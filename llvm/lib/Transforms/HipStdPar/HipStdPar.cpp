//===----- HipStdPar.cpp - HIP C++ Standard Parallelism Support Passes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements two passes that enable HIP C++ Standard Parallelism
// Support:
//
// 1. AcceleratorCodeSelection (required): Given that only algorithms are
//    accelerated, and that the accelerated implementation exists in the form of
//    a compute kernel, we assume that only the kernel, and all functions
//    reachable from it, constitute code that the user expects the accelerator
//    to execute. Thus, we identify the set of all functions reachable from
//    kernels, and then remove all unreachable ones. This last part is necessary
//    because it is possible for code that the user did not expect to execute on
//    an accelerator to contain constructs that cannot be handled by the target
//    BE, which cannot be provably demonstrated to be dead code in general, and
//    thus can lead to mis-compilation. The degenerate case of this is when a
//    Module contains no kernels (the parent TU had no algorithm invocations fit
//    for acceleration), which we handle by completely emptying said module.
//    **NOTE**: The above does not handle indirectly reachable functions i.e.
//              it is possible to obtain a case where the target of an indirect
//              call is otherwise unreachable and thus is removed; this
//              restriction is aligned with the current `-hipstdpar` limitations
//              and will be relaxed in the future.
//
// 2. AllocationInterposition (required only when on-demand paging is
//    unsupported): Some accelerators or operating systems might not support
//    transparent on-demand paging. Thus, they would only be able to access
//    memory that is allocated by an accelerator-aware mechanism. For such cases
//    the user can opt into enabling allocation / deallocation interposition,
//    whereby we replace calls to known allocation / deallocation functions with
//    calls to runtime implemented equivalents that forward the requests to
//    accelerator-aware interfaces. We also support freeing system allocated
//    memory that ends up in one of the runtime equivalents, since this can
//    happen if e.g. a library that was compiled without interposition returns
//    an allocation that can be validly passed to `free`.
//
// 3. MathFixup (required): Some accelerators might have an incomplete
//    implementation for the intrinsics used to implement some of the math
//    functions in <cmath> / their corresponding libcall lowerings. Since this
//    can vary quite significantly between accelerators, we replace calls to a
//    set of intrinsics / lib functions known to be problematic with calls to a
//    HIPSTDPAR specific forwarding layer, which gives an uniform interface for
//    accelerators to implement in their own runtime components. This pass
//    should run before AcceleratorCodeSelection so as to prevent the spurious
//    removal of the HIPSTDPAR specific forwarding functions.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HipStdPar/HipStdPar.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cassert>
#include <string>
#include <utility>

using namespace llvm;

template<typename T>
static inline void eraseFromModule(T &ToErase) {
  ToErase.replaceAllUsesWith(PoisonValue::get(ToErase.getType()));
  ToErase.eraseFromParent();
}

static bool checkIfSupported(GlobalVariable &G) {
  if (!G.isThreadLocal())
    return true;

  G.dropDroppableUses();

  if (!G.isConstantUsed())
    return true;

  std::string W;
  raw_string_ostream OS(W);

  OS << "Accelerator does not support the thread_local variable "
    << G.getName();

  Instruction *I = nullptr;
  SmallVector<User *> Tmp(G.users());
  SmallPtrSet<User *, 5> Visited;
  do {
    auto U = std::move(Tmp.back());
    Tmp.pop_back();

    if (!Visited.insert(U).second)
      continue;

    if (isa<Instruction>(U))
      I = cast<Instruction>(U);
    else
      Tmp.insert(Tmp.end(), U->user_begin(), U->user_end());
  } while (!I && !Tmp.empty());

  assert(I && "thread_local global should have at least one non-constant use.");

  G.getContext().diagnose(
    DiagnosticInfoUnsupported(*I->getParent()->getParent(), W,
                              I->getDebugLoc(), DS_Error));

  return false;
}

static inline void clearModule(Module &M) { // TODO: simplify.
  while (!M.functions().empty())
    eraseFromModule(*M.begin());
  while (!M.globals().empty())
    eraseFromModule(*M.globals().begin());
  while (!M.aliases().empty())
    eraseFromModule(*M.aliases().begin());
  while (!M.ifuncs().empty())
    eraseFromModule(*M.ifuncs().begin());
}

static SmallVector<std::reference_wrapper<Use>>
collectIndirectableUses(GlobalVariable *G) {
  // We are interested only in use chains that end in an Instruction.
  SmallVector<std::reference_wrapper<Use>> Uses;

  SmallVector<std::reference_wrapper<Use>> Stack(G->use_begin(), G->use_end());
  while (!Stack.empty()) {
    Use &U = Stack.pop_back_val();
    if (isa<Instruction>(U.getUser()))
      Uses.emplace_back(U);
    else
      transform(U.getUser()->uses(), std::back_inserter(Stack),
                [](auto &&U) { return std::ref(U); });
  }

  return Uses;
}

static inline GlobalVariable *getGlobalForName(GlobalVariable *G) {
  // Create an anonymous global which stores the variable's name, which will be
  // used by the HIPSTDPAR runtime to look up the program-wide symbol.
  LLVMContext &Ctx = G->getContext();
  auto *CDS = ConstantDataArray::getString(Ctx, G->getName());

  GlobalVariable *N = G->getParent()->getOrInsertGlobal("", CDS->getType());
  N->setInitializer(CDS);
  N->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
  N->setConstant(true);

  return N;
}

static inline GlobalVariable *getIndirectionGlobal(Module *M) {
  // Create an anonymous global which stores a pointer to a pointer, which will
  // be externally initialised by the HIPSTDPAR runtime with the address of the
  // program-wide symbol.
  Type *PtrTy = PointerType::get(
      M->getContext(), M->getDataLayout().getDefaultGlobalsAddressSpace());
  GlobalVariable *NewG = M->getOrInsertGlobal("", PtrTy);

  NewG->setInitializer(PoisonValue::get(NewG->getValueType()));
  NewG->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
  NewG->setConstant(true);
  NewG->setExternallyInitialized(true);

  return NewG;
}

static Constant *
appendIndirectedGlobal(const GlobalVariable *IndirectionTable,
                       SmallVector<Constant *> &SymbolIndirections,
                       GlobalVariable *ToIndirect) {
  Module *M = ToIndirect->getParent();

  auto *InitTy = cast<StructType>(IndirectionTable->getValueType());
  auto *SymbolListTy = cast<StructType>(InitTy->getStructElementType(2));
  Type *NameTy = SymbolListTy->getElementType(0);
  Type *IndirectTy = SymbolListTy->getElementType(1);

  Constant *NameG = getGlobalForName(ToIndirect);
  Constant *IndirectG = getIndirectionGlobal(M);
  Constant *Entry = ConstantStruct::get(
      SymbolListTy, {ConstantExpr::getAddrSpaceCast(NameG, NameTy),
                     ConstantExpr::getAddrSpaceCast(IndirectG, IndirectTy)});
  SymbolIndirections.push_back(Entry);

  return IndirectG;
}

static void fillIndirectionTable(GlobalVariable *IndirectionTable,
                                 SmallVector<Constant *> Indirections) {
  Module *M = IndirectionTable->getParent();
  size_t SymCnt = Indirections.size();

  auto *InitTy = cast<StructType>(IndirectionTable->getValueType());
  Type *SymbolListTy = InitTy->getStructElementType(1);
  auto *SymbolTy = cast<StructType>(InitTy->getStructElementType(2));

  Constant *Count = ConstantInt::get(InitTy->getStructElementType(0), SymCnt);
  M->removeGlobalVariable(IndirectionTable);
  GlobalVariable *Symbols =
      M->getOrInsertGlobal("", ArrayType::get(SymbolTy, SymCnt));
  Symbols->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
  Symbols->setInitializer(
      ConstantArray::get(ArrayType::get(SymbolTy, SymCnt), {Indirections}));
  Symbols->setConstant(true);

  Constant *ASCSymbols = ConstantExpr::getAddrSpaceCast(Symbols, SymbolListTy);
  Constant *Init = ConstantStruct::get(
      InitTy, {Count, ASCSymbols, PoisonValue::get(SymbolTy)});
  M->insertGlobalVariable(IndirectionTable);
  IndirectionTable->setInitializer(Init);
}

static void replaceWithIndirectUse(const Use &U, const GlobalVariable *G,
                                   Constant *IndirectedG) {
  auto *I = cast<Instruction>(U.getUser());

  IRBuilder<> Builder(I);
  unsigned OpIdx = U.getOperandNo();
  Value *Op = I->getOperand(OpIdx);

  // We walk back up the use chain, which could be an arbitrarily long sequence
  // of constexpr AS casts, ptr-to-int and GEP instructions, until we reach the
  // indirected global.
  while (auto *CE = dyn_cast<ConstantExpr>(Op)) {
    assert((CE->getOpcode() == Instruction::GetElementPtr ||
            CE->getOpcode() == Instruction::AddrSpaceCast ||
            CE->getOpcode() == Instruction::PtrToInt) &&
           "Only GEP, ASCAST or PTRTOINT constant uses supported!");

    Instruction *NewI = Builder.Insert(CE->getAsInstruction());
    I->replaceUsesOfWith(Op, NewI);
    I = NewI;
    Op = I->getOperand(0);
    OpIdx = 0;
    Builder.SetInsertPoint(I);
  }

  assert(Op == G && "Must reach indirected global!");

  I->setOperand(OpIdx, Builder.CreateLoad(G->getType(), IndirectedG));
}

static inline bool isValidIndirectionTable(GlobalVariable *IndirectionTable) {
  std::string W;
  raw_string_ostream OS(W);

  Type *Ty = IndirectionTable->getValueType();
  bool Valid = false;

  if (!isa<StructType>(Ty)) {
    OS << "The Indirection Table must be a struct type; ";
    Ty->print(OS);
    OS << " is incorrect.\n";
  } else if (cast<StructType>(Ty)->getNumElements() != 3u) {
    OS << "The Indirection Table must have 3 elements; "
       << cast<StructType>(Ty)->getNumElements() << " is incorrect.\n";
  } else if (!isa<IntegerType>(cast<StructType>(Ty)->getStructElementType(0))) {
    OS << "The first element in the Indirection Table must be an integer; ";
    cast<StructType>(Ty)->getStructElementType(0)->print(OS);
    OS << " is incorrect.\n";
  } else if (!isa<PointerType>(cast<StructType>(Ty)->getStructElementType(1))) {
    OS << "The second element in the Indirection Table must be a pointer; ";
    cast<StructType>(Ty)->getStructElementType(1)->print(OS);
    OS << " is incorrect.\n";
  } else if (!isa<StructType>(cast<StructType>(Ty)->getStructElementType(2))) {
    OS << "The third element in the Indirection Table must be a struct type; ";
    cast<StructType>(Ty)->getStructElementType(2)->print(OS);
    OS << " is incorrect.\n";
  } else {
    Valid = true;
  }

  if (!Valid)
    IndirectionTable->getContext().diagnose(DiagnosticInfoGeneric(W, DS_Error));

  return Valid;
}

static void indirectGlobals(GlobalVariable *IndirectionTable,
                            SmallVector<GlobalVariable *> ToIndirect) {
  // We replace globals with an indirected access via a pointer that will get
  // set by the HIPSTDPAR runtime, using their accessible, program-wide unique
  // address as set by the host linker-loader.
  SmallVector<Constant *> SymbolIndirections;
  for (auto &&G : ToIndirect) {
    SmallVector<std::reference_wrapper<Use>> Uses = collectIndirectableUses(G);

    if (Uses.empty())
      continue;

    Constant *IndirectedGlobal =
        appendIndirectedGlobal(IndirectionTable, SymbolIndirections, G);

    for_each(Uses,
             [=](auto &&U) { replaceWithIndirectUse(U, G, IndirectedGlobal); });

    eraseFromModule(*G);
  }

  if (SymbolIndirections.empty())
    return;

  fillIndirectionTable(IndirectionTable, std::move(SymbolIndirections));
}

static inline void maybeHandleGlobals(Module &M) {
  unsigned GlobAS = M.getDataLayout().getDefaultGlobalsAddressSpace();

  SmallVector<GlobalVariable *> ToIndirect;
  for (auto &&G : M.globals()) {
    if (!checkIfSupported(G))
      return clearModule(M);
    if (G.getAddressSpace() != GlobAS)
      continue;
    if (G.isConstant() && G.hasInitializer() && G.hasAtLeastLocalUnnamedAddr())
      continue;

    ToIndirect.push_back(&G);
  }

  if (ToIndirect.empty())
    return;

  if (auto *IT = M.getNamedGlobal("__hipstdpar_symbol_indirection_table")) {
    if (!isValidIndirectionTable(IT))
      return clearModule(M);
    return indirectGlobals(IT, std::move(ToIndirect));
  } else {
    for (auto &&G : ToIndirect) {
      // We will internalise these, so we provide a poison initialiser.
      if (!G->hasInitializer())
        G->setInitializer(PoisonValue::get(G->getValueType()));
    }
  }
}

template<unsigned N>
static inline void removeUnreachableFunctions(
  const SmallPtrSet<const Function *, N>& Reachable, Module &M) {
  removeFromUsedLists(M, [&](Constant *C) {
    if (auto F = dyn_cast<Function>(C))
      return !Reachable.contains(F);

    return false;
  });

  SmallVector<std::reference_wrapper<Function>> ToRemove;
  copy_if(M, std::back_inserter(ToRemove), [&](auto &&F) {
    return !F.isIntrinsic() && !Reachable.contains(&F);
  });

  for_each(ToRemove, eraseFromModule<Function>);
}

static inline bool isAcceleratorExecutionRoot(const Function *F) {
    if (!F)
      return false;

    return F->getCallingConv() == CallingConv::AMDGPU_KERNEL;
}

static inline bool checkIfSupported(const Function *F, const CallBase *CB) {
  const auto Dx = F->getName().rfind("__hipstdpar_unsupported");

  if (Dx == StringRef::npos)
    return true;

  const auto N = F->getName().substr(0, Dx);

  std::string W;
  raw_string_ostream OS(W);

  if (N == "__ASM")
    OS << "Accelerator does not support the ASM block:\n"
      << cast<ConstantDataArray>(CB->getArgOperand(0))->getAsCString();
  else
    OS << "Accelerator does not support the " << N << " function.";

  auto Caller = CB->getParent()->getParent();

  Caller->getContext().diagnose(
    DiagnosticInfoUnsupported(*Caller, W, CB->getDebugLoc(), DS_Error));

  return false;
}

PreservedAnalyses
  HipStdParAcceleratorCodeSelectionPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  auto &CGA = MAM.getResult<CallGraphAnalysis>(M);

  SmallPtrSet<const Function *, 32> Reachable;
  for (auto &&CGN : CGA) {
    if (!isAcceleratorExecutionRoot(CGN.first))
      continue;

    Reachable.insert(CGN.first);

    SmallVector<const Function *> Tmp({CGN.first});
    do {
      auto F = std::move(Tmp.back());
      Tmp.pop_back();

      for (auto &&N : *CGA[F]) {
        if (!N.second)
          continue;
        if (!N.second->getFunction())
          continue;
        if (Reachable.contains(N.second->getFunction()))
          continue;

        if (!checkIfSupported(N.second->getFunction(),
                              dyn_cast<CallBase>(*N.first)))
          return PreservedAnalyses::none();

        Reachable.insert(N.second->getFunction());
        Tmp.push_back(N.second->getFunction());
      }
    } while (!std::empty(Tmp));
  }

  if (std::empty(Reachable))
    clearModule(M);
  else
    removeUnreachableFunctions(Reachable, M);

  maybeHandleGlobals(M);

  return PreservedAnalyses::none();
}

static constexpr std::pair<StringLiteral, StringLiteral> ReplaceMap[]{
    {"aligned_alloc", "__hipstdpar_aligned_alloc"},
    {"calloc", "__hipstdpar_calloc"},
    {"free", "__hipstdpar_free"},
    {"malloc", "__hipstdpar_malloc"},
    {"memalign", "__hipstdpar_aligned_alloc"},
    {"mmap", "__hipstdpar_mmap"},
    {"munmap", "__hipstdpar_munmap"},
    {"posix_memalign", "__hipstdpar_posix_aligned_alloc"},
    {"realloc", "__hipstdpar_realloc"},
    {"reallocarray", "__hipstdpar_realloc_array"},
    {"_ZdaPv", "__hipstdpar_operator_delete"},
    {"_ZdaPvm", "__hipstdpar_operator_delete_sized"},
    {"_ZdaPvSt11align_val_t", "__hipstdpar_operator_delete_aligned"},
    {"_ZdaPvmSt11align_val_t", "__hipstdpar_operator_delete_aligned_sized"},
    {"_ZdlPv", "__hipstdpar_operator_delete"},
    {"_ZdlPvm", "__hipstdpar_operator_delete_sized"},
    {"_ZdlPvSt11align_val_t", "__hipstdpar_operator_delete_aligned"},
    {"_ZdlPvmSt11align_val_t", "__hipstdpar_operator_delete_aligned_sized"},
    {"_Znam", "__hipstdpar_operator_new"},
    {"_ZnamRKSt9nothrow_t", "__hipstdpar_operator_new_nothrow"},
    {"_ZnamSt11align_val_t", "__hipstdpar_operator_new_aligned"},
    {"_ZnamSt11align_val_tRKSt9nothrow_t",
     "__hipstdpar_operator_new_aligned_nothrow"},

    {"_Znwm", "__hipstdpar_operator_new"},
    {"_ZnwmRKSt9nothrow_t", "__hipstdpar_operator_new_nothrow"},
    {"_ZnwmSt11align_val_t", "__hipstdpar_operator_new_aligned"},
    {"_ZnwmSt11align_val_tRKSt9nothrow_t",
     "__hipstdpar_operator_new_aligned_nothrow"},
    {"__builtin_calloc", "__hipstdpar_calloc"},
    {"__builtin_free", "__hipstdpar_free"},
    {"__builtin_malloc", "__hipstdpar_malloc"},
    {"__builtin_operator_delete", "__hipstdpar_operator_delete"},
    {"__builtin_operator_new", "__hipstdpar_operator_new"},
    {"__builtin_realloc", "__hipstdpar_realloc"},
    {"__libc_calloc", "__hipstdpar_calloc"},
    {"__libc_free", "__hipstdpar_free"},
    {"__libc_malloc", "__hipstdpar_malloc"},
    {"__libc_memalign", "__hipstdpar_aligned_alloc"},
    {"__libc_realloc", "__hipstdpar_realloc"}};

static constexpr std::pair<StringLiteral, StringLiteral> HiddenMap[]{
    // hidden_malloc and hidden_free are only kept for backwards compatibility /
    // legacy purposes, and we should remove them in the future
    {"__hipstdpar_hidden_malloc", "__libc_malloc"},
    {"__hipstdpar_hidden_free", "__libc_free"},
    {"__hipstdpar_hidden_memalign", "__libc_memalign"},
    {"__hipstdpar_hidden_mmap", "mmap"},
    {"__hipstdpar_hidden_munmap", "munmap"}};

PreservedAnalyses
HipStdParAllocationInterpositionPass::run(Module &M, ModuleAnalysisManager&) {
  SmallDenseMap<StringRef, StringRef> AllocReplacements(std::cbegin(ReplaceMap),
                                                        std::cend(ReplaceMap));

  for (auto &&F : M) {
    if (!F.hasName())
      continue;
    auto It = AllocReplacements.find(F.getName());
    if (It == AllocReplacements.end())
      continue;

    if (auto R = M.getFunction(It->second)) {
      F.replaceAllUsesWith(R);
    } else {
      std::string W;
      raw_string_ostream OS(W);

      OS << "cannot be interposed, missing: " << AllocReplacements[F.getName()]
        << ". Tried to run the allocation interposition pass without the "
        << "replacement functions available.";

      F.getContext().diagnose(DiagnosticInfoUnsupported(F, W,
                                                        F.getSubprogram(),
                                                        DS_Warning));
    }
  }

  for (auto &&HR : HiddenMap) {
    if (auto F = M.getFunction(HR.first)) {
      auto R = M.getOrInsertFunction(HR.second, F->getFunctionType(),
                                     F->getAttributes());
      F->replaceAllUsesWith(R.getCallee());

      eraseFromModule(*F);
    }
  }

  return PreservedAnalyses::none();
}

static constexpr std::pair<StringLiteral, StringLiteral> MathLibToHipStdPar[]{
    {"acosh", "__hipstdpar_acosh_f64"},
    {"acoshf", "__hipstdpar_acosh_f32"},
    {"asinh", "__hipstdpar_asinh_f64"},
    {"asinhf", "__hipstdpar_asinh_f32"},
    {"atanh", "__hipstdpar_atanh_f64"},
    {"atanhf", "__hipstdpar_atanh_f32"},
    {"cbrt", "__hipstdpar_cbrt_f64"},
    {"cbrtf", "__hipstdpar_cbrt_f32"},
    {"erf", "__hipstdpar_erf_f64"},
    {"erff", "__hipstdpar_erf_f32"},
    {"erfc", "__hipstdpar_erfc_f64"},
    {"erfcf", "__hipstdpar_erfc_f32"},
    {"fdim", "__hipstdpar_fdim_f64"},
    {"fdimf", "__hipstdpar_fdim_f32"},
    {"expm1", "__hipstdpar_expm1_f64"},
    {"expm1f", "__hipstdpar_expm1_f32"},
    {"hypot", "__hipstdpar_hypot_f64"},
    {"hypotf", "__hipstdpar_hypot_f32"},
    {"ilogb", "__hipstdpar_ilogb_f64"},
    {"ilogbf", "__hipstdpar_ilogb_f32"},
    {"lgamma", "__hipstdpar_lgamma_f64"},
    {"lgammaf", "__hipstdpar_lgamma_f32"},
    {"log1p", "__hipstdpar_log1p_f64"},
    {"log1pf", "__hipstdpar_log1p_f32"},
    {"logb", "__hipstdpar_logb_f64"},
    {"logbf", "__hipstdpar_logb_f32"},
    {"nextafter", "__hipstdpar_nextafter_f64"},
    {"nextafterf", "__hipstdpar_nextafter_f32"},
    {"nexttoward", "__hipstdpar_nexttoward_f64"},
    {"nexttowardf", "__hipstdpar_nexttoward_f32"},
    {"remainder", "__hipstdpar_remainder_f64"},
    {"remainderf", "__hipstdpar_remainder_f32"},
    {"remquo", "__hipstdpar_remquo_f64"},
    {"remquof", "__hipstdpar_remquo_f32"},
    {"scalbln", "__hipstdpar_scalbln_f64"},
    {"scalblnf", "__hipstdpar_scalbln_f32"},
    {"scalbn", "__hipstdpar_scalbn_f64"},
    {"scalbnf", "__hipstdpar_scalbn_f32"},
    {"tgamma", "__hipstdpar_tgamma_f64"},
    {"tgammaf", "__hipstdpar_tgamma_f32"}};

PreservedAnalyses HipStdParMathFixupPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  if (M.empty())
    return PreservedAnalyses::all();

  SmallVector<std::pair<Function *, std::string>> ToReplace;
  for (auto &&F : M) {
    if (!F.hasName())
      continue;

    StringRef N = F.getName();
    Intrinsic::ID ID = F.getIntrinsicID();

    switch (ID) {
    case Intrinsic::not_intrinsic: {
      auto It =
          find_if(MathLibToHipStdPar, [&](auto &&M) { return M.first == N; });
      if (It == std::cend(MathLibToHipStdPar))
        continue;
      ToReplace.emplace_back(&F, It->second);
      break;
    }
    case Intrinsic::acos:
    case Intrinsic::asin:
    case Intrinsic::atan:
    case Intrinsic::atan2:
    case Intrinsic::cosh:
    case Intrinsic::modf:
    case Intrinsic::sinh:
    case Intrinsic::tan:
    case Intrinsic::tanh:
      break;
    default: {
      if (F.getReturnType()->isDoubleTy()) {
        switch (ID) {
        case Intrinsic::cos:
        case Intrinsic::exp:
        case Intrinsic::exp2:
        case Intrinsic::log:
        case Intrinsic::log10:
        case Intrinsic::log2:
        case Intrinsic::pow:
        case Intrinsic::sin:
          break;
        default:
          continue;
        }
        break;
      }
      continue;
    }
    }

    ToReplace.emplace_back(&F, N);
    llvm::replace(ToReplace.back().second, '.', '_');
    StringRef Prefix = "llvm";
    ToReplace.back().second.replace(0, Prefix.size(), "__hipstdpar");
  }
  for (auto &&[F, NewF] : ToReplace)
    F->replaceAllUsesWith(
        M.getOrInsertFunction(NewF, F->getFunctionType()).getCallee());

  return PreservedAnalyses::none();
}
