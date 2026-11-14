//===- SPIRVLowerDynamicLocalMemory.cpp - Lower dynamic local memory ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic local memory, i.e. an unsized external Workgroup array such as
// CUDA's `extern __shared__ T Buf[]`, has a size only known at launch. OpenCL
// and Level Zero runtimes cannot specify the size of such an array and instead
// they rely on `ptr addrspace(3)` kernel arguments to provide such memory.
//
// This pass appends a `ptr addrspace(3)` parameter to every kernel of a module
// referencing dynamic local memory. Each kernel stores the parameter into an
// internal Workgroup variable on entry, and every use of the dynamic local
// memory is replaced with a load of that variable. All declarations of dynamic
// local memory alias the same memory, as in CUDA.
//
// Going through a variable rather than threading the pointer through every
// function keeps indirect calls and functions shared between kernels working.
// All work-items store the same value, and each loads it only after its own
// store. All kernels get the parameter, whether they access the memory or not,
// so the runtime can treat the kernels of a module alike.
//
// The lowering changes the kernel signatures, so it is limited to Intel GPUs,
// whose runtime sets the argument. To add support of dynamic shared memory
// into OpenCL backends, this pass will also be needed.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "spirv-lower-dynamic-local-memory"

using namespace llvm;

namespace {

// Maximum alignment of SLM arguments provided by
// Level Zero driver. Applicable only to Intel GPUs.
constexpr Align LocalMemArgAlign = Align::Constant<32>();

/// Returns whether \p GV is dynamic local memory: an external Workgroup array
/// without a size.
bool isDynamicLocalMemory(const GlobalVariable &GV) {
  if (!GV.isDeclaration() ||
      GV.getAddressSpace() !=
          storageClassToAddressSpace(SPIRV::StorageClass::Workgroup))
    return false;
  const auto *ArrTy = dyn_cast<ArrayType>(GV.getValueType());
  return ArrTy && ArrTy->getNumElements() == 0;
}

/// Replaces the kernel \p F with a copy taking an additional trailing
/// parameter of type \p ParamTy, and returns the copy. Kernels cannot be
/// called, so only their other uses need updating.
Function *appendKernelParam(Function &F, Type *ParamTy) {
  FunctionType *OldTy = F.getFunctionType();
  SmallVector<Type *> Params(OldTy->params());
  Params.push_back(ParamTy);
  FunctionType *NewTy =
      FunctionType::get(OldTy->getReturnType(), Params, OldTy->isVarArg());

  Function *NewF = Function::Create(NewTy, F.getLinkage(), F.getAddressSpace());
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);
  NewF->copyAttributesFrom(&F);
  NewF->setComdat(F.getComdat());
  NewF->copyMetadata(&F, 0);
  NewF->splice(NewF->begin(), &F);
  for (auto [OldArg, NewArg] : zip(F.args(), NewF->args())) {
    OldArg.replaceAllUsesWith(&NewArg);
    NewArg.takeName(&OldArg);
  }
  NewF->getArg(NewF->arg_size() - 1)->setName("dyn_local_mem");

  NewF->takeName(&F);
  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();
  return NewF;
}

void updateMemoryEffects(Function *F, MemoryEffects NewEffects) {
  SmallVector<Function *> Worklist;
  SmallPtrSet<Function *, 16> Visited;
  Worklist.push_back(F);

  while (!Worklist.empty()) {
    Function *Current = Worklist.pop_back_val();

    // Record visited functions, skip if function was
    // already processed.
    if (!Visited.insert(Current).second)
      continue;

    Current->setMemoryEffects(Current->getMemoryEffects() | NewEffects);

    // Add callers of the function into worklist
    for (auto &U : Current->uses()) {
      if (auto *CB = dyn_cast<CallBase>(U.getUser())) {
        if (CB->isCallee(&U)) {
          auto *Caller = CB->getCaller();
          Worklist.push_back(Caller);

          // Update call instruction with new effects
          CB->setMemoryEffects(CB->getMemoryEffects() | NewEffects);
        }
      }
    }
  }
}

bool lowerDynamicLocalMemory(Module &M, const SPIRVTargetMachine &TM) {
  // Shaders have no kernel arguments, so the pass is disabled.
  // It is only used for OpenCL runtime, because OpenCL has no API
  // to size extern workgroup array. Runtimes can check global
  // variable `__spirv_dynamic_local_memory_present` to check
  // if module contains the additional SLM argument.
  // UnknownOS implies OpenCL
  // https://llvm.org/docs/SPIRVUsage.html#target-triples.
  if (TM.getSubtargetImpl()->isShader() ||
      (M.getTargetTriple().getOS() != Triple::UnknownOS &&
       M.getTargetTriple().getOS() != Triple::OpenCL))
    return false;

  SmallVector<GlobalVariable *> DynMems;
  for (GlobalVariable &GV : M.globals())
    if (isDynamicLocalMemory(GV) && !GV.use_empty())
      DynMems.push_back(&GV);
  if (DynMems.empty())
    return false;

  SmallVector<Function *> Kernels;
  for (Function &F : M)
    if (!F.isDeclaration() && F.getCallingConv() == CallingConv::SPIR_KERNEL)
      Kernels.push_back(&F);
  // Leave the memory to be resolved by linking with a module defining kernels.
  if (Kernels.empty())
    return false;

  for (GlobalVariable *GV : DynMems)
    if (GV->getAlign().valueOrOne() > LocalMemArgAlign)
      reportFatalUsageError(
          "dynamic local memory '" + GV->getName() +
          "' requires an alignment of " + Twine(GV->getAlign()->value()) +
          " bytes, but at most " + Twine(LocalMemArgAlign.value()) +
          " bytes are supported");

  unsigned LocalAS = storageClassToAddressSpace(SPIRV::StorageClass::Workgroup);
  auto *LocalPtrTy = PointerType::get(M.getContext(), LocalAS);
  auto *Slot = new GlobalVariable(
      M, LocalPtrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      PoisonValue::get(LocalPtrTy), "__spirv_dynamic_local_memory",
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal, LocalAS);
  Slot->setAlignment(M.getDataLayout().getPointerABIAlignment(LocalAS));
  Slot->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  // Marker detected by liboffload to add SLM argument
  auto *MarkerTy = Type::getInt8Ty(M.getContext());
  new GlobalVariable(
      M, MarkerTy, /*isConstant=*/false, GlobalValue::ExternalLinkage,
      ConstantInt::get(MarkerTy, 1), "__spirv_dynamic_local_memory_present",
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
      storageClassToAddressSpace(SPIRV::StorageClass::CrossWorkgroup));

  // Replace the uses of the memory, which must all be in instructions.
  SmallVector<Constant *> DynMemConsts(DynMems.begin(), DynMems.end());
  convertUsersOfConstantsToInstructions(DynMemConsts);
  for (GlobalVariable *GV : DynMems) {
    DenseMap<Function *, SmallVector<Use *>> UsesOfGV;
    for (Use &U : GV->uses()) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      if (!I)
        reportFatalUsageError("dynamic local memory '" + GV->getName() +
                              "' cannot be referenced by a constant");

      Function *F = I->getFunction();
      UsesOfGV[F].push_back(&U);
    }

    for (auto &[F, Uses] : UsesOfGV) {
      IRBuilder<> Builder(F->getEntryBlock().getFirstInsertionPt());
      auto *LoadedSlot = Builder.CreateLoad(LocalPtrTy, Slot, "dyn_local_mem");

      for (Use *U : Uses)
        U->set(LoadedSlot);

      // Modify memory effects of the function and it's callers, a function that
      // would reference SLM pointer might have ModeRefInfo::NoModRef, but
      // adding the load requires change it to at least ModRefInfo::Ref
      updateMemoryEffects(F, MemoryEffects::otherMemOnly(ModRefInfo::Ref));
    }

    GV->eraseFromParent();
  }

  // Give every kernel the memory as an argument stored on entry.
  for (Function *F : Kernels) {
    Function *NewF = appendKernelParam(*F, LocalPtrTy);
    IRBuilder<> Builder(NewF->getEntryBlock().getFirstNonPHIOrAlloca());
    Builder.CreateStore(NewF->getArg(NewF->arg_size() - 1), Slot);
    NewF->setMemoryEffects(NewF->getMemoryEffects() |
                           MemoryEffects::otherMemOnly(ModRefInfo::ModRef));
  }

  return true;
}

class SPIRVLowerDynamicLocalMemoryLegacy : public ModulePass {
public:
  static char ID;
  SPIRVLowerDynamicLocalMemoryLegacy(const SPIRVTargetMachine &TM)
      : ModulePass(ID), TM(TM) {}

  StringRef getPassName() const override {
    return "SPIRV lower dynamic local memory";
  }

  bool runOnModule(Module &M) override {
    return lowerDynamicLocalMemory(M, TM);
  }

private:
  const SPIRVTargetMachine &TM;
};

} // namespace

PreservedAnalyses
SPIRVLowerDynamicLocalMemoryPass::run(Module &M, ModuleAnalysisManager &AM) {
  return lowerDynamicLocalMemory(M, TM) ? PreservedAnalyses::none()
                                        : PreservedAnalyses::all();
}

char SPIRVLowerDynamicLocalMemoryLegacy::ID = 0;

INITIALIZE_PASS(SPIRVLowerDynamicLocalMemoryLegacy,
                "spirv-lower-dynamic-local-memory",
                "Lower SPIR-V dynamic local memory to kernel arguments", false,
                false)

ModulePass *
llvm::createSPIRVLowerDynamicLocalMemoryPass(const SPIRVTargetMachine &TM) {
  return new SPIRVLowerDynamicLocalMemoryLegacy(TM);
}
