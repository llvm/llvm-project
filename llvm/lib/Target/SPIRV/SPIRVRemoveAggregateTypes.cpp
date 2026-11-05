//===-- SPIRVRemoveAggregateTypes.cpp - remove aggregate types -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass modifies function signatures and indirect or inline asm call sites
// containing aggregate arguments and/or return values before IRTranslator,
// replacing the aggregates by i32 placeholders. Information about the original
// types is stored in metadata. It is used during call lowering to restore
// correct SPIR-V types of function arguments and return values.
//
// The placeholders leave IR that generic passes do not expect, such as an
// aggregate PHI with an i32 operand, so the pass runs after them, as the first
// SPIR-V pass of addISelPrepare. It clones functions, so it also runs before
// any pass that records instructions in SPIRVGlobalRegistry.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

class SPIRVRemoveAggregateTypesImpl {
  const SPIRVTargetMachine &TM;
  Function *removeAggregateTypesFromSignature(Function *F);
  bool removeAggregateTypesFromCalls(Function *F);

public:
  SPIRVRemoveAggregateTypesImpl(const SPIRVTargetMachine &TM) : TM(TM) {}
  bool runOnModule(Module &M);
};

class SPIRVRemoveAggregateTypesLegacy : public ModulePass {
  const SPIRVTargetMachine &TM;

public:
  static char ID;
  SPIRVRemoveAggregateTypesLegacy(const SPIRVTargetMachine &TM)
      : ModulePass(ID), TM(TM) {}

  bool runOnModule(Module &M) override {
    return SPIRVRemoveAggregateTypesImpl(TM).runOnModule(M);
  }

  StringRef getPassName() const override {
    return "SPIRV remove aggregate types";
  }
};

} // namespace

char SPIRVRemoveAggregateTypesLegacy::ID = 0;

INITIALIZE_PASS(SPIRVRemoveAggregateTypesLegacy, "spirv-remove-aggregate-types",
                "SPIRV remove aggregate types", false, false)

static void
addFunctionTypeMutation(NamedMDNode *NMD,
                        SmallVector<std::pair<int, Type *>> ChangedTys,
                        StringRef Name, StringRef AsmConstraints = "") {

  LLVMContext &Ctx = NMD->getParent()->getContext();
  Type *I32Ty = IntegerType::getInt32Ty(Ctx);

  SmallVector<Metadata *> MDArgs;
  MDArgs.push_back(MDString::get(Ctx, Name));
  transform(ChangedTys, std::back_inserter(MDArgs), [=, &Ctx](auto &&CTy) {
    return MDNode::get(
        Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32Ty, CTy.first, true)),
              ValueAsMetadata::get(Constant::getNullValue(CTy.second))});
  });
  if (!AsmConstraints.empty())
    MDArgs.push_back(MDNode::get(Ctx, MDString::get(Ctx, AsmConstraints)));
  NMD->addOperand(MDNode::get(Ctx, MDArgs));
}

// Returns F if aggregate argument/return types are not present or cloned F
// function with the types replaced by i32 types. The change in types is
// noted in 'spv.cloned_funcs' metadata for later restoration.
Function *
SPIRVRemoveAggregateTypesImpl::removeAggregateTypesFromSignature(Function *F) {
  bool IsRetAggr = F->getReturnType()->isAggregateType();
  // Allow intrinsics with aggregate return/argument types to reach GlobalISel.
  // Renaming/mutating the signature of an intrinsic would desync its name from
  // its argument types and break the IR verifier.
  if (F->isIntrinsic())
    return F;

  IRBuilder<> B(*F->getParent());

  bool HasAggrArg = llvm::any_of(F->args(), [](Argument &Arg) {
    return Arg.getType()->isAggregateType();
  });
  bool DoClone = IsRetAggr || HasAggrArg;
  if (!DoClone)
    return F;
  SmallVector<std::pair<int, Type *>, 4> ChangedTypes;
  Type *RetType = IsRetAggr ? B.getInt32Ty() : F->getReturnType();
  if (IsRetAggr)
    ChangedTypes.push_back(std::pair<int, Type *>(-1, F->getReturnType()));
  SmallVector<Type *, 4> ArgTypes;
  for (const auto &Arg : F->args()) {
    if (Arg.getType()->isAggregateType()) {
      ArgTypes.push_back(B.getInt32Ty());
      ChangedTypes.push_back(
          std::pair<int, Type *>(Arg.getArgNo(), Arg.getType()));
    } else
      ArgTypes.push_back(Arg.getType());
  }
  FunctionType *NewFTy =
      FunctionType::get(RetType, ArgTypes, F->getFunctionType()->isVarArg());
  Function *NewF =
      Function::Create(NewFTy, F->getLinkage(), F->getAddressSpace(),
                       F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  auto NewFArgIt = NewF->arg_begin();
  for (auto &Arg : F->args()) {
    StringRef ArgName = Arg.getName();
    NewFArgIt->setName(ArgName);
    VMap[&Arg] = &(*NewFArgIt++);
  }
  SmallVector<ReturnInst *, 8> Returns;

  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  NewF->takeName(F);
  NewF->setComdat(F->getComdat());

  addFunctionTypeMutation(
      NewF->getParent()->getOrInsertNamedMetadata("spv.cloned_funcs"),
      std::move(ChangedTypes), NewF->getName());

  for (User *U : F->users()) {
    if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledFunction() == F)
      CB->mutateFunctionType(NewF->getFunctionType());
  }
  // NewF keeps F's address space, so their pointer types match and
  // RAUW is safe despite the differing signatures.
  assert(F->getType() == NewF->getType() &&
         "RAUW requires F and NewF to share the same pointer type");
  F->replaceAllUsesWith(NewF);

  // register the mutation
  if (RetType != F->getReturnType())
    TM.getSubtarget<SPIRVSubtarget>(*F).getSPIRVGlobalRegistry()->addMutated(
        NewF, F->getReturnType());
  return NewF;
}

static std::string fixMultiOutputConstraintString(StringRef Constraints) {
  // We should only have one =r return for the made up ASM type.
  SmallVector<StringRef> Tmp;
  SplitString(Constraints, Tmp, ",");
  std::string SafeConstraints("=r,");
  for (unsigned I = 0u; I != Tmp.size() - 1; ++I) {
    if (Tmp[I].starts_with('=') && (Tmp[I][1] == '&' || isalnum(Tmp[I][1])))
      continue;
    SafeConstraints.append(Tmp[I]).append({','});
  }
  SafeConstraints.append(Tmp.back());

  return SafeConstraints;
}

// Mutates indirect and inline ASM callsites iff aggregate argument/return types
// are present with the types replaced by i32 types. The change in types is
// noted in 'spv.mutated_callsites' metadata for later restoration. For ASM we
// also have to mutate the constraint string as IRTranslator tries to handle
// multiple outputs and expects an aggregate return type in their presence.
bool SPIRVRemoveAggregateTypesImpl::removeAggregateTypesFromCalls(Function *F) {
  if (F->isDeclaration() || F->isIntrinsic())
    return false;

  SmallVector<std::pair<CallBase *, FunctionType *>> Calls;
  for (auto &&I : instructions(F)) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (!CB->getCalledOperand() || CB->getCalledFunction())
        continue;
      if (CB->getType()->isAggregateType() ||
          any_of(CB->args(),
                 [](auto &&Arg) { return Arg->getType()->isAggregateType(); }))
        Calls.emplace_back(CB, nullptr);
    }
  }

  if (Calls.empty())
    return false;

  IRBuilder<> B(*F->getParent());

  unsigned MutatedCallIdx = 0;
  for (auto &&[CB, NewFnTy] : Calls) {
    SmallVector<std::pair<int, Type *>> ChangedTypes;
    SmallVector<Type *> NewArgTypes;

    Type *RetTy = CB->getType();
    if (RetTy->isAggregateType()) {
      ChangedTypes.emplace_back(-1, RetTy);
      RetTy = B.getInt32Ty();
    }

    for (auto &&Arg : CB->args()) {
      if (Arg->getType()->isAggregateType()) {
        NewArgTypes.push_back(B.getInt32Ty());
        ChangedTypes.emplace_back(Arg.getOperandNo(), Arg->getType());
      } else {
        NewArgTypes.push_back(Arg->getType());
      }
    }
    NewFnTy = FunctionType::get(RetTy, NewArgTypes,
                                CB->getFunctionType()->isVarArg());

    // Keyed via instruction metadata, not a name.
    std::string Key =
        ("spv.mutated_callsite." + F->getName() + "." + Twine(MutatedCallIdx++))
            .str();
    CB->setMetadata(
        "spv.mutated_callsite",
        MDNode::get(F->getContext(), MDString::get(F->getContext(), Key)));

    std::string Constraints;
    if (auto *ASM = dyn_cast<InlineAsm>(CB->getCalledOperand())) {
      Constraints = ASM->getConstraintString();

      CB->setCalledOperand(InlineAsm::get(
          NewFnTy, ASM->getAsmString(),
          fixMultiOutputConstraintString(Constraints), ASM->hasSideEffects(),
          ASM->isAlignStack(), ASM->getDialect(), ASM->canThrow()));
    }

    addFunctionTypeMutation(
        F->getParent()->getOrInsertNamedMetadata("spv.mutated_callsites"),
        std::move(ChangedTypes), Key, Constraints);
  }

  for (auto &&[CB, NewFTy] : Calls) {
    if (NewFTy->getReturnType() != CB->getType())
      TM.getSubtarget<SPIRVSubtarget>(*F).getSPIRVGlobalRegistry()->addMutated(
          CB, CB->getType());
    CB->mutateFunctionType(NewFTy);
  }

  return true;
}

bool SPIRVRemoveAggregateTypesImpl::runOnModule(Module &M) {
  bool Changed = false;
  std::vector<Function *> FuncsWorklist;
  for (Function &F : M) {
    Changed |= removeAggregateTypesFromCalls(&F);
    FuncsWorklist.push_back(&F);
  }

  for (auto *F : FuncsWorklist) {
    Function *NewF = removeAggregateTypesFromSignature(F);

    if (NewF != F) {
      F->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

PreservedAnalyses
SPIRVRemoveAggregateTypesPass::run(Module &M, ModuleAnalysisManager &AM) {
  return SPIRVRemoveAggregateTypesImpl(TM).runOnModule(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

ModulePass *
llvm::createSPIRVRemoveAggregateTypesPass(const SPIRVTargetMachine &TM) {
  return new SPIRVRemoveAggregateTypesLegacy(TM);
}
