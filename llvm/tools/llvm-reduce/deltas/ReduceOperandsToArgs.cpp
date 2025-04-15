//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReduceOperandsToArgs.h"
#include "Utils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

static bool canReplaceFunction(const Function &F) {
  // TODO: Add controls to avoid ABI breaks (e.g. don't break main)
  return true;
}

static bool canReduceUse(Use &Op) {
  Value *Val = Op.get();
  Type *Ty = Val->getType();

  // Only replace operands that can be passed-by-value.
  if (!Ty->isFirstClassType())
    return false;

  // Don't pass labels/metadata as arguments.
  if (Ty->isLabelTy() || Ty->isMetadataTy() || Ty->isTokenTy())
    return false;

  // No need to replace values that are already arguments.
  if (isa<Argument>(Val))
    return false;

  // Do not replace literals.
  if (isa<ConstantData>(Val))
    return false;

  // Do not convert direct function calls to indirect calls.
  if (auto *CI = dyn_cast<CallBase>(Op.getUser()))
    if (&CI->getCalledOperandUse() == &Op)
      return false;

  return true;
}

/// Goes over OldF calls and replaces them with a call to NewF.
static void replaceFunctionCalls(Function *OldF, Function *NewF) {
  SmallVector<CallBase *> Callers;
  for (Use &U : OldF->uses()) {
    auto *CI = dyn_cast<CallBase>(U.getUser());
    if (!CI || !CI->isCallee(&U)) // RAUW can handle these fine.
      continue;

    Function *CalledF = CI->getCalledFunction();
    if (CalledF == OldF) {
      Callers.push_back(CI);
    } else {
      // The call may have undefined behavior by calling a function with a
      // mismatched signature. In this case, do not bother adjusting the
      // callsites to pad with any new arguments.

      // TODO: Better QoI to try to add new arguments to the end, and ignore
      // existing mismatches.
      assert(!CalledF && CI->getCalledOperand()->stripPointerCasts() == OldF &&
             "only expected call and function signature mismatch");
    }
  }

  // Call arguments for NewF.
  SmallVector<Value *> Args(NewF->arg_size(), nullptr);

  // Fill up the additional parameters with default values.
  for (auto ArgIdx : llvm::seq<size_t>(OldF->arg_size(), NewF->arg_size())) {
    Type *NewArgTy = NewF->getArg(ArgIdx)->getType();
    Args[ArgIdx] = getDefaultValue(NewArgTy);
  }

  for (CallBase *CI : Callers) {
    // Preserve the original function arguments.
    for (auto Z : zip_first(CI->args(), Args))
      std::get<1>(Z) = std::get<0>(Z);

    // Also preserve operand bundles.
    SmallVector<OperandBundleDef> OperandBundles;
    CI->getOperandBundlesAsDefs(OperandBundles);

    // Create the new function call.
    CallBase *NewCI;
    if (auto *II = dyn_cast<InvokeInst>(CI)) {
      NewCI = InvokeInst::Create(NewF, II->getNormalDest(), II->getUnwindDest(),
                                 Args, OperandBundles, CI->getName());
    } else {
      assert(isa<CallInst>(CI));
      NewCI = CallInst::Create(NewF, Args, OperandBundles, CI->getName());
    }
    NewCI->setCallingConv(NewF->getCallingConv());
    NewCI->setAttributes(CI->getAttributes());

    if (isa<FPMathOperator>(NewCI))
      NewCI->setFastMathFlags(CI->getFastMathFlags());

    NewCI->copyMetadata(*CI);

    // Do the replacement for this use.
    if (!CI->use_empty())
      CI->replaceAllUsesWith(NewCI);
    ReplaceInstWithInst(CI, NewCI);
  }
}

/// Add a new function argument to @p F for each use in @OpsToReplace, and
/// replace those operand values with the new function argument.
static void substituteOperandWithArgument(Function *OldF,
                                          ArrayRef<Use *> OpsToReplace) {
  if (OpsToReplace.empty())
    return;

  SetVector<Value *> UniqueValues;
  for (Use *Op : OpsToReplace)
    UniqueValues.insert(Op->get());

  // Determine the new function's signature.
  SmallVector<Type *> NewArgTypes;
  llvm::append_range(NewArgTypes, OldF->getFunctionType()->params());
  size_t ArgOffset = NewArgTypes.size();
  for (Value *V : UniqueValues)
    NewArgTypes.push_back(V->getType());
  FunctionType *FTy =
      FunctionType::get(OldF->getFunctionType()->getReturnType(), NewArgTypes,
                        OldF->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF =
      Function::Create(FTy, OldF->getLinkage(), OldF->getAddressSpace(),
                       OldF->getName(), OldF->getParent());

  // In order to preserve function order, we move NewF behind OldF
  NewF->removeFromParent();
  OldF->getParent()->getFunctionList().insertAfter(OldF->getIterator(), NewF);

  // Preserve the parameters of OldF.
  ValueToValueMapTy VMap;
  for (auto Z : zip_first(OldF->args(), NewF->args())) {
    Argument &OldArg = std::get<0>(Z);
    Argument &NewArg = std::get<1>(Z);

    NewArg.takeName(&OldArg); // Copy the name over...
    VMap[&OldArg] = &NewArg;  // Add mapping to VMap
  }

  LLVMContext &Ctx = OldF->getContext();

  // Adjust the new parameters.
  ValueToValueMapTy OldValMap;
  for (auto Z : zip_first(UniqueValues, drop_begin(NewF->args(), ArgOffset))) {
    Value *OldVal = std::get<0>(Z);
    Argument &NewArg = std::get<1>(Z);

    NewArg.setName(OldVal->getName());
    OldValMap[OldVal] = &NewArg;
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, OldF, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", /*CodeInfo=*/nullptr);

  // Replace the actual operands.
  for (Use *Op : OpsToReplace) {
    Argument *NewArg = cast<Argument>(OldValMap.lookup(Op->get()));
    auto *NewUser = cast<Instruction>(VMap.lookup(Op->getUser()));

    // Try to preserve any information contained metadata annotations as the
    // equivalent parameter attributes if possible.
    if (auto *MDSrcInst = dyn_cast<Instruction>(Op)) {
      AttrBuilder AB(Ctx);
      NewArg->addAttrs(AB.addFromEquivalentMetadata(*MDSrcInst));
    }

    if (PHINode *NewPhi = dyn_cast<PHINode>(NewUser)) {
      PHINode *OldPhi = cast<PHINode>(Op->getUser());
      BasicBlock *OldBB = OldPhi->getIncomingBlock(*Op);
      NewPhi->setIncomingValueForBlock(cast<BasicBlock>(VMap.lookup(OldBB)),
                                       NewArg);
    } else
      NewUser->setOperand(Op->getOperandNo(), NewArg);
  }

  // Replace all OldF uses with NewF.
  replaceFunctionCalls(OldF, NewF);

  NewF->takeName(OldF);
  OldF->replaceAllUsesWith(NewF);
  OldF->eraseFromParent();
}

void llvm::reduceOperandsToArgsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();

  SmallVector<Use *> OperandsToReduce;
  for (Function &F : make_early_inc_range(Program.functions())) {
    if (!canReplaceFunction(F))
      continue;
    OperandsToReduce.clear();
    for (Instruction &I : instructions(&F)) {
      for (Use &Op : I.operands()) {
        if (!canReduceUse(Op))
          continue;
        if (O.shouldKeep())
          continue;

        OperandsToReduce.push_back(&Op);
      }
    }

    substituteOperandWithArgument(&F, OperandsToReduce);
  }
}
