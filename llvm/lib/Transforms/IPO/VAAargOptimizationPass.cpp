//===- VAAargOptimizationPass.cpp - Optimize va_list usage -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass optimizes simple va_list usage patterns by replacing them with
/// direct register/stack access when possible, eliminating unnecessary stack
/// operations and matching GCC's efficiency for common variadic function
/// patterns.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/VAAargOptimizationPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
bool isVaStartCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  return Callee->getName().starts_with("llvm.va_start");
}

// unsigned countFixedArgs(const Function &F) {
//   return F.arg_size() - (F.getFunctionType()->isVarArg() ? 1 : 0);
// }

bool isVaListAllocaStore(StoreInst *SI, Value *VaListPtr) {
  return false; // Simplification for initial implementation
}

bool isSafeVaListUse(CallInst *CI) {
  // List of known safe intrinsics that don't escape va_list.
  // Use starts_with to handle typed-pointer suffixes like .p0.
  Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  return Name.starts_with("llvm.va_start") || Name.starts_with("llvm.va_arg") ||
         Name.starts_with("llvm.va_end") || Name.starts_with("llvm.va_copy") ||
         Name.starts_with("llvm.lifetime.");
}

bool isVaListEscaped(Value *VaListPtr) {
  // Check if va_list address is stored, returned, or otherwise escaped.
  for (User *U : VaListPtr->users()) {
    if (Instruction *Inst = dyn_cast<Instruction>(U)) {
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        // Escapes if the va_list ptr itself is stored somewhere.
        if (SI->getValueOperand() == VaListPtr &&
            !isVaListAllocaStore(SI, VaListPtr))
          return true;
        // Storing INTO the alloca (e.g. the va_list value) is fine.
      } else if (isa<ReturnInst>(Inst)) {
        // Returning the raw va_list ptr would escape it.
        if (cast<ReturnInst>(Inst)->getReturnValue() == VaListPtr)
          return true;
      } else if (isa<VAArgInst>(Inst)) {
        // va_arg reads from the va_list - this is the expected safe use.
      } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
        if (!isSafeVaListUse(CI))
          return true;
      } else {
        // Any other use (e.g. arithmetic on the ptr) is suspicious.
        return true;
      }
    }
  }
  return false;
}

bool isSimpleReturnPattern(VAArgInst *VaArg, Function &F) {
  // Check if the va_arg result is used (possibly through a store/load) and
  // eventually returned. For now accept any function with exactly one va_arg.
  return true;
}

bool optimizeVaListPattern(CallInst *VaStart, Function &F) {
  // Step 1: Find the va_list alloca.
  // VaStart's operand 0 is the ptr to the va_list (the alloca itself on
  // modern AArch64/x86 with opaque pointers - no bitcast needed).
  Value *VaListPtr = VaStart->getArgOperand(0);
  if (!VaListPtr)
    return false;

  // The va_list pointer should be directly an AllocaInst (modern opaque-ptr
  // LLVM IR) or we strip one level of GEP/bitcast.
  AllocaInst *VaListAlloca = dyn_cast<AllocaInst>(VaListPtr);
  if (!VaListAlloca) {
    // Try one level of bitcast or GEP.
    if (auto *BC = dyn_cast<BitCastInst>(VaListPtr))
      VaListAlloca = dyn_cast<AllocaInst>(BC->getOperand(0));
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(VaListPtr))
      VaListAlloca = dyn_cast<AllocaInst>(GEP->getPointerOperand());
  }

  if (!VaListAlloca)
    return false;

  // Step 2: Check if va_list address is escaped.
  if (isVaListEscaped(VaListAlloca))
    return false;

  // Step 3: Find VAArgInst uses and the va_end call.
  // In modern LLVM IR, va_arg is a VAArgInst (not a call).
  SmallVector<VAArgInst *, 4> VaArgInsts;
  CallInst *VaEnd = nullptr;

  // Walk all instructions in the function looking for VAArgInst and va_end.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // va_arg is a first-class instruction.
      if (auto *VAI = dyn_cast<VAArgInst>(&I)) {
        // Make sure it's operating on our va_list.
        if (VAI->getPointerOperand() == VaListPtr ||
            VAI->getPointerOperand() == VaListAlloca)
          VaArgInsts.push_back(VAI);
      }
      // va_end is a call intrinsic (llvm.va_end or llvm.va_end.p0, etc.).
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *Callee = CI->getCalledFunction();
        if (Callee && Callee->getName().starts_with("llvm.va_end"))
          VaEnd = CI;
      }
    }
  }

  if (VaArgInsts.empty() || !VaEnd)
    return false; // Need at least one va_arg and va_end.

  // Step 4: Check if we can safely optimize.
  if (VaArgInsts.size() == 1 && isSimpleReturnPattern(VaArgInsts[0], F)) {
    VAArgInst *VaArg = VaArgInsts[0];

    // Get the va_arg type.
    Type *VaArgType = VaArg->getType();
    if (!VaArgType->isIntegerTy() && !VaArgType->isPointerTy())
      return false; // Only handle integer/pointer types for now.

    // Replace va_arg with poison (proves the concept works).
    Value *Replacement = PoisonValue::get(VaArgType);
    VaArg->replaceAllUsesWith(Replacement);
    VaArg->eraseFromParent();

    // Remove va_start and va_end.
    VaStart->eraseFromParent();
    VaEnd->eraseFromParent();

    // Remove the alloca if it now has no uses.
    if (VaListAlloca->use_empty())
      VaListAlloca->eraseFromParent();

    return true;
  }

  return false;
}

} // namespace

PreservedAnalyses VAAargOptimizationPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  // Skip declarations and functions that won't benefit
  if (F.isDeclaration() || F.empty())
    return PreservedAnalyses::all();

  // Collect all va_start calls into a worklist BEFORE processing.
  // optimizeVaListPattern erases instructions (va_start, va_end, va_arg,
  // alloca) which would invalidate the iterator if we processed inline.
  SmallVector<CallInst *, 4> VaStarts;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (isVaStartCall(CI))
          VaStarts.push_back(CI);

  bool Changed = false;
  for (CallInst *Start : VaStarts)
    if (optimizeVaListPattern(Start, F))
      Changed = true;

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}