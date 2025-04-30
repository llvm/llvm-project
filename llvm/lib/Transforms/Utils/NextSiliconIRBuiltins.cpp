//===- NextSiliconIRBuiltins.cpp - Convert unsupported instructions to
// NextSilicon IR builtins ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// NextSiliconIRBuiltins pass converts unsupported code into the code based on
// the NextSilicon IR builtins.
//
//===------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/NextSiliconIRBuiltins.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>

using namespace llvm;

#define DEBUG_TYPE "ir-builtins"

// Convert 128-bit cmpxchg instruction to call to compiler generated function.
// This function will be defined in the nextatomic static IR library.
static bool convertCmpXchg(Module &M, AtomicCmpXchgInst *CmpXchg) {
  LLVMContext &Context = M.getContext();
  auto *Int64Ty = Type::getInt64Ty(Context);
  auto *Int1Ty = Type::getInt1Ty(Context);

  auto *Ptr = CmpXchg->getPointerOperand();
  auto *Cmp = CmpXchg->getCompareOperand();
  auto *NewVal = CmpXchg->getNewValOperand();
  auto *CmpTy = Cmp->getType();
  auto *NewValTy = NewVal->getType();
  auto *CmpIntTy = dyn_cast<IntegerType>(CmpTy);
  auto *NewValIntTy = dyn_cast<IntegerType>(NewValTy);
  if (!CmpIntTy || !NewValIntTy)
    return false;

  // Check if the cmpxchg instruction is 128-bit.
  if (CmpIntTy->getBitWidth() != 128 || NewValIntTy->getBitWidth() != 128)
    return false;

  // Conversion:
  // Input:
  //   %9 = load i128, ptr %1, align 16
  //   %10 = cmpxchg ptr %0, i128 %9, i128 %8 acquire monotonic, align 16
  // Uutput:
  //   %10 = call i128 @__ns_atomic_compare_exchange_16(ptr %0, ptr %1,
  //   i128 %8, i32 1, i32 0)

  // Find load instruction for Cmp to get the address.
  LoadInst *CmpLdInst = dyn_cast<LoadInst>(Cmp);
  if (!CmpLdInst)
    return false;
  // Find the address of Cmp from Load Inst.
  Value *CmpAddr = CmpLdInst->getPointerOperand();

  // Create a new function call to __ns_atomic_compare_exchange_16.
  // The function signature is:
  // bool __ns_atomic_compare_exchange_16(int128 *Ptr, int128 *CmpAddr, int128
  // NewVal, int32_t SuccessOrder, int32_t FailureOrder);
  FunctionCallee FC = M.getOrInsertFunction(
      "__ns_atomic_compare_exchange_16", Int1Ty, Ptr->getType(),
      CmpAddr->getType(), Int64Ty, Int64Ty, Type::getInt32Ty(Context),
      Type::getInt32Ty(Context));
  IRBuilder<> Builder(CmpXchg);
  // Split 128-bit ints NewVal into two 64-bit ints.
  Value *NewValLow = Builder.CreateTrunc(NewVal, Type::getInt64Ty(Context));
  Value *NewValShift = Builder.CreateAShr(NewVal, 64);
  Value *NewValHigh =
      Builder.CreateTrunc(NewValShift, Type::getInt64Ty(Context));
  // Except for the CmpAddr, other arguments are the cmpxchg operands.
  Value *Args[] = {Ptr,
                   CmpAddr,
                   NewValLow,
                   NewValHigh,
                   Builder.getInt32(1),
                   Builder.getInt32(0)};
  CallInst *NewCmpXchg = Builder.CreateCall(FC, Args);
  // Get all uses of CmpXchg.
  SmallVector<Use *, 8> Uses;
  for (Use &U : CmpXchg->uses())
    Uses.push_back(&U);
  // Replate all uses of CmpXchg[0] with load from CmpAddr.
  CmpLdInst = Builder.CreateLoad(CmpIntTy, CmpAddr);
  // Create aggregate value {CmpLdInst, NewCmpXchg} and RAUW.
  Value *PoisonVal = PoisonValue::get(CmpXchg->getType());
  Builder.CreateInsertValue(PoisonVal, CmpLdInst, 0);
  Builder.CreateInsertValue(PoisonVal, NewCmpXchg, 1);
  CmpXchg->replaceAllUsesWith(PoisonVal);
  CmpXchg->eraseFromParent();

  return true;
}

static bool processInsts(Function &F) {
  Module &M = *F.getParent();
  bool Changed = false;

  for (auto II = inst_begin(&F), IE = inst_end(&F); II != IE;) {
    Instruction *I = &*II++;
    switch (I->getOpcode()) {
    case Instruction::AtomicCmpXchg: {
      // Search for 128-bit cmpxchg instructions, which are not supported in NS
      // system, and should be converted into call to compiler generated
      // function.
      AtomicCmpXchgInst *CmpXchg = dyn_cast<AtomicCmpXchgInst>(I);
      if (!CmpXchg)
        break;
      Changed |= convertCmpXchg(M, CmpXchg);

      break;
    }
    default: {
      continue;
    }
    }
  }

  return Changed;
}

static bool runImpl(Module &M) {
  bool Changed = false;

  for (Function &F : M.functions())
    Changed |= processInsts(F);

  return Changed;
}

PreservedAnalyses NextSiliconIRBuiltinsPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  if (!runImpl(M))
    return PreservedAnalyses::all();

  // Be conservative for now, optimize later if necessary
  return PreservedAnalyses::none();
}