//===- InferAddressSpacePrepre.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InferAddressSpacesPrepare.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Scalar.h"
#include <limits>

#define DEBUG_TYPE "infer-address-spaces-prepare"

using namespace llvm;

extern cl::opt<bool> AssumeDefaultIsFlatAddressSpace;
static const unsigned UninitializedAddressSpace =
    std::numeric_limits<unsigned>::max();

namespace {

class InferAddressSpacesPrepare : public FunctionPass {
  unsigned FlatAddrSpace = 0;

public:
  static char ID;

  InferAddressSpacesPrepare()
      : FunctionPass(ID), FlatAddrSpace(UninitializedAddressSpace) {
    initializeInferAddressSpacesPass(*PassRegistry::getPassRegistry());
  }
  InferAddressSpacesPrepare(unsigned AS) : FunctionPass(ID), FlatAddrSpace(AS) {
    initializeInferAddressSpacesPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

class InferAddressSpacesPrepareImpl {
  Function *F = nullptr;
  const TargetTransformInfo *TTI = nullptr;

  /// Target specific address space which uses of should be replaced if
  /// possible.
  unsigned FlatAddrSpace = 0;

public:
  InferAddressSpacesPrepareImpl(const TargetTransformInfo *TTI,
                                unsigned FlatAddrSpace)
      : TTI(TTI), FlatAddrSpace(FlatAddrSpace) {}
  bool run(Function &F);
};

// Find the ptrtoint instruction that the value of inttoptr is derived from.
static Value *getPtrToIntRecursively(Value *Val, int Depth) {
  Instruction *Inst = dyn_cast<Instruction>(Val);
  if (!Inst)
    return nullptr;
  if (auto *P2I = dyn_cast<PtrToIntInst>(Val))
    return P2I;

  Depth--;
  if (Depth <= 0)
    return nullptr;

  if (!isa<BinaryOperator>(Inst) && !isa<UnaryInstruction>(Inst))
    return nullptr;

  // Recursively look up each operand to find the ptrtoint instruction.
  for (unsigned J = 0, E = Inst->getNumOperands(); J != E; ++J) {
    Inst->getOperand(J)->dump();
    if (auto *P2I = getPtrToIntRecursively(Inst->getOperand(J), Depth))
      return P2I;
  }
  return nullptr;
}

bool InferAddressSpacesPrepareImpl::run(Function &CurFn) {
  bool Changed = false;
  F = &CurFn;

  if (AssumeDefaultIsFlatAddressSpace)
    FlatAddrSpace = 0;

  if (FlatAddrSpace == UninitializedAddressSpace) {
    FlatAddrSpace = TTI->getFlatAddressSpace();
    if (FlatAddrSpace == UninitializedAddressSpace)
      return false;
  }

  SmallVector<std::pair<unsigned, unsigned>, 2> AsLSBSizePairs;
  if (!TTI->getMutableLSBSizeInAddrSpaces(AsLSBSizePairs))
    return false;

  SmallVector<std::pair<Instruction *, Instruction *>, 8> IntPtrPairs;
  for (Instruction &I : instructions(F)) {
    auto *I2P = dyn_cast<IntToPtrInst>(&I);
    if (!I2P)
      continue;
    if (I2P->getAddressSpace() != FlatAddrSpace)
      continue;
    auto *P2I = getPtrToIntRecursively(I2P, 6);
    if (!P2I)
      continue;
    IntPtrPairs.push_back({I2P, cast<PtrToIntInst>(P2I)});
  }

  if (!IntPtrPairs.empty())
    Changed = true;
  // Create ptr.bit.diff intrinsic to infer if there is any bit changes in the
  // address high bits.
  // call void @llvm.ptr.bit.diff.i64(ptr %src, ptr %derived, i32 AS, i64 diff)
  // The InstCombine pass may deduce the diff to be 0 in compiling time, so that
  // InferAddressSpaces pass knows the address space doesn't change after
  // ptrtoint and integer computations.
  for (auto Iter : IntPtrPairs) {
    IntToPtrInst *I2P = cast<IntToPtrInst>(Iter.first);
    PtrToIntInst *P2I = cast<PtrToIntInst>(Iter.second);
    IRBuilder<> B(I2P->getParent(), ++cast<Instruction>(I2P)->getIterator());

    for (auto [AS, LSBSize] : AsLSBSizePairs) {
      // Duplicate the Xor instruction to facilitate InstCombine.
      auto *Xor = B.CreateXor(I2P->getOperand(0), P2I);
      auto DL = F->getDataLayout();
      APInt Mask =
          ~((APInt(DL.getAddressSizeInBits(FlatAddrSpace), 1) << LSBSize) - 1);
      auto *And = B.CreateAnd(
          Xor, ConstantInt::get(I2P->getOperand(0)->getType(), Mask));
      B.CreateIntrinsic(
          Type::getVoidTy(F->getContext()), Intrinsic::ptr_bit_diff,
          {I2P, P2I->getPointerOperand(),
           ConstantInt::get(Type::getInt32Ty(F->getContext()), AS), And});
    }
  }

  return Changed;
}

} // end anonymous namespace

char InferAddressSpacesPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(InferAddressSpacesPrepare, DEBUG_TYPE,
                      "Infer address spaces prepare", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(InferAddressSpacesPrepare, DEBUG_TYPE,
                    "Infer address spaces prepare", false, false)

PreservedAnalyses
InferAddressSpacesPreparePass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = InferAddressSpacesPrepareImpl(
                     &AM.getResult<TargetIRAnalysis>(F), FlatAddrSpace)
                     .run(F);
  if (Changed) {
    PreservedAnalyses PA;
    return PA;
  }
  return PreservedAnalyses::all();
}

// Prepare for address spaces inferring
bool InferAddressSpacesPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  return InferAddressSpacesPrepareImpl(
             &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F),
             FlatAddrSpace)
      .run(F);
}

FunctionPass *llvm::createInferAddressSpacesPreparePass(unsigned AddressSpace) {
  return new InferAddressSpacesPrepare(AddressSpace);
}

InferAddressSpacesPreparePass::InferAddressSpacesPreparePass()
    : FlatAddrSpace(UninitializedAddressSpace) {}
InferAddressSpacesPreparePass::InferAddressSpacesPreparePass(
    unsigned AddressSpace)
    : FlatAddrSpace(AddressSpace) {}
