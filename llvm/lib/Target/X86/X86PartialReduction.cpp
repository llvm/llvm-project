//===-- X86PartialReduction.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass looks for add instructions used by a horizontal reduction to see
// if we might be able to use pmaddwd or psadbw. Some cases of this require
// cross basic block knowledge and can't be done in SelectionDAG.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "X86TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "x86-partial-reduction"

namespace {

class X86PartialReduction : public FunctionPass {
  const DataLayout *DL;
  const X86Subtarget *ST;

public:
  static char ID; // Pass identification, replacement for typeid.

  X86PartialReduction() : FunctionPass(ID) { }

  bool runOnFunction(Function &Fn) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override {
    return "X86 Partial Reduction";
  }

private:
  bool tryMAddPattern(BinaryOperator *BO);
  bool tryMAddReplacement(Value *Op, BinaryOperator *Add);

  bool trySADPattern(BinaryOperator *BO);
  bool trySADReplacement(Value *Op, BinaryOperator *Add);
};
}

FunctionPass *llvm::createX86PartialReductionPass() {
  return new X86PartialReduction();
}

char X86PartialReduction::ID = 0;

INITIALIZE_PASS(X86PartialReduction, DEBUG_TYPE,
                "X86 Partial Reduction", false, false)

static bool isVectorReductionOp(const BinaryOperator &BO) {
  if (!BO.getType()->isVectorTy())
    return false;

  unsigned Opcode = BO.getOpcode();

  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    break;
  case Instruction::FAdd:
  case Instruction::FMul:
    if (auto *FPOp = dyn_cast<FPMathOperator>(&BO))
      if (FPOp->getFastMathFlags().isFast())
        break;
    LLVM_FALLTHROUGH;
  default:
    return false;
  }

  unsigned ElemNum = BO.getType()->getVectorNumElements();
  // Ensure the reduction size is a power of 2.
  if (!isPowerOf2_32(ElemNum))
    return false;

  unsigned ElemNumToReduce = ElemNum;

  // Do DFS search on the def-use chain from the given instruction. We only
  // allow four kinds of operations during the search until we reach the
  // instruction that extracts the first element from the vector:
  //
  //   1. The reduction operation of the same opcode as the given instruction.
  //
  //   2. PHI node.
  //
  //   3. ShuffleVector instruction together with a reduction operation that
  //      does a partial reduction.
  //
  //   4. ExtractElement that extracts the first element from the vector, and we
  //      stop searching the def-use chain here.
  //
  // 3 & 4 above perform a reduction on all elements of the vector. We push defs
  // from 1-3 to the stack to continue the DFS. The given instruction is not
  // a reduction operation if we meet any other instructions other than those
  // listed above.

  SmallVector<const User *, 16> UsersToVisit{&BO};
  SmallPtrSet<const User *, 16> Visited;
  bool ReduxExtracted = false;

  while (!UsersToVisit.empty()) {
    auto User = UsersToVisit.back();
    UsersToVisit.pop_back();
    if (!Visited.insert(User).second)
      continue;

    for (const auto *U : User->users()) {
      auto *Inst = dyn_cast<Instruction>(U);
      if (!Inst)
        return false;

      if (Inst->getOpcode() == Opcode || isa<PHINode>(U)) {
        if (auto *FPOp = dyn_cast<FPMathOperator>(Inst))
          if (!isa<PHINode>(FPOp) && !FPOp->getFastMathFlags().isFast())
            return false;
        UsersToVisit.push_back(U);
      } else if (auto *ShufInst = dyn_cast<ShuffleVectorInst>(U)) {
        // Detect the following pattern: A ShuffleVector instruction together
        // with a reduction that do partial reduction on the first and second
        // ElemNumToReduce / 2 elements, and store the result in
        // ElemNumToReduce / 2 elements in another vector.

        unsigned ResultElements = ShufInst->getType()->getVectorNumElements();
        if (ResultElements < ElemNum)
          return false;

        if (ElemNumToReduce == 1)
          return false;
        if (!isa<UndefValue>(U->getOperand(1)))
          return false;
        for (unsigned i = 0; i < ElemNumToReduce / 2; ++i)
          if (ShufInst->getMaskValue(i) != int(i + ElemNumToReduce / 2))
            return false;
        for (unsigned i = ElemNumToReduce / 2; i < ElemNum; ++i)
          if (ShufInst->getMaskValue(i) != -1)
            return false;

        // There is only one user of this ShuffleVector instruction, which
        // must be a reduction operation.
        if (!U->hasOneUse())
          return false;

        auto *U2 = dyn_cast<BinaryOperator>(*U->user_begin());
        if (!U2 || U2->getOpcode() != Opcode)
          return false;

        // Check operands of the reduction operation.
        if ((U2->getOperand(0) == U->getOperand(0) && U2->getOperand(1) == U) ||
            (U2->getOperand(1) == U->getOperand(0) && U2->getOperand(0) == U)) {
          UsersToVisit.push_back(U2);
          ElemNumToReduce /= 2;
        } else
          return false;
      } else if (isa<ExtractElementInst>(U)) {
        // At this moment we should have reduced all elements in the vector.
        if (ElemNumToReduce != 1)
          return false;

        auto *Val = dyn_cast<ConstantInt>(U->getOperand(1));
        if (!Val || !Val->isZero())
          return false;

        ReduxExtracted = true;
      } else
        return false;
    }
  }
  return ReduxExtracted;
}

bool X86PartialReduction::tryMAddReplacement(Value *Op, BinaryOperator *Add) {
  BasicBlock *BB = Add->getParent();

  auto *BO = dyn_cast<BinaryOperator>(Op);
  if (!BO || BO->getOpcode() != Instruction::Mul || !BO->hasOneUse() ||
      BO->getParent() != BB)
    return false;

  Value *LHS = BO->getOperand(0);
  Value *RHS = BO->getOperand(1);

  // LHS and RHS should be only used once or if they are the same then only
  // used twice. Only check this when SSE4.1 is enabled and we have zext/sext
  // instructions, otherwise we use punpck to emulate zero extend in stages. The
  // trunc/ we need to do likely won't introduce new instructions in that case.
  if (ST->hasSSE41()) {
    if (LHS == RHS) {
      if (!isa<Constant>(LHS) && !LHS->hasNUses(2))
        return false;
    } else {
      if (!isa<Constant>(LHS) && !LHS->hasOneUse())
        return false;
      if (!isa<Constant>(RHS) && !RHS->hasOneUse())
        return false;
    }
  }

  auto canShrinkOp = [&](Value *Op) {
    if (isa<Constant>(Op) && ComputeNumSignBits(Op, *DL, 0, nullptr, BO) > 16)
      return true;
    if (auto *Cast = dyn_cast<CastInst>(Op)) {
      if (Cast->getParent() == BB &&
          (Cast->getOpcode() == Instruction::SExt ||
           Cast->getOpcode() == Instruction::ZExt) &&
          ComputeNumSignBits(Op, *DL, 0, nullptr, BO) > 16)
        return true;
    }

    return false;
  };

  // Both Ops need to be shrinkable.
  if (!canShrinkOp(LHS) && !canShrinkOp(RHS))
    return false;

  IRBuilder<> Builder(Add);

  Type *MulTy = Op->getType();
  unsigned NumElts = MulTy->getVectorNumElements();

  // Extract even elements and odd elements and add them together. This will
  // be pattern matched by SelectionDAG to pmaddwd. This instruction will be
  // half the original width.
  SmallVector<uint32_t, 16> EvenMask(NumElts / 2);
  SmallVector<uint32_t, 16> OddMask(NumElts / 2);
  for (int i = 0, e = NumElts / 2; i != e; ++i) {
    EvenMask[i] = i * 2;
    OddMask[i] = i * 2 + 1;
  }
  Value *EvenElts = Builder.CreateShuffleVector(BO, BO, EvenMask);
  Value *OddElts = Builder.CreateShuffleVector(BO, BO, OddMask);
  Value *MAdd = Builder.CreateAdd(EvenElts, OddElts);

  // Concatenate zeroes to extend back to the original type.
  SmallVector<uint32_t, 32> ConcatMask(NumElts);
  std::iota(ConcatMask.begin(), ConcatMask.end(), 0);
  Value *Zero = Constant::getNullValue(MAdd->getType());
  Value *Concat = Builder.CreateShuffleVector(MAdd, Zero, ConcatMask);

  // Replaces the use of mul in the original Add with the pmaddwd and zeroes.
  Add->replaceUsesOfWith(BO, Concat);
  Add->setHasNoSignedWrap(false);
  Add->setHasNoUnsignedWrap(false);

  return true;
}

// Try to replace operans of this add with pmaddwd patterns.
bool X86PartialReduction::tryMAddPattern(BinaryOperator *BO) {
  if (!ST->hasSSE2())
    return false;

  // Need at least 8 elements.
  if (BO->getType()->getVectorNumElements() < 8)
    return false;

  // Element type should be i32.
  if (!BO->getType()->getVectorElementType()->isIntegerTy(32))
    return false;

  bool Changed = false;
  Changed |= tryMAddReplacement(BO->getOperand(0), BO);
  Changed |= tryMAddReplacement(BO->getOperand(1), BO);
  return Changed;
}

bool X86PartialReduction::trySADReplacement(Value *Op, BinaryOperator *Add) {
  // Operand should be a select.
  auto *SI = dyn_cast<SelectInst>(Op);
  if (!SI)
    return false;

  // Select needs to implement absolute value.
  Value *LHS, *RHS;
  auto SPR = matchSelectPattern(SI, LHS, RHS);
  if (SPR.Flavor != SPF_ABS)
    return false;

  // Need a subtract of two values.
  auto *Sub = dyn_cast<BinaryOperator>(LHS);
  if (!Sub || Sub->getOpcode() != Instruction::Sub)
    return false;

  // Look for zero extend from i8.
  auto getZeroExtendedVal = [](Value *Op) -> Value * {
    if (auto *ZExt = dyn_cast<ZExtInst>(Op))
      if (ZExt->getOperand(0)->getType()->getVectorElementType()->isIntegerTy(8))
        return ZExt->getOperand(0);

    return nullptr;
  };

  // Both operands of the subtract should be extends from vXi8.
  Value *Op0 = getZeroExtendedVal(Sub->getOperand(0));
  Value *Op1 = getZeroExtendedVal(Sub->getOperand(1));
  if (!Op0 || !Op1)
    return false;

  IRBuilder<> Builder(Add);

  Type *OpTy = Op->getType();
  unsigned NumElts = OpTy->getVectorNumElements();

  unsigned IntrinsicNumElts;
  Intrinsic::ID IID;
  if (ST->hasBWI() && NumElts >= 64) {
    IID = Intrinsic::x86_avx512_psad_bw_512;
    IntrinsicNumElts = 64;
  } else if (ST->hasAVX2() && NumElts >= 32) {
    IID = Intrinsic::x86_avx2_psad_bw;
    IntrinsicNumElts = 32;
  } else {
    IID = Intrinsic::x86_sse2_psad_bw;
    IntrinsicNumElts = 16;
  }

  Function *PSADBWFn = Intrinsic::getDeclaration(Add->getModule(), IID);

  if (NumElts < 16) {
    // Pad input with zeroes.
    SmallVector<uint32_t, 32> ConcatMask(16);
    for (unsigned i = 0; i != NumElts; ++i)
      ConcatMask[i] = i;
    for (unsigned i = NumElts; i != 16; ++i)
      ConcatMask[i] = (i % NumElts) + NumElts;

    Value *Zero = Constant::getNullValue(Op0->getType());
    Op0 = Builder.CreateShuffleVector(Op0, Zero, ConcatMask);
    Op1 = Builder.CreateShuffleVector(Op1, Zero, ConcatMask);
    NumElts = 16;
  }

  // Intrinsics produce vXi64 and need to be casted to vXi32.
  Type *I32Ty = VectorType::get(Builder.getInt32Ty(), IntrinsicNumElts / 4);

  assert(NumElts % IntrinsicNumElts == 0 && "Unexpected number of elements!");
  unsigned NumSplits = NumElts / IntrinsicNumElts;

  // First collect the pieces we need.
  SmallVector<Value *, 4> Ops(NumSplits);
  for (unsigned i = 0; i != NumSplits; ++i) {
    SmallVector<uint32_t, 64> ExtractMask(IntrinsicNumElts);
    std::iota(ExtractMask.begin(), ExtractMask.end(), i * IntrinsicNumElts);
    Value *ExtractOp0 = Builder.CreateShuffleVector(Op0, Op0, ExtractMask);
    Value *ExtractOp1 = Builder.CreateShuffleVector(Op1, Op0, ExtractMask);
    Ops[i] = Builder.CreateCall(PSADBWFn, {ExtractOp0, ExtractOp1});
    Ops[i] = Builder.CreateBitCast(Ops[i], I32Ty);
  }

  assert(isPowerOf2_32(NumSplits) && "Expected power of 2 splits");
  unsigned Stages = Log2_32(NumSplits);
  for (unsigned s = Stages; s > 0; --s) {
    unsigned NumConcatElts = Ops[0]->getType()->getVectorNumElements() * 2;
    for (unsigned i = 0; i != 1U << (s - 1); ++i) {
      SmallVector<uint32_t, 64> ConcatMask(NumConcatElts);
      std::iota(ConcatMask.begin(), ConcatMask.end(), 0);
      Ops[i] = Builder.CreateShuffleVector(Ops[i*2], Ops[i*2+1], ConcatMask);
    }
  }

  // At this point the final value should be in Ops[0]. Now we need to adjust
  // it to the final original type.
  NumElts = OpTy->getVectorNumElements();
  if (NumElts == 2) {
    // Extract down to 2 elements.
    Ops[0] = Builder.CreateShuffleVector(Ops[0], Ops[0], {0, 1});
  } else if (NumElts >= 8) {
    SmallVector<uint32_t, 32> ConcatMask(NumElts);
    unsigned SubElts = Ops[0]->getType()->getVectorNumElements();
    for (unsigned i = 0; i != SubElts; ++i)
      ConcatMask[i] = i;
    for (unsigned i = SubElts; i != NumElts; ++i)
      ConcatMask[i] = (i % SubElts) + SubElts;

    Value *Zero = Constant::getNullValue(Ops[0]->getType());
    Ops[0] = Builder.CreateShuffleVector(Ops[0], Zero, ConcatMask);
  }

  // Replaces the uses of Op in Add with the new sequence.
  Add->replaceUsesOfWith(Op, Ops[0]);
  Add->setHasNoSignedWrap(false);
  Add->setHasNoUnsignedWrap(false);

  return false;
}

bool X86PartialReduction::trySADPattern(BinaryOperator *BO) {
  if (!ST->hasSSE2())
    return false;

  // TODO: There's nothing special about i32, any integer type above i16 should
  // work just as well.
  if (!BO->getType()->getVectorElementType()->isIntegerTy(32))
    return false;

  bool Changed = false;
  Changed |= trySADReplacement(BO->getOperand(0), BO);
  Changed |= trySADReplacement(BO->getOperand(1), BO);
  return Changed;
}

bool X86PartialReduction::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<X86TargetMachine>();
  ST = TM.getSubtargetImpl(F);

  DL = &F.getParent()->getDataLayout();

  bool MadeChange = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *BO = dyn_cast<BinaryOperator>(&I);
      if (!BO)
        continue;

      if (!isVectorReductionOp(*BO))
        continue;

      if (BO->getOpcode() == Instruction::Add) {
        if (tryMAddPattern(BO)) {
          MadeChange = true;
          continue;
        }
        if (trySADPattern(BO)) {
          MadeChange = true;
          continue;
        }
      }
    }
  }

  return MadeChange;
}
