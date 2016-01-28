//===---- DemandedBits.cpp - Determine demanded bits ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a demanded bits analysis. A demanded bit is one that
// contributes to a result; bits that are not demanded can be either zero or
// one without affecting control or data flow. For example in this sequence:
//
//   %1 = add i32 %x, %y
//   %2 = trunc i32 %1 to i16
//
// Only the lowest 16 bits of %1 are demanded; the rest are removed by the
// trunc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "demanded-bits"

char DemandedBits::ID = 0;
INITIALIZE_PASS_BEGIN(DemandedBits, "demanded-bits", "Demanded bits analysis",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DemandedBits, "demanded-bits", "Demanded bits analysis",
                    false, false)

DemandedBits::DemandedBits() : FunctionPass(ID), F(nullptr), Analyzed(false) {
  initializeDemandedBitsPass(*PassRegistry::getPassRegistry());
}

void DemandedBits::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesAll();
}

static bool isAlwaysLive(Instruction *I) {
  return isa<TerminatorInst>(I) || isa<DbgInfoIntrinsic>(I) ||
      I->isEHPad() || I->mayHaveSideEffects();
}

void DemandedBits::determineLiveOperandBits(
    const Instruction *UserI, const Instruction *I, unsigned OperandNo,
    const APInt &AOut, APInt &AB, APInt &KnownZero, APInt &KnownOne,
    APInt &KnownZero2, APInt &KnownOne2) {
  unsigned BitWidth = AB.getBitWidth();

  // We're called once per operand, but for some instructions, we need to
  // compute known bits of both operands in order to determine the live bits of
  // either (when both operands are instructions themselves). We don't,
  // however, want to do this twice, so we cache the result in APInts that live
  // in the caller. For the two-relevant-operands case, both operand values are
  // provided here.
  auto ComputeKnownBits =
      [&](unsigned BitWidth, const Value *V1, const Value *V2) {
        const DataLayout &DL = I->getModule()->getDataLayout();
        KnownZero = APInt(BitWidth, 0);
        KnownOne = APInt(BitWidth, 0);
        computeKnownBits(const_cast<Value *>(V1), KnownZero, KnownOne, DL, 0,
                         AC, UserI, DT);

        if (V2) {
          KnownZero2 = APInt(BitWidth, 0);
          KnownOne2 = APInt(BitWidth, 0);
          computeKnownBits(const_cast<Value *>(V2), KnownZero2, KnownOne2, DL,
                           0, AC, UserI, DT);
        }
      };

  switch (UserI->getOpcode()) {
  default: break;
  case Instruction::Call:
  case Instruction::Invoke:
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(UserI))
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::bswap:
        // The alive bits of the input are the swapped alive bits of
        // the output.
        AB = AOut.byteSwap();
        break;
      case Intrinsic::ctlz:
        if (OperandNo == 0) {
          // We need some output bits, so we need all bits of the
          // input to the left of, and including, the leftmost bit
          // known to be one.
          ComputeKnownBits(BitWidth, I, nullptr);
          AB = APInt::getHighBitsSet(BitWidth,
                 std::min(BitWidth, KnownOne.countLeadingZeros()+1));
        }
        break;
      case Intrinsic::cttz:
        if (OperandNo == 0) {
          // We need some output bits, so we need all bits of the
          // input to the right of, and including, the rightmost bit
          // known to be one.
          ComputeKnownBits(BitWidth, I, nullptr);
          AB = APInt::getLowBitsSet(BitWidth,
                 std::min(BitWidth, KnownOne.countTrailingZeros()+1));
        }
        break;
      }
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // Find the highest live output bit. We don't need any more input
    // bits than that (adds, and thus subtracts, ripple only to the
    // left).
    AB = APInt::getLowBitsSet(BitWidth, AOut.getActiveBits());
    break;
  case Instruction::Shl:
    if (OperandNo == 0)
      if (ConstantInt *CI =
            dyn_cast<ConstantInt>(UserI->getOperand(1))) {
        uint64_t ShiftAmt = CI->getLimitedValue(BitWidth-1);
        AB = AOut.lshr(ShiftAmt);

        // If the shift is nuw/nsw, then the high bits are not dead
        // (because we've promised that they *must* be zero).
        const ShlOperator *S = cast<ShlOperator>(UserI);
        if (S->hasNoSignedWrap())
          AB |= APInt::getHighBitsSet(BitWidth, ShiftAmt+1);
        else if (S->hasNoUnsignedWrap())
          AB |= APInt::getHighBitsSet(BitWidth, ShiftAmt);
      }
    break;
  case Instruction::LShr:
    if (OperandNo == 0)
      if (ConstantInt *CI =
            dyn_cast<ConstantInt>(UserI->getOperand(1))) {
        uint64_t ShiftAmt = CI->getLimitedValue(BitWidth-1);
        AB = AOut.shl(ShiftAmt);

        // If the shift is exact, then the low bits are not dead
        // (they must be zero).
        if (cast<LShrOperator>(UserI)->isExact())
          AB |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
      }
    break;
  case Instruction::AShr:
    if (OperandNo == 0)
      if (ConstantInt *CI =
            dyn_cast<ConstantInt>(UserI->getOperand(1))) {
        uint64_t ShiftAmt = CI->getLimitedValue(BitWidth-1);
        AB = AOut.shl(ShiftAmt);
        // Because the high input bit is replicated into the
        // high-order bits of the result, if we need any of those
        // bits, then we must keep the highest input bit.
        if ((AOut & APInt::getHighBitsSet(BitWidth, ShiftAmt))
            .getBoolValue())
          AB.setBit(BitWidth-1);

        // If the shift is exact, then the low bits are not dead
        // (they must be zero).
        if (cast<AShrOperator>(UserI)->isExact())
          AB |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
      }
    break;
  case Instruction::And:
    AB = AOut;

    // For bits that are known zero, the corresponding bits in the
    // other operand are dead (unless they're both zero, in which
    // case they can't both be dead, so just mark the LHS bits as
    // dead).
    if (OperandNo == 0) {
      ComputeKnownBits(BitWidth, I, UserI->getOperand(1));
      AB &= ~KnownZero2;
    } else {
      if (!isa<Instruction>(UserI->getOperand(0)))
        ComputeKnownBits(BitWidth, UserI->getOperand(0), I);
      AB &= ~(KnownZero & ~KnownZero2);
    }
    break;
  case Instruction::Or:
    AB = AOut;

    // For bits that are known one, the corresponding bits in the
    // other operand are dead (unless they're both one, in which
    // case they can't both be dead, so just mark the LHS bits as
    // dead).
    if (OperandNo == 0) {
      ComputeKnownBits(BitWidth, I, UserI->getOperand(1));
      AB &= ~KnownOne2;
    } else {
      if (!isa<Instruction>(UserI->getOperand(0)))
        ComputeKnownBits(BitWidth, UserI->getOperand(0), I);
      AB &= ~(KnownOne & ~KnownOne2);
    }
    break;
  case Instruction::Xor:
  case Instruction::PHI:
    AB = AOut;
    break;
  case Instruction::Trunc:
    AB = AOut.zext(BitWidth);
    break;
  case Instruction::ZExt:
    AB = AOut.trunc(BitWidth);
    break;
  case Instruction::SExt:
    AB = AOut.trunc(BitWidth);
    // Because the high input bit is replicated into the
    // high-order bits of the result, if we need any of those
    // bits, then we must keep the highest input bit.
    if ((AOut & APInt::getHighBitsSet(AOut.getBitWidth(),
                                      AOut.getBitWidth() - BitWidth))
        .getBoolValue())
      AB.setBit(BitWidth-1);
    break;
  case Instruction::Select:
    if (OperandNo != 0)
      AB = AOut;
    break;
  case Instruction::ICmp:
    // Count the number of leading zeroes in each operand.
    ComputeKnownBits(BitWidth, UserI->getOperand(0), UserI->getOperand(1));
    auto NumLeadingZeroes = std::min(KnownZero.countLeadingOnes(),
                                     KnownZero2.countLeadingOnes());
    AB = ~APInt::getHighBitsSet(BitWidth, NumLeadingZeroes);
    break;
  }
}

bool DemandedBits::runOnFunction(Function& Fn) {
  F = &Fn;
  Analyzed = false;
  return false;
}

void DemandedBits::performAnalysis() {
  if (Analyzed)
    // Analysis already completed for this function.
    return;
  Analyzed = true;
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  
  Visited.clear();
  AliveBits.clear();

  SmallVector<Instruction*, 128> Worklist;

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : instructions(*F)) {
    if (!isAlwaysLive(&I))
      continue;

    DEBUG(dbgs() << "DemandedBits: Root: " << I << "\n");
    // For integer-valued instructions, set up an initial empty set of alive
    // bits and add the instruction to the work list. For other instructions
    // add their operands to the work list (for integer values operands, mark
    // all bits as live).
    if (IntegerType *IT = dyn_cast<IntegerType>(I.getType())) {
      if (!AliveBits.count(&I)) {
        AliveBits[&I] = APInt(IT->getBitWidth(), 0);
        Worklist.push_back(&I);
      }

      continue;
    }

    // Non-integer-typed instructions...
    for (Use &OI : I.operands()) {
      if (Instruction *J = dyn_cast<Instruction>(OI)) {
        if (IntegerType *IT = dyn_cast<IntegerType>(J->getType()))
          AliveBits[J] = APInt::getAllOnesValue(IT->getBitWidth());
        Worklist.push_back(J);
      }
    }
    // To save memory, we don't add I to the Visited set here. Instead, we
    // check isAlwaysLive on every instruction when searching for dead
    // instructions later (we need to check isAlwaysLive for the
    // integer-typed instructions anyway).
  }

  // Propagate liveness backwards to operands.
  while (!Worklist.empty()) {
    Instruction *UserI = Worklist.pop_back_val();

    DEBUG(dbgs() << "DemandedBits: Visiting: " << *UserI);
    APInt AOut;
    if (UserI->getType()->isIntegerTy()) {
      AOut = AliveBits[UserI];
      DEBUG(dbgs() << " Alive Out: " << AOut);
    }
    DEBUG(dbgs() << "\n");

    if (!UserI->getType()->isIntegerTy())
      Visited.insert(UserI);

    APInt KnownZero, KnownOne, KnownZero2, KnownOne2;
    // Compute the set of alive bits for each operand. These are anded into the
    // existing set, if any, and if that changes the set of alive bits, the
    // operand is added to the work-list.
    for (Use &OI : UserI->operands()) {
      if (Instruction *I = dyn_cast<Instruction>(OI)) {
        if (IntegerType *IT = dyn_cast<IntegerType>(I->getType())) {
          unsigned BitWidth = IT->getBitWidth();
          APInt AB = APInt::getAllOnesValue(BitWidth);
          if (UserI->getType()->isIntegerTy() && !AOut &&
              !isAlwaysLive(UserI)) {
            AB = APInt(BitWidth, 0);
          } else {
            // If all bits of the output are dead, then all bits of the input
            // Bits of each operand that are used to compute alive bits of the
            // output are alive, all others are dead.
            determineLiveOperandBits(UserI, I, OI.getOperandNo(), AOut, AB,
                                     KnownZero, KnownOne,
                                     KnownZero2, KnownOne2);
          }

          // If we've added to the set of alive bits (or the operand has not
          // been previously visited), then re-queue the operand to be visited
          // again.
          APInt ABPrev(BitWidth, 0);
          auto ABI = AliveBits.find(I);
          if (ABI != AliveBits.end())
            ABPrev = ABI->second;

          APInt ABNew = AB | ABPrev;
          if (ABNew != ABPrev || ABI == AliveBits.end()) {
            AliveBits[I] = std::move(ABNew);
            Worklist.push_back(I);
          }
        } else if (!Visited.count(I)) {
          Worklist.push_back(I);
        }
      }
    }
  }
}

APInt DemandedBits::getDemandedBits(Instruction *I) {
  performAnalysis();
  
  const DataLayout &DL = I->getParent()->getModule()->getDataLayout();
  if (AliveBits.count(I))
    return AliveBits[I];
  return APInt::getAllOnesValue(DL.getTypeSizeInBits(I->getType()));
}

bool DemandedBits::isInstructionDead(Instruction *I) {
  performAnalysis();

  return !Visited.count(I) && AliveBits.find(I) == AliveBits.end() &&
    !isAlwaysLive(I);
}

void DemandedBits::print(raw_ostream &OS, const Module *M) const {
  // This is gross. But the alternative is making all the state mutable
  // just because of this one debugging method.
  const_cast<DemandedBits*>(this)->performAnalysis();
  for (auto &KV : AliveBits) {
    OS << "DemandedBits: 0x" << utohexstr(KV.second.getLimitedValue()) << " for "
       << *KV.first << "\n";
  }
}

FunctionPass *llvm::createDemandedBitsPass() {
  return new DemandedBits();
}
