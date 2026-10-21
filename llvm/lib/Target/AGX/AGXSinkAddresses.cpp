//===-- AGXSinkAddresses.cpp - Device addresses next to their accesses ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device loads and stores take a 64-bit base and a sign or zero extended
// 32-bit index scaled by 1, 2 or 4 elements, or a 16-bit element offset.
// Instruction selection folds an address of that form only when its pieces
// are in the block of the access. Addresses computed once and used in later
// blocks (after the expansion of divisions or the structurizer's flow
// blocks, or as PTX computes them, through integer arithmetic) are copied
// into the block of each access that can fold them; the copies cost
// nothing, and the shared 64-bit additions they replace disappear.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-sink-addresses"

namespace {

class AGXSinkAddresses : public FunctionPass {
public:
  static char ID;
  AGXSinkAddresses() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Sink Addresses"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // namespace

char AGXSinkAddresses::ID = 0;

INITIALIZE_PASS(AGXSinkAddresses, DEBUG_TYPE, "AGX Sink Addresses", false,
                false)

FunctionPass *llvm::createAGXSinkAddressesPass() {
  return new AGXSinkAddresses();
}

static bool isIndexExtension(Value *V) {
  return (isa<SExtInst>(V) || isa<ZExtInst>(V)) &&
         cast<CastInst>(V)->getSrcTy()->isIntegerTy(32) &&
         V->getType()->isIntegerTy(64);
}

/// Whether Off, scaled by 2^Extra bytes, is an offset a device access of
/// 2^ElemLog2-byte elements folds. Appends the instructions computing it,
/// operands first.
static bool foldableOffset(Value *Off, unsigned ElemLog2, unsigned Extra,
                           SmallVectorImpl<Instruction *> &Chain) {
  if (auto *C = dyn_cast<ConstantInt>(Off)) {
    int64_t Bytes = C->getSExtValue() * (int64_t(1) << Extra);
    return Bytes % (int64_t(1) << ElemLog2) == 0 &&
           isInt<16>(Bytes >> ElemLog2);
  }
  Value *X;
  const APInt *C;
  unsigned Scale = Extra;
  auto *I = dyn_cast<Instruction>(Off);
  if (!I)
    return false;
  if (match(I, m_Shl(m_Value(X), m_APInt(C))) && C->ult(8)) {
    Scale += C->getZExtValue();
  } else if (match(I, m_Mul(m_Value(X), m_APInt(C))) && C->isPowerOf2()) {
    Scale += C->logBase2();
  } else {
    X = I;
    I = nullptr;
  }
  if (!isIndexExtension(X) || Scale < ElemLog2 || Scale > ElemLog2 + 2)
    return false;
  Chain.push_back(cast<Instruction>(X));
  if (I)
    Chain.push_back(I);
  return true;
}

namespace {
/// A device address as a base and a foldable offset: the 64-bit integer
/// Base + Off (Int), or the element Off of type Elt from the pointer Base.
struct Address {
  Value *Base = nullptr;
  Value *Off = nullptr;
  Type *Elt = nullptr;
  bool Int = false;
  SmallVector<Instruction *, 4> Chain; // computing Off, operands first
  SmallVector<Instruction *, 4> Rest;  // the other instructions of the address
};
} // namespace

/// Splits P into a base and an offset that an access of 2^ElemLog2-byte
/// elements folds, also when the offset was written as the base, as in
/// (gep i8, (inttoptr offset), base).
static bool splitAddress(Value *P, unsigned ElemLog2, Address &A) {
  Value *Terms[2];
  if (auto *ITP = dyn_cast<IntToPtrInst>(P)) {
    auto *Add = dyn_cast<BinaryOperator>(ITP->getOperand(0));
    if (!Add || !Add->getType()->isIntegerTy(64) ||
        !(Add->getOpcode() == Instruction::Add ||
          (Add->getOpcode() == Instruction::Or &&
           cast<PossiblyDisjointInst>(Add)->isDisjoint())))
      return false;
    Terms[0] = Add->getOperand(0);
    Terms[1] = Add->getOperand(1);
    A.Rest = {Add, ITP};
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(P)) {
    if (GEP->getNumIndices() != 1)
      return false;
    TypeSize Size =
        GEP->getDataLayout().getTypeAllocSize(GEP->getSourceElementType());
    if (Size.isScalable() || !isPowerOf2_64(Size.getFixedValue()))
      return false;
    auto *ITP = dyn_cast<IntToPtrInst>(GEP->getPointerOperand());
    if (Size.getFixedValue() == 1 && ITP &&
        ITP->getOperand(0)->getType()->isIntegerTy(64) &&
        GEP->getOperand(1)->getType()->isIntegerTy(64)) {
      Terms[0] = ITP->getOperand(0);
      Terms[1] = GEP->getOperand(1);
      A.Rest = {ITP, GEP};
    } else {
      if (!foldableOffset(GEP->getOperand(1), ElemLog2,
                          Log2_64(Size.getFixedValue()), A.Chain))
        return false;
      A.Base = GEP->getPointerOperand();
      A.Off = GEP->getOperand(1);
      A.Elt = GEP->getSourceElementType();
      A.Rest = {GEP};
      return true;
    }
  } else {
    return false;
  }
  for (unsigned K = 0; K < 2; ++K)
    if (foldableOffset(Terms[K], ElemLog2, 0, A.Chain)) {
      A.Base = Terms[1 - K];
      A.Off = Terms[K];
      A.Int = true;
      return true;
    }
  return false;
}

bool AGXSinkAddresses::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  bool Changed = false;
  SmallVector<WeakTrackingVH, 16> Originals;
  for (BasicBlock &BB : F) {
    // The copy of each address in this block.
    DenseMap<Value *, Value *> Copies;
    for (Instruction &I : BB) {
      Value *Ptr;
      Type *Ty;
      if (auto *L = dyn_cast<LoadInst>(&I)) {
        Ptr = L->getPointerOperand();
        Ty = L->getType();
      } else if (auto *S = dyn_cast<StoreInst>(&I)) {
        Ptr = S->getPointerOperand();
        Ty = S->getValueOperand()->getType();
      } else {
        continue;
      }
      if (Ptr->getType()->getPointerAddressSpace() != AGX::DEVICE_ADDRESS ||
          !Ty->isSized())
        continue;
      // Accesses wider than 32 bits are made of 32-bit elements.
      unsigned EltBits = Ty->isPointerTy() ? 64 : Ty->getScalarSizeInBits();
      if (!isPowerOf2_32(EltBits) || EltBits < 8)
        continue;
      unsigned ElemLog2 = Log2_32(std::min(EltBits, 32u) / 8);
      Address A;
      if (!splitAddress(Ptr, ElemLog2, A))
        continue;
      auto InBlock = [&](Instruction *C) { return C->getParent() == &BB; };
      // Already selectable: the offset is computed here, the base used as is.
      if (all_of(A.Chain, InBlock) && all_of(A.Rest, InBlock) &&
          (A.Int ? isa<IntToPtrInst>(Ptr) : true))
        continue;
      Value *&Copy = Copies[Ptr];
      if (!Copy) {
        IRBuilder<> B(&I);
        Value *Off = A.Off;
        DenseMap<Instruction *, Instruction *> Map;
        for (Instruction *C : A.Chain) {
          Instruction *N = C->clone();
          N->setName(C->getName() + ".sunk");
          for (Use &U : N->operands())
            if (auto *Op = dyn_cast<Instruction>(U.get()))
              if (Instruction *M = Map.lookup(Op))
                U.set(M);
          B.Insert(N);
          Map[C] = N;
          Originals.push_back(C);
        }
        if (auto *OffI = dyn_cast<Instruction>(Off))
          Off = Map.lookup(OffI);
        Value *New;
        if (A.Int)
          New = B.CreateIntToPtr(B.CreateAdd(A.Base, Off), Ptr->getType());
        else
          New = B.CreateGEP(A.Elt, A.Base, Off);
        for (Instruction *R : A.Rest)
          Originals.push_back(R);
        Copy = New;
      }
      I.replaceUsesOfWith(Ptr, Copy);
      Changed = true;
    }
  }
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(Originals);
  return Changed;
}
