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
// Threadgroup and private pointers are 32 bits, but PTX computes their
// addresses in 64-bit integers. 64-bit arithmetic whose results only reach
// such pointers or truncations (InstCombine narrows it only where each step
// has one use) is redone in 32 bits.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
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

static bool isNarrowPointerSpace(unsigned AS) {
  return AS == AGX::THREADGROUP_ADDRESS || AS == AGX::PRIVATE_ADDRESS;
}

/// Whether the low 32 bits of I depend only on the low 32 bits of its
/// operands that are themselves narrowed.
static bool isNarrowable(const Instruction &I) {
  if (!I.getType()->isIntegerTy(64))
    return false;
  switch (I.getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::PHI:
    return true;
  case Instruction::Shl: {
    auto *C = dyn_cast<ConstantInt>(I.getOperand(1));
    return C && C->getZExtValue() < 32;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
    return I.getOperand(0)->getType()->getIntegerBitWidth() <= 32;
  case Instruction::PtrToInt:
    return true;
  default:
    return false;
  }
}

/// Whether a use only needs the low 32 bits of its value.
static bool isNarrowUse(const User *U) {
  if (auto *ITP = dyn_cast<IntToPtrInst>(U))
    return isNarrowPointerSpace(ITP->getType()->getPointerAddressSpace());
  if (auto *T = dyn_cast<TruncInst>(U))
    return T->getType()->getIntegerBitWidth() <= 32;
  return false;
}

/// An index computation narrowed also where its 64-bit value stays in use
/// (typically folded into device addresses): an extension, possibly scaled.
static bool isCheapIndex(const Instruction &I) {
  if (isa<SExtInst>(I) || isa<ZExtInst>(I))
    return true;
  if (I.getOpcode() != Instruction::Shl && I.getOpcode() != Instruction::Mul)
    return false;
  return isa<ConstantInt>(I.getOperand(1)) &&
         (isa<SExtInst>(I.getOperand(0)) || isa<ZExtInst>(I.getOperand(0)));
}

static bool narrowLocalAddresses(Function &F) {
  // The narrowable instructions all of whose uses only need 32 bits, to a
  // fixed point; at least one use must be a narrow pointer.
  SetVector<Instruction *> Set;
  for (Instruction &I : instructions(F))
    if (isNarrowable(I))
      Set.insert(&I);
  bool Removed = true;
  while (Removed) {
    Removed = false;
    for (Instruction *I : SmallVector<Instruction *, 32>(Set.begin(), Set.end())) {
      bool OK = !I->use_empty() &&
                (isCheapIndex(*I) || all_of(I->users(), [&](User *U) {
                   return isNarrowUse(U) || (isa<Instruction>(U) &&
                                             Set.contains(cast<Instruction>(U)));
                 }));
      if (!OK) {
        Set.remove(I);
        Removed = true;
      }
    }
  }
  // Only worth it where an address is computed.
  SmallVector<Instruction *, 16> Roots;
  for (Instruction *I : Set)
    if (any_of(I->users(), [](User *U) { return isa<IntToPtrInst>(U); }))
      Roots.push_back(I);
  if (Roots.empty())
    return false;

  Type *I32 = Type::getInt32Ty(F.getContext());
  DenseMap<Value *, Value *> Narrow;
  SmallVector<std::pair<PHINode *, PHINode *>, 4> Phis;
  // The 32-bit value of V, created next to its definition.
  std::function<Value *(Value *)> Get = [&](Value *V) -> Value * {
    if (Value *N = Narrow.lookup(V))
      return N;
    Value *N;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Set.contains(I)) {
      // A leaf: truncated where it is defined.
      if (auto *C = dyn_cast<Constant>(V)) {
        N = ConstantFoldCastOperand(Instruction::Trunc, C, I32,
                                    F.getDataLayout());
        if (!N)
          N = ConstantExpr::getTrunc(C, I32);
      } else {
        IRBuilder<> B(F.getContext());
        if (auto *A = dyn_cast<Argument>(V))
          B.SetInsertPoint(F.getEntryBlock().getFirstInsertionPt());
        else
          B.SetInsertPoint(
              cast<Instruction>(V)->getInsertionPointAfterDef().value());
        N = B.CreateTrunc(V, I32, V->getName() + ".lo");
      }
      return Narrow[V] = N;
    }
    if (auto *P = dyn_cast<PHINode>(I)) {
      PHINode *NP = PHINode::Create(I32, P->getNumIncomingValues(),
                                    P->getName() + ".lo", P->getIterator());
      Narrow[V] = NP;
      Phis.push_back({P, NP});
      return NP;
    }
    IRBuilder<> B(I);
    switch (I->getOpcode()) {
    case Instruction::SExt:
    case Instruction::ZExt: {
      Value *X = I->getOperand(0);
      N = X->getType() == I32
              ? X
              : B.CreateCast(cast<CastInst>(I)->getOpcode(), X, I32);
      break;
    }
    case Instruction::PtrToInt:
      N = B.CreatePtrToInt(I->getOperand(0), I32, I->getName() + ".lo");
      break;
    default:
      N = B.CreateBinOp(cast<BinaryOperator>(I)->getOpcode(),
                        Get(I->getOperand(0)), Get(I->getOperand(1)),
                        I->getName() + ".lo");
      break;
    }
    return Narrow[V] = N;
  };
  for (Instruction *I : Set)
    Get(I);
  // Phi operands, which may lead back to the phis themselves.
  for (unsigned K = 0; K < Phis.size(); ++K) {
    auto [P, NP] = Phis[K];
    for (unsigned In = 0; In < P->getNumIncomingValues(); ++In) {
      Value *V = P->getIncomingValue(In);
      Value *N;
      if (auto *I = dyn_cast<Instruction>(V); I && !Set.contains(I)) {
        // Truncate a leaf at the end of the incoming block.
        IRBuilder<> B(P->getIncomingBlock(In)->getTerminator());
        N = B.CreateTrunc(V, I32, V->getName() + ".lo");
      } else {
        N = Get(V);
      }
      NP->addIncoming(N, P->getIncomingBlock(In));
    }
  }
  // Point the narrow uses at the 32-bit values.
  for (auto [V, N] : Narrow) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Set.contains(I))
      continue;
    for (User *U : make_early_inc_range(I->users())) {
      if (!isNarrowUse(U))
        continue;
      if (auto *ITP = dyn_cast<IntToPtrInst>(U)) {
        IRBuilder<> B(ITP);
        ITP->replaceAllUsesWith(B.CreateIntToPtr(N, ITP->getType()));
        ITP->eraseFromParent();
      } else if (auto *T = dyn_cast<TruncInst>(U)) {
        IRBuilder<> B(T);
        T->replaceAllUsesWith(B.CreateTrunc(N, T->getType()));
        T->eraseFromParent();
      }
    }
  }
  // The 64-bit instructions still used outside the set (cheap indices) stay,
  // with the set members they use; the others only use each other now
  // (phis in cycles too).
  SmallPtrSet<Instruction *, 16> Keep;
  SmallVector<Instruction *, 16> Work;
  for (Instruction *I : Set)
    if (any_of(I->users(), [&](User *U) {
          return !isa<Instruction>(U) || !Set.contains(cast<Instruction>(U));
        }))
      Work.push_back(I);
  while (!Work.empty()) {
    Instruction *I = Work.pop_back_val();
    if (!Keep.insert(I).second)
      continue;
    for (Value *Op : I->operands())
      if (auto *J = dyn_cast<Instruction>(Op); J && Set.contains(J))
        Work.push_back(J);
  }
  SmallVector<Instruction *, 16> Erase;
  for (Instruction *I : Set)
    if (!Keep.contains(I))
      Erase.push_back(I);
  for (Instruction *I : Erase)
    I->dropAllReferences();
  for (Instruction *I : Erase)
    I->eraseFromParent();
  return true;
}

bool AGXSinkAddresses::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  bool Changed = narrowLocalAddresses(F);
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
