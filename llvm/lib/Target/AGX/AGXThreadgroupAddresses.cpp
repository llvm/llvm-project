//===-- AGXThreadgroupAddresses.cpp - Threadgroup addresses as base + offset ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PTX computes threadgroup (shared) and private addresses in integers, so
// their pointers are inttoptr of an integer expression: the accesses of two
// unrolled loop iterations, a[k] and a[k + 2], are inttoptr((k + i) * 4 + s)
// and inttoptr(((k | 2) + i) * 4 + s), which the load/store vectorizer cannot
// relate. Each such pointer becomes a constant offset (a getelementptr) from
// the pointer of the address without its constant terms, rebuilt from the
// instructions already there: both accesses above then use the same base
// pointer, at offsets 0 and 8, and the vectorizer can join them.
//
// The pointers are 32 bits and the getelementptr is not inbounds: the
// address is the same modulo 2^32, whatever the arithmetic wrapped.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-threadgroup-addresses"

static cl::opt<bool> AGXThreadgroupAddressOffsets(
    "agx-threadgroup-address-offsets", cl::init(true), cl::Hidden,
    cl::desc("Write threadgroup and private addresses as base + constant "
             "for the load/store vectorizer"));

namespace {

class AGXThreadgroupAddresses : public FunctionPass {
public:
  static char ID;
  AGXThreadgroupAddresses() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Threadgroup Addresses";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

/// An integer expression split into a base without constant terms (null for
/// a constant) and a constant.
struct Split {
  Value *Base;
  APInt Offset;
};

class Splitter {
  DominatorTree &DT;
  Instruction *At;

public:
  Splitter(DominatorTree &DT, Instruction *At) : DT(DT), At(At) {}

  /// An instruction computing Op(A, B) that dominates the insertion point,
  /// the existing one if there is one.
  Value *binary(Instruction::BinaryOps Op, Value *A, Value *B) {
    for (User *U : A->users()) {
      auto *I = dyn_cast<BinaryOperator>(U);
      if (!I || I->getOpcode() != Op || !DT.dominates(I, At))
        continue;
      if ((I->getOperand(0) == A && I->getOperand(1) == B) ||
          (I->isCommutative() && I->getOperand(0) == B &&
           I->getOperand(1) == A))
        return I;
    }
    return BinaryOperator::Create(Op, A, B, "", At->getIterator());
  }

  Split split(Value *V, unsigned Depth = 0) {
    unsigned Bits = V->getType()->getIntegerBitWidth();
    if (auto *C = dyn_cast<ConstantInt>(V))
      return {nullptr, C->getValue()};
    Value *A, *B;
    const APInt *K;
    // (a constant expression add: a global's address plus a constant, which
    // the patterns below no longer match in constants)
    auto *CE = dyn_cast<ConstantExpr>(V);
    bool ConstantAdd = CE && CE->getOpcode() == Instruction::Add;
    if (ConstantAdd) {
      A = CE->getOperand(0);
      B = CE->getOperand(1);
    }
    if (Depth < 8) {
      if (ConstantAdd || match(V, m_Add(m_Value(A), m_Value(B))) ||
          match(V, m_DisjointOr(m_Value(A), m_Value(B)))) {
        Split SA = split(A, Depth + 1), SB = split(B, Depth + 1);
        if (SA.Offset.isZero() && SB.Offset.isZero())
          return {V, APInt(Bits, 0)};
        Value *Base = !SA.Base   ? SB.Base
                      : !SB.Base ? SA.Base
                                 : binary(Instruction::Add, SA.Base, SB.Base);
        return {Base, SA.Offset + SB.Offset};
      }
      if (match(V, m_Sub(m_Value(A), m_APInt(K)))) {
        Split SA = split(A, Depth + 1);
        return {SA.Base, SA.Offset - *K};
      }
      if (match(V, m_Shl(m_Value(A), m_APInt(K))) && K->ult(Bits)) {
        Split SA = split(A, Depth + 1);
        if (SA.Offset.isZero())
          return {V, APInt(Bits, 0)};
        Value *Base =
            SA.Base ? binary(Instruction::Shl, SA.Base,
                             ConstantInt::get(V->getType(), *K))
                    : nullptr;
        return {Base, SA.Offset.shl(*K)};
      }
      if (match(V, m_Mul(m_Value(A), m_APInt(K)))) {
        Split SA = split(A, Depth + 1);
        if (SA.Offset.isZero())
          return {V, APInt(Bits, 0)};
        Value *Base =
            SA.Base ? binary(Instruction::Mul, SA.Base,
                             ConstantInt::get(V->getType(), *K))
                    : nullptr;
        return {Base, SA.Offset * *K};
      }
    }
    return {V, APInt(Bits, 0)};
  }
};

} // namespace

char AGXThreadgroupAddresses::ID = 0;

INITIALIZE_PASS_BEGIN(AGXThreadgroupAddresses, DEBUG_TYPE,
                      "AGX Threadgroup Addresses", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AGXThreadgroupAddresses, DEBUG_TYPE,
                    "AGX Threadgroup Addresses", false, false)

FunctionPass *llvm::createAGXThreadgroupAddressesPass() {
  return new AGXThreadgroupAddresses();
}

/// An instruction that dominates At and casts V to Ty (zext or inttoptr), the
/// existing one if there is one.
static Value *cast(Instruction::CastOps Op, Value *V, Type *Ty,
                   Instruction *At, DominatorTree &DT) {
  for (User *U : V->users())
    if (auto *C = dyn_cast<CastInst>(U))
      if (C->getOpcode() == Op && C->getType() == Ty && DT.dominates(C, At))
        return C;
  if (!CastInst::castIsValid(Op, V, Ty))
    return nullptr;
  return CastInst::Create(Op, V, Ty, "", At->getIterator());
}

bool AGXThreadgroupAddresses::runOnFunction(Function &F) {
  if (!AGXThreadgroupAddressOffsets)
    return false;
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SmallVector<IntToPtrInst *, 32> Work;
  for (Instruction &I : instructions(F)) {
    auto *P = dyn_cast<IntToPtrInst>(&I);
    if (!P || !P->getType()->isPointerTy() ||
        !P->getOperand(0)->getType()->isIntegerTy())
      continue;
    unsigned AS = P->getType()->getPointerAddressSpace();
    if (AS == AGX::THREADGROUP_ADDRESS || AS == AGX::PRIVATE_ADDRESS)
      Work.push_back(P);
  }
  bool Changed = false;
  for (IntToPtrInst *P : Work) {
    Value *V = P->getOperand(0);
    // inttoptr (zext i32 X): X is the 32-bit address
    Value *X = V;
    if (auto *Z = dyn_cast<ZExtInst>(V))
      X = Z->getOperand(0);
    Splitter S(DT, P);
    Split Sp = S.split(X);
    if (Sp.Offset.isZero() || !Sp.Base)
      continue;
    // the base as a 32-bit pointer directly (inttoptr of the zext to 64 bits
    // is the same pointer)
    Value *Base = cast(Instruction::IntToPtr, Sp.Base, P->getType(), P, DT);
    if (!Base)
      continue;
    IRBuilder<> B(P);
    Value *G = B.CreatePtrAdd(
        Base, B.getInt32(static_cast<uint32_t>(Sp.Offset.getZExtValue())));
    G->takeName(P);
    // the old address arithmetic may now be dead: later passes delete it (it
    // may also have been reused as another address's base)
    P->replaceAllUsesWith(G);
    P->eraseFromParent();
    Changed = true;
  }
  return Changed;
}
