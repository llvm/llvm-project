//===- LowerPointerVectors.cpp - Lower vector-of-pointer values -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIR/Metal has no vector-of-pointer values: a `<N x ptr>` gets a single
// pointee slot in the AIR type table, yet its scalar lanes can carry
// conflicting pointees. This transform lowers the whole pointer-vector web to
// `<N x i64>` and reconstructs pointer semantics only at the consuming edges
// (ptrtoint, extractelement, generic pointer-vector consumers).
//
//===----------------------------------------------------------------------===//

#include "LowerPointerVectors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>

using namespace llvm;

namespace llvm {
namespace metal {

void lowerVectorPointerToInt(Module &M) {
  auto isPtrVec = [](Type *T) -> FixedVectorType * {
    auto *VT = dyn_cast<FixedVectorType>(T);
    if (VT && VT->getElementType()->isPointerTy())
      return VT;
    return nullptr;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  const DataLayout &DL = M.getDataLayout();
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<Instruction *, 16> PtrVecDefs;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isPtrVec(I.getType()))
          PtrVecDefs.push_back(&I);
    if (PtrVecDefs.empty())
      continue;

    DenseMap<Value *, Value *> IntOf;
    auto intVecTy = [&](FixedVectorType *PVT) {
      return FixedVectorType::get(I64, PVT->getNumElements());
    };

    std::function<Value *(Value *, IRBuilder<> &)> asIntVec =
        [&](Value *V, IRBuilder<> &B) -> Value * {
      if (auto *It = IntOf.lookup(V))
        return It;
      auto *PVT = cast<FixedVectorType>(V->getType());
      if (isa<UndefValue>(V))
        return UndefValue::get(intVecTy(PVT));
      if (isa<ConstantAggregateZero>(V) ||
          (isa<Constant>(V) && cast<Constant>(V)->isNullValue()))
        return ConstantAggregateZero::get(intVecTy(PVT));

      return B.CreatePtrToInt(V, intVecTy(PVT));
    };

    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        IRBuilder<> B(PN);
        auto *NewPN =
            B.CreatePHI(intVecTy(cast<FixedVectorType>(PN->getType())),
                        PN->getNumIncomingValues());
        IntOf[PN] = NewPN;
      }

    for (Instruction *I : PtrVecDefs) {
      if (isa<PHINode>(I))
        continue;
      IRBuilder<> B(I);
      Value *Repl = nullptr;
      if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        Value *Vec = asIntVec(IE->getOperand(0), B);
        Value *Sc = B.CreatePtrToInt(IE->getOperand(1), I64);
        Repl = B.CreateInsertElement(Vec, Sc, IE->getOperand(2));
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(I)) {
        Value *A = asIntVec(SV->getOperand(0), B);
        Value *Bv = asIntVec(SV->getOperand(1), B);
        Repl = B.CreateShuffleVector(A, Bv, SV->getShuffleMask());
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *T = asIntVec(Sel->getTrueValue(), B);
        Value *Fv = asIntVec(Sel->getFalseValue(), B);
        Repl = B.CreateSelect(Sel->getCondition(), T, Fv);
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {

        Repl = asIntVec(BC->getOperand(0), B);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        auto *PVT = cast<FixedVectorType>(GEP->getType());
        unsigned N = PVT->getNumElements();
        Value *Base = asIntVec(GEP->getPointerOperand(), B);
        auto *VecI64 = FixedVectorType::get(I64, N);
        Value *Off = ConstantAggregateZero::get(VecI64);
        auto splat = [&](Value *S) -> Value * {
          if (S->getType()->isVectorTy())
            return B.CreateSExtOrTrunc(S, VecI64);
          Value *W = B.CreateSExtOrTrunc(S, I64);
          return B.CreateVectorSplat(N, W);
        };
        for (gep_type_iterator GTI = gep_type_begin(GEP),
                               GTE = gep_type_end(GEP);
             GTI != GTE; ++GTI) {
          Value *Idx = GTI.getOperand();
          if (StructType *STy = GTI.getStructTypeOrNull()) {
            uint64_t FieldNo = cast<ConstantInt>(Idx)->getZExtValue();
            uint64_t FieldOff =
                DL.getStructLayout(STy)->getElementOffset(FieldNo);
            Off = B.CreateAdd(
                Off, B.CreateVectorSplat(N, ConstantInt::get(I64, FieldOff)));
          } else {
            uint64_t Stride =
                DL.getTypeAllocSize(GTI.getIndexedType()).getFixedValue();
            Value *Scaled = B.CreateMul(
                splat(Idx),
                B.CreateVectorSplat(N, ConstantInt::get(I64, Stride)));
            Off = B.CreateAdd(Off, Scaled);
          }
        }
        Repl = B.CreateAdd(Base, Off);
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled vector-of-pointer "
                                 "producer '") +
                           I->getOpcodeName() + "' in function '" +
                           F.getName() + "'");
      }
      IntOf[I] = Repl;
    }

    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        auto *NewPN = cast<PHINode>(IntOf[PN]);
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          IRBuilder<> B(PN->getIncomingBlock(J)->getTerminator());
          NewPN->addIncoming(asIntVec(PN->getIncomingValue(J), B),
                             PN->getIncomingBlock(J));
        }
      }

    for (Instruction *I : PtrVecDefs) {
      Value *Int = IntOf.lookup(I);
      if (!Int)
        continue;
      SmallVector<Use *, 8> Uses;
      for (Use &U : I->uses())
        Uses.push_back(&U);
      for (Use *U : Uses) {
        auto *User = cast<Instruction>(U->getUser());
        if (IntOf.count(User))
          continue;
        IRBuilder<> B(User);
        if (auto *P2I = dyn_cast<PtrToIntInst>(User)) {
          Value *V = Int;
          if (P2I->getType() != Int->getType())
            V = B.CreateZExtOrTrunc(Int, P2I->getType());
          P2I->replaceAllUsesWith(V);
          continue;
        }
        if (auto *EE = dyn_cast<ExtractElementInst>(User)) {
          Value *Sc = B.CreateExtractElement(Int, EE->getIndexOperand());
          EE->replaceAllUsesWith(B.CreateIntToPtr(Sc, EE->getType()));
          continue;
        }

        U->set(B.CreateIntToPtr(Int, I->getType()));
      }
    }

    for (Instruction *I : reverse(PtrVecDefs)) {
      if (!IntOf.count(I))
        continue;
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
    }

    SmallVector<Instruction *, 8> Dead;
    for (auto &BB : F)
      for (auto &I : BB)
        if ((isa<PtrToIntInst>(I) || isa<ExtractElementInst>(I)) &&
            I.use_empty() && isPtrVec(I.getOperand(0)->getType()) &&
            IntOf.count(I.getOperand(0)))
          Dead.push_back(&I);
    for (Instruction *I : Dead)
      I->eraseFromParent();
    for (Instruction *I : reverse(PtrVecDefs))
      if (IntOf.count(I) && I->use_empty())
        I->eraseFromParent();
  }
}

}
}
