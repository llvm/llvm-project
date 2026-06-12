//===- LowerVectorSelect.cpp - Branchless vector-select lowering ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vector-condition SELECT form (VSELECT) is rejected by the AGX JIT (see
// the SelectInst emission in FunctionWriter). The mid-end vectorizers produce
// vector-condition selects on `where`/`clamp`. Rather than scalarize into a
// per-lane extract/select/insert chain (which doubles register pressure and
// relies on the JIT re-fusing the lanes), lower each into a branchless bitmask
// blend that stays fully vectorized:
//
//   mask = sext <N x i1> cond to <N x iW>      ; all-ones / all-zeros per lane
//   res  = (a_bits & mask) | (b_bits & ~mask)  ; bit-reinterpreted operands
//
// where iW is the element bit width. Float operands are bitcast to an integer
// vector of matching width and the result bitcast back.
//
//===----------------------------------------------------------------------===//

#include "LowerVectorSelect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

void lowerVectorSelects(Module &M) {
  SmallVector<SelectInst *, 16> Sels;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *Sel = dyn_cast<SelectInst>(&I))
          if (Sel->getCondition()->getType()->isVectorTy())
            Sels.push_back(Sel);

  for (SelectInst *Sel : Sels) {
    auto *VT = cast<FixedVectorType>(Sel->getType());
    Type *ElemTy = VT->getElementType();
    unsigned BitWidth = ElemTy->getPrimitiveSizeInBits();
    if (BitWidth == 0)
      continue;

    IRBuilder<> B(Sel);
    auto *IntElemTy = B.getIntNTy(BitWidth);
    auto *IntVecTy = FixedVectorType::get(IntElemTy, VT->getNumElements());
    bool NeedCast = IntVecTy != VT;

    Value *TV = Sel->getTrueValue();
    Value *FV = Sel->getFalseValue();
    Value *Mask = B.CreateSExt(Sel->getCondition(), IntVecTy);

    auto toBits = [&](Value *V) -> Value * {
      if (auto *C = dyn_cast<Constant>(V)) {
        if (C->isNullValue())
          return ConstantInt::getNullValue(IntVecTy);
        if (C->isAllOnesValue())
          return ConstantInt::getAllOnesValue(IntVecTy);
      }
      return NeedCast ? B.CreateBitCast(V, IntVecTy) : V;
    };
    auto *TC = dyn_cast<Constant>(TV);
    auto *FC = dyn_cast<Constant>(FV);

    Value *Blend;
    if (FC && FC->isNullValue())
      Blend = B.CreateAnd(toBits(TV), Mask);
    else if (TC && TC->isNullValue())
      Blend = B.CreateAnd(toBits(FV), B.CreateNot(Mask));
    else if (FC && FC->isAllOnesValue())
      Blend = B.CreateOr(toBits(TV), B.CreateNot(Mask));
    else if (TC && TC->isAllOnesValue())
      Blend = B.CreateOr(toBits(FV), Mask);
    else
      Blend = B.CreateOr(B.CreateAnd(toBits(TV), Mask),
                         B.CreateAnd(toBits(FV), B.CreateNot(Mask)));

    if (NeedCast && Blend->getType() == IntVecTy)
      Blend = B.CreateBitCast(Blend, VT);

    Sel->replaceAllUsesWith(Blend);
    Sel->eraseFromParent();
  }
}

} // namespace metal
} // namespace llvm
