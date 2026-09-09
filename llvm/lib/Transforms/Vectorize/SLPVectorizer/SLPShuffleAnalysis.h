//===- SLPShuffleAnalysis.h - SLP shuffle analysis base ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It defines the base class for
// shuffle cost estimation and shuffle instruction emission. It does not depend
// on BoUpSLP or any other SLP-private type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPSHUFFLEANALYSIS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPSHUFFLEANALYSIS_H

#include "SLPUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>

namespace llvm::slpvectorizer {

/// The base class for shuffle instruction emission and shuffle cost estimation.
class BaseShuffleAnalysis {
protected:
  Type *ScalarTy = nullptr;

  BaseShuffleAnalysis(Type *ScalarTy) : ScalarTy(ScalarTy) {}

  /// V is expected to be a vectorized value.
  /// When REVEC is disabled, there is no difference between VF and
  /// VNumElements.
  /// When REVEC is enabled, VF is VNumElements / ScalarTyNumElements.
  /// e.g., if ScalarTy is <4 x Ty> and V1 is <8 x Ty>, 2 is returned instead
  /// of 8.
  unsigned getVF(Value *V) const {
    assert(V && "V cannot be nullptr");
    assert(isa<FixedVectorType>(V->getType()) &&
           "V does not have FixedVectorType");
    assert(ScalarTy && "ScalarTy cannot be nullptr");
    unsigned ScalarTyNumElements = getNumElements(ScalarTy);
    unsigned VNumElements =
        cast<FixedVectorType>(V->getType())->getNumElements();
    assert(VNumElements > ScalarTyNumElements &&
           "the number of elements of V is not large enough");
    assert(VNumElements % ScalarTyNumElements == 0 &&
           "the number of elements of V is not a vectorized value");
    return VNumElements / ScalarTyNumElements;
  }

  /// Checks if the mask is an identity mask.
  /// \param IsStrict if is true the function returns false if mask size does
  /// not match vector size.
  static bool isIdentityMask(ArrayRef<int> Mask, const FixedVectorType *VecTy,
                             bool IsStrict) {
    int Limit = Mask.size();
    int VF = VecTy->getNumElements();
    int Index = -1;
    if (VF == Limit && ShuffleVectorInst::isIdentityMask(Mask, Limit))
      return true;
    if (!IsStrict) {
      // Consider extract subvector starting from index 0.
      if (ShuffleVectorInst::isExtractSubvectorMask(Mask, VF, Index) &&
          Index == 0)
        return true;
      // All VF-size submasks are identity (e.g.
      // <poison,poison,poison,poison,0,1,2,poison,poison,1,2,3> etc. for VF 4).
      if (Limit % VF == 0 && all_of(seq<int>(0, Limit / VF), [=](int Idx) {
            ArrayRef<int> Slice = Mask.slice(Idx * VF, VF);
            return all_of(Slice, equal_to(PoisonMaskElem)) ||
                   ShuffleVectorInst::isIdentityMask(Slice, VF);
          }))
        return true;
    }
    return false;
  }

  /// Tries to combine 2 different masks into single one.
  /// \param LocalVF Vector length of the permuted input vector. \p Mask may
  /// change the size of the vector, \p LocalVF is the original size of the
  /// shuffled vector.
  static void combineMasks(unsigned LocalVF, SmallVectorImpl<int> &Mask,
                           ArrayRef<int> ExtMask) {
    unsigned VF = Mask.size();
    SmallVector<int> NewMask(ExtMask.size(), PoisonMaskElem);
    for (int I = 0, Sz = ExtMask.size(); I < Sz; ++I) {
      if (ExtMask[I] == PoisonMaskElem)
        continue;
      int MaskedIdx = Mask[ExtMask[I] % VF];
      NewMask[I] =
          MaskedIdx == PoisonMaskElem ? PoisonMaskElem : MaskedIdx % LocalVF;
    }
    Mask.swap(NewMask);
  }

  /// Looks through shuffles trying to reduce final number of shuffles in the
  /// code. The function looks through the previously emitted shuffle
  /// instructions and properly mark indices in mask as undef.
  /// For example, given the code
  /// \code
  /// %s1 = shufflevector <2 x ty> %0, poison, <1, 0>
  /// %s2 = shufflevector <2 x ty> %1, poison, <1, 0>
  /// \endcode
  /// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 3, 2>, it will
  /// look through %s1 and %s2 and select vectors %0 and %1 with mask
  /// <0, 1, 2, 3> for the shuffle.
  /// If 2 operands are of different size, the smallest one will be resized and
  /// the mask recalculated properly.
  /// For example, given the code
  /// \code
  /// %s1 = shufflevector <2 x ty> %0, poison, <1, 0, 1, 0>
  /// %s2 = shufflevector <2 x ty> %1, poison, <1, 0, 1, 0>
  /// \endcode
  /// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 5, 4>, it will
  /// look through %s1 and %s2 and select vectors %0 and %1 with mask
  /// <0, 1, 2, 3> for the shuffle.
  /// So, it tries to transform permutations to simple vector merge, if
  /// possible.
  /// \param V The input vector which must be shuffled using the given \p Mask.
  /// If the better candidate is found, \p V is set to this best candidate
  /// vector.
  /// \param Mask The input mask for the shuffle. If the best candidate is found
  /// during looking-through-shuffles attempt, it is updated accordingly.
  /// \param SinglePermute true if the shuffle operation is originally a
  /// single-value-permutation. In this case the look-through-shuffles procedure
  /// may look for resizing shuffles as the best candidates.
  /// \return true if the shuffle results in the non-resizing identity shuffle
  /// (and thus can be ignored), false - otherwise.
  static bool peekThroughShuffles(Value *&V, SmallVectorImpl<int> &Mask,
                                  bool SinglePermute) {
    Value *Op = V;
    ShuffleVectorInst *IdentityOp = nullptr;
    SmallVector<int> IdentityMask;
    while (auto *SV = dyn_cast<ShuffleVectorInst>(Op)) {
      // Exit if not a fixed vector type or changing size shuffle.
      auto *SVTy = dyn_cast<FixedVectorType>(SV->getType());
      if (!SVTy)
        break;
      // Remember the identity or broadcast mask, if it is not a resizing
      // shuffle. If no better candidates are found, this Op and Mask will be
      // used in the final shuffle.
      if (isIdentityMask(Mask, SVTy, /*IsStrict=*/false)) {
        if (!IdentityOp || !SinglePermute ||
            (isIdentityMask(Mask, SVTy, /*IsStrict=*/true) &&
             !ShuffleVectorInst::isZeroEltSplatMask(IdentityMask,
                                                    IdentityMask.size()))) {
          IdentityOp = SV;
          // Store current mask in the IdentityMask so later we did not lost
          // this info if IdentityOp is selected as the best candidate for the
          // permutation.
          IdentityMask.assign(Mask);
        }
      }
      // Remember the broadcast mask. If no better candidates are found, this Op
      // and Mask will be used in the final shuffle.
      // Zero splat can be used as identity too, since it might be used with
      // mask <0, 1, 2, ...>, i.e. identity mask without extra reshuffling.
      // E.g. if need to shuffle the vector with the mask <3, 1, 2, 0>, which is
      // expensive, the analysis founds out, that the source vector is just a
      // broadcast, this original mask can be transformed to identity mask <0,
      // 1, 2, 3>.
      // \code
      // %0 = shuffle %v, poison, zeroinitalizer
      // %res = shuffle %0, poison, <3, 1, 2, 0>
      // \endcode
      // may be transformed to
      // \code
      // %0 = shuffle %v, poison, zeroinitalizer
      // %res = shuffle %0, poison, <0, 1, 2, 3>
      // \endcode
      if (SV->isZeroEltSplat()) {
        IdentityOp = SV;
        IdentityMask.assign(Mask);
      }
      int LocalVF = Mask.size();
      if (auto *SVOpTy =
              dyn_cast<FixedVectorType>(SV->getOperand(0)->getType()))
        LocalVF = SVOpTy->getNumElements();
      SmallVector<int> ExtMask(Mask.size(), PoisonMaskElem);
      for (auto [Idx, I] : enumerate(Mask)) {
        if (I == PoisonMaskElem ||
            static_cast<unsigned>(I) >= SV->getShuffleMask().size())
          continue;
        ExtMask[Idx] = SV->getMaskValue(I);
      }
      bool IsOp1Undef = isUndefVector</*isPoisonOnly=*/true>(
                            SV->getOperand(0),
                            buildUseMask(LocalVF, ExtMask, UseMask::FirstArg))
                            .all();
      bool IsOp2Undef = isUndefVector</*isPoisonOnly=*/true>(
                            SV->getOperand(1),
                            buildUseMask(LocalVF, ExtMask, UseMask::SecondArg))
                            .all();
      if (!IsOp1Undef && !IsOp2Undef) {
        // Update mask and mark undef elems.
        for (int &I : Mask) {
          if (I == PoisonMaskElem)
            continue;
          if (SV->getMaskValue(I % SV->getShuffleMask().size()) ==
              PoisonMaskElem)
            I = PoisonMaskElem;
        }
        break;
      }
      SmallVector<int> ShuffleMask(SV->getShuffleMask());
      combineMasks(LocalVF, ShuffleMask, Mask);
      Mask.swap(ShuffleMask);
      if (IsOp2Undef)
        Op = SV->getOperand(0);
      else
        Op = SV->getOperand(1);
    }
    if (auto *OpTy = dyn_cast<FixedVectorType>(Op->getType());
        !OpTy || !isIdentityMask(Mask, OpTy, SinglePermute) ||
        ShuffleVectorInst::isZeroEltSplatMask(Mask, Mask.size())) {
      if (IdentityOp) {
        V = IdentityOp;
        assert(Mask.size() == IdentityMask.size() &&
               "Expected masks of same sizes.");
        // Clear known poison elements.
        for (auto [I, Idx] : enumerate(Mask))
          if (Idx == PoisonMaskElem)
            IdentityMask[I] = PoisonMaskElem;
        Mask.swap(IdentityMask);
        auto *Shuffle = dyn_cast<ShuffleVectorInst>(V);
        return SinglePermute &&
               (isIdentityMask(Mask, cast<FixedVectorType>(V->getType()),
                               /*IsStrict=*/true) ||
                (Shuffle && Mask.size() == Shuffle->getShuffleMask().size() &&
                 Shuffle->isZeroEltSplat() &&
                 ShuffleVectorInst::isZeroEltSplatMask(Mask, Mask.size()) &&
                 all_of(enumerate(Mask), [&](const auto &P) {
                   return P.value() == PoisonMaskElem ||
                          Shuffle->getShuffleMask()[P.index()] == 0;
                 })));
      }
      V = Op;
      return false;
    }
    V = Op;
    return true;
  }

  /// Smart shuffle instruction emission, walks through shuffles trees and
  /// tries to find the best matching vector for the actual shuffle
  /// instruction.
  template <typename T, typename ShuffleBuilderTy, typename... Args>
  static T createShuffle(Value *V1, Value *V2, ArrayRef<int> Mask,
                         ShuffleBuilderTy &Builder, Type *ScalarTy, bool ReVec,
                         Args... Arguments) {
    assert(V1 && "Expected at least one vector value.");
    unsigned ScalarTyNumElements = getNumElements(ScalarTy);
    SmallVector<int> NewMask(Mask);
    if (ScalarTyNumElements != 1) {
      assert(ReVec && "FixedVectorType is not expected.");
      transformScalarShuffleIndiciesToVector(ScalarTyNumElements, NewMask);
      Mask = NewMask;
    }
    if (V2)
      Builder.resizeToMatch(V1, V2);
    int VF = Mask.size();
    if (auto *FTy = dyn_cast<FixedVectorType>(V1->getType()))
      VF = FTy->getNumElements();
    if (V2 && !isUndefVector</*IsPoisonOnly=*/true>(
                   V2, buildUseMask(VF, Mask, UseMask::SecondArg))
                   .all()) {
      // Peek through shuffles.
      Value *Op1 = V1;
      Value *Op2 = V2;
      int VF =
          cast<VectorType>(V1->getType())->getElementCount().getKnownMinValue();
      SmallVector<int> CombinedMask1(Mask.size(), PoisonMaskElem);
      SmallVector<int> CombinedMask2(Mask.size(), PoisonMaskElem);
      for (int I = 0, E = Mask.size(); I < E; ++I) {
        if (Mask[I] < VF)
          CombinedMask1[I] = Mask[I];
        else
          CombinedMask2[I] = Mask[I] - VF;
      }
      Value *PrevOp1;
      Value *PrevOp2;
      do {
        PrevOp1 = Op1;
        PrevOp2 = Op2;
        (void)peekThroughShuffles(Op1, CombinedMask1, /*SinglePermute=*/false);
        (void)peekThroughShuffles(Op2, CombinedMask2, /*SinglePermute=*/false);
        // Check if we have 2 resizing shuffles - need to peek through operands
        // again.
        if (auto *SV1 = dyn_cast<ShuffleVectorInst>(Op1))
          if (auto *SV2 = dyn_cast<ShuffleVectorInst>(Op2)) {
            SmallVector<int> ExtMask1(Mask.size(), PoisonMaskElem);
            for (auto [Idx, I] : enumerate(CombinedMask1)) {
              if (I == PoisonMaskElem)
                continue;
              ExtMask1[Idx] = SV1->getMaskValue(I);
            }
            SmallBitVector UseMask1 = buildUseMask(
                cast<FixedVectorType>(SV1->getOperand(1)->getType())
                    ->getNumElements(),
                ExtMask1, UseMask::SecondArg);
            SmallVector<int> ExtMask2(CombinedMask2.size(), PoisonMaskElem);
            for (auto [Idx, I] : enumerate(CombinedMask2)) {
              if (I == PoisonMaskElem)
                continue;
              ExtMask2[Idx] = SV2->getMaskValue(I);
            }
            SmallBitVector UseMask2 = buildUseMask(
                cast<FixedVectorType>(SV2->getOperand(1)->getType())
                    ->getNumElements(),
                ExtMask2, UseMask::SecondArg);
            if (SV1->getOperand(0)->getType() ==
                    SV2->getOperand(0)->getType() &&
                SV1->getOperand(0)->getType() != SV1->getType() &&
                isUndefVector(SV1->getOperand(1), UseMask1).all() &&
                isUndefVector(SV2->getOperand(1), UseMask2).all()) {
              Op1 = SV1->getOperand(0);
              Op2 = SV2->getOperand(0);
              SmallVector<int> ShuffleMask1(SV1->getShuffleMask());
              int LocalVF = ShuffleMask1.size();
              if (auto *FTy = dyn_cast<FixedVectorType>(Op1->getType()))
                LocalVF = FTy->getNumElements();
              combineMasks(LocalVF, ShuffleMask1, CombinedMask1);
              CombinedMask1.swap(ShuffleMask1);
              SmallVector<int> ShuffleMask2(SV2->getShuffleMask());
              LocalVF = ShuffleMask2.size();
              if (auto *FTy = dyn_cast<FixedVectorType>(Op2->getType()))
                LocalVF = FTy->getNumElements();
              combineMasks(LocalVF, ShuffleMask2, CombinedMask2);
              CombinedMask2.swap(ShuffleMask2);
            }
          }
      } while (PrevOp1 != Op1 || PrevOp2 != Op2);
      Builder.resizeToMatch(Op1, Op2);
      VF = std::max(cast<VectorType>(Op1->getType())
                        ->getElementCount()
                        .getKnownMinValue(),
                    cast<VectorType>(Op2->getType())
                        ->getElementCount()
                        .getKnownMinValue());
      for (int I = 0, E = Mask.size(); I < E; ++I) {
        if (CombinedMask2[I] != PoisonMaskElem) {
          assert(CombinedMask1[I] == PoisonMaskElem &&
                 "Expected undefined mask element");
          CombinedMask1[I] = CombinedMask2[I] + (Op1 == Op2 ? 0 : VF);
        }
      }
      if (Op1 == Op2 &&
          (ShuffleVectorInst::isIdentityMask(CombinedMask1, VF) ||
           (ShuffleVectorInst::isZeroEltSplatMask(CombinedMask1, VF) &&
            isa<ShuffleVectorInst>(Op1) &&
            cast<ShuffleVectorInst>(Op1)->getShuffleMask() ==
                ArrayRef(CombinedMask1))))
        return Builder.createIdentity(Op1);
      return Builder.createShuffleVector(
          Op1, Op1 == Op2 ? PoisonValue::get(Op1->getType()) : Op2,
          CombinedMask1);
    }
    if (isa<PoisonValue>(V1))
      return Builder.createPoison(
          cast<VectorType>(V1->getType())->getElementType(), Mask.size());
    bool IsIdentity = peekThroughShuffles(V1, NewMask, /*SinglePermute=*/true);
    assert(V1 && "Expected non-null value after looking through shuffles.");

    if (!IsIdentity)
      return Builder.createShuffleVector(V1, NewMask, Arguments...);
    return Builder.createIdentity(V1);
  }

  /// Transforms mask \p CommonMask per given \p Mask to make proper set after
  /// shuffle emission.
  static void transformMaskAfterShuffle(MutableArrayRef<int> CommonMask,
                                        ArrayRef<int> Mask) {
    for (unsigned I : seq<unsigned>(CommonMask.size()))
      if (Mask[I] != PoisonMaskElem)
        CommonMask[I] = I;
  }
};

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPSHUFFLEANALYSIS_H
