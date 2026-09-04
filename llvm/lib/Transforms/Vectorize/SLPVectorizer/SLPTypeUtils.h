//===- SLPTypeUtils.h - SLP Vectorizer type/width helpers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free type and vector
// width helpers that do not depend on BoUpSLP or any other SLP-private type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTYPEUTILS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTYPEUTILS_H

namespace llvm {
class FixedVectorType;
class TargetTransformInfo;
class Type;
class Value;
} // namespace llvm

namespace llvm::slpvectorizer {

/// Predicate for the element types that the SLP vectorizer supports.
///
/// The most important thing to filter here are types which are invalid in LLVM
/// vectors. We also filter target specific types which have absolutely no
/// meaningful vectorization path such as x86_fp80 and ppc_f128. This just
/// avoids spending time checking the cost model and realizing that they will
/// be inevitably scalarized.
bool isValidElementType(Type *Ty, bool ReVec);

/// Returns the "element type" of the given value/instruction \p V.
/// For stores, returns the stored value type; for insertelement (when \p ReVec
/// is off), the inserted operand type. For compares, the default is to return
/// the result type (i1); when \p LookThroughCmp is true, returns the type of
/// the compared operands instead, which is needed for vector width
/// calculations (the width is determined by the operand type, not the i1
/// result).
Type *getValueType(Value *V, bool ReVec, bool LookThroughCmp = false);

/// \returns the vector type of ScalarTy based on vectorization factor.
Type *getWidenedType(Type *ScalarTy, unsigned VF);

/// Returns the number of elements of the given type \p Ty, not less than \p Sz,
/// which forms type, which splits by \p TTI into whole vector types during
/// legalization.
unsigned getFullVectorNumberOfElements(const TargetTransformInfo &TTI, Type *Ty,
                                       unsigned Sz, bool ReVec);

/// Returns the number of elements of the given type \p Ty, not greater than \p
/// Sz, which forms type, which splits by \p TTI into whole vector types during
/// legalization.
unsigned getFloorFullVectorNumberOfElements(const TargetTransformInfo &TTI,
                                            Type *Ty, unsigned Sz, bool ReVec);

/// For a non-power-of-2 \p NumElts-wide integer div/rem \p Opcode, returns the
/// padded full-register vector type if padding is structurally possible, or
/// nullptr if the vector already fills a register or the opcode is not
/// div/rem. Does not check profitability.
FixedVectorType *getMaskedDivRemType(const TargetTransformInfo &TTI,
                                     unsigned Opcode, Type *ScalarTy,
                                     unsigned NumElts, bool ReVec);

/// Returns true if widened type of \p Ty elements with size \p Sz represents
/// full vector type, i.e. adding extra element results in extra parts upon type
/// legalization.
bool hasFullVectorsOrPowerOf2(const TargetTransformInfo &TTI, Type *Ty,
                              unsigned Sz, bool ReVec);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTYPEUTILS_H
