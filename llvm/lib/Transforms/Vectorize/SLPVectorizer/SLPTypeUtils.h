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

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTYPEUTILS_H
