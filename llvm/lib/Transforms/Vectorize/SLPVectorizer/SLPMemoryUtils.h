//===- SLPMemoryUtils.h - SLP pointer/stride helpers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free pointer and
// stride helpers that do not depend on BoUpSLP or any other SLP-private type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPMEMORYUTILS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPMEMORYUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"

namespace llvm {
class DataLayout;
class SCEV;
class ScalarEvolution;
class TargetLibraryInfo;
class Type;
class Value;
} // namespace llvm

namespace llvm::slpvectorizer {

/// \p MaxDepth is the recursion limit for getUnderlyingObject.
bool arePointersCompatible(Value *Ptr1, Value *Ptr2,
                           const TargetLibraryInfo &TLI, unsigned MaxDepth,
                           bool CompareOpcodes = true);

/// Calculates minimal alignment as a common alignment.
template <typename T> Align computeCommonAlignment(ArrayRef<Value *> VL);

/// Checks if the provided list of pointers \p Pointers represents the strided
/// pointers for type ElemTy. If they are not, nullptr is returned.
/// Otherwise, SCEV* of the stride value is returned.
/// If `PointerOps` can be rearranged into the following sequence:
/// ```
/// %x + c_0 * stride,
/// %x + c_1 * stride,
/// %x + c_2 * stride
/// ...
/// ```
/// where each `c_i` is constant. The SCEV of the `stride` will be returned.
const SCEV *calculateRtStride(ArrayRef<Value *> PointerOps, Type *ElemTy,
                              const DataLayout &DL, ScalarEvolution &SE,
                              SmallVectorImpl<unsigned> &SortedIndices);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPMEMORYUTILS_H
