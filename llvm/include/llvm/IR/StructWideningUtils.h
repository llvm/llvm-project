//===---- StructWideningUtils.h - Utils for widening scalar to vector calls
//--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_STRUCTWIDENINGUTILS_H
#define LLVM_IR_STRUCTWIDENINGUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"

namespace llvm {

/// A helper for converting structs of scalar types to structs of vector types.
/// Note: Only unpacked literal struct types are supported.
Type *ToWideStructTy(StructType *StructTy, ElementCount EC);

/// A helper for converting structs of vector types to structs of scalar types.
/// Note: Only unpacked literal struct types are supported.
Type *ToNarrowStructTy(StructType *StructTy);

/// Returns true if `StructTy` is an unpacked literal struct where all elements
/// are vectors of matching element count. This does not include empty structs.
bool isWideStructTy(StructType *StructTy);

} // namespace llvm

#endif
