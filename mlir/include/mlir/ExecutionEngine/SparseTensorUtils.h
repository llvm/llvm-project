//===- SparseTensorUtils.h - Enums shared with the runtime ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines several enums shared between
// Transforms/SparseTensorConversion.cpp and ExecutionEngine/SparseUtils.cpp
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_
#define MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_

#include <cinttypes>

extern "C" {

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index". For now, we simply assume that
/// type is 64-bit, but targets with different "index" bit widths should link
/// with an alternatively built runtime support library.
// TODO: support such targets?
using index_type = uint64_t;

/// Encoding of overhead types (both pointer overhead and indices
/// overhead), for "overloading" @newSparseTensor.
enum class OverheadType : uint32_t {
  kIndex = 0,
  kU64 = 1,
  kU32 = 2,
  kU16 = 3,
  kU8 = 4
};

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
enum class PrimaryType : uint32_t {
  kF64 = 1,
  kF32 = 2,
  kI64 = 3,
  kI32 = 4,
  kI16 = 5,
  kI8 = 6,
  kC64 = 7,
  kC32 = 8
};

/// The actions performed by @newSparseTensor.
enum class Action : uint32_t {
  kEmpty = 0,
  kFromFile = 1,
  kFromCOO = 2,
  kSparseToSparse = 3,
  kEmptyCOO = 4,
  kToCOO = 5,
  kToIterator = 6
};

/// This enum mimics `SparseTensorEncodingAttr::DimLevelType` for
/// breaking dependency cycles.  `SparseTensorEncodingAttr::DimLevelType`
/// is the source of truth and this enum should be kept consistent with it.
enum class DimLevelType : uint8_t {
  kDense = 0,
  kCompressed = 1,
  kSingleton = 2
};

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_
