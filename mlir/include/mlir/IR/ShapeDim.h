//===- ShapeDim.h - MLIR ShapeDim Class --------------------------------------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ShapeDim class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SHAPEDIM_H
#define MLIR_IR_SHAPEDIM_H

#include <cassert>
#include <cstdint>
#include <limits>

namespace mlir {

struct ShapeDim {
  /// Deprecated. Use ShapeDim::fixed(), ShapeDim::scalable(), or
  /// ShapeDim::kDynamic() instead.
  constexpr ShapeDim(int64_t size) : size(size) {}

  /// Deprecated. Use an explicit conversion instead.
  constexpr operator int64_t() const { return size; }

  /// Construct a scalable dimension.
  constexpr static ShapeDim scalable(int64_t size) {
    assert(size > 0);
    return ShapeDim{-size};
  }

  /// Construct a fixed dimension.
  constexpr static ShapeDim fixed(int64_t size) {
    assert(size >= 0);
    return ShapeDim{size};
  }

  /// Construct a dynamic dimension.
  constexpr static ShapeDim kDynamic() { return ShapeDim{}; }

  /// Returns whether this is a dynamic dimension.
  constexpr bool isDynamic() const { return size == kDynamic(); }

  /// Returns whether this is a scalable dimension.
  constexpr bool isScalable() const { return size < 0 && !isDynamic(); }

  /// Returns whether this is a fixed dimension.
  constexpr bool isFixed() const { return size >= 0; }

  /// Asserts a dimension is fixed and returns its size.
  constexpr int64_t fixedSize() const {
    assert(isFixed());
    return size;
  };

  /// Asserts a dimension is scalable and returns its size.
  constexpr int64_t scalableSize() const {
    assert(isScalable());
    return -size;
  }

  /// Returns the minimum (runtime) size for this dimension.
  constexpr int64_t minSize() const {
    if (isScalable())
      return scalableSize();
    if (isFixed())
      return fixedSize();
    return 0;
  }

private:
  constexpr explicit ShapeDim()
      : ShapeDim(std::numeric_limits<int64_t>::min()) {}

  int64_t size;
};

} // namespace mlir

#endif
