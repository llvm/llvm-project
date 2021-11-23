//=- Transforms.h - X86Vector Dialect Transformation Entrypoints -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86VECTOR_TRANSFORMS_H
#define MLIR_DIALECT_X86VECTOR_TRANSFORMS_H

#include "mlir/IR/Value.h"

namespace mlir {

class ImplicitLocOpBuilder;
class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

namespace x86vector {

/// Helper class to factor out the creation and extraction of masks from nibs.
struct MaskHelper {
  /// b01 captures the lower 2 bits, b67 captures the higher 2 bits.
  /// Meant to be used with instructions such as mm256ShufflePs.
  template <unsigned b67, unsigned b45, unsigned b23, unsigned b01>
  static int8_t shuffle() {
    static_assert(b01 <= 0x03, "overflow");
    static_assert(b23 <= 0x03, "overflow");
    static_assert(b45 <= 0x03, "overflow");
    static_assert(b67 <= 0x03, "overflow");
    return static_cast<int8_t>((b67 << 6) | (b45 << 4) | (b23 << 2) | b01);
  }
  /// b01 captures the lower 2 bits, b67 captures the higher 2 bits.
  static void extractShuffle(int8_t mask, int8_t &b01, int8_t &b23, int8_t &b45,
                             int8_t &b67) {
    b67 = (mask & (0x03 << 6)) >> 6;
    b45 = (mask & (0x03 << 4)) >> 4;
    b23 = (mask & (0x03 << 2)) >> 2;
    b01 = mask & 0x03;
  }
  /// b03 captures the lower 4 bits, b47 captures the higher 4 bits.
  /// Meant to be used with instructions such as mm256Permute2f128Ps.
  template <unsigned b47, unsigned b03>
  static int8_t permute() {
    static_assert(b03 <= 0x0f, "overflow");
    static_assert(b47 <= 0x0f, "overflow");
    return static_cast<int8_t>((b47 << 4) + b03);
  }
  /// b03 captures the lower 4 bits, b47 captures the higher 4 bits.
  static void extractPermute(int8_t mask, int8_t &b03, int8_t &b47) {
    b47 = (mask & (0x0f << 4)) >> 4;
    b03 = mask & 0x0f;
  }
};

//===----------------------------------------------------------------------===//
/// Helpers extracted from:
///   - clang/lib/Headers/avxintrin.h
///   - clang/test/CodeGen/X86/avx-builtins.c
///   - clang/test/CodeGen/X86/avx2-builtins.c
///   - clang/test/CodeGen/X86/avx-shuffle-builtins.c
/// as well as the Intel Intrinsics Guide
/// (https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
/// make it easier to just implement known good lowerings.
/// All intrinsics correspond 1-1 to the Intel definition.
//===----------------------------------------------------------------------===//

namespace avx2 {

/// Lower to vector.shuffle v1, v2, [0, 8, 1, 9, 4, 12, 5, 13].
Value mm256UnpackLoPs(ImplicitLocOpBuilder &b, Value v1, Value v2);

/// Lower to vector.shuffle v1, v2, [0, 8, 1, 9, 4, 12, 5, 13].
Value mm256UnpackHiPs(ImplicitLocOpBuilder &b, Value v1, Value v2);

///                            a  a   b   b  a  a   b   b
/// Take an 8 bit mask, 2 bit for each position of a[0, 3)  **and** b[0, 4):
///                                 0:127    |         128:255
///                            b01  b23  C8  D8  |  b01+4 b23+4 C8+4 D8+4
Value mm256ShufflePs(ImplicitLocOpBuilder &b, Value v1, Value v2, int8_t mask);

// imm[0:1] out of imm[0:3] is:
//    0             1           2             3
// a[0:127] or a[128:255] or b[0:127] or b[128:255]    |
//          a[0:127] or a[128:255] or b[0:127] or b[128:255]
//             0             1           2             3
// imm[0:1] out of imm[4:7].
Value mm256Permute2f128Ps(ImplicitLocOpBuilder &b, Value v1, Value v2,
                          int8_t mask);

/// 4x8xf32-specific AVX2 transpose lowering.
void transpose4x8xf32(ImplicitLocOpBuilder &ib, MutableArrayRef<Value> vs);

/// 8x8xf32-specific AVX2 transpose lowering.
void transpose8x8xf32(ImplicitLocOpBuilder &ib, MutableArrayRef<Value> vs);

/// Structure to control the behavior of specialized AVX2 transpose lowering.
struct TransposeLoweringOptions {
  bool lower4x8xf32_ = false;
  TransposeLoweringOptions &lower4x8xf32(bool lower = true) {
    lower4x8xf32_ = lower;
    return *this;
  }
  bool lower8x8xf32_ = false;
  TransposeLoweringOptions &lower8x8xf32(bool lower = true) {
    lower8x8xf32_ = lower;
    return *this;
  }
};

/// Options for controlling specialized AVX2 lowerings.
struct LoweringOptions {
  /// Configure specialized vector lowerings.
  TransposeLoweringOptions transposeOptions;
  LoweringOptions &setTransposeOptions(TransposeLoweringOptions options) {
    transposeOptions = options;
    return *this;
  }
};

/// Insert specialized transpose lowering patterns.
void populateSpecializedTransposeLoweringPatterns(
    RewritePatternSet &patterns, LoweringOptions options = LoweringOptions(),
    int benefit = 10);

} // namespace avx2
} // namespace x86vector

/// Collect a set of patterns to lower X86Vector ops to ops that map to LLVM
/// intrinsics.
void populateX86VectorLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns);

/// Configure the target to support lowering X86Vector ops to ops that map to
/// LLVM intrinsics.
void configureX86VectorLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // MLIR_DIALECT_X86VECTOR_TRANSFORMS_H
