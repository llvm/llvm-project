//===- AVXTranspose.cpp - Lower Vector transpose to AVX -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements vector.transpose rewrites as AVX patterns for particular
// sizes of interest.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;
using namespace mlir::x86vector::avx2;
using namespace mlir::x86vector::avx2::inline_asm;
using namespace mlir::x86vector::avx2::intrin;

Value mlir::x86vector::avx2::inline_asm::mm256BlendPsAsm(
    ImplicitLocOpBuilder &b, Value v1, Value v2, uint8_t mask) {
  auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel);
  const auto *asmTp = "vblendps $0, $1, $2, {0}";
  const auto *asmCstr =
      "=x,x,x"; // Careful: constraint parser is very brittle: no ws!
  SmallVector<Value> asmVals{v1, v2};
  auto asmStr = llvm::formatv(asmTp, llvm::format_hex(mask, /*width=*/2)).str();
  auto asmOp = LLVM::InlineAsmOp::create(
      b, v1.getType(), /*operands=*/asmVals, /*asm_string=*/asmStr,
      /*constraints=*/asmCstr, /*has_side_effects=*/false,
      /*is_align_stack=*/false, LLVM::TailCallKind::None,
      /*asm_dialect=*/asmDialectAttr,
      /*operand_attrs=*/ArrayAttr());
  return asmOp.getResult(0);
}

Value mlir::x86vector::avx2::intrin::mm256UnpackLoPs(ImplicitLocOpBuilder &b,
                                                     Value v1, Value v2) {
  return vector::ShuffleOp::create(b, v1, v2,
                                   ArrayRef<int64_t>{0, 8, 1, 9, 4, 12, 5, 13});
}

Value mlir::x86vector::avx2::intrin::mm256UnpackHiPs(ImplicitLocOpBuilder &b,
                                                     Value v1, Value v2) {
  return vector::ShuffleOp::create(
      b, v1, v2, ArrayRef<int64_t>{2, 10, 3, 11, 6, 14, 7, 15});
}
///                            a  a   b   b  a  a   b   b
/// Takes an 8 bit mask, 2 bit for each position of a[0, 3)  **and** b[0, 4):
///                                 0:127    |         128:255
///                            b01  b23  C8  D8  |  b01+4 b23+4 C8+4 D8+4
Value mlir::x86vector::avx2::intrin::mm256ShufflePs(ImplicitLocOpBuilder &b,
                                                    Value v1, Value v2,
                                                    uint8_t mask) {
  uint8_t b01, b23, b45, b67;
  MaskHelper::extractShuffle(mask, b01, b23, b45, b67);
  SmallVector<int64_t> shuffleMask = {
      b01, b23, b45 + 8, b67 + 8, b01 + 4, b23 + 4, b45 + 8 + 4, b67 + 8 + 4};
  return vector::ShuffleOp::create(b, v1, v2, shuffleMask);
}

// imm[0:1] out of imm[0:3] is:
//    0             1           2             3
// a[0:127] or a[128:255] or b[0:127] or b[128:255]    |
//          a[0:127] or a[128:255] or b[0:127] or b[128:255]
//             0             1           2             3
// imm[0:1] out of imm[4:7].
Value mlir::x86vector::avx2::intrin::mm256Permute2f128Ps(
    ImplicitLocOpBuilder &b, Value v1, Value v2, uint8_t mask) {
  SmallVector<int64_t> shuffleMask;
  auto appendToMask = [&](uint8_t control) {
    if (control == 0)
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{0, 1, 2, 3});
    else if (control == 1)
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{4, 5, 6, 7});
    else if (control == 2)
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{8, 9, 10, 11});
    else if (control == 3)
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{12, 13, 14, 15});
    else
      llvm_unreachable("control > 3 : overflow");
  };
  uint8_t b03, b47;
  MaskHelper::extractPermute(mask, b03, b47);
  appendToMask(b03);
  appendToMask(b47);
  return vector::ShuffleOp::create(b, v1, v2, shuffleMask);
}

/// If bit i of `mask` is zero, take f32@i from v1 else take it from v2.
Value mlir::x86vector::avx2::intrin::mm256BlendPs(ImplicitLocOpBuilder &b,
                                                  Value v1, Value v2,
                                                  uint8_t mask) {
  SmallVector<int64_t, 8> shuffleMask;
  for (int i = 0; i < 8; ++i) {
    bool isSet = mask & (1 << i);
    shuffleMask.push_back(!isSet ? i : i + 8);
  }
  return vector::ShuffleOp::create(b, v1, v2, shuffleMask);
}

/// AVX2 4x8xf32-specific transpose lowering using a "C intrinsics" model.
void mlir::x86vector::avx2::transpose4x8xf32(ImplicitLocOpBuilder &ib,
                                             MutableArrayRef<Value> vs) {
#ifndef NDEBUG
  auto vt = VectorType::get({8}, Float32Type::get(ib.getContext()));
  assert(vs.size() == 4 && "expects 4 vectors");
  assert(llvm::all_of(ValueRange{vs}.getTypes(),
                      [&](Type t) { return t == vt; }) &&
         "expects all types to be vector<8xf32>");
#endif

  Value t0 = mm256UnpackLoPs(ib, vs[0], vs[1]);
  Value t1 = mm256UnpackHiPs(ib, vs[0], vs[1]);
  Value t2 = mm256UnpackLoPs(ib, vs[2], vs[3]);
  Value t3 = mm256UnpackHiPs(ib, vs[2], vs[3]);
  Value s0 = mm256ShufflePs(ib, t0, t2, MaskHelper::shuffle<1, 0, 1, 0>());
  Value s1 = mm256ShufflePs(ib, t0, t2, MaskHelper::shuffle<3, 2, 3, 2>());
  Value s2 = mm256ShufflePs(ib, t1, t3, MaskHelper::shuffle<1, 0, 1, 0>());
  Value s3 = mm256ShufflePs(ib, t1, t3, MaskHelper::shuffle<3, 2, 3, 2>());
  vs[0] = mm256Permute2f128Ps(ib, s0, s1, MaskHelper::permute<2, 0>());
  vs[1] = mm256Permute2f128Ps(ib, s2, s3, MaskHelper::permute<2, 0>());
  vs[2] = mm256Permute2f128Ps(ib, s0, s1, MaskHelper::permute<3, 1>());
  vs[3] = mm256Permute2f128Ps(ib, s2, s3, MaskHelper::permute<3, 1>());
}

/// AVX2 8x8xf32-specific transpose lowering using a "C intrinsics" model.
void mlir::x86vector::avx2::transpose8x8xf32(ImplicitLocOpBuilder &ib,
                                             MutableArrayRef<Value> vs) {
  auto vt = VectorType::get({8}, Float32Type::get(ib.getContext()));
  (void)vt;
  assert(vs.size() == 8 && "expects 8 vectors");
  assert(llvm::all_of(ValueRange{vs}.getTypes(),
                      [&](Type t) { return t == vt; }) &&
         "expects all types to be vector<8xf32>");

  Value t0 = mm256UnpackLoPs(ib, vs[0], vs[1]);
  Value t1 = mm256UnpackHiPs(ib, vs[0], vs[1]);
  Value t2 = mm256UnpackLoPs(ib, vs[2], vs[3]);
  Value t3 = mm256UnpackHiPs(ib, vs[2], vs[3]);
  Value t4 = mm256UnpackLoPs(ib, vs[4], vs[5]);
  Value t5 = mm256UnpackHiPs(ib, vs[4], vs[5]);
  Value t6 = mm256UnpackLoPs(ib, vs[6], vs[7]);
  Value t7 = mm256UnpackHiPs(ib, vs[6], vs[7]);

  using inline_asm::mm256BlendPsAsm;
  Value sh0 = mm256ShufflePs(ib, t0, t2, MaskHelper::shuffle<1, 0, 3, 2>());
  Value sh2 = mm256ShufflePs(ib, t1, t3, MaskHelper::shuffle<1, 0, 3, 2>());
  Value sh4 = mm256ShufflePs(ib, t4, t6, MaskHelper::shuffle<1, 0, 3, 2>());
  Value sh6 = mm256ShufflePs(ib, t5, t7, MaskHelper::shuffle<1, 0, 3, 2>());

  Value s0 =
      mm256BlendPsAsm(ib, t0, sh0, MaskHelper::blend<0, 0, 1, 1, 0, 0, 1, 1>());
  Value s1 =
      mm256BlendPsAsm(ib, t2, sh0, MaskHelper::blend<1, 1, 0, 0, 1, 1, 0, 0>());
  Value s2 =
      mm256BlendPsAsm(ib, t1, sh2, MaskHelper::blend<0, 0, 1, 1, 0, 0, 1, 1>());
  Value s3 =
      mm256BlendPsAsm(ib, t3, sh2, MaskHelper::blend<1, 1, 0, 0, 1, 1, 0, 0>());
  Value s4 =
      mm256BlendPsAsm(ib, t4, sh4, MaskHelper::blend<0, 0, 1, 1, 0, 0, 1, 1>());
  Value s5 =
      mm256BlendPsAsm(ib, t6, sh4, MaskHelper::blend<1, 1, 0, 0, 1, 1, 0, 0>());
  Value s6 =
      mm256BlendPsAsm(ib, t5, sh6, MaskHelper::blend<0, 0, 1, 1, 0, 0, 1, 1>());
  Value s7 =
      mm256BlendPsAsm(ib, t7, sh6, MaskHelper::blend<1, 1, 0, 0, 1, 1, 0, 0>());

  vs[0] = mm256Permute2f128Ps(ib, s0, s4, MaskHelper::permute<2, 0>());
  vs[1] = mm256Permute2f128Ps(ib, s1, s5, MaskHelper::permute<2, 0>());
  vs[2] = mm256Permute2f128Ps(ib, s2, s6, MaskHelper::permute<2, 0>());
  vs[3] = mm256Permute2f128Ps(ib, s3, s7, MaskHelper::permute<2, 0>());
  vs[4] = mm256Permute2f128Ps(ib, s0, s4, MaskHelper::permute<3, 1>());
  vs[5] = mm256Permute2f128Ps(ib, s1, s5, MaskHelper::permute<3, 1>());
  vs[6] = mm256Permute2f128Ps(ib, s2, s6, MaskHelper::permute<3, 1>());
  vs[7] = mm256Permute2f128Ps(ib, s3, s7, MaskHelper::permute<3, 1>());
}

/// Rewrite AVX2-specific vector.transpose, for the supported cases and
/// depending on the `TransposeLoweringOptions`. The lowering supports 2-D
/// transpose cases and n-D cases that have been decomposed into 2-D
/// transposition slices. For example, a 3-D transpose:
///
///   %0 = vector.transpose %arg0, [2, 0, 1]
///      : vector<1024x2048x4096xf32> to vector<4096x1024x2048xf32>
///
/// could be sliced into 2-D transposes by tiling two of its dimensions to one
/// of the vector lengths supported by the AVX2 patterns (e.g., 4x8):
///
///   %0 = vector.transpose %arg0, [2, 0, 1]
///      : vector<1x4x8xf32> to vector<8x1x4xf32>
///
/// This lowering will analyze the n-D vector.transpose and determine if it's a
/// supported 2-D transposition slice where any of the AVX2 patterns can be
/// applied.
class TransposeOpLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  TransposeOpLowering(LoweringOptions loweringOptions, MLIRContext *context,
                      int benefit)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        loweringOptions(loweringOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Check if the source vector type is supported. AVX2 patterns can only be
    // applied to f32 vector types with two dimensions greater than one.
    VectorType srcType = op.getSourceVectorType();
    if (!srcType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "Unsupported vector element type");

    auto srcGtOneDims = mlir::vector::isTranspose2DSlice(op);
    if (failed(srcGtOneDims))
      return rewriter.notifyMatchFailure(
          op, "expected transposition on a 2D slice");

    // Retrieve the sizes of the two dimensions greater than one to be
    // transposed.
    int64_t m = srcType.getDimSize(std::get<0>(srcGtOneDims.value()));
    int64_t n = srcType.getDimSize(std::get<1>(srcGtOneDims.value()));

    auto applyRewrite = [&]() {
      ImplicitLocOpBuilder ib(loc, rewriter);
      SmallVector<Value> vs;

      // Reshape the n-D input vector with only two dimensions greater than one
      // to a 2-D vector.
      auto flattenedType =
          VectorType::get({n * m}, op.getSourceVectorType().getElementType());
      auto reshInputType = VectorType::get({m, n}, srcType.getElementType());
      auto reshInput =
          vector::ShapeCastOp::create(ib, flattenedType, op.getVector());
      reshInput = vector::ShapeCastOp::create(ib, reshInputType, reshInput);

      // Extract 1-D vectors from the higher-order dimension of the input
      // vector.
      for (int64_t i = 0; i < m; ++i)
        vs.push_back(vector::ExtractOp::create(ib, reshInput, i));

      // Transpose set of 1-D vectors.
      if (m == 4)
        transpose4x8xf32(ib, vs);
      if (m == 8)
        transpose8x8xf32(ib, vs);

      // Insert transposed 1-D vectors into the higher-order dimension of the
      // output vector.
      Value res = arith::ConstantOp::create(ib, reshInputType,
                                            ib.getZeroAttr(reshInputType));
      for (int64_t i = 0; i < m; ++i)
        res = vector::InsertOp::create(ib, vs[i], res, i);

      // The output vector still has the shape of the input vector (e.g., 4x8).
      // We have to transpose their dimensions and retrieve its original rank
      // (e.g., 1x8x1x4x1).
      res = vector::ShapeCastOp::create(ib, flattenedType, res);
      res = vector::ShapeCastOp::create(ib, op.getResultVectorType(), res);
      rewriter.replaceOp(op, res);
      return success();
    };

    if (loweringOptions.transposeOptions.lower4x8xf32_ && m == 4 && n == 8)
      return applyRewrite();
    if (loweringOptions.transposeOptions.lower8x8xf32_ && m == 8 && n == 8)
      return applyRewrite();
    return failure();
  }

private:
  LoweringOptions loweringOptions;
};

void mlir::x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
    RewritePatternSet &patterns, LoweringOptions options, int benefit) {
  patterns.add<TransposeOpLowering>(options, patterns.getContext(), benefit);
}
