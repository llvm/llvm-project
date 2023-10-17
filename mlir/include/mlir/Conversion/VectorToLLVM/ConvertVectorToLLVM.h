//===- ConvertVectorToLLVM.h - Utils to convert from the vector dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;

/// Collect a set of patterns to convert from Vector contractions to LLVM Matrix
/// Intrinsics. To lower to assembly, the LLVM flag -lower-matrix-intrinsics
/// will be needed when invoking LLVM.
void populateVectorToLLVMMatrixConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool reassociateFPReductions = false, bool force32BitVectorIndices = false);

/// Base class to convert a `vector.mask` operation while matching traits
/// of the maskable operation nested inside. A
/// `ConvertVectorMaskOpToLLVMPattern` instance matches against a `vector.mask`
/// operation. The `matchAndRewrite` method performs a second match against the
/// maskable operation `MaskedOp`. Finally, it invokes the virtual method
/// `matchAndRewriteMaskableOp` to be implemented by the concrete conversion
/// classes. This method can match against specific traits of the `vector.mask`
/// and the maskable operation. It must replace the `vector.mask` operation.
template <class MaskedOp>
class ConvertVectorMaskOpToLLVMPattern
    : public ConvertOpToLLVMPattern<vector::MaskOp> {
public:
  using ConvertOpToLLVMPattern<vector::MaskOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MaskOp maskOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Match against the maskable operation kind.
    auto maskedOp = llvm::dyn_cast_or_null<MaskedOp>(maskOp.getMaskableOp());
    if (!maskedOp)
      return failure();
    return matchAndRewriteMaskableOp(maskOp, maskedOp, rewriter);
  }

protected:
  virtual LogicalResult
  matchAndRewriteMaskableOp(vector::MaskOp maskOp, MaskedOp maskableOp,
                            ConversionPatternRewriter &rewriter) const = 0;
};

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
