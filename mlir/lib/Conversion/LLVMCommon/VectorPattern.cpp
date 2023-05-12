//===- VectorPattern.cpp - Vector conversion pattern to the LLVM dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

// For >1-D vector types, extracts the necessary information to iterate over all
// 1-D subvectors in the underlying llrepresentation of the n-D vector
// Iterates on the llvm array type until we hit a non-array type (which is
// asserted to be an llvm vector type).
LLVM::detail::NDVectorTypeInfo
LLVM::detail::extractNDVectorTypeInfo(VectorType vectorType,
                                      LLVMTypeConverter &converter) {
  assert(vectorType.getRank() > 1 && "expected >1D vector type");
  NDVectorTypeInfo info;
  info.llvmNDVectorTy = converter.convertType(vectorType);
  if (!info.llvmNDVectorTy || !LLVM::isCompatibleType(info.llvmNDVectorTy)) {
    info.llvmNDVectorTy = nullptr;
    return info;
  }
  info.arraySizes.reserve(vectorType.getRank() - 1);
  auto llvmTy = info.llvmNDVectorTy;
  while (isa<LLVM::LLVMArrayType>(llvmTy)) {
    info.arraySizes.push_back(
        cast<LLVM::LLVMArrayType>(llvmTy).getNumElements());
    llvmTy = cast<LLVM::LLVMArrayType>(llvmTy).getElementType();
  }
  if (!LLVM::isCompatibleVectorType(llvmTy))
    return info;
  info.llvm1DVectorTy = llvmTy;
  return info;
}

// Express `linearIndex` in terms of coordinates of `basis`.
// Returns the empty vector when linearIndex is out of the range [0, P] where
// P is the product of all the basis coordinates.
//
// Prerequisites:
//   Basis is an array of nonnegative integers (signed type inherited from
//   vector shape type).
SmallVector<int64_t, 4> LLVM::detail::getCoordinates(ArrayRef<int64_t> basis,
                                                     unsigned linearIndex) {
  SmallVector<int64_t, 4> res;
  res.reserve(basis.size());
  for (unsigned basisElement : llvm::reverse(basis)) {
    res.push_back(linearIndex % basisElement);
    linearIndex = linearIndex / basisElement;
  }
  if (linearIndex > 0)
    return {};
  std::reverse(res.begin(), res.end());
  return res;
}

// Iterate of linear index, convert to coords space and insert splatted 1-D
// vector in each position.
void LLVM::detail::nDVectorIterate(const LLVM::detail::NDVectorTypeInfo &info,
                                   OpBuilder &builder,
                                   function_ref<void(ArrayRef<int64_t>)> fun) {
  unsigned ub = 1;
  for (auto s : info.arraySizes)
    ub *= s;
  for (unsigned linearIndex = 0; linearIndex < ub; ++linearIndex) {
    auto coords = getCoordinates(info.arraySizes, linearIndex);
    // Linear index is out of bounds, we are done.
    if (coords.empty())
      break;
    assert(coords.size() == info.arraySizes.size());
    fun(coords);
  }
}

LogicalResult LLVM::detail::handleMultidimensionalVectors(
    Operation *op, ValueRange operands, LLVMTypeConverter &typeConverter,
    std::function<Value(Type, ValueRange)> createOperand,
    ConversionPatternRewriter &rewriter) {
  auto resultNDVectorType = cast<VectorType>(op->getResult(0).getType());
  auto resultTypeInfo =
      extractNDVectorTypeInfo(resultNDVectorType, typeConverter);
  auto result1DVectorTy = resultTypeInfo.llvm1DVectorTy;
  auto resultNDVectoryTy = resultTypeInfo.llvmNDVectorTy;
  auto loc = op->getLoc();
  Value desc = rewriter.create<LLVM::UndefOp>(loc, resultNDVectoryTy);
  nDVectorIterate(resultTypeInfo, rewriter, [&](ArrayRef<int64_t> position) {
    // For this unrolled `position` corresponding to the `linearIndex`^th
    // element, extract operand vectors
    SmallVector<Value, 4> extractedOperands;
    for (const auto &operand : llvm::enumerate(operands)) {
      extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, operand.value(), position));
    }
    Value newVal = createOperand(result1DVectorTy, extractedOperands);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, newVal, position);
  });
  rewriter.replaceOp(op, desc);
  return success();
}

LogicalResult LLVM::detail::vectorOneToOneRewrite(
    Operation *op, StringRef targetOp, ValueRange operands,
    ArrayRef<NamedAttribute> targetAttrs, LLVMTypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  assert(!operands.empty());

  // Cannot convert ops if their operands are not of LLVM type.
  if (!llvm::all_of(operands.getTypes(), isCompatibleType))
    return failure();

  auto llvmNDVectorTy = operands[0].getType();
  if (!isa<LLVM::LLVMArrayType>(llvmNDVectorTy))
    return oneToOneRewrite(op, targetOp, operands, targetAttrs, typeConverter,
                           rewriter);

  auto callback = [op, targetOp, targetAttrs, &rewriter](Type llvm1DVectorTy,
                                                         ValueRange operands) {
    return rewriter
        .create(op->getLoc(), rewriter.getStringAttr(targetOp), operands,
                llvm1DVectorTy, targetAttrs)
        ->getResult(0);
  };

  return handleMultidimensionalVectors(op, operands, typeConverter, callback,
                                       rewriter);
}
