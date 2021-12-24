//===- VectorToROCDL.cpp - Vector to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// Vector operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToROCDL/VectorToROCDL.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;

static LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ValueRange operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferReadOp xferOp,
    Type &vecTy, Value &dwordConfig, Value &vindex, Value &offsetSizeInBytes,
    Value &glc, Value &slc) {
  rewriter.replaceOpWithNewOp<ROCDL::MubufLoadOp>(
      xferOp, vecTy, dwordConfig, vindex, offsetSizeInBytes, glc, slc);
  return success();
}

static LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ValueRange operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferWriteOp xferOp,
    Type &vecTy, Value &dwordConfig, Value &vindex, Value &offsetSizeInBytes,
    Value &glc, Value &slc) {
  auto adaptor = TransferWriteOpAdaptor(operands, xferOp->getAttrDictionary());
  rewriter.replaceOpWithNewOp<ROCDL::MubufStoreOp>(xferOp, adaptor.vector(),
                                                   dwordConfig, vindex,
                                                   offsetSizeInBytes, glc, slc);
  return success();
}

namespace {
/// Conversion pattern that converts a 1-D vector transfer read/write.
/// Note that this conversion pass only converts vector x2 or x4 f32
/// types. For unsupported cases, they will fall back to the vector to
/// llvm conversion pattern.
template <typename ConcreteOp>
class VectorTransferConversion : public ConvertOpToLLVMPattern<ConcreteOp> {
public:
  using ConvertOpToLLVMPattern<ConcreteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConcreteOp xferOp, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();

    if (!xferOp.permutation_map().isMinorIdentity())
      return failure();

    // Have it handled in vector->llvm conversion pass.
    if (xferOp.isDimInBounds(0))
      return failure();

    auto toLLVMTy = [&](Type t) {
      return this->getTypeConverter()->convertType(t);
    };
    auto vecTy = toLLVMTy(xferOp.getVectorType());
    unsigned vecWidth = LLVM::getVectorNumElements(vecTy).getFixedValue();
    Location loc = xferOp->getLoc();

    // The backend result vector scalarization have trouble scalarize
    // <1 x ty> result, exclude the x1 width from the lowering.
    if (vecWidth != 2 && vecWidth != 4)
      return failure();

    // Obtain dataPtr and elementType from the memref.
    auto memRefType = xferOp.getShapedType().template dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();
    // MUBUF instruction operate only on addresspace 0(unified) or 1(global)
    // In case of 3(LDS): fall back to vector->llvm pass
    // In case of 5(VGPR): wrong
    if ((memRefType.getMemorySpaceAsInt() != 0) &&
        (memRefType.getMemorySpaceAsInt() != 1))
      return failure();

    // Note that the dataPtr starts at the offset address specified by
    // indices, so no need to calculate offset size in bytes again in
    // the MUBUF instruction.
    Value dataPtr = this->getStridedElementPtr(
        loc, memRefType, adaptor.source(), adaptor.indices(), rewriter);

    // 1. Create and fill a <4 x i32> dwordConfig with:
    //    1st two elements holding the address of dataPtr.
    //    3rd element: -1.
    //    4th element: 0x27000.
    SmallVector<int32_t, 4> constConfigAttr{0, 0, -1, 0x27000};
    Type i32Ty = rewriter.getIntegerType(32);
    VectorType i32Vecx4 = VectorType::get(4, i32Ty);
    Value constConfig = rewriter.create<LLVM::ConstantOp>(
        loc, toLLVMTy(i32Vecx4),
        DenseElementsAttr::get(i32Vecx4, ArrayRef<int32_t>(constConfigAttr)));

    // Treat first two element of <4 x i32> as i64, and save the dataPtr
    // to it.
    Type i64Ty = rewriter.getIntegerType(64);
    Value i64x2Ty = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::getFixedVectorType(toLLVMTy(i64Ty), 2), constConfig);
    Value dataPtrAsI64 = rewriter.create<LLVM::PtrToIntOp>(
        loc, toLLVMTy(i64Ty).template cast<Type>(), dataPtr);
    Value zero = this->createIndexConstant(rewriter, loc, 0);
    Value dwordConfig = rewriter.create<LLVM::InsertElementOp>(
        loc, LLVM::getFixedVectorType(toLLVMTy(i64Ty), 2), i64x2Ty,
        dataPtrAsI64, zero);
    dwordConfig =
        rewriter.create<LLVM::BitcastOp>(loc, toLLVMTy(i32Vecx4), dwordConfig);

    // 2. Rewrite op as a buffer read or write.
    Value int1False = rewriter.create<LLVM::ConstantOp>(
        loc, toLLVMTy(rewriter.getIntegerType(1)),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    Value int32Zero = rewriter.create<LLVM::ConstantOp>(
        loc, toLLVMTy(i32Ty),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
    return replaceTransferOpWithMubuf(
        rewriter, adaptor.getOperands(), *this->getTypeConverter(), loc, xferOp,
        vecTy, dwordConfig, int32Zero, int32Zero, int1False, int1False);
  }
};
} // namespace

void mlir::populateVectorToROCDLConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<VectorTransferConversion<TransferReadOp>,
               VectorTransferConversion<TransferWriteOp>>(converter);
}

namespace {
struct LowerVectorToROCDLPass
    : public ConvertVectorToROCDLBase<LowerVectorToROCDLPass> {
  void runOnOperation() override;
};
} // namespace

void LowerVectorToROCDLPass::runOnOperation() {
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());

  populateVectorToROCDLConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addLegalDialect<ROCDL::ROCDLDialect>();

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToROCDLPass() {
  return std::make_unique<LowerVectorToROCDLPass>();
}
