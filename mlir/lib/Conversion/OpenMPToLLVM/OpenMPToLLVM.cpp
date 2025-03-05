//===- OpenMPToLLVM.cpp - conversion from OpenMP to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTOPENMPTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pattern that converts the region arguments in a single-region OpenMP
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
struct RegionOpConversion : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpType curOp, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<OpType>(
        curOp.getLoc(), TypeRange(), adaptor.getOperands(), curOp->getAttrs());
    rewriter.inlineRegionBefore(curOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *this->getTypeConverter())))
      return failure();

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <typename T>
struct RegionLessOpWithVarOperandsConversion
    : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(T curOp, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return failure();
    SmallVector<Value> convertedOperands;
    assert(curOp.getNumVariableOperands() ==
               curOp.getOperation()->getNumOperands() &&
           "unexpected non-variable operands");
    for (unsigned idx = 0; idx < curOp.getNumVariableOperands(); ++idx) {
      Value originalVariableOperand = curOp.getVariableOperand(idx);
      if (!originalVariableOperand)
        return failure();
      if (isa<MemRefType>(originalVariableOperand.getType())) {
        // TODO: Support memref type in variable operands
        return rewriter.notifyMatchFailure(curOp,
                                           "memref is not supported yet");
      }
      convertedOperands.emplace_back(adaptor.getOperands()[idx]);
    }

    rewriter.replaceOpWithNewOp<T>(curOp, resTypes, convertedOperands,
                                   curOp->getAttrs());
    return success();
  }
};

template <typename T>
struct RegionOpWithVarOperandsConversion : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(T curOp, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return failure();
    SmallVector<Value> convertedOperands;
    assert(curOp.getNumVariableOperands() ==
               curOp.getOperation()->getNumOperands() &&
           "unexpected non-variable operands");
    for (unsigned idx = 0; idx < curOp.getNumVariableOperands(); ++idx) {
      Value originalVariableOperand = curOp.getVariableOperand(idx);
      if (!originalVariableOperand)
        return failure();
      if (isa<MemRefType>(originalVariableOperand.getType())) {
        // TODO: Support memref type in variable operands
        return rewriter.notifyMatchFailure(curOp,
                                           "memref is not supported yet");
      }
      convertedOperands.emplace_back(adaptor.getOperands()[idx]);
    }
    auto newOp = rewriter.create<T>(curOp.getLoc(), resTypes, convertedOperands,
                                    curOp->getAttrs());
    rewriter.inlineRegionBefore(curOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *this->getTypeConverter())))
      return failure();

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <typename T>
struct RegionLessOpConversion : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(T curOp, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return failure();

    rewriter.replaceOpWithNewOp<T>(curOp, resTypes, adaptor.getOperands(),
                                   curOp->getAttrs());
    return success();
  }
};

struct AtomicReadOpConversion
    : public ConvertOpToLLVMPattern<omp::AtomicReadOp> {
  using ConvertOpToLLVMPattern<omp::AtomicReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(omp::AtomicReadOp curOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    Type curElementType = curOp.getElementType();
    auto newOp = rewriter.create<omp::AtomicReadOp>(
        curOp.getLoc(), TypeRange(), adaptor.getOperands(), curOp->getAttrs());
    TypeAttr typeAttr = TypeAttr::get(converter->convertType(curElementType));
    newOp.setElementTypeAttr(typeAttr);
    rewriter.eraseOp(curOp);
    return success();
  }
};

struct MapInfoOpConversion : public ConvertOpToLLVMPattern<omp::MapInfoOp> {
  using ConvertOpToLLVMPattern<omp::MapInfoOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(omp::MapInfoOp curOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();

    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return failure();

    // Copy attributes of the curOp except for the typeAttr which should
    // be converted
    SmallVector<NamedAttribute> newAttrs;
    for (NamedAttribute attr : curOp->getAttrs()) {
      if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
        Type newAttr = converter->convertType(typeAttr.getValue());
        newAttrs.emplace_back(attr.getName(), TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }

    rewriter.replaceOpWithNewOp<omp::MapInfoOp>(
        curOp, resTypes, adaptor.getOperands(), newAttrs);
    return success();
  }
};

struct DeclMapperOpConversion
    : public ConvertOpToLLVMPattern<omp::DeclareMapperOp> {
  using ConvertOpToLLVMPattern<omp::DeclareMapperOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(omp::DeclareMapperOp curOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<NamedAttribute> newAttrs;
    newAttrs.emplace_back(curOp.getSymNameAttrName(), curOp.getSymNameAttr());
    newAttrs.emplace_back(
        curOp.getTypeAttrName(),
        TypeAttr::get(converter->convertType(curOp.getType())));

    auto newOp = rewriter.create<omp::DeclareMapperOp>(
        curOp.getLoc(), TypeRange(), adaptor.getOperands(), newAttrs);
    rewriter.inlineRegionBefore(curOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *this->getTypeConverter())))
      return failure();

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <typename OpType>
struct MultiRegionOpConversion : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  void forwardOpAttrs(OpType curOp, OpType newOp) const {}

  LogicalResult
  matchAndRewrite(OpType curOp, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<OpType>(
        curOp.getLoc(), TypeRange(), curOp.getSymNameAttr(),
        TypeAttr::get(this->getTypeConverter()->convertType(
            curOp.getTypeAttr().getValue())));
    forwardOpAttrs(curOp, newOp);

    for (unsigned idx = 0; idx < curOp.getNumRegions(); idx++) {
      rewriter.inlineRegionBefore(curOp.getRegion(idx), newOp.getRegion(idx),
                                  newOp.getRegion(idx).end());
      if (failed(rewriter.convertRegionTypes(&newOp.getRegion(idx),
                                             *this->getTypeConverter())))
        return failure();
    }

    rewriter.eraseOp(curOp);
    return success();
  }
};

template <>
void MultiRegionOpConversion<omp::PrivateClauseOp>::forwardOpAttrs(
    omp::PrivateClauseOp curOp, omp::PrivateClauseOp newOp) const {
  newOp.setDataSharingType(curOp.getDataSharingType());
}
} // namespace

void mlir::configureOpenMPToLLVMConversionLegality(
    ConversionTarget &target, const LLVMTypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<
      omp::AtomicReadOp, omp::AtomicWriteOp, omp::CancellationPointOp,
      omp::CancelOp, omp::CriticalDeclareOp, omp::DeclareMapperInfoOp,
      omp::FlushOp, omp::MapBoundsOp, omp::MapInfoOp, omp::OrderedOp,
      omp::ScanOp, omp::TargetEnterDataOp, omp::TargetExitDataOp,
      omp::TargetUpdateOp, omp::ThreadprivateOp, omp::YieldOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
      });
  target.addDynamicallyLegalOp<
      omp::AtomicUpdateOp, omp::CriticalOp, omp::DeclareMapperOp,
      omp::DeclareReductionOp, omp::DistributeOp, omp::LoopNestOp, omp::LoopOp,
      omp::MasterOp, omp::OrderedRegionOp, omp::ParallelOp,
      omp::PrivateClauseOp, omp::SectionOp, omp::SectionsOp, omp::SimdOp,
      omp::SingleOp, omp::TargetDataOp, omp::TargetOp, omp::TaskgroupOp,
      omp::TaskloopOp, omp::TaskOp, omp::TeamsOp,
      omp::WsloopOp>([&](Operation *op) {
    return std::all_of(op->getRegions().begin(), op->getRegions().end(),
                       [&](Region &region) {
                         return typeConverter.isLegal(&region);
                       }) &&
           typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<omp::PrivateClauseOp>(
      [&](omp::PrivateClauseOp op) -> bool {
        return std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&](Region &region) {
                             return typeConverter.isLegal(&region);
                           }) &&
               typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes()) &&
               typeConverter.isLegal(op.getType());
      });
}

void mlir::populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  // This type is allowed when converting OpenMP to LLVM Dialect, it carries
  // bounds information for map clauses and the operation and type are
  // discarded on lowering to LLVM-IR from the OpenMP dialect.
  converter.addConversion(
      [&](omp::MapBoundsType type) -> Type { return type; });

  patterns.add<
      AtomicReadOpConversion, DeclMapperOpConversion, MapInfoOpConversion,
      MultiRegionOpConversion<omp::DeclareReductionOp>,
      MultiRegionOpConversion<omp::PrivateClauseOp>,
      RegionLessOpConversion<omp::CancellationPointOp>,
      RegionLessOpConversion<omp::CancelOp>,
      RegionLessOpConversion<omp::CriticalDeclareOp>,
      RegionLessOpConversion<omp::DeclareMapperInfoOp>,
      RegionLessOpConversion<omp::OrderedOp>,
      RegionLessOpConversion<omp::ScanOp>,
      RegionLessOpConversion<omp::TargetEnterDataOp>,
      RegionLessOpConversion<omp::TargetExitDataOp>,
      RegionLessOpConversion<omp::TargetUpdateOp>,
      RegionLessOpConversion<omp::YieldOp>,
      RegionLessOpWithVarOperandsConversion<omp::AtomicWriteOp>,
      RegionLessOpWithVarOperandsConversion<omp::FlushOp>,
      RegionLessOpWithVarOperandsConversion<omp::MapBoundsOp>,
      RegionLessOpWithVarOperandsConversion<omp::ThreadprivateOp>,
      RegionOpConversion<omp::AtomicCaptureOp>,
      RegionOpConversion<omp::CriticalOp>,
      RegionOpConversion<omp::DistributeOp>,
      RegionOpConversion<omp::LoopNestOp>, RegionOpConversion<omp::LoopOp>,
      RegionOpConversion<omp::MaskedOp>, RegionOpConversion<omp::MasterOp>,
      RegionOpConversion<omp::OrderedRegionOp>,
      RegionOpConversion<omp::ParallelOp>, RegionOpConversion<omp::SectionOp>,
      RegionOpConversion<omp::SectionsOp>, RegionOpConversion<omp::SimdOp>,
      RegionOpConversion<omp::SingleOp>, RegionOpConversion<omp::TargetDataOp>,
      RegionOpConversion<omp::TargetOp>, RegionOpConversion<omp::TaskgroupOp>,
      RegionOpConversion<omp::TaskloopOp>, RegionOpConversion<omp::TaskOp>,
      RegionOpConversion<omp::TeamsOp>, RegionOpConversion<omp::WsloopOp>,
      RegionOpWithVarOperandsConversion<omp::AtomicUpdateOp>>(converter);
}

namespace {
struct ConvertOpenMPToLLVMPass
    : public impl::ConvertOpenMPToLLVMPassBase<ConvertOpenMPToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertOpenMPToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to OpenMP operations with LLVM IR dialect
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  cf::populateAssertToLLVMConversionPattern(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateOpenMPToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<omp::BarrierOp, omp::FlushOp, omp::TaskwaitOp,
                    omp::TaskyieldOp, omp::TerminatorOp>();
  configureOpenMPToLLVMConversionLegality(target, converter);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//
namespace {
/// Implement the interface to convert OpenMP to LLVM.
struct OpenMPToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    configureOpenMPToLLVMConversionLegality(target, typeConverter);
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertOpenMPToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, omp::OpenMPDialect *dialect) {
    dialect->addInterfaces<OpenMPToLLVMDialectInterface>();
  });
}
