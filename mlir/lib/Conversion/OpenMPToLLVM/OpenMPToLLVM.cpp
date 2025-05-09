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

/// A pattern that converts the result and operand types, attributes, and region
/// arguments of an OpenMP operation to the LLVM dialect.
///
/// Attributes are copied verbatim by default, and only translated if they are
/// type attributes.
///
/// Region bodies, if any, are not modified and expected to either be processed
/// by the conversion infrastructure or already contain ops compatible with LLVM
/// dialect types.
template <typename T>
struct OpenMPOpConversion : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Translate result types.
    const TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();
    SmallVector<Type> resTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), resTypes)))
      return failure();

    // Translate type attributes.
    // They are kept unmodified except if they are type attributes.
    SmallVector<NamedAttribute> convertedAttrs;
    for (NamedAttribute attr : op->getAttrs()) {
      if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
        Type convertedType = converter->convertType(typeAttr.getValue());
        convertedAttrs.emplace_back(attr.getName(),
                                    TypeAttr::get(convertedType));
      } else {
        convertedAttrs.push_back(attr);
      }
    }

    // Translate operands.
    SmallVector<Value> convertedOperands;
    convertedOperands.reserve(op->getNumOperands());
    for (auto [originalOperand, convertedOperand] :
         llvm::zip_equal(op->getOperands(), adaptor.getOperands())) {
      if (!originalOperand)
        return failure();

      // TODO: Revisit whether we need to trigger an error specifically for this
      // set of operations. Consider removing this check or updating the list.
      if constexpr (llvm::is_one_of<T, omp::AtomicUpdateOp, omp::AtomicWriteOp,
                                    omp::FlushOp, omp::MapBoundsOp,
                                    omp::ThreadprivateOp>::value) {
        if (isa<MemRefType>(originalOperand.getType())) {
          // TODO: Support memref type in variable operands
          return rewriter.notifyMatchFailure(op, "memref is not supported yet");
        }
      }
      convertedOperands.push_back(convertedOperand);
    }

    // Create new operation.
    auto newOp = rewriter.create<T>(op.getLoc(), resTypes, convertedOperands,
                                    convertedAttrs);

    // Translate regions.
    for (auto [originalRegion, convertedRegion] :
         llvm::zip_equal(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(originalRegion, convertedRegion,
                                  convertedRegion.end());
      if (failed(rewriter.convertRegionTypes(&convertedRegion,
                                             *this->getTypeConverter())))
        return failure();
    }

    // Delete old operation and replace result uses with those of the new one.
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void mlir::configureOpenMPToLLVMConversionLegality(
    ConversionTarget &target, const LLVMTypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes()) &&
           std::all_of(op->getRegions().begin(), op->getRegions().end(),
                       [&](Region &region) {
                         return typeConverter.isLegal(&region);
                       }) &&
           std::all_of(op->getAttrs().begin(), op->getAttrs().end(),
                       [&](NamedAttribute attr) {
                         auto typeAttr = dyn_cast<TypeAttr>(attr.getValue());
                         return !typeAttr ||
                                typeConverter.isLegal(typeAttr.getValue());
                       });
  });
}

/// Add an `OpenMPOpConversion<T>` conversion pattern for each operation type
/// passed as template argument.
template <typename... Ts>
static inline RewritePatternSet &
addOpenMPOpConversions(LLVMTypeConverter &converter,
                       RewritePatternSet &patterns) {
  return patterns.add<OpenMPOpConversion<Ts>...>(converter);
}

void mlir::populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  // This type is allowed when converting OpenMP to LLVM Dialect, it carries
  // bounds information for map clauses and the operation and type are
  // discarded on lowering to LLVM-IR from the OpenMP dialect.
  converter.addConversion(
      [&](omp::MapBoundsType type) -> Type { return type; });

  // Add conversions for all OpenMP operations.
  addOpenMPOpConversions<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >(converter, patterns);
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
