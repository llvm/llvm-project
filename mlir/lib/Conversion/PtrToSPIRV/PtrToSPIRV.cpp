//===- PtrToSPIRV.cpp - Ptr to SPIR-V dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PtrToSPIRV/PtrToSPIRV.h"

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTPTRTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static FailureOr<Type> getAddressType(spirv::TargetEnvAttr targetAttr,
                                      MLIRContext *context) {
  spirv::AddressingModel addressingModel =
      spirv::getAddressingModel(targetAttr, /*use64bitAddress=*/true);
  if (addressingModel == spirv::AddressingModel::PhysicalStorageBuffer64)
    return IntegerType::get(context, 64);

  return failure();
}

static LogicalResult getMemoryAccessAttrs(std::optional<int64_t> alignment,
                                          Builder &builder,
                                          spirv::MemoryAccessAttr &accessAttr,
                                          IntegerAttr &alignmentAttr) {
  if (!alignment)
    return success();
  if (*alignment > std::numeric_limits<uint32_t>::max())
    return failure();

  accessAttr = spirv::MemoryAccessAttr::get(builder.getContext(),
                                            spirv::MemoryAccess::Aligned);
  alignmentAttr = builder.getI32IntegerAttr(*alignment);
  return success();
}

static bool hasInvariantAttr(ptr::LoadOp op) { return op.getInvariant(); }

static bool hasInvariantAttr(ptr::StoreOp) { return false; }

template <typename OpTy>
static LogicalResult checkSupportedPtrMemoryOp(OpTy op,
                                               PatternRewriter &rewriter) {
  StringRef opName = op->getName().getStringRef();
  if (op.getVolatile_() || op.getNontemporal() || hasInvariantAttr(op) ||
      op.getInvariantGroup())
    return rewriter.notifyMatchFailure(
        op, llvm::Twine("unsupported ") + opName +
                " memory operand for SPIR-V lowering");
  if (op.getOrdering() != ptr::AtomicOrdering::not_atomic)
    return rewriter.notifyMatchFailure(op, llvm::Twine("unsupported atomic ") +
                                               opName + " for SPIR-V lowering");
  return success();
}

static FailureOr<Value>
castAddressToPointeeType(Operation *op, Value address, Type pointeeType,
                         spirv::StorageClass storageClass, Location loc,
                         PatternRewriter &rewriter) {
  if (!isa<IntegerType>(address.getType()))
    return rewriter.notifyMatchFailure(op, "expected integer address operand");
  if (storageClass != spirv::StorageClass::PhysicalStorageBuffer)
    return rewriter.notifyMatchFailure(
        op, "only PhysicalStorageBuffer pointer materialization is supported");

  auto typedPtrType = spirv::PointerType::get(pointeeType, storageClass);
  return spirv::ConvertUToPtrOp::create(rewriter, loc, typedPtrType, address)
      .getResult();
}

struct PtrTypeOffsetOpPattern final
    : public OpConversionPattern<ptr::TypeOffsetOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(ptr::TypeOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");

    llvm::TypeSize typeSize = op.getTypeSize();
    if (typeSize.isScalable())
      return rewriter.notifyMatchFailure(op, "scalable type size");

    auto intType = dyn_cast<IntegerType>(convertedType);
    if (!intType)
      return rewriter.notifyMatchFailure(
          op, "converted result type is not an integer");

    auto attr = rewriter.getIntegerAttr(intType, typeSize.getFixedValue());
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(op, intType, attr);
    return success();
  }
};

struct PtrAddOpPattern final : public OpConversionPattern<ptr::PtrAddOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(ptr::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");

    auto addressType = dyn_cast<IntegerType>(convertedType);
    if (!addressType)
      return rewriter.notifyMatchFailure(
          op, "converted result type is not an integer address");
    if (adaptor.getBase().getType() != addressType)
      return rewriter.notifyMatchFailure(
          op, "converted base pointer type does not match result type");

    Location loc = op.getLoc();
    Value offset = adaptor.getOffset();
    if (!isa<IntegerType>(offset.getType()))
      return rewriter.notifyMatchFailure(op, "offset is not an integer");
    if (offset.getType() != addressType)
      offset = spirv::UConvertOp::create(rewriter, loc, addressType, offset);

    rewriter.replaceOpWithNewOp<spirv::IAddOp>(op, addressType,
                                               adaptor.getBase(), offset);
    return success();
  }
};

struct PtrLoadOpPattern final : public OpConversionPattern<ptr::LoadOp> {
  PtrLoadOpPattern(const SPIRVTypeConverter &typeConverter,
                   MLIRContext *context, spirv::StorageClass storageClass)
      : OpConversionPattern<ptr::LoadOp>(typeConverter, context),
        storageClass(storageClass) {}

  LogicalResult
  matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkSupportedPtrMemoryOp(op, rewriter)))
      return rewriter.notifyMatchFailure(
          op, "unsupported ptr.load for SPIR-V lowering");

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");

    FailureOr<Value> ptr =
        castAddressToPointeeType(op, adaptor.getPtr(), convertedType,
                                 storageClass, op.getLoc(), rewriter);
    if (failed(ptr))
      return rewriter.notifyMatchFailure(
          op, "could not materialize SPIR-V pointer for ptr.load");

    spirv::MemoryAccessAttr accessAttr;
    IntegerAttr alignmentAttr;
    if (failed(getMemoryAccessAttrs(op.getAlignment(), rewriter, accessAttr,
                                    alignmentAttr)))
      return rewriter.notifyMatchFailure(op, "invalid alignment requirement");

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(op, convertedType, *ptr,
                                               accessAttr, alignmentAttr);
    return success();
  }

private:
  spirv::StorageClass storageClass;
};

struct PtrStoreOpPattern final : public OpConversionPattern<ptr::StoreOp> {
  PtrStoreOpPattern(const SPIRVTypeConverter &typeConverter,
                    MLIRContext *context, spirv::StorageClass storageClass)
      : OpConversionPattern<ptr::StoreOp>(typeConverter, context),
        storageClass(storageClass) {}

  LogicalResult
  matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkSupportedPtrMemoryOp(op, rewriter)))
      return rewriter.notifyMatchFailure(
          op, "unsupported ptr.store for SPIR-V lowering");

    Type valueType = adaptor.getValue().getType();
    FailureOr<Value> ptr = castAddressToPointeeType(
        op, adaptor.getPtr(), valueType, storageClass, op.getLoc(), rewriter);
    if (failed(ptr))
      return rewriter.notifyMatchFailure(
          op, "could not materialize SPIR-V pointer for ptr.store");

    spirv::MemoryAccessAttr accessAttr;
    IntegerAttr alignmentAttr;
    if (failed(getMemoryAccessAttrs(op.getAlignment(), rewriter, accessAttr,
                                    alignmentAttr)))
      return rewriter.notifyMatchFailure(op, "invalid alignment requirement");

    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, *ptr, adaptor.getValue(),
                                                accessAttr, alignmentAttr);
    return success();
  }

private:
  spirv::StorageClass storageClass;
};

static FailureOr<spirv::StorageClass> parseStorageClass(StringRef storageClass,
                                                        Operation *op) {
  std::optional<spirv::StorageClass> parsed =
      spirv::symbolizeStorageClass(storageClass);
  if (!parsed)
    return op->emitError() << "invalid SPIR-V storage class: " << storageClass;
  return *parsed;
}

struct ConvertPtrToSPIRVPass final
    : public impl::ConvertPtrToSPIRVPassBase<ConvertPtrToSPIRVPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    SPIRVTypeConverter typeConverter(targetAttr);

    FailureOr<spirv::StorageClass> storageClass =
        parseStorageClass(this->storageClass, op);
    if (failed(storageClass))
      return signalPassFailure();
    if (*storageClass != spirv::StorageClass::PhysicalStorageBuffer) {
      op->emitError()
          << "ptr-to-SPIR-V currently only supports PhysicalStorageBuffer";
      return signalPassFailure();
    }
    if (failed(getAddressType(targetAttr, op->getContext()))) {
      op->emitError()
          << "ptr-to-SPIR-V requires PhysicalStorageBuffer64 addressing";
      return signalPassFailure();
    }

    ptr::populatePtrToSPIRVTypeConversions(typeConverter);

    target->addLegalOp<UnrealizedConversionCastOp>();
    target->addIllegalDialect<ptr::PtrDialect>();

    RewritePatternSet patterns(&getContext());
    ptr::populatePtrToSPIRVPatterns(typeConverter, patterns, *storageClass);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::ptr::populatePtrToSPIRVTypeConversions(
    SPIRVTypeConverter &typeConverter) {
  spirv::TargetEnvAttr targetAttr = typeConverter.getTargetEnv().getAttr();
  typeConverter.addConversion([targetAttr](
                                  ptr::PtrType type) -> std::optional<Type> {
    FailureOr<Type> addressType = getAddressType(targetAttr, type.getContext());
    if (failed(addressType))
      return std::nullopt;
    return *addressType;
  });
}

void mlir::ptr::populatePtrToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns,
    spirv::StorageClass storageClass) {
  patterns.add<PtrAddOpPattern, PtrTypeOffsetOpPattern>(typeConverter,
                                                        patterns.getContext());
  patterns.add<PtrLoadOpPattern, PtrStoreOpPattern>(
      typeConverter, patterns.getContext(), storageClass);
}

std::unique_ptr<OperationPass<>> mlir::ptr::createConvertPtrToSPIRVPass() {
  return std::make_unique<ConvertPtrToSPIRVPass>();
}
