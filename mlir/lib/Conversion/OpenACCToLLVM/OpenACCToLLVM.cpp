//===- OpenACCToLLVM.cpp - Prepare OpenACC data for LLVM translation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/OpenACCToLLVM/ConvertOpenACCToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DataDescriptor implementation
//===----------------------------------------------------------------------===//

constexpr StringRef getStructName() { return "openacc_data"; }

/// Construct a helper for the given descriptor value.
DataDescriptor::DataDescriptor(Value descriptor) : StructBuilder(descriptor) {
  assert(value != nullptr && "value cannot be null");
}

/// Builds IR creating an `undef` value of the data descriptor.
DataDescriptor DataDescriptor::undef(OpBuilder &builder, Location loc,
                                     Type basePtrTy, Type ptrTy) {
  Type descriptorType = LLVM::LLVMStructType::getNewIdentified(
      builder.getContext(), getStructName(),
      {basePtrTy, ptrTy, builder.getI64Type()});
  Value descriptor = builder.create<LLVM::UndefOp>(loc, descriptorType);
  return DataDescriptor(descriptor);
}

/// Check whether the type is a valid data descriptor.
bool DataDescriptor::isValid(Value descriptor) {
  if (auto type = descriptor.getType().dyn_cast<LLVM::LLVMStructType>()) {
    if (type.isIdentified() && type.getName().startswith(getStructName()) &&
        type.getBody().size() == 3 &&
        (type.getBody()[kPtrBasePosInDataDescriptor]
             .isa<LLVM::LLVMPointerType>() ||
         type.getBody()[kPtrBasePosInDataDescriptor]
             .isa<LLVM::LLVMStructType>()) &&
        type.getBody()[kPtrPosInDataDescriptor].isa<LLVM::LLVMPointerType>() &&
        type.getBody()[kSizePosInDataDescriptor].isInteger(64))
      return true;
  }
  return false;
}

/// Builds IR inserting the base pointer value into the descriptor.
void DataDescriptor::setBasePointer(OpBuilder &builder, Location loc,
                                    Value basePtr) {
  setPtr(builder, loc, kPtrBasePosInDataDescriptor, basePtr);
}

/// Builds IR inserting the pointer value into the descriptor.
void DataDescriptor::setPointer(OpBuilder &builder, Location loc, Value ptr) {
  setPtr(builder, loc, kPtrPosInDataDescriptor, ptr);
}

/// Builds IR inserting the size value into the descriptor.
void DataDescriptor::setSize(OpBuilder &builder, Location loc, Value size) {
  setPtr(builder, loc, kSizePosInDataDescriptor, size);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

template <typename Op>
class LegalizeDataOpForLLVMTranslation : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &builder) const override {
    Location loc = op.getLoc();
    TypeConverter *converter = ConvertToLLVMPattern::getTypeConverter();

    unsigned numDataOperand = op.getNumDataOperands();

    // Keep the non data operands without modification.
    auto nonDataOperands =
        operands.take_front(operands.size() - numDataOperand);
    SmallVector<Value> convertedOperands;
    convertedOperands.append(nonDataOperands.begin(), nonDataOperands.end());

    // Go over the data operand and legalize them for translation.
    for (unsigned idx = 0; idx < numDataOperand; ++idx) {
      Value originalDataOperand = op.getDataOperand(idx);

      // Traverse operands that were converted to MemRefDescriptors.
      if (auto memRefType =
              originalDataOperand.getType().dyn_cast<MemRefType>()) {
        Type structType = converter->convertType(memRefType);
        Value memRefDescriptor = builder
                                     .create<LLVM::DialectCastOp>(
                                         loc, structType, originalDataOperand)
                                     .getResult();

        // Calculate the size of the memref and get the pointer to the allocated
        // buffer.
        SmallVector<Value> sizes;
        SmallVector<Value> strides;
        Value sizeBytes;
        ConvertToLLVMPattern::getMemRefDescriptorSizes(
            loc, memRefType, {}, builder, sizes, strides, sizeBytes);
        MemRefDescriptor descriptor(memRefDescriptor);
        Value dataPtr = descriptor.alignedPtr(builder, loc);
        auto ptrType = descriptor.getElementPtrType();

        auto descr = DataDescriptor::undef(builder, loc, structType, ptrType);
        descr.setBasePointer(builder, loc, memRefDescriptor);
        descr.setPointer(builder, loc, dataPtr);
        descr.setSize(builder, loc, sizeBytes);
        convertedOperands.push_back(descr);
      } else if (originalDataOperand.getType().isa<LLVM::LLVMPointerType>()) {
        convertedOperands.push_back(originalDataOperand);
      } else {
        // Type not supported.
        return builder.notifyMatchFailure(op, "unsupported type");
      }
    }

    builder.replaceOpWithNewOp<Op>(op, TypeRange(), convertedOperands,
                                   op.getOperation()->getAttrs());

    return success();
  }
};
} // namespace

void mlir::populateOpenACCToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::DataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::EnterDataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::ExitDataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::ParallelOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::UpdateOp>>(converter);
}

namespace {
struct ConvertOpenACCToLLVMPass
    : public ConvertOpenACCToLLVMBase<ConvertOpenACCToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToLLVMPass::runOnOperation() {
  auto op = getOperation();
  auto *context = op.getContext();

  // Convert to OpenACC operations with LLVM IR dialect
  RewritePatternSet patterns(context);
  LLVMTypeConverter converter(context);
  populateOpenACCToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();

  auto allDataOperandsAreConverted = [](ValueRange operands) {
    for (Value operand : operands) {
      if (!DataDescriptor::isValid(operand) &&
          !operand.getType().isa<LLVM::LLVMPointerType>())
        return false;
    }
    return true;
  };

  target.addDynamicallyLegalOp<acc::DataOp>(
      [allDataOperandsAreConverted](acc::DataOp op) {
        return allDataOperandsAreConverted(op.copyOperands()) &&
               allDataOperandsAreConverted(op.copyinOperands()) &&
               allDataOperandsAreConverted(op.copyinReadonlyOperands()) &&
               allDataOperandsAreConverted(op.copyoutOperands()) &&
               allDataOperandsAreConverted(op.copyoutZeroOperands()) &&
               allDataOperandsAreConverted(op.createOperands()) &&
               allDataOperandsAreConverted(op.createZeroOperands()) &&
               allDataOperandsAreConverted(op.noCreateOperands()) &&
               allDataOperandsAreConverted(op.presentOperands()) &&
               allDataOperandsAreConverted(op.deviceptrOperands()) &&
               allDataOperandsAreConverted(op.attachOperands());
      });

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [allDataOperandsAreConverted](acc::EnterDataOp op) {
        return allDataOperandsAreConverted(op.copyinOperands()) &&
               allDataOperandsAreConverted(op.createOperands()) &&
               allDataOperandsAreConverted(op.createZeroOperands()) &&
               allDataOperandsAreConverted(op.attachOperands());
      });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [allDataOperandsAreConverted](acc::ExitDataOp op) {
        return allDataOperandsAreConverted(op.copyoutOperands()) &&
               allDataOperandsAreConverted(op.deleteOperands()) &&
               allDataOperandsAreConverted(op.detachOperands());
      });

  target.addDynamicallyLegalOp<acc::ParallelOp>(
      [allDataOperandsAreConverted](acc::ParallelOp op) {
        return allDataOperandsAreConverted(op.reductionOperands()) &&
               allDataOperandsAreConverted(op.copyOperands()) &&
               allDataOperandsAreConverted(op.copyinOperands()) &&
               allDataOperandsAreConverted(op.copyinReadonlyOperands()) &&
               allDataOperandsAreConverted(op.copyoutOperands()) &&
               allDataOperandsAreConverted(op.copyoutZeroOperands()) &&
               allDataOperandsAreConverted(op.createOperands()) &&
               allDataOperandsAreConverted(op.createZeroOperands()) &&
               allDataOperandsAreConverted(op.noCreateOperands()) &&
               allDataOperandsAreConverted(op.presentOperands()) &&
               allDataOperandsAreConverted(op.devicePtrOperands()) &&
               allDataOperandsAreConverted(op.attachOperands()) &&
               allDataOperandsAreConverted(op.gangPrivateOperands()) &&
               allDataOperandsAreConverted(op.gangFirstPrivateOperands());
      });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [allDataOperandsAreConverted](acc::UpdateOp op) {
        return allDataOperandsAreConverted(op.hostOperands()) &&
               allDataOperandsAreConverted(op.deviceOperands());
      });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertOpenACCToLLVMPass() {
  return std::make_unique<ConvertOpenACCToLLVMPass>();
}
