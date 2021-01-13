//===- GPUOpsLowering.h - GPU FuncOp / ReturnOp lowering -------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {

template <unsigned AllocaAddrSpace>
struct GPUFuncOpLowering : ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  using ConvertOpToLLVMPattern<gpu::GPUFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "func op is not expected to have operands");
    Location loc = gpuFuncOp.getLoc();

    SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
    workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
    for (auto en : llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
      Value attribution = en.value();

      auto type = attribution.getType().dyn_cast<MemRefType>();
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      uint64_t numElements = type.getNumElements();

      auto elementType = typeConverter->convertType(type.getElementType())
                             .template cast<Type>();
      auto arrayType = LLVM::LLVMArrayType::get(elementType, numElements);
      std::string name = std::string(
          llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), en.index()));
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(),
          gpu::GPUDialect::getWorkgroupAddressSpace());
      workgroupBuffers.push_back(globalOp);
    }

    // Rewrite the original GPU function to an LLVM function.
    auto funcType = typeConverter->convertType(gpuFuncOp.getType())
                        .template cast<LLVM::LLVMPointerType>()
                        .getElementType();

    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversion(
        gpuFuncOp.front().getNumArguments());
    getTypeConverter()->convertFunctionSignature(
        gpuFuncOp.getType(), /*isVariadic=*/false, signatureConversion);

    // Create the new function operation. Only copy those attributes that are
    // not specific to function modeling.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : gpuFuncOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == impl::getTypeAttrName() ||
          attr.first == gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName())
        continue;
      attributes.push_back(attr);
    }
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
        LLVM::Linkage::External, attributes);

    {
      // Insert operations that correspond to converted workgroup and private
      // memory attributions to the body of the function. This must operate on
      // the original function, before the body region is inlined in the new
      // function to maintain the relation between block arguments and the
      // parent operation that assigns their semantics.
      OpBuilder::InsertionGuard guard(rewriter);

      // Rewrite workgroup memory attributions to addresses of global buffers.
      rewriter.setInsertionPointToStart(&gpuFuncOp.front());
      unsigned numProperArguments = gpuFuncOp.getNumArguments();
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);

      Value zero = nullptr;
      if (!workgroupBuffers.empty())
        zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                                 rewriter.getI32IntegerAttr(0));
      for (auto en : llvm::enumerate(workgroupBuffers)) {
        LLVM::GlobalOp global = en.value();
        Value address = rewriter.create<LLVM::AddressOfOp>(loc, global);
        auto elementType =
            global.getType().cast<LLVM::LLVMArrayType>().getElementType();
        Value memory = rewriter.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(elementType, global.addr_space()),
            address, ArrayRef<Value>{zero, zero});

        // Build a memref descriptor pointing to the buffer to plug with the
        // existing memref infrastructure. This may use more registers than
        // otherwise necessary given that memref sizes are fixed, but we can try
        // and canonicalize that away later.
        Value attribution = gpuFuncOp.getWorkgroupAttributions()[en.index()];
        auto type = attribution.getType().cast<MemRefType>();
        auto descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, memory);
        signatureConversion.remapInput(numProperArguments + en.index(), descr);
      }

      // Rewrite private memory attributions to alloca'ed buffers.
      unsigned numWorkgroupAttributions =
          gpuFuncOp.getNumWorkgroupAttributions();
      auto int64Ty = IntegerType::get(rewriter.getContext(), 64);
      for (auto en : llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
        Value attribution = en.value();
        auto type = attribution.getType().cast<MemRefType>();
        assert(type && type.hasStaticShape() &&
               "unexpected type in attribution");

        // Explicitly drop memory space when lowering private memory
        // attributions since NVVM models it as `alloca`s in the default
        // memory space and does not support `alloca`s with addrspace(5).
        auto ptrType = LLVM::LLVMPointerType::get(
            typeConverter->convertType(type.getElementType())
                .template cast<Type>(),
            AllocaAddrSpace);
        Value numElements = rewriter.create<LLVM::ConstantOp>(
            gpuFuncOp.getLoc(), int64Ty,
            rewriter.getI64IntegerAttr(type.getNumElements()));
        Value allocated = rewriter.create<LLVM::AllocaOp>(
            gpuFuncOp.getLoc(), ptrType, numElements, /*alignment=*/0);
        auto descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, allocated);
        signatureConversion.remapInput(
            numProperArguments + numWorkgroupAttributions + en.index(), descr);
      }
    }

    // Move the region to the new function, update the entry block signature.
    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                                llvmFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &llvmFuncOp.getBody(), *typeConverter, &signatureConversion)))
      return failure();

    rewriter.eraseOp(gpuFuncOp);
    return success();
  }
};

struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_
