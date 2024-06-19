//===- GPUOpsLowering.h - GPU FuncOp / ReturnOp lowering -------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {

/// Lowering for gpu.dynamic.shared.memory to LLVM dialect. The pattern first
/// create a 0-sized global array symbol similar as LLVM expects. It constructs
/// a memref descriptor with these values and return it.
struct GPUDynamicSharedMemoryOpLowering
    : public ConvertOpToLLVMPattern<gpu::DynamicSharedMemoryOp> {
  using ConvertOpToLLVMPattern<
      gpu::DynamicSharedMemoryOp>::ConvertOpToLLVMPattern;
  GPUDynamicSharedMemoryOpLowering(const LLVMTypeConverter &converter,
                                   unsigned alignmentBit = 0)
      : ConvertOpToLLVMPattern<gpu::DynamicSharedMemoryOp>(converter),
        alignmentBit(alignmentBit) {}

  LogicalResult
  matchAndRewrite(gpu::DynamicSharedMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Alignment bit
  unsigned alignmentBit;
};

struct GPUFuncOpLowering : ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  GPUFuncOpLowering(
      const LLVMTypeConverter &converter, unsigned allocaAddrSpace,
      unsigned workgroupAddrSpace, StringAttr kernelAttributeName,
      std::optional<StringAttr> kernelBlockSizeAttributeName = std::nullopt)
      : ConvertOpToLLVMPattern<gpu::GPUFuncOp>(converter),
        allocaAddrSpace(allocaAddrSpace),
        workgroupAddrSpace(workgroupAddrSpace),
        kernelAttributeName(kernelAttributeName),
        kernelBlockSizeAttributeName(kernelBlockSizeAttributeName) {}

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// The address space to use for `alloca`s in private memory.
  unsigned allocaAddrSpace;
  /// The address space to use declaring workgroup memory.
  unsigned workgroupAddrSpace;

  /// The attribute name to use instead of `gpu.kernel`.
  StringAttr kernelAttributeName;

  /// The attribute name to to set block size
  std::optional<StringAttr> kernelBlockSizeAttributeName;
};

/// The lowering of gpu.printf to a call to HIP hostcalls
///
/// Simplifies llvm/lib/Transforms/Utils/AMDGPUEmitPrintf.cpp, as we don't have
/// to deal with %s (even if there were first-class strings in MLIR, they're not
/// legal input to gpu.printf) or non-constant format strings
struct GPUPrintfOpToHIPLowering : public ConvertOpToLLVMPattern<gpu::PrintfOp> {
  using ConvertOpToLLVMPattern<gpu::PrintfOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// The lowering of gpu.printf to a call to an external printf() function
///
/// This pass will add a declaration of printf() to the GPUModule if needed
/// and separate out the format strings into global constants. For some
/// runtimes, such as OpenCL on AMD, this is sufficient setup, as the compiler
/// will lower printf calls to appropriate device-side code
struct GPUPrintfOpToLLVMCallLowering
    : public ConvertOpToLLVMPattern<gpu::PrintfOp> {
  GPUPrintfOpToLLVMCallLowering(const LLVMTypeConverter &converter,
                                int addressSpace = 0)
      : ConvertOpToLLVMPattern<gpu::PrintfOp>(converter),
        addressSpace(addressSpace) {}

  LogicalResult
  matchAndRewrite(gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  int addressSpace;
};

/// Lowering of gpu.printf to a vprintf standard library.
struct GPUPrintfOpToVPrintfLowering
    : public ConvertOpToLLVMPattern<gpu::PrintfOp> {
  using ConvertOpToLLVMPattern<gpu::PrintfOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

namespace impl {
/// Unrolls op if it's operating on vectors.
LogicalResult scalarizeVectorOp(Operation *op, ValueRange operands,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &converter);
} // namespace impl

/// Rewriting that unrolls SourceOp to scalars if it's operating on vectors.
template <typename SourceOp>
struct ScalarizeVectorOpLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return impl::scalarizeVectorOp(op, adaptor.getOperands(), rewriter,
                                   *this->getTypeConverter());
  }
};
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUOPSLOWERING_H_
