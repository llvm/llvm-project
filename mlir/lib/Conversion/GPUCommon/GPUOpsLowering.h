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

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Find or create an external function declaration in the given module.
LLVM::LLVMFuncOp getOrDefineFunction(gpu::GPUModuleOp moduleOp, Location loc,
                                     OpBuilder &b, StringRef name,
                                     LLVM::LLVMFunctionType type);

/// Create a global that contains the given string. If a global with the same
/// string already exists in the module, return that global.
LLVM::GlobalOp getOrCreateStringConstant(OpBuilder &b, Location loc,
                                         gpu::GPUModuleOp moduleOp, Type llvmI8,
                                         StringRef namePrefix, StringRef str,
                                         uint64_t alignment = 0,
                                         unsigned addrSpace = 0);

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lowering for gpu.dynamic.shared.memory to LLVM dialect. The pattern first
/// create a 0-sized global array symbol similar as LLVM expects. It constructs
/// a memref descriptor with these values and return it.
struct GPUDynamicSharedMemoryOpLowering
    : public ConvertOpToLLVMPattern<gpu::DynamicSharedMemoryOp> {
  using ConvertOpToLLVMPattern<
      gpu::DynamicSharedMemoryOp>::ConvertOpToLLVMPattern;
  GPUDynamicSharedMemoryOpLowering(const LLVMTypeConverter &converter,
                                   unsigned alignmentBit = 0,
                                   PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<gpu::DynamicSharedMemoryOp>(converter, benefit),
        alignmentBit(alignmentBit) {}

  LogicalResult
  matchAndRewrite(gpu::DynamicSharedMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Alignment bit
  unsigned alignmentBit;
};

struct GPUFuncOpLoweringOptions {
  /// The address space to use for `alloca`s in private memory.
  unsigned allocaAddrSpace;
  /// The address space to use declaring workgroup memory.
  unsigned workgroupAddrSpace;

  /// The attribute name to use instead of `gpu.kernel`. Null if no attribute
  /// should be used.
  StringAttr kernelAttributeName;
  /// The attribute name to to set block size. Null if no attribute should be
  /// used.
  StringAttr kernelBlockSizeAttributeName;

  /// The calling convention to use for kernel functions.
  LLVM::CConv kernelCallingConvention = LLVM::CConv::C;
  /// The calling convention to use for non-kernel functions.
  LLVM::CConv nonKernelCallingConvention = LLVM::CConv::C;

  /// Whether to encode workgroup attributions as additional arguments instead
  /// of a global variable.
  bool encodeWorkgroupAttributionsAsArguments = false;
};

struct GPUFuncOpLowering : ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  GPUFuncOpLowering(const LLVMTypeConverter &converter,
                    const GPUFuncOpLoweringOptions &options,
                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<gpu::GPUFuncOp>(converter, benefit),
        allocaAddrSpace(options.allocaAddrSpace),
        workgroupAddrSpace(options.workgroupAddrSpace),
        kernelAttributeName(options.kernelAttributeName),
        kernelBlockSizeAttributeName(options.kernelBlockSizeAttributeName),
        kernelCallingConvention(options.kernelCallingConvention),
        nonKernelCallingConvention(options.nonKernelCallingConvention),
        encodeWorkgroupAttributionsAsArguments(
            options.encodeWorkgroupAttributionsAsArguments) {}

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// The address space to use for `alloca`s in private memory.
  unsigned allocaAddrSpace;
  /// The address space to use declaring workgroup memory.
  unsigned workgroupAddrSpace;

  /// The attribute name to use instead of `gpu.kernel`. Null if no attribute
  /// should be used.
  StringAttr kernelAttributeName;
  /// The attribute name to to set block size. Null if no attribute should be
  /// used.
  StringAttr kernelBlockSizeAttributeName;

  /// The calling convention to use for kernel functions
  LLVM::CConv kernelCallingConvention;
  /// The calling convention to use for non-kernel functions
  LLVM::CConv nonKernelCallingConvention;

  /// Whether to encode workgroup attributions as additional arguments instead
  /// of a global variable.
  bool encodeWorkgroupAttributionsAsArguments;
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
                  ConversionPatternRewriter &rewriter) const override;
};

namespace impl {
/// Unrolls op to array/vector elements.
LogicalResult scalarizeVectorOp(Operation *op, ValueRange operands,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &converter);
} // namespace impl

/// Unrolls SourceOp to array/vector elements.
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
