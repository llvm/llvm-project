//===- ConvertLaunchFuncToGpuRuntimeCalls.cpp - MLIR GPU lowering passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu.launch_func op into a sequence of
// GPU runtime calls. As most of GPU runtimes does not have a stable published
// ABI, this pass uses a slim runtime layer that builds on top of the public
// API from GPU runtime headers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_GPUTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr const char *kGpuBinaryStorageSuffix = "_gpubin_cst";

namespace {

class GpuToLLVMConversionPass
    : public impl::GpuToLLVMConversionPassBase<GpuToLLVMConversionPass> {
public:
  using Base::Base;

  // Run the dialect converter on the module.
  void runOnOperation() override;
};

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern : public ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter) {}

protected:
  Value getNumElements(ConversionPatternRewriter &rewriter, Location loc,
                       MemRefType type, MemRefDescriptor desc) const {
    return type.hasStaticShape()
               ? ConvertToLLVMPattern::createIndexConstant(
                     rewriter, loc, type.getNumElements())
               // For identity maps (verified by caller), the number of
               // elements is stride[0] * size[0].
               : rewriter.create<LLVM::MulOp>(loc,
                                              desc.stride(rewriter, loc, 0),
                                              desc.size(rewriter, loc, 0));
  }

  MLIRContext *context = &this->getTypeConverter()->getContext();

  Type llvmVoidType = LLVM::LLVMVoidType::get(context);
  LLVM::LLVMPointerType llvmPointerType =
      this->getTypeConverter()->getPointerType(IntegerType::get(context, 8));
  Type llvmPointerPointerType =
      this->getTypeConverter()->getPointerType(llvmPointerType);
  Type llvmInt8Type = IntegerType::get(context, 8);
  Type llvmInt16Type = IntegerType::get(context, 16);
  Type llvmInt32Type = IntegerType::get(context, 32);
  Type llvmInt64Type = IntegerType::get(context, 64);
  Type llvmInt8PointerType =
      this->getTypeConverter()->getPointerType(llvmInt8Type);
  Type llvmInt64PointerType =
      this->getTypeConverter()->getPointerType(llvmInt64Type);
  Type llvmIntPtrType = IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  FunctionCallBuilder moduleLoadCallBuilder = {
      "mgpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType /* void *cubin */}};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "mgpuModuleUnload", llvmVoidType, {llvmPointerType /* void *module */}};
  FunctionCallBuilder moduleGetFunctionCallBuilder = {
      "mgpuModuleGetFunction",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "mgpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
          llvmPointerPointerType  /* void **extra */
      }};
  FunctionCallBuilder streamCreateCallBuilder = {
      "mgpuStreamCreate", llvmPointerType /* void *stream */, {}};
  FunctionCallBuilder streamDestroyCallBuilder = {
      "mgpuStreamDestroy", llvmVoidType, {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "mgpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamWaitEventCallBuilder = {
      "mgpuStreamWaitEvent",
      llvmVoidType,
      {llvmPointerType /* void *stream */, llvmPointerType /* void *event */}};
  FunctionCallBuilder eventCreateCallBuilder = {
      "mgpuEventCreate", llvmPointerType /* void *event */, {}};
  FunctionCallBuilder eventDestroyCallBuilder = {
      "mgpuEventDestroy", llvmVoidType, {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventSynchronizeCallBuilder = {
      "mgpuEventSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventRecordCallBuilder = {
      "mgpuEventRecord",
      llvmVoidType,
      {llvmPointerType /* void *event */, llvmPointerType /* void *stream */}};
  FunctionCallBuilder hostRegisterCallBuilder = {
      "mgpuMemHostRegisterMemRef",
      llvmVoidType,
      {llvmIntPtrType /* intptr_t rank */,
       llvmPointerType /* void *memrefDesc */,
       llvmIntPtrType /* intptr_t elementSizeBytes */}};
  FunctionCallBuilder hostUnregisterCallBuilder = {
      "mgpuMemHostUnregisterMemRef",
      llvmVoidType,
      {llvmIntPtrType /* intptr_t rank */,
       llvmPointerType /* void *memrefDesc */,
       llvmIntPtrType /* intptr_t elementSizeBytes */}};
  FunctionCallBuilder allocCallBuilder = {
      "mgpuMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder deallocCallBuilder = {
      "mgpuMemFree",
      llvmVoidType,
      {llvmPointerType /* void *ptr */, llvmPointerType /* void *stream */}};
  FunctionCallBuilder memcpyCallBuilder = {
      "mgpuMemcpy",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmPointerType /* void *src */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder memset16CallBuilder = {
      "mgpuMemset16",
      llvmVoidType,
      {llvmPointerType /* void *dst */,
       llvmInt16Type /* unsigned short value */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder memset32CallBuilder = {
      "mgpuMemset32",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmInt32Type /* unsigned int value */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder setDefaultDeviceCallBuilder = {
      "mgpuSetDefaultDevice",
      llvmVoidType,
      {llvmInt32Type /* uint32_t devIndex */}};
  FunctionCallBuilder createDnVecCallBuilder = {
      "mgpuCreateDnVec",
      llvmPointerType,
      {llvmIntPtrType, llvmPointerType, llvmInt32Type,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder destroyDnVecCallBuilder = {
      "mgpuDestroyDnVec",
      llvmVoidType,
      {llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createDnMatCallBuilder = {
      "mgpuCreateDnMat",
      llvmPointerType,
      {llvmIntPtrType, llvmIntPtrType, llvmPointerType, llvmInt32Type,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder destroyDnMatCallBuilder = {
      "mgpuDestroyDnMat",
      llvmVoidType,
      {llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createCooCallBuilder = {
      "mgpuCreateCoo",
      llvmPointerType,
      {llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, llvmPointerType,
       llvmPointerType, llvmPointerType, llvmInt32Type, llvmInt32Type,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder createCooAoSCallBuilder = {
      "mgpuCreateCooAoS", // deprecated in cuSPARSE 11.2
      llvmPointerType,
      {llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmInt32Type,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder createCsrCallBuilder = {
      "mgpuCreateCsr",
      llvmPointerType,
      {llvmIntPtrType, llvmIntPtrType, llvmIntPtrType, llvmPointerType,
       llvmPointerType, llvmPointerType, llvmInt32Type, llvmInt32Type,
       llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder destroySpMatCallBuilder = {
      "mgpuDestroySpMat",
      llvmVoidType,
      {llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder spMVBufferSizeCallBuilder = {
      "mgpuSpMVBufferSize",
      llvmIntPtrType,
      {llvmInt32Type, llvmPointerType, llvmPointerType, llvmPointerType,
       llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder spMVCallBuilder = {
      "mgpuSpMV",
      llvmVoidType,
      {llvmInt32Type, llvmPointerType, llvmPointerType, llvmPointerType,
       llvmInt32Type, llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createSpMMBufferSizeCallBuilder = {
      "mgpuSpMMBufferSize",
      llvmIntPtrType,
      {llvmInt32Type, llvmInt32Type, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createSpMMCallBuilder = {
      "mgpuSpMM",
      llvmVoidType,
      {llvmInt32Type, llvmInt32Type, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmPointerType,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder createSDDMMBufferSizeCallBuilder = {
      "mgpuSDDMMBufferSize",
      llvmIntPtrType,
      {llvmInt32Type, llvmInt32Type, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createSDDMMCallBuilder = {
      "mgpuSDDMM",
      llvmVoidType,
      {llvmInt32Type, llvmInt32Type, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmPointerType,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder createLtDnMatCallBuilder = {
      "mgpuCreateCuSparseLtDnMat",
      llvmVoidType,
      {llvmPointerType, llvmIntPtrType, llvmIntPtrType, llvmPointerType,
       llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder destroyCuSparseLtSpMatBuilder = {
      "mgpuDestroyCuSparseLtSpMat",
      llvmVoidType,
      {llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder destroyCuSparseLtDnMatBuilder = {
      "mgpuDestroyCuSparseLtDnMat",
      llvmVoidType,
      {llvmPointerType, llvmPointerType /* void *stream */}};
  FunctionCallBuilder create2To4SpMatCallBuilder = {
      "mgpuCusparseLtCreate2To4SpMat",
      llvmVoidType,
      {llvmPointerType, llvmIntPtrType, llvmIntPtrType, llvmPointerType,
       llvmInt32Type, llvmPointerType /* void *stream */}};
  FunctionCallBuilder createCuSparseLtSpMMBufferSizeBuilder = {
      "mgpuCuSparseLtSpMMBufferSize",
      llvmVoidType,
      {llvmPointerType, llvmInt32Type, llvmInt32Type, llvmPointerType,
       llvmPointerType, llvmPointerType, llvmInt32Type,
       llvmPointerType /*void *stream*/}};
  FunctionCallBuilder createCuSparseLtSpMMBuilder = {
      "mgpuCuSparseLtSpMM",
      llvmVoidType,
      {llvmPointerType, llvmPointerType, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmPointerType, llvmPointerType /*void *stream*/}};
};

/// A rewrite pattern to convert gpu.host_register operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertHostRegisterOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::HostRegisterOp> {
public:
  ConvertHostRegisterOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::HostRegisterOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::HostRegisterOp hostRegisterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertHostUnregisterOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::HostUnregisterOp> {
public:
  ConvertHostUnregisterOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::HostUnregisterOp>(typeConverter) {
  }

private:
  LogicalResult
  matchAndRewrite(gpu::HostUnregisterOp hostUnregisterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.alloc operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp> {
public:
  ConvertAllocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.dealloc operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp> {
public:
  ConvertDeallocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::DeallocOp deallocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertAsyncYieldToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<async::YieldOp> {
public:
  ConvertAsyncYieldToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<async::YieldOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(async::YieldOp yieldOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.wait operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertWaitOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp> {
public:
  ConvertWaitOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.wait async operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertWaitAsyncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp> {
public:
  ConvertWaitAsyncOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite patter to convert gpu.launch_func operations into a sequence of
/// GPU runtime calls. Currently it supports CUDA and ROCm (HIP).
///
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * moduleLoad        -- loads the module given the cubin / hsaco data
/// * moduleGetFunction -- gets a handle to the actual kernel function
/// * getStreamHelper   -- initializes a new compute stream on GPU
/// * launchKernel      -- launches the kernel on a stream
/// * streamSynchronize -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class ConvertLaunchFuncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp> {
public:
  ConvertLaunchFuncOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter,
                                             StringRef gpuBinaryAnnotation,
                                             bool kernelBarePtrCallConv)
      : ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation),
        kernelBarePtrCallConv(kernelBarePtrCallConv) {}

private:
  Value generateParamsArray(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                            OpBuilder &builder) const;
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder &builder) const;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  llvm::SmallString<32> gpuBinaryAnnotation;
  bool kernelBarePtrCallConv;
};

class EraseGpuModuleOpPattern : public OpRewritePattern<gpu::GPUModuleOp> {
  using OpRewritePattern<gpu::GPUModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::GPUModuleOp op,
                                PatternRewriter &rewriter) const override {
    // GPU kernel modules are no longer necessary since we have a global
    // constant with the CUBIN, or HSACO data.
    rewriter.eraseOp(op);
    return success();
  }
};

/// A rewrite pattern to convert gpu.memcpy operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertMemcpyOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::MemcpyOp> {
public:
  ConvertMemcpyOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::MemcpyOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::MemcpyOp memcpyOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.memset operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertMemsetOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::MemsetOp> {
public:
  ConvertMemsetOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::MemsetOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::MemsetOp memsetOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.set_default_device to a GPU runtime call.
/// Currently supports CUDA and ROCm (HIP)
class ConvertSetDefaultDeviceOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SetDefaultDeviceOp> {
public:
  ConvertSetDefaultDeviceOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SetDefaultDeviceOp>(
            typeConverter) {}

  LogicalResult
  matchAndRewrite(gpu::SetDefaultDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertCreateDnTensorOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::CreateDnTensorOp> {
public:
  ConvertCreateDnTensorOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::CreateDnTensorOp>(typeConverter) {
  }

private:
  LogicalResult
  matchAndRewrite(gpu::CreateDnTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertDestroyDnTensorOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::DestroyDnTensorOp> {
public:
  ConvertDestroyDnTensorOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::DestroyDnTensorOp>(
            typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::DestroyDnTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertCreateCooOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::CreateCooOp> {
public:
  ConvertCreateCooOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::CreateCooOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::CreateCooOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertCreateCooAoSOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::CreateCooAoSOp> {
public:
  ConvertCreateCooAoSOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::CreateCooAoSOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::CreateCooAoSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertCreateCsrOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::CreateCsrOp> {
public:
  ConvertCreateCsrOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::CreateCsrOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::CreateCsrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertCreate2To4SpMatOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::Create2To4SpMatOp> {
public:
  ConvertCreate2To4SpMatOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::Create2To4SpMatOp>(
            typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::Create2To4SpMatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertDestroySpMatOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::DestroySpMatOp> {
public:
  ConvertDestroySpMatOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::DestroySpMatOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::DestroySpMatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSpMVBufferSizeOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SpMVBufferSizeOp> {
public:
  ConvertSpMVBufferSizeOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SpMVBufferSizeOp>(typeConverter) {
  }

private:
  LogicalResult
  matchAndRewrite(gpu::SpMVBufferSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSpMVOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SpMVOp> {
public:
  ConvertSpMVOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SpMVOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::SpMVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSpMMBufferSizeOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SpMMBufferSizeOp> {
public:
  ConvertSpMMBufferSizeOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SpMMBufferSizeOp>(typeConverter) {
  }

private:
  LogicalResult
  matchAndRewrite(gpu::SpMMBufferSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSDDMMBufferSizeOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SDDMMBufferSizeOp> {
public:
  ConvertSDDMMBufferSizeOpToGpuRuntimeCallPattern(
      LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SDDMMBufferSizeOp>(
            typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::SDDMMBufferSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSpMMOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SpMMOp> {
public:
  ConvertSpMMOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SpMMOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::SpMMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertSDDMMOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::SDDMMOp> {
public:
  ConvertSDDMMOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::SDDMMOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::SDDMMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

void GpuToLLVMConversionPass::runOnOperation() {
  LowerToLLVMOptions options(&getContext());
  options.useOpaquePointers = useOpaquePointers;
  options.useBarePtrCallConv = hostBarePtrCallConv;

  LLVMTypeConverter converter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  LLVMConversionTarget target(getContext());

  target.addIllegalDialect<gpu::GPUDialect>();

  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                    target);
  populateGpuToLLVMConversionPatterns(converter, patterns, gpuBinaryAnnotation,
                                      kernelBarePtrCallConv);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

LLVM::CallOp FunctionCallBuilder::create(Location loc, OpBuilder &builder,
                                         ArrayRef<Value> arguments) const {
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto function = [&] {
    if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
      return function;
    return OpBuilder::atBlockEnd(module.getBody())
        .create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
  }();
  return builder.create<LLVM::CallOp>(loc, function, arguments);
}

// Corresponding to cusparseIndexType_t defined in cusparse.h.
static int32_t getCuSparseIndexTypeFrom(Type type) {
  if (type.isa<IndexType>())
    return 3; // CUSPARSE_INDEX_64I
  else
    return 2; // CUSPARSE_INDEX_32I
  // TODO: add support to CUSPARSE_INDEX_16U: 1
}

static int32_t getCuSparseLtDataTypeFrom(Type type) {
  if (type.isF16())
    return 0; // CUSPARSE_COMPUTE_16F,
  if (type.isInteger(32))
    return 1; // CUSPARSE_COMPUTE_32I
  llvm_unreachable("unsupported type");
  // TODO: add support to TF32
}

// Corresponding to cudaDataType_t defined in CUDA library_types.h.
static int32_t getCuSparseDataTypeFrom(Type type) {
  if (llvm::isa<ComplexType>(type)) {
    // get the element type
    auto elementType = type.cast<ComplexType>().getElementType();
    if (elementType.isBF16())
      return 15; // CUDA_C_16BF
    if (elementType.isF16())
      return 6; // CUDA_C_16F
    if (elementType.isF32())
      return 4; // CUDA_C_32F
    if (elementType.isF64())
      return 5; // CUDA_C_64F
    if (elementType.isInteger(8))
      return 7; // CUDA_C_8I
    if (elementType.isInteger(16))
      return 21; // CUDA_C_16I
    if (elementType.isInteger(32))
      return 11; // CUDA_C_32I
  }
  if (type.isBF16())
    return 14; // CUDA_R_16BF
  if (type.isF16())
    return 2; // CUDA_R_16F
  if (type.isF32())
    return 0; // CUDA_R_32F
  if (type.isF64())
    return 1; // CUDA_R_64F
  if (type.isInteger(8))
    return 3; // CUDA_R_8I
  if (type.isInteger(16))
    return 20; // CUDA_R_16I
  if (type.isInteger(32))
    return 10; // CUDA_R_32I

  llvm_unreachable("unsupported element type");
}

// TODO:  We may want a run-time (of the mlir compiler) disablement/warning:
// cusparseLt currently won't work for cuda architecture <8.0 and will trigger a
// runtime (of the CUDA program) error , but it might be great if we could at
// least output a warning when we found the target architecture is <8.0 and the
// user still wants to use cusparseLt. to make sure when lowering gpu sparse
// dialect to llvm calls, the cusparselt calls are disabled for cuda
// architecture <8.0
static bool is2To4Sparsity(Value spMat) {
  if (auto op = spMat.getDefiningOp<gpu::Create2To4SpMatOp>())
    return true;
  if (auto op = spMat.getDefiningOp<gpu::CreateCooOp>())
    return false;
  if (auto op = spMat.getDefiningOp<gpu::CreateCsrOp>())
    return false;
  if (auto op = spMat.getDefiningOp<gpu::CreateCooAoSOp>())
    return false;
  // Print the spMat defining op
  spMat.getDefiningOp()->print(llvm::errs());
  llvm_unreachable("cannot find spmat def");
}

static bool isSpMMCusparseLtOp(Value op) {
  for (Operation *user : op.getUsers()) {
    auto spmmOp = dyn_cast<gpu::SpMMOp>(user);
    // If the other operator is 50% sparsity then we should use cusparseLt
    if (!spmmOp)
      continue;
    if (is2To4Sparsity(spmmOp.getSpmatA()))
      return true;
  }
  return false;
}

// Returns whether all operands are of LLVM type.
static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      }))
    return rewriter.notifyMatchFailure(
        op, "Cannot convert if operands aren't of LLVM type.");
  return success();
}

static LogicalResult
isAsyncWithOneDependency(ConversionPatternRewriter &rewriter,
                         gpu::AsyncOpInterface op) {
  if (op.getAsyncDependencies().size() != 1)
    return rewriter.notifyMatchFailure(
        op, "Can only convert with exactly one async dependency.");

  if (!op.getAsyncToken())
    return rewriter.notifyMatchFailure(op, "Can convert only async version.");

  return success();
}

LogicalResult ConvertHostRegisterOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::HostRegisterOp hostRegisterOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto *op = hostRegisterOp.getOperation();
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)))
    return failure();

  Location loc = op->getLoc();

  auto memRefType = hostRegisterOp.getValue().getType();
  auto elementType = cast<UnrankedMemRefType>(memRefType).getElementType();
  auto elementSize = getSizeInBytes(loc, elementType, rewriter);

  auto arguments = getTypeConverter()->promoteOperands(
      loc, op->getOperands(), adaptor.getOperands(), rewriter);
  arguments.push_back(elementSize);
  hostRegisterCallBuilder.create(loc, rewriter, arguments);

  rewriter.eraseOp(op);
  return success();
}

LogicalResult ConvertHostUnregisterOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::HostUnregisterOp hostUnregisterOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Operation *op = hostUnregisterOp.getOperation();
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)))
    return failure();

  Location loc = op->getLoc();

  auto memRefType = hostUnregisterOp.getValue().getType();
  auto elementType = cast<UnrankedMemRefType>(memRefType).getElementType();
  auto elementSize = getSizeInBytes(loc, elementType, rewriter);

  auto arguments = getTypeConverter()->promoteOperands(
      loc, op->getOperands(), adaptor.getOperands(), rewriter);
  arguments.push_back(elementSize);
  hostUnregisterCallBuilder.create(loc, rewriter, arguments);

  rewriter.eraseOp(op);
  return success();
}

LogicalResult ConvertAllocOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::AllocOp allocOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.getHostShared())
    return rewriter.notifyMatchFailure(
        allocOp, "host_shared allocation is not supported");

  MemRefType memRefType = allocOp.getType();

  if (failed(areAllLLVMTypes(allocOp, adaptor.getOperands(), rewriter)) ||
      !isConvertibleAndHasIdentityMaps(memRefType) ||
      failed(isAsyncWithOneDependency(rewriter, allocOp)))
    return failure();

  auto loc = allocOp.getLoc();

  // Get shape of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.
  SmallVector<Value, 4> shape;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(), rewriter,
                           shape, strides, sizeBytes);

  // Allocate the underlying buffer and store a pointer to it in the MemRef
  // descriptor.
  Type elementPtrType = this->getElementPtrType(memRefType);
  auto stream = adaptor.getAsyncDependencies().front();
  Value allocatedPtr =
      allocCallBuilder.create(loc, rewriter, {sizeBytes, stream}).getResult();
  if (!getTypeConverter()->useOpaquePointers())
    allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, allocatedPtr);

  // No alignment.
  Value alignedPtr = allocatedPtr;

  // Create the MemRef descriptor.
  auto memRefDescriptor = this->createMemRefDescriptor(
      loc, memRefType, allocatedPtr, alignedPtr, shape, strides, rewriter);

  rewriter.replaceOp(allocOp, {memRefDescriptor, stream});

  return success();
}

LogicalResult ConvertDeallocOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::DeallocOp deallocOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(deallocOp, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, deallocOp)))
    return failure();

  Location loc = deallocOp.getLoc();

  Value pointer =
      MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers())
    pointer = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pointer);
  Value stream = adaptor.getAsyncDependencies().front();
  deallocCallBuilder.create(loc, rewriter, {pointer, stream});

  rewriter.replaceOp(deallocOp, {stream});
  return success();
}

static bool isGpuAsyncTokenType(Value value) {
  return isa<gpu::AsyncTokenType>(value.getType());
}

// Converts !gpu.async.token operands of `async.yield` to runtime calls. The
// !gpu.async.token are lowered to stream within the async.execute region, but
// are passed as events between them. For each !gpu.async.token operand, we
// create an event and record it on the stream.
LogicalResult ConvertAsyncYieldToGpuRuntimeCallPattern::matchAndRewrite(
    async::YieldOp yieldOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (llvm::none_of(yieldOp.getOperands(), isGpuAsyncTokenType))
    return rewriter.notifyMatchFailure(yieldOp, "no gpu async token operand");

  Location loc = yieldOp.getLoc();
  SmallVector<Value, 4> newOperands(adaptor.getOperands());
  llvm::SmallDenseSet<Value> streams;
  for (auto &operand : yieldOp->getOpOperands()) {
    if (!isGpuAsyncTokenType(operand.get()))
      continue;
    auto idx = operand.getOperandNumber();
    auto stream = adaptor.getOperands()[idx];
    auto event = eventCreateCallBuilder.create(loc, rewriter, {}).getResult();
    eventRecordCallBuilder.create(loc, rewriter, {event, stream});
    newOperands[idx] = event;
    streams.insert(stream);
  }
  for (auto stream : streams)
    streamDestroyCallBuilder.create(loc, rewriter, {stream});

  rewriter.updateRootInPlace(yieldOp,
                             [&] { yieldOp->setOperands(newOperands); });
  return success();
}

// Returns whether `value` is the result of an LLVM::CallOp to `functionName`.
static bool isDefinedByCallTo(Value value, StringRef functionName) {
  assert(isa<LLVM::LLVMPointerType>(value.getType()));
  if (auto defOp = value.getDefiningOp<LLVM::CallOp>())
    return defOp.getCallee()->equals(functionName);
  return false;
}

// Converts `gpu.wait` to runtime calls. The converted op synchronizes the host
// with the stream/event operands. The operands are destroyed. That is, it
// assumes that it is not used afterwards or elsewhere. Otherwise we will get a
// runtime error. Eventually, we should guarantee this property.
LogicalResult ConvertWaitOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::WaitOp waitOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (waitOp.getAsyncToken())
    return rewriter.notifyMatchFailure(waitOp, "Cannot convert async op.");

  Location loc = waitOp.getLoc();

  for (auto operand : adaptor.getOperands()) {
    if (isDefinedByCallTo(operand, streamCreateCallBuilder.functionName)) {
      // The converted operand's definition created a stream.
      streamSynchronizeCallBuilder.create(loc, rewriter, {operand});
      streamDestroyCallBuilder.create(loc, rewriter, {operand});
    } else {
      // Otherwise the converted operand is an event. This assumes that we use
      // events in control flow code as well.
      eventSynchronizeCallBuilder.create(loc, rewriter, {operand});
      eventDestroyCallBuilder.create(loc, rewriter, {operand});
    }
  }

  rewriter.eraseOp(waitOp);
  return success();
}

// Converts `gpu.wait async` to runtime calls. The converted op creates a new
// stream that is synchronized with stream/event operands. The operands are
// destroyed. That is, it assumes that it is not used afterwards or elsewhere.
// Otherwise we will get a runtime error. Eventually, we should guarantee this
// property.
LogicalResult ConvertWaitAsyncOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::WaitOp waitOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!waitOp.getAsyncToken())
    return rewriter.notifyMatchFailure(waitOp, "Can only convert async op.");

  Location loc = waitOp.getLoc();

  auto insertionPoint = rewriter.saveInsertionPoint();
  SmallVector<Value, 1> events;
  for (auto pair :
       llvm::zip(waitOp.getAsyncDependencies(), adaptor.getOperands())) {
    auto operand = std::get<1>(pair);
    if (isDefinedByCallTo(operand, streamCreateCallBuilder.functionName)) {
      // The converted operand's definition created a stream. Insert an event
      // into the stream just after the last use of the original token operand.
      auto *defOp = std::get<0>(pair).getDefiningOp();
      rewriter.setInsertionPointAfter(defOp);
      auto event = eventCreateCallBuilder.create(loc, rewriter, {}).getResult();
      eventRecordCallBuilder.create(loc, rewriter, {event, operand});
      events.push_back(event);
    } else {
      // Otherwise the converted operand is an event. This assumes that we use
      // events in control flow code as well.
      events.push_back(operand);
    }
  }
  rewriter.restoreInsertionPoint(insertionPoint);
  auto stream = streamCreateCallBuilder.create(loc, rewriter, {}).getResult();
  for (auto event : events)
    streamWaitEventCallBuilder.create(loc, rewriter, {stream, event});
  for (auto event : events)
    eventDestroyCallBuilder.create(loc, rewriter, {event});
  rewriter.replaceOp(waitOp, {stream});

  return success();
}

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertLaunchFuncOpToGpuRuntimeCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launchOp, OpAdaptor adaptor, OpBuilder &builder) const {
  auto loc = launchOp.getLoc();
  auto numKernelOperands = launchOp.getNumKernelOperands();
  SmallVector<Value, 4> arguments;
  if (kernelBarePtrCallConv) {
    // Hack the bare pointer value on just for the argument promotion
    LLVMTypeConverter *converter = getTypeConverter();
    LowerToLLVMOptions options = converter->getOptions();
    LowerToLLVMOptions overrideToMatchKernelOpts = options;
    overrideToMatchKernelOpts.useBarePtrCallConv = true;
    converter->dangerousSetOptions(overrideToMatchKernelOpts);
    arguments = converter->promoteOperands(
        loc, launchOp.getOperands().take_back(numKernelOperands),
        adaptor.getOperands().take_back(numKernelOperands), builder);
    converter->dangerousSetOptions(options);
  } else {
    arguments = getTypeConverter()->promoteOperands(
        loc, launchOp.getOperands().take_back(numKernelOperands),
        adaptor.getOperands().take_back(numKernelOperands), builder);
  }

  auto numArguments = arguments.size();
  SmallVector<Type, 4> argumentTypes;
  argumentTypes.reserve(numArguments);
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType());
  auto structType = LLVM::LLVMStructType::getNewIdentified(context, StringRef(),
                                                           argumentTypes);
  auto one = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type, 1);
  auto structPtr = builder.create<LLVM::AllocaOp>(
      loc, getTypeConverter()->getPointerType(structType), structType, one,
      /*alignment=*/0);
  auto arraySize =
      builder.create<LLVM::ConstantOp>(loc, llvmInt32Type, numArguments);
  auto arrayPtr = builder.create<LLVM::AllocaOp>(
      loc, llvmPointerPointerType, llvmPointerType, arraySize, /*alignment=*/0);
  for (const auto &en : llvm::enumerate(arguments)) {
    Value fieldPtr = builder.create<LLVM::GEPOp>(
        loc, getTypeConverter()->getPointerType(argumentTypes[en.index()]),
        structType, structPtr, ArrayRef<LLVM::GEPArg>{0, en.index()});
    builder.create<LLVM::StoreOp>(loc, en.value(), fieldPtr);
    auto elementPtr = builder.create<LLVM::GEPOp>(
        loc, llvmPointerPointerType, llvmPointerType, arrayPtr,
        ArrayRef<LLVM::GEPArg>{en.index()});
    if (!getTypeConverter()->useOpaquePointers())
      fieldPtr =
          builder.create<LLVM::BitcastOp>(loc, llvmPointerType, fieldPtr);
    builder.create<LLVM::StoreOp>(loc, fieldPtr, elementPtr);
  }
  return arrayPtr;
}

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
// The code is essentially:
//
// llvm.global constant @kernel_name("function_name\00")
// func(...) {
//   %0 = llvm.addressof @kernel_name
//   %1 = llvm.constant (0 : index)
//   %2 = llvm.getelementptr %0[%1, %1] : !llvm<"i8*">
// }
Value ConvertLaunchFuncOpToGpuRuntimeCallPattern::generateKernelNameConstant(
    StringRef moduleName, StringRef name, Location loc,
    OpBuilder &builder) const {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(name.begin(), name.end());
  kernelName.push_back('\0');

  std::string globalName =
      std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, name));
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      LLVM::Linkage::Internal, getTypeConverter()->useOpaquePointers());
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute, or a
// hsaco in the 'rocdl.hsaco' attribute of the kernel function in the IR.
//
// %0 = call %binarygetter
// %1 = call %moduleLoad(%0)
// %2 = <see generateKernelNameConstant>
// %3 = call %moduleGetFunction(%1, %2)
// %4 = call %streamCreate()
// %5 = <see generateParamsArray>
// call %launchKernel(%3, <launchOp operands 0..5>, 0, %4, %5, nullptr)
// call %streamSynchronize(%4)
// call %streamDestroy(%4)
// call %moduleUnload(%1)
//
// If the op is async, the stream corresponds to the (single) async dependency
// as well as the async token the op produces.
LogicalResult ConvertLaunchFuncOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(launchOp, adaptor.getOperands(), rewriter)))
    return failure();

  if (launchOp.getAsyncDependencies().size() > 1)
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert with more than one async dependency.");

  // Fail when the synchronous version of the op has async dependencies. The
  // lowering destroys the stream, and we do not want to check that there is no
  // use of the stream after this op.
  if (!launchOp.getAsyncToken() && !launchOp.getAsyncDependencies().empty())
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert non-async op with async dependencies.");

  Location loc = launchOp.getLoc();

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launchOp, launchOp.getKernelModuleName());
  assert(kernelModule && "expected a kernel module");

  auto binaryAttr =
      kernelModule->getAttrOfType<StringAttr>(gpuBinaryAnnotation);
  if (!binaryAttr) {
    kernelModule.emitOpError()
        << "missing " << gpuBinaryAnnotation << " attribute";
    return failure();
  }

  SmallString<128> nameBuffer(kernelModule.getName());
  nameBuffer.append(kGpuBinaryStorageSuffix);
  Value data = LLVM::createGlobalString(
      loc, rewriter, nameBuffer.str(), binaryAttr.getValue(),
      LLVM::Linkage::Internal, getTypeConverter()->useOpaquePointers());

  auto module = moduleLoadCallBuilder.create(loc, rewriter, data);
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto kernelName = generateKernelNameConstant(
      launchOp.getKernelModuleName().getValue(),
      launchOp.getKernelName().getValue(), loc, rewriter);
  auto function = moduleGetFunctionCallBuilder.create(
      loc, rewriter, {module.getResult(), kernelName});
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type, 0);
  Value stream =
      adaptor.getAsyncDependencies().empty()
          ? streamCreateCallBuilder.create(loc, rewriter, {}).getResult()
          : adaptor.getAsyncDependencies().front();
  // Create array of pointers to kernel arguments.
  auto kernelParams = generateParamsArray(launchOp, adaptor, rewriter);
  auto nullpointer = rewriter.create<LLVM::NullOp>(loc, llvmPointerPointerType);
  Value dynamicSharedMemorySize = launchOp.getDynamicSharedMemorySize()
                                      ? launchOp.getDynamicSharedMemorySize()
                                      : zero;
  launchKernelCallBuilder.create(
      loc, rewriter,
      {function.getResult(), adaptor.getGridSizeX(), adaptor.getGridSizeY(),
       adaptor.getGridSizeZ(), adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
       adaptor.getBlockSizeZ(), dynamicSharedMemorySize, stream, kernelParams,
       /*extra=*/nullpointer});

  if (launchOp.getAsyncToken()) {
    // Async launch: make dependent ops use the same stream.
    rewriter.replaceOp(launchOp, {stream});
  } else {
    // Synchronize with host and destroy stream. This must be the stream created
    // above (with no other uses) because we check that the synchronous version
    // does not have any async dependencies.
    streamSynchronizeCallBuilder.create(loc, rewriter, stream);
    streamDestroyCallBuilder.create(loc, rewriter, stream);
    rewriter.eraseOp(launchOp);
  }
  moduleUnloadCallBuilder.create(loc, rewriter, module.getResult());

  return success();
}

static Value bitAndAddrspaceCast(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 LLVM::LLVMPointerType destinationType,
                                 Value sourcePtr,
                                 LLVMTypeConverter &typeConverter) {
  auto sourceTy = cast<LLVM::LLVMPointerType>(sourcePtr.getType());
  if (destinationType.getAddressSpace() != sourceTy.getAddressSpace())
    sourcePtr = rewriter.create<LLVM::AddrSpaceCastOp>(
        loc,
        typeConverter.getPointerType(sourceTy.getElementType(),
                                     destinationType.getAddressSpace()),
        sourcePtr);

  if (typeConverter.useOpaquePointers())
    return sourcePtr;

  return rewriter.create<LLVM::BitcastOp>(loc, destinationType, sourcePtr);
}

LogicalResult ConvertMemcpyOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::MemcpyOp memcpyOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto memRefType = cast<MemRefType>(memcpyOp.getSrc().getType());

  if (failed(areAllLLVMTypes(memcpyOp, adaptor.getOperands(), rewriter)) ||
      !isConvertibleAndHasIdentityMaps(memRefType) ||
      failed(isAsyncWithOneDependency(rewriter, memcpyOp)))
    return failure();

  auto loc = memcpyOp.getLoc();

  MemRefDescriptor srcDesc(adaptor.getSrc());
  Value numElements = getNumElements(rewriter, loc, memRefType, srcDesc);

  Type elementPtrType = getElementPtrType(memRefType);
  Value nullPtr = rewriter.create<LLVM::NullOp>(loc, elementPtrType);
  Value gepPtr = rewriter.create<LLVM::GEPOp>(
      loc, elementPtrType,
      typeConverter->convertType(memRefType.getElementType()), nullPtr,
      numElements);
  auto sizeBytes =
      rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

  auto src = bitAndAddrspaceCast(loc, rewriter, llvmPointerType,
                                 srcDesc.alignedPtr(rewriter, loc),
                                 *getTypeConverter());
  auto dst = bitAndAddrspaceCast(
      loc, rewriter, llvmPointerType,
      MemRefDescriptor(adaptor.getDst()).alignedPtr(rewriter, loc),
      *getTypeConverter());

  auto stream = adaptor.getAsyncDependencies().front();
  memcpyCallBuilder.create(loc, rewriter, {dst, src, sizeBytes, stream});

  rewriter.replaceOp(memcpyOp, {stream});

  return success();
}

LogicalResult ConvertMemsetOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::MemsetOp memsetOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto memRefType = cast<MemRefType>(memsetOp.getDst().getType());

  if (failed(areAllLLVMTypes(memsetOp, adaptor.getOperands(), rewriter)) ||
      !isConvertibleAndHasIdentityMaps(memRefType) ||
      failed(isAsyncWithOneDependency(rewriter, memsetOp)))
    return failure();

  auto loc = memsetOp.getLoc();

  Type valueType = adaptor.getValue().getType();
  unsigned bitWidth = valueType.getIntOrFloatBitWidth();
  // Ints and floats of 16 or 32 bit width are allowed.
  if (!valueType.isIntOrFloat() || (bitWidth != 16 && bitWidth != 32)) {
    return rewriter.notifyMatchFailure(
        memsetOp, "value must be a 16 or 32 bit int or float");
  }

  unsigned valueTypeWidth = valueType.getIntOrFloatBitWidth();
  Type bitCastType = valueTypeWidth == 32 ? llvmInt32Type : llvmInt16Type;

  MemRefDescriptor dstDesc(adaptor.getDst());
  Value numElements = getNumElements(rewriter, loc, memRefType, dstDesc);

  auto value =
      rewriter.create<LLVM::BitcastOp>(loc, bitCastType, adaptor.getValue());
  auto dst = bitAndAddrspaceCast(loc, rewriter, llvmPointerType,
                                 dstDesc.alignedPtr(rewriter, loc),
                                 *getTypeConverter());

  auto stream = adaptor.getAsyncDependencies().front();
  FunctionCallBuilder builder =
      valueTypeWidth == 32 ? memset32CallBuilder : memset16CallBuilder;
  builder.create(loc, rewriter, {dst, value, numElements, stream});

  rewriter.replaceOp(memsetOp, {stream});
  return success();
}

LogicalResult ConvertSetDefaultDeviceOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SetDefaultDeviceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  setDefaultDeviceCallBuilder.create(loc, rewriter, {adaptor.getDevIndex()});
  rewriter.replaceOp(op, {});
  return success();
}

template <typename T>
static Value genConstInt32From(OpBuilder &builder, Location loc, T TValue) {
  Type llvmInt32Type = builder.getIntegerType(32);
  return builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                          static_cast<int32_t>(TValue));
}

LogicalResult ConvertCreateDnTensorOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::CreateDnTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  Value pTensor =
      MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers())
    pTensor = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pTensor);
  Type dType = op.getMemref().getType().getElementType();
  auto dtp = genConstInt32From(rewriter, loc, getCuSparseDataTypeFrom(dType));

  SmallVector<Value, 4> dims;
  for (Value dim : adaptor.getDims()) {
    dims.push_back(dim);
  }

  Value handle;
  // TODO: For now, we track the use of the handle and lower it to cusparse /
  // cusparseLt accordingly. If in a block, both cusparse and cusparseLt are
  // used, we require two separate Creation ops to be the correct logic. In
  // future, we may add support to using one handle in sparse tensor / GPU
  // dialect in both cusparse and cusparseLt. use the cusparseLt create call if
  // the dnmat is used with spmat with 2:4 sparsity
  if (dims.size() == 2) {
    if (isSpMMCusparseLtOp(op.getDnTensor())) {
      auto handleSz = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(11032));
      handle = rewriter.create<LLVM::AllocaOp>(
          loc, llvmInt8PointerType, llvmInt8Type, handleSz, /*alignment=*/16);
      handle = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, handle);

      createLtDnMatCallBuilder
          .create(loc, rewriter,
                  {handle, dims[0], dims[1], pTensor, dtp, stream})
          .getResult();
    } else {
      handle =
          createDnMatCallBuilder
              .create(loc, rewriter, {dims[0], dims[1], pTensor, dtp, stream})
              .getResult();
    }
  } else {
    assert(dims.size() == 1 && "Only 1D and 2D tensors are supported");
    handle = createDnVecCallBuilder
                 .create(loc, rewriter, {dims[0], pTensor, dtp, stream})
                 .getResult();
  }
  rewriter.replaceOp(op, {handle, stream});
  return success();
}

LogicalResult ConvertDestroyDnTensorOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::DestroyDnTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  auto definingOp = op.getDnTensor().getDefiningOp<gpu::CreateDnTensorOp>();
  SmallVector<Value, 4> dims;
  for (Value dim : definingOp.getDims()) {
    dims.push_back(dim);
  }
  if (dims.size() == 2) {
    // Use the cusparseLt destroy call if the dnmat is used with spmat with
    // 2:4 sparsity
    if (isSpMMCusparseLtOp(op.getDnTensor())) {
      destroyCuSparseLtDnMatBuilder.create(loc, rewriter,
                                           {adaptor.getDnTensor(), stream});
    } else {
      destroyDnMatCallBuilder.create(loc, rewriter,
                                     {adaptor.getDnTensor(), stream});
    }
  } else {
    assert(dims.size() == 1 && "Only 1D and 2D tensors are supported");
    destroyDnVecCallBuilder.create(loc, rewriter,
                                   {adaptor.getDnTensor(), stream});
  }
  rewriter.replaceOp(op, {stream});
  return success();
}

LogicalResult ConvertCreateCooOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::CreateCooOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  Value pRowIdxs =
      MemRefDescriptor(adaptor.getRowIdxs()).allocatedPtr(rewriter, loc);
  Value pColIdxs =
      MemRefDescriptor(adaptor.getColIdxs()).allocatedPtr(rewriter, loc);
  Value pValues =
      MemRefDescriptor(adaptor.getValues()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers()) {
    pRowIdxs = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pRowIdxs);
    pColIdxs = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pColIdxs);
    pValues = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pValues);
  }
  Type iType =
      llvm::cast<MemRefType>(op.getColIdxs().getType()).getElementType();
  Type dType =
      llvm::cast<MemRefType>(op.getValues().getType()).getElementType();
  auto itp = genConstInt32From(rewriter, loc, getCuSparseIndexTypeFrom(iType));
  auto dtp = genConstInt32From(rewriter, loc, getCuSparseDataTypeFrom(dType));
  auto handle =
      createCooCallBuilder
          .create(loc, rewriter,
                  {adaptor.getRows(), adaptor.getCols(), adaptor.getNnz(),
                   pRowIdxs, pColIdxs, pValues, itp, dtp, stream})
          .getResult();
  rewriter.replaceOp(op, {handle, stream});
  return success();
}

LogicalResult ConvertCreateCooAoSOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::CreateCooAoSOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  Value pIdxs = MemRefDescriptor(adaptor.getIdxs()).allocatedPtr(rewriter, loc);
  Value pValues =
      MemRefDescriptor(adaptor.getValues()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers()) {
    pIdxs = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pIdxs);
    pValues = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pValues);
  }
  Type iType = llvm::cast<MemRefType>(op.getIdxs().getType()).getElementType();
  Type dType =
      llvm::cast<MemRefType>(op.getValues().getType()).getElementType();
  auto itp = genConstInt32From(rewriter, loc, getCuSparseIndexTypeFrom(iType));
  auto dtp = genConstInt32From(rewriter, loc, getCuSparseDataTypeFrom(dType));
  auto handle =
      createCooAoSCallBuilder
          .create(loc, rewriter,
                  {adaptor.getRows(), adaptor.getCols(), adaptor.getNnz(),
                   pIdxs, pValues, itp, dtp, stream})
          .getResult();
  rewriter.replaceOp(op, {handle, stream});
  return success();
}

LogicalResult ConvertCreateCsrOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::CreateCsrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  Value pRowPos =
      MemRefDescriptor(adaptor.getRowPos()).allocatedPtr(rewriter, loc);
  Value pColIdxs =
      MemRefDescriptor(adaptor.getColIdxs()).allocatedPtr(rewriter, loc);
  Value pValues =
      MemRefDescriptor(adaptor.getValues()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers()) {
    pRowPos = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pRowPos);
    pColIdxs = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pColIdxs);
    pValues = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pValues);
  }
  Type pType =
      llvm::cast<MemRefType>(op.getRowPos().getType()).getElementType();
  Type iType =
      llvm::cast<MemRefType>(op.getColIdxs().getType()).getElementType();
  Type dType =
      llvm::cast<MemRefType>(op.getValues().getType()).getElementType();
  auto ptp = genConstInt32From(rewriter, loc, getCuSparseIndexTypeFrom(pType));
  auto itp = genConstInt32From(rewriter, loc, getCuSparseIndexTypeFrom(iType));
  auto dtp = genConstInt32From(rewriter, loc, getCuSparseDataTypeFrom(dType));
  auto handle =
      createCsrCallBuilder
          .create(loc, rewriter,
                  {adaptor.getRows(), adaptor.getCols(), adaptor.getNnz(),
                   pRowPos, pColIdxs, pValues, ptp, itp, dtp, stream})
          .getResult();
  rewriter.replaceOp(op, {handle, stream});
  return success();
}

LogicalResult ConvertCreate2To4SpMatOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::Create2To4SpMatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  Value pMat =
      MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers())
    pMat = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pMat);
  Type dType =
      llvm::cast<MemRefType>(op.getMemref().getType()).getElementType();
  auto dtp = genConstInt32From(rewriter, loc, getCuSparseDataTypeFrom(dType));

  // CUDA runner asserts the size is 44104 bytes.
  auto handleSz = rewriter.create<LLVM::ConstantOp>(
      loc, getIndexType(), rewriter.getIndexAttr(44104));
  Value handle = rewriter.create<LLVM::AllocaOp>(
      loc, llvmInt8PointerType, llvmInt8Type, handleSz, /*alignment=*/16);
  handle = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, handle);

  create2To4SpMatCallBuilder
      .create(loc, rewriter,
              {handle, adaptor.getRows(), adaptor.getCols(), pMat, dtp, stream})
      .getResult();
  rewriter.replaceOp(op, {handle, stream});
  return success();
}

LogicalResult ConvertDestroySpMatOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::DestroySpMatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto stream = adaptor.getAsyncDependencies().front();
  // Use the cusparseLt destroy call if the spmat is 2:4 sparsity
  if (is2To4Sparsity(op.getSpmat())) {
    destroyCuSparseLtSpMatBuilder.create(loc, rewriter,
                                         {adaptor.getSpmat(), stream});

  } else {
    destroySpMatCallBuilder.create(loc, rewriter, {adaptor.getSpmat(), stream});
  }
  rewriter.replaceOp(op, {stream});
  return success();
}

LogicalResult ConvertSpMVBufferSizeOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SpMVBufferSizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto modeA = genConstInt32From(rewriter, loc, op.getModeA());
  auto computeType = genConstInt32From(
      rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));
  auto stream = adaptor.getAsyncDependencies().front();
  auto bufferSize = spMVBufferSizeCallBuilder
                        .create(loc, rewriter,
                                {modeA, adaptor.getSpmatA(), adaptor.getDnX(),
                                 adaptor.getDnY(), computeType, stream})
                        .getResult();
  rewriter.replaceOp(op, {bufferSize, stream});
  return success();
}

LogicalResult ConvertSpMVOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SpMVOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto modeA = genConstInt32From(rewriter, loc, adaptor.getModeA());
  auto computeType = genConstInt32From(
      rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));
  auto stream = adaptor.getAsyncDependencies().front();
  Value pBuf =
      MemRefDescriptor(adaptor.getBuffer()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers())
    pBuf = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pBuf);
  spMVCallBuilder.create(loc, rewriter,
                         {modeA, adaptor.getSpmatA(), adaptor.getDnX(),
                          adaptor.getDnY(), computeType, pBuf, stream});
  rewriter.replaceOp(op, {stream});
  return success();
}

LogicalResult ConvertSpMMBufferSizeOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SpMMBufferSizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto modeA = genConstInt32From(rewriter, loc, adaptor.getModeA());
  auto modeB = genConstInt32From(rewriter, loc, adaptor.getModeB());
  auto stream = adaptor.getAsyncDependencies().front();
  Value bufferSize;
  if (is2To4Sparsity(op.getSpmatA())) {
    auto computeType = genConstInt32From(
        rewriter, loc, getCuSparseLtDataTypeFrom(adaptor.getComputeType()));
    auto three = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                   rewriter.getIndexAttr(3));
    auto bufferSize = rewriter.create<LLVM::AllocaOp>(
        loc, llvmInt64PointerType, llvmInt64Type, three, /*alignment=*/16);
    createCuSparseLtSpMMBufferSizeBuilder
        .create(loc, rewriter,
                {bufferSize, modeA, modeB, adaptor.getSpmatA(),
                 adaptor.getDnmatB(), adaptor.getDnmatC(), computeType, stream})
        .getResult();

    auto bufferSizePtr1 = rewriter.create<LLVM::GEPOp>(
        loc, llvmInt64PointerType, llvmInt64PointerType, bufferSize,
        ValueRange{rewriter.create<LLVM::ConstantOp>(
            loc, getIndexType(), rewriter.getIndexAttr(1))});
    auto bufferSizePtr2 = rewriter.create<LLVM::GEPOp>(
        loc, llvmInt64PointerType, llvmInt64PointerType, bufferSize,
        ValueRange{rewriter.create<LLVM::ConstantOp>(
            loc, getIndexType(), rewriter.getIndexAttr(2))});
    auto bufferSize0 =
        rewriter.create<LLVM::LoadOp>(loc, llvmInt64Type, bufferSize);
    auto bufferSize1 =
        rewriter.create<LLVM::LoadOp>(loc, llvmInt64Type, bufferSizePtr1);
    auto bufferSize2 =
        rewriter.create<LLVM::LoadOp>(loc, llvmInt64Type, bufferSizePtr2);

    rewriter.replaceOp(op, {bufferSize0, bufferSize1, bufferSize2, stream});
  } else {
    auto computeType = genConstInt32From(
        rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));
    bufferSize =
        createSpMMBufferSizeCallBuilder
            .create(loc, rewriter,
                    {modeA, modeB, adaptor.getSpmatA(), adaptor.getDnmatB(),
                     adaptor.getDnmatC(), computeType, stream})
            .getResult();
    rewriter.replaceOp(op, {bufferSize, stream});
  }
  return success();
}

LogicalResult ConvertSDDMMBufferSizeOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SDDMMBufferSizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto modeA = genConstInt32From(rewriter, loc, adaptor.getModeA());
  auto modeB = genConstInt32From(rewriter, loc, adaptor.getModeB());
  auto computeType = genConstInt32From(
      rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));
  auto stream = adaptor.getAsyncDependencies().front();
  auto bufferSize =
      createSDDMMBufferSizeCallBuilder
          .create(loc, rewriter,
                  {modeA, modeB, adaptor.getDnmatA(), adaptor.getDnmatB(),
                   adaptor.getSpmatC(), computeType, stream})
          .getResult();
  rewriter.replaceOp(op, {bufferSize, stream});
  return success();
}

LogicalResult ConvertSpMMOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SpMMOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto modeA = genConstInt32From(rewriter, loc, adaptor.getModeA());
  auto modeB = genConstInt32From(rewriter, loc, adaptor.getModeB());
  auto computeType = genConstInt32From(
      rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));

  auto stream = adaptor.getAsyncDependencies().front();

  // Lower to cusparseLt if applicable
  if (is2To4Sparsity(op.getSpmatA())) {
    SmallVector<Value> pBufs;
    for (Value buffer : adaptor.getBuffers()) {
      Value pBuf = MemRefDescriptor(buffer).allocatedPtr(rewriter, loc);
      if (!getTypeConverter()->useOpaquePointers())
        pBuf = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pBuf);
      pBufs.push_back(pBuf);
    }
    createCuSparseLtSpMMBuilder.create(
        loc, rewriter,
        {adaptor.getSpmatA(), adaptor.getDnmatB(), adaptor.getDnmatC(),
         pBufs[0], pBufs[1], pBufs[2], stream});
  } else {
    Value pBuf = MemRefDescriptor(adaptor.getBuffers().front())
                     .allocatedPtr(rewriter, loc);
    if (!getTypeConverter()->useOpaquePointers())
      pBuf = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pBuf);
    createSpMMCallBuilder.create(loc, rewriter,
                                 {modeA, modeB, adaptor.getSpmatA(),
                                  adaptor.getDnmatB(), adaptor.getDnmatC(),
                                  computeType, pBuf, stream});
  }
  rewriter.replaceOp(op, {stream});
  return success();
}

template <typename T>
static void addOpaquePointerConversion(LLVMTypeConverter &converter) {
  converter.addConversion([&converter](T) -> Type {
    return converter.getPointerType(
        IntegerType::get(&converter.getContext(), 8));
  });
}

LogicalResult ConvertSDDMMOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::SDDMMOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, op)))
    return failure();
  Location loc = op.getLoc();
  auto computeType = genConstInt32From(
      rewriter, loc, getCuSparseDataTypeFrom(adaptor.getComputeType()));
  auto modeA = genConstInt32From(rewriter, loc, adaptor.getModeA());
  auto modeB = genConstInt32From(rewriter, loc, adaptor.getModeB());
  auto stream = adaptor.getAsyncDependencies().front();
  Value pBuf =
      MemRefDescriptor(adaptor.getBuffer()).allocatedPtr(rewriter, loc);
  if (!getTypeConverter()->useOpaquePointers())
    pBuf = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pBuf);
  createSDDMMCallBuilder.create(loc, rewriter,
                                {modeA, modeB, adaptor.getDnmatA(),
                                 adaptor.getDnmatB(), adaptor.getSpmatC(),
                                 computeType, pBuf, stream});
  rewriter.replaceOp(op, {stream});
  return success();
}

void mlir::populateGpuToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns,
                                               StringRef gpuBinaryAnnotation,
                                               bool kernelBarePtrCallConv) {
  addOpaquePointerConversion<gpu::AsyncTokenType>(converter);
  addOpaquePointerConversion<gpu::SparseDnTensorHandleType>(converter);
  addOpaquePointerConversion<gpu::SparseSpMatHandleType>(converter);

  patterns.add<ConvertAllocOpToGpuRuntimeCallPattern,
               ConvertDeallocOpToGpuRuntimeCallPattern,
               ConvertHostRegisterOpToGpuRuntimeCallPattern,
               ConvertHostUnregisterOpToGpuRuntimeCallPattern,
               ConvertMemcpyOpToGpuRuntimeCallPattern,
               ConvertMemsetOpToGpuRuntimeCallPattern,
               ConvertSetDefaultDeviceOpToGpuRuntimeCallPattern,
               ConvertWaitAsyncOpToGpuRuntimeCallPattern,
               ConvertWaitOpToGpuRuntimeCallPattern,
               ConvertAsyncYieldToGpuRuntimeCallPattern,
               ConvertCreateDnTensorOpToGpuRuntimeCallPattern,
               ConvertDestroyDnTensorOpToGpuRuntimeCallPattern,
               ConvertCreateCooOpToGpuRuntimeCallPattern,
               ConvertCreateCooAoSOpToGpuRuntimeCallPattern,
               ConvertCreateCsrOpToGpuRuntimeCallPattern,
               ConvertCreate2To4SpMatOpToGpuRuntimeCallPattern,
               ConvertDestroySpMatOpToGpuRuntimeCallPattern,
               ConvertSpMVBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSpMVOpToGpuRuntimeCallPattern,
               ConvertSpMMBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSpMMOpToGpuRuntimeCallPattern,
               ConvertSDDMMBufferSizeOpToGpuRuntimeCallPattern,
               ConvertSDDMMOpToGpuRuntimeCallPattern>(converter);
  patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, gpuBinaryAnnotation, kernelBarePtrCallConv);
  patterns.add<EraseGpuModuleOpPattern>(&converter.getContext());
}
