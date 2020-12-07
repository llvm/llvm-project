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

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

static constexpr const char *kGpuBinaryStorageSuffix = "_gpubin_cst";

namespace {

class GpuToLLVMConversionPass
    : public GpuToLLVMConversionPassBase<GpuToLLVMConversionPass> {
public:
  GpuToLLVMConversionPass(StringRef gpuBinaryAnnotation) {
    if (!gpuBinaryAnnotation.empty())
      this->gpuBinaryAnnotation = gpuBinaryAnnotation.str();
  }

  // Run the dialect converter on the module.
  void runOnOperation() override;
};

class FunctionCallBuilder {
public:
  FunctionCallBuilder(StringRef functionName, LLVM::LLVMType returnType,
                      ArrayRef<LLVM::LLVMType> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMType::getFunctionTy(returnType, argumentTypes,
                                                   /*isVarArg=*/false)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments) const;

private:
  StringRef functionName;
  LLVM::LLVMType functionType;
};

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern : public ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter) {}

protected:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  LLVM::LLVMType llvmVoidType = LLVM::LLVMType::getVoidTy(context);
  LLVM::LLVMType llvmPointerType = LLVM::LLVMType::getInt8PtrTy(context);
  LLVM::LLVMType llvmPointerPointerType = llvmPointerType.getPointerTo();
  LLVM::LLVMType llvmInt8Type = LLVM::LLVMType::getInt8Ty(context);
  LLVM::LLVMType llvmInt32Type = LLVM::LLVMType::getInt32Ty(context);
  LLVM::LLVMType llvmInt64Type = LLVM::LLVMType::getInt64Ty(context);
  LLVM::LLVMType llvmIntPtrType = LLVM::LLVMType::getIntNTy(
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
  FunctionCallBuilder allocCallBuilder = {
      "mgpuMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder deallocCallBuilder = {
      "mgpuMemFree",
      llvmVoidType,
      {llvmPointerType /* void *ptr */, llvmPointerType /* void *stream */}};
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
  matchAndRewrite(gpu::HostRegisterOp hostRegisterOp, ArrayRef<Value> operands,
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
  matchAndRewrite(gpu::AllocOp allocOp, ArrayRef<Value> operands,
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
  matchAndRewrite(gpu::DeallocOp deallocOp, ArrayRef<Value> operands,
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
  matchAndRewrite(gpu::WaitOp waitOp, ArrayRef<Value> operands,
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
  matchAndRewrite(gpu::WaitOp waitOp, ArrayRef<Value> operands,
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
                                             StringRef gpuBinaryAnnotation)
      : ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation) {}

private:
  Value generateParamsArray(gpu::LaunchFuncOp launchOp,
                            ArrayRef<Value> operands, OpBuilder &builder) const;
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder &builder) const;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

  llvm::SmallString<32> gpuBinaryAnnotation;
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
} // namespace

void GpuToLLVMConversionPass::runOnOperation() {
  LLVMTypeConverter converter(&getContext());
  OwningRewritePatternList patterns;
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateGpuToLLVMConversionPatterns(converter, patterns, gpuBinaryAnnotation);

  LLVMConversionTarget target(getContext());
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
    return OpBuilder(module.getBody()->getTerminator())
        .create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
  }();
  return builder.create<LLVM::CallOp>(
      loc, const_cast<LLVM::LLVMType &>(functionType).getFunctionResultType(),
      builder.getSymbolRefAttr(function), arguments);
}

// Returns whether all operands are of LLVM type.
static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return value.getType().isa<LLVM::LLVMType>();
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
    gpu::HostRegisterOp hostRegisterOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto *op = hostRegisterOp.getOperation();
  if (failed(areAllLLVMTypes(op, operands, rewriter)))
    return failure();

  Location loc = op->getLoc();

  auto memRefType = hostRegisterOp.value().getType();
  auto elementType = memRefType.cast<UnrankedMemRefType>().getElementType();
  auto elementSize = getSizeInBytes(loc, elementType, rewriter);

  auto arguments = getTypeConverter()->promoteOperands(loc, op->getOperands(),
                                                       operands, rewriter);
  arguments.push_back(elementSize);
  hostRegisterCallBuilder.create(loc, rewriter, arguments);

  rewriter.eraseOp(op);
  return success();
}

LogicalResult ConvertAllocOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::AllocOp allocOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  MemRefType memRefType = allocOp.getType();

  if (failed(areAllLLVMTypes(allocOp, operands, rewriter)) ||
      !isSupportedMemRefType(memRefType) ||
      failed(isAsyncWithOneDependency(rewriter, allocOp)))
    return failure();

  auto loc = allocOp.getLoc();

  // Get shape of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.
  SmallVector<Value, 4> shape;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  getMemRefDescriptorSizes(loc, memRefType, operands, rewriter, shape, strides,
                           sizeBytes);

  // Allocate the underlying buffer and store a pointer to it in the MemRef
  // descriptor.
  Type elementPtrType = this->getElementPtrType(memRefType);
  auto adaptor = gpu::AllocOpAdaptor(operands, allocOp->getAttrDictionary());
  auto stream = adaptor.asyncDependencies().front();
  Value allocatedPtr =
      allocCallBuilder.create(loc, rewriter, {sizeBytes, stream}).getResult(0);
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
    gpu::DeallocOp deallocOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(deallocOp, operands, rewriter)) ||
      failed(isAsyncWithOneDependency(rewriter, deallocOp)))
    return failure();

  Location loc = deallocOp.getLoc();

  auto adaptor =
      gpu::DeallocOpAdaptor(operands, deallocOp->getAttrDictionary());
  Value pointer =
      MemRefDescriptor(adaptor.memref()).allocatedPtr(rewriter, loc);
  auto casted = rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pointer);
  Value stream = adaptor.asyncDependencies().front();
  deallocCallBuilder.create(loc, rewriter, {casted, stream});

  rewriter.replaceOp(deallocOp, {stream});
  return success();
}

// Converts `gpu.wait` to runtime calls. The operands are all CUDA or ROCm
// streams (i.e. void*). The converted op synchronizes the host with every
// stream and then destroys it. That is, it assumes that the stream is not used
// afterwards. In case this isn't correct, we will get a runtime error.
// Eventually, we will have a pass that guarantees this property.
LogicalResult ConvertWaitOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::WaitOp waitOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (waitOp.asyncToken())
    return rewriter.notifyMatchFailure(waitOp, "Cannot convert async op.");

  Location loc = waitOp.getLoc();

  for (auto asyncDependency : operands)
    streamSynchronizeCallBuilder.create(loc, rewriter, {asyncDependency});
  for (auto asyncDependency : operands)
    streamDestroyCallBuilder.create(loc, rewriter, {asyncDependency});

  rewriter.eraseOp(waitOp);
  return success();
}

// Converts `gpu.wait async` to runtime calls. The result is a new stream that
// is synchronized with all operands, which are CUDA or ROCm streams (i.e.
// void*). We create and record an event after the definition of the stream
// and make the new stream wait on that event before destroying it again. This
// assumes that there is no other use between the definition and this op, and
// the plan is to have a pass that guarantees this property.
LogicalResult ConvertWaitAsyncOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::WaitOp waitOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!waitOp.asyncToken())
    return rewriter.notifyMatchFailure(waitOp, "Can only convert async op.");

  Location loc = waitOp.getLoc();

  auto insertionPoint = rewriter.saveInsertionPoint();
  SmallVector<Value, 1> events;
  for (auto pair : llvm::zip(waitOp.asyncDependencies(), operands)) {
    auto token = std::get<0>(pair);
    if (auto *defOp = token.getDefiningOp()) {
      rewriter.setInsertionPointAfter(defOp);
    } else {
      // If we can't find the defining op, we record the event at block start,
      // which is late and therefore misses parallelism, but still valid.
      rewriter.setInsertionPointToStart(waitOp->getBlock());
    }
    auto event = eventCreateCallBuilder.create(loc, rewriter, {}).getResult(0);
    auto stream = std::get<1>(pair);
    eventRecordCallBuilder.create(loc, rewriter, {event, stream});
    events.push_back(event);
  }
  rewriter.restoreInsertionPoint(insertionPoint);
  auto stream = streamCreateCallBuilder.create(loc, rewriter, {}).getResult(0);
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
    gpu::LaunchFuncOp launchOp, ArrayRef<Value> operands,
    OpBuilder &builder) const {
  auto loc = launchOp.getLoc();
  auto numKernelOperands = launchOp.getNumKernelOperands();
  auto arguments = getTypeConverter()->promoteOperands(
      loc, launchOp.getOperands().take_back(numKernelOperands),
      operands.take_back(numKernelOperands), builder);
  auto numArguments = arguments.size();
  SmallVector<LLVM::LLVMType, 4> argumentTypes;
  argumentTypes.reserve(numArguments);
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType().cast<LLVM::LLVMType>());
  auto structType = LLVM::LLVMType::createStructTy(argumentTypes, StringRef());
  auto one = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                              builder.getI32IntegerAttr(1));
  auto structPtr = builder.create<LLVM::AllocaOp>(
      loc, structType.getPointerTo(), one, /*alignment=*/0);
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, llvmInt32Type, builder.getI32IntegerAttr(numArguments));
  auto arrayPtr = builder.create<LLVM::AllocaOp>(loc, llvmPointerPointerType,
                                                 arraySize, /*alignment=*/0);
  auto zero = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                               builder.getI32IntegerAttr(0));
  for (auto en : llvm::enumerate(arguments)) {
    auto index = builder.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, builder.getI32IntegerAttr(en.index()));
    auto fieldPtr = builder.create<LLVM::GEPOp>(
        loc, argumentTypes[en.index()].getPointerTo(), structPtr,
        ArrayRef<Value>{zero, index.getResult()});
    builder.create<LLVM::StoreOp>(loc, en.value(), fieldPtr);
    auto elementPtr = builder.create<LLVM::GEPOp>(loc, llvmPointerPointerType,
                                                  arrayPtr, index.getResult());
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, llvmPointerType, fieldPtr);
    builder.create<LLVM::StoreOp>(loc, casted, elementPtr);
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
      LLVM::Linkage::Internal);
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
    gpu::LaunchFuncOp launchOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(launchOp, operands, rewriter)))
    return failure();

  if (launchOp.asyncDependencies().size() > 1)
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert with more than one async dependency.");

  // Fail when the synchronous version of the op has async dependencies. The
  // lowering destroys the stream, and we do not want to check that there is no
  // use of the stream after this op.
  if (!launchOp.asyncToken() && !launchOp.asyncDependencies().empty())
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert non-async op with async dependencies.");

  Location loc = launchOp.getLoc();

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launchOp, launchOp.getKernelModuleName());
  assert(kernelModule && "expected a kernel module");

  auto binaryAttr = kernelModule.getAttrOfType<StringAttr>(gpuBinaryAnnotation);
  if (!binaryAttr) {
    kernelModule.emitOpError()
        << "missing " << gpuBinaryAnnotation << " attribute";
    return failure();
  }

  SmallString<128> nameBuffer(kernelModule.getName());
  nameBuffer.append(kGpuBinaryStorageSuffix);
  Value data =
      LLVM::createGlobalString(loc, rewriter, nameBuffer.str(),
                               binaryAttr.getValue(), LLVM::Linkage::Internal);

  auto module = moduleLoadCallBuilder.create(loc, rewriter, data);
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto kernelName = generateKernelNameConstant(
      launchOp.getKernelModuleName(), launchOp.getKernelName(), loc, rewriter);
  auto function = moduleGetFunctionCallBuilder.create(
      loc, rewriter, {module.getResult(0), kernelName});
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                rewriter.getI32IntegerAttr(0));
  auto adaptor =
      gpu::LaunchFuncOpAdaptor(operands, launchOp->getAttrDictionary());
  Value stream =
      adaptor.asyncDependencies().empty()
          ? streamCreateCallBuilder.create(loc, rewriter, {}).getResult(0)
          : adaptor.asyncDependencies().front();
  // Create array of pointers to kernel arguments.
  auto kernelParams = generateParamsArray(launchOp, operands, rewriter);
  auto nullpointer = rewriter.create<LLVM::NullOp>(loc, llvmPointerPointerType);
  launchKernelCallBuilder.create(loc, rewriter,
                                 {function.getResult(0), launchOp.gridSizeX(),
                                  launchOp.gridSizeY(), launchOp.gridSizeZ(),
                                  launchOp.blockSizeX(), launchOp.blockSizeY(),
                                  launchOp.blockSizeZ(),
                                  /*sharedMemBytes=*/zero, stream, kernelParams,
                                  /*extra=*/nullpointer});

  if (launchOp.asyncToken()) {
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
  moduleUnloadCallBuilder.create(loc, rewriter, module.getResult(0));

  return success();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createGpuToLLVMConversionPass(StringRef gpuBinaryAnnotation) {
  return std::make_unique<GpuToLLVMConversionPass>(gpuBinaryAnnotation);
}

void mlir::populateGpuToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    StringRef gpuBinaryAnnotation) {
  converter.addConversion(
      [context = &converter.getContext()](gpu::AsyncTokenType type) -> Type {
        return LLVM::LLVMType::getInt8PtrTy(context);
      });
  patterns.insert<ConvertAllocOpToGpuRuntimeCallPattern,
                  ConvertDeallocOpToGpuRuntimeCallPattern,
                  ConvertHostRegisterOpToGpuRuntimeCallPattern,
                  ConvertWaitAsyncOpToGpuRuntimeCallPattern,
                  ConvertWaitOpToGpuRuntimeCallPattern>(converter);
  patterns.insert<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, gpuBinaryAnnotation);
  patterns.insert<EraseGpuModuleOpPattern>(&converter.getContext());
}
