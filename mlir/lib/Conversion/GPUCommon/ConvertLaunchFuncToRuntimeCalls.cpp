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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

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
  MLIRContext *context = &this->typeConverter.getContext();

  LLVM::LLVMType llvmVoidType = LLVM::LLVMType::getVoidTy(context);
  LLVM::LLVMType llvmPointerType = LLVM::LLVMType::getInt8PtrTy(context);
  LLVM::LLVMType llvmPointerPointerType = llvmPointerType.getPointerTo();
  LLVM::LLVMType llvmInt8Type = LLVM::LLVMType::getInt8Ty(context);
  LLVM::LLVMType llvmInt32Type = LLVM::LLVMType::getInt32Ty(context);
  LLVM::LLVMType llvmInt64Type = LLVM::LLVMType::getInt64Ty(context);
  LLVM::LLVMType llvmIntPtrType = LLVM::LLVMType::getIntNTy(
      context, this->typeConverter.getPointerBitwidth(0));

  FunctionCallBuilder moduleLoadCallBuilder = {
      "mgpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType /* void *cubin */}};
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
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

// Returns whether value is of LLVM type.
static bool isLLVMType(Value value) {
  return value.getType().isa<LLVM::LLVMType>();
}

LogicalResult ConvertHostRegisterOpToGpuRuntimeCallPattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!llvm::all_of(operands, isLLVMType))
    return rewriter.notifyMatchFailure(
        op, "Cannot convert if operands aren't of LLVM type.");

  Location loc = op->getLoc();

  auto memRefType = cast<gpu::HostRegisterOp>(op).value().getType();
  auto elementType = memRefType.cast<UnrankedMemRefType>().getElementType();
  auto elementSize = getSizeInBytes(loc, elementType, rewriter);

  auto arguments =
      typeConverter.promoteOperands(loc, op->getOperands(), operands, rewriter);
  arguments.push_back(elementSize);
  hostRegisterCallBuilder.create(loc, rewriter, arguments);

  rewriter.eraseOp(op);
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
  auto arguments = typeConverter.promoteOperands(
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
LogicalResult ConvertLaunchFuncOpToGpuRuntimeCallPattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!llvm::all_of(operands, isLLVMType))
    return rewriter.notifyMatchFailure(
        op, "Cannot convert if operands aren't of LLVM type.");

  auto launchOp = cast<gpu::LaunchFuncOp>(op);
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
  // Grab the global stream needed for execution.
  auto stream = streamCreateCallBuilder.create(loc, rewriter, {});
  // Create array of pointers to kernel arguments.
  auto kernelParams = generateParamsArray(launchOp, operands, rewriter);
  auto nullpointer = rewriter.create<LLVM::NullOp>(loc, llvmPointerPointerType);
  launchKernelCallBuilder.create(
      loc, rewriter,
      {function.getResult(0), launchOp.gridSizeX(), launchOp.gridSizeY(),
       launchOp.gridSizeZ(), launchOp.blockSizeX(), launchOp.blockSizeY(),
       launchOp.blockSizeZ(), zero, /* sharedMemBytes */
       stream.getResult(0),         /* stream */
       kernelParams,                /* kernel params */
       nullpointer /* extra */});
  streamSynchronizeCallBuilder.create(loc, rewriter, stream.getResult(0));

  rewriter.eraseOp(op);
  return success();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createGpuToLLVMConversionPass(StringRef gpuBinaryAnnotation) {
  return std::make_unique<GpuToLLVMConversionPass>(gpuBinaryAnnotation);
}

void mlir::populateGpuToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    StringRef gpuBinaryAnnotation) {
  patterns.insert<ConvertHostRegisterOpToGpuRuntimeCallPattern>(converter);
  patterns.insert<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, gpuBinaryAnnotation);
  patterns.insert<EraseGpuModuleOpPattern>(&converter.getContext());
}
