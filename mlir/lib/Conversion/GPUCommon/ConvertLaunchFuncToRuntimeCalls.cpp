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

class GpuLaunchFuncToGpuRuntimeCallsPass
    : public ConvertGpuLaunchFuncToGpuRuntimeCallsBase<
          GpuLaunchFuncToGpuRuntimeCallsPass> {
public:
  GpuLaunchFuncToGpuRuntimeCallsPass(StringRef gpuBinaryAnnotation) {
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
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "mgpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
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
  void addParamToArray(OpBuilder &builder, Location loc, Value param,
                       Value array, unsigned pos, Value one) const;
  Value generateParamsArray(gpu::LaunchFuncOp launchOp, unsigned numArguments,
                            OpBuilder &builder) const;
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

void GpuLaunchFuncToGpuRuntimeCallsPass::runOnOperation() {
  LLVMTypeConverter converter(&getContext());
  OwningRewritePatternList patterns;
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

/// Emits the IR with the following structure:
///
///   %data = llvm.alloca 1 x type-of(<param>)
///   llvm.store <param>, %data
///   %typeErased = llvm.bitcast %data to !llvm<"i8*">
///   %addr = llvm.getelementptr <array>[<pos>]
///   llvm.store %typeErased, %addr
///
/// This is necessary to construct the array of arguments passed to the kernel
/// function as accepted by cuLaunchKernel, i.e. as a void** that points to
/// array of stack-allocated type-erased pointers to the actual arguments.
void ConvertLaunchFuncOpToGpuRuntimeCallPattern::addParamToArray(
    OpBuilder &builder, Location loc, Value param, Value array, unsigned pos,
    Value one) const {
  auto memLocation = builder.create<LLVM::AllocaOp>(
      loc, param.getType().cast<LLVM::LLVMType>().getPointerTo(), one,
      /*alignment=*/0);
  builder.create<LLVM::StoreOp>(loc, param, memLocation);
  auto casted =
      builder.create<LLVM::BitcastOp>(loc, llvmPointerType, memLocation);

  auto index = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                builder.getI32IntegerAttr(pos));
  auto gep = builder.create<LLVM::GEPOp>(loc, llvmPointerPointerType, array,
                                         index.getResult());
  builder.create<LLVM::StoreOp>(loc, casted, gep);
}

// Generates a parameters array to be used with a CUDA / ROCm (HIP) kernel
// launch call. The arguments are extracted from the launchOp.
// The generated code is essentially as follows:
//
// %array = alloca(numparams * sizeof(void *))
// for (i : [0, NumKernelOperands))
//   %array[i] = cast<void*>(KernelOperand[i])
// return %array
Value ConvertLaunchFuncOpToGpuRuntimeCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launchOp, unsigned numArguments,
    OpBuilder &builder) const {
  auto numKernelOperands = launchOp.getNumKernelOperands();
  Location loc = launchOp.getLoc();
  auto one = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                              builder.getI32IntegerAttr(1));
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, llvmInt32Type, builder.getI32IntegerAttr(numArguments));
  auto array = builder.create<LLVM::AllocaOp>(loc, llvmPointerPointerType,
                                              arraySize, /*alignment=*/0);

  unsigned pos = 0;
  for (unsigned idx = 0; idx < numKernelOperands; ++idx) {
    auto operand = launchOp.getKernelOperand(idx);
    auto llvmType = operand.getType().cast<LLVM::LLVMType>();

    // Assume all struct arguments come from MemRef. If this assumption does not
    // hold anymore then we `launchOp` to lower from MemRefType and not after
    // LLVMConversion has taken place and the MemRef information is lost.
    if (!llvmType.isStructTy()) {
      addParamToArray(builder, loc, operand, array, pos++, one);
      continue;
    }

    // Put individual components of a memref descriptor into the flat argument
    // list. We cannot use unpackMemref from LLVM lowering here because we have
    // no access to MemRefType that had been lowered away.
    for (int32_t j = 0, ej = llvmType.getStructNumElements(); j < ej; ++j) {
      auto elemType = llvmType.getStructElementType(j);
      if (elemType.isArrayTy()) {
        for (int32_t k = 0, ek = elemType.getArrayNumElements(); k < ek; ++k) {
          Value elem = builder.create<LLVM::ExtractValueOp>(
              loc, elemType.getArrayElementType(), operand,
              builder.getI32ArrayAttr({j, k}));
          addParamToArray(builder, loc, elem, array, pos++, one);
        }
      } else {
        assert((elemType.isIntegerTy() || elemType.isFloatTy() ||
                elemType.isDoubleTy() || elemType.isPointerTy()) &&
               "expected scalar type");
        Value strct = builder.create<LLVM::ExtractValueOp>(
            loc, elemType, operand, builder.getI32ArrayAttr(j));
        addParamToArray(builder, loc, strct, array, pos++, one);
      }
    }
  }

  return array;
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
  Location loc = op->getLoc();
  auto launchOp = cast<gpu::LaunchFuncOp>(op);
  auto moduleOp = op->getParentOfType<ModuleOp>();

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule =
      moduleOp.lookupSymbol<gpu::GPUModuleOp>(launchOp.getKernelModuleName());
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
  // Grab the global stream needed for execution.
  auto stream = streamCreateCallBuilder.create(loc, rewriter, {});

  // Get the launch target.
  auto gpuFuncOp = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      launchOp, launchOp.kernel());
  if (!gpuFuncOp) {
    launchOp.emitOpError() << "corresponding kernel function not found";
    return failure();
  }
  // Build array of kernel parameters.
  auto kernelParams =
      generateParamsArray(launchOp, gpuFuncOp.getNumArguments(), rewriter);

  // Invoke the function with required arguments.
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                rewriter.getI32IntegerAttr(0));
  auto nullpointer =
      rewriter.create<LLVM::IntToPtrOp>(loc, llvmPointerPointerType, zero);
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
mlir::createConvertGpuLaunchFuncToGpuRuntimeCallsPass(
    StringRef gpuBinaryAnnotation) {
  return std::make_unique<GpuLaunchFuncToGpuRuntimeCallsPass>(
      gpuBinaryAnnotation);
}

void mlir::populateGpuToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    StringRef gpuBinaryAnnotation) {
  patterns.insert<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, gpuBinaryAnnotation);
  patterns.insert<EraseGpuModuleOpPattern>(&converter.getContext());
}
