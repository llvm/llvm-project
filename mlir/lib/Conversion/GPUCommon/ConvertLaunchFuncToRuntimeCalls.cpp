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

// To avoid name mangling, these are defined in the mini-runtime file.
static constexpr const char *kGpuModuleLoadName = "mgpuModuleLoad";
static constexpr const char *kGpuModuleGetFunctionName =
    "mgpuModuleGetFunction";
static constexpr const char *kGpuLaunchKernelName = "mgpuLaunchKernel";
static constexpr const char *kGpuGetStreamHelperName = "mgpuGetStreamHelper";
static constexpr const char *kGpuStreamSynchronizeName =
    "mgpuStreamSynchronize";
static constexpr const char *kGpuMemHostRegisterName = "mgpuMemHostRegister";
static constexpr const char *kGpuBinaryStorageSuffix = "_gpubin_cst";

namespace {

/// A pass to convert gpu.launch_func operations into a sequence of GPU
/// runtime calls. Currently it supports CUDA and ROCm (HIP).
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
class GpuLaunchFuncToGpuRuntimeCallsPass
    : public ConvertGpuLaunchFuncToGpuRuntimeCallsBase<
          GpuLaunchFuncToGpuRuntimeCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    const llvm::Module &module = llvmDialect->getLLVMModule();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmPointerPointerType = llvmPointerType.getPointerTo();
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    llvmIntPtrType = LLVM::LLVMType::getIntNTy(
        llvmDialect, module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() { return llvmVoidType; }

  LLVM::LLVMType getPointerType() { return llvmPointerType; }

  LLVM::LLVMType getPointerPointerType() { return llvmPointerPointerType; }

  LLVM::LLVMType getInt8Type() { return llvmInt8Type; }

  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  LLVM::LLVMType getIntPtrType() {
    const llvm::Module &module = getLLVMDialect()->getLLVMModule();
    return LLVM::LLVMType::getIntNTy(
        getLLVMDialect(), module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getGpuRuntimeResultType() {
    // This is declared as an enum in both CUDA and ROCm (HIP), but helpers
    // use i32.
    return getInt32Type();
  }

  // Allocate a void pointer on the stack.
  Value allocatePointer(OpBuilder &builder, Location loc) {
    auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(1));
    return builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), one,
                                          /*alignment=*/0);
  }

  void declareGpuRuntimeFunctions(Location loc);
  void addParamToList(OpBuilder &builder, Location loc, Value param, Value list,
                      unsigned pos, Value one);
  Value setupParamsArray(gpu::LaunchFuncOp launchOp, OpBuilder &builder);
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  // Run the dialect converter on the module.
  void runOnOperation() override {
    // Cache the LLVMDialect for the current module.
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    // Cache the used LLVM types.
    initializeCachedTypes();

    getOperation().walk(
        [this](mlir::gpu::LaunchFuncOp op) { translateGpuLaunchCalls(op); });

    // GPU kernel modules are no longer necessary since we have a global
    // constant with the CUBIN, or HSACO data.
    for (auto m :
         llvm::make_early_inc_range(getOperation().getOps<gpu::GPUModuleOp>()))
      m.erase();
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmPointerPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmIntPtrType;
};

} // anonymous namespace

// Adds declarations for the needed helper functions from the runtime wrappers.
// The types in comments give the actual types expected/returned but the API
// uses void pointers. This is fine as they have the same linkage in C.
void GpuLaunchFuncToGpuRuntimeCallsPass::declareGpuRuntimeFunctions(
    Location loc) {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getBody()->getTerminator());
  if (!module.lookupSymbol(kGpuModuleLoadName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuModuleLoadName,
        LLVM::LLVMType::getFunctionTy(
            getGpuRuntimeResultType(),
            {
                getPointerPointerType(), /* CUmodule *module */
                getPointerType()         /* void *cubin */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kGpuModuleGetFunctionName)) {
    // The helper uses void* instead of CUDA's opaque CUmodule and
    // CUfunction, or ROCm (HIP)'s opaque hipModule_t and hipFunction_t.
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuModuleGetFunctionName,
        LLVM::LLVMType::getFunctionTy(
            getGpuRuntimeResultType(),
            {
                getPointerPointerType(), /* void **function */
                getPointerType(),        /* void *module */
                getPointerType()         /* char *name */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kGpuLaunchKernelName)) {
    // Other than the CUDA or ROCm (HIP) api, the wrappers use uintptr_t to
    // match the LLVM type if MLIR's index type, which the GPU dialect uses.
    // Furthermore, they use void* instead of CUDA's opaque CUfunction and
    // CUstream, or ROCm (HIP)'s opaque hipFunction_t and hipStream_t.
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuLaunchKernelName,
        LLVM::LLVMType::getFunctionTy(
            getGpuRuntimeResultType(),
            {
                getPointerType(),        /* void* f */
                getIntPtrType(),         /* intptr_t gridXDim */
                getIntPtrType(),         /* intptr_t gridyDim */
                getIntPtrType(),         /* intptr_t gridZDim */
                getIntPtrType(),         /* intptr_t blockXDim */
                getIntPtrType(),         /* intptr_t blockYDim */
                getIntPtrType(),         /* intptr_t blockZDim */
                getInt32Type(),          /* unsigned int sharedMemBytes */
                getPointerType(),        /* void *hstream */
                getPointerPointerType(), /* void **kernelParams */
                getPointerPointerType()  /* void **extra */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kGpuGetStreamHelperName)) {
    // Helper function to get the current GPU compute stream. Uses void*
    // instead of CUDA's opaque CUstream, or ROCm (HIP)'s opaque hipStream_t.
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuGetStreamHelperName,
        LLVM::LLVMType::getFunctionTy(getPointerType(), /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kGpuStreamSynchronizeName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuStreamSynchronizeName,
        LLVM::LLVMType::getFunctionTy(getGpuRuntimeResultType(),
                                      getPointerType() /* CUstream stream */,
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kGpuMemHostRegisterName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kGpuMemHostRegisterName,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {
                                          getPointerType(), /* void *ptr */
                                          getInt64Type()    /* int64 sizeBytes*/
                                      },
                                      /*isVarArg=*/false));
  }
}

/// Emits the IR with the following structure:
///
///   %data = llvm.alloca 1 x type-of(<param>)
///   llvm.store <param>, %data
///   %typeErased = llvm.bitcast %data to !llvm<"i8*">
///   %addr = llvm.getelementptr <list>[<pos>]
///   llvm.store %typeErased, %addr
///
/// This is necessary to construct the list of arguments passed to the kernel
/// function as accepted by cuLaunchKernel, i.e. as a void** that points to list
/// of stack-allocated type-erased pointers to the actual arguments.
void GpuLaunchFuncToGpuRuntimeCallsPass::addParamToList(OpBuilder &builder,
                                                        Location loc,
                                                        Value param, Value list,
                                                        unsigned pos,
                                                        Value one) {
  auto memLocation = builder.create<LLVM::AllocaOp>(
      loc, param.getType().cast<LLVM::LLVMType>().getPointerTo(), one,
      /*alignment=*/1);
  builder.create<LLVM::StoreOp>(loc, param, memLocation);
  auto casted =
      builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);

  auto index = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(pos));
  auto gep = builder.create<LLVM::GEPOp>(loc, getPointerPointerType(), list,
                                         ArrayRef<Value>{index});
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
Value GpuLaunchFuncToGpuRuntimeCallsPass::setupParamsArray(
    gpu::LaunchFuncOp launchOp, OpBuilder &builder) {

  // Get the launch target.
  auto gpuFunc = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      launchOp, launchOp.kernel());
  if (!gpuFunc)
    return {};

  unsigned numArgs = gpuFunc.getNumArguments();

  auto numKernelOperands = launchOp.getNumKernelOperands();
  Location loc = launchOp.getLoc();
  auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                              builder.getI32IntegerAttr(1));
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(numArgs));
  auto array = builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(),
                                              arraySize, /*alignment=*/0);

  unsigned pos = 0;
  for (unsigned idx = 0; idx < numKernelOperands; ++idx) {
    auto operand = launchOp.getKernelOperand(idx);
    auto llvmType = operand.getType().cast<LLVM::LLVMType>();

    // Assume all struct arguments come from MemRef. If this assumption does not
    // hold anymore then we `launchOp` to lower from MemRefType and not after
    // LLVMConversion has taken place and the MemRef information is lost.
    if (!llvmType.isStructTy()) {
      addParamToList(builder, loc, operand, array, pos++, one);
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
          addParamToList(builder, loc, elem, array, pos++, one);
        }
      } else {
        assert((elemType.isIntegerTy() || elemType.isFloatTy() ||
                elemType.isDoubleTy() || elemType.isPointerTy()) &&
               "expected scalar type");
        Value strct = builder.create<LLVM::ExtractValueOp>(
            loc, elemType, operand, builder.getI32ArrayAttr(j));
        addParamToList(builder, loc, strct, array, pos++, one);
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
Value GpuLaunchFuncToGpuRuntimeCallsPass::generateKernelNameConstant(
    StringRef moduleName, StringRef name, Location loc, OpBuilder &builder) {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(name.begin(), name.end());
  kernelName.push_back('\0');

  std::string globalName =
      std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, name));
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      LLVM::Linkage::Internal, llvmDialect);
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute, or a
// hsaco in the 'rocdl.hsaco' attribute of the kernel function in the IR.
//
// %0 = call %binarygetter
// %1 = alloca sizeof(void*)
// call %moduleLoad(%2, %1)
// %2 = alloca sizeof(void*)
// %3 = load %1
// %4 = <see generateKernelNameConstant>
// call %moduleGetFunction(%2, %3, %4)
// %5 = call %getStreamHelper()
// %6 = load %2
// %7 = <see setupParamsArray>
// call %launchKernel(%6, <launchOp operands 0..5>, 0, %5, %7, nullptr)
// call %streamSynchronize(%5)
void GpuLaunchFuncToGpuRuntimeCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();
  declareGpuRuntimeFunctions(loc);

  auto zero = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                               builder.getI32IntegerAttr(0));
  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule = getOperation().lookupSymbol<gpu::GPUModuleOp>(
      launchOp.getKernelModuleName());
  assert(kernelModule && "expected a kernel module");

  auto binaryAttr = kernelModule.getAttrOfType<StringAttr>(gpuBinaryAnnotation);
  if (!binaryAttr) {
    kernelModule.emitOpError()
        << "missing " << gpuBinaryAnnotation << " attribute";
    return signalPassFailure();
  }

  SmallString<128> nameBuffer(kernelModule.getName());
  nameBuffer.append(kGpuBinaryStorageSuffix);
  Value data = LLVM::createGlobalString(
      loc, builder, nameBuffer.str(), binaryAttr.getValue(),
      LLVM::Linkage::Internal, getLLVMDialect());

  // Emit the load module call to load the module data. Error checking is done
  // in the called helper function.
  auto gpuModule = allocatePointer(builder, loc);
  auto gpuModuleLoad =
      getOperation().lookupSymbol<LLVM::LLVMFuncOp>(kGpuModuleLoadName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getGpuRuntimeResultType()},
                               builder.getSymbolRefAttr(gpuModuleLoad),
                               ArrayRef<Value>{gpuModule, data});
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto gpuOwningModuleRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), gpuModule);
  auto kernelName = generateKernelNameConstant(
      launchOp.getKernelModuleName(), launchOp.getKernelName(), loc, builder);
  auto gpuFunction = allocatePointer(builder, loc);
  auto gpuModuleGetFunction =
      getOperation().lookupSymbol<LLVM::LLVMFuncOp>(kGpuModuleGetFunctionName);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getGpuRuntimeResultType()},
      builder.getSymbolRefAttr(gpuModuleGetFunction),
      ArrayRef<Value>{gpuFunction, gpuOwningModuleRef, kernelName});
  // Grab the global stream needed for execution.
  auto gpuGetStreamHelper =
      getOperation().lookupSymbol<LLVM::LLVMFuncOp>(kGpuGetStreamHelperName);
  auto gpuStream = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()},
      builder.getSymbolRefAttr(gpuGetStreamHelper), ArrayRef<Value>{});
  // Invoke the function with required arguments.
  auto gpuLaunchKernel =
      getOperation().lookupSymbol<LLVM::LLVMFuncOp>(kGpuLaunchKernelName);
  auto gpuFunctionRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), gpuFunction);
  auto paramsArray = setupParamsArray(launchOp, builder);
  if (!paramsArray) {
    launchOp.emitOpError() << "cannot pass given parameters to the kernel";
    return signalPassFailure();
  }
  auto nullpointer =
      builder.create<LLVM::IntToPtrOp>(loc, getPointerPointerType(), zero);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getGpuRuntimeResultType()},
      builder.getSymbolRefAttr(gpuLaunchKernel),
      ArrayRef<Value>{gpuFunctionRef, launchOp.getOperand(0),
                      launchOp.getOperand(1), launchOp.getOperand(2),
                      launchOp.getOperand(3), launchOp.getOperand(4),
                      launchOp.getOperand(5), zero, /* sharedMemBytes */
                      gpuStream.getResult(0),       /* stream */
                      paramsArray,                  /* kernel params */
                      nullpointer /* extra */});
  // Sync on the stream to make it synchronous.
  auto gpuStreamSync =
      getOperation().lookupSymbol<LLVM::LLVMFuncOp>(kGpuStreamSynchronizeName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getGpuRuntimeResultType()},
                               builder.getSymbolRefAttr(gpuStreamSync),
                               ArrayRef<Value>(gpuStream.getResult(0)));
  launchOp.erase();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToGpuRuntimeCallsPass() {
  return std::make_unique<GpuLaunchFuncToGpuRuntimeCallsPass>();
}
