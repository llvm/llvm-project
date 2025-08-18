//===- ObjectHandler.cpp - Implements base ObjectManager attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `OffloadingLLVMTranslationAttrInterface` for the
// `SelectObject` attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace mlir;

namespace {
// Implementation of the `OffloadingLLVMTranslationAttrInterface` model.
class SelectObjectAttrImpl
    : public gpu::OffloadingLLVMTranslationAttrInterface::FallbackModel<
          SelectObjectAttrImpl> {
  // Returns the selected object for embedding.
  gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) const;

public:
  // Translates a `gpu.binary`, embedding the binary into a host LLVM module as
  // global binary string which gets loaded/unloaded into a global module
  // object through a global ctor/dtor.
  LogicalResult embedBinary(Attribute attribute, Operation *operation,
                            llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) const;

  // Translates a `gpu.launch_func` to a sequence of LLVM instructions resulting
  // in a kernel launch call.
  LogicalResult launchKernel(Attribute attribute,
                             Operation *launchFuncOperation,
                             Operation *binaryOperation,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) const;
};
} // namespace

gpu::ObjectAttr
SelectObjectAttrImpl::getSelectedObject(gpu::BinaryOp op) const {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();

  // Obtain the index of the object to select.
  int64_t index = -1;
  if (Attribute target =
          cast<gpu::SelectObjectAttr>(op.getOffloadingHandlerAttr())
              .getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto [i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("the requested target object couldn't be found");
    return nullptr;
  }
  return mlir::dyn_cast<gpu::ObjectAttr>(objects[index]);
}

static Twine getModuleIdentifier(StringRef moduleName) {
  return moduleName + "_module";
}

namespace llvm {
static LogicalResult embedBinaryImpl(StringRef moduleName,
                                     gpu::ObjectAttr object, Module &module) {

  // Embed the object as a global string.
  // Add null for assembly output for JIT paths that expect null-terminated
  // strings.
  bool addNull = (object.getFormat() == gpu::CompilationTarget::Assembly);
  StringRef serializedStr = object.getObject().getValue();
  Constant *serializedCst =
      ConstantDataArray::getString(module.getContext(), serializedStr, addNull);
  GlobalVariable *serializedObj =
      new GlobalVariable(module, serializedCst->getType(), true,
                         GlobalValue::LinkageTypes::InternalLinkage,
                         serializedCst, moduleName + "_binary");
  serializedObj->setAlignment(MaybeAlign(8));
  serializedObj->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  // Default JIT optimization level.
  auto optLevel = APInt::getZero(32);

  if (DictionaryAttr objectProps = object.getProperties()) {
    if (auto section = dyn_cast_or_null<StringAttr>(
            objectProps.get(gpu::elfSectionName))) {
      serializedObj->setSection(section.getValue());
    }
    // Check if there's an optimization level embedded in the object.
    if (auto optAttr = dyn_cast_or_null<IntegerAttr>(objectProps.get("O")))
      optLevel = optAttr.getValue();
  }

  IRBuilder<> builder(module.getContext());
  auto i32Ty = builder.getInt32Ty();
  auto i64Ty = builder.getInt64Ty();
  auto ptrTy = builder.getPtrTy(0);
  auto voidTy = builder.getVoidTy();

  // Embed the module as a global object.
  auto *modulePtr = new GlobalVariable(
      module, ptrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      /*Initializer=*/ConstantPointerNull::get(ptrTy),
      getModuleIdentifier(moduleName));

  auto *loadFn = Function::Create(FunctionType::get(voidTy, /*IsVarArg=*/false),
                                  GlobalValue::InternalLinkage,
                                  moduleName + "_load", module);
  loadFn->setSection(".text.startup");
  auto *loadBlock = BasicBlock::Create(module.getContext(), "entry", loadFn);
  builder.SetInsertPoint(loadBlock);
  Value *moduleObj = [&] {
    if (object.getFormat() == gpu::CompilationTarget::Assembly) {
      FunctionCallee moduleLoadFn = module.getOrInsertFunction(
          "mgpuModuleLoadJIT", FunctionType::get(ptrTy, {ptrTy, i32Ty}, false));
      Constant *optValue = ConstantInt::get(i32Ty, optLevel);
      return builder.CreateCall(moduleLoadFn, {serializedObj, optValue});
    } else {
      FunctionCallee moduleLoadFn = module.getOrInsertFunction(
          "mgpuModuleLoad", FunctionType::get(ptrTy, {ptrTy, i64Ty}, false));
      Constant *binarySize =
          ConstantInt::get(i64Ty, serializedStr.size() + (addNull ? 1 : 0));
      return builder.CreateCall(moduleLoadFn, {serializedObj, binarySize});
    }
  }();
  builder.CreateStore(moduleObj, modulePtr);
  builder.CreateRetVoid();
  appendToGlobalCtors(module, loadFn, /*Priority=*/123);

  auto *unloadFn = Function::Create(
      FunctionType::get(voidTy, /*IsVarArg=*/false),
      GlobalValue::InternalLinkage, moduleName + "_unload", module);
  unloadFn->setSection(".text.startup");
  auto *unloadBlock =
      BasicBlock::Create(module.getContext(), "entry", unloadFn);
  builder.SetInsertPoint(unloadBlock);
  FunctionCallee moduleUnloadFn = module.getOrInsertFunction(
      "mgpuModuleUnload", FunctionType::get(voidTy, ptrTy, false));
  builder.CreateCall(moduleUnloadFn, builder.CreateLoad(ptrTy, modulePtr));
  builder.CreateRetVoid();
  appendToGlobalDtors(module, unloadFn, /*Priority=*/123);

  return success();
}
} // namespace llvm

LogicalResult SelectObjectAttrImpl::embedBinary(
    Attribute attribute, Operation *operation, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  assert(operation && "The binary operation must be non null.");
  if (!operation)
    return failure();

  auto op = mlir::dyn_cast<gpu::BinaryOp>(operation);
  if (!op) {
    operation->emitError("operation must be a GPU binary");
    return failure();
  }

  gpu::ObjectAttr object = getSelectedObject(op);
  if (!object)
    return failure();

  return embedBinaryImpl(op.getName(), object,
                         *moduleTranslation.getLLVMModule());
}

namespace llvm {
namespace {
class LaunchKernel {
public:
  LaunchKernel(Module &module, IRBuilderBase &builder,
               mlir::LLVM::ModuleTranslation &moduleTranslation);
  // Get the kernel launch callee.
  FunctionCallee getKernelLaunchFn();

  // Get the kernel launch callee.
  FunctionCallee getClusterKernelLaunchFn();

  // Get the module function callee.
  FunctionCallee getModuleFunctionFn();

  // Get the stream create callee.
  FunctionCallee getStreamCreateFn();

  // Get the stream destroy callee.
  FunctionCallee getStreamDestroyFn();

  // Get the stream sync callee.
  FunctionCallee getStreamSyncFn();

  // Ger or create the function name global string.
  Value *getOrCreateFunctionName(StringRef moduleName, StringRef kernelName);

  // Create the void* kernel array for passing the arguments.
  Value *createKernelArgArray(mlir::gpu::LaunchFuncOp op);

  // Create the full kernel launch.
  llvm::LogicalResult createKernelLaunch(mlir::gpu::LaunchFuncOp op,
                                         mlir::gpu::ObjectAttr object);

private:
  Module &module;
  IRBuilderBase &builder;
  mlir::LLVM::ModuleTranslation &moduleTranslation;
  Type *i32Ty{};
  Type *i64Ty{};
  Type *voidTy{};
  Type *intPtrTy{};
  PointerType *ptrTy{};
};
} // namespace
} // namespace llvm

LogicalResult SelectObjectAttrImpl::launchKernel(
    Attribute attribute, Operation *launchFuncOperation,
    Operation *binaryOperation, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  assert(launchFuncOperation && "The launch func operation must be non null.");
  if (!launchFuncOperation)
    return failure();

  auto launchFuncOp = mlir::dyn_cast<gpu::LaunchFuncOp>(launchFuncOperation);
  if (!launchFuncOp) {
    launchFuncOperation->emitError("operation must be a GPU launch func Op.");
    return failure();
  }

  auto binOp = mlir::dyn_cast<gpu::BinaryOp>(binaryOperation);
  if (!binOp) {
    binaryOperation->emitError("operation must be a GPU binary.");
    return failure();
  }
  gpu::ObjectAttr object = getSelectedObject(binOp);
  if (!object)
    return failure();

  return llvm::LaunchKernel(*moduleTranslation.getLLVMModule(), builder,
                            moduleTranslation)
      .createKernelLaunch(launchFuncOp, object);
}

llvm::LaunchKernel::LaunchKernel(
    Module &module, IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation)
    : module(module), builder(builder), moduleTranslation(moduleTranslation) {
  i32Ty = builder.getInt32Ty();
  i64Ty = builder.getInt64Ty();
  ptrTy = builder.getPtrTy(0);
  voidTy = builder.getVoidTy();
  intPtrTy = builder.getIntPtrTy(module.getDataLayout());
}

llvm::FunctionCallee llvm::LaunchKernel::getKernelLaunchFn() {
  return module.getOrInsertFunction(
      "mgpuLaunchKernel",
      FunctionType::get(voidTy,
                        ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy,
                                          intPtrTy, intPtrTy, intPtrTy, i32Ty,
                                          ptrTy, ptrTy, ptrTy, i64Ty}),
                        false));
}

llvm::FunctionCallee llvm::LaunchKernel::getClusterKernelLaunchFn() {
  return module.getOrInsertFunction(
      "mgpuLaunchClusterKernel",
      FunctionType::get(
          voidTy,
          ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                            intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                            i32Ty, ptrTy, ptrTy, ptrTy}),
          false));
}

llvm::FunctionCallee llvm::LaunchKernel::getModuleFunctionFn() {
  return module.getOrInsertFunction(
      "mgpuModuleGetFunction",
      FunctionType::get(ptrTy, ArrayRef<Type *>({ptrTy, ptrTy}), false));
}

llvm::FunctionCallee llvm::LaunchKernel::getStreamCreateFn() {
  return module.getOrInsertFunction("mgpuStreamCreate",
                                    FunctionType::get(ptrTy, false));
}

llvm::FunctionCallee llvm::LaunchKernel::getStreamDestroyFn() {
  return module.getOrInsertFunction(
      "mgpuStreamDestroy",
      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
}

llvm::FunctionCallee llvm::LaunchKernel::getStreamSyncFn() {
  return module.getOrInsertFunction(
      "mgpuStreamSynchronize",
      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
}

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
llvm::Value *llvm::LaunchKernel::getOrCreateFunctionName(StringRef moduleName,
                                                         StringRef kernelName) {
  std::string globalName =
      std::string(formatv("{0}_{1}_name", moduleName, kernelName));

  if (GlobalVariable *gv = module.getGlobalVariable(globalName, true))
    return gv;

  return builder.CreateGlobalString(kernelName, globalName);
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
llvm::Value *
llvm::LaunchKernel::createKernelArgArray(mlir::gpu::LaunchFuncOp op) {
  SmallVector<Value *> args =
      moduleTranslation.lookupValues(op.getKernelOperands());
  SmallVector<Type *> structTypes(args.size(), nullptr);

  for (auto [i, arg] : llvm::enumerate(args))
    structTypes[i] = arg->getType();

  Type *structTy = StructType::create(module.getContext(), structTypes);
  Value *argStruct = builder.CreateAlloca(structTy, 0u);
  Value *argArray = builder.CreateAlloca(
      ptrTy, ConstantInt::get(intPtrTy, structTypes.size()));

  for (auto [i, arg] : enumerate(args)) {
    Value *structMember = builder.CreateStructGEP(structTy, argStruct, i);
    builder.CreateStore(arg, structMember);
    Value *arrayMember = builder.CreateConstGEP1_32(ptrTy, argArray, i);
    builder.CreateStore(structMember, arrayMember);
  }
  return argArray;
}

// Emits LLVM IR to launch a kernel function:
// %1 = load %global_module_object
// %2 = call @mgpuModuleGetFunction(%1, %global_kernel_name)
// %3 = call @mgpuStreamCreate()
// %4 = <see createKernelArgArray()>
// call @mgpuLaunchKernel(%2, ..., %3, %4, ...)
// call @mgpuStreamSynchronize(%3)
// call @mgpuStreamDestroy(%3)
llvm::LogicalResult
llvm::LaunchKernel::createKernelLaunch(mlir::gpu::LaunchFuncOp op,
                                       mlir::gpu::ObjectAttr object) {
  auto llvmValue = [&](mlir::Value value) -> Value * {
    Value *v = moduleTranslation.lookupValue(value);
    assert(v && "Value has not been translated.");
    return v;
  };

  // Get grid dimensions.
  mlir::gpu::KernelDim3 grid = op.getGridSizeOperandValues();
  Value *gx = llvmValue(grid.x), *gy = llvmValue(grid.y),
        *gz = llvmValue(grid.z);

  // Get block dimensions.
  mlir::gpu::KernelDim3 block = op.getBlockSizeOperandValues();
  Value *bx = llvmValue(block.x), *by = llvmValue(block.y),
        *bz = llvmValue(block.z);

  // Get dynamic shared memory size.
  Value *dynamicMemorySize = nullptr;
  if (mlir::Value dynSz = op.getDynamicSharedMemorySize())
    dynamicMemorySize = llvmValue(dynSz);
  else
    dynamicMemorySize = ConstantInt::get(i32Ty, 0);

  // Create the argument array.
  Value *argArray = createKernelArgArray(op);

  // Load the kernel function.
  StringRef moduleName = op.getKernelModuleName().getValue();
  Twine moduleIdentifier = getModuleIdentifier(moduleName);
  Value *modulePtr = module.getGlobalVariable(moduleIdentifier.str(), true);
  if (!modulePtr)
    return op.emitError() << "Couldn't find the binary: " << moduleIdentifier;
  Value *moduleObj = builder.CreateLoad(ptrTy, modulePtr);
  Value *functionName = getOrCreateFunctionName(moduleName, op.getKernelName());
  Value *moduleFunction =
      builder.CreateCall(getModuleFunctionFn(), {moduleObj, functionName});

  // Get the stream to use for execution. If there's no async object then create
  // a stream to make a synchronous kernel launch.
  Value *stream = nullptr;
  // Sync & destroy the stream, for synchronous launches.
  auto destroyStream = make_scope_exit([&]() {
    builder.CreateCall(getStreamSyncFn(), {stream});
    builder.CreateCall(getStreamDestroyFn(), {stream});
  });
  if (mlir::Value asyncObject = op.getAsyncObject()) {
    stream = llvmValue(asyncObject);
    destroyStream.release();
  } else {
    stream = builder.CreateCall(getStreamCreateFn(), {});
  }

  llvm::Constant *paramsCount =
      llvm::ConstantInt::get(i64Ty, op.getNumKernelOperands());

  // Create the launch call.
  Value *nullPtr = ConstantPointerNull::get(ptrTy);

  // Launch kernel with clusters if cluster size is specified.
  if (op.hasClusterSize()) {
    mlir::gpu::KernelDim3 cluster = op.getClusterSizeOperandValues();
    Value *cx = llvmValue(cluster.x), *cy = llvmValue(cluster.y),
          *cz = llvmValue(cluster.z);
    builder.CreateCall(
        getClusterKernelLaunchFn(),
        ArrayRef<Value *>({moduleFunction, cx, cy, cz, gx, gy, gz, bx, by, bz,
                           dynamicMemorySize, stream, argArray, nullPtr}));
  } else {
    builder.CreateCall(getKernelLaunchFn(),
                       ArrayRef<Value *>({moduleFunction, gx, gy, gz, bx, by,
                                          bz, dynamicMemorySize, stream,
                                          argArray, nullPtr, paramsCount}));
  }

  return success();
}

void mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    SelectObjectAttr::attachInterface<SelectObjectAttrImpl>(*ctx);
  });
}
