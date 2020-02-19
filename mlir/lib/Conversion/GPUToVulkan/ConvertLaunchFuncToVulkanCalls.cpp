//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu.launch_func op into a sequence of
// Vulkan runtime calls. The Vulkan runtime API surface is huge so currently we
// don't expose separate external functions in IR for each of them, instead we
// expose a few external functions to wrapper libraries which manages Vulkan
// runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"

using namespace mlir;

static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kSPIRVBinary = "SPIRV_BIN";

namespace {

/// A pass to convert gpu.launch_func operation into a sequence of Vulkan
/// runtime calls.
///
/// * setBinaryShader      -- sets the binary shader data
/// * setEntryPoint        -- sets the entry point name
/// * setNumWorkGroups     -- sets the number of a local workgroups
/// * runOnVulkan          -- runs vulkan runtime
///
class GpuLaunchFuncToVulkanCalssPass
    : public ModulePass<GpuLaunchFuncToVulkanCalssPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  LLVM::LLVMType getVoidType() { return llvmVoidType; }
  LLVM::LLVMType getPointerType() { return llvmPointerType; }
  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  /// Creates a LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);

  /// Creates a LLVM constant for each dimension of local workgroup and
  /// populates the given `numWorkGroups`.
  LogicalResult createNumWorkGroups(Location loc, OpBuilder &builder,
                                    mlir::gpu::LaunchFuncOp launchOp,
                                    SmallVectorImpl<Value> &numWorkGroups);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Translates the given `launcOp` op to the sequence of Vulkan runtime calls
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  void runOnModule() override;

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
};

} // anonymous namespace

void GpuLaunchFuncToVulkanCalssPass::runOnModule() {
  initializeCachedTypes();

  getModule().walk(
      [this](mlir::gpu::LaunchFuncOp op) { translateGpuLaunchCalls(op); });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
    spirvModule.erase();
}

void GpuLaunchFuncToVulkanCalssPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(), {getInt32Type(), getInt32Type(), getInt32Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {getPointerType(), getInt32Type()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {},
                                      /*isVarArg=*/false));
  }
}

Value GpuLaunchFuncToVulkanCalssPass::createEntryPointNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  SmallString<16> shaderName(name.begin(), name.end());
  // Append `\0` to follow C style string given that LLVM::createGlobalString()
  // won't handle this directly for us.
  shaderName.push_back('\0');

  std::string entryPointGlobalName = (name + "_spv_entry_point_name").str();
  return LLVM::createGlobalString(loc, builder, entryPointGlobalName,
                                  shaderName, LLVM::Linkage::Internal,
                                  getLLVMDialect());
}

LogicalResult GpuLaunchFuncToVulkanCalssPass::createBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {
  bool done = false;
  SmallVector<uint32_t, 0> binary;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (done)
      return spirvModule.emitError("should only contain one 'spv.module' op");
    done = true;

    if (failed(spirv::serialize(spirvModule, binary)))
      return failure();
  }

  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());
  return success();
}

LogicalResult GpuLaunchFuncToVulkanCalssPass::createNumWorkGroups(
    Location loc, OpBuilder &builder, mlir::gpu::LaunchFuncOp launchOp,
    SmallVectorImpl<Value> &numWorkGroups) {
  for (auto index : llvm::seq(0, 3)) {
    auto numWorkGroupDimConstant = dyn_cast_or_null<ConstantOp>(
        launchOp.getOperand(index).getDefiningOp());

    if (!numWorkGroupDimConstant)
      return failure();

    auto numWorkGroupDimValue =
        numWorkGroupDimConstant.getValue().cast<IntegerAttr>().getInt();
    numWorkGroups.push_back(builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(numWorkGroupDimValue)));
  }

  return success();
}

void GpuLaunchFuncToVulkanCalssPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(
          GpuLaunchFuncToVulkanCalssPass::createBinaryShader(module, binary)))
    return signalPassFailure();

  // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
  // that data to runtime call.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary, StringRef(binary.data(), binary.size()),
      LLVM::Linkage::Internal, getLLVMDialect());
  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(binary.size()));
  // Create call to `setBinaryShader` runtime function with the given pointer to
  // SPIR-V binary and binary size.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetBinaryShader),
                               ArrayRef<Value>{ptrToSPIRVBinary, binarySize});

  // Create LLVM global with entry point name.
  Value entryPointName =
      createEntryPointNameConstant(launchOp.kernel(), loc, builder);
  // Create call to `setEntryPoint` runtime function with the given pointer to
  // entry point name.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetEntryPoint),
                               ArrayRef<Value>{entryPointName});

  // Create number of local workgroup for each dimension.
  SmallVector<Value, 3> numWorkGroups;
  if (failed(createNumWorkGroups(loc, builder, launchOp, numWorkGroups)))
    return signalPassFailure();

  // Create call `setNumWorkGroups` runtime function with the given numbers of
  // local workgroup.
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getVoidType()},
      builder.getSymbolRefAttr(kSetNumWorkGroups),
      ArrayRef<Value>{numWorkGroups[0], numWorkGroups[1], numWorkGroups[2]});

  // Create call to `runOnVulkan` runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kRunOnVulkan),
                               ArrayRef<Value>{});

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanCallsPass() {
  return std::make_unique<GpuLaunchFuncToVulkanCalssPass>();
}

static PassRegistration<GpuLaunchFuncToVulkanCalssPass>
    pass("launch-func-to-vulkan",
         "Convert gpu.launch_func op to Vulkan runtime calls");
