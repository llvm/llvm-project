//===- ConvertGPULaunchFuncToVulkanLaunchFunc.cpp - MLIR conversion pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu launch function into a vulkan
// launch function. Creates a SPIR-V binary shader from the `spirv::ModuleOp`
// using `spirv::serialize` function, attaches binary data and entry point name
// as an attributes to vulkan launch call op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

// A pass to convert gpu launch op to vulkan launch call op, by creating a
// SPIR-V binary shader from `spirv::ModuleOp` using `spirv::serialize`
// function and attaching binary data and entry point name as an attributes to
// created vulkan launch call op.
class ConvertGpuLaunchFuncToVulkanLaunchFunc
    : public ModulePass<ConvertGpuLaunchFuncToVulkanLaunchFunc> {
public:
  void runOnModule() override;

private:
  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  /// Converts the given `luanchOp` to vulkan launch call.
  void convertGpuLaunchFunc(gpu::LaunchFuncOp launchOp);

  /// Checks where the given type is supported by Vulkan runtime.
  bool isSupportedType(Type type) {
    // TODO(denis0x0D): Handle other types.
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>())
      return memRefType.hasRank() && memRefType.getRank() == 1;
    return false;
  }

  /// Declares the vulkan launch function. Returns an error if the any type of
  /// operand is unsupported by Vulkan runtime.
  LogicalResult declareVulkanLaunchFunc(Location loc,
                                        gpu::LaunchFuncOp launchOp);

};

} // anonymous namespace

void ConvertGpuLaunchFuncToVulkanLaunchFunc::runOnModule() {
  bool done = false;
  getModule().walk([this, &done](gpu::LaunchFuncOp op) {
    if (done) {
      op.emitError("should only contain one 'gpu::LaunchFuncOp' op");
      return signalPassFailure();
    }
    done = true;
    convertGpuLaunchFunc(op);
  });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
    spirvModule.erase();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::declareVulkanLaunchFunc(
    Location loc, gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(getModule().getBody()->getTerminator());
  // TODO: Workgroup size is written into the kernel. So to properly modelling
  // vulkan launch, we cannot have the local workgroup size configuration here.
  SmallVector<Type, 8> vulkanLaunchTypes{launchOp.getOperandTypes()};

  // Check that all operands have supported types except those for the launch
  // configuration.
  for (auto type : llvm::drop_begin(vulkanLaunchTypes, 6)) {
    if (!isSupportedType(type))
      return launchOp.emitError() << type << " is unsupported to run on Vulkan";
  }

  // Declare vulkan launch function.
  builder.create<FuncOp>(
      loc, kVulkanLaunch,
      FunctionType::get(vulkanLaunchTypes, ArrayRef<Type>{}, loc->getContext()),
      ArrayRef<NamedAttribute>{});

  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::createBinaryShader(
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

void ConvertGpuLaunchFuncToVulkanLaunchFunc::convertGpuLaunchFunc(
    gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(createBinaryShader(module, binary)))
    return signalPassFailure();

  // Declare vulkan launch function.
  if (failed(declareVulkanLaunchFunc(loc, launchOp)))
    return signalPassFailure();

  // Create vulkan launch call op.
  auto vulkanLaunchCallOp = builder.create<CallOp>(
      loc, ArrayRef<Type>{}, builder.getSymbolRefAttr(kVulkanLaunch),
      launchOp.getOperands());

  // Set SPIR-V binary shader data as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVBlobAttrName,
      StringAttr::get({binary.data(), binary.size()}, loc->getContext()));

  // Set entry point name as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVEntryPointAttrName,
      StringAttr::get(launchOp.kernel(), loc->getContext()));

  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}

static PassRegistration<ConvertGpuLaunchFuncToVulkanLaunchFunc>
    pass("convert-gpu-launch-to-vulkan-launch",
         "Convert gpu.launch_func to vulkanLaunch external call");
