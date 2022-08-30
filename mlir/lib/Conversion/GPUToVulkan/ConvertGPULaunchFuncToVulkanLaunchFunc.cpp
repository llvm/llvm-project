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

#include "../PassDetail.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/SPIRV/Serialization.h"

using namespace mlir;

static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

/// A pass to convert gpu launch op to vulkan launch call op, by creating a
/// SPIR-V binary shader from `spirv::ModuleOp` using `spirv::serialize`
/// function and attaching binary data and entry point name as an attributes to
/// created vulkan launch call op.
class ConvertGpuLaunchFuncToVulkanLaunchFunc
    : public ConvertGpuLaunchFuncToVulkanLaunchFuncBase<
          ConvertGpuLaunchFuncToVulkanLaunchFunc> {
public:
  void runOnOperation() override;

private:
  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  /// Converts the given `launchOp` to vulkan launch call.
  void convertGpuLaunchFunc(gpu::LaunchFuncOp launchOp);

  /// Checks where the given type is supported by Vulkan runtime.
  bool isSupportedType(Type type) {
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>()) {
      auto elementType = memRefType.getElementType();
      return memRefType.hasRank() &&
             (memRefType.getRank() >= 1 && memRefType.getRank() <= 3) &&
             (elementType.isIntOrFloat());
    }
    return false;
  }

  /// Declares the vulkan launch function. Returns an error if the any type of
  /// operand is unsupported by Vulkan runtime.
  LogicalResult declareVulkanLaunchFunc(Location loc,
                                        gpu::LaunchFuncOp launchOp);

private:
  /// The number of vulkan launch configuration operands, placed at the leading
  /// positions of the operand list.
  static constexpr unsigned kVulkanLaunchNumConfigOperands = 3;
};

} // namespace

void ConvertGpuLaunchFuncToVulkanLaunchFunc::runOnOperation() {
  bool done = false;
  getOperation().walk([this, &done](gpu::LaunchFuncOp op) {
    if (done) {
      op.emitError("should only contain one 'gpu::LaunchFuncOp' op");
      return signalPassFailure();
    }
    done = true;
    convertGpuLaunchFunc(op);
  });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getOperation().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getOperation().getOps<spirv::ModuleOp>()))
    spirvModule.erase();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::declareVulkanLaunchFunc(
    Location loc, gpu::LaunchFuncOp launchOp) {
  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());

  // Workgroup size is written into the kernel. So to properly modelling
  // vulkan launch, we have to skip local workgroup size configuration here.
  SmallVector<Type, 8> gpuLaunchTypes(launchOp.getOperandTypes());
  // The first kVulkanLaunchNumConfigOperands of the gpu.launch_func op are the
  // same as the config operands for the vulkan launch call op.
  SmallVector<Type, 8> vulkanLaunchTypes(gpuLaunchTypes.begin(),
                                         gpuLaunchTypes.begin() +
                                             kVulkanLaunchNumConfigOperands);
  vulkanLaunchTypes.append(gpuLaunchTypes.begin() +
                               gpu::LaunchOp::kNumConfigOperands,
                           gpuLaunchTypes.end());

  // Check that all operands have supported types except those for the
  // launch configuration.
  for (auto type :
       llvm::drop_begin(vulkanLaunchTypes, kVulkanLaunchNumConfigOperands)) {
    if (!isSupportedType(type))
      return launchOp.emitError() << type << " is unsupported to run on Vulkan";
  }

  // Declare vulkan launch function.
  auto funcType = builder.getFunctionType(vulkanLaunchTypes, {});
  builder.create<func::FuncOp>(loc, kVulkanLaunch, funcType).setPrivate();

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
  ModuleOp module = getOperation();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(createBinaryShader(module, binary)))
    return signalPassFailure();

  // Declare vulkan launch function.
  if (failed(declareVulkanLaunchFunc(loc, launchOp)))
    return signalPassFailure();

  SmallVector<Value, 8> gpuLaunchOperands(launchOp.getOperands());
  SmallVector<Value, 8> vulkanLaunchOperands(
      gpuLaunchOperands.begin(),
      gpuLaunchOperands.begin() + kVulkanLaunchNumConfigOperands);
  vulkanLaunchOperands.append(gpuLaunchOperands.begin() +
                                  gpu::LaunchOp::kNumConfigOperands,
                              gpuLaunchOperands.end());

  // Create vulkan launch call op.
  auto vulkanLaunchCallOp = builder.create<func::CallOp>(
      loc, TypeRange{}, SymbolRefAttr::get(builder.getContext(), kVulkanLaunch),
      vulkanLaunchOperands);

  // Set SPIR-V binary shader data as an attribute.
  vulkanLaunchCallOp->setAttr(
      kSPIRVBlobAttrName,
      builder.getStringAttr(StringRef(binary.data(), binary.size())));

  // Set entry point name as an attribute.
  vulkanLaunchCallOp->setAttr(kSPIRVEntryPointAttrName,
                              launchOp.getKernelName());

  launchOp.erase();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}
