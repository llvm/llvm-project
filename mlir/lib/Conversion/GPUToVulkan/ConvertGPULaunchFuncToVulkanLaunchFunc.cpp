//===- ConvertGPULaunchFuncToVulkanLaunchFunc.cpp - MLIR conversion pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu launch function into a vulkan
// launch function. Extracts the SPIR-V from a `gpu::BinaryOp` and attaches it
// along with the entry point name as attributes to a Vulkan launch call op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPULAUNCHFUNCTOVULKANLAUNCHFUNC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kSPIRVElementTypesAttrName = "spirv_element_types";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

/// A pass to convert gpu launch op to vulkan launch call op, by extracting a
/// SPIR-V binary shader from a `gpu::BinaryOp` and attaching binary data and
/// entry point name as an attributes to created vulkan launch call op.
class ConvertGpuLaunchFuncToVulkanLaunchFunc
    : public impl::ConvertGpuLaunchFuncToVulkanLaunchFuncBase<
          ConvertGpuLaunchFuncToVulkanLaunchFunc> {
public:
  void runOnOperation() override;

private:
  /// Extracts a SPIR-V binary shader from the given `module`, if any.
  /// Note that this also removes the binary from the IR.
  FailureOr<StringAttr> getBinaryShader(ModuleOp module);

  /// Converts the given `launchOp` to vulkan launch call.
  void convertGpuLaunchFunc(gpu::LaunchFuncOp launchOp);

  /// Checks where the given type is supported by Vulkan runtime.
  bool isSupportedType(Type type) {
    if (auto memRefType = dyn_cast_or_null<MemRefType>(type)) {
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

FailureOr<StringAttr>
ConvertGpuLaunchFuncToVulkanLaunchFunc::getBinaryShader(ModuleOp module) {
  bool done = false;
  StringAttr binaryAttr;
  gpu::BinaryOp binaryToErase;
  for (auto gpuBinary : module.getOps<gpu::BinaryOp>()) {
    if (done)
      return gpuBinary.emitError("should only contain one 'gpu.binary' op");
    done = true;

    ArrayRef<Attribute> objects = gpuBinary.getObjectsAttr().getValue();
    if (objects.size() != 1)
      return gpuBinary.emitError("should only contain a single object");

    auto object = cast<gpu::ObjectAttr>(objects[0]);

    if (!isa<spirv::TargetEnvAttr>(object.getTarget()))
      return gpuBinary.emitError(
          "should contain an object with a SPIR-V target environment");

    binaryAttr = object.getObject();
    binaryToErase = gpuBinary;
  }
  if (!done)
    return module.emitError("should contain a 'gpu.binary' op");

  // Remove the binary to avoid confusing later conversion passes.
  binaryToErase.erase();
  return binaryAttr;
}

void ConvertGpuLaunchFuncToVulkanLaunchFunc::convertGpuLaunchFunc(
    gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getOperation();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  FailureOr<StringAttr> binaryAttr = getBinaryShader(module);
  // Extract SPIR-V from `gpu.binary` op.
  if (failed(binaryAttr))
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
  vulkanLaunchCallOp->setAttr(kSPIRVBlobAttrName, *binaryAttr);

  // Set entry point name as an attribute.
  vulkanLaunchCallOp->setAttr(kSPIRVEntryPointAttrName,
                              launchOp.getKernelName());

  // Add MemRef element types before they're lost when lowering to LLVM.
  SmallVector<Type> elementTypes;
  for (Type type : llvm::drop_begin(launchOp.getOperandTypes(),
                                    gpu::LaunchOp::kNumConfigOperands)) {
    // The below cast always succeeds as it has already been verified in
    // 'declareVulkanLaunchFunc' that these are MemRefs with compatible element
    // types.
    elementTypes.push_back(cast<MemRefType>(type).getElementType());
  }
  vulkanLaunchCallOp->setAttr(kSPIRVElementTypesAttrName,
                              builder.getTypeArrayAttr(elementTypes));

  launchOp.erase();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}
