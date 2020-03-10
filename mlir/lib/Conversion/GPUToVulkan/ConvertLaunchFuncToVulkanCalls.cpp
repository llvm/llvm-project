//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert vulkan launch call into a sequence of
// Vulkan runtime calls. The Vulkan runtime API surface is huge so currently we
// don't expose separate external functions in IR for each of them, instead we
// expose a few external functions to wrapper libraries which manages Vulkan
// runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"

using namespace mlir;

static constexpr const char *kBindResource = "bindResource";
static constexpr const char *kDeinitVulkan = "deinitVulkan";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kInitVulkan = "initVulkan";
static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kSPIRVBinary = "SPIRV_BIN";
static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

/// A pass to convert vulkan launch func into a sequence of Vulkan
/// runtime calls in the following order:
///
/// * initVulkan           -- initializes vulkan runtime
/// * bindResource         -- binds resource
/// * setBinaryShader      -- sets the binary shader data
/// * setEntryPoint        -- sets the entry point name
/// * setNumWorkGroups     -- sets the number of a local workgroups
/// * runOnVulkan          -- runs vulkan runtime
/// * deinitVulkan         -- deinitializes vulkan runtime
///
class VulkanLaunchFuncToVulkanCallsPass
    : public ModulePass<VulkanLaunchFuncToVulkanCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmFloatType = LLVM::LLVMType::getFloatTy(llvmDialect);
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  LLVM::LLVMType getFloatType() { return llvmFloatType; }
  LLVM::LLVMType getVoidType() { return llvmVoidType; }
  LLVM::LLVMType getPointerType() { return llvmPointerType; }
  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }
  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  /// Creates a LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Checks whether the given LLVM::CallOp is a vulkan launch call op.
  bool isVulkanLaunchCallOp(LLVM::CallOp callOp) {
    return (callOp.callee() && callOp.callee().getValue() == kVulkanLaunch &&
            callOp.getNumOperands() >= 6);
  }

  /// Translates the given `vulkanLaunchCallOp` to the sequence of Vulkan
  /// runtime calls.
  void translateVulkanLaunchCall(LLVM::CallOp vulkanLaunchCallOp);

  /// Creates call to `bindResource` for each resource operand.
  void createBindResourceCalls(LLVM::CallOp vulkanLaunchCallOp,
                               Value vulkanRuntiem);

public:
  void runOnModule() override;

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmFloatType;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
};

/// Represents operand adaptor for vulkan launch call operation, to simplify an
/// access to the lowered memref.
// TODO: We should use 'emit-c-wrappers' option to lower memref type:
// https://mlir.llvm.org/docs/ConversionToLLVMDialect/#c-compatible-wrapper-emission.
struct VulkanLaunchOpOperandAdaptor {
  VulkanLaunchOpOperandAdaptor(ArrayRef<Value> values) { operands = values; }
  VulkanLaunchOpOperandAdaptor(const VulkanLaunchOpOperandAdaptor &) = delete;
  VulkanLaunchOpOperandAdaptor
  operator=(const VulkanLaunchOpOperandAdaptor &) = delete;

  /// Returns a tuple with a pointer to the memory and the size for the index-th
  /// resource.
  std::tuple<Value, Value> getResourceDescriptor1D(uint32_t index) {
    assert(index < getResourceCount1D());
    // 1D memref calling convention according to "ConversionToLLVMDialect.md":
    // 0. Allocated pointer.
    // 1. Aligned pointer.
    // 2. Offset.
    // 3. Size in dim 0.
    // 4. Stride in dim 0.
    return {operands[numConfigOps + index * loweredMemRefNumOps1D],
            operands[numConfigOps + index * loweredMemRefNumOps1D + 3]};
  }

  /// Returns the number of resources assuming all operands lowered from
  /// 1D memref.
  uint32_t getResourceCount1D() {
    return (operands.size() - numConfigOps) / loweredMemRefNumOps1D;
  }

private:
  /// The number of operands of lowered 1D memref.
  static constexpr const uint32_t loweredMemRefNumOps1D = 5;
  /// The number of the first config operands.
  static constexpr const uint32_t numConfigOps = 6;
  ArrayRef<Value> operands;
};

} // anonymous namespace

void VulkanLaunchFuncToVulkanCallsPass::runOnModule() {
  initializeCachedTypes();
  getModule().walk([this](LLVM::CallOp op) {
    if (isVulkanLaunchCallOp(op))
      translateVulkanLaunchCall(op);
  });
}

void VulkanLaunchFuncToVulkanCallsPass::createBindResourceCalls(
    LLVM::CallOp vulkanLaunchCallOp, Value vulkanRuntime) {
  if (vulkanLaunchCallOp.getNumOperands() == 6)
    return;
  OpBuilder builder(vulkanLaunchCallOp);
  Location loc = vulkanLaunchCallOp.getLoc();

  // Create LLVM constant for the descriptor set index.
  // Bind all resources to the `0` descriptor set, the same way as `GPUToSPIRV`
  // pass does.
  Value descriptorSet = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(0));

  auto operands = SmallVector<Value, 32>{vulkanLaunchCallOp.getOperands()};
  VulkanLaunchOpOperandAdaptor vkLaunchOperandAdaptor(operands);

  for (auto resourceIdx :
       llvm::seq<uint32_t>(0, vkLaunchOperandAdaptor.getResourceCount1D())) {
    // Create LLVM constant for the descriptor binding index.
    Value descriptorBinding = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(resourceIdx));
    // Get a pointer to the memory and size of that memory.
    auto resourceDescriptor =
        vkLaunchOperandAdaptor.getResourceDescriptor1D(resourceIdx);
    // Create call to `bindResource`.
    builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{getVoidType()},
        builder.getSymbolRefAttr(kBindResource),
        ArrayRef<Value>{vulkanRuntime, descriptorSet, descriptorBinding,
                        // Pointer to the memory.
                        std::get<0>(resourceDescriptor),
                        // Size of the memory.
                        std::get<1>(resourceDescriptor)});
  }
}

void VulkanLaunchFuncToVulkanCallsPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {getPointerType(), getPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(),
            {getPointerType(), getInt64Type(), getInt64Type(), getInt64Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(), {getPointerType(), getPointerType(), getInt32Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kBindResource)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kBindResource,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(),
            {getPointerType(), getInt32Type(), getInt32Type(),
             getFloatType().getPointerTo(), getInt64Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kInitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kInitVulkan,
        LLVM::LLVMType::getFunctionTy(getPointerType(), {},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kDeinitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kDeinitVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }
}

Value VulkanLaunchFuncToVulkanCallsPass::createEntryPointNameConstant(
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

void VulkanLaunchFuncToVulkanCallsPass::translateVulkanLaunchCall(
    LLVM::CallOp vulkanLaunchCallOp) {
  OpBuilder builder(vulkanLaunchCallOp);
  Location loc = vulkanLaunchCallOp.getLoc();

  // Check that `kSPIRVBinary` and `kSPIRVEntryPoint` are present in attributes
  // for the given vulkan launch call.
  auto spirvBlobAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVBlobAttrName);
  if (!spirvBlobAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVBlobAttrName << " attribute";
    return signalPassFailure();
  }

  auto entryPointNameAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVEntryPointAttrName);
  if (!entryPointNameAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVEntryPointAttrName << " attribute";
    return signalPassFailure();
  }

  // Create call to `initVulkan`.
  auto initVulkanCall = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()},
      builder.getSymbolRefAttr(kInitVulkan), ArrayRef<Value>{});
  // The result of `initVulkan` function is a pointer to Vulkan runtime, we
  // need to pass that pointer to each Vulkan runtime call.
  auto vulkanRuntime = initVulkanCall.getResult(0);

  // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
  // that data to runtime call.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary, spirvBlobAttr.getValue(),
      LLVM::Linkage::Internal, getLLVMDialect());

  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(spirvBlobAttr.getValue().size()));

  // Create call to `bindResource` for each resource operand.
  createBindResourceCalls(vulkanLaunchCallOp, vulkanRuntime);

  // Create call to `setBinaryShader` runtime function with the given pointer to
  // SPIR-V binary and binary size.
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getVoidType()},
      builder.getSymbolRefAttr(kSetBinaryShader),
      ArrayRef<Value>{vulkanRuntime, ptrToSPIRVBinary, binarySize});
  // Create LLVM global with entry point name.
  Value entryPointName =
      createEntryPointNameConstant(entryPointNameAttr.getValue(), loc, builder);
  // Create call to `setEntryPoint` runtime function with the given pointer to
  // entry point name.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetEntryPoint),
                               ArrayRef<Value>{vulkanRuntime, entryPointName});

  // Create number of local workgroup for each dimension.
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getVoidType()},
      builder.getSymbolRefAttr(kSetNumWorkGroups),
      ArrayRef<Value>{vulkanRuntime, vulkanLaunchCallOp.getOperand(0),
                      vulkanLaunchCallOp.getOperand(1),
                      vulkanLaunchCallOp.getOperand(2)});

  // Create call to `runOnVulkan` runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kRunOnVulkan),
                               ArrayRef<Value>{vulkanRuntime});

  // Create call to 'deinitVulkan' runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kDeinitVulkan),
                               ArrayRef<Value>{vulkanRuntime});

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  vulkanLaunchCallOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertVulkanLaunchFuncToVulkanCallsPass() {
  return std::make_unique<VulkanLaunchFuncToVulkanCallsPass>();
}

static PassRegistration<VulkanLaunchFuncToVulkanCallsPass>
    pass("launch-func-to-vulkan",
         "Convert vulkanLaunch external call to Vulkan runtime external calls");
