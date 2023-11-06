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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVULKANLAUNCHFUNCTOVULKANCALLSPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr const char *kCInterfaceVulkanLaunch =
    "_mlir_ciface_vulkanLaunch";
static constexpr const char *kDeinitVulkan = "deinitVulkan";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kInitVulkan = "initVulkan";
static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kSPIRVBinary = "SPIRV_BIN";
static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kSPIRVElementTypesAttrName = "spirv_element_types";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

/// A pass to convert vulkan launch call op into a sequence of Vulkan
/// runtime calls in the following order:
///
/// * initVulkan           -- initializes vulkan runtime
/// * bindMemRef           -- binds memref
/// * setBinaryShader      -- sets the binary shader data
/// * setEntryPoint        -- sets the entry point name
/// * setNumWorkGroups     -- sets the number of a local workgroups
/// * runOnVulkan          -- runs vulkan runtime
/// * deinitVulkan         -- deinitializes vulkan runtime
///
class VulkanLaunchFuncToVulkanCallsPass
    : public impl::ConvertVulkanLaunchFuncToVulkanCallsPassBase<
          VulkanLaunchFuncToVulkanCallsPass> {
private:
  void initializeCachedTypes() {
    llvmFloatType = Float32Type::get(&getContext());
    llvmVoidType = LLVM::LLVMVoidType::get(&getContext());
    llvmPointerType = LLVM::LLVMPointerType::get(&getContext());
    llvmInt32Type = IntegerType::get(&getContext(), 32);
    llvmInt64Type = IntegerType::get(&getContext(), 64);
  }

  Type getMemRefType(uint32_t rank, Type elemenType) {
    // According to the MLIR doc memref argument is converted into a
    // pointer-to-struct argument of type:
    // template <typename Elem, size_t Rank>
    // struct {
    //   Elem *allocated;
    //   Elem *aligned;
    //   int64_t offset;
    //   int64_t sizes[Rank]; // omitted when rank == 0
    //   int64_t strides[Rank]; // omitted when rank == 0
    // };
    auto llvmArrayRankElementSizeType =
        LLVM::LLVMArrayType::get(getInt64Type(), rank);

    // Create a type
    // `!llvm<"{ `element-type`*, `element-type`*, i64,
    // [`rank` x i64], [`rank` x i64]}">`.
    return LLVM::LLVMStructType::getLiteral(
        &getContext(),
        {llvmPointerType, llvmPointerType, getInt64Type(),
         llvmArrayRankElementSizeType, llvmArrayRankElementSizeType});
  }

  Type getVoidType() { return llvmVoidType; }
  Type getPointerType() { return llvmPointerType; }
  Type getInt32Type() { return llvmInt32Type; }
  Type getInt64Type() { return llvmInt64Type; }

  /// Creates an LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Checks whether the given LLVM::CallOp is a vulkan launch call op.
  bool isVulkanLaunchCallOp(LLVM::CallOp callOp) {
    return (callOp.getCallee() && *callOp.getCallee() == kVulkanLaunch &&
            callOp.getNumOperands() >= kVulkanLaunchNumConfigOperands);
  }

  /// Checks whether the given LLVM::CallOp is a "ci_face" vulkan launch call
  /// op.
  bool isCInterfaceVulkanLaunchCallOp(LLVM::CallOp callOp) {
    return (callOp.getCallee() &&
            *callOp.getCallee() == kCInterfaceVulkanLaunch &&
            callOp.getNumOperands() >= kVulkanLaunchNumConfigOperands);
  }

  /// Translates the given `vulkanLaunchCallOp` to the sequence of Vulkan
  /// runtime calls.
  void translateVulkanLaunchCall(LLVM::CallOp vulkanLaunchCallOp);

  /// Creates call to `bindMemRef` for each memref operand.
  void createBindMemRefCalls(LLVM::CallOp vulkanLaunchCallOp,
                             Value vulkanRuntime);

  /// Collects SPIRV attributes from the given `vulkanLaunchCallOp`.
  void collectSPIRVAttributes(LLVM::CallOp vulkanLaunchCallOp);

  /// Deduces a rank from the given 'launchCallArg`.
  LogicalResult deduceMemRefRank(Value launchCallArg, uint32_t &rank);

  /// Returns a string representation from the given `type`.
  StringRef stringifyType(Type type) {
    if (isa<Float32Type>(type))
      return "Float";
    if (isa<Float16Type>(type))
      return "Half";
    if (auto intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 32)
        return "Int32";
      if (intType.getWidth() == 16)
        return "Int16";
      if (intType.getWidth() == 8)
        return "Int8";
    }

    llvm_unreachable("unsupported type");
  }

public:
  using Base::Base;

  void runOnOperation() override;

private:
  Type llvmFloatType;
  Type llvmVoidType;
  Type llvmPointerType;
  Type llvmInt32Type;
  Type llvmInt64Type;

  struct SPIRVAttributes {
    StringAttr blob;
    StringAttr entryPoint;
    SmallVector<Type> elementTypes;
  };

  // TODO: Use an associative array to support multiple vulkan launch calls.
  SPIRVAttributes spirvAttributes;
  /// The number of vulkan launch configuration operands, placed at the leading
  /// positions of the operand list.
  static constexpr unsigned kVulkanLaunchNumConfigOperands = 3;
};

} // namespace

void VulkanLaunchFuncToVulkanCallsPass::runOnOperation() {
  initializeCachedTypes();

  // Collect SPIR-V attributes such as `spirv_blob` and
  // `spirv_entry_point_name`.
  getOperation().walk([this](LLVM::CallOp op) {
    if (isVulkanLaunchCallOp(op))
      collectSPIRVAttributes(op);
  });

  // Convert vulkan launch call op into a sequence of Vulkan runtime calls.
  getOperation().walk([this](LLVM::CallOp op) {
    if (isCInterfaceVulkanLaunchCallOp(op))
      translateVulkanLaunchCall(op);
  });
}

void VulkanLaunchFuncToVulkanCallsPass::collectSPIRVAttributes(
    LLVM::CallOp vulkanLaunchCallOp) {
  // Check that `kSPIRVBinary` and `kSPIRVEntryPoint` are present in attributes
  // for the given vulkan launch call.
  auto spirvBlobAttr =
      vulkanLaunchCallOp->getAttrOfType<StringAttr>(kSPIRVBlobAttrName);
  if (!spirvBlobAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVBlobAttrName << " attribute";
    return signalPassFailure();
  }

  auto spirvEntryPointNameAttr =
      vulkanLaunchCallOp->getAttrOfType<StringAttr>(kSPIRVEntryPointAttrName);
  if (!spirvEntryPointNameAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVEntryPointAttrName << " attribute";
    return signalPassFailure();
  }

  auto spirvElementTypesAttr =
      vulkanLaunchCallOp->getAttrOfType<ArrayAttr>(kSPIRVElementTypesAttrName);
  if (!spirvElementTypesAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVElementTypesAttrName << " attribute";
    return signalPassFailure();
  }
  if (llvm::any_of(spirvElementTypesAttr,
                   [](Attribute attr) { return !isa<TypeAttr>(attr); })) {
    vulkanLaunchCallOp.emitError()
        << "expected " << spirvElementTypesAttr << " to be an array of types";
    return signalPassFailure();
  }

  spirvAttributes.blob = spirvBlobAttr;
  spirvAttributes.entryPoint = spirvEntryPointNameAttr;
  spirvAttributes.elementTypes =
      llvm::to_vector(spirvElementTypesAttr.getAsValueRange<mlir::TypeAttr>());
}

void VulkanLaunchFuncToVulkanCallsPass::createBindMemRefCalls(
    LLVM::CallOp cInterfaceVulkanLaunchCallOp, Value vulkanRuntime) {
  if (cInterfaceVulkanLaunchCallOp.getNumOperands() ==
      kVulkanLaunchNumConfigOperands)
    return;
  OpBuilder builder(cInterfaceVulkanLaunchCallOp);
  Location loc = cInterfaceVulkanLaunchCallOp.getLoc();

  // Create LLVM constant for the descriptor set index.
  // Bind all memrefs to the `0` descriptor set, the same way as `GPUToSPIRV`
  // pass does.
  Value descriptorSet =
      builder.create<LLVM::ConstantOp>(loc, getInt32Type(), 0);

  for (auto [index, ptrToMemRefDescriptor] :
       llvm::enumerate(cInterfaceVulkanLaunchCallOp.getOperands().drop_front(
           kVulkanLaunchNumConfigOperands))) {
    // Create LLVM constant for the descriptor binding index.
    Value descriptorBinding =
        builder.create<LLVM::ConstantOp>(loc, getInt32Type(), index);

    if (index >= spirvAttributes.elementTypes.size()) {
      cInterfaceVulkanLaunchCallOp.emitError()
          << kSPIRVElementTypesAttrName << " missing element type for "
          << ptrToMemRefDescriptor;
      return signalPassFailure();
    }

    uint32_t rank = 0;
    Type type = spirvAttributes.elementTypes[index];
    if (failed(deduceMemRefRank(ptrToMemRefDescriptor, rank))) {
      cInterfaceVulkanLaunchCallOp.emitError()
          << "invalid memref descriptor " << ptrToMemRefDescriptor.getType();
      return signalPassFailure();
    }

    auto symbolName =
        llvm::formatv("bindMemRef{0}D{1}", rank, stringifyType(type)).str();
    // Create call to `bindMemRef`.
    builder.create<LLVM::CallOp>(
        loc, TypeRange(), StringRef(symbolName.data(), symbolName.size()),
        ValueRange{vulkanRuntime, descriptorSet, descriptorBinding,
                   ptrToMemRefDescriptor});
  }
}

LogicalResult
VulkanLaunchFuncToVulkanCallsPass::deduceMemRefRank(Value launchCallArg,
                                                    uint32_t &rank) {
  // Deduce the rank from the type used to allocate the lowered MemRef.
  auto alloca = launchCallArg.getDefiningOp<LLVM::AllocaOp>();
  if (!alloca)
    return failure();

  std::optional<Type> elementType = alloca.getElemType();
  assert(elementType && "expected to work with opaque pointers");
  auto llvmDescriptorTy = dyn_cast<LLVM::LLVMStructType>(*elementType);
  // template <typename Elem, size_t Rank>
  // struct {
  //   Elem *allocated;
  //   Elem *aligned;
  //   int64_t offset;
  //   int64_t sizes[Rank]; // omitted when rank == 0
  //   int64_t strides[Rank]; // omitted when rank == 0
  // };
  if (!llvmDescriptorTy)
    return failure();

  if (llvmDescriptorTy.getBody().size() == 3) {
    rank = 0;
    return success();
  }
  rank =
      cast<LLVM::LLVMArrayType>(llvmDescriptorTy.getBody()[3]).getNumElements();
  return success();
}

void VulkanLaunchFuncToVulkanCallsPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getOperation();
  auto builder = OpBuilder::atBlockEnd(module.getBody());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    {getPointerType(), getPointerType()}));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    {getPointerType(), getInt64Type(),
                                     getInt64Type(), getInt64Type()}));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMFunctionType::get(
            getVoidType(),
            {getPointerType(), getPointerType(), getInt32Type()}));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMFunctionType::get(getVoidType(), {getPointerType()}));
  }

  for (unsigned i = 1; i <= 3; i++) {
    SmallVector<Type, 5> types{
        Float32Type::get(&getContext()), IntegerType::get(&getContext(), 32),
        IntegerType::get(&getContext(), 16), IntegerType::get(&getContext(), 8),
        Float16Type::get(&getContext())};
    for (auto type : types) {
      std::string fnName = "bindMemRef" + std::to_string(i) + "D" +
                           std::string(stringifyType(type));
      if (isa<Float16Type>(type))
        type = IntegerType::get(&getContext(), 16);
      if (!module.lookupSymbol(fnName)) {
        auto fnType = LLVM::LLVMFunctionType::get(
            getVoidType(),
            {llvmPointerType, getInt32Type(), getInt32Type(), llvmPointerType},
            /*isVarArg=*/false);
        builder.create<LLVM::LLVMFuncOp>(loc, fnName, fnType);
      }
    }
  }

  if (!module.lookupSymbol(kInitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kInitVulkan, LLVM::LLVMFunctionType::get(getPointerType(), {}));
  }

  if (!module.lookupSymbol(kDeinitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kDeinitVulkan,
        LLVM::LLVMFunctionType::get(getVoidType(), {getPointerType()}));
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
                                  shaderName, LLVM::Linkage::Internal);
}

void VulkanLaunchFuncToVulkanCallsPass::translateVulkanLaunchCall(
    LLVM::CallOp cInterfaceVulkanLaunchCallOp) {
  OpBuilder builder(cInterfaceVulkanLaunchCallOp);
  Location loc = cInterfaceVulkanLaunchCallOp.getLoc();
  // Create call to `initVulkan`.
  auto initVulkanCall = builder.create<LLVM::CallOp>(
      loc, TypeRange{getPointerType()}, kInitVulkan);
  // The result of `initVulkan` function is a pointer to Vulkan runtime, we
  // need to pass that pointer to each Vulkan runtime call.
  auto vulkanRuntime = initVulkanCall.getResult();

  // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
  // that data to runtime call.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary, spirvAttributes.blob.getValue(),
      LLVM::Linkage::Internal);

  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), spirvAttributes.blob.getValue().size());

  // Create call to `bindMemRef` for each memref operand.
  createBindMemRefCalls(cInterfaceVulkanLaunchCallOp, vulkanRuntime);

  // Create call to `setBinaryShader` runtime function with the given pointer to
  // SPIR-V binary and binary size.
  builder.create<LLVM::CallOp>(
      loc, TypeRange(), kSetBinaryShader,
      ValueRange{vulkanRuntime, ptrToSPIRVBinary, binarySize});
  // Create LLVM global with entry point name.
  Value entryPointName = createEntryPointNameConstant(
      spirvAttributes.entryPoint.getValue(), loc, builder);
  // Create call to `setEntryPoint` runtime function with the given pointer to
  // entry point name.
  builder.create<LLVM::CallOp>(loc, TypeRange(), kSetEntryPoint,
                               ValueRange{vulkanRuntime, entryPointName});

  // Create number of local workgroup for each dimension.
  builder.create<LLVM::CallOp>(
      loc, TypeRange(), kSetNumWorkGroups,
      ValueRange{vulkanRuntime, cInterfaceVulkanLaunchCallOp.getOperand(0),
                 cInterfaceVulkanLaunchCallOp.getOperand(1),
                 cInterfaceVulkanLaunchCallOp.getOperand(2)});

  // Create call to `runOnVulkan` runtime function.
  builder.create<LLVM::CallOp>(loc, TypeRange(), kRunOnVulkan,
                               ValueRange{vulkanRuntime});

  // Create call to 'deinitVulkan' runtime function.
  builder.create<LLVM::CallOp>(loc, TypeRange(), kDeinitVulkan,
                               ValueRange{vulkanRuntime});

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  cInterfaceVulkanLaunchCallOp.erase();
}
