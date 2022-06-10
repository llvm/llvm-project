//===- ConvertLaunchFuncToLLVMCalls.cpp - MLIR GPU launch to LLVM pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes to convert `gpu.launch_func` op into a sequence
// of LLVM calls that emulate the host and device sides.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

static constexpr const char kSPIRVModule[] = "__spv__";

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns the string name of the `DescriptorSet` decoration.
static std::string descriptorSetName() {
  return llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::DescriptorSet));
}

/// Returns the string name of the `Binding` decoration.
static std::string bindingName() {
  return llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::Binding));
}

/// Calculates the index of the kernel's operand that is represented by the
/// given global variable with the `bind` attribute. We assume that the index of
/// each kernel's operand is mapped to (descriptorSet, binding) by the map:
///   i -> (0, i)
/// which is implemented under `LowerABIAttributesPass`.
static unsigned calculateGlobalIndex(spirv::GlobalVariableOp op) {
  IntegerAttr binding = op->getAttrOfType<IntegerAttr>(bindingName());
  return binding.getInt();
}

/// Copies the given number of bytes from src to dst pointers.
static void copy(Location loc, Value dst, Value src, Value size,
                 OpBuilder &builder) {
  MLIRContext *context = builder.getContext();
  auto llvmI1Type = IntegerType::get(context, 1);
  Value isVolatile = builder.create<LLVM::ConstantOp>(
      loc, llvmI1Type, builder.getBoolAttr(false));
  builder.create<LLVM::MemcpyOp>(loc, dst, src, size, isVolatile);
}

/// Encodes the binding and descriptor set numbers into a new symbolic name.
/// The name is specified by
///   {kernel_module_name}_{variable_name}_descriptor_set{ds}_binding{b}
/// to avoid symbolic conflicts, where 'ds' and 'b' are descriptor set and
/// binding numbers.
static std::string
createGlobalVariableWithBindName(spirv::GlobalVariableOp op,
                                 StringRef kernelModuleName) {
  IntegerAttr descriptorSet =
      op->getAttrOfType<IntegerAttr>(descriptorSetName());
  IntegerAttr binding = op->getAttrOfType<IntegerAttr>(bindingName());
  return llvm::formatv("{0}_{1}_descriptor_set{2}_binding{3}",
                       kernelModuleName.str(), op.sym_name().str(),
                       std::to_string(descriptorSet.getInt()),
                       std::to_string(binding.getInt()));
}

/// Returns true if the given global variable has both a descriptor set number
/// and a binding number.
static bool hasDescriptorSetAndBinding(spirv::GlobalVariableOp op) {
  IntegerAttr descriptorSet =
      op->getAttrOfType<IntegerAttr>(descriptorSetName());
  IntegerAttr binding = op->getAttrOfType<IntegerAttr>(bindingName());
  return descriptorSet && binding;
}

/// Fills `globalVariableMap` with SPIR-V global variables that represent kernel
/// arguments from the given SPIR-V module. We assume that the module contains a
/// single entry point function. Hence, all `spv.GlobalVariable`s with a bind
/// attribute are kernel arguments.
static LogicalResult getKernelGlobalVariables(
    spirv::ModuleOp module,
    DenseMap<uint32_t, spirv::GlobalVariableOp> &globalVariableMap) {
  auto entryPoints = module.getOps<spirv::EntryPointOp>();
  if (!llvm::hasSingleElement(entryPoints)) {
    return module.emitError(
        "The module must contain exactly one entry point function");
  }
  auto globalVariables = module.getOps<spirv::GlobalVariableOp>();
  for (auto globalOp : globalVariables) {
    if (hasDescriptorSetAndBinding(globalOp))
      globalVariableMap[calculateGlobalIndex(globalOp)] = globalOp;
  }
  return success();
}

/// Encodes the SPIR-V module's symbolic name into the name of the entry point
/// function.
static LogicalResult encodeKernelName(spirv::ModuleOp module) {
  StringRef spvModuleName = module.sym_name().getValue();
  // We already know that the module contains exactly one entry point function
  // based on `getKernelGlobalVariables()` call. Update this function's name
  // to:
  //   {spv_module_name}_{function_name}
  auto entryPoint = *module.getOps<spirv::EntryPointOp>().begin();
  StringRef funcName = entryPoint.fn();
  auto funcOp = module.lookupSymbol<spirv::FuncOp>(entryPoint.fnAttr());
  StringAttr newFuncName =
      StringAttr::get(module->getContext(), spvModuleName + "_" + funcName);
  if (failed(SymbolTable::replaceAllSymbolUses(funcOp, newFuncName, module)))
    return failure();
  SymbolTable::setSymbolName(funcOp, newFuncName);
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

/// Structure to group information about the variables being copied.
struct CopyInfo {
  Value dst;
  Value src;
  Value size;
};

/// This pattern emulates a call to the kernel in LLVM dialect. For that, we
/// copy the data to the global variable (emulating device side), call the
/// kernel as a normal void LLVM function, and copy the data back (emulating the
/// host side).
class GPULaunchLowering : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
  using ConvertOpToLLVMPattern<gpu::LaunchFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *op = launchOp.getOperation();
    MLIRContext *context = rewriter.getContext();
    auto module = launchOp->getParentOfType<ModuleOp>();

    // Get the SPIR-V module that represents the gpu kernel module. The module
    // is named:
    //   __spv__{kernel_module_name}
    // based on GPU to SPIR-V conversion.
    StringRef kernelModuleName = launchOp.getKernelModuleName().getValue();
    std::string spvModuleName = kSPIRVModule + kernelModuleName.str();
    auto spvModule = module.lookupSymbol<spirv::ModuleOp>(
        StringAttr::get(context, spvModuleName));
    if (!spvModule) {
      return launchOp.emitOpError("SPIR-V kernel module '")
             << spvModuleName << "' is not found";
    }

    // Declare kernel function in the main module so that it later can be linked
    // with its definition from the kernel module. We know that the kernel
    // function would have no arguments and the data is passed via global
    // variables. The name of the kernel will be
    //   {spv_module_name}_{kernel_function_name}
    // to avoid symbolic name conflicts.
    StringRef kernelFuncName = launchOp.getKernelName().getValue();
    std::string newKernelFuncName = spvModuleName + "_" + kernelFuncName.str();
    auto kernelFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, newKernelFuncName));
    if (!kernelFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      kernelFunc = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), newKernelFuncName,
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                      ArrayRef<Type>()));
      rewriter.setInsertionPoint(launchOp);
    }

    // Get all global variables associated with the kernel operands.
    DenseMap<uint32_t, spirv::GlobalVariableOp> globalVariableMap;
    if (failed(getKernelGlobalVariables(spvModule, globalVariableMap)))
      return failure();

    // Traverse kernel operands that were converted to MemRefDescriptors. For
    // each operand, create a global variable and copy data from operand to it.
    Location loc = launchOp.getLoc();
    SmallVector<CopyInfo, 4> copyInfo;
    auto numKernelOperands = launchOp.getNumKernelOperands();
    auto kernelOperands = adaptor.getOperands().take_back(numKernelOperands);
    for (const auto &operand : llvm::enumerate(kernelOperands)) {
      // Check if the kernel's operand is a ranked memref.
      auto memRefType = launchOp.getKernelOperand(operand.index())
                            .getType()
                            .dyn_cast<MemRefType>();
      if (!memRefType)
        return failure();

      // Calculate the size of the memref and get the pointer to the allocated
      // buffer.
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 4> strides;
      Value sizeBytes;
      getMemRefDescriptorSizes(loc, memRefType, {}, rewriter, sizes, strides,
                               sizeBytes);
      MemRefDescriptor descriptor(operand.value());
      Value src = descriptor.allocatedPtr(rewriter, loc);

      // Get the global variable in the SPIR-V module that is associated with
      // the kernel operand. Construct its new name and create a corresponding
      // LLVM dialect global variable.
      spirv::GlobalVariableOp spirvGlobal = globalVariableMap[operand.index()];
      auto pointeeType =
          spirvGlobal.type().cast<spirv::PointerType>().getPointeeType();
      auto dstGlobalType = typeConverter->convertType(pointeeType);
      if (!dstGlobalType)
        return failure();
      std::string name =
          createGlobalVariableWithBindName(spirvGlobal, spvModuleName);
      // Check if this variable has already been created.
      auto dstGlobal = module.lookupSymbol<LLVM::GlobalOp>(name);
      if (!dstGlobal) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        dstGlobal = rewriter.create<LLVM::GlobalOp>(
            loc, dstGlobalType,
            /*isConstant=*/false, LLVM::Linkage::Linkonce, name, Attribute(),
            /*alignment=*/0);
        rewriter.setInsertionPoint(launchOp);
      }

      // Copy the data from src operand pointer to dst global variable. Save
      // src, dst and size so that we can copy data back after emulating the
      // kernel call.
      Value dst = rewriter.create<LLVM::AddressOfOp>(loc, dstGlobal);
      copy(loc, dst, src, sizeBytes, rewriter);

      CopyInfo info;
      info.dst = dst;
      info.src = src;
      info.size = sizeBytes;
      copyInfo.push_back(info);
    }
    // Create a call to the kernel and copy the data back.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, kernelFunc,
                                              ArrayRef<Value>());
    for (CopyInfo info : copyInfo)
      copy(loc, info.src, info.dst, info.size, rewriter);
    return success();
  }
};

class LowerHostCodeToLLVM
    : public LowerHostCodeToLLVMBase<LowerHostCodeToLLVM> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Erase the GPU module.
    for (auto gpuModule :
         llvm::make_early_inc_range(module.getOps<gpu::GPUModuleOp>()))
      gpuModule.erase();

    // Specify options to lower to LLVM and pull in the conversion patterns.
    LowerToLLVMOptions options(module.getContext());
    options.emitCWrappers = true;
    auto *context = module.getContext();
    RewritePatternSet patterns(context);
    LLVMTypeConverter typeConverter(context, options);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<GPULaunchLowering>(typeConverter);

    // Pull in SPIR-V type conversion patterns to convert SPIR-V global
    // variable's type to LLVM dialect type.
    populateSPIRVToLLVMTypeConversion(typeConverter);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Finally, modify the kernel function in SPIR-V modules to avoid symbolic
    // conflicts.
    for (auto spvModule : module.getOps<spirv::ModuleOp>())
      (void)encodeKernelName(spvModule);
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::createLowerHostCodeToLLVMPass() {
  return std::make_unique<LowerHostCodeToLLVM>();
}
