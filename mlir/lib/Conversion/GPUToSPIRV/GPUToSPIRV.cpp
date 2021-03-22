//===- GPUToSPIRV.cpp - GPU to SPIR-V Patterns ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert GPU dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

static constexpr const char kSPIRVModule[] = "__spv__";

namespace {
/// Pattern lowering GPU block/thread size/id to loading SPIR-V invocation
/// builtin variables.
template <typename SourceOp, spirv::BuiltIn builtin>
class LaunchConfigConversion : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern lowering subgroup size/id to loading SPIR-V invocation
/// builtin variables.
template <typename SourceOp, spirv::BuiltIn builtin>
class SingleDimLaunchConfigConversion : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// This is separate because in Vulkan workgroup size is exposed to shaders via
/// a constant with WorkgroupSize decoration. So here we cannot generate a
/// builtin variable; instead the information in the `spv.entry_point_abi`
/// attribute on the surrounding FuncOp is used to replace the gpu::BlockDimOp.
class WorkGroupSizeConversion : public OpConversionPattern<gpu::BlockDimOp> {
public:
  using OpConversionPattern<gpu::BlockDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BlockDimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a kernel function in GPU dialect within a spv.module.
class GPUFuncOpConversion final : public OpConversionPattern<gpu::GPUFuncOp> {
public:
  using OpConversionPattern<gpu::GPUFuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  SmallVector<int32_t, 3> workGroupSizeAsInt32;
};

/// Pattern to convert a gpu.module to a spv.module.
class GPUModuleConversion final : public OpConversionPattern<gpu::GPUModuleOp> {
public:
  using OpConversionPattern<gpu::GPUModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp moduleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a gpu.return into a SPIR-V return.
// TODO: This can go to DRR when GPU return has operands.
class GPUReturnOpConversion final : public OpConversionPattern<gpu::ReturnOp> {
public:
  using OpConversionPattern<gpu::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// Builtins.
//===----------------------------------------------------------------------===//

static Optional<int32_t> getLaunchConfigIndex(Operation *op) {
  auto dimAttr = op->getAttrOfType<StringAttr>("dimension");
  if (!dimAttr)
    return llvm::None;

  return llvm::StringSwitch<Optional<int32_t>>(dimAttr.getValue())
      .Case("x", 0)
      .Case("y", 1)
      .Case("z", 2)
      .Default(llvm::None);
}

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult LaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto index = getLaunchConfigIndex(op);
  if (!index)
    return failure();

  // SPIR-V invocation builtin variables are a vector of type <3xi32>
  auto spirvBuiltin = spirv::getBuiltinVariableValue(op, builtin, rewriter);
  rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
      op, rewriter.getIntegerType(32), spirvBuiltin,
      rewriter.getI32ArrayAttr({index.getValue()}));
  return success();
}

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult
SingleDimLaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto spirvBuiltin = spirv::getBuiltinVariableValue(op, builtin, rewriter);
  rewriter.replaceOp(op, spirvBuiltin);
  return success();
}

LogicalResult WorkGroupSizeConversion::matchAndRewrite(
    gpu::BlockDimOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto index = getLaunchConfigIndex(op);
  if (!index)
    return failure();

  auto workGroupSizeAttr = spirv::lookupLocalWorkGroupSize(op);
  auto val = workGroupSizeAttr.getValue<int32_t>(index.getValue());
  auto convertedType =
      getTypeConverter()->convertType(op.getResult().getType());
  if (!convertedType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
      op, convertedType, IntegerAttr::get(convertedType, val));
  return success();
}

//===----------------------------------------------------------------------===//
// GPUFuncOp
//===----------------------------------------------------------------------===//

// Legalizes a GPU function as an entry SPIR-V function.
static spirv::FuncOp
lowerAsEntryFunction(gpu::GPUFuncOp funcOp, TypeConverter &typeConverter,
                     ConversionPatternRewriter &rewriter,
                     spirv::EntryPointABIAttr entryPointInfo,
                     ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    funcOp.emitError("SPIR-V lowering only supports entry functions"
                     "with no return values right now");
    return nullptr;
  }
  if (!argABIInfo.empty() && fnType.getNumInputs() != argABIInfo.size()) {
    funcOp.emitError(
        "lowering as entry functions requires ABI info for all arguments "
        "or none of them");
    return nullptr;
  }
  // Update the signature to valid SPIR-V types and add the ABI
  // attributes. These will be "materialized" by using the
  // LowerABIAttributesPass.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    for (auto argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));
  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.first == impl::getTypeAttrName() ||
        namedAttr.first == SymbolTable::getSymbolAttrName())
      continue;
    newFuncOp->setAttr(namedAttr.first, namedAttr.second);
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                         &signatureConverter)))
    return nullptr;
  rewriter.eraseOp(funcOp);

  if (failed(spirv::setABIAttrs(newFuncOp, entryPointInfo, argABIInfo)))
    return nullptr;
  return newFuncOp;
}

/// Populates `argABI` with spv.interface_var_abi attributes for lowering
/// gpu.func to spv.func if no arguments have the attributes set
/// already. Returns failure if any argument has the ABI attribute set already.
static LogicalResult
getDefaultABIAttrs(MLIRContext *context, gpu::GPUFuncOp funcOp,
                   SmallVectorImpl<spirv::InterfaceVarABIAttr> &argABI) {
  spirv::TargetEnvAttr targetEnv = spirv::lookupTargetEnvOrDefault(funcOp);
  if (!spirv::needsInterfaceVarABIAttrs(targetEnv))
    return success();

  for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    if (funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
            argIndex, spirv::getInterfaceVarABIAttrName()))
      return failure();
    // Vulkan's interface variable requirements needs scalars to be wrapped in a
    // struct. The struct held in storage buffer.
    Optional<spirv::StorageClass> sc;
    if (funcOp.getArgument(argIndex).getType().isIntOrIndexOrFloat())
      sc = spirv::StorageClass::StorageBuffer;
    argABI.push_back(spirv::getInterfaceVarABIAttr(0, argIndex, sc, context));
  }
  return success();
}

LogicalResult GPUFuncOpConversion::matchAndRewrite(
    gpu::GPUFuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!gpu::GPUDialect::isKernel(funcOp))
    return failure();

  SmallVector<spirv::InterfaceVarABIAttr, 4> argABI;
  if (failed(getDefaultABIAttrs(rewriter.getContext(), funcOp, argABI))) {
    argABI.clear();
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
      // If the ABI is already specified, use it.
      auto abiAttr = funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
          argIndex, spirv::getInterfaceVarABIAttrName());
      if (!abiAttr) {
        funcOp.emitRemark(
            "match failure: missing 'spv.interface_var_abi' attribute at "
            "argument ")
            << argIndex;
        return failure();
      }
      argABI.push_back(abiAttr);
    }
  }

  auto entryPointAttr = spirv::lookupEntryPointABI(funcOp);
  if (!entryPointAttr) {
    funcOp.emitRemark("match failure: missing 'spv.entry_point_abi' attribute");
    return failure();
  }
  spirv::FuncOp newFuncOp = lowerAsEntryFunction(
      funcOp, *getTypeConverter(), rewriter, entryPointAttr, argABI);
  if (!newFuncOp)
    return failure();
  newFuncOp->removeAttr(Identifier::get(
      gpu::GPUDialect::getKernelFuncAttrName(), rewriter.getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleOp with gpu.module.
//===----------------------------------------------------------------------===//

LogicalResult GPUModuleConversion::matchAndRewrite(
    gpu::GPUModuleOp moduleOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  spirv::TargetEnvAttr targetEnv = spirv::lookupTargetEnvOrDefault(moduleOp);
  spirv::AddressingModel addressingModel = spirv::getAddressingModel(targetEnv);
  FailureOr<spirv::MemoryModel> memoryModel = spirv::getMemoryModel(targetEnv);
  if (failed(memoryModel))
    return moduleOp.emitRemark("match failure: could not selected memory model "
                               "based on 'spv.target_env'");

  // Add a keyword to the module name to avoid symbolic conflict.
  std::string spvModuleName = (kSPIRVModule + moduleOp.getName()).str();
  auto spvModule = rewriter.create<spirv::ModuleOp>(
      moduleOp.getLoc(), addressingModel, memoryModel.getValue(),
      StringRef(spvModuleName));

  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.body();
  rewriter.inlineRegionBefore(moduleOp.body(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spv.module build method adds a block with a terminator. Remove that
  // block. The terminator of the module op in the remaining block will be
  // legalized later.
  rewriter.eraseBlock(&spvModuleRegion.back());
  rewriter.eraseOp(moduleOp);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU return inside kernel functions to SPIR-V return.
//===----------------------------------------------------------------------===//

LogicalResult GPUReturnOpConversion::matchAndRewrite(
    gpu::ReturnOp returnOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!operands.empty())
    return failure();

  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU To SPIRV Patterns.
//===----------------------------------------------------------------------===//

namespace {
#include "GPUToSPIRV.cpp.inc"
}

void mlir::populateGPUToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  auto *context = patterns.getContext();
  populateWithGenerated(context, patterns);
  patterns.insert<
      GPUFuncOpConversion, GPUModuleConversion, GPUReturnOpConversion,
      LaunchConfigConversion<gpu::BlockIdOp, spirv::BuiltIn::WorkgroupId>,
      LaunchConfigConversion<gpu::GridDimOp, spirv::BuiltIn::NumWorkgroups>,
      LaunchConfigConversion<gpu::ThreadIdOp,
                             spirv::BuiltIn::LocalInvocationId>,
      SingleDimLaunchConfigConversion<gpu::SubgroupIdOp,
                                      spirv::BuiltIn::SubgroupId>,
      SingleDimLaunchConfigConversion<gpu::NumSubgroupsOp,
                                      spirv::BuiltIn::NumSubgroups>,
      SingleDimLaunchConfigConversion<gpu::SubgroupSizeOp,
                                      spirv::BuiltIn::SubgroupSize>,
      WorkGroupSizeConversion>(typeConverter, context);
}
