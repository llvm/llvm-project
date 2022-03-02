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
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

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
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern lowering subgroup size/id to loading SPIR-V invocation
/// builtin variables.
template <typename SourceOp, spirv::BuiltIn builtin>
class SingleDimLaunchConfigConversion : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
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
  matchAndRewrite(gpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a kernel function in GPU dialect within a spv.module.
class GPUFuncOpConversion final : public OpConversionPattern<gpu::GPUFuncOp> {
public:
  using OpConversionPattern<gpu::GPUFuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  SmallVector<int32_t, 3> workGroupSizeAsInt32;
};

/// Pattern to convert a gpu.module to a spv.module.
class GPUModuleConversion final : public OpConversionPattern<gpu::GPUModuleOp> {
public:
  using OpConversionPattern<gpu::GPUModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp moduleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class GPUModuleEndConversion final
    : public OpConversionPattern<gpu::ModuleEndOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ModuleEndOp endOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(endOp);
    return success();
  }
};

/// Pattern to convert a gpu.return into a SPIR-V return.
// TODO: This can go to DRR when GPU return has operands.
class GPUReturnOpConversion final : public OpConversionPattern<gpu::ReturnOp> {
public:
  using OpConversionPattern<gpu::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a gpu.barrier op into a spv.ControlBarrier op.
class GPUBarrierConversion final : public OpConversionPattern<gpu::BarrierOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp barrierOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// Builtins.
//===----------------------------------------------------------------------===//

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult LaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, typename SourceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
  auto indexType = typeConverter->getIndexType();

  // SPIR-V invocation builtin variables are a vector of type <3xi32>
  auto spirvBuiltin =
      spirv::getBuiltinVariableValue(op, builtin, indexType, rewriter);
  rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
      op, indexType, spirvBuiltin,
      rewriter.getI32ArrayAttr({static_cast<int32_t>(op.dimension())}));
  return success();
}

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult
SingleDimLaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, typename SourceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
  auto indexType = typeConverter->getIndexType();

  auto spirvBuiltin =
      spirv::getBuiltinVariableValue(op, builtin, indexType, rewriter);
  rewriter.replaceOp(op, spirvBuiltin);
  return success();
}

LogicalResult WorkGroupSizeConversion::matchAndRewrite(
    gpu::BlockDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto workGroupSizeAttr = spirv::lookupLocalWorkGroupSize(op);
  auto val = workGroupSizeAttr
                 .getValues<int32_t>()[static_cast<int32_t>(op.dimension())];
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
    for (const auto &argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));
  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.getName() == FunctionOpInterface::getTypeAttrName() ||
        namedAttr.getName() == SymbolTable::getSymbolAttrName())
      continue;
    newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                         &signatureConverter)))
    return nullptr;
  rewriter.eraseOp(funcOp);

  // Set the attributes for argument and the function.
  StringRef argABIAttrName = spirv::getInterfaceVarABIAttrName();
  for (auto argIndex : llvm::seq<unsigned>(0, argABIInfo.size())) {
    newFuncOp.setArgAttr(argIndex, argABIAttrName, argABIInfo[argIndex]);
  }
  newFuncOp->setAttr(spirv::getEntryPointABIAttrName(), entryPointInfo);

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
    gpu::GPUFuncOp funcOp, OpAdaptor adaptor,
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
  newFuncOp->removeAttr(
      rewriter.getStringAttr(gpu::GPUDialect::getKernelFuncAttrName()));
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleOp with gpu.module.
//===----------------------------------------------------------------------===//

LogicalResult GPUModuleConversion::matchAndRewrite(
    gpu::GPUModuleOp moduleOp, OpAdaptor adaptor,
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
      moduleOp.getLoc(), addressingModel, memoryModel.getValue(), llvm::None,
      StringRef(spvModuleName));

  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.getRegion();
  rewriter.inlineRegionBefore(moduleOp.body(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spv.module build method adds a block. Remove that.
  rewriter.eraseBlock(&spvModuleRegion.back());
  rewriter.eraseOp(moduleOp);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU return inside kernel functions to SPIR-V return.
//===----------------------------------------------------------------------===//

LogicalResult GPUReturnOpConversion::matchAndRewrite(
    gpu::ReturnOp returnOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!adaptor.getOperands().empty())
    return failure();

  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Barrier.
//===----------------------------------------------------------------------===//

LogicalResult GPUBarrierConversion::matchAndRewrite(
    gpu::BarrierOp barrierOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *context = getContext();
  // Both execution and memory scope should be workgroup.
  auto scope = spirv::ScopeAttr::get(context, spirv::Scope::Workgroup);
  // Require acquire and release memory semantics for workgroup memory.
  auto memorySemantics = spirv::MemorySemanticsAttr::get(
      context, spirv::MemorySemantics::WorkgroupMemory |
                   spirv::MemorySemantics::AcquireRelease);
  rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(barrierOp, scope, scope,
                                                       memorySemantics);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU To SPIRV Patterns.
//===----------------------------------------------------------------------===//

void mlir::populateGPUToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.add<
      GPUBarrierConversion, GPUFuncOpConversion, GPUModuleConversion,
      GPUModuleEndConversion, GPUReturnOpConversion,
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
      WorkGroupSizeConversion>(typeConverter, patterns.getContext());
}
