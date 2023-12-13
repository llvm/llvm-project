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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

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
/// builtin variable; instead the information in the `spirv.entry_point_abi`
/// attribute on the surrounding FuncOp is used to replace the gpu::BlockDimOp.
class WorkGroupSizeConversion : public OpConversionPattern<gpu::BlockDimOp> {
public:
  WorkGroupSizeConversion(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context, /*benefit*/ 10) {}

  LogicalResult
  matchAndRewrite(gpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a kernel function in GPU dialect within a spirv.module.
class GPUFuncOpConversion final : public OpConversionPattern<gpu::GPUFuncOp> {
public:
  using OpConversionPattern<gpu::GPUFuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  SmallVector<int32_t, 3> workGroupSizeAsInt32;
};

/// Pattern to convert a gpu.module to a spirv.module.
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

/// Pattern to convert a gpu.barrier op into a spirv.ControlBarrier op.
class GPUBarrierConversion final : public OpConversionPattern<gpu::BarrierOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp barrierOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a gpu.shuffle op into a spirv.GroupNonUniformShuffle op.
class GPUShuffleConversion final : public OpConversionPattern<gpu::ShuffleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp shuffleOp, OpAdaptor adaptor,
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
  Type indexType = typeConverter->getIndexType();

  // For Vulkan, these SPIR-V builtin variables are required to be a vector of
  // type <3xi32> by the spec:
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/NumWorkgroups.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/WorkgroupId.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/WorkgroupSize.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/LocalInvocationId.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/LocalInvocationId.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/GlobalInvocationId.html
  //
  // For OpenCL, it depends on the Physical32/Physical64 addressing model:
  // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_built_in_variables
  bool forShader =
      typeConverter->getTargetEnv().allows(spirv::Capability::Shader);
  Type builtinType = forShader ? rewriter.getIntegerType(32) : indexType;

  Value vector =
      spirv::getBuiltinVariableValue(op, builtin, builtinType, rewriter);
  Value dim = rewriter.create<spirv::CompositeExtractOp>(
      op.getLoc(), builtinType, vector,
      rewriter.getI32ArrayAttr({static_cast<int32_t>(op.getDimension())}));
  if (forShader && builtinType != indexType)
    dim = rewriter.create<spirv::UConvertOp>(op.getLoc(), indexType, dim);
  rewriter.replaceOp(op, dim);
  return success();
}

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult
SingleDimLaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, typename SourceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
  Type indexType = typeConverter->getIndexType();
  Type i32Type = rewriter.getIntegerType(32);

  // For Vulkan, these SPIR-V builtin variables are required to be a vector of
  // type i32 by the spec:
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/NumSubgroups.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/SubgroupId.html
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/SubgroupSize.html
  //
  // For OpenCL, they are also required to be i32:
  // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_built_in_variables
  Value builtinValue =
      spirv::getBuiltinVariableValue(op, builtin, i32Type, rewriter);
  if (i32Type != indexType)
    builtinValue = rewriter.create<spirv::UConvertOp>(op.getLoc(), indexType,
                                                      builtinValue);
  rewriter.replaceOp(op, builtinValue);
  return success();
}

LogicalResult WorkGroupSizeConversion::matchAndRewrite(
    gpu::BlockDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  DenseI32ArrayAttr workGroupSizeAttr = spirv::lookupLocalWorkGroupSize(op);
  if (!workGroupSizeAttr)
    return failure();

  int val =
      workGroupSizeAttr.asArrayRef()[static_cast<int32_t>(op.getDimension())];
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
lowerAsEntryFunction(gpu::GPUFuncOp funcOp, const TypeConverter &typeConverter,
                     ConversionPatternRewriter &rewriter,
                     spirv::EntryPointABIAttr entryPointInfo,
                     ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
  auto fnType = funcOp.getFunctionType();
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
    for (const auto &argType :
         enumerate(funcOp.getFunctionType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      if (!convertedType)
        return nullptr;
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               std::nullopt));
  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.getName() == funcOp.getFunctionTypeAttrName() ||
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

/// Populates `argABI` with spirv.interface_var_abi attributes for lowering
/// gpu.func to spirv.func if no arguments have the attributes set
/// already. Returns failure if any argument has the ABI attribute set already.
static LogicalResult
getDefaultABIAttrs(const spirv::TargetEnv &targetEnv, gpu::GPUFuncOp funcOp,
                   SmallVectorImpl<spirv::InterfaceVarABIAttr> &argABI) {
  if (!spirv::needsInterfaceVarABIAttrs(targetEnv))
    return success();

  for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    if (funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
            argIndex, spirv::getInterfaceVarABIAttrName()))
      return failure();
    // Vulkan's interface variable requirements needs scalars to be wrapped in a
    // struct. The struct held in storage buffer.
    std::optional<spirv::StorageClass> sc;
    if (funcOp.getArgument(argIndex).getType().isIntOrIndexOrFloat())
      sc = spirv::StorageClass::StorageBuffer;
    argABI.push_back(
        spirv::getInterfaceVarABIAttr(0, argIndex, sc, funcOp.getContext()));
  }
  return success();
}

LogicalResult GPUFuncOpConversion::matchAndRewrite(
    gpu::GPUFuncOp funcOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!gpu::GPUDialect::isKernel(funcOp))
    return failure();

  auto *typeConverter = getTypeConverter<SPIRVTypeConverter>();
  SmallVector<spirv::InterfaceVarABIAttr, 4> argABI;
  if (failed(
          getDefaultABIAttrs(typeConverter->getTargetEnv(), funcOp, argABI))) {
    argABI.clear();
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
      // If the ABI is already specified, use it.
      auto abiAttr = funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
          argIndex, spirv::getInterfaceVarABIAttrName());
      if (!abiAttr) {
        funcOp.emitRemark(
            "match failure: missing 'spirv.interface_var_abi' attribute at "
            "argument ")
            << argIndex;
        return failure();
      }
      argABI.push_back(abiAttr);
    }
  }

  auto entryPointAttr = spirv::lookupEntryPointABI(funcOp);
  if (!entryPointAttr) {
    funcOp.emitRemark(
        "match failure: missing 'spirv.entry_point_abi' attribute");
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
  auto *typeConverter = getTypeConverter<SPIRVTypeConverter>();
  const spirv::TargetEnv &targetEnv = typeConverter->getTargetEnv();
  spirv::AddressingModel addressingModel = spirv::getAddressingModel(
      targetEnv, typeConverter->getOptions().use64bitIndex);
  FailureOr<spirv::MemoryModel> memoryModel = spirv::getMemoryModel(targetEnv);
  if (failed(memoryModel))
    return moduleOp.emitRemark(
        "cannot deduce memory model from 'spirv.target_env'");

  // Add a keyword to the module name to avoid symbolic conflict.
  std::string spvModuleName = (kSPIRVModule + moduleOp.getName()).str();
  auto spvModule = rewriter.create<spirv::ModuleOp>(
      moduleOp.getLoc(), addressingModel, *memoryModel, std::nullopt,
      StringRef(spvModuleName));

  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.getRegion();
  rewriter.inlineRegionBefore(moduleOp.getBodyRegion(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spirv.module build method adds a block. Remove that.
  rewriter.eraseBlock(&spvModuleRegion.back());

  // Some of the patterns call `lookupTargetEnv` during conversion and they
  // will fail if called after GPUModuleConversion and we don't preserve
  // `TargetEnv` attribute.
  // Copy TargetEnvAttr only if it is attached directly to the GPUModuleOp.
  if (auto attr = moduleOp->getAttrOfType<spirv::TargetEnvAttr>(
          spirv::getTargetEnvAttrName()))
    spvModule->setAttr(spirv::getTargetEnvAttrName(), attr);

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
// Shuffle
//===----------------------------------------------------------------------===//

LogicalResult GPUShuffleConversion::matchAndRewrite(
    gpu::ShuffleOp shuffleOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Require the shuffle width to be the same as the target's subgroup size,
  // given that for SPIR-V non-uniform subgroup ops, we cannot select
  // participating invocations.
  auto targetEnv = getTypeConverter<SPIRVTypeConverter>()->getTargetEnv();
  unsigned subgroupSize =
      targetEnv.getAttr().getResourceLimits().getSubgroupSize();
  IntegerAttr widthAttr;
  if (!matchPattern(shuffleOp.getWidth(), m_Constant(&widthAttr)) ||
      widthAttr.getValue().getZExtValue() != subgroupSize)
    return rewriter.notifyMatchFailure(
        shuffleOp, "shuffle width and target subgroup size mismatch");

  Location loc = shuffleOp.getLoc();
  Value trueVal = spirv::ConstantOp::getOne(rewriter.getI1Type(),
                                            shuffleOp.getLoc(), rewriter);
  auto scope = rewriter.getAttr<spirv::ScopeAttr>(spirv::Scope::Subgroup);
  Value result;

  switch (shuffleOp.getMode()) {
  case gpu::ShuffleMode::XOR:
    result = rewriter.create<spirv::GroupNonUniformShuffleXorOp>(
        loc, scope, adaptor.getValue(), adaptor.getOffset());
    break;
  case gpu::ShuffleMode::IDX:
    result = rewriter.create<spirv::GroupNonUniformShuffleOp>(
        loc, scope, adaptor.getValue(), adaptor.getOffset());
    break;
  default:
    return rewriter.notifyMatchFailure(shuffleOp, "unimplemented shuffle mode");
  }

  rewriter.replaceOp(shuffleOp, {result, trueVal});
  return success();
}

//===----------------------------------------------------------------------===//
// Group ops
//===----------------------------------------------------------------------===//

template <typename UniformOp, typename NonUniformOp>
static Value createGroupReduceOpImpl(OpBuilder &builder, Location loc,
                                     Value arg, bool isGroup, bool isUniform) {
  Type type = arg.getType();
  auto scope = mlir::spirv::ScopeAttr::get(builder.getContext(),
                                           isGroup ? spirv::Scope::Workgroup
                                                   : spirv::Scope::Subgroup);
  auto groupOp = spirv::GroupOperationAttr::get(builder.getContext(),
                                                spirv::GroupOperation::Reduce);
  if (isUniform) {
    return builder.create<UniformOp>(loc, type, scope, groupOp, arg)
        .getResult();
  }
  return builder.create<NonUniformOp>(loc, type, scope, groupOp, arg, Value{})
      .getResult();
}

static std::optional<Value> createGroupReduceOp(OpBuilder &builder,
                                                Location loc, Value arg,
                                                gpu::AllReduceOperation opType,
                                                bool isGroup, bool isUniform) {
  enum class ElemType { Float, Boolean, Integer };
  using FuncT = Value (*)(OpBuilder &, Location, Value, bool, bool);
  struct OpHandler {
    gpu::AllReduceOperation kind;
    ElemType elemType;
    FuncT func;
  };

  Type type = arg.getType();
  ElemType elementType;
  if (isa<FloatType>(type)) {
    elementType = ElemType::Float;
  } else if (auto intTy = dyn_cast<IntegerType>(type)) {
    elementType = (intTy.getIntOrFloatBitWidth() == 1) ? ElemType::Boolean
                                                       : ElemType::Integer;
  } else {
    return std::nullopt;
  }

  // TODO(https://github.com/llvm/llvm-project/issues/73459): The SPIR-V spec
  // does not specify how -0.0 / +0.0 and NaN values are handled in *FMin/*FMax
  // reduction ops. We should account possible precision requirements in this
  // conversion.

  using ReduceType = gpu::AllReduceOperation;
  const OpHandler handlers[] = {
      {ReduceType::ADD, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupIAddOp,
                                spirv::GroupNonUniformIAddOp>},
      {ReduceType::ADD, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFAddOp,
                                spirv::GroupNonUniformFAddOp>},
      {ReduceType::MUL, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupIMulKHROp,
                                spirv::GroupNonUniformIMulOp>},
      {ReduceType::MUL, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFMulKHROp,
                                spirv::GroupNonUniformFMulOp>},
      {ReduceType::MINUI, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupUMinOp,
                                spirv::GroupNonUniformUMinOp>},
      {ReduceType::MINSI, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupSMinOp,
                                spirv::GroupNonUniformSMinOp>},
      {ReduceType::MINF, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFMinOp,
                                spirv::GroupNonUniformFMinOp>},
      {ReduceType::MAXUI, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupUMaxOp,
                                spirv::GroupNonUniformUMaxOp>},
      {ReduceType::MAXSI, ElemType::Integer,
       &createGroupReduceOpImpl<spirv::GroupSMaxOp,
                                spirv::GroupNonUniformSMaxOp>},
      {ReduceType::MAXF, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFMaxOp,
                                spirv::GroupNonUniformFMaxOp>},
      {ReduceType::MINIMUMF, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFMinOp,
                                spirv::GroupNonUniformFMinOp>},
      {ReduceType::MAXIMUMF, ElemType::Float,
       &createGroupReduceOpImpl<spirv::GroupFMaxOp,
                                spirv::GroupNonUniformFMaxOp>}};

  for (const OpHandler &handler : handlers)
    if (handler.kind == opType && elementType == handler.elemType)
      return handler.func(builder, loc, arg, isGroup, isUniform);

  return std::nullopt;
}

/// Pattern to convert a gpu.all_reduce op into a SPIR-V group op.
class GPUAllReduceConversion final
    : public OpConversionPattern<gpu::AllReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::AllReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = op.getOp();

    // gpu.all_reduce can have either reduction op attribute or reduction
    // region. Only attribute version is supported.
    if (!opType)
      return failure();

    auto result =
        createGroupReduceOp(rewriter, op.getLoc(), adaptor.getValue(), *opType,
                            /*isGroup*/ true, op.getUniform());
    if (!result)
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }
};

/// Pattern to convert a gpu.subgroup_reduce op into a SPIR-V group op.
class GPUSubgroupReduceConversion final
    : public OpConversionPattern<gpu::SubgroupReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = op.getOp();
    auto result =
        createGroupReduceOp(rewriter, op.getLoc(), adaptor.getValue(), opType,
                            /*isGroup*/ false, op.getUniform());
    if (!result)
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GPU To SPIRV Patterns.
//===----------------------------------------------------------------------===//

void mlir::populateGPUToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.add<
      GPUBarrierConversion, GPUFuncOpConversion, GPUModuleConversion,
      GPUModuleEndConversion, GPUReturnOpConversion, GPUShuffleConversion,
      LaunchConfigConversion<gpu::BlockIdOp, spirv::BuiltIn::WorkgroupId>,
      LaunchConfigConversion<gpu::GridDimOp, spirv::BuiltIn::NumWorkgroups>,
      LaunchConfigConversion<gpu::BlockDimOp, spirv::BuiltIn::WorkgroupSize>,
      LaunchConfigConversion<gpu::ThreadIdOp,
                             spirv::BuiltIn::LocalInvocationId>,
      LaunchConfigConversion<gpu::GlobalIdOp,
                             spirv::BuiltIn::GlobalInvocationId>,
      SingleDimLaunchConfigConversion<gpu::SubgroupIdOp,
                                      spirv::BuiltIn::SubgroupId>,
      SingleDimLaunchConfigConversion<gpu::NumSubgroupsOp,
                                      spirv::BuiltIn::NumSubgroups>,
      SingleDimLaunchConfigConversion<gpu::SubgroupSizeOp,
                                      spirv::BuiltIn::SubgroupSize>,
      WorkGroupSizeConversion, GPUAllReduceConversion,
      GPUSubgroupReduceConversion>(typeConverter, patterns.getContext());
}
