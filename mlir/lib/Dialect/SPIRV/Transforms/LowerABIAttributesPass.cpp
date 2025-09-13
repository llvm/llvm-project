//===- LowerABIAttributesPass.cpp - Decorate composite type ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower attributes that specify the shader ABI
// for the functions in the generated SPIR-V module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVLOWERABIATTRIBUTESPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

/// Creates a global variable for an argument based on the ABI info.
static spirv::GlobalVariableOp
createGlobalVarForEntryPointArgument(OpBuilder &builder, spirv::FuncOp funcOp,
                                     unsigned argIndex,
                                     spirv::InterfaceVarABIAttr abiInfo) {
  auto spirvModule = funcOp->getParentOfType<spirv::ModuleOp>();
  if (!spirvModule)
    return nullptr;

  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  builder.setInsertionPoint(funcOp.getOperation());
  std::string varName =
      funcOp.getName().str() + "_arg_" + std::to_string(argIndex);

  // Get the type of variable. If this is a scalar/vector type and has an ABI
  // info create a variable of type !spirv.ptr<!spirv.struct<elementType>>. If
  // not it must already be a !spirv.ptr<!spirv.struct<...>>.
  auto varType = funcOp.getFunctionType().getInput(argIndex);
  if (cast<spirv::SPIRVType>(varType).isScalarOrVector()) {
    auto storageClass = abiInfo.getStorageClass();
    if (!storageClass)
      return nullptr;
    varType =
        spirv::PointerType::get(spirv::StructType::get(varType), *storageClass);
  }
  auto varPtrType = cast<spirv::PointerType>(varType);
  Type pointeeType = varPtrType.getPointeeType();

  // Images are an opaque type and so we can just return a pointer to an image.
  // Note that currently only sampled images are supported in the SPIR-V
  // lowering.
  if (isa<spirv::SampledImageType>(pointeeType))
    return spirv::GlobalVariableOp::create(builder, funcOp.getLoc(), varType,
                                           varName, abiInfo.getDescriptorSet(),
                                           abiInfo.getBinding());

  auto varPointeeType = cast<spirv::StructType>(pointeeType);

  // Set the offset information.
  varPointeeType =
      cast<spirv::StructType>(VulkanLayoutUtils::decorateType(varPointeeType));

  if (!varPointeeType)
    return nullptr;

  varType =
      spirv::PointerType::get(varPointeeType, varPtrType.getStorageClass());

  return spirv::GlobalVariableOp::create(builder, funcOp.getLoc(), varType,
                                         varName, abiInfo.getDescriptorSet(),
                                         abiInfo.getBinding());
}

/// Creates a global variable for an argument or result based on the ABI info.
static spirv::GlobalVariableOp
createGlobalVarForGraphEntryPoint(OpBuilder &builder, spirv::GraphARMOp graphOp,
                                  unsigned index, bool isArg,
                                  spirv::InterfaceVarABIAttr abiInfo) {
  auto spirvModule = graphOp->getParentOfType<spirv::ModuleOp>();
  if (!spirvModule)
    return nullptr;

  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  builder.setInsertionPoint(graphOp.getOperation());
  std::string varName = llvm::formatv("{}_{}_{}", graphOp.getName(),
                                      isArg ? "arg" : "res", index);

  Type varType = isArg ? graphOp.getFunctionType().getInput(index)
                       : graphOp.getFunctionType().getResult(index);

  auto pointerType = spirv::PointerType::get(
      varType,
      abiInfo.getStorageClass().value_or(spirv::StorageClass::UniformConstant));

  return spirv::GlobalVariableOp::create(builder, graphOp.getLoc(), pointerType,
                                         varName, abiInfo.getDescriptorSet(),
                                         abiInfo.getBinding());
}

/// Gets the global variables that need to be specified as interface variable
/// with an spirv.EntryPointOp. Traverses the body of a entry function to do so.
static LogicalResult
getInterfaceVariables(mlir::FunctionOpInterface funcOp,
                      SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp->getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  spirv::TargetEnvAttr targetEnvAttr = spirv::lookupTargetEnv(funcOp);
  spirv::TargetEnv targetEnv(targetEnvAttr);

  SetVector<Operation *> interfaceVarSet;

  // TODO: This should in reality traverse the entry function
  // call graph and collect all the interfaces. For now, just traverse the
  // instructions in this function.
  funcOp.walk([&](spirv::AddressOfOp addressOfOp) {
    auto var =
        module.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.getVariable());
    // Per SPIR-V spec: "Before version 1.4, the interface's
    // storage classes are limited to the Input and Output storage classes.
    // Starting with version 1.4, the interface's storage classes are all
    // storage classes used in declaring all global variables referenced by the
    // entry point’s call tree."
    const spirv::StorageClass storageClass =
        cast<spirv::PointerType>(var.getType()).getStorageClass();
    if ((targetEnvAttr && targetEnv.getVersion() >= spirv::Version::V_1_4) ||
        (llvm::is_contained(
            {spirv::StorageClass::Input, spirv::StorageClass::Output},
            storageClass))) {
      interfaceVarSet.insert(var.getOperation());
    }
  });
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
        funcOp.getContext(), cast<spirv::GlobalVariableOp>(var).getSymName()));
  }
  return success();
}

/// Lowers the entry point attribute.
static LogicalResult lowerEntryPointABIAttr(spirv::FuncOp funcOp,
                                            OpBuilder &builder) {
  auto entryPointAttrName = spirv::getEntryPointABIAttrName();
  auto entryPointAttr =
      funcOp->getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName);
  if (!entryPointAttr) {
    return failure();
  }

  spirv::TargetEnvAttr targetEnvAttr = spirv::lookupTargetEnv(funcOp);
  spirv::TargetEnv targetEnv(targetEnvAttr);

  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  auto spirvModule = funcOp->getParentOfType<spirv::ModuleOp>();
  builder.setInsertionPointToEnd(spirvModule.getBody());

  // Adds the spirv.EntryPointOp after collecting all the interface variables
  // needed.
  SmallVector<Attribute, 1> interfaceVars;
  if (failed(getInterfaceVariables(funcOp, interfaceVars))) {
    return failure();
  }

  FailureOr<spirv::ExecutionModel> executionModel =
      spirv::getExecutionModel(targetEnvAttr);
  if (failed(executionModel))
    return funcOp.emitRemark("lower entry point failure: could not select "
                             "execution model based on 'spirv.target_env'");

  spirv::EntryPointOp::create(builder, funcOp.getLoc(), *executionModel, funcOp,
                              interfaceVars);

  // Specifies the spirv.ExecutionModeOp.
  if (DenseI32ArrayAttr workgroupSizeAttr = entryPointAttr.getWorkgroupSize()) {
    std::optional<ArrayRef<spirv::Capability>> caps =
        spirv::getCapabilities(spirv::ExecutionMode::LocalSize);
    if (!caps || targetEnv.allows(*caps)) {
      spirv::ExecutionModeOp::create(builder, funcOp.getLoc(), funcOp,
                                     spirv::ExecutionMode::LocalSize,
                                     workgroupSizeAttr.asArrayRef());
      // Erase workgroup size.
      entryPointAttr = spirv::EntryPointABIAttr::get(
          entryPointAttr.getContext(), DenseI32ArrayAttr(),
          entryPointAttr.getSubgroupSize(), entryPointAttr.getTargetWidth());
    }
  }
  if (std::optional<int> subgroupSize = entryPointAttr.getSubgroupSize()) {
    std::optional<ArrayRef<spirv::Capability>> caps =
        spirv::getCapabilities(spirv::ExecutionMode::SubgroupSize);
    if (!caps || targetEnv.allows(*caps)) {
      spirv::ExecutionModeOp::create(builder, funcOp.getLoc(), funcOp,
                                     spirv::ExecutionMode::SubgroupSize,
                                     *subgroupSize);
      // Erase subgroup size.
      entryPointAttr = spirv::EntryPointABIAttr::get(
          entryPointAttr.getContext(), entryPointAttr.getWorkgroupSize(),
          std::nullopt, entryPointAttr.getTargetWidth());
    }
  }
  if (std::optional<int> targetWidth = entryPointAttr.getTargetWidth()) {
    std::optional<ArrayRef<spirv::Capability>> caps =
        spirv::getCapabilities(spirv::ExecutionMode::SignedZeroInfNanPreserve);
    if (!caps || targetEnv.allows(*caps)) {
      spirv::ExecutionModeOp::create(
          builder, funcOp.getLoc(), funcOp,
          spirv::ExecutionMode::SignedZeroInfNanPreserve, *targetWidth);
      // Erase target width.
      entryPointAttr = spirv::EntryPointABIAttr::get(
          entryPointAttr.getContext(), entryPointAttr.getWorkgroupSize(),
          entryPointAttr.getSubgroupSize(), std::nullopt);
    }
  }
  if (entryPointAttr.getWorkgroupSize() || entryPointAttr.getSubgroupSize() ||
      entryPointAttr.getTargetWidth())
    funcOp->setAttr(entryPointAttrName, entryPointAttr);
  else
    funcOp->removeAttr(entryPointAttrName);
  return success();
}

namespace {
/// A pattern to convert function signature according to interface variable ABI
/// attributes.
///
/// Specifically, this pattern creates global variables according to interface
/// variable ABI attributes attached to function arguments and converts all
/// function argument uses to those global variables. This is necessary because
/// Vulkan requires all shader entry points to be of void(void) type.
class ProcessInterfaceVarABI final : public OpConversionPattern<spirv::FuncOp> {
public:
  using OpConversionPattern<spirv::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(spirv::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A pattern to convert graph signature according to interface variable ABI
/// attributes.
///
/// Specifically, this pattern creates global variables according to interface
/// variable ABI attributes attached to graph arguments and results.
class ProcessGraphInterfaceVarABI final
    : public OpConversionPattern<spirv::GraphARMOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(spirv::GraphARMOp graphOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pass to implement the ABI information specified as attributes.
class LowerABIAttributesPass final
    : public spirv::impl::SPIRVLowerABIAttributesPassBase<
          LowerABIAttributesPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult ProcessInterfaceVarABI::matchAndRewrite(
    spirv::FuncOp funcOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!funcOp->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName())) {
    // TODO: Non-entry point functions are not handled.
    return failure();
  }
  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getFunctionType().getNumInputs());

  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  auto indexType = typeConverter.getIndexType();

  auto attrName = spirv::getInterfaceVarABIAttrName();

  OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
  rewriter.setInsertionPointToStart(&funcOp.front());

  for (const auto &argType :
       llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    auto abiInfo = funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
        argType.index(), attrName);
    if (!abiInfo) {
      // TODO: For non-entry point functions, it should be legal
      // to pass around scalar/vector values and return a scalar/vector. For now
      // non-entry point functions are not handled in this ABI lowering and will
      // produce an error.
      return failure();
    }
    spirv::GlobalVariableOp var = createGlobalVarForEntryPointArgument(
        rewriter, funcOp, argType.index(), abiInfo);
    if (!var)
      return failure();

    // Insert spirv::AddressOf and spirv::AccessChain operations.
    Value replacement =
        spirv::AddressOfOp::create(rewriter, funcOp.getLoc(), var);
    // Check if the arg is a scalar or vector type. In that case, the value
    // needs to be loaded into registers.
    // TODO: This is loading value of the scalar into registers
    // at the start of the function. It is probably better to do the load just
    // before the use. There might be multiple loads and currently there is no
    // easy way to replace all uses with a sequence of operations.
    if (cast<spirv::SPIRVType>(argType.value()).isScalarOrVector()) {
      auto zero =
          spirv::ConstantOp::getZero(indexType, funcOp.getLoc(), rewriter);
      auto loadPtr = spirv::AccessChainOp::create(
          rewriter, funcOp.getLoc(), replacement, zero.getConstant());
      replacement = spirv::LoadOp::create(rewriter, funcOp.getLoc(), loadPtr);
    }
    signatureConverter.remapInput(argType.index(), replacement);
  }
  if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(),
                                         &signatureConverter)))
    return failure();

  // Creates a new function with the update signature.
  rewriter.modifyOpInPlace(funcOp, [&] {
    funcOp.setType(
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(), {}));
  });
  return success();
}

LogicalResult ProcessGraphInterfaceVarABI::matchAndRewrite(
    spirv::GraphARMOp graphOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Non-entry point graphs are not handled.
  if (!graphOp.getEntryPoint().value_or(false))
    return failure();

  TypeConverter::SignatureConversion signatureConverter(
      graphOp.getFunctionType().getNumInputs());

  StringRef attrName = spirv::getInterfaceVarABIAttrName();
  SmallVector<Attribute, 4> interfaceVars;

  // Convert arguments.
  unsigned numInputs = graphOp.getFunctionType().getNumInputs();
  unsigned numResults = graphOp.getFunctionType().getNumResults();
  for (unsigned index = 0; index < numInputs; ++index) {
    auto abiInfo =
        graphOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(index, attrName);
    if (!abiInfo)
      return failure();
    spirv::GlobalVariableOp var = createGlobalVarForGraphEntryPoint(
        rewriter, graphOp, index, true, abiInfo);
    if (!var)
      return failure();
    interfaceVars.push_back(
        SymbolRefAttr::get(rewriter.getContext(), var.getSymName()));
  }

  for (unsigned index = 0; index < numResults; ++index) {
    auto abiInfo = graphOp.getResultAttrOfType<spirv::InterfaceVarABIAttr>(
        index, attrName);
    if (!abiInfo)
      return failure();
    spirv::GlobalVariableOp var = createGlobalVarForGraphEntryPoint(
        rewriter, graphOp, index, false, abiInfo);
    if (!var)
      return failure();
    interfaceVars.push_back(
        SymbolRefAttr::get(rewriter.getContext(), var.getSymName()));
  }

  // Update graph signature.
  rewriter.modifyOpInPlace(graphOp, [&] {
    for (unsigned index = 0; index < numInputs; ++index) {
      graphOp.removeArgAttr(index, attrName);
    }
    for (unsigned index = 0; index < numResults; ++index) {
      graphOp.removeResultAttr(index, rewriter.getStringAttr(attrName));
    }
  });

  spirv::GraphEntryPointARMOp::create(rewriter, graphOp.getLoc(), graphOp,
                                      interfaceVars);
  return success();
}

void LowerABIAttributesPass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  spirv::ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  spirv::TargetEnvAttr targetEnvAttr = spirv::lookupTargetEnv(module);
  if (!targetEnvAttr) {
    module->emitOpError("missing SPIR-V target env attribute");
    return signalPassFailure();
  }
  spirv::TargetEnv targetEnv(targetEnvAttr);

  SPIRVTypeConverter typeConverter(targetEnv);

  // Insert a bitcast in the case of a pointer type change.
  typeConverter.addSourceMaterialization([](OpBuilder &builder,
                                            spirv::PointerType type,
                                            ValueRange inputs, Location loc) {
    if (inputs.size() != 1 || !isa<spirv::PointerType>(inputs[0].getType()))
      return Value();
    return spirv::BitcastOp::create(builder, loc, type, inputs[0]).getResult();
  });

  RewritePatternSet patterns(context);
  patterns.add<ProcessInterfaceVarABI, ProcessGraphInterfaceVarABI>(
      typeConverter, context);

  ConversionTarget target(*context);
  // "Legal" function ops should have no interface variable ABI attributes.
  target.addDynamicallyLegalOp<spirv::FuncOp>([&](spirv::FuncOp op) {
    StringRef attrName = spirv::getInterfaceVarABIAttrName();
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i)
      if (op.getArgAttr(i, attrName))
        return false;
    return true;
  });
  target.addDynamicallyLegalOp<spirv::GraphARMOp>([&](spirv::GraphARMOp op) {
    StringRef attrName = spirv::getInterfaceVarABIAttrName();
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i)
      if (op.getArgAttr(i, attrName))
        return false;
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
      if (op.getResultAttr(i, attrName))
        return false;
    return true;
  });

  // All other SPIR-V ops are legal.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return op->getDialect()->getNamespace() ==
           spirv::SPIRVDialect::getDialectNamespace();
  });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();

  // Walks over all the FuncOps in spirv::ModuleOp to lower the entry point
  // attributes.
  OpBuilder builder(context);
  SmallVector<spirv::FuncOp, 1> entryPointFns;
  auto entryPointAttrName = spirv::getEntryPointABIAttrName();
  module.walk([&](spirv::FuncOp funcOp) {
    if (funcOp->getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName)) {
      entryPointFns.push_back(funcOp);
    }
  });
  for (auto fn : entryPointFns) {
    if (failed(lowerEntryPointABIAttr(fn, builder))) {
      return signalPassFailure();
    }
  }
}
