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

#include "PassDetail.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

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
  // info create a variable of type !spv.ptr<!spv.struct<elementType>>. If not
  // it must already be a !spv.ptr<!spv.struct<...>>.
  auto varType = funcOp.getType().getInput(argIndex);
  if (varType.cast<spirv::SPIRVType>().isScalarOrVector()) {
    auto storageClass = abiInfo.getStorageClass();
    if (!storageClass)
      return nullptr;
    varType =
        spirv::PointerType::get(spirv::StructType::get(varType), *storageClass);
  }
  auto varPtrType = varType.cast<spirv::PointerType>();
  auto varPointeeType = varPtrType.getPointeeType().cast<spirv::StructType>();

  // Set the offset information.
  varPointeeType =
      VulkanLayoutUtils::decorateType(varPointeeType).cast<spirv::StructType>();

  if (!varPointeeType)
    return nullptr;

  varType =
      spirv::PointerType::get(varPointeeType, varPtrType.getStorageClass());

  return builder.create<spirv::GlobalVariableOp>(
      funcOp.getLoc(), varType, varName, abiInfo.getDescriptorSet(),
      abiInfo.getBinding());
}

/// Gets the global variables that need to be specified as interface variable
/// with an spv.EntryPointOp. Traverses the body of a entry function to do so.
static LogicalResult
getInterfaceVariables(spirv::FuncOp funcOp,
                      SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp->getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  llvm::SetVector<Operation *> interfaceVarSet;

  // TODO: This should in reality traverse the entry function
  // call graph and collect all the interfaces. For now, just traverse the
  // instructions in this function.
  funcOp.walk([&](spirv::AddressOfOp addressOfOp) {
    auto var =
        module.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.variable());
    // TODO: Per SPIR-V spec: "Before version 1.4, the interface’s
    // storage classes are limited to the Input and Output storage classes.
    // Starting with version 1.4, the interface’s storage classes are all
    // storage classes used in declaring all global variables referenced by the
    // entry point’s call tree." We should consider the target environment here.
    switch (var.type().cast<spirv::PointerType>().getStorageClass()) {
    case spirv::StorageClass::Input:
    case spirv::StorageClass::Output:
      interfaceVarSet.insert(var.getOperation());
      break;
    default:
      break;
    }
  });
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
        cast<spirv::GlobalVariableOp>(var).sym_name(), funcOp.getContext()));
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

  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  auto spirvModule = funcOp->getParentOfType<spirv::ModuleOp>();
  builder.setInsertionPoint(spirvModule.body().front().getTerminator());

  // Adds the spv.EntryPointOp after collecting all the interface variables
  // needed.
  SmallVector<Attribute, 1> interfaceVars;
  if (failed(getInterfaceVariables(funcOp, interfaceVars))) {
    return failure();
  }

  spirv::TargetEnvAttr targetEnv = spirv::lookupTargetEnv(funcOp);
  FailureOr<spirv::ExecutionModel> executionModel =
      spirv::getExecutionModel(targetEnv);
  if (failed(executionModel))
    return funcOp.emitRemark("lower entry point failure: could not select "
                             "execution model based on 'spv.target_env'");

  builder.create<spirv::EntryPointOp>(
      funcOp.getLoc(), executionModel.getValue(), funcOp, interfaceVars);

  // Specifies the spv.ExecutionModeOp.
  auto localSizeAttr = entryPointAttr.local_size();
  SmallVector<int32_t, 3> localSize(localSizeAttr.getValues<int32_t>());
  builder.create<spirv::ExecutionModeOp>(
      funcOp.getLoc(), funcOp, spirv::ExecutionMode::LocalSize, localSize);
  funcOp.removeAttr(entryPointAttrName);
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
  matchAndRewrite(spirv::FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pass to implement the ABI information specified as attributes.
class LowerABIAttributesPass final
    : public SPIRVLowerABIAttributesBase<LowerABIAttributesPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult ProcessInterfaceVarABI::matchAndRewrite(
    spirv::FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!funcOp->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName())) {
    // TODO: Non-entry point functions are not handled.
    return failure();
  }
  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getType().getNumInputs());

  auto attrName = spirv::getInterfaceVarABIAttrName();
  for (auto argType : llvm::enumerate(funcOp.getType().getInputs())) {
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

    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.front());
    // Insert spirv::AddressOf and spirv::AccessChain operations.
    Value replacement =
        rewriter.create<spirv::AddressOfOp>(funcOp.getLoc(), var);
    // Check if the arg is a scalar or vector type. In that case, the value
    // needs to be loaded into registers.
    // TODO: This is loading value of the scalar into registers
    // at the start of the function. It is probably better to do the load just
    // before the use. There might be multiple loads and currently there is no
    // easy way to replace all uses with a sequence of operations.
    if (argType.value().cast<spirv::SPIRVType>().isScalarOrVector()) {
      auto indexType = SPIRVTypeConverter::getIndexType(funcOp.getContext());
      auto zero =
          spirv::ConstantOp::getZero(indexType, funcOp.getLoc(), rewriter);
      auto loadPtr = rewriter.create<spirv::AccessChainOp>(
          funcOp.getLoc(), replacement, zero.constant());
      replacement = rewriter.create<spirv::LoadOp>(funcOp.getLoc(), loadPtr);
    }
    signatureConverter.remapInput(argType.index(), replacement);
  }
  if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(),
                                         &signatureConverter)))
    return failure();

  // Creates a new function with the update signature.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), llvm::None));
  });
  return success();
}

void LowerABIAttributesPass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  spirv::ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  spirv::TargetEnv targetEnv(spirv::lookupTargetEnv(module));

  SPIRVTypeConverter typeConverter(targetEnv);

  // Insert a bitcast in the case of a pointer type change.
  typeConverter.addSourceMaterialization([](OpBuilder &builder,
                                            spirv::PointerType type,
                                            ValueRange inputs, Location loc) {
    if (inputs.size() != 1 || !inputs[0].getType().isa<spirv::PointerType>())
      return Value();
    return builder.create<spirv::BitcastOp>(loc, type, inputs[0]).getResult();
  });

  OwningRewritePatternList patterns;
  patterns.insert<ProcessInterfaceVarABI>(typeConverter, context);

  ConversionTarget target(*context);
  // "Legal" function ops should have no interface variable ABI attributes.
  target.addDynamicallyLegalOp<spirv::FuncOp>([&](spirv::FuncOp op) {
    StringRef attrName = spirv::getInterfaceVarABIAttrName();
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i)
      if (op.getArgAttr(i, attrName))
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

std::unique_ptr<OperationPass<spirv::ModuleOp>>
mlir::spirv::createLowerABIAttributesPass() {
  return std::make_unique<LowerABIAttributesPass>();
}
