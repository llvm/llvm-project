//===- TosaToSPIRVTosa.cpp - TOSA to SPIR-V Graph/TOSA patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert TOSA IR to SPIR-V Graph/TOSA.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "tosa-to-spirv-tosa-pattern"

namespace mlir::tosa {
namespace {

// Allows users to specify descriptor sets and binding ids on the source
// function inputs and outputs. Use a source-side GraphARM attribute because
// `spirv.interface_var_abi` is verified by the SPIR-V dialect before this
// conversion runs, and result attrs are only accepted on `spirv.ARM.Graph`.
constexpr StringLiteral graphARMInterfaceVarABIAttrName =
    "grapharm.interface_var_abi";

void copyFuncAttrsToGraph(func::FuncOp funcOp, func::FuncOpAdaptor adaptor,
                          spirv::GraphARMOp graphOp) {
  for (NamedAttribute attr : adaptor.getAttributes()) {
    StringRef attrName = attr.getName().getValue();
    if (llvm::is_contained({SymbolTable::getSymbolAttrName(),
                            funcOp.getFunctionTypeAttrName().getValue(),
                            funcOp.getArgAttrsAttrName().getValue(),
                            funcOp.getResAttrsAttrName().getValue(),
                            graphOp.getEntryPointAttrName().getValue()},
                           attrName))
      continue;

    graphOp->setAttr(attr.getName(), attr.getValue());
  }
}

struct FuncGraphConvert final : OpConversionPattern<func::FuncOp> {
  FuncGraphConvert(SPIRVTypeConverter &typeConverter, MLIRContext *context,
                   spirv::TargetEnvAttr targetAttr)
      : OpConversionPattern<func::FuncOp>(typeConverter, context),
        targetAttr(targetAttr) {}

private:
  spirv::TargetEnvAttr targetAttr;

  // Prefer an explicit source-side GraphARM ABI annotation, then preserve an
  // already-canonical SPIR-V ABI annotation, and otherwise synthesize the
  // default descriptor set and binding id.
  void normalizeInterfaceVarABIAttr(spirv::GraphARMOp graphOp,
                                    MLIRContext *context, unsigned index,
                                    bool isResult,
                                    uint32_t defaultDescriptorSet,
                                    uint32_t defaultBinding) const {
    auto abiInfo =
        isResult ? graphOp.getResultAttrOfType<spirv::InterfaceVarABIAttr>(
                       index, graphARMInterfaceVarABIAttrName)
                 : graphOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
                       index, graphARMInterfaceVarABIAttrName);

    if (!abiInfo) {
      abiInfo = isResult
                    ? graphOp.getResultAttrOfType<spirv::InterfaceVarABIAttr>(
                          index, spirv::getInterfaceVarABIAttrName())
                    : graphOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
                          index, spirv::getInterfaceVarABIAttrName());
    }

    if (!abiInfo) {
      abiInfo = spirv::InterfaceVarABIAttr::get(
          defaultDescriptorSet, defaultBinding, std::nullopt, context);
    }

    if (isResult) {
      graphOp.setResultAttr(index, spirv::getInterfaceVarABIAttrName(),
                            abiInfo);
      graphOp.removeResultAttr(index, graphARMInterfaceVarABIAttrName);
    } else {
      graphOp.setArgAttr(index, spirv::getInterfaceVarABIAttrName(), abiInfo);
      graphOp.removeArgAttr(index, graphARMInterfaceVarABIAttrName);
    }
  }

  void normalizeInterfaceVarABIAttrs(spirv::GraphARMOp graphOp,
                                     MLIRContext *context, unsigned inputs,
                                     unsigned outputs) const {
    constexpr uint32_t defaultDescriptorSet = 0;
    for (auto argIndex : llvm::seq<unsigned>(0, inputs)) {
      normalizeInterfaceVarABIAttr(graphOp, context, argIndex, false,
                                   defaultDescriptorSet, argIndex);
    }
    for (auto resIndex : llvm::seq<unsigned>(0, outputs)) {
      normalizeInterfaceVarABIAttr(graphOp, context, resIndex, true,
                                   defaultDescriptorSet, resIndex + inputs);
    }
  }

public:
  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, func::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = rewriter.getContext();

    StringRef name = adaptor.getSymName();
    auto spvModule = spirv::ModuleOp::create(
        rewriter, funcOp.getLoc(), spirv::AddressingModel::Logical,
        spirv::MemoryModel::Vulkan, std::nullopt,
        ("_spirv_tosa_" + name).str());
    spvModule->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

    rewriter.setInsertionPoint(spvModule.getBody(), spvModule.begin());

    FunctionType ftype = adaptor.getFunctionType();
    ArrayAttr argAttrs = adaptor.getArgAttrsAttr();
    ArrayAttr resAttrs = adaptor.getResAttrsAttr();

    TypeConverter::SignatureConversion signatureConverter(ftype.getNumInputs());
    if (failed(typeConverter->convertSignatureArgs(ftype.getInputs(),
                                                   signatureConverter))) {
      return funcOp.emitError("failed to convert function argument types");
    }

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(ftype.getResults(),
                                                newResultTypes))) {
      return funcOp.emitError("failed to convert function result types");
    }

    // TOSA graphs cannot contain nested funcs, so the converted GraphARM op is
    // an entry point.
    auto entryPointAttr = BoolAttr::get(context, true);
    auto graphTy = GraphType::get(
        context, signatureConverter.getConvertedTypes(), newResultTypes);
    auto graphOp =
        spirv::GraphARMOp::create(rewriter, funcOp.getLoc(), graphTy, argAttrs,
                                  resAttrs, entryPointAttr, name);
    copyFuncAttrsToGraph(funcOp, adaptor, graphOp);

    rewriter.inlineRegionBefore(funcOp.getBody(), graphOp.getBody(),
                                graphOp.end());
    if (failed(rewriter.convertRegionTypes(
            &graphOp.getBody(), *getTypeConverter(), &signatureConverter))) {
      return funcOp.emitError("failed to convert function regions");
    }

    normalizeInterfaceVarABIAttrs(graphOp, context, ftype.getNumInputs(),
                                  ftype.getNumResults());

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Converts func.return to spirv.ARM.GraphOutputs.
struct ReturnGraphOutputConvert final : OpConversionPattern<func::ReturnOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::GraphOutputsARMOp>(
        returnOp, adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateTosaToSPIRVTosaConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns,
    spirv::TargetEnvAttr targetAttr) {
  patterns.add<FuncGraphConvert>(typeConverter, patterns.getContext(),
                                 targetAttr);
  patterns.add<ReturnGraphOutputConvert>(typeConverter, patterns.getContext());
}

} // namespace mlir::tosa
