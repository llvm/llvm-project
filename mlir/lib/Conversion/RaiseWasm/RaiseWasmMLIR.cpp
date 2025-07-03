//===- RaiseWasmMLIR.cpp - Convert Wasm to less abstract dialects ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of wasm operations to standard dialects ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RaiseWasm/RaiseWasmMLIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>

#define DEBUG_TYPE "wasm-convert"

namespace mlir {
#define GEN_PASS_DEF_RAISEWASMMLIR
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::wasmssa;
namespace {

template <typename SourceOp, typename TargetIntOp, typename TargetFPOp>
struct IntFPDispatchMappingConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = srcOp.getRhs().getType();
    if (type.isInteger()) {
      rewriter.replaceOpWithNewOp<TargetIntOp>(srcOp, srcOp->getResultTypes(),
                                               adaptor.getOperands());
      return success();
    }
    if (!type.isFloat())
      return failure();
    rewriter.replaceOpWithNewOp<TargetFPOp>(srcOp, srcOp->getResultTypes(),
                                            adaptor.getOperands());
    return success();
  }
};

using WasmAddOpConversion =
    IntFPDispatchMappingConversion<AddOp, arith::AddIOp, arith::AddFOp>;
using WasmMulOpConversion =
    IntFPDispatchMappingConversion<MulOp, arith::MulIOp, arith::MulFOp>;
using WasmSubOpConversion =
    IntFPDispatchMappingConversion<SubOp, arith::SubIOp, arith::SubFOp>;

/// Convert a k-ary source operation \p SourceOp into an operation \p TargetOp.
/// Both \p SourceOp and \p TargetOp must have the same number of operands.
template <typename SourceOp, typename TargetOp>
struct OpMappingConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(srcOp, srcOp->getResultTypes(),
                                          adaptor.getOperands());
    return success();
  }
};

using WasmAndOpConversion = OpMappingConversion<AndOp, arith::AndIOp>;
using WasmCeilOpConversion = OpMappingConversion<CeilOp, math::CeilOp>;
/// TODO: SIToFP and UIToFP don't allow specification of the floating point
/// rounding mode
using WasmConvertSOpConversion =
    OpMappingConversion<ConvertSOp, arith::SIToFPOp>;
using WasmConvertUOpConversion =
    OpMappingConversion<ConvertUOp, arith::UIToFPOp>;
using WasmDemoteOpConversion = OpMappingConversion<DemoteOp, arith::TruncFOp>;
using WasmDivFPOpConversion = OpMappingConversion<DivOp, arith::DivFOp>;
using WasmDivSIOpConversion = OpMappingConversion<DivSIOp, arith::DivSIOp>;
using WasmDivUIOpConversion = OpMappingConversion<DivUIOp, arith::DivUIOp>;
using WasmExtendSOpConversion =
    OpMappingConversion<ExtendSI32Op, arith::ExtSIOp>;
using WasmExtendUOpConversion =
    OpMappingConversion<ExtendUI32Op, arith::ExtUIOp>;
using WasmFloorOpConversion = OpMappingConversion<FloorOp, math::FloorOp>;
using WasmMaxOpConversion = OpMappingConversion<MaxOp, arith::MaximumFOp>;
using WasmMinOpConversion = OpMappingConversion<MinOp, arith::MinimumFOp>;
using WasmOrOpConversion = OpMappingConversion<OrOp, arith::OrIOp>;
using WasmPromoteOpConversion = OpMappingConversion<PromoteOp, arith::ExtFOp>;
using WasmRemSIOpConversion = OpMappingConversion<RemSIOp, arith::RemSIOp>;
using WasmRemUIOpConversion = OpMappingConversion<RemUIOp, arith::RemUIOp>;
using WasmReinterpretOpConversion =
    OpMappingConversion<ReinterpretOp, arith::BitcastOp>;
using WasmShLOpConversion = OpMappingConversion<ShLOp, arith::ShLIOp>;
using WasmShRSOpConversion = OpMappingConversion<ShRSOp, arith::ShRSIOp>;
using WasmShRUOpConversion = OpMappingConversion<ShRUOp, arith::ShRUIOp>;
using WasmXOrOpConversion = OpMappingConversion<XOrOp, arith::XOrIOp>;
using WasmNegOpConversion = OpMappingConversion<NegOp, arith::NegFOp>;
using WasmCopySignOpConversion =
    OpMappingConversion<CopySignOp, math::CopySignOp>;
using WasmClzOpConversion =
    OpMappingConversion<ClzOp, math::CountLeadingZerosOp>;
using WasmCtzOpConversion =
    OpMappingConversion<CtzOp, math::CountTrailingZerosOp>;
using WasmPopCntOpConversion = OpMappingConversion<PopCntOp, math::CtPopOp>;
using WasmAbsOpConversion = OpMappingConversion<AbsOp, math::AbsFOp>;
using WasmTruncOpConversion = OpMappingConversion<TruncOp, math::TruncOp>;
using WasmSqrtOpConversion = OpMappingConversion<SqrtOp, math::SqrtOp>;
using WasmWrapOpConversion = OpMappingConversion<WrapOp, arith::TruncIOp>;

struct WasmCallOpConversion : OpConversionPattern<FuncCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncCallOp funcCallOp, FuncCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        funcCallOp, funcCallOp.getCallee(), funcCallOp.getResults().getTypes(),
        funcCallOp.getOperands());
    return success();
  }
};

struct WasmConstOpConversion : OpConversionPattern<ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstOp constOp, ConstOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, constOp.getValue());
    return success();
  }
};

struct WasmFuncImportOpConversion : OpConversionPattern<FuncImportOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncImportOp funcImportOp, FuncImportOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nFunc = rewriter.replaceOpWithNewOp<func::FuncOp>(
        funcImportOp, funcImportOp.getSymName(), funcImportOp.getType());
    nFunc.setVisibility(SymbolTable::Visibility::Private);
    return success();
  }
};

struct WasmFuncOpConversion : OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp funcOp, FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFunc = rewriter.create<func::FuncOp>(
        funcOp->getLoc(), funcOp.getSymName(), funcOp.getFunctionType());
    rewriter.cloneRegionBefore(funcOp.getBody(), newFunc.getBody(),
                               newFunc.getBody().end());
    Block *oldEntryBlock = &newFunc.getBody().front();
    auto blockArgTypes = oldEntryBlock->getArgumentTypes();
    TypeConverter::SignatureConversion sC{oldEntryBlock->getNumArguments()};
    auto numArgs = blockArgTypes.size();
    for (size_t i = 0; i < numArgs; ++i) {
      auto argType = dyn_cast<LocalRefType>(blockArgTypes[i]);
      if (!argType)
        return failure();
      sC.addInputs(i, argType.getElementType());
    }

    rewriter.applySignatureConversion(oldEntryBlock, sC, getTypeConverter());
    rewriter.replaceOp(funcOp, newFunc);
    return success();
  }
};

struct WasmGlobalImportOpConverter : OpConversionPattern<GlobalImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GlobalImportOp gIOp, GlobalImportOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefGOp = rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        gIOp, gIOp.getSymNameAttr(), rewriter.getStringAttr("nested"),
        TypeAttr::get(MemRefType::get({1}, gIOp.getType())), Attribute{},
        /*constant*/ UnitAttr{},
        /*alignment*/ IntegerAttr{});
    memrefGOp.setConstant(!gIOp.getIsMutable());
    return success();
  }
};

template <typename CRTP, typename OriginOpType>
struct GlobalOpConverter : OpConversionPattern<GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GlobalOp globalOp, GlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReturnOp rop;
    globalOp->walk([&rop](ReturnOp op) { rop = op; });

    if (rop->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(
          globalOp, "globalOp initializer should return one value exactly");

    auto initializerOp =
        dyn_cast<OriginOpType>(rop->getOperand(0).getDefiningOp());

    if (!initializerOp)
      return rewriter.notifyMatchFailure(
          globalOp, "invalid initializer op type for this pattern");

    return static_cast<CRTP const *>(this)->handleInitializer(
        globalOp, rewriter, initializerOp);
  }
};

struct WasmGlobalWithConstInitConversion
    : GlobalOpConverter<WasmGlobalWithConstInitConversion, ConstOp> {
  using GlobalOpConverter::GlobalOpConverter;
  LogicalResult handleInitializer(GlobalOp globalOp,
                                  ConversionPatternRewriter &rewriter,
                                  ConstOp constInit) const {
    auto initializer =
        DenseElementsAttr::get(RankedTensorType::get({1}, globalOp.getType()),
                               ArrayRef<Attribute>{constInit.getValueAttr()});
    auto globalReplacement = rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        globalOp, globalOp.getSymNameAttr(), rewriter.getStringAttr("private"),
        TypeAttr::get(MemRefType::get({1}, globalOp.getType())), initializer,
        /*constant*/ UnitAttr{},
        /*alignment*/ IntegerAttr{});
    globalReplacement.setConstant(!globalOp.getIsMutable());
    return success();
  }
};

struct WasmGlobalWithGetGlobalInitConversion
    : GlobalOpConverter<WasmGlobalWithGetGlobalInitConversion, GlobalGetOp> {
  using GlobalOpConverter::GlobalOpConverter;
  LogicalResult handleInitializer(GlobalOp globalOp,
                                  ConversionPatternRewriter &rewriter,
                                  GlobalGetOp constInit) const {
    auto globalReplacement = rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        globalOp, globalOp.getSymNameAttr(), rewriter.getStringAttr("private"),
        TypeAttr::get(MemRefType::get({1}, globalOp.getType())),
        rewriter.getUnitAttr(),
        /*constant*/ UnitAttr{},
        /*alignment*/ IntegerAttr{});
    globalReplacement.setConstant(!globalOp.getIsMutable());
    auto loc = globalOp.getLoc();
    auto initializerName = (globalOp.getSymName() + "::initializer").str();
    auto globalInitializer = rewriter.create<func::FuncOp>(
        loc, initializerName, FunctionType::get(getContext(), {}, {}));
    globalInitializer->setAttr(rewriter.getStringAttr("initializer"),
                               rewriter.getUnitAttr());
    auto *initializerBody = globalInitializer.addEntryBlock();
    auto sip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initializerBody);
    auto srcGlobalPtr = rewriter.create<memref::GetGlobalOp>(
        loc, MemRefType::get({1}, constInit.getType()), constInit.getGlobal());
    auto destGlobalPtr = rewriter.create<memref::GetGlobalOp>(
        loc, globalReplacement.getType(), globalReplacement.getSymName());
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    auto loadSrc =
        rewriter.create<memref::LoadOp>(loc, srcGlobalPtr, ValueRange{idx});
    rewriter.create<memref::StoreOp>(
        loc, loadSrc.getResult(), destGlobalPtr.getResult(), ValueRange{idx});
    rewriter.create<func::ReturnOp>(loc);
    rewriter.restoreInsertionPoint(sip);
    return success();
  }
};

inline TypedAttr getInitializerAttr(Type t) {
  assert(t.isIntOrFloat() &&
         "This helper is intended to use with int and float types");
  if (t.isInteger())
    return IntegerAttr::get(t, 0);
  if (t.isFloat())
    return FloatAttr::get(t, 0.);
  return TypedAttr{};
}

struct WasmLocalConversion : OpConversionPattern<LocalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LocalOp localOp, LocalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto alloca = rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        localOp,
        MemRefType::get({}, localOp.getResult().getType().getElementType()));
    auto initializer = rewriter.create<arith::ConstantOp>(
        localOp->getLoc(),
        getInitializerAttr(localOp.getResult().getType().getElementType()));
    rewriter.create<memref::StoreOp>(localOp->getLoc(), initializer.getResult(),
                                     alloca.getResult());
    return success();
  }
};

struct WasmLocalGetConversion : OpConversionPattern<LocalGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LocalGetOp localGetOp, LocalGetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        localGetOp, localGetOp.getResult().getType(), adaptor.getLocalVar(),
        ValueRange{});
    return success();
  }
};

struct WasmLocalSetConversion : OpConversionPattern<LocalSetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LocalSetOp localSetOp, LocalSetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        localSetOp, adaptor.getValue(), adaptor.getLocalVar(), ValueRange{});
    return success();
  }
};

struct WasmLocalTeeConversion : OpConversionPattern<LocalTeeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LocalTeeOp localTeeOp, LocalTeeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<memref::StoreOp>(localTeeOp->getLoc(), adaptor.getValue(),
                                     adaptor.getLocalVar());
    rewriter.replaceOp(localTeeOp, adaptor.getValue());
    return success();
  }
};

struct WasmReturnOpConversion : OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp,
                                                adaptor.getOperands());
    return success();
  }
};

struct RaiseWasmMLIRPass : public impl::RaiseWasmMLIRBase<RaiseWasmMLIRPass> {
  void runOnOperation() override {
    ConversionTarget target{getContext()};
    target.addIllegalDialect<WasmSSADialect>();
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect,
                           cf::ControlFlowDialect, func::FuncDialect,
                           memref::MemRefDialect, math::MathDialect>();
    RewritePatternSet patterns(&getContext());
    TypeConverter tc{};
    tc.addConversion([](Type type) -> std::optional<Type> { return type; });
    tc.addConversion([](LocalRefType type) -> std::optional<Type> {
      return MemRefType::get({}, type.getElementType());
    });
    tc.addTargetMaterialization([](OpBuilder &builder, MemRefType destType,
                                   ValueRange values, Location loc) -> Value {
      if (values.size() != 1 ||
          values.front().getType() != destType.getElementType())
        return {};
      auto localVar = builder.create<memref::AllocaOp>(loc, destType);
      builder.create<memref::StoreOp>(loc, values.front(),
                                      localVar.getResult());
      return localVar.getResult();
    });
    populateRaiseWasmMLIRConversionPatterns(tc, patterns);

    llvm::DenseMap<StringAttr, StringAttr> idxSymToImportSym{};
    auto *topOp = getOperation();
    topOp->walk([&idxSymToImportSym, this](ImportOpInterface importOp) {
      auto const qualifiedImportName = importOp.getQualifiedImportName();
      auto qualNameAttr = StringAttr::get(&getContext(), qualifiedImportName);
      idxSymToImportSym.insert(
          std::make_pair(importOp.getSymbolName(), qualNameAttr));
    });

    if (failed(applyFullConversion(topOp, target, std::move(patterns))))
      return signalPassFailure();

    auto symTable = SymbolTable{topOp};
    for (auto &[oldName, newName] : idxSymToImportSym) {
      if (failed(symTable.rename(oldName, newName)))
        return signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateRaiseWasmMLIRConversionPatterns(
    TypeConverter &tc, RewritePatternSet &patternSet) {
  auto *ctx = patternSet.getContext();
  // Disable clang-format in patternSet for readability + small diffs.
  // clang-format off
  patternSet
      .add<
           WasmAbsOpConversion,
           WasmAddOpConversion,
           WasmAndOpConversion,
           WasmCallOpConversion,
           WasmCeilOpConversion,
           WasmClzOpConversion,
           WasmConstOpConversion,
           WasmConvertSOpConversion,
           WasmConvertUOpConversion,
           WasmCopySignOpConversion,
           WasmCtzOpConversion,
           WasmDemoteOpConversion,
           WasmDivFPOpConversion,
           WasmDivSIOpConversion,
           WasmDivUIOpConversion,
           WasmExtendSOpConversion,
           WasmExtendUOpConversion,
           WasmFloorOpConversion,
           WasmFuncImportOpConversion,
           WasmFuncOpConversion,
           WasmGlobalImportOpConverter,
           WasmGlobalWithConstInitConversion,
           WasmGlobalWithGetGlobalInitConversion,
           WasmLocalConversion,
           WasmLocalGetConversion,
           WasmLocalSetConversion,
           WasmLocalTeeConversion,
           WasmMaxOpConversion,
           WasmMinOpConversion,
           WasmMulOpConversion,
           WasmNegOpConversion,
           WasmOrOpConversion,
           WasmPopCntOpConversion,
           WasmPromoteOpConversion,
           WasmReinterpretOpConversion,
           WasmRemSIOpConversion,
           WasmRemUIOpConversion,
           WasmReturnOpConversion,
           WasmShLOpConversion,
           WasmShRSOpConversion,
           WasmShRUOpConversion,
           WasmSqrtOpConversion,
           WasmSubOpConversion,
           WasmTruncOpConversion,
           WasmWrapOpConversion,
           WasmXOrOpConversion
           >(tc, ctx);
  // clang-format on
}

std::unique_ptr<Pass> mlir::createRaiseWasmMLIRPass() {
  return std::make_unique<RaiseWasmMLIRPass>();
}
