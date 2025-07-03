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

template <typename SourceOp, typename TargetOp, typename AttrType,
          typename ValType, ValType flag>
struct ComparisonOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpRes =
        rewriter
            .create<TargetOp>(srcOp.getLoc(), rewriter.getI1Type(),
                              AttrType::get(rewriter.getContext(), flag),
                              adaptor.getLhs(), adaptor.getRhs())
            .getResult();
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(srcOp, rewriter.getI32Type(),
                                                cmpRes);

    return success();
  }
};

template <typename SourceOp, arith::CmpFPredicate compFlag>
using FPComparisonConversion =
    ComparisonOpConversion<SourceOp, arith::CmpFOp, arith::CmpFPredicateAttr,
                           arith::CmpFPredicate, compFlag>;

template <typename SourceOp, arith::CmpIPredicate compFlag>
using IntComparisonConversion =
    ComparisonOpConversion<SourceOp, arith::CmpIOp, arith::CmpIPredicateAttr,
                           arith::CmpIPredicate, compFlag>;

using WasmLtSIOpConversion =
    IntComparisonConversion<LtSIOp, arith::CmpIPredicate::slt>;
using WasmLeSIOpConversion =
    IntComparisonConversion<LeSIOp, arith::CmpIPredicate::sle>;
using WasmGtSIOpConversion =
    IntComparisonConversion<GtSIOp, arith::CmpIPredicate::sgt>;
using WasmGeSIOpConversion =
    IntComparisonConversion<GeSIOp, arith::CmpIPredicate::sge>;
using WasmLtUIOpConversion =
    IntComparisonConversion<LtUIOp, arith::CmpIPredicate::ult>;
using WasmLeUIOpConversion =
    IntComparisonConversion<LeUIOp, arith::CmpIPredicate::ule>;
using WasmGtUIOpConversion =
    IntComparisonConversion<GtUIOp, arith::CmpIPredicate::ugt>;
using WasmGeUIOpConversion =
    IntComparisonConversion<GeUIOp, arith::CmpIPredicate::uge>;
using WasmLtOpConversion =
    FPComparisonConversion<LtOp, arith::CmpFPredicate::OLT>;
using WasmLeOpConversion =
    FPComparisonConversion<LeOp, arith::CmpFPredicate::OLE>;
using WasmGtOpConversion =
    FPComparisonConversion<GtOp, arith::CmpFPredicate::OGT>;
using WasmGeOpConversion =
    FPComparisonConversion<GeOp, arith::CmpFPredicate::OGE>;

template <typename SourceOp, arith::CmpIPredicate IntFlag,
          arith::CmpFPredicate FloatFlag>
struct IntFpComparisonOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value comparisonResult;
    if (srcOp.getLhs().getType().isInteger())
      comparisonResult =
          rewriter
              .create<arith::CmpIOp>(
                  srcOp.getLoc(), rewriter.getI1Type(),
                  arith::CmpIPredicateAttr::get(rewriter.getContext(), IntFlag),
                  adaptor.getLhs(), adaptor.getRhs())
              .getResult();
    else if (srcOp.getLhs().getType().isFloat())
      comparisonResult =
          rewriter
              .create<arith::CmpFOp>(srcOp.getLoc(), rewriter.getI1Type(),
                                     arith::CmpFPredicateAttr::get(
                                         rewriter.getContext(), FloatFlag),
                                     adaptor.getLhs(), adaptor.getRhs())
              .getResult();
    else
      return rewriter.notifyMatchFailure(
          srcOp.getLoc(), "Unsupported datatype for comparison OP.");

    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(srcOp, rewriter.getI32Type(),
                                                comparisonResult);
    return success();
  }
};

using WasmEqOpConversion =
    IntFpComparisonOpConversion<EqOp, arith::CmpIPredicate::eq,
                                arith::CmpFPredicate::OEQ>;
using WasmNeOpConversion =
    IntFpComparisonOpConversion<NeOp, arith::CmpIPredicate::ne,
                                arith::CmpFPredicate::ONE>;

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

struct WasmEqzOpConversion : OpConversionPattern<EqzOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EqzOp eqzOp, EqzOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = eqzOp->getLoc();
    auto zero =
        rewriter
            .create<arith::ConstantOp>(
                loc, rewriter.getIntegerAttr(adaptor.getInput().getType(), 0))
            .getResult();
    auto cmpRes = rewriter
                      .create<arith::CmpIOp>(
                          loc, rewriter.getI1Type(),
                          arith::CmpIPredicateAttr::get(
                              rewriter.getContext(), arith::CmpIPredicate::eq),
                          adaptor.getInput(), zero)
                      .getResult();
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(eqzOp, rewriter.getI32Type(),
                                                cmpRes);

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
    auto newFunc =
        func::FuncOp::create(rewriter, funcOp->getLoc(), funcOp.getSymName(),
                             funcOp.getFunctionType());
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
    ReturnOp rop = globalOp.getInitTerminator();

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
    auto globalInitializer =
        func::FuncOp::create(rewriter, loc, initializerName,
                             FunctionType::get(getContext(), {}, {}));
    globalInitializer->setAttr(rewriter.getStringAttr("initializer"),
                               rewriter.getUnitAttr());
    auto *initializerBody = globalInitializer.addEntryBlock();
    auto sip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initializerBody);
    auto srcGlobalPtr = memref::GetGlobalOp::create(
        rewriter, loc, MemRefType::get({1}, constInit.getType()),
        constInit.getGlobal());
    auto destGlobalPtr =
        memref::GetGlobalOp::create(rewriter, loc, globalReplacement.getType(),
                                    globalReplacement.getSymName());
    auto idx = arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
    auto loadSrc =
        memref::LoadOp::create(rewriter, loc, srcGlobalPtr, ValueRange{idx});
    memref::StoreOp::create(rewriter, loc, loadSrc.getResult(),
                            destGlobalPtr.getResult(), ValueRange{idx});
    func::ReturnOp::create(rewriter, loc);
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
    auto initializer = arith::ConstantOp::create(
        rewriter, localOp->getLoc(),
        getInitializerAttr(localOp.getResult().getType().getElementType()));
    memref::StoreOp::create(rewriter, localOp->getLoc(),
                            initializer.getResult(), alloca.getResult());
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
    memref::StoreOp::create(rewriter, localTeeOp->getLoc(), adaptor.getValue(),
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
      auto localVar = memref::AllocaOp::create(builder, loc, destType);
      memref::StoreOp::create(builder, loc, values.front(),
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
           WasmEqOpConversion,
           WasmEqzOpConversion,
           WasmExtendSOpConversion,
           WasmExtendUOpConversion,
           WasmFloorOpConversion,
           WasmFuncImportOpConversion,
           WasmFuncOpConversion,
           WasmGeOpConversion,
           WasmGeSIOpConversion,
           WasmGeUIOpConversion,
           WasmGlobalImportOpConverter,
           WasmGlobalWithConstInitConversion,
           WasmGlobalWithGetGlobalInitConversion,
           WasmGtOpConversion,
           WasmGtSIOpConversion,
           WasmGtUIOpConversion,
           WasmLeOpConversion,
           WasmLeSIOpConversion,
           WasmLeUIOpConversion,
           WasmLocalConversion,
           WasmLocalGetConversion,
           WasmLocalSetConversion,
           WasmLocalTeeConversion,
           WasmLtOpConversion,
           WasmLtSIOpConversion,
           WasmLtUIOpConversion,
           WasmMaxOpConversion,
           WasmMinOpConversion,
           WasmMulOpConversion,
           WasmNeOpConversion,
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

std::unique_ptr<Pass> createRaiseWasmMLIRPass() {
  return std::make_unique<RaiseWasmMLIRPass>();
}
