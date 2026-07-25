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
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"
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

/// Lower a rotate to a series of bitwise operations. Intended for us
/// in dialects that do not natively support rotate operations.
///
/// Result stays in the wasm dialect. It will then subsequently be lowered to
/// the target dialect.
///
/// The rotate will be lowered to a pattern like so:
///
/// (val LHSShiftOp (bits & (width-1))) | (val RHSShiftOp (-bits & (width-1)))
///
/// Where LHSShiftOp and RHSShiftOp are shift operations. Concretely,
///
/// rotr = (val >> (bits & (width - 1))) | (val << (-bits & (width - 1)))
/// rotl = (val << (bits & (width - 1))) | (val >> (-bits & (width - 1)))
///
/// Using this variant ensures that our rotate is defined in the target dialect.
///
/// \p SourceOp - Rotate operation to replace.
/// \p LHSShiftOp - Shift operation to use on the left-hand side of the OR.
/// \p RHSShiftOp - Shift operation to use on the right-hand side of the OR.
template <typename SourceOp, typename LHSShiftOp, typename RHSShiftOp>
struct RotateOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Type ty = srcOp->getResultTypes()[0];
    const Location loc = srcOp->getLoc();
    const Value val = adaptor.getVal();
    const Value bits = adaptor.getBits();
    const unsigned width = ty.getIntOrFloatBitWidth();

    // Materialize (width - 1) for use in both sides of the expression.
    auto cstWidthMinusOne =
        ConstOp::create(rewriter, loc, IntegerAttr::get(ty, width - 1));

    // Form the left-hand side of the OR:
    // (val (lhs shift op) (bits & (width - 1)))
    auto orLHS = LHSShiftOp::create(
        rewriter, loc, val,
        AndOp::create(rewriter, loc, bits, cstWidthMinusOne));

    // Form the right-hand side of the OR:
    // (val (rhs shift op) (-bits & (width - 1)))
    auto orRHS = RHSShiftOp::create(
        rewriter, loc, val,
        // (-bits & (width - 1))
        AndOp::create(rewriter, loc,
                      // 0 - bits == -bits
                      SubOp::create(rewriter, loc,
                                    ConstOp::create(rewriter, loc,
                                                    IntegerAttr::get(ty, 0)),
                                    bits),
                      cstWidthMinusOne));

    // OR together the two shifts and replace the rotate with the new
    // expression.
    rewriter.replaceOpWithNewOp<OrOp>(srcOp, orLHS, orRHS);
    return success();
  }
};

using WasmRotrOpConversion = RotateOpConversion<RotrOp, ShRUOp, ShLOp>;
using WasmRotlOpConversion = RotateOpConversion<RotlOp, ShLOp, ShRUOp>;

template <typename SourceOp, typename TargetOp, typename AttrType,
          typename ValType, ValType flag>
struct ComparisonOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp srcOp, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpRes =
        TargetOp::create(rewriter, srcOp.getLoc(), rewriter.getI1Type(),
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
          arith::CmpIOp::create(
              rewriter, srcOp.getLoc(), rewriter.getI1Type(),
              arith::CmpIPredicateAttr::get(rewriter.getContext(), IntFlag),
              adaptor.getLhs(), adaptor.getRhs())
              .getResult();
    else if (srcOp.getLhs().getType().isFloat())
      comparisonResult =
          arith::CmpFOp::create(
              rewriter, srcOp.getLoc(), rewriter.getI1Type(),
              arith::CmpFPredicateAttr::get(rewriter.getContext(), FloatFlag),
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
    auto zero = arith::ConstantOp::create(
                    rewriter, loc,
                    rewriter.getIntegerAttr(adaptor.getInput().getType(), 0))
                    .getResult();
    auto cmpRes = arith::CmpIOp::create(
                      rewriter, loc, rewriter.getI1Type(),
                      arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                                    arith::CmpIPredicate::eq),
                      adaptor.getInput(), zero)
                      .getResult();
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(eqzOp, rewriter.getI32Type(),
                                                cmpRes);

    return success();
  }
};

struct WasmExtendLowBitsOpConversion : OpConversionPattern<ExtendLowBitsSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtendLowBitsSOp extendLowBytesSOp,
                  ExtendLowBitsSOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto truncWidth = extendLowBytesSOp.getBitsToTake().getInt();
    auto truncation = arith::TruncIOp::create(
        rewriter, extendLowBytesSOp->getLoc(),
        rewriter.getIntegerType(truncWidth), adaptor.getInput());
    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
        extendLowBytesSOp, extendLowBytesSOp.getResult().getType(),
        truncation.getResult());
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
  ///
  /// Control flow conversion needs shared state for tracking which block
  /// corresponds to which operation at which level.
  ///
  /// This class handles such tracking and performs the conversion of control
  /// flow related ops contained in a function.
  class CFRewriterVisitor {
  private:
    using branch_to_dest_t = llvm::DenseMap<LabelBranchingOpInterface, Block *>;
    Value getCompResultAsI1(Value compResult,
                            ConversionPatternRewriter &rewriter) {
      auto testValue = arith::ConstantOp::create(rewriter, compResult.getLoc(),
                                                 rewriter.getI32IntegerAttr(0));
      auto flag = arith::CmpIOp::create(
                      rewriter, compResult.getLoc(), rewriter.getIntegerType(1),
                      arith::CmpIPredicate::ne, compResult, testValue)
                      .getResult();
      return flag;
    }

    void replaceNestLevelWithBranch(BlockOp blockOp,
                                    llvm::ArrayRef<Block *> regionsToEntry,
                                    ConversionPatternRewriter &rewriter) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(blockOp, regionsToEntry[0],
                                                blockOp->getOperands());
    }

    void replaceNestLevelWithBranch(LoopOp loopOp,
                                    llvm::ArrayRef<Block *> regionsToEntry,
                                    ConversionPatternRewriter &rewriter) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(loopOp, regionsToEntry[0],
                                                loopOp->getOperands());
    }

    void replaceNestLevelWithBranch(IfOp ifOp,
                                    llvm::ArrayRef<Block *> regionsToEntry,
                                    ConversionPatternRewriter &rewriter) {
      Block *falseDest =
          regionsToEntry.size() == 2 ? regionsToEntry[1] : ifOp.getTarget();
      auto flag = getCompResultAsI1(ifOp.getCondition(), rewriter);
      rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
          ifOp, flag, regionsToEntry[0], ifOp.getInputs(), falseDest,
          ifOp.getInputs());
    }

    template <typename LevelType>
    LogicalResult
    replaceNestLevelWithBranchWrapper(LabelLevelOpInterface nestingOp,
                                      llvm::ArrayRef<Block *> regionsToEntry,
                                      ConversionPatternRewriter &rewriter) {
      auto cast = dyn_cast<LevelType>(nestingOp.getOperation());
      if (!cast)
        return failure();
      replaceNestLevelWithBranch(cast, regionsToEntry, rewriter);
      return success();
    }

    template <typename... LevelTypes>
    LogicalResult inlineNestDispatcher(LabelLevelOpInterface nestingOp,
                                       ConversionPatternRewriter &rewriter) {
      auto sip = rewriter.saveInsertionPoint();
      Block *blockSuccessor = nestingOp->getSuccessor(0);
      llvm::SmallVector<Block *, 2> regionEntries;
      LLVM_DEBUG(llvm::dbgs()
                     << "Starting inlining blocks for " << nestingOp << "\n";);
      for (auto &region : nestingOp->getRegions()) {
        if (region.empty())
          continue;
        regionEntries.push_back(&region.front());
        /// Inline blocks of nested ops
        llvm::SmallVector<LabelLevelOpInterface> nestedOps{
            region.getOps<LabelLevelOpInterface>()};
        for (auto nestedOp : nestedOps) {
          LLVM_DEBUG(llvm::dbgs() << " Found nested op: " << nestedOp);
          if (failed(inlineBlocks(nestedOp, rewriter)))
            return failure();
        }
        rewriter.inlineRegionBefore(region, blockSuccessor);
      }
      LLVM_DEBUG(llvm::dbgs() << "End of region inlining\n");
      LLVM_DEBUG(llvm::dbgs() << "Replacing initial op with branching\n");
      rewriter.setInsertionPoint(nestingOp);
      auto res = success(
          (... || succeeded(replaceNestLevelWithBranchWrapper<LevelTypes>(
                      nestingOp, regionEntries, rewriter))));
      rewriter.restoreInsertionPoint(sip);
      if (failed(res))
        return emitError(nestingOp->getLoc(),
                         "Unable to inline the operation regions.");
      return success();
    }

    /// Take a nesting level defining op and inline it in the parent region.
    LogicalResult inlineBlocks(LabelLevelOpInterface nestingOp,
                               ConversionPatternRewriter &rewriter) {
      return inlineNestDispatcher<BlockOp, IfOp, LoopOp>(nestingOp, rewriter);
    }

    llvm::FailureOr<Block *> getBlockFor(LabelBranchingOpInterface branchOp) {
      auto destIter = branchToDest.find(branchOp);
      if (destIter == branchToDest.end())
        return branchOp->emitError("No indexed label op for this operation: ")
               << branchOp;
      return destIter->second;
    }

    inline void convertBranch(BranchIfOp brOp, Block *dest,
                              ConversionPatternRewriter &rewriter) {
      auto flag = getCompResultAsI1(brOp.getCondition(), rewriter);
      rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
          brOp, flag, dest, brOp.getInputs(), brOp.getElseSuccessor(),
          ValueRange{});
    }

    inline void convertBranch(BlockReturnOp brOp, Block *dest,
                              ConversionPatternRewriter &rewriter) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(brOp, dest, brOp.getInputs());
    }

    template <typename LevelInterfaceT>
    inline LogicalResult
    convertBranchWrapper(LabelBranchingOpInterface branchOp, Block *dest,
                         ConversionPatternRewriter &rewriter) {
      auto cast = dyn_cast<LevelInterfaceT>(branchOp.getOperation());
      if (!cast)
        return failure();
      auto sip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(branchOp);
      convertBranch(cast, dest, rewriter);
      rewriter.restoreInsertionPoint(sip);
      return success();
    }

    template <typename... BranchInterfaceT>
    LogicalResult convertBranchDispatch(LabelBranchingOpInterface branchOp,
                                        ConversionPatternRewriter &rewriter) {
      auto dest = getBlockFor(branchOp);
      if (failed(dest))
        return failure();
      auto res =
          success((... || succeeded(convertBranchWrapper<BranchInterfaceT>(
                              branchOp, *dest, rewriter))));
      if (failed(res))
        return emitError(branchOp->getLoc(), "No known converter for op ")
               << branchOp;
      return res;
    }

    LogicalResult convertBranch(LabelBranchingOpInterface branchOp,
                                ConversionPatternRewriter &rewriter) {
      return convertBranchDispatch<BlockReturnOp, BranchIfOp>(branchOp,
                                                              rewriter);
    }

    func::FuncOp func;
    branch_to_dest_t branchToDest;

  public:
    CFRewriterVisitor(func::FuncOp func) : func{func} {
      func.walk([this](LabelBranchingOpInterface branchOp) {
        branchToDest.insert({branchOp, branchOp.getTarget()});
      });
    }
    LogicalResult rewrite(ConversionPatternRewriter &rewriter) {
      llvm::SmallVector<LabelLevelOpInterface> nestingOps{
          func.getOps<LabelLevelOpInterface>()};
      for (auto nestingOp : nestingOps)
        if (failed(inlineBlocks(nestingOp, rewriter)))
          return failure();

      auto res =
          func->walk([this, &rewriter](LabelBranchingOpInterface branchOp) {
            if (failed(convertBranch(branchOp, rewriter)))
              return WalkResult::interrupt();
            return WalkResult::advance();
          });
      return failure(res.wasInterrupted());
    }
  };

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
    CFRewriterVisitor cfRewriter{newFunc};
    return cfRewriter.rewrite(rewriter);
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

struct WasmMemoryOpConversion : OpConversionPattern<MemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemOp memOp, MemOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = memOp.getLoc();
    auto bufferType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    auto bufferPtrType = MemRefType::get({1}, bufferType);
    auto memVisibility = memOp.getVisibility();
    // Convert to StringAttr since memref::GlobalOp expects visibility as a
    // string attribute.
    mlir::StringAttr visAttr;
    if (memVisibility == mlir::SymbolTable::Visibility::Public)
      visAttr = mlir::StringAttr::get(memOp->getContext(), "public");
    else if (memVisibility == mlir::SymbolTable::Visibility::Private)
      visAttr = mlir::StringAttr::get(memOp->getContext(), "private");
    else
      visAttr = mlir::StringAttr::get(memOp->getContext(), "nested");

    auto memPtr = rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        memOp, memOp.getSymNameAttr(), visAttr, TypeAttr::get(bufferPtrType),
        /*initialValue*/ rewriter.getUnitAttr(),
        /*constant*/ UnitAttr{}, /*alignment*/ IntegerAttr{});
    auto initializerName = (memPtr.getSymName() + "::initializer").str();
    auto memInitializer =
        func::FuncOp::create(rewriter, loc, initializerName,
                             FunctionType::get(getContext(), {}, {}));
    memInitializer->setAttr(rewriter.getStringAttr("initializer"),
                            rewriter.getUnitAttr());
    auto *initializerBody = memInitializer.addEntryBlock();
    auto sip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initializerBody);
    auto memRefPtr = memref::GetGlobalOp::create(
        rewriter, loc, MemRefType::get({1}, bufferType), memPtr.getSymName());
    auto alloc = memref::AllocOp::create(
        rewriter, loc,
        MemRefType::get({memOp.getLimits().getMin()}, rewriter.getI8Type()));
    auto castOp =
        memref::CastOp::create(rewriter, loc, bufferType, alloc.getResult());
    auto idx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    memref::StoreOp::create(rewriter, loc, castOp.getResult(),
                            memRefPtr.getResult(), ValueRange{idx.getResult()});
    func::ReturnOp::create(rewriter, loc);
    rewriter.restoreInsertionPoint(sip);
    func::CallOp::create(rewriter, loc, memInitializer);
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
           WasmExtendLowBitsOpConversion,
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
           WasmMemoryOpConversion,
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
           WasmRotlOpConversion,
           WasmRotrOpConversion,
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
