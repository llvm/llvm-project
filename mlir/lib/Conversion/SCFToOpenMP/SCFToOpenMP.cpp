//===- SCFToOpenMP.cpp - Structured Control Flow to OpenMP conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into OpenMP
// parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "../PassDetail.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Matches a block containing a "simple" reduction. The expected shape of the
/// block is as follows.
///
///   ^bb(%arg0, %arg1):
///     %0 = OpTy(%arg0, %arg1)
///     scf.reduce.return %0
template <typename... OpTy>
static bool matchSimpleReduction(Block &block) {
  if (block.empty() || llvm::hasSingleElement(block) ||
      std::next(block.begin(), 2) != block.end())
    return false;

  if (block.getNumArguments() != 2)
    return false;

  SmallVector<Operation *, 4> combinerOps;
  Value reducedVal = matchReduction({block.getArguments()[1]},
                                    /*redPos=*/0, combinerOps);

  if (!reducedVal || !reducedVal.isa<BlockArgument>() ||
      combinerOps.size() != 1)
    return false;

  return isa<OpTy...>(combinerOps[0]) &&
         isa<scf::ReduceReturnOp>(block.back()) &&
         block.front().getOperands() == block.getArguments();
}

/// Matches a block containing a select-based min/max reduction. The types of
/// select and compare operations are provided as template arguments. The
/// comparison predicates suitable for min and max are provided as function
/// arguments. If a reduction is matched, `ifMin` will be set if the reduction
/// compute the minimum and unset if it computes the maximum, otherwise it
/// remains unmodified. The expected shape of the block is as follows.
///
///   ^bb(%arg0, %arg1):
///     %0 = CompareOpTy(<one-of-predicates>, %arg0, %arg1)
///     %1 = SelectOpTy(%0, %arg0, %arg1)  // %arg0, %arg1 may be swapped here.
///     scf.reduce.return %1
template <
    typename CompareOpTy, typename SelectOpTy,
    typename Predicate = decltype(std::declval<CompareOpTy>().predicate())>
static bool
matchSelectReduction(Block &block, ArrayRef<Predicate> lessThanPredicates,
                     ArrayRef<Predicate> greaterThanPredicates, bool &isMin) {
  static_assert(llvm::is_one_of<SelectOpTy, SelectOp, LLVM::SelectOp>::value,
                "only std and llvm select ops are supported");

  // Expect exactly three operations in the block.
  if (block.empty() || llvm::hasSingleElement(block) ||
      std::next(block.begin(), 2) == block.end() ||
      std::next(block.begin(), 3) != block.end())
    return false;

  // Check op kinds.
  auto compare = dyn_cast<CompareOpTy>(block.front());
  auto select = dyn_cast<SelectOpTy>(block.front().getNextNode());
  auto terminator = dyn_cast<scf::ReduceReturnOp>(block.back());
  if (!compare || !select || !terminator)
    return false;

  // Block arguments must be compared.
  if (compare->getOperands() != block.getArguments())
    return false;

  // Detect whether the comparison is less-than or greater-than, otherwise bail.
  bool isLess;
  if (llvm::find(lessThanPredicates, compare.predicate()) !=
      lessThanPredicates.end()) {
    isLess = true;
  } else if (llvm::find(greaterThanPredicates, compare.predicate()) !=
             greaterThanPredicates.end()) {
    isLess = false;
  } else {
    return false;
  }

  if (select.condition() != compare.getResult())
    return false;

  // Detect if the operands are swapped between cmpf and select. Match the
  // comparison type with the requested type or with the opposite of the
  // requested type if the operands are swapped. Use generic accessors because
  // std and LLVM versions of select have different operand names but identical
  // positions.
  constexpr unsigned kTrueValue = 1;
  constexpr unsigned kFalseValue = 2;
  bool sameOperands = select.getOperand(kTrueValue) == compare.lhs() &&
                      select.getOperand(kFalseValue) == compare.rhs();
  bool swappedOperands = select.getOperand(kTrueValue) == compare.rhs() &&
                         select.getOperand(kFalseValue) == compare.lhs();
  if (!sameOperands && !swappedOperands)
    return false;

  if (select.getResult() != terminator.result())
    return false;

  // The reduction is a min if it uses less-than predicates with same operands
  // or greather-than predicates with swapped operands. Similarly for max.
  isMin = (isLess && sameOperands) || (!isLess && swappedOperands);
  return isMin || (isLess & swappedOperands) || (!isLess && sameOperands);
}

/// Returns the float semantics for the given float type.
static const llvm::fltSemantics &fltSemanticsForType(FloatType type) {
  if (type.isF16())
    return llvm::APFloat::IEEEhalf();
  if (type.isF32())
    return llvm::APFloat::IEEEsingle();
  if (type.isF64())
    return llvm::APFloat::IEEEdouble();
  if (type.isF128())
    return llvm::APFloat::IEEEquad();
  if (type.isBF16())
    return llvm::APFloat::BFloat();
  if (type.isF80())
    return llvm::APFloat::x87DoubleExtended();
  llvm_unreachable("unknown float type");
}

/// Returns an attribute with the minimum (if `min` is set) or the maximum value
/// (otherwise) for the given float type.
static Attribute minMaxValueForFloat(Type type, bool min) {
  auto fltType = type.cast<FloatType>();
  return FloatAttr::get(
      type, llvm::APFloat::getLargest(fltSemanticsForType(fltType), min));
}

/// Returns an attribute with the signed integer minimum (if `min` is set) or
/// the maximum value (otherwise) for the given integer type, regardless of its
/// signedness semantics (only the width is considered).
static Attribute minMaxValueForSignedInt(Type type, bool min) {
  auto intType = type.cast<IntegerType>();
  unsigned bitwidth = intType.getWidth();
  return IntegerAttr::get(type, min ? llvm::APInt::getSignedMinValue(bitwidth)
                                    : llvm::APInt::getSignedMaxValue(bitwidth));
}

/// Returns an attribute with the unsigned integer minimum (if `min` is set) or
/// the maximum value (otherwise) for the given integer type, regardless of its
/// signedness semantics (only the width is considered).
static Attribute minMaxValueForUnsignedInt(Type type, bool min) {
  auto intType = type.cast<IntegerType>();
  unsigned bitwidth = intType.getWidth();
  return IntegerAttr::get(type, min ? llvm::APInt::getNullValue(bitwidth)
                                    : llvm::APInt::getAllOnesValue(bitwidth));
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the reduction combiner carried over from `reduce`.
static omp::ReductionDeclareOp createDecl(PatternRewriter &builder,
                                          SymbolTable &symbolTable,
                                          scf::ReduceOp reduce,
                                          Attribute initValue) {
  OpBuilder::InsertionGuard guard(builder);
  auto decl = builder.create<omp::ReductionDeclareOp>(
      reduce.getLoc(), "__scf_reduction", reduce.operand().getType());
  symbolTable.insert(decl);

  Type type = reduce.operand().getType();
  builder.createBlock(&decl.initializerRegion(), decl.initializerRegion().end(),
                      {type});
  builder.setInsertionPointToEnd(&decl.initializerRegion().back());
  Value init =
      builder.create<LLVM::ConstantOp>(reduce.getLoc(), type, initValue);
  builder.create<omp::YieldOp>(reduce.getLoc(), init);

  Operation *terminator = &reduce.getRegion().front().back();
  assert(isa<scf::ReduceReturnOp>(terminator) &&
         "expected reduce op to be terminated by redure return");
  builder.setInsertionPoint(terminator);
  builder.replaceOpWithNewOp<omp::YieldOp>(terminator,
                                           terminator->getOperands());
  builder.inlineRegionBefore(reduce.getRegion(), decl.reductionRegion(),
                             decl.reductionRegion().end());
  return decl;
}

/// Adds an atomic reduction combiner to the given OpenMP reduction declaration
/// using llvm.atomicrmw of the given kind.
static omp::ReductionDeclareOp addAtomicRMW(OpBuilder &builder,
                                            LLVM::AtomicBinOp atomicKind,
                                            omp::ReductionDeclareOp decl,
                                            scf::ReduceOp reduce) {
  OpBuilder::InsertionGuard guard(builder);
  Type type = reduce.operand().getType();
  Type ptrType = LLVM::LLVMPointerType::get(type);
  builder.createBlock(&decl.atomicReductionRegion(),
                      decl.atomicReductionRegion().end(), {ptrType, ptrType});
  Block *atomicBlock = &decl.atomicReductionRegion().back();
  builder.setInsertionPointToEnd(atomicBlock);
  Value loaded = builder.create<LLVM::LoadOp>(reduce.getLoc(),
                                              atomicBlock->getArgument(1));
  builder.create<LLVM::AtomicRMWOp>(reduce.getLoc(), type, atomicKind,
                                    atomicBlock->getArgument(0), loaded,
                                    LLVM::AtomicOrdering::monotonic);
  builder.create<omp::YieldOp>(reduce.getLoc(), ArrayRef<Value>());
  return decl;
}

/// Creates an OpenMP reduction declaration that corresponds to the given SCF
/// reduction and returns it. Recognizes common reductions in order to identify
/// the neutral value, necessary for the OpenMP declaration. If the reduction
/// cannot be recognized, returns null.
static omp::ReductionDeclareOp declareReduction(PatternRewriter &builder,
                                                scf::ReduceOp reduce) {
  Operation *container = SymbolTable::getNearestSymbolTable(reduce);
  SymbolTable symbolTable(container);

  // Insert reduction declarations in the symbol-table ancestor before the
  // ancestor of the current insertion point.
  Operation *insertionPoint = reduce;
  while (insertionPoint->getParentOp() != container)
    insertionPoint = insertionPoint->getParentOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertionPoint);

  assert(llvm::hasSingleElement(reduce.getRegion()) &&
         "expected reduction region to have a single element");

  // Match simple binary reductions that can be expressed with atomicrmw.
  Type type = reduce.operand().getType();
  Block &reduction = reduce.getRegion().front();
  if (matchSimpleReduction<arith::AddFOp, LLVM::FAddOp>(reduction)) {
    omp::ReductionDeclareOp decl = createDecl(builder, symbolTable, reduce,
                                              builder.getFloatAttr(type, 0.0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::fadd, decl, reduce);
  }
  if (matchSimpleReduction<arith::AddIOp, LLVM::AddOp>(reduction)) {
    omp::ReductionDeclareOp decl = createDecl(builder, symbolTable, reduce,
                                              builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::add, decl, reduce);
  }
  if (matchSimpleReduction<arith::OrIOp, LLVM::OrOp>(reduction)) {
    omp::ReductionDeclareOp decl = createDecl(builder, symbolTable, reduce,
                                              builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_or, decl, reduce);
  }
  if (matchSimpleReduction<arith::XOrIOp, LLVM::XOrOp>(reduction)) {
    omp::ReductionDeclareOp decl = createDecl(builder, symbolTable, reduce,
                                              builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_xor, decl, reduce);
  }
  if (matchSimpleReduction<arith::AndIOp, LLVM::AndOp>(reduction)) {
    omp::ReductionDeclareOp decl = createDecl(
        builder, symbolTable, reduce,
        builder.getIntegerAttr(
            type, llvm::APInt::getAllOnesValue(type.getIntOrFloatBitWidth())));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_and, decl, reduce);
  }

  // Match simple binary reductions that cannot be expressed with atomicrmw.
  // TODO: add atomic region using cmpxchg (which needs atomic load to be
  // available as an op).
  if (matchSimpleReduction<arith::MulFOp, LLVM::FMulOp>(reduction)) {
    return createDecl(builder, symbolTable, reduce,
                      builder.getFloatAttr(type, 1.0));
  }

  // Match select-based min/max reductions.
  bool isMin;
  if (matchSelectReduction<arith::CmpFOp, SelectOp>(
          reduction, {arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE},
          {arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE}, isMin) ||
      matchSelectReduction<LLVM::FCmpOp, LLVM::SelectOp>(
          reduction, {LLVM::FCmpPredicate::olt, LLVM::FCmpPredicate::ole},
          {LLVM::FCmpPredicate::ogt, LLVM::FCmpPredicate::oge}, isMin)) {
    return createDecl(builder, symbolTable, reduce,
                      minMaxValueForFloat(type, !isMin));
  }
  if (matchSelectReduction<arith::CmpIOp, SelectOp>(
          reduction, {arith::CmpIPredicate::slt, arith::CmpIPredicate::sle},
          {arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge}, isMin) ||
      matchSelectReduction<LLVM::ICmpOp, LLVM::SelectOp>(
          reduction, {LLVM::ICmpPredicate::slt, LLVM::ICmpPredicate::sle},
          {LLVM::ICmpPredicate::sgt, LLVM::ICmpPredicate::sge}, isMin)) {
    omp::ReductionDeclareOp decl = createDecl(
        builder, symbolTable, reduce, minMaxValueForSignedInt(type, !isMin));
    return addAtomicRMW(builder,
                        isMin ? LLVM::AtomicBinOp::min : LLVM::AtomicBinOp::max,
                        decl, reduce);
  }
  if (matchSelectReduction<arith::CmpIOp, SelectOp>(
          reduction, {arith::CmpIPredicate::ult, arith::CmpIPredicate::ule},
          {arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge}, isMin) ||
      matchSelectReduction<LLVM::ICmpOp, LLVM::SelectOp>(
          reduction, {LLVM::ICmpPredicate::ugt, LLVM::ICmpPredicate::ule},
          {LLVM::ICmpPredicate::ugt, LLVM::ICmpPredicate::uge}, isMin)) {
    omp::ReductionDeclareOp decl = createDecl(
        builder, symbolTable, reduce, minMaxValueForUnsignedInt(type, !isMin));
    return addAtomicRMW(
        builder, isMin ? LLVM::AtomicBinOp::umin : LLVM::AtomicBinOp::umax,
        decl, reduce);
  }

  return nullptr;
}

namespace {

struct ParallelOpLowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    // Replace SCF yield with OpenMP yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(parallelOp.getBody());
      assert(llvm::hasSingleElement(parallelOp.region()) &&
             "expected scf.parallel to have one block");
      rewriter.replaceOpWithNewOp<omp::YieldOp>(
          parallelOp.getBody()->getTerminator(), ValueRange());
    }

    // Declare reductions.
    // TODO: consider checking it here is already a compatible reduction
    // declaration and use it instead of redeclaring.
    SmallVector<Attribute> reductionDeclSymbols;
    for (auto reduce : parallelOp.getOps<scf::ReduceOp>()) {
      omp::ReductionDeclareOp decl = declareReduction(rewriter, reduce);
      if (!decl)
        return failure();
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(rewriter.getContext(), decl.sym_name()));
    }

    // Allocate reduction variables. Make sure the we don't overflow the stack
    // with local `alloca`s by saving and restoring the stack pointer.
    Location loc = parallelOp.getLoc();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(1));
    SmallVector<Value> reductionVariables;
    reductionVariables.reserve(parallelOp.getNumReductions());
    Value token = rewriter.create<LLVM::StackSaveOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    for (Value init : parallelOp.initVals()) {
      assert((LLVM::isCompatibleType(init.getType()) ||
              init.getType().isa<LLVM::PointerElementTypeInterface>()) &&
             "cannot create a reduction variable if the type is not an LLVM "
             "pointer element");
      Value storage = rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(init.getType()), one, 0);
      rewriter.create<LLVM::StoreOp>(loc, init, storage);
      reductionVariables.push_back(storage);
    }

    // Replace the reduction operations contained in this loop. Must be done
    // here rather than in a separate pattern to have access to the list of
    // reduction variables.
    for (auto pair :
         llvm::zip(parallelOp.getOps<scf::ReduceOp>(), reductionVariables)) {
      OpBuilder::InsertionGuard guard(rewriter);
      scf::ReduceOp reduceOp = std::get<0>(pair);
      rewriter.setInsertionPoint(reduceOp);
      rewriter.replaceOpWithNewOp<omp::ReductionOp>(
          reduceOp, reduceOp.operand(), std::get<1>(pair));
    }

    // Create the parallel wrapper.
    auto ompParallel = rewriter.create<omp::ParallelOp>(loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ompParallel.region());

      // Replace SCF yield with OpenMP yield.
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToEnd(parallelOp.getBody());
        assert(llvm::hasSingleElement(parallelOp.region()) &&
               "expected scf.parallel to have one block");
        rewriter.replaceOpWithNewOp<omp::YieldOp>(
            parallelOp.getBody()->getTerminator(), ValueRange());
      }

      // Replace the loop.
      auto loop = rewriter.create<omp::WsLoopOp>(
          parallelOp.getLoc(), parallelOp.lowerBound(), parallelOp.upperBound(),
          parallelOp.step());
      rewriter.create<omp::TerminatorOp>(loc);

      rewriter.inlineRegionBefore(parallelOp.region(), loop.region(),
                                  loop.region().begin());
      if (!reductionVariables.empty()) {
        loop.reductionsAttr(
            ArrayAttr::get(rewriter.getContext(), reductionDeclSymbols));
        loop.reduction_varsMutable().append(reductionVariables);
      }
    }

    // Load loop results.
    SmallVector<Value> results;
    results.reserve(reductionVariables.size());
    for (Value variable : reductionVariables) {
      Value res = rewriter.create<LLVM::LoadOp>(loc, variable);
      results.push_back(res);
    }
    rewriter.replaceOp(parallelOp, results);

    rewriter.create<LLVM::StackRestoreOp>(loc, token);
    return success();
  }
};

/// Applies the conversion patterns in the given function.
static LogicalResult applyPatterns(ModuleOp module) {
  ConversionTarget target(*module.getContext());
  target.addIllegalOp<scf::ReduceOp, scf::ReduceReturnOp, scf::ParallelOp>();
  target.addLegalDialect<omp::OpenMPDialect, LLVM::LLVMDialect>();

  RewritePatternSet patterns(module.getContext());
  patterns.add<ParallelOpLowering>(module.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

/// A pass converting SCF operations to OpenMP operations.
struct SCFToOpenMPPass : public ConvertSCFToOpenMPBase<SCFToOpenMPPass> {
  /// Pass entry point.
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation())))
      signalPassFailure();
  }
};

} // end namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertSCFToOpenMPPass() {
  return std::make_unique<SCFToOpenMPPass>();
}
