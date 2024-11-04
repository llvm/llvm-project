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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSCFTOOPENMPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

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

  if (!reducedVal || !isa<BlockArgument>(reducedVal) || combinerOps.size() != 1)
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
    typename Predicate = decltype(std::declval<CompareOpTy>().getPredicate())>
static bool
matchSelectReduction(Block &block, ArrayRef<Predicate> lessThanPredicates,
                     ArrayRef<Predicate> greaterThanPredicates, bool &isMin) {
  static_assert(
      llvm::is_one_of<SelectOpTy, arith::SelectOp, LLVM::SelectOp>::value,
      "only arithmetic and llvm select ops are supported");

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
  if (llvm::is_contained(lessThanPredicates, compare.getPredicate())) {
    isLess = true;
  } else if (llvm::is_contained(greaterThanPredicates,
                                compare.getPredicate())) {
    isLess = false;
  } else {
    return false;
  }

  if (select.getCondition() != compare.getResult())
    return false;

  // Detect if the operands are swapped between cmpf and select. Match the
  // comparison type with the requested type or with the opposite of the
  // requested type if the operands are swapped. Use generic accessors because
  // std and LLVM versions of select have different operand names but identical
  // positions.
  constexpr unsigned kTrueValue = 1;
  constexpr unsigned kFalseValue = 2;
  bool sameOperands = select.getOperand(kTrueValue) == compare.getLhs() &&
                      select.getOperand(kFalseValue) == compare.getRhs();
  bool swappedOperands = select.getOperand(kTrueValue) == compare.getRhs() &&
                         select.getOperand(kFalseValue) == compare.getLhs();
  if (!sameOperands && !swappedOperands)
    return false;

  if (select.getResult() != terminator.getResult())
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
  auto fltType = cast<FloatType>(type);
  return FloatAttr::get(
      type, llvm::APFloat::getLargest(fltSemanticsForType(fltType), min));
}

/// Returns an attribute with the signed integer minimum (if `min` is set) or
/// the maximum value (otherwise) for the given integer type, regardless of its
/// signedness semantics (only the width is considered).
static Attribute minMaxValueForSignedInt(Type type, bool min) {
  auto intType = cast<IntegerType>(type);
  unsigned bitwidth = intType.getWidth();
  return IntegerAttr::get(type, min ? llvm::APInt::getSignedMinValue(bitwidth)
                                    : llvm::APInt::getSignedMaxValue(bitwidth));
}

/// Returns an attribute with the unsigned integer minimum (if `min` is set) or
/// the maximum value (otherwise) for the given integer type, regardless of its
/// signedness semantics (only the width is considered).
static Attribute minMaxValueForUnsignedInt(Type type, bool min) {
  auto intType = cast<IntegerType>(type);
  unsigned bitwidth = intType.getWidth();
  return IntegerAttr::get(type, min ? llvm::APInt::getZero(bitwidth)
                                    : llvm::APInt::getAllOnes(bitwidth));
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the `reductionIndex`-th reduction combiner carried
/// over from `reduce`.
static omp::DeclareReductionOp
createDecl(PatternRewriter &builder, SymbolTable &symbolTable,
           scf::ReduceOp reduce, int64_t reductionIndex, Attribute initValue) {
  OpBuilder::InsertionGuard guard(builder);
  Type type = reduce.getOperands()[reductionIndex].getType();
  auto decl = builder.create<omp::DeclareReductionOp>(reduce.getLoc(),
                                                      "__scf_reduction", type);
  symbolTable.insert(decl);

  builder.createBlock(&decl.getInitializerRegion(),
                      decl.getInitializerRegion().end(), {type},
                      {reduce.getOperands()[reductionIndex].getLoc()});
  builder.setInsertionPointToEnd(&decl.getInitializerRegion().back());
  Value init =
      builder.create<LLVM::ConstantOp>(reduce.getLoc(), type, initValue);
  builder.create<omp::YieldOp>(reduce.getLoc(), init);

  Operation *terminator =
      &reduce.getReductions()[reductionIndex].front().back();
  assert(isa<scf::ReduceReturnOp>(terminator) &&
         "expected reduce op to be terminated by redure return");
  builder.setInsertionPoint(terminator);
  builder.replaceOpWithNewOp<omp::YieldOp>(terminator,
                                           terminator->getOperands());
  builder.inlineRegionBefore(reduce.getReductions()[reductionIndex],
                             decl.getReductionRegion(),
                             decl.getReductionRegion().end());
  return decl;
}

/// Adds an atomic reduction combiner to the given OpenMP reduction declaration
/// using llvm.atomicrmw of the given kind.
static omp::DeclareReductionOp addAtomicRMW(OpBuilder &builder,
                                            LLVM::AtomicBinOp atomicKind,
                                            omp::DeclareReductionOp decl,
                                            scf::ReduceOp reduce,
                                            int64_t reductionIndex) {
  OpBuilder::InsertionGuard guard(builder);
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Location reduceOperandLoc = reduce.getOperands()[reductionIndex].getLoc();
  builder.createBlock(&decl.getAtomicReductionRegion(),
                      decl.getAtomicReductionRegion().end(), {ptrType, ptrType},
                      {reduceOperandLoc, reduceOperandLoc});
  Block *atomicBlock = &decl.getAtomicReductionRegion().back();
  builder.setInsertionPointToEnd(atomicBlock);
  Value loaded = builder.create<LLVM::LoadOp>(reduce.getLoc(), decl.getType(),
                                              atomicBlock->getArgument(1));
  builder.create<LLVM::AtomicRMWOp>(reduce.getLoc(), atomicKind,
                                    atomicBlock->getArgument(0), loaded,
                                    LLVM::AtomicOrdering::monotonic);
  builder.create<omp::YieldOp>(reduce.getLoc(), ArrayRef<Value>());
  return decl;
}

/// Creates an OpenMP reduction declaration that corresponds to the given SCF
/// reduction and returns it. Recognizes common reductions in order to identify
/// the neutral value, necessary for the OpenMP declaration. If the reduction
/// cannot be recognized, returns null.
static omp::DeclareReductionOp declareReduction(PatternRewriter &builder,
                                                scf::ReduceOp reduce,
                                                int64_t reductionIndex) {
  Operation *container = SymbolTable::getNearestSymbolTable(reduce);
  SymbolTable symbolTable(container);

  // Insert reduction declarations in the symbol-table ancestor before the
  // ancestor of the current insertion point.
  Operation *insertionPoint = reduce;
  while (insertionPoint->getParentOp() != container)
    insertionPoint = insertionPoint->getParentOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertionPoint);

  assert(llvm::hasSingleElement(reduce.getReductions()[reductionIndex]) &&
         "expected reduction region to have a single element");

  // Match simple binary reductions that can be expressed with atomicrmw.
  Type type = reduce.getOperands()[reductionIndex].getType();
  Block &reduction = reduce.getReductions()[reductionIndex].front();
  if (matchSimpleReduction<arith::AddFOp, LLVM::FAddOp>(reduction)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   builder.getFloatAttr(type, 0.0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::fadd, decl, reduce,
                        reductionIndex);
  }
  if (matchSimpleReduction<arith::AddIOp, LLVM::AddOp>(reduction)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::add, decl, reduce,
                        reductionIndex);
  }
  if (matchSimpleReduction<arith::OrIOp, LLVM::OrOp>(reduction)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_or, decl, reduce,
                        reductionIndex);
  }
  if (matchSimpleReduction<arith::XOrIOp, LLVM::XOrOp>(reduction)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   builder.getIntegerAttr(type, 0));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_xor, decl, reduce,
                        reductionIndex);
  }
  if (matchSimpleReduction<arith::AndIOp, LLVM::AndOp>(reduction)) {
    omp::DeclareReductionOp decl = createDecl(
        builder, symbolTable, reduce, reductionIndex,
        builder.getIntegerAttr(
            type, llvm::APInt::getAllOnes(type.getIntOrFloatBitWidth())));
    return addAtomicRMW(builder, LLVM::AtomicBinOp::_and, decl, reduce,
                        reductionIndex);
  }

  // Match simple binary reductions that cannot be expressed with atomicrmw.
  // TODO: add atomic region using cmpxchg (which needs atomic load to be
  // available as an op).
  if (matchSimpleReduction<arith::MulFOp, LLVM::FMulOp>(reduction)) {
    return createDecl(builder, symbolTable, reduce, reductionIndex,
                      builder.getFloatAttr(type, 1.0));
  }
  if (matchSimpleReduction<arith::MulIOp, LLVM::MulOp>(reduction)) {
    return createDecl(builder, symbolTable, reduce, reductionIndex,
                      builder.getIntegerAttr(type, 1));
  }

  // Match select-based min/max reductions.
  bool isMin;
  if (matchSelectReduction<arith::CmpFOp, arith::SelectOp>(
          reduction, {arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE},
          {arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE}, isMin) ||
      matchSelectReduction<LLVM::FCmpOp, LLVM::SelectOp>(
          reduction, {LLVM::FCmpPredicate::olt, LLVM::FCmpPredicate::ole},
          {LLVM::FCmpPredicate::ogt, LLVM::FCmpPredicate::oge}, isMin)) {
    return createDecl(builder, symbolTable, reduce, reductionIndex,
                      minMaxValueForFloat(type, !isMin));
  }
  if (matchSelectReduction<arith::CmpIOp, arith::SelectOp>(
          reduction, {arith::CmpIPredicate::slt, arith::CmpIPredicate::sle},
          {arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge}, isMin) ||
      matchSelectReduction<LLVM::ICmpOp, LLVM::SelectOp>(
          reduction, {LLVM::ICmpPredicate::slt, LLVM::ICmpPredicate::sle},
          {LLVM::ICmpPredicate::sgt, LLVM::ICmpPredicate::sge}, isMin)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   minMaxValueForSignedInt(type, !isMin));
    return addAtomicRMW(builder,
                        isMin ? LLVM::AtomicBinOp::min : LLVM::AtomicBinOp::max,
                        decl, reduce, reductionIndex);
  }
  if (matchSelectReduction<arith::CmpIOp, arith::SelectOp>(
          reduction, {arith::CmpIPredicate::ult, arith::CmpIPredicate::ule},
          {arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge}, isMin) ||
      matchSelectReduction<LLVM::ICmpOp, LLVM::SelectOp>(
          reduction, {LLVM::ICmpPredicate::ugt, LLVM::ICmpPredicate::ule},
          {LLVM::ICmpPredicate::ugt, LLVM::ICmpPredicate::uge}, isMin)) {
    omp::DeclareReductionOp decl =
        createDecl(builder, symbolTable, reduce, reductionIndex,
                   minMaxValueForUnsignedInt(type, !isMin));
    return addAtomicRMW(
        builder, isMin ? LLVM::AtomicBinOp::umin : LLVM::AtomicBinOp::umax,
        decl, reduce, reductionIndex);
  }

  return nullptr;
}

namespace {

struct ParallelOpLowering : public OpRewritePattern<scf::ParallelOp> {
  static constexpr unsigned kUseOpenMPDefaultNumThreads = 0;
  unsigned numThreads;

  ParallelOpLowering(MLIRContext *context,
                     unsigned numThreads = kUseOpenMPDefaultNumThreads)
      : OpRewritePattern<scf::ParallelOp>(context), numThreads(numThreads) {}

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    // Declare reductions.
    // TODO: consider checking it here is already a compatible reduction
    // declaration and use it instead of redeclaring.
    SmallVector<Attribute> reductionDeclSymbols;
    SmallVector<omp::DeclareReductionOp> ompReductionDecls;
    auto reduce = cast<scf::ReduceOp>(parallelOp.getBody()->getTerminator());
    for (int64_t i = 0, e = parallelOp.getNumReductions(); i < e; ++i) {
      omp::DeclareReductionOp decl = declareReduction(rewriter, reduce, i);
      ompReductionDecls.push_back(decl);
      if (!decl)
        return failure();
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(rewriter.getContext(), decl.getSymName()));
    }

    // Allocate reduction variables. Make sure the we don't overflow the stack
    // with local `alloca`s by saving and restoring the stack pointer.
    Location loc = parallelOp.getLoc();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(1));
    SmallVector<Value> reductionVariables;
    reductionVariables.reserve(parallelOp.getNumReductions());
    auto ptrType = LLVM::LLVMPointerType::get(parallelOp.getContext());
    for (Value init : parallelOp.getInitVals()) {
      assert((LLVM::isCompatibleType(init.getType()) ||
              isa<LLVM::PointerElementTypeInterface>(init.getType())) &&
             "cannot create a reduction variable if the type is not an LLVM "
             "pointer element");
      Value storage =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, init.getType(), one, 0);
      rewriter.create<LLVM::StoreOp>(loc, init, storage);
      reductionVariables.push_back(storage);
    }

    // Replace the reduction operations contained in this loop. Must be done
    // here rather than in a separate pattern to have access to the list of
    // reduction variables.
    for (auto [x, y, rD] : llvm::zip_equal(
             reductionVariables, reduce.getOperands(), ompReductionDecls)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(reduce);
      Region &redRegion = rD.getReductionRegion();
      // The SCF dialect by definition contains only structured operations
      // and hence the SCF reduction region will contain a single block.
      // The ompReductionDecls region is a copy of the SCF reduction region
      // and hence has the same property.
      assert(redRegion.hasOneBlock() &&
             "expect reduction region to have one block");
      Value pvtRedVar = parallelOp.getRegion().addArgument(x.getType(), loc);
      Value pvtRedVal = rewriter.create<LLVM::LoadOp>(reduce.getLoc(),
                                                      rD.getType(), pvtRedVar);
      // Make a copy of the reduction combiner region in the body
      mlir::OpBuilder builder(rewriter.getContext());
      builder.setInsertionPoint(reduce);
      mlir::IRMapping mapper;
      assert(redRegion.getNumArguments() == 2 &&
             "expect reduction region to have two arguments");
      mapper.map(redRegion.getArgument(0), pvtRedVal);
      mapper.map(redRegion.getArgument(1), y);
      for (auto &op : redRegion.getOps()) {
        Operation *cloneOp = builder.clone(op, mapper);
        if (auto yieldOp = dyn_cast<omp::YieldOp>(*cloneOp)) {
          assert(yieldOp && yieldOp.getResults().size() == 1 &&
                 "expect YieldOp in reduction region to return one result");
          Value redVal = yieldOp.getResults()[0];
          rewriter.create<LLVM::StoreOp>(loc, redVal, pvtRedVar);
          rewriter.eraseOp(yieldOp);
          break;
        }
      }
    }
    rewriter.eraseOp(reduce);

    Value numThreadsVar;
    if (numThreads > 0) {
      numThreadsVar = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(numThreads));
    }
    // Create the parallel wrapper.
    auto ompParallel = rewriter.create<omp::ParallelOp>(
        loc,
        /* if_expr_var = */ Value{},
        /* num_threads_var = */ numThreadsVar,
        /* allocate_vars = */ llvm::SmallVector<Value>{},
        /* allocators_vars = */ llvm::SmallVector<Value>{},
        /* reduction_vars = */ llvm::SmallVector<Value>{},
        /* reduction_vars_isbyref = */ DenseBoolArrayAttr{},
        /* reductions = */ ArrayAttr{},
        /* proc_bind_val = */ omp::ClauseProcBindKindAttr{},
        /* private_vars = */ ValueRange(),
        /* privatizers = */ nullptr);
    {

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ompParallel.getRegion());

      // Replace the loop.
      {
        OpBuilder::InsertionGuard allocaGuard(rewriter);
        // Create worksharing loop wrapper.
        auto wsloopOp = rewriter.create<omp::WsloopOp>(parallelOp.getLoc());
        if (!reductionVariables.empty()) {
          wsloopOp.setReductionsAttr(
              ArrayAttr::get(rewriter.getContext(), reductionDeclSymbols));
          wsloopOp.getReductionVarsMutable().append(reductionVariables);
          llvm::SmallVector<bool> byRefVec;
          // false because these reductions always reduce scalars and so do
          // not need to pass by reference
          byRefVec.resize(reductionVariables.size(), false);
          wsloopOp.setReductionVarsByref(
              DenseBoolArrayAttr::get(rewriter.getContext(), byRefVec));
        }
        rewriter.create<omp::TerminatorOp>(loc); // omp.parallel terminator.

        // The wrapper's entry block arguments will define the reduction
        // variables.
        llvm::SmallVector<mlir::Type> reductionTypes;
        reductionTypes.reserve(reductionVariables.size());
        llvm::transform(reductionVariables, std::back_inserter(reductionTypes),
                        [](mlir::Value v) { return v.getType(); });
        rewriter.createBlock(
            &wsloopOp.getRegion(), {}, reductionTypes,
            llvm::SmallVector<mlir::Location>(reductionVariables.size(),
                                              parallelOp.getLoc()));

        rewriter.setInsertionPoint(
            rewriter.create<omp::TerminatorOp>(parallelOp.getLoc()));

        // Create loop nest and populate region with contents of scf.parallel.
        auto loopOp = rewriter.create<omp::LoopNestOp>(
            parallelOp.getLoc(), parallelOp.getLowerBound(),
            parallelOp.getUpperBound(), parallelOp.getStep());

        rewriter.inlineRegionBefore(parallelOp.getRegion(), loopOp.getRegion(),
                                    loopOp.getRegion().begin());

        // Remove reduction-related block arguments from omp.loop_nest and
        // redirect uses to the corresponding omp.wsloop block argument.
        mlir::Block &loopOpEntryBlock = loopOp.getRegion().front();
        unsigned numLoops = parallelOp.getNumLoops();
        rewriter.replaceAllUsesWith(
            loopOpEntryBlock.getArguments().drop_front(numLoops),
            wsloopOp.getRegion().getArguments());
        loopOpEntryBlock.eraseArguments(
            numLoops, loopOpEntryBlock.getNumArguments() - numLoops);

        Block *ops =
            rewriter.splitBlock(&loopOpEntryBlock, loopOpEntryBlock.begin());
        rewriter.setInsertionPointToStart(&loopOpEntryBlock);

        auto scope = rewriter.create<memref::AllocaScopeOp>(parallelOp.getLoc(),
                                                            TypeRange());
        rewriter.create<omp::YieldOp>(loc, ValueRange());
        Block *scopeBlock = rewriter.createBlock(&scope.getBodyRegion());
        rewriter.mergeBlocks(ops, scopeBlock);
        rewriter.setInsertionPointToEnd(&*scope.getBodyRegion().begin());
        rewriter.create<memref::AllocaScopeReturnOp>(loc, ValueRange());
      }
    }

    // Load loop results.
    SmallVector<Value> results;
    results.reserve(reductionVariables.size());
    for (auto [variable, type] :
         llvm::zip(reductionVariables, parallelOp.getResultTypes())) {
      Value res = rewriter.create<LLVM::LoadOp>(loc, type, variable);
      results.push_back(res);
    }
    rewriter.replaceOp(parallelOp, results);

    return success();
  }
};

/// Applies the conversion patterns in the given function.
static LogicalResult applyPatterns(ModuleOp module, unsigned numThreads) {
  ConversionTarget target(*module.getContext());
  target.addIllegalOp<scf::ReduceOp, scf::ReduceReturnOp, scf::ParallelOp>();
  target.addLegalDialect<omp::OpenMPDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect>();

  RewritePatternSet patterns(module.getContext());
  patterns.add<ParallelOpLowering>(module.getContext(), numThreads);
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

/// A pass converting SCF operations to OpenMP operations.
struct SCFToOpenMPPass
    : public impl::ConvertSCFToOpenMPPassBase<SCFToOpenMPPass> {

  using Base::Base;

  /// Pass entry point.
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation(), numThreads)))
      signalPassFailure();
  }
};

} // namespace
