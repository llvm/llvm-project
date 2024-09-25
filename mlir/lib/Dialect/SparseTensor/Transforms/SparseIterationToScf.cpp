
#include "Utils/CodegenUtils.h"
#include "Utils/LoopEmitter.h"
#include "Utils/SparseTensorIterator.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

static void convertLevelType(SparseTensorEncodingAttr enc, Level lvl,
                             SmallVectorImpl<Type> &fields) {
  // Position and coordinate buffer in the sparse structure.
  if (enc.getLvlType(lvl).isWithPosLT())
    fields.push_back(enc.getPosMemRefType());
  if (enc.getLvlType(lvl).isWithCrdLT())
    fields.push_back(enc.getCrdMemRefType());
  // One index for shape bound (result from lvlOp).
  fields.push_back(IndexType::get(enc.getContext()));
}

static std::optional<LogicalResult>
convertIterSpaceType(IterSpaceType itSp, SmallVectorImpl<Type> &fields) {

  auto idxTp = IndexType::get(itSp.getContext());
  for (Level l = itSp.getLoLvl(); l < itSp.getHiLvl(); l++)
    convertLevelType(itSp.getEncoding(), l, fields);

  // Two indices for lower and upper bound (we only need one pair for the last
  // iteration space).
  fields.append({idxTp, idxTp});
  return success();
}

static std::optional<LogicalResult>
convertIteratorType(IteratorType itTp, SmallVectorImpl<Type> &fields) {
  // The actually Iterator Values (that are updated every iteration).
  auto idxTp = IndexType::get(itTp.getContext());
  // TODO: handle batch dimension.
  assert(itTp.getEncoding().getBatchLvlRank() == 0);
  if (!itTp.isUnique()) {
    // Segment high for non-unique iterator.
    fields.push_back(idxTp);
  }
  fields.push_back(idxTp);
  return success();
}

static ValueRange
genCoIterateBranchNest(PatternRewriter &rewriter, Location loc, CoIterateOp op,
                       Value loopCrd,
                       ArrayRef<std::unique_ptr<SparseIterator>> iters,
                       ArrayRef<Region *> subCases, ArrayRef<Value> userReduc) {
  if (subCases.empty())
    return userReduc;

  // The current branch that we are handling.
  Region *b = subCases.front();
  Value casePred = constantI1(rewriter, loc, true);
  I64BitSet caseBits = op.getRegionDefinedSpace(b->getRegionNumber());
  for (unsigned i : caseBits.bits()) {
    SparseIterator *it = iters[i].get();
    Value pred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                it->getCrd(), loopCrd);
    casePred = rewriter.create<arith::AndIOp>(loc, casePred, pred);
  }
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(
      loc, ValueRange(userReduc).getTypes(), casePred, /*else=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Erase the empty block.
  rewriter.eraseBlock(&ifOp.getThenRegion().front());
  // Set up block arguments: user-provided values -> loop coord -> iterators.
  SmallVector<Value> blockArgs(userReduc);
  blockArgs.push_back(loopCrd);
  for (unsigned idx : caseBits.bits())
    llvm::append_range(blockArgs, iters[idx]->getCursor());

  IRMapping mapping;
  for (auto [from, to] :
       llvm::zip_equal(b->front().getArguments(), blockArgs)) {
    mapping.map(from, to);
  }

  // Clone the region, we can not erase the region now because the same region
  // might be a subcase for multiple lattice point.
  rewriter.cloneRegionBefore(*b, ifOp.getThenRegion(),
                             ifOp.getThenRegion().begin(), mapping);

  // replace sparse_tensor::YieldOp -> scf::YieldOp
  auto spY = cast<sparse_tensor::YieldOp>(&ifOp.getThenRegion().front().back());
  ValueRange yields = spY.getResults();
  rewriter.eraseOp(spY);
  rewriter.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  rewriter.create<scf::YieldOp>(loc, yields);

  // Generates remaining case recursively.
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  ValueRange res = genCoIterateBranchNest(rewriter, loc, op, loopCrd, iters,
                                          subCases.drop_front(), userReduc);
  if (!res.empty())
    rewriter.create<scf::YieldOp>(loc, res);

  rewriter.setInsertionPointAfter(ifOp);
  return ifOp.getResults();
}

static ValueRange genLoopWithIterator(
    PatternRewriter &rewriter, Location loc, SparseIterator *it,
    ValueRange reduc,
    function_ref<SmallVector<Value>(PatternRewriter &rewriter, Location loc,
                                    Region &loopBody, SparseIterator *it,
                                    ValueRange reduc)>
        bodyBuilder) {
  if (it->iteratableByFor()) {
    auto [lo, hi] = it->genForCond(rewriter, loc);
    Value step = constantIndex(rewriter, loc, 1);
    scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step, reduc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      // Erase the implicit yield operation created by ForOp when there is no
      // yielding values.
      if (!forOp.getBody()->empty())
        rewriter.eraseOp(&forOp.getBody()->front());
      assert(forOp.getBody()->empty());

      it->linkNewScope(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
      SmallVector<Value> ret = bodyBuilder(rewriter, loc, forOp.getBodyRegion(),
                                           it, forOp.getRegionIterArgs());

      rewriter.setInsertionPointToEnd(forOp.getBody());
      rewriter.create<scf::YieldOp>(loc, ret);
    }
    return forOp.getResults();
  }

  SmallVector<Value> ivs(reduc);
  llvm::append_range(ivs, it->getCursor());

  TypeRange types = ValueRange(ivs).getTypes();
  auto whileOp = rewriter.create<scf::WhileOp>(loc, types, ivs);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    // Generates loop conditions.
    SmallVector<Location> l(types.size(), loc);
    Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, types, l);
    rewriter.setInsertionPointToStart(before);
    ValueRange bArgs = before->getArguments();
    auto [whileCond, remArgs] = it->genWhileCond(rewriter, loc, bArgs);
    rewriter.create<scf::ConditionOp>(loc, whileCond, before->getArguments());

    // Delegates loop body generation.
    Region &dstRegion = whileOp.getAfter();
    Block *after = rewriter.createBlock(&dstRegion, {}, types, l);
    ValueRange aArgs = whileOp.getAfterArguments();
    it->linkNewScope(aArgs.drop_front(reduc.size()));
    aArgs = aArgs.take_front(reduc.size());

    rewriter.setInsertionPointToStart(after);
    SmallVector<Value> ret = bodyBuilder(rewriter, loc, dstRegion, it, aArgs);
    rewriter.setInsertionPointToEnd(after);

    // Forward loops
    SmallVector<Value> yields;
    llvm::append_range(yields, ret);
    llvm::append_range(yields, it->forward(rewriter, loc));
    rewriter.create<scf::YieldOp>(loc, yields);
  }
  return whileOp.getResults().drop_front(it->getCursor().size());
}

namespace {

/// Sparse codegen rule for number of entries operator.
class ExtractIterSpaceConverter
    : public OneToNOpConversionPattern<ExtractIterSpaceOp> {
public:
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractIterSpaceOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // Construct the iteration space.
    SparseIterationSpace space(loc, rewriter, op.getTensor(), 0,
                               op.getLvlRange(), adaptor.getParentIter());

    SmallVector<Value> result = space.toValues();
    rewriter.replaceOp(op, result, resultMapping);
    return success();
  }
};

/// Sparse codegen rule for number of entries operator.
class ExtractValOpConverter : public OneToNOpConversionPattern<ExtractValOp> {
public:
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractValOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value pos = adaptor.getIterator().back();
    Value valBuf = rewriter.create<ToValuesOp>(loc, op.getTensor());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, valBuf, pos);
    return success();
  }
};

class SparseIterateOpConverter : public OneToNOpConversionPattern<IterateOp> {
public:
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(IterateOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (!op.getCrdUsedLvls().empty())
      return rewriter.notifyMatchFailure(
          op, "non-empty coordinates list not implemented.");

    Location loc = op.getLoc();

    auto iterSpace = SparseIterationSpace::fromValues(
        op.getIterSpace().getType(), adaptor.getIterSpace(), 0);

    std::unique_ptr<SparseIterator> it =
        iterSpace.extractIterator(rewriter, loc);

    SmallVector<Value> ivs;
    for (ValueRange inits : adaptor.getInitArgs())
      llvm::append_range(ivs, inits);

    // Type conversion on iterate op block.
    OneToNTypeMapping blockTypeMapping(op.getBody()->getArgumentTypes());
    if (failed(typeConverter->convertSignatureArgs(
            op.getBody()->getArgumentTypes(), blockTypeMapping)))
      return rewriter.notifyMatchFailure(
          op, "failed to convert iterate region argurment types");
    rewriter.applySignatureConversion(op.getBody(), blockTypeMapping);

    Block *block = op.getBody();
    ValueRange ret = genLoopWithIterator(
        rewriter, loc, it.get(), ivs,
        [block](PatternRewriter &rewriter, Location loc, Region &loopBody,
                SparseIterator *it, ValueRange reduc) -> SmallVector<Value> {
          SmallVector<Value> blockArgs(reduc);
          // TODO: Also appends coordinates if used.
          // blockArgs.push_back(it->deref(rewriter, loc));
          llvm::append_range(blockArgs, it->getCursor());

          Block *dstBlock = &loopBody.getBlocks().front();
          rewriter.inlineBlockBefore(block, dstBlock, dstBlock->end(),
                                     blockArgs);
          auto yield = llvm::cast<sparse_tensor::YieldOp>(dstBlock->back());
          // We can not use ValueRange as the operation holding the values will
          // be destoryed.
          SmallVector<Value> result(yield.getResults());
          rewriter.eraseOp(yield);
          return result;
        });

    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    rewriter.replaceOp(op, ret, resultMapping);
    return success();
  }
};

class SparseCoIterateOpConverter
    : public OneToNOpConversionPattern<CoIterateOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(CoIterateOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    assert(op.getSpaceDim() == 1 && "Not implemented");
    Location loc = op.getLoc();

    I64BitSet denseBits(0);
    for (auto [idx, spaceTp] : llvm::enumerate(op.getIterSpaces().getTypes()))
      if (all_of(cast<IterSpaceType>(spaceTp).getLvlTypes(), isDenseLT))
        denseBits.set(idx);

    // If there exists a case that only contains dense spaces. I.e., case
    // bits is a subset of dense bits, or when there is a full empty case (due
    // to complements), we need a universal pointer to forward the coiteration
    // loop.
    bool needUniv =
        any_of(op.getRegionDefinedSpaces(), [denseBits](I64BitSet caseBits) {
          // A case for complement.
          if (caseBits.count() == 0)
            return true;
          // An all-dense case.
          return caseBits.isSubSetOf(denseBits);
        });
    assert(!needUniv && "Not implemented");
    (void)needUniv;

    for (Region &region : op.getCaseRegions()) {
      // Do a one-shot type conversion on all region blocks, since the same
      // region might be used multiple time.
      Block *block = &region.getBlocks().front();
      OneToNTypeMapping blockTypeMapping(block->getArgumentTypes());
      if (failed(typeConverter->convertSignatureArgs(block->getArgumentTypes(),
                                                     blockTypeMapping))) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert coiterate region argurment types");
      }

      rewriter.applySignatureConversion(block, blockTypeMapping);
    }

    SmallVector<SparseIterationSpace> spaces;
    SmallVector<std::unique_ptr<SparseIterator>> iters;
    for (auto [spaceTp, spaceVals] : llvm::zip_equal(
             op.getIterSpaces().getTypes(), adaptor.getIterSpaces())) {
      // TODO: do we really need tid?
      spaces.push_back(SparseIterationSpace::fromValues(
          cast<IterSpaceType>(spaceTp), spaceVals, /*tid=*/0));
      // Extract the iterator.
      iters.push_back(spaces.back().extractIterator(rewriter, loc));
    }

    auto getFilteredIters = [&iters](I64BitSet caseBits) {
      // Retrives a vector of pointers to the iterators used in the case.
      SmallVector<SparseIterator *> validIters;
      for (auto idx : caseBits.bits())
        validIters.push_back(iters[idx].get());
      return validIters;
    };

    // Get a flattened user-provided loop reduction values.
    SmallVector<Value> userReduc;
    for (ValueRange r : adaptor.getInitArgs())
      llvm::append_range(userReduc, r);

    // TODO: we need to sort the cases such that they appears in lexical order.
    // Although sparsification always generates cases in that order, it might
    // not be the case for human-written code.

    // Generates a loop sequence, one loop per case.
    for (auto [r, caseBits] :
         llvm::zip_equal(op.getCaseRegions(), op.getRegionDefinedSpaces())) {
      assert(caseBits.count() > 0 && "Complement space not implemented");

      // Retrives a vector of pointers to the iterators used in the case.
      SmallVector<SparseIterator *> validIters = getFilteredIters(caseBits);

      if (validIters.size() > 1) {
        auto [loop, loopCrd] =
            genCoIteration(rewriter, loc, validIters, userReduc,
                           /*uniIdx=*/nullptr, /*userReducFirst=*/true);

        // 1st. find all the cases that is a strict subset of the current case
        // condition, for which we generate one branch per case inside the loop.
        // The subcases are never empty, it must contains at least the current
        // region itself.
        // TODO: these cases should be sorted.
        SmallVector<Region *> subCases = op.getSubCasesOf(r.getRegionNumber());
        assert(!subCases.empty());

        ValueRange res = genCoIterateBranchNest(rewriter, loc, op, loopCrd,
                                                iters, subCases, userReduc);

        SmallVector<Value> nextIterYields(res);
        // 2nd. foward the loop.
        for (SparseIterator *it : validIters) {
          Value cmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, it->getCrd(), loopCrd);
          it->forwardIf(rewriter, loc, cmp);
          llvm::append_range(nextIterYields, it->getCursor());
        }
        rewriter.create<scf::YieldOp>(loc, nextIterYields);

        // Exit the loop, relink the iterator SSA value.
        rewriter.setInsertionPointAfter(loop);
        ValueRange iterVals = loop->getResults().drop_front(userReduc.size());
        for (SparseIterator *it : validIters)
          iterVals = it->linkNewScope(iterVals);
        assert(iterVals.empty());

        ValueRange curResult = loop->getResults().take_front(userReduc.size());
        userReduc.assign(curResult.begin(), curResult.end());
      } else {
        // This is a simple iteration loop.
        assert(caseBits.count() == 1);

        Block *block = &r.getBlocks().front();
        ValueRange curResult = genLoopWithIterator(
            rewriter, loc, validIters.front(), userReduc,
            /*bodyBuilder=*/
            [block](PatternRewriter &rewriter, Location loc, Region &dstRegion,
                    SparseIterator *it,
                    ValueRange reduc) -> SmallVector<Value> {
              SmallVector<Value> blockArgs(reduc);
              blockArgs.push_back(it->deref(rewriter, loc));
              llvm::append_range(blockArgs, it->getCursor());

              Block *dstBlock = &dstRegion.getBlocks().front();
              rewriter.inlineBlockBefore(
                  block, dstBlock, rewriter.getInsertionPoint(), blockArgs);
              auto yield = llvm::cast<sparse_tensor::YieldOp>(dstBlock->back());
              SmallVector<Value> result(yield.getResults());
              rewriter.eraseOp(yield);
              return result;
            });

        userReduc.assign(curResult.begin(), curResult.end());
      }
    }

    rewriter.replaceOp(op, userReduc);
    return success();
  }
};

} // namespace

mlir::SparseIterationTypeConverter::SparseIterationTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertIteratorType);
  addConversion(convertIterSpaceType);

  addSourceMaterialization([](OpBuilder &builder, IterSpaceType spTp,
                              ValueRange inputs,
                              Location loc) -> std::optional<Value> {
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange(spTp), inputs)
        .getResult(0);
  });
}

void mlir::populateLowerSparseIterationToSCFPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {

  IterateOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  patterns.add<ExtractIterSpaceConverter, ExtractValOpConverter,
               SparseIterateOpConverter, SparseCoIterateOpConverter>(
      converter, patterns.getContext());
}
