
#include "Utils/CodegenUtils.h"
#include "Utils/SparseTensorIterator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

static std::optional<LogicalResult>
convertIterSpaceType(IterSpaceType itSp, SmallVectorImpl<Type> &fields) {
  if (itSp.getSpaceDim() > 1)
    llvm_unreachable("Not implemented.");

  auto idxTp = IndexType::get(itSp.getContext());
  // FIXME: this assumes that the Pos/CrdBitWidth in sparse tensor encoding is
  // overriden to non-default values.
  auto sparseMemRef = MemRefType::get({ShapedType::kDynamic}, idxTp);
  for (LevelType lt : itSp.getLvlTypes()) {
    // Position and coordinate buffer in the sparse structure.
    if (lt.isWithPosLT())
      fields.push_back(sparseMemRef);
    if (lt.isWithCrdLT())
      fields.push_back(sparseMemRef);
  }
  // Two indices for lower and upper bound.
  fields.append({idxTp, idxTp});
  return success();
}

static std::optional<LogicalResult>
convertIteratorType(IteratorType itTp, SmallVectorImpl<Type> &fields) {
  if (itTp.getSpaceDim() > 1)
    llvm_unreachable("Not implemented.");

  auto idxTp = IndexType::get(itTp.getContext());
  // TODO: This assumes there is no batch dimenstion in the sparse tensor.
  if (!itTp.isUnique()) {
    // Segment high for non-unqiue iterator.
    fields.push_back(idxTp);
  }
  fields.push_back(idxTp);
  return success();
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
    if (op.getSpaceDim() > 1)
      llvm_unreachable("Not implemented.");
    Location loc = op.getLoc();

    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    std::unique_ptr<SparseTensorLevel> lvl =
        makeSparseTensorLevel(rewriter, loc, op.getTensor(), 0, op.getLoLvl());

    SmallVector<Value> result = llvm::to_vector(lvl->getLvlBuffers());
    if (!op.getParentIter()) {
      // TODO: handle batch.
      std::pair<Value, Value> bounds = lvl->peekRangeAt(
          rewriter, loc, /*batchPrefix*/ {}, constantIndex(rewriter, loc, 0));
      result.append({bounds.first, bounds.second});
    } else {
      llvm_unreachable("Not implemented.");
    }

    rewriter.replaceOp(op, result, resultMapping);
    return success();
  }
};

class SparseIterateOpConverter : public OneToNOpConversionPattern<IterateOp> {
public:
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(IterateOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (op.getSpaceDim() > 1 || !op.getCrdUsedLvls().empty())
      llvm_unreachable("Not implemented.");

    Location loc = op.getLoc();

    LevelType lt = op.getIterSpace().getType().getLvlTypes().front();

    ValueRange buffers = adaptor.getIterSpace().take_front(2);
    // TODO: Introduce a class to represent a sparse iter_space, which is a
    // combination of sparse levels and posRange.
    // ValueRange posRange = adaptor.getIterSpace().take_front(2);

    std::unique_ptr<SparseTensorLevel> stl = makeSparseTensorLevel(
        lt, /*sz=*/nullptr, buffers, /*tid=*/0, /*lvl=*/0);

    // TODO: decouple sparse iterator with sparse levels.
    std::unique_ptr<SparseIterator> it = makeSimpleIterator(*stl);

    // FIXME: only works for the first level.
    it->genInit(rewriter, loc, /*parent*/ nullptr);
    if (it->iteratableByFor()) {
      // TODO
      llvm_unreachable("not yet implemented.");
    } else {
      SmallVector<Value> ivs;
      llvm::append_range(ivs, it->getCursor());
      for (ValueRange inits : adaptor.getInitArgs())
        llvm::append_range(ivs, inits);

      assert(llvm::all_of(ivs, [](Value v) { return v != nullptr; }));

      TypeRange types = ValueRange(ivs).getTypes();
      auto whileOp = rewriter.create<scf::WhileOp>(loc, types, ivs);
      SmallVector<Location> l(types.size(), op.getIterator().getLoc());

      // Generates loop conditions.
      Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, types, l);
      rewriter.setInsertionPointToStart(before);
      ValueRange bArgs = before->getArguments();
      auto [whileCond, remArgs] = it->genWhileCond(rewriter, loc, bArgs);
      assert(remArgs.size() == adaptor.getInitArgs().size());
      rewriter.create<scf::ConditionOp>(loc, whileCond, before->getArguments());

      // Generates loop body.
      Block *loopBody = op.getBody();
      OneToNTypeMapping bodyTypeMapping(loopBody->getArgumentTypes());
      if (failed(typeConverter->convertSignatureArgs(
              loopBody->getArgumentTypes(), bodyTypeMapping)))
        return failure();

      rewriter.applySignatureConversion(loopBody, bodyTypeMapping);
      Region &dstRegion = whileOp.getAfter();
      // TODO: handle uses of coordinate!
      rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());
      ValueRange aArgs = whileOp.getAfterArguments();
      auto yieldOp = llvm::cast<sparse_tensor::YieldOp>(
          whileOp.getAfterBody()->getTerminator());

      rewriter.setInsertionPointToEnd(whileOp.getAfterBody());

      aArgs = it->linkNewScope(aArgs);
      ValueRange nx = it->forward(rewriter, loc);
      SmallVector<Value> yields;
      llvm::append_range(yields, nx);
      llvm::append_range(yields, yieldOp.getResults());

      // replace sparse_tensor.yield with scf.yield.
      yieldOp->erase();
      rewriter.create<scf::YieldOp>(loc, yields);

      const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
      rewriter.replaceOp(
          op, whileOp.getResults().drop_front(it->getCursor().size()),
          resultMapping);
    }
    return success();
  }
};

} // namespace

mlir::SparseIterationTypeConverter::SparseIterationTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertIterSpaceType);
  addConversion(convertIteratorType);
}

void mlir::populateLowerSparseIterationToSCFPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ExtractIterSpaceConverter, SparseIterateOpConverter>(
      converter, patterns.getContext());
}
