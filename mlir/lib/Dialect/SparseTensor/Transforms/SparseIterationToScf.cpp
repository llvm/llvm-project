
#include "Utils/CodegenUtils.h"
#include "Utils/SparseTensorIterator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

void convertLevelType(SparseTensorEncodingAttr enc, Level lvl,
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

    if (it->iteratableByFor()) {
      auto [lo, hi] = it->genForCond(rewriter, loc);
      Value step = constantIndex(rewriter, loc, 1);
      SmallVector<Value> ivs;
      for (ValueRange inits : adaptor.getInitArgs())
        llvm::append_range(ivs, inits);
      scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step, ivs);

      Block *loopBody = op.getBody();
      OneToNTypeMapping bodyTypeMapping(loopBody->getArgumentTypes());
      if (failed(typeConverter->convertSignatureArgs(
              loopBody->getArgumentTypes(), bodyTypeMapping)))
        return failure();
      rewriter.applySignatureConversion(loopBody, bodyTypeMapping);

      rewriter.eraseBlock(forOp.getBody());
      Region &dstRegion = forOp.getRegion();
      rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

      auto yieldOp =
          llvm::cast<sparse_tensor::YieldOp>(forOp.getBody()->getTerminator());

      rewriter.setInsertionPointToEnd(forOp.getBody());
      // replace sparse_tensor.yield with scf.yield.
      rewriter.create<scf::YieldOp>(loc, yieldOp.getResults());
      rewriter.eraseOp(yieldOp);

      const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
      rewriter.replaceOp(op, forOp.getResults(), resultMapping);
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
      rewriter.eraseOp(yieldOp);
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
  patterns.add<ExtractIterSpaceConverter, SparseIterateOpConverter>(
      converter, patterns.getContext());
}
