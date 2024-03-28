
#include "Utils/CodegenUtils.h"
#include "Utils/SparseTensorLevel.h"

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

} // namespace

mlir::SparseIterationTypeConverter::SparseIterationTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertIterSpaceType);
}

void mlir::populateLowerSparseIterationToSCFPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ExtractIterSpaceConverter>(converter, patterns.getContext());
}
