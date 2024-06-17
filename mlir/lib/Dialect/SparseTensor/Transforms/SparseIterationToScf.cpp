
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

} // namespace

mlir::SparseIterationTypeConverter::SparseIterationTypeConverter() {
  addConversion([](Type type) { return type; });
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
  patterns.add<ExtractIterSpaceConverter>(converter, patterns.getContext());
}
