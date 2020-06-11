//===- StdOpToCall.cpp - lowering std ops to appropriate runtime call -----===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

template <typename OpType>
struct StdOpToCallConverter : public OpRewritePattern<OpType> {
public:
  StdOpToCallConverter(MLIRContext *context, Type matchingType,
                       StringRef funcName)
      : OpRewritePattern<OpType>(context), matchingType(matchingType),
        funcName(funcName) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    auto resultTypes = op.getOperation()->getResultTypes();
    for (auto type : resultTypes) {
      if (type != matchingType)
        return failure();
    }

    auto operands = op.getOperation()->getOperands();
    SmallVector<Type, 2> operandTypes;
    for (auto operand : operands) {
      auto type = operand.getType();
      if (type != matchingType)
        return failure();
      operandTypes.push_back(type);
    }

    auto module = op.template getParentOfType<ModuleOp>();
    auto callee = getOrInsertFunc(module, funcName, resultTypes, operandTypes,
                                  op.getLoc());
    OpBuilder builder(op);
    auto callOp =
        builder.create<CallOp>(op.getLoc(), callee, resultTypes, operands);
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertFunc(ModuleOp module, StringRef symbol,
                                           ArrayRef<Type> resultTypes,
                                           ArrayRef<Type> operandTypes,
                                           Location loc) {
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType(resultTypes, operandTypes);
    ArrayRef<NamedAttribute> attrs;
    builder.create<FuncOp>(loc, symbol, funcType, attrs);
    return SymbolRefAttr::get(symbol, context);
  }

  Type matchingType;
  StringRef funcName;
};

struct TanhLowering : public TanhLoweringBase<TanhLowering> {
  void runOnFunction() override {
    auto &context = getContext();

    OwningRewritePatternList patterns;
    patterns.insert<StdOpToCallConverter<TanhOp>>(
        &context, FloatType::getF32(&context), "tanhf");
    patterns.insert<StdOpToCallConverter<TanhOp>>(
        &context, FloatType::getF64(&context), "tanh");

    ConversionTarget target(context);
    target.addIllegalOp<TanhOp>();
    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTanhLoweringPass() {
  return std::make_unique<TanhLowering>();
}
