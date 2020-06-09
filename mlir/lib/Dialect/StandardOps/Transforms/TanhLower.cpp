//===- TanhLower.cpp - Code to perform tanh lowering to appropriate runtime call ---------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Support/DebugStringHelper.h"

using namespace mlir;

#define PASS_NAME "tanh-lower"

namespace {
struct TanhConverter : public OpRewritePattern<TanhOp> {
private:
  static FlatSymbolRefAttr getOrInsertFunc(ModuleOp module, StringRef symbol, Type opType, Location loc) {
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(module.getBodyRegion());
    std::array<Type, 1> inputs{opType};
    ArrayRef<Type> results{opType};
    auto funcType = builder.getFunctionType(inputs, results);
    ArrayRef<NamedAttribute> attrs{};
    builder.create<FuncOp>(loc, symbol, funcType, attrs);
    return SymbolRefAttr::get(symbol, context);
  }

public:
  using OpRewritePattern<TanhOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TanhOp op,
                                PatternRewriter &rewriter) const final {
    Type resultType = op.getResult().getType();
    const char* funcName;
    if (resultType.isF32())
      funcName = static_cast<const char*>("tanhf");
    else if (resultType.isF64())
      funcName = static_cast<const char*>("tanh");
    else
      return failure();
    
    FlatSymbolRefAttr funcToCall = getOrInsertFunc(op.template getParentOfType<ModuleOp>(), funcName, resultType, op.getLoc());
    OpBuilder opBuilder(op);
    SmallVector<Value, 1> args;
    args.push_back(op.getOperand());
    auto newCallOp = opBuilder.create<CallOp>(op.getOperation()->getLoc(), funcToCall,
                               ArrayRef<Type>{resultType}, args);
    rewriter.replaceOp(op, newCallOp.getResult(0));
    return success();
  }
};

struct TanhLower : public TanhLowerBase<TanhLower> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<TanhConverter>(&getContext());

    ConversionTarget target(getContext());
    if (failed(mlir::applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTanhLowerPass() {
  return std::make_unique<TanhLower>();
}
