//===- TestPatterns.cpp - Test dialect pattern driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace test;

// Native function for testing NativeCodeCall
static Value chooseOperand(Value input1, Value input2, BoolAttr choice) {
  return choice.getValue() ? input1 : input2;
}

static void createOpI(PatternRewriter &rewriter, Location loc, Value input) {
  rewriter.create<OpI>(loc, input);
}

static void handleNoResultOp(PatternRewriter &rewriter,
                             OpSymbolBindingNoResult op) {
  // Turn the no result op to a one-result op.
  rewriter.create<OpSymbolBindingB>(op.getLoc(), op.getOperand().getType(),
                                    op.getOperand());
}

static bool getFirstI32Result(Operation *op, Value &value) {
  if (!Type(op->getResult(0).getType()).isSignlessInteger(32))
    return false;
  value = op->getResult(0);
  return true;
}

static Value bindNativeCodeCallResult(Value value) { return value; }

static SmallVector<Value, 2> bindMultipleNativeCodeCallResult(Value input1,
                                                              Value input2) {
  return SmallVector<Value, 2>({input2, input1});
}

// Test that natives calls are only called once during rewrites.
// OpM_Test will return Pi, increased by 1 for each subsequent calls.
// This let us check the number of times OpM_Test was called by inspecting
// the returned value in the MLIR output.
static int64_t opMIncreasingValue = 314159265;
static Attribute opMTest(PatternRewriter &rewriter, Value val) {
  int64_t i = opMIncreasingValue++;
  return rewriter.getIntegerAttr(rewriter.getIntegerType(32), i);
}

namespace {
#include "TestPatterns.inc"
} // namespace

//===----------------------------------------------------------------------===//
// Test Reduce Pattern Interface
//===----------------------------------------------------------------------===//

void test::populateTestReductionPatterns(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

//===----------------------------------------------------------------------===//
// Canonicalizer Driver.
//===----------------------------------------------------------------------===//

namespace {
struct FoldingPattern : public RewritePattern {
public:
  FoldingPattern(MLIRContext *context)
      : RewritePattern(TestOpInPlaceFoldAnchor::getOperationName(),
                       /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Exercise OperationFolder API for a single-result operation that is folded
    // upon construction. The operation being created through the folder has an
    // in-place folder, and it should be still present in the output.
    // Furthermore, the folder should not crash when attempting to recover the
    // (unchanged) operation result.
    OperationFolder folder(op->getContext());
    Value result = folder.create<TestOpInPlaceFold>(
        rewriter, op->getLoc(), rewriter.getIntegerType(32), op->getOperand(0),
        rewriter.getI32IntegerAttr(0));
    assert(result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This pattern creates a foldable operation at the entry point of the block.
/// This tests the situation where the operation folder will need to replace an
/// operation with a previously created constant that does not initially
/// dominate the operation to replace.
struct FolderInsertBeforePreviouslyFoldedConstantPattern
    : public OpRewritePattern<TestCastOp> {
public:
  using OpRewritePattern<TestCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TestCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("test_fold_before_previously_folded_op"))
      return failure();
    rewriter.setInsertionPointToStart(op->getBlock());

    auto constOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getBoolAttr(true));
    rewriter.replaceOpWithNewOp<TestCastOp>(op, rewriter.getI32Type(),
                                            Value(constOp));
    return success();
  }
};

/// This pattern matches test.op_commutative2 with the first operand being
/// another test.op_commutative2 with a constant on the right side and fold it
/// away by propagating it as its result. This is intend to check that patterns
/// are applied after the commutative property moves constant to the right.
struct FolderCommutativeOp2WithConstant
    : public OpRewritePattern<TestCommutative2Op> {
public:
  using OpRewritePattern<TestCommutative2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TestCommutative2Op op,
                                PatternRewriter &rewriter) const override {
    auto operand =
        dyn_cast_or_null<TestCommutative2Op>(op->getOperand(0).getDefiningOp());
    if (!operand)
      return failure();
    Attribute constInput;
    if (!matchPattern(operand->getOperand(1), m_Constant(&constInput)))
      return failure();
    rewriter.replaceOp(op, operand->getOperand(1));
    return success();
  }
};

/// This pattern matches test.any_attr_of_i32_str ops. In case of an integer
/// attribute with value smaller than MaxVal, it increments the value by 1.
template <int MaxVal>
struct IncrementIntAttribute : public OpRewritePattern<AnyAttrOfOp> {
  using OpRewritePattern<AnyAttrOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AnyAttrOfOp op,
                                PatternRewriter &rewriter) const override {
    auto intAttr = op.getAttr().dyn_cast<IntegerAttr>();
    if (!intAttr)
      return failure();
    int64_t val = intAttr.getInt();
    if (val >= MaxVal)
      return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op.setAttrAttr(rewriter.getI32IntegerAttr(val + 1)); });
    return success();
  }
};

/// This patterns adds an "eligible" attribute to "foo.maybe_eligible_op".
struct MakeOpEligible : public RewritePattern {
  MakeOpEligible(MLIRContext *context)
      : RewritePattern("foo.maybe_eligible_op", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("eligible"))
      return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op->setAttr("eligible", rewriter.getUnitAttr()); });
    return success();
  }
};

/// This pattern hoists eligible ops out of a "test.one_region_op".
struct HoistEligibleOps : public OpRewritePattern<test::OneRegionOp> {
  using OpRewritePattern<test::OneRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(test::OneRegionOp op,
                                PatternRewriter &rewriter) const override {
    Operation *terminator = op.getRegion().front().getTerminator();
    Operation *toBeHoisted = terminator->getOperands()[0].getDefiningOp();
    if (toBeHoisted->getParentOp() != op)
      return failure();
    if (!toBeHoisted->hasAttr("eligible"))
      return failure();
    toBeHoisted->moveBefore(op);
    return success();
  }
};

struct TestPatternDriver
    : public PassWrapper<TestPatternDriver, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPatternDriver)

  TestPatternDriver() = default;
  TestPatternDriver(const TestPatternDriver &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-patterns"; }
  StringRef getDescription() const final { return "Run test dialect patterns"; }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);

    // Verify named pattern is generated with expected name.
    patterns.add<FoldingPattern, TestNamedPatternRule,
                 FolderInsertBeforePreviouslyFoldedConstantPattern,
                 FolderCommutativeOp2WithConstant, HoistEligibleOps,
                 MakeOpEligible>(&getContext());

    // Additional patterns for testing the GreedyPatternRewriteDriver.
    patterns.insert<IncrementIntAttribute<3>>(&getContext());

    GreedyRewriteConfig config;
    config.useTopDownTraversal = this->useTopDownTraversal;
    config.maxIterations = this->maxIterations;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }

  Option<bool> useTopDownTraversal{
      *this, "top-down",
      llvm::cl::desc("Seed the worklist in general top-down order"),
      llvm::cl::init(GreedyRewriteConfig().useTopDownTraversal)};
  Option<int> maxIterations{
      *this, "max-iterations",
      llvm::cl::desc("Max. iterations in the GreedyRewriteConfig"),
      llvm::cl::init(GreedyRewriteConfig().maxIterations)};
};

struct TestStrictPatternDriver
    : public PassWrapper<TestStrictPatternDriver, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStrictPatternDriver)

  TestStrictPatternDriver() = default;
  TestStrictPatternDriver(const TestStrictPatternDriver &other) {
    strictMode = other.strictMode;
  }

  StringRef getArgument() const final { return "test-strict-pattern-driver"; }
  StringRef getDescription() const final {
    return "Test strict mode of pattern driver";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<InsertSameOp, ReplaceWithNewOp, EraseOp>(ctx);
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (opName == "test.insert_same_op" ||
          opName == "test.replace_with_new_op" || opName == "test.erase_op") {
        ops.push_back(op);
      }
    });

    GreedyRewriteStrictness mode;
    if (strictMode == "AnyOp") {
      mode = GreedyRewriteStrictness::AnyOp;
    } else if (strictMode == "ExistingAndNewOps") {
      mode = GreedyRewriteStrictness::ExistingAndNewOps;
    } else if (strictMode == "ExistingOps") {
      mode = GreedyRewriteStrictness::ExistingOps;
    } else {
      llvm_unreachable("invalid strictness option");
    }

    // Check if these transformations introduce visiting of operations that
    // are not in the `ops` set (The new created ops are valid). An invalid
    // operation will trigger the assertion while processing.
    bool changed = false;
    bool allErased = false;
    (void)applyOpPatternsAndFold(ArrayRef(ops), std::move(patterns), mode,
                                 &changed, &allErased);
    Builder b(ctx);
    getOperation()->setAttr("pattern_driver_changed", b.getBoolAttr(changed));
    getOperation()->setAttr("pattern_driver_all_erased",
                            b.getBoolAttr(allErased));
  }

  Option<std::string> strictMode{
      *this, "strictness",
      llvm::cl::desc("Can be {AnyOp, ExistingAndNewOps, ExistingOps}"),
      llvm::cl::init("AnyOp")};

private:
  // New inserted operation is valid for further transformation.
  class InsertSameOp : public RewritePattern {
  public:
    InsertSameOp(MLIRContext *context)
        : RewritePattern("test.insert_same_op", /*benefit=*/1, context) {}

    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
      if (op->hasAttr("skip"))
        return failure();

      Operation *newOp =
          rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                          op->getOperands(), op->getResultTypes());
      op->setAttr("skip", rewriter.getBoolAttr(true));
      newOp->setAttr("skip", rewriter.getBoolAttr(true));

      return success();
    }
  };

  // Replace an operation may introduce the re-visiting of its users.
  class ReplaceWithNewOp : public RewritePattern {
  public:
    ReplaceWithNewOp(MLIRContext *context)
        : RewritePattern("test.replace_with_new_op", /*benefit=*/1, context) {}

    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
      Operation *newOp;
      if (op->hasAttr("create_erase_op")) {
        newOp = rewriter.create(
            op->getLoc(),
            OperationName("test.erase_op", op->getContext()).getIdentifier(),
            ValueRange(), TypeRange());
      } else {
        newOp = rewriter.create(
            op->getLoc(),
            OperationName("test.new_op", op->getContext()).getIdentifier(),
            op->getOperands(), op->getResultTypes());
      }
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }
  };

  // Remove an operation may introduce the re-visiting of its opreands.
  class EraseOp : public RewritePattern {
  public:
    EraseOp(MLIRContext *context)
        : RewritePattern("test.erase_op", /*benefit=*/1, context) {}
    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
      rewriter.eraseOp(op);
      return success();
    }
  };
};

} // namespace

//===----------------------------------------------------------------------===//
// ReturnType Driver.
//===----------------------------------------------------------------------===//

namespace {
// Generate ops for each instance where the type can be successfully inferred.
template <typename OpTy>
static void invokeCreateWithInferredReturnType(Operation *op) {
  auto *context = op->getContext();
  auto fop = op->getParentOfType<func::FuncOp>();
  auto location = UnknownLoc::get(context);
  OpBuilder b(op);
  b.setInsertionPointAfter(op);

  // Use permutations of 2 args as operands.
  assert(fop.getNumArguments() >= 2);
  for (int i = 0, e = fop.getNumArguments(); i < e; ++i) {
    for (int j = 0; j < e; ++j) {
      std::array<Value, 2> values = {{fop.getArgument(i), fop.getArgument(j)}};
      SmallVector<Type, 2> inferredReturnTypes;
      if (succeeded(OpTy::inferReturnTypes(
              context, std::nullopt, values, op->getAttrDictionary(),
              op->getRegions(), inferredReturnTypes))) {
        OperationState state(location, OpTy::getOperationName());
        // TODO: Expand to regions.
        OpTy::build(b, state, values, op->getAttrs());
        (void)b.create(state);
      }
    }
  }
}

static void reifyReturnShape(Operation *op) {
  OpBuilder b(op);

  // Use permutations of 2 args as operands.
  auto shapedOp = cast<OpWithShapedTypeInferTypeInterfaceOp>(op);
  SmallVector<Value, 2> shapes;
  if (failed(shapedOp.reifyReturnTypeShapes(b, op->getOperands(), shapes)) ||
      !llvm::hasSingleElement(shapes))
    return;
  for (const auto &it : llvm::enumerate(shapes)) {
    op->emitRemark() << "value " << it.index() << ": "
                     << it.value().getDefiningOp();
  }
}

struct TestReturnTypeDriver
    : public PassWrapper<TestReturnTypeDriver, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReturnTypeDriver)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  StringRef getArgument() const final { return "test-return-type"; }
  StringRef getDescription() const final { return "Run return type functions"; }

  void runOnOperation() override {
    if (getOperation().getName() == "testCreateFunctions") {
      std::vector<Operation *> ops;
      // Collect ops to avoid triggering on inserted ops.
      for (auto &op : getOperation().getBody().front())
        ops.push_back(&op);
      // Generate test patterns for each, but skip terminator.
      for (auto *op : llvm::ArrayRef(ops).drop_back()) {
        // Test create method of each of the Op classes below. The resultant
        // output would be in reverse order underneath `op` from which
        // the attributes and regions are used.
        invokeCreateWithInferredReturnType<OpWithInferTypeInterfaceOp>(op);
        invokeCreateWithInferredReturnType<
            OpWithShapedTypeInferTypeInterfaceOp>(op);
      };
      return;
    }
    if (getOperation().getName() == "testReifyFunctions") {
      std::vector<Operation *> ops;
      // Collect ops to avoid triggering on inserted ops.
      for (auto &op : getOperation().getBody().front())
        if (isa<OpWithShapedTypeInferTypeInterfaceOp>(op))
          ops.push_back(&op);
      // Generate test patterns for each, but skip terminator.
      for (auto *op : ops)
        reifyReturnShape(op);
    }
  }
};
} // namespace

namespace {
struct TestDerivedAttributeDriver
    : public PassWrapper<TestDerivedAttributeDriver,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDerivedAttributeDriver)

  StringRef getArgument() const final { return "test-derived-attr"; }
  StringRef getDescription() const final {
    return "Run test derived attributes";
  }
  void runOnOperation() override;
};
} // namespace

void TestDerivedAttributeDriver::runOnOperation() {
  getOperation().walk([](DerivedAttributeOpInterface dOp) {
    auto dAttr = dOp.materializeDerivedAttributes();
    if (!dAttr)
      return;
    for (auto d : dAttr)
      dOp.emitRemark() << d.getName().getValue() << " = " << d.getValue();
  });
}

//===----------------------------------------------------------------------===//
// Legalization Driver.
//===----------------------------------------------------------------------===//

namespace {
//===----------------------------------------------------------------------===//
// Region-Block Rewrite Testing

/// This pattern is a simple pattern that inlines the first region of a given
/// operation into the parent region.
struct TestRegionRewriteBlockMovement : public ConversionPattern {
  TestRegionRewriteBlockMovement(MLIRContext *ctx)
      : ConversionPattern("test.region", 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Inline this region into the parent region.
    auto &parentRegion = *op->getParentRegion();
    auto &opRegion = op->getRegion(0);
    if (op->getAttr("legalizer.should_clone"))
      rewriter.cloneRegionBefore(opRegion, parentRegion, parentRegion.end());
    else
      rewriter.inlineRegionBefore(opRegion, parentRegion, parentRegion.end());

    if (op->getAttr("legalizer.erase_old_blocks")) {
      while (!opRegion.empty())
        rewriter.eraseBlock(&opRegion.front());
    }

    // Drop this operation.
    rewriter.eraseOp(op);
    return success();
  }
};
/// This pattern is a simple pattern that generates a region containing an
/// illegal operation.
struct TestRegionRewriteUndo : public RewritePattern {
  TestRegionRewriteUndo(MLIRContext *ctx)
      : RewritePattern("test.region_builder", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    // Create the region operation with an entry block containing arguments.
    OperationState newRegion(op->getLoc(), "test.region");
    newRegion.addRegion();
    auto *regionOp = rewriter.create(newRegion);
    auto *entryBlock = rewriter.createBlock(&regionOp->getRegion(0));
    entryBlock->addArgument(rewriter.getIntegerType(64),
                            rewriter.getUnknownLoc());

    // Add an explicitly illegal operation to ensure the conversion fails.
    rewriter.create<ILLegalOpF>(op->getLoc(), rewriter.getIntegerType(32));
    rewriter.create<TestValidOp>(op->getLoc(), ArrayRef<Value>());

    // Drop this operation.
    rewriter.eraseOp(op);
    return success();
  }
};
/// A simple pattern that creates a block at the end of the parent region of the
/// matched operation.
struct TestCreateBlock : public RewritePattern {
  TestCreateBlock(MLIRContext *ctx)
      : RewritePattern("test.create_block", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    Region &region = *op->getParentRegion();
    Type i32Type = rewriter.getIntegerType(32);
    Location loc = op->getLoc();
    rewriter.createBlock(&region, region.end(), {i32Type, i32Type}, {loc, loc});
    rewriter.create<TerminatorOp>(loc);
    rewriter.replaceOp(op, {});
    return success();
  }
};

/// A simple pattern that creates a block containing an invalid operation in
/// order to trigger the block creation undo mechanism.
struct TestCreateIllegalBlock : public RewritePattern {
  TestCreateIllegalBlock(MLIRContext *ctx)
      : RewritePattern("test.create_illegal_block", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    Region &region = *op->getParentRegion();
    Type i32Type = rewriter.getIntegerType(32);
    Location loc = op->getLoc();
    rewriter.createBlock(&region, region.end(), {i32Type, i32Type}, {loc, loc});
    // Create an illegal op to ensure the conversion fails.
    rewriter.create<ILLegalOpF>(loc, i32Type);
    rewriter.create<TerminatorOp>(loc);
    rewriter.replaceOp(op, {});
    return success();
  }
};

/// A simple pattern that tests the undo mechanism when replacing the uses of a
/// block argument.
struct TestUndoBlockArgReplace : public ConversionPattern {
  TestUndoBlockArgReplace(MLIRContext *ctx)
      : ConversionPattern("test.undo_block_arg_replace", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto illegalOp =
        rewriter.create<ILLegalOpF>(op->getLoc(), rewriter.getF32Type());
    rewriter.replaceUsesOfBlockArgument(op->getRegion(0).getArgument(0),
                                        illegalOp);
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

/// A rewrite pattern that tests the undo mechanism when erasing a block.
struct TestUndoBlockErase : public ConversionPattern {
  TestUndoBlockErase(MLIRContext *ctx)
      : ConversionPattern("test.undo_block_erase", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Block *secondBlock = &*std::next(op->getRegion(0).begin());
    rewriter.setInsertionPointToStart(secondBlock);
    rewriter.create<ILLegalOpF>(op->getLoc(), rewriter.getF32Type());
    rewriter.eraseBlock(secondBlock);
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Type-Conversion Rewrite Testing

/// This patterns erases a region operation that has had a type conversion.
struct TestDropOpSignatureConversion : public ConversionPattern {
  TestDropOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, "test.drop_region_op", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Region &region = op->getRegion(0);
    Block *entry = &region.front();

    // Convert the original entry arguments.
    TypeConverter &converter = *getTypeConverter();
    TypeConverter::SignatureConversion result(entry->getNumArguments());
    if (failed(converter.convertSignatureArgs(entry->getArgumentTypes(),
                                              result)) ||
        failed(rewriter.convertRegionTypes(&region, converter, &result)))
      return failure();

    // Convert the region signature and just drop the operation.
    rewriter.eraseOp(op);
    return success();
  }
};
/// This pattern simply updates the operands of the given operation.
struct TestPassthroughInvalidOp : public ConversionPattern {
  TestPassthroughInvalidOp(MLIRContext *ctx)
      : ConversionPattern("test.invalid", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TestValidOp>(op, std::nullopt, operands,
                                             std::nullopt);
    return success();
  }
};
/// This pattern handles the case of a split return value.
struct TestSplitReturnType : public ConversionPattern {
  TestSplitReturnType(MLIRContext *ctx)
      : ConversionPattern("test.return", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Check for a return of F32.
    if (op->getNumOperands() != 1 || !op->getOperand(0).getType().isF32())
      return failure();

    // Check if the first operation is a cast operation, if it is we use the
    // results directly.
    auto *defOp = operands[0].getDefiningOp();
    if (auto packerOp =
            llvm::dyn_cast_or_null<UnrealizedConversionCastOp>(defOp)) {
      rewriter.replaceOpWithNewOp<TestReturnOp>(op, packerOp.getOperands());
      return success();
    }

    // Otherwise, fail to match.
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Multi-Level Type-Conversion Rewrite Testing
struct TestChangeProducerTypeI32ToF32 : public ConversionPattern {
  TestChangeProducerTypeI32ToF32(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is I32, change the type to F32.
    if (!Type(*op->result_type_begin()).isSignlessInteger(32))
      return failure();
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF32Type());
    return success();
  }
};
struct TestChangeProducerTypeF32ToF64 : public ConversionPattern {
  TestChangeProducerTypeF32ToF64(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is F32, change the type to F64.
    if (!Type(*op->result_type_begin()).isF32())
      return rewriter.notifyMatchFailure(op, "expected single f32 operand");
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF64Type());
    return success();
  }
};
struct TestChangeProducerTypeF32ToInvalid : public ConversionPattern {
  TestChangeProducerTypeF32ToInvalid(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 10, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Always convert to B16, even though it is not a legal type. This tests
    // that values are unmapped correctly.
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getBF16Type());
    return success();
  }
};
struct TestUpdateConsumerType : public ConversionPattern {
  TestUpdateConsumerType(MLIRContext *ctx)
      : ConversionPattern("test.type_consumer", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Verify that the incoming operand has been successfully remapped to F64.
    if (!operands[0].getType().isF64())
      return failure();
    rewriter.replaceOpWithNewOp<TestTypeConsumerOp>(op, operands[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Non-Root Replacement Rewrite Testing
/// This pattern generates an invalid operation, but replaces it before the
/// pattern is finished. This checks that we don't need to legalize the
/// temporary op.
struct TestNonRootReplacement : public RewritePattern {
  TestNonRootReplacement(MLIRContext *ctx)
      : RewritePattern("test.replace_non_root", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto resultType = *op->result_type_begin();
    auto illegalOp = rewriter.create<ILLegalOpF>(op->getLoc(), resultType);
    auto legalOp = rewriter.create<LegalOpB>(op->getLoc(), resultType);

    rewriter.replaceOp(illegalOp, {legalOp});
    rewriter.replaceOp(op, {illegalOp});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Recursive Rewrite Testing
/// This pattern is applied to the same operation multiple times, but has a
/// bounded recursion.
struct TestBoundedRecursiveRewrite
    : public OpRewritePattern<TestRecursiveRewriteOp> {
  using OpRewritePattern<TestRecursiveRewriteOp>::OpRewritePattern;

  void initialize() {
    // The conversion target handles bounding the recursion of this pattern.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(TestRecursiveRewriteOp op,
                                PatternRewriter &rewriter) const final {
    // Decrement the depth of the op in-place.
    rewriter.updateRootInPlace(op, [&] {
      op->setAttr("depth", rewriter.getI64IntegerAttr(op.getDepth() - 1));
    });
    return success();
  }
};

struct TestNestedOpCreationUndoRewrite
    : public OpRewritePattern<IllegalOpWithRegionAnchor> {
  using OpRewritePattern<IllegalOpWithRegionAnchor>::OpRewritePattern;

  LogicalResult matchAndRewrite(IllegalOpWithRegionAnchor op,
                                PatternRewriter &rewriter) const final {
    // rewriter.replaceOpWithNewOp<IllegalOpWithRegion>(op);
    rewriter.replaceOpWithNewOp<IllegalOpWithRegion>(op);
    return success();
  };
};

// This pattern matches `test.blackhole` and delete this op and its producer.
struct TestReplaceEraseOp : public OpRewritePattern<BlackHoleOp> {
  using OpRewritePattern<BlackHoleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlackHoleOp op,
                                PatternRewriter &rewriter) const final {
    Operation *producer = op.getOperand().getDefiningOp();
    // Always erase the user before the producer, the framework should handle
    // this correctly.
    rewriter.eraseOp(op);
    rewriter.eraseOp(producer);
    return success();
  };
};

// This pattern replaces explicitly illegal op with explicitly legal op,
// but in addition creates unregistered operation.
struct TestCreateUnregisteredOp : public OpRewritePattern<ILLegalOpG> {
  using OpRewritePattern<ILLegalOpG>::OpRewritePattern;

  LogicalResult matchAndRewrite(ILLegalOpG op,
                                PatternRewriter &rewriter) const final {
    IntegerAttr attr = rewriter.getI32IntegerAttr(0);
    Value val = rewriter.create<arith::ConstantOp>(op->getLoc(), attr);
    rewriter.replaceOpWithNewOp<LegalOpC>(op, val);
    return success();
  };
};
} // namespace

namespace {
struct TestTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;
  TestTypeConverter() {
    addConversion(convertType);
    addArgumentMaterialization(materializeCast);
    addSourceMaterialization(materializeCast);
  }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    // Drop I16 types.
    if (t.isSignlessInteger(16))
      return success();

    // Convert I64 to F64.
    if (t.isSignlessInteger(64)) {
      results.push_back(FloatType::getF64(t.getContext()));
      return success();
    }

    // Convert I42 to I43.
    if (t.isInteger(42)) {
      results.push_back(IntegerType::get(t.getContext(), 43));
      return success();
    }

    // Split F32 into F16,F16.
    if (t.isF32()) {
      results.assign(2, FloatType::getF16(t.getContext()));
      return success();
    }

    // Otherwise, convert the type directly.
    results.push_back(t);
    return success();
  }

  /// Hook for materializing a conversion. This is necessary because we generate
  /// 1->N type mappings.
  static std::optional<Value> materializeCast(OpBuilder &builder,
                                              Type resultType,
                                              ValueRange inputs, Location loc) {
    return builder.create<TestCastOp>(loc, resultType, inputs).getResult();
  }
};

struct TestLegalizePatternDriver
    : public PassWrapper<TestLegalizePatternDriver, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLegalizePatternDriver)

  StringRef getArgument() const final { return "test-legalize-patterns"; }
  StringRef getDescription() const final {
    return "Run test dialect legalization patterns";
  }
  /// The mode of conversion to use with the driver.
  enum class ConversionMode { Analysis, Full, Partial };

  TestLegalizePatternDriver(ConversionMode mode) : mode(mode) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, test::TestDialect>();
  }

  void runOnOperation() override {
    TestTypeConverter converter;
    mlir::RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    patterns
        .add<TestRegionRewriteBlockMovement, TestRegionRewriteUndo,
             TestCreateBlock, TestCreateIllegalBlock, TestUndoBlockArgReplace,
             TestUndoBlockErase, TestPassthroughInvalidOp, TestSplitReturnType,
             TestChangeProducerTypeI32ToF32, TestChangeProducerTypeF32ToF64,
             TestChangeProducerTypeF32ToInvalid, TestUpdateConsumerType,
             TestNonRootReplacement, TestBoundedRecursiveRewrite,
             TestNestedOpCreationUndoRewrite, TestReplaceEraseOp,
             TestCreateUnregisteredOp>(&getContext());
    patterns.add<TestDropOpSignatureConversion>(&getContext(), converter);
    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                              converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);

    // Define the conversion target used for the test.
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<LegalOpA, LegalOpB, LegalOpC, TestCastOp, TestValidOp,
                      TerminatorOp>();
    target
        .addIllegalOp<ILLegalOpF, TestRegionBuilderOp, TestOpWithRegionFold>();
    target.addDynamicallyLegalOp<TestReturnOp>([](TestReturnOp op) {
      // Don't allow F32 operands.
      return llvm::none_of(op.getOperandTypes(),
                           [](Type type) { return type.isF32(); });
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return converter.isLegal(op); });

    // TestCreateUnregisteredOp creates `arith.constant` operation,
    // which was not added to target intentionally to test
    // correct error code from conversion driver.
    target.addDynamicallyLegalOp<ILLegalOpG>([](ILLegalOpG) { return false; });

    // Expect the type_producer/type_consumer operations to only operate on f64.
    target.addDynamicallyLegalOp<TestTypeProducerOp>(
        [](TestTypeProducerOp op) { return op.getType().isF64(); });
    target.addDynamicallyLegalOp<TestTypeConsumerOp>([](TestTypeConsumerOp op) {
      return op.getOperand().getType().isF64();
    });

    // Check support for marking certain operations as recursively legal.
    target.markOpRecursivelyLegal<func::FuncOp, ModuleOp>([](Operation *op) {
      return static_cast<bool>(
          op->getAttrOfType<UnitAttr>("test.recursively_legal"));
    });

    // Mark the bound recursion operation as dynamically legal.
    target.addDynamicallyLegalOp<TestRecursiveRewriteOp>(
        [](TestRecursiveRewriteOp op) { return op.getDepth() == 0; });

    // Handle a partial conversion.
    if (mode == ConversionMode::Partial) {
      DenseSet<Operation *> unlegalizedOps;
      if (failed(applyPartialConversion(
              getOperation(), target, std::move(patterns), &unlegalizedOps))) {
        getOperation()->emitRemark() << "applyPartialConversion failed";
      }
      // Emit remarks for each legalizable operation.
      for (auto *op : unlegalizedOps)
        op->emitRemark() << "op '" << op->getName() << "' is not legalizable";
      return;
    }

    // Handle a full conversion.
    if (mode == ConversionMode::Full) {
      // Check support for marking unknown operations as dynamically legal.
      target.markUnknownOpDynamicallyLegal([](Operation *op) {
        return (bool)op->getAttrOfType<UnitAttr>("test.dynamically_legal");
      });

      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        getOperation()->emitRemark() << "applyFullConversion failed";
      }
      return;
    }

    // Otherwise, handle an analysis conversion.
    assert(mode == ConversionMode::Analysis);

    // Analyze the convertible operations.
    DenseSet<Operation *> legalizedOps;
    if (failed(applyAnalysisConversion(getOperation(), target,
                                       std::move(patterns), legalizedOps)))
      return signalPassFailure();

    // Emit remarks for each legalizable operation.
    for (auto *op : legalizedOps)
      op->emitRemark() << "op '" << op->getName() << "' is legalizable";
  }

  /// The mode of conversion to use.
  ConversionMode mode;
};
} // namespace

static llvm::cl::opt<TestLegalizePatternDriver::ConversionMode>
    legalizerConversionMode(
        "test-legalize-mode",
        llvm::cl::desc("The legalization mode to use with the test driver"),
        llvm::cl::init(TestLegalizePatternDriver::ConversionMode::Partial),
        llvm::cl::values(
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Analysis,
                       "analysis", "Perform an analysis conversion"),
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Full, "full",
                       "Perform a full conversion"),
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Partial,
                       "partial", "Perform a partial conversion")));

//===----------------------------------------------------------------------===//
// ConversionPatternRewriter::getRemappedValue testing. This method is used
// to get the remapped value of an original value that was replaced using
// ConversionPatternRewriter.
namespace {
struct TestRemapValueTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  TestRemapValueTypeConverter() {
    addConversion(
        [](Float32Type type) { return Float64Type::get(type.getContext()); });
    addConversion([](Type type) { return type; });
  }
};

/// Converter that replaces a one-result one-operand OneVResOneVOperandOp1 with
/// a one-operand two-result OneVResOneVOperandOp1 by replicating its original
/// operand twice.
///
/// Example:
///   %1 = test.one_variadic_out_one_variadic_in1"(%0)
/// is replaced with:
///   %1 = test.one_variadic_out_one_variadic_in1"(%0, %0)
struct OneVResOneVOperandOp1Converter
    : public OpConversionPattern<OneVResOneVOperandOp1> {
  using OpConversionPattern<OneVResOneVOperandOp1>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OneVResOneVOperandOp1 op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origOps = op.getOperands();
    assert(std::distance(origOps.begin(), origOps.end()) == 1 &&
           "One operand expected");
    Value origOp = *origOps.begin();
    SmallVector<Value, 2> remappedOperands;
    // Replicate the remapped original operand twice. Note that we don't used
    // the remapped 'operand' since the goal is testing 'getRemappedValue'.
    remappedOperands.push_back(rewriter.getRemappedValue(origOp));
    remappedOperands.push_back(rewriter.getRemappedValue(origOp));

    rewriter.replaceOpWithNewOp<OneVResOneVOperandOp1>(op, op.getResultTypes(),
                                                       remappedOperands);
    return success();
  }
};

/// A rewriter pattern that tests that blocks can be merged.
struct TestRemapValueInRegion
    : public OpConversionPattern<TestRemappedValueRegionOp> {
  using OpConversionPattern<TestRemappedValueRegionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestRemappedValueRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block &block = op.getBody().front();
    Operation *terminator = block.getTerminator();

    // Merge the block into the parent region.
    Block *parentBlock = op->getBlock();
    Block *finalBlock = rewriter.splitBlock(parentBlock, op->getIterator());
    rewriter.mergeBlocks(&block, parentBlock, ValueRange());
    rewriter.mergeBlocks(finalBlock, parentBlock, ValueRange());

    // Replace the results of this operation with the remapped terminator
    // values.
    SmallVector<Value> terminatorOperands;
    if (failed(rewriter.getRemappedValues(terminator->getOperands(),
                                          terminatorOperands)))
      return failure();

    rewriter.eraseOp(terminator);
    rewriter.replaceOp(op, terminatorOperands);
    return success();
  }
};

struct TestRemappedValue
    : public mlir::PassWrapper<TestRemappedValue, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRemappedValue)

  StringRef getArgument() const final { return "test-remapped-value"; }
  StringRef getDescription() const final {
    return "Test public remapped value mechanism in ConversionPatternRewriter";
  }
  void runOnOperation() override {
    TestRemapValueTypeConverter typeConverter;

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<OneVResOneVOperandOp1Converter>(&getContext());
    patterns.add<TestChangeProducerTypeF32ToF64, TestUpdateConsumerType>(
        &getContext());
    patterns.add<TestRemapValueInRegion>(typeConverter, &getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, func::FuncOp, TestReturnOp>();

    // Expect the type_producer/type_consumer operations to only operate on f64.
    target.addDynamicallyLegalOp<TestTypeProducerOp>(
        [](TestTypeProducerOp op) { return op.getType().isF64(); });
    target.addDynamicallyLegalOp<TestTypeConsumerOp>([](TestTypeConsumerOp op) {
      return op.getOperand().getType().isF64();
    });

    // We make OneVResOneVOperandOp1 legal only when it has more that one
    // operand. This will trigger the conversion that will replace one-operand
    // OneVResOneVOperandOp1 with two-operand OneVResOneVOperandOp1.
    target.addDynamicallyLegalOp<OneVResOneVOperandOp1>(
        [](Operation *op) { return op->getNumOperands() > 1; });

    if (failed(mlir::applyFullConversion(getOperation(), target,
                                         std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Test patterns without a specific root operation kind
//===----------------------------------------------------------------------===//

namespace {
/// This pattern matches and removes any operation in the test dialect.
struct RemoveTestDialectOps : public RewritePattern {
  RemoveTestDialectOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<TestDialect>(op->getDialect()))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct TestUnknownRootOpDriver
    : public mlir::PassWrapper<TestUnknownRootOpDriver,
                               OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUnknownRootOpDriver)

  StringRef getArgument() const final {
    return "test-legalize-unknown-root-patterns";
  }
  StringRef getDescription() const final {
    return "Test public remapped value mechanism in ConversionPatternRewriter";
  }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RemoveTestDialectOps>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<TestDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Test patterns that uses operations and types defined at runtime
//===----------------------------------------------------------------------===//

namespace {
/// This pattern matches dynamic operations 'test.one_operand_two_results' and
/// replace them with dynamic operations 'test.generic_dynamic_op'.
struct RewriteDynamicOp : public RewritePattern {
  RewriteDynamicOp(MLIRContext *context)
      : RewritePattern("test.dynamic_one_operand_two_results", /*benefit=*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    assert(op->getName().getStringRef() ==
               "test.dynamic_one_operand_two_results" &&
           "rewrite pattern should only match operations with the right name");

    OperationState state(op->getLoc(), "test.dynamic_generic",
                         op->getOperands(), op->getResultTypes(),
                         op->getAttrs());
    auto *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct TestRewriteDynamicOpDriver
    : public PassWrapper<TestRewriteDynamicOpDriver,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRewriteDynamicOpDriver)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TestDialect>();
  }
  StringRef getArgument() const final { return "test-rewrite-dynamic-op"; }
  StringRef getDescription() const final {
    return "Test rewritting on dynamic operations";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteDynamicOp>(&getContext());

    ConversionTarget target(getContext());
    target.addIllegalOp(
        OperationName("test.dynamic_one_operand_two_results", &getContext()));
    target.addLegalOp(OperationName("test.dynamic_generic", &getContext()));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Test type conversions
//===----------------------------------------------------------------------===//

namespace {
struct TestTypeConversionProducer
    : public OpConversionPattern<TestTypeProducerOp> {
  using OpConversionPattern<TestTypeProducerOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TestTypeProducerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = op.getType();
    Type convertedType = getTypeConverter()
                             ? getTypeConverter()->convertType(resultType)
                             : resultType;
    if (resultType.isa<FloatType>())
      resultType = rewriter.getF64Type();
    else if (resultType.isInteger(16))
      resultType = rewriter.getIntegerType(64);
    else if (resultType.isa<test::TestRecursiveType>() &&
             convertedType != resultType)
      resultType = convertedType;
    else
      return failure();

    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, resultType);
    return success();
  }
};

/// Call signature conversion and then fail the rewrite to trigger the undo
/// mechanism.
struct TestSignatureConversionUndo
    : public OpConversionPattern<TestSignatureConversionUndoOp> {
  using OpConversionPattern<TestSignatureConversionUndoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestSignatureConversionUndoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    (void)rewriter.convertRegionTypes(&op->getRegion(0), *getTypeConverter());
    return failure();
  }
};

/// Call signature conversion without providing a type converter to handle
/// materializations.
struct TestTestSignatureConversionNoConverter
    : public OpConversionPattern<TestSignatureConversionNoConverterOp> {
  TestTestSignatureConversionNoConverter(TypeConverter &converter,
                                         MLIRContext *context)
      : OpConversionPattern<TestSignatureConversionNoConverterOp>(context),
        converter(converter) {}

  LogicalResult
  matchAndRewrite(TestSignatureConversionNoConverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Region &region = op->getRegion(0);
    Block *entry = &region.front();

    // Convert the original entry arguments.
    TypeConverter::SignatureConversion result(entry->getNumArguments());
    if (failed(
            converter.convertSignatureArgs(entry->getArgumentTypes(), result)))
      return failure();
    rewriter.updateRootInPlace(
        op, [&] { rewriter.applySignatureConversion(&region, result); });
    return success();
  }

  TypeConverter &converter;
};

/// Just forward the operands to the root op. This is essentially a no-op
/// pattern that is used to trigger target materialization.
struct TestTypeConsumerForward
    : public OpConversionPattern<TestTypeConsumerOp> {
  using OpConversionPattern<TestTypeConsumerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestTypeConsumerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct TestTypeConversionAnotherProducer
    : public OpRewritePattern<TestAnotherTypeProducerOp> {
  using OpRewritePattern<TestAnotherTypeProducerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TestAnotherTypeProducerOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, op.getType());
    return success();
  }
};

struct TestTypeConversionDriver
    : public PassWrapper<TestTypeConversionDriver, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeConversionDriver)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TestDialect>();
  }
  StringRef getArgument() const final {
    return "test-legalize-type-conversion";
  }
  StringRef getDescription() const final {
    return "Test various type conversion functionalities in DialectConversion";
  }

  void runOnOperation() override {
    // Initialize the type converter.
    TypeConverter converter;

    /// Add the legal set of type conversions.
    converter.addConversion([](Type type) -> Type {
      // Treat F64 as legal.
      if (type.isF64())
        return type;
      // Allow converting BF16/F16/F32 to F64.
      if (type.isBF16() || type.isF16() || type.isF32())
        return FloatType::getF64(type.getContext());
      // Otherwise, the type is illegal.
      return nullptr;
    });
    converter.addConversion([](IntegerType type, SmallVectorImpl<Type> &) {
      // Drop all integer types.
      return success();
    });
    converter.addConversion(
        // Convert a recursive self-referring type into a non-self-referring
        // type named "outer_converted_type" that contains a SimpleAType.
        [&](test::TestRecursiveType type, SmallVectorImpl<Type> &results,
            ArrayRef<Type> callStack) -> std::optional<LogicalResult> {
          // If the type is already converted, return it to indicate that it is
          // legal.
          if (type.getName() == "outer_converted_type") {
            results.push_back(type);
            return success();
          }

          // If the type is on the call stack more than once (it is there at
          // least once because of the _current_ call, which is always the last
          // element on the stack), we've hit the recursive case. Just return
          // SimpleAType here to create a non-recursive type as a result.
          if (llvm::is_contained(callStack.drop_back(), type)) {
            results.push_back(test::SimpleAType::get(type.getContext()));
            return success();
          }

          // Convert the body recursively.
          auto result = test::TestRecursiveType::get(type.getContext(),
                                                     "outer_converted_type");
          if (failed(result.setBody(converter.convertType(type.getBody()))))
            return failure();
          results.push_back(result);
          return success();
        });

    /// Add the legal set of type materializations.
    converter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      // Allow casting from F64 back to F32.
      if (!resultType.isF16() && inputs.size() == 1 &&
          inputs[0].getType().isF64())
        return builder.create<TestCastOp>(loc, resultType, inputs).getResult();
      // Allow producing an i32 or i64 from nothing.
      if ((resultType.isInteger(32) || resultType.isInteger(64)) &&
          inputs.empty())
        return builder.create<TestTypeProducerOp>(loc, resultType);
      // Allow producing an i64 from an integer.
      if (resultType.isa<IntegerType>() && inputs.size() == 1 &&
          inputs[0].getType().isa<IntegerType>())
        return builder.create<TestCastOp>(loc, resultType, inputs).getResult();
      // Otherwise, fail.
      return nullptr;
    });

    // Initialize the conversion target.
    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<TestTypeProducerOp>([](TestTypeProducerOp op) {
      auto recursiveType = op.getType().dyn_cast<test::TestRecursiveType>();
      return op.getType().isF64() || op.getType().isInteger(64) ||
             (recursiveType &&
              recursiveType.getName() == "outer_converted_type");
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<TestCastOp>([&](TestCastOp op) {
      // Allow casts from F64 to F32.
      return (*op.operand_type_begin()).isF64() && op.getType().isF32();
    });
    target.addDynamicallyLegalOp<TestSignatureConversionNoConverterOp>(
        [&](TestSignatureConversionNoConverterOp op) {
          return converter.isLegal(op.getRegion().front().getArgumentTypes());
        });

    // Initialize the set of rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<TestTypeConsumerForward, TestTypeConversionProducer,
                 TestSignatureConversionUndo,
                 TestTestSignatureConversionNoConverter>(converter,
                                                         &getContext());
    patterns.add<TestTypeConversionAnotherProducer>(&getContext());
    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                              converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Test Target Materialization With No Uses
//===----------------------------------------------------------------------===//

namespace {
struct ForwardOperandPattern : public OpConversionPattern<TestTypeChangerOp> {
  using OpConversionPattern<TestTypeChangerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestTypeChangerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

struct TestTargetMaterializationWithNoUses
    : public PassWrapper<TestTargetMaterializationWithNoUses,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTargetMaterializationWithNoUses)

  StringRef getArgument() const final {
    return "test-target-materialization-with-no-uses";
  }
  StringRef getDescription() const final {
    return "Test a special case of target materialization in DialectConversion";
  }

  void runOnOperation() override {
    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });
    converter.addConversion([](IntegerType intTy) -> Type {
      if (intTy.getWidth() == 16)
        return IntegerType::get(intTy.getContext(), 64);
      return intTy;
    });
    converter.addTargetMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return builder.create<TestCastOp>(loc, type, inputs).getResult();
        });

    ConversionTarget target(getContext());
    target.addIllegalOp<TestTypeChangerOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ForwardOperandPattern>(converter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Test Block Merging
//===----------------------------------------------------------------------===//

namespace {
/// A rewriter pattern that tests that blocks can be merged.
struct TestMergeBlock : public OpConversionPattern<TestMergeBlocksOp> {
  using OpConversionPattern<TestMergeBlocksOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestMergeBlocksOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block &firstBlock = op.getBody().front();
    Operation *branchOp = firstBlock.getTerminator();
    Block *secondBlock = &*(std::next(op.getBody().begin()));
    auto succOperands = branchOp->getOperands();
    SmallVector<Value, 2> replacements(succOperands);
    rewriter.eraseOp(branchOp);
    rewriter.mergeBlocks(secondBlock, &firstBlock, replacements);
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

/// A rewrite pattern to tests the undo mechanism of blocks being merged.
struct TestUndoBlocksMerge : public ConversionPattern {
  TestUndoBlocksMerge(MLIRContext *ctx)
      : ConversionPattern("test.undo_blocks_merge", /*benefit=*/1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Block &firstBlock = op->getRegion(0).front();
    Operation *branchOp = firstBlock.getTerminator();
    Block *secondBlock = &*(std::next(op->getRegion(0).begin()));
    rewriter.setInsertionPointToStart(secondBlock);
    rewriter.create<ILLegalOpF>(op->getLoc(), rewriter.getF32Type());
    auto succOperands = branchOp->getOperands();
    SmallVector<Value, 2> replacements(succOperands);
    rewriter.eraseOp(branchOp);
    rewriter.mergeBlocks(secondBlock, &firstBlock, replacements);
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

/// A rewrite mechanism to inline the body of the op into its parent, when both
/// ops can have a single block.
struct TestMergeSingleBlockOps
    : public OpConversionPattern<SingleBlockImplicitTerminatorOp> {
  using OpConversionPattern<
      SingleBlockImplicitTerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SingleBlockImplicitTerminatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SingleBlockImplicitTerminatorOp parentOp =
        op->getParentOfType<SingleBlockImplicitTerminatorOp>();
    if (!parentOp)
      return failure();
    Block &innerBlock = op.getRegion().front();
    TerminatorOp innerTerminator =
        cast<TerminatorOp>(innerBlock.getTerminator());
    rewriter.mergeBlockBefore(&innerBlock, op);
    rewriter.eraseOp(innerTerminator);
    rewriter.eraseOp(op);
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

struct TestMergeBlocksPatternDriver
    : public PassWrapper<TestMergeBlocksPatternDriver,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMergeBlocksPatternDriver)

  StringRef getArgument() const final { return "test-merge-blocks"; }
  StringRef getDescription() const final {
    return "Test Merging operation in ConversionPatternRewriter";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TestMergeBlock, TestUndoBlocksMerge, TestMergeSingleBlockOps>(
        context);
    ConversionTarget target(*context);
    target.addLegalOp<func::FuncOp, ModuleOp, TerminatorOp, TestBranchOp,
                      TestTypeConsumerOp, TestTypeProducerOp, TestReturnOp>();
    target.addIllegalOp<ILLegalOpF>();

    /// Expect the op to have a single block after legalization.
    target.addDynamicallyLegalOp<TestMergeBlocksOp>(
        [&](TestMergeBlocksOp op) -> bool {
          return llvm::hasSingleElement(op.getBody());
        });

    /// Only allow `test.br` within test.merge_blocks op.
    target.addDynamicallyLegalOp<TestBranchOp>([&](TestBranchOp op) -> bool {
      return op->getParentOfType<TestMergeBlocksOp>();
    });

    /// Expect that all nested test.SingleBlockImplicitTerminator ops are
    /// inlined.
    target.addDynamicallyLegalOp<SingleBlockImplicitTerminatorOp>(
        [&](SingleBlockImplicitTerminatorOp op) -> bool {
          return !op->getParentOfType<SingleBlockImplicitTerminatorOp>();
        });

    DenseSet<Operation *> unlegalizedOps;
    (void)applyPartialConversion(getOperation(), target, std::move(patterns),
                                 &unlegalizedOps);
    for (auto *op : unlegalizedOps)
      op->emitRemark() << "op '" << op->getName() << "' is not legalizable";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Test Selective Replacement
//===----------------------------------------------------------------------===//

namespace {
/// A rewrite mechanism to inline the body of the op into its parent, when both
/// ops can have a single block.
struct TestSelectiveOpReplacementPattern : public OpRewritePattern<TestCastOp> {
  using OpRewritePattern<TestCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TestCastOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getNumOperands() != 2)
      return failure();
    OperandRange operands = op.getOperands();

    // Replace non-terminator uses with the first operand.
    rewriter.replaceOpWithIf(op, operands[0], [](OpOperand &operand) {
      return operand.getOwner()->hasTrait<OpTrait::IsTerminator>();
    });
    // Replace everything else with the second operand if the operation isn't
    // dead.
    rewriter.replaceOp(op, op.getOperand(1));
    return success();
  }
};

struct TestSelectiveReplacementPatternDriver
    : public PassWrapper<TestSelectiveReplacementPatternDriver,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestSelectiveReplacementPatternDriver)

  StringRef getArgument() const final {
    return "test-pattern-selective-replacement";
  }
  StringRef getDescription() const final {
    return "Test selective replacement in the PatternRewriter";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TestSelectiveOpReplacementPattern>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PassRegistration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {
void registerPatternsTestPass() {
  PassRegistration<TestReturnTypeDriver>();

  PassRegistration<TestDerivedAttributeDriver>();

  PassRegistration<TestPatternDriver>();
  PassRegistration<TestStrictPatternDriver>();

  PassRegistration<TestLegalizePatternDriver>([] {
    return std::make_unique<TestLegalizePatternDriver>(legalizerConversionMode);
  });

  PassRegistration<TestRemappedValue>();

  PassRegistration<TestUnknownRootOpDriver>();

  PassRegistration<TestTypeConversionDriver>();
  PassRegistration<TestTargetMaterializationWithNoUses>();

  PassRegistration<TestRewriteDynamicOpDriver>();

  PassRegistration<TestMergeBlocksPatternDriver>();
  PassRegistration<TestSelectiveReplacementPatternDriver>();
}
} // namespace test
} // namespace mlir
