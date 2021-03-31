//===- TestPatterns.cpp - Test dialect pattern driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::test;

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
  rewriter.create<OpSymbolBindingB>(op.getLoc(), op.operand().getType(),
                                    op.operand());
}

// Test that natives calls are only called once during rewrites.
// OpM_Test will return Pi, increased by 1 for each subsequent calls.
// This let us check the number of times OpM_Test was called by inspecting
// the returned value in the MLIR output.
static int64_t opMIncreasingValue = 314159265;
static Attribute OpMTest(PatternRewriter &rewriter, Value val) {
  int64_t i = opMIncreasingValue++;
  return rewriter.getIntegerAttr(rewriter.getIntegerType(32), i);
}

namespace {
#include "TestPatterns.inc"
} // end anonymous namespace

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

struct TestPatternDriver : public PassWrapper<TestPatternDriver, FunctionPass> {
  void runOnFunction() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);

    // Verify named pattern is generated with expected name.
    patterns.add<FoldingPattern, TestNamedPatternRule>(&getContext());

    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ReturnType Driver.
//===----------------------------------------------------------------------===//

namespace {
// Generate ops for each instance where the type can be successfully inferred.
template <typename OpTy>
static void invokeCreateWithInferredReturnType(Operation *op) {
  auto *context = op->getContext();
  auto fop = op->getParentOfType<FuncOp>();
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
              context, llvm::None, values, op->getAttrDictionary(),
              op->getRegions(), inferredReturnTypes))) {
        OperationState state(location, OpTy::getOperationName());
        // TODO: Expand to regions.
        OpTy::build(b, state, values, op->getAttrs());
        (void)b.createOperation(state);
      }
    }
  }
}

static void reifyReturnShape(Operation *op) {
  OpBuilder b(op);

  // Use permutations of 2 args as operands.
  auto shapedOp = cast<OpWithShapedTypeInferTypeInterfaceOp>(op);
  SmallVector<Value, 2> shapes;
  if (failed(shapedOp.reifyReturnTypeShapes(b, shapes)) ||
      !llvm::hasSingleElement(shapes))
    return;
  for (auto it : llvm::enumerate(shapes)) {
    op->emitRemark() << "value " << it.index() << ": "
                     << it.value().getDefiningOp();
  }
}

struct TestReturnTypeDriver
    : public PassWrapper<TestReturnTypeDriver, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnFunction() override {
    if (getFunction().getName() == "testCreateFunctions") {
      std::vector<Operation *> ops;
      // Collect ops to avoid triggering on inserted ops.
      for (auto &op : getFunction().getBody().front())
        ops.push_back(&op);
      // Generate test patterns for each, but skip terminator.
      for (auto *op : llvm::makeArrayRef(ops).drop_back()) {
        // Test create method of each of the Op classes below. The resultant
        // output would be in reverse order underneath `op` from which
        // the attributes and regions are used.
        invokeCreateWithInferredReturnType<OpWithInferTypeInterfaceOp>(op);
        invokeCreateWithInferredReturnType<
            OpWithShapedTypeInferTypeInterfaceOp>(op);
      };
      return;
    }
    if (getFunction().getName() == "testReifyFunctions") {
      std::vector<Operation *> ops;
      // Collect ops to avoid triggering on inserted ops.
      for (auto &op : getFunction().getBody().front())
        if (isa<OpWithShapedTypeInferTypeInterfaceOp>(op))
          ops.push_back(&op);
      // Generate test patterns for each, but skip terminator.
      for (auto *op : ops)
        reifyReturnShape(op);
    }
  }
};
} // end anonymous namespace

namespace {
struct TestDerivedAttributeDriver
    : public PassWrapper<TestDerivedAttributeDriver, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestDerivedAttributeDriver::runOnFunction() {
  getFunction().walk([](DerivedAttributeOpInterface dOp) {
    auto dAttr = dOp.materializeDerivedAttributes();
    if (!dAttr)
      return;
    for (auto d : dAttr)
      dOp.emitRemark() << d.first << " = " << d.second;
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
    auto *regionOp = rewriter.createOperation(newRegion);
    auto *entryBlock = rewriter.createBlock(&regionOp->getRegion(0));
    entryBlock->addArgument(rewriter.getIntegerType(64));

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
    rewriter.createBlock(&region, region.end(), {i32Type, i32Type});
    rewriter.create<TerminatorOp>(op->getLoc());
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
    rewriter.createBlock(&region, region.end(), {i32Type, i32Type});
    // Create an illegal op to ensure the conversion fails.
    rewriter.create<ILLegalOpF>(op->getLoc(), i32Type);
    rewriter.create<TerminatorOp>(op->getLoc());
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
    rewriter.replaceOpWithNewOp<TestValidOp>(op, llvm::None, operands,
                                             llvm::None);
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
    if (auto packerOp = llvm::dyn_cast_or_null<TestCastOp>(defOp)) {
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
  TestBoundedRecursiveRewrite(MLIRContext *ctx)
      : OpRewritePattern<TestRecursiveRewriteOp>(ctx) {
    // The conversion target handles bounding the recursion of this pattern.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(TestRecursiveRewriteOp op,
                                PatternRewriter &rewriter) const final {
    // Decrement the depth of the op in-place.
    rewriter.updateRootInPlace(op, [&] {
      op->setAttr("depth", rewriter.getI64IntegerAttr(op.depth() - 1));
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
} // namespace

namespace {
struct TestTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;
  TestTypeConverter() {
    addConversion(convertType);
    addArgumentMaterialization(materializeCast);
    addSourceMaterialization(materializeCast);

    /// Materialize the cast for one-to-one conversion from i64 to f64.
    const auto materializeOneToOneCast =
        [](OpBuilder &builder, IntegerType resultType, ValueRange inputs,
           Location loc) -> Optional<Value> {
      if (resultType.getWidth() == 42 && inputs.size() == 1)
        return builder.create<TestCastOp>(loc, resultType, inputs).getResult();
      return llvm::None;
    };
    addArgumentMaterialization(materializeOneToOneCast);
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
  static Optional<Value> materializeCast(OpBuilder &builder, Type resultType,
                                         ValueRange inputs, Location loc) {
    if (inputs.size() == 1)
      return inputs[0];
    return builder.create<TestCastOp>(loc, resultType, inputs).getResult();
  }
};

struct TestLegalizePatternDriver
    : public PassWrapper<TestLegalizePatternDriver, OperationPass<ModuleOp>> {
  /// The mode of conversion to use with the driver.
  enum class ConversionMode { Analysis, Full, Partial };

  TestLegalizePatternDriver(ConversionMode mode) : mode(mode) {}

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
             TestNestedOpCreationUndoRewrite, TestReplaceEraseOp>(
            &getContext());
    patterns.add<TestDropOpSignatureConversion>(&getContext(), converter);
    mlir::populateFuncOpTypeConversionPattern(patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);

    // Define the conversion target used for the test.
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<LegalOpA, LegalOpB, TestCastOp, TestValidOp,
                      TerminatorOp>();
    target
        .addIllegalOp<ILLegalOpF, TestRegionBuilderOp, TestOpWithRegionFold>();
    target.addDynamicallyLegalOp<TestReturnOp>([](TestReturnOp op) {
      // Don't allow F32 operands.
      return llvm::none_of(op.getOperandTypes(),
                           [](Type type) { return type.isF32(); });
    });
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getType()) &&
             converter.isLegal(&op.getBody());
    });

    // Expect the type_producer/type_consumer operations to only operate on f64.
    target.addDynamicallyLegalOp<TestTypeProducerOp>(
        [](TestTypeProducerOp op) { return op.getType().isF64(); });
    target.addDynamicallyLegalOp<TestTypeConsumerOp>([](TestTypeConsumerOp op) {
      return op.getOperand().getType().isF64();
    });

    // Check support for marking certain operations as recursively legal.
    target.markOpRecursivelyLegal<FuncOp, ModuleOp>([](Operation *op) {
      return static_cast<bool>(
          op->getAttrOfType<UnitAttr>("test.recursively_legal"));
    });

    // Mark the bound recursion operation as dynamically legal.
    target.addDynamicallyLegalOp<TestRecursiveRewriteOp>(
        [](TestRecursiveRewriteOp op) { return op.depth() == 0; });

    // Handle a partial conversion.
    if (mode == ConversionMode::Partial) {
      DenseSet<Operation *> unlegalizedOps;
      (void)applyPartialConversion(getOperation(), target, std::move(patterns),
                                   &unlegalizedOps);
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

      (void)applyFullConversion(getOperation(), target, std::move(patterns));
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
} // end anonymous namespace

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
  matchAndRewrite(OneVResOneVOperandOp1 op, ArrayRef<Value> operands,
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

struct TestRemappedValue
    : public mlir::PassWrapper<TestRemappedValue, FunctionPass> {
  void runOnFunction() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<OneVResOneVOperandOp1Converter>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, FuncOp, TestReturnOp>();
    // We make OneVResOneVOperandOp1 legal only when it has more that one
    // operand. This will trigger the conversion that will replace one-operand
    // OneVResOneVOperandOp1 with two-operand OneVResOneVOperandOp1.
    target.addDynamicallyLegalOp<OneVResOneVOperandOp1>(
        [](Operation *op) -> bool {
          return std::distance(op->operand_begin(), op->operand_end()) > 1;
        });

    if (failed(mlir::applyFullConversion(getFunction(), target,
                                         std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

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
    : public mlir::PassWrapper<TestUnknownRootOpDriver, FunctionPass> {
  void runOnFunction() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RemoveTestDialectOps>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<TestDialect>();
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
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
  matchAndRewrite(TestTypeProducerOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = op.getType();
    if (resultType.isa<FloatType>())
      resultType = rewriter.getF64Type();
    else if (resultType.isInteger(16))
      resultType = rewriter.getIntegerType(64);
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
  matchAndRewrite(TestSignatureConversionUndoOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    (void)rewriter.convertRegionTypes(&op->getRegion(0), *getTypeConverter());
    return failure();
  }
};

/// Just forward the operands to the root op. This is essentially a no-op
/// pattern that is used to trigger target materialization.
struct TestTypeConsumerForward
    : public OpConversionPattern<TestTypeConsumerOp> {
  using OpConversionPattern<TestTypeConsumerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestTypeConsumerOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&] { op->setOperands(operands); });
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TestDialect>();
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
      return op.getType().isF64() || op.getType().isInteger(64);
    });
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<TestCastOp>([&](TestCastOp op) {
      // Allow casts from F64 to F32.
      return (*op.operand_type_begin()).isF64() && op.getType().isF32();
    });

    // Initialize the set of rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<TestTypeConsumerForward, TestTypeConversionProducer,
                 TestSignatureConversionUndo>(converter, &getContext());
    patterns.add<TestTypeConversionAnotherProducer>(&getContext());
    mlir::populateFuncOpTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Test Block Merging
//===----------------------------------------------------------------------===//

namespace {
/// A rewriter pattern that tests that blocks can be merged.
struct TestMergeBlock : public OpConversionPattern<TestMergeBlocksOp> {
  using OpConversionPattern<TestMergeBlocksOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TestMergeBlocksOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Block &firstBlock = op.body().front();
    Operation *branchOp = firstBlock.getTerminator();
    Block *secondBlock = &*(std::next(op.body().begin()));
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
  matchAndRewrite(SingleBlockImplicitTerminatorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SingleBlockImplicitTerminatorOp parentOp =
        op->getParentOfType<SingleBlockImplicitTerminatorOp>();
    if (!parentOp)
      return failure();
    Block &innerBlock = op.region().front();
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
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TestMergeBlock, TestUndoBlocksMerge, TestMergeSingleBlockOps>(
        context);
    ConversionTarget target(*context);
    target.addLegalOp<FuncOp, ModuleOp, TerminatorOp, TestBranchOp,
                      TestTypeConsumerOp, TestTypeProducerOp, TestReturnOp>();
    target.addIllegalOp<ILLegalOpF>();

    /// Expect the op to have a single block after legalization.
    target.addDynamicallyLegalOp<TestMergeBlocksOp>(
        [&](TestMergeBlocksOp op) -> bool {
          return llvm::hasSingleElement(op.body());
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
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TestSelectiveOpReplacementPattern>(context);
    (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                       std::move(patterns));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PassRegistration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {
void registerPatternsTestPass() {
  PassRegistration<TestReturnTypeDriver>("test-return-type",
                                         "Run return type functions");

  PassRegistration<TestDerivedAttributeDriver>("test-derived-attr",
                                               "Run test derived attributes");

  PassRegistration<TestPatternDriver>("test-patterns",
                                      "Run test dialect patterns");

  PassRegistration<TestLegalizePatternDriver>(
      "test-legalize-patterns", "Run test dialect legalization patterns", [] {
        return std::make_unique<TestLegalizePatternDriver>(
            legalizerConversionMode);
      });

  PassRegistration<TestRemappedValue>(
      "test-remapped-value",
      "Test public remapped value mechanism in ConversionPatternRewriter");

  PassRegistration<TestUnknownRootOpDriver>(
      "test-legalize-unknown-root-patterns",
      "Test public remapped value mechanism in ConversionPatternRewriter");

  PassRegistration<TestTypeConversionDriver>(
      "test-legalize-type-conversion",
      "Test various type conversion functionalities in DialectConversion");

  PassRegistration<TestMergeBlocksPatternDriver>{
      "test-merge-blocks",
      "Test Merging operation in ConversionPatternRewriter"};
  PassRegistration<TestSelectiveReplacementPatternDriver>{
      "test-pattern-selective-replacement",
      "Test selective replacement in the PatternRewriter"};
}
} // namespace test
} // namespace mlir
