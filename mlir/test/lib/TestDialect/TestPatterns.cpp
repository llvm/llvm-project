//===- TestPatterns.cpp - Test dialect pattern driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;

// Native function for testing NativeCodeCall
static Value chooseOperand(Value input1, Value input2, BoolAttr choice) {
  return choice.getValue() ? input1 : input2;
}

static void createOpI(PatternRewriter &rewriter, Value input) {
  rewriter.create<OpI>(rewriter.getUnknownLoc(), input);
}

static void handleNoResultOp(PatternRewriter &rewriter,
                             OpSymbolBindingNoResult op) {
  // Turn the no result op to a one-result op.
  rewriter.create<OpSymbolBindingB>(op.getLoc(), op.operand().getType(),
                                    op.operand());
}

namespace {
#include "TestPatterns.inc"
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Canonicalizer Driver.
//===----------------------------------------------------------------------===//

namespace {
struct TestPatternDriver : public FunctionPass<TestPatternDriver> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);

    // Verify named pattern is generated with expected name.
    patterns.insert<TestNamedPatternRule>(&getContext());

    applyPatternsGreedily(getFunction(), patterns);
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
      if (succeeded(OpTy::inferReturnTypes(context, llvm::None, values,
                                           op->getAttrs(), op->getRegions(),
                                           inferredReturnTypes))) {
        OperationState state(location, OpTy::getOperationName());
        // TODO(jpienaar): Expand to regions.
        OpTy::build(&b, state, values, op->getAttrs());
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
  if (failed(shapedOp.reifyReturnTypeShapes(b, shapes)))
    return;
  for (auto it : llvm::enumerate(shapes))
    op->emitRemark() << "value " << it.index() << ": "
                     << it.value().getDefiningOp();
}

struct TestReturnTypeDriver : public FunctionPass<TestReturnTypeDriver> {
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

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Inline this region into the parent region.
    auto &parentRegion = *op->getParentRegion();
    if (op->getAttr("legalizer.should_clone"))
      rewriter.cloneRegionBefore(op->getRegion(0), parentRegion,
                                 parentRegion.end());
    else
      rewriter.inlineRegionBefore(op->getRegion(0), parentRegion,
                                  parentRegion.end());

    // Drop this operation.
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};
/// This pattern is a simple pattern that generates a region containing an
/// illegal operation.
struct TestRegionRewriteUndo : public RewritePattern {
  TestRegionRewriteUndo(MLIRContext *ctx)
      : RewritePattern("test.region_builder", 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op,
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
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Type-Conversion Rewrite Testing

/// This patterns erases a region operation that has had a type conversion.
struct TestDropOpSignatureConversion : public ConversionPattern {
  TestDropOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern("test.drop_region_op", 1, ctx), converter(converter) {
  }
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Region &region = op->getRegion(0);
    Block *entry = &region.front();

    // Convert the original entry arguments.
    TypeConverter::SignatureConversion result(entry->getNumArguments());
    for (unsigned i = 0, e = entry->getNumArguments(); i != e; ++i)
      if (failed(converter.convertSignatureArg(
              i, entry->getArgument(i).getType(), result)))
        return matchFailure();

    // Convert the region signature and just drop the operation.
    rewriter.applySignatureConversion(&region, result);
    rewriter.eraseOp(op);
    return matchSuccess();
  }

  /// The type converter to use when rewriting the signature.
  TypeConverter &converter;
};
/// This pattern simply updates the operands of the given operation.
struct TestPassthroughInvalidOp : public ConversionPattern {
  TestPassthroughInvalidOp(MLIRContext *ctx)
      : ConversionPattern("test.invalid", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TestValidOp>(op, llvm::None, operands,
                                             llvm::None);
    return matchSuccess();
  }
};
/// This pattern handles the case of a split return value.
struct TestSplitReturnType : public ConversionPattern {
  TestSplitReturnType(MLIRContext *ctx)
      : ConversionPattern("test.return", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Check for a return of F32.
    if (op->getNumOperands() != 1 || !op->getOperand(0).getType().isF32())
      return matchFailure();

    // Check if the first operation is a cast operation, if it is we use the
    // results directly.
    auto *defOp = operands[0].getDefiningOp();
    if (auto packerOp = llvm::dyn_cast_or_null<TestCastOp>(defOp)) {
      rewriter.replaceOpWithNewOp<TestReturnOp>(op, packerOp.getOperands());
      return matchSuccess();
    }

    // Otherwise, fail to match.
    return matchFailure();
  }
};

//===----------------------------------------------------------------------===//
// Multi-Level Type-Conversion Rewrite Testing
struct TestChangeProducerTypeI32ToF32 : public ConversionPattern {
  TestChangeProducerTypeI32ToF32(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is I32, change the type to F32.
    if (!Type(*op->result_type_begin()).isSignlessInteger(32))
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF32Type());
    return matchSuccess();
  }
};
struct TestChangeProducerTypeF32ToF64 : public ConversionPattern {
  TestChangeProducerTypeF32ToF64(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is F32, change the type to F64.
    if (!Type(*op->result_type_begin()).isF32())
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF64Type());
    return matchSuccess();
  }
};
struct TestChangeProducerTypeF32ToInvalid : public ConversionPattern {
  TestChangeProducerTypeF32ToInvalid(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 10, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Always convert to B16, even though it is not a legal type. This tests
    // that values are unmapped correctly.
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getBF16Type());
    return matchSuccess();
  }
};
struct TestUpdateConsumerType : public ConversionPattern {
  TestUpdateConsumerType(MLIRContext *ctx)
      : ConversionPattern("test.type_consumer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Verify that the incoming operand has been successfully remapped to F64.
    if (!operands[0].getType().isF64())
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeConsumerOp>(op, operands[0]);
    return matchSuccess();
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

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const final {
    auto resultType = *op->result_type_begin();
    auto illegalOp = rewriter.create<ILLegalOpF>(op->getLoc(), resultType);
    auto legalOp = rewriter.create<LegalOpB>(op->getLoc(), resultType);

    rewriter.replaceOp(illegalOp, {legalOp});
    rewriter.replaceOp(op, {illegalOp});
    return matchSuccess();
  }
};
} // namespace

namespace {
struct TestTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;
  TestTypeConverter() { addConversion(convertType); }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    // Drop I16 types.
    if (t.isSignlessInteger(16))
      return success();

    // Convert I64 to F64.
    if (t.isSignlessInteger(64)) {
      results.push_back(FloatType::getF64(t.getContext()));
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

  /// Override the hook to materialize a conversion. This is necessary because
  /// we generate 1->N type mappings.
  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value> inputs,
                                   Location loc) override {
    return rewriter.create<TestCastOp>(loc, resultType, inputs);
  }
};

struct TestLegalizePatternDriver
    : public ModulePass<TestLegalizePatternDriver> {
  /// The mode of conversion to use with the driver.
  enum class ConversionMode { Analysis, Full, Partial };

  TestLegalizePatternDriver(ConversionMode mode) : mode(mode) {}

  void runOnModule() override {
    TestTypeConverter converter;
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    patterns
        .insert<TestRegionRewriteBlockMovement, TestRegionRewriteUndo,
                TestPassthroughInvalidOp, TestSplitReturnType,
                TestChangeProducerTypeI32ToF32, TestChangeProducerTypeF32ToF64,
                TestChangeProducerTypeF32ToInvalid, TestUpdateConsumerType,
                TestNonRootReplacement>(&getContext());
    patterns.insert<TestDropOpSignatureConversion>(&getContext(), converter);
    mlir::populateFuncOpTypeConversionPattern(patterns, &getContext(),
                                              converter);

    // Define the conversion target used for the test.
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addLegalOp<LegalOpA, LegalOpB, TestCastOp, TestValidOp>();
    target
        .addIllegalOp<ILLegalOpF, TestRegionBuilderOp, TestOpWithRegionFold>();
    target.addDynamicallyLegalOp<TestReturnOp>([](TestReturnOp op) {
      // Don't allow F32 operands.
      return llvm::none_of(op.getOperandTypes(),
                           [](Type type) { return type.isF32(); });
    });
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

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

    // Handle a partial conversion.
    if (mode == ConversionMode::Partial) {
      (void)applyPartialConversion(getModule(), target, patterns, &converter);
      return;
    }

    // Handle a full conversion.
    if (mode == ConversionMode::Full) {
      // Check support for marking unknown operations as dynamically legal.
      target.markUnknownOpDynamicallyLegal([](Operation *op) {
        return (bool)op->getAttrOfType<UnitAttr>("test.dynamically_legal");
      });

      (void)applyFullConversion(getModule(), target, patterns, &converter);
      return;
    }

    // Otherwise, handle an analysis conversion.
    assert(mode == ConversionMode::Analysis);

    // Analyze the convertible operations.
    DenseSet<Operation *> legalizedOps;
    if (failed(applyAnalysisConversion(getModule(), target, patterns,
                                       legalizedOps, &converter)))
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
// to get the remapped value of a original value that was replaced using
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

  PatternMatchResult
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
    return matchSuccess();
  }
};

struct TestRemappedValue : public mlir::FunctionPass<TestRemappedValue> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<OneVResOneVOperandOp1Converter>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp, TestReturnOp>();
    // We make OneVResOneVOperandOp1 legal only when it has more that one
    // operand. This will trigger the conversion that will replace one-operand
    // OneVResOneVOperandOp1 with two-operand OneVResOneVOperandOp1.
    target.addDynamicallyLegalOp<OneVResOneVOperandOp1>(
        [](Operation *op) -> bool {
          return std::distance(op->operand_begin(), op->operand_end()) > 1;
        });

    if (failed(mlir::applyFullConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

namespace mlir {
void registerPatternsTestPass() {
  mlir::PassRegistration<TestReturnTypeDriver>("test-return-type",
                                               "Run return type functions");

  mlir::PassRegistration<TestPatternDriver>("test-patterns",
                                            "Run test dialect patterns");

  mlir::PassRegistration<TestLegalizePatternDriver>(
      "test-legalize-patterns", "Run test dialect legalization patterns", [] {
        return std::make_unique<TestLegalizePatternDriver>(
            legalizerConversionMode);
      });

  PassRegistration<TestRemappedValue>(
      "test-remapped-value",
      "Test public remapped value mechanism in ConversionPatternRewriter");
}
} // namespace mlir
