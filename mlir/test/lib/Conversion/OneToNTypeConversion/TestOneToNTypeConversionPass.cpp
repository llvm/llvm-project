//===- TestOneToNTypeConversionPass.cpp - Test pass 1:N type conv. utils --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;

namespace {
/// Test pass that exercises the (poor-man's) 1:N type conversion mechanisms
/// in `applyPartialOneToNConversion` by converting built-in tuples to the
/// elements they consist of as well as some dummy ops operating on these
/// tuples.
struct TestOneToNTypeConversionPass
    : public PassWrapper<TestOneToNTypeConversionPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOneToNTypeConversionPass)

  TestOneToNTypeConversionPass() = default;
  TestOneToNTypeConversionPass(const TestOneToNTypeConversionPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
  }

  StringRef getArgument() const final {
    return "test-one-to-n-type-conversion";
  }

  StringRef getDescription() const final {
    return "Test pass for 1:N type conversion";
  }

  Option<bool> convertFuncOps{*this, "convert-func-ops",
                              llvm::cl::desc("Enable conversion on func ops"),
                              llvm::cl::init(false)};

  Option<bool> convertTupleOps{*this, "convert-tuple-ops",
                               llvm::cl::desc("Enable conversion on tuple ops"),
                               llvm::cl::init(false)};

  void runOnOperation() override;
};

} // namespace

namespace mlir {
namespace test {
void registerTestOneToNTypeConversionPass() {
  PassRegistration<TestOneToNTypeConversionPass>();
}
} // namespace test
} // namespace mlir

namespace {

/// Test pattern on for the `make_tuple` op from the test dialect that converts
/// this kind of op into it's "decomposed" form, i.e., the elements of the tuple
/// that is being produced by `test.make_tuple`, which are really just the
/// operands of this op.
class ConvertMakeTupleOp
    : public OneToNOpConversionPattern<::test::MakeTupleOp> {
public:
  using OneToNOpConversionPattern<
      ::test::MakeTupleOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(::test::MakeTupleOp op,
                                OneToNPatternRewriter &rewriter,
                                const OneToNTypeMapping &operandMapping,
                                const OneToNTypeMapping &resultMapping,
                                ValueRange convertedOperands) const override {
    // Simply replace the current op with the converted operands.
    rewriter.replaceOp(op, convertedOperands, resultMapping);
    return success();
  }
};

/// Test pattern on for the `get_tuple_element` op from the test dialect that
/// converts this kind of op into it's "decomposed" form, i.e., instead of
/// "physically" extracting one element from the tuple, we forward the one
/// element of the decomposed form that is being extracted (or the several
/// elements in case that element is a nested tuple).
class ConvertGetTupleElementOp
    : public OneToNOpConversionPattern<::test::GetTupleElementOp> {
public:
  using OneToNOpConversionPattern<
      ::test::GetTupleElementOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(::test::GetTupleElementOp op,
                                OneToNPatternRewriter &rewriter,
                                const OneToNTypeMapping &operandMapping,
                                const OneToNTypeMapping &resultMapping,
                                ValueRange convertedOperands) const override {
    // Construct mapping for tuple element types.
    auto stateType = op->getOperand(0).getType().cast<TupleType>();
    TypeRange originalElementTypes = stateType.getTypes();
    OneToNTypeMapping elementMapping(originalElementTypes);
    if (failed(typeConverter->convertSignatureArgs(originalElementTypes,
                                                   elementMapping)))
      return failure();

    // Compute converted operands corresponding to original input tuple.
    ValueRange convertedTuple =
        operandMapping.getConvertedValues(convertedOperands, 0);

    // Got those converted operands that correspond to the index-th element of
    // the original input tuple.
    size_t index = op.getIndex();
    ValueRange extractedElement =
        elementMapping.getConvertedValues(convertedTuple, index);

    rewriter.replaceOp(op, extractedElement, resultMapping);

    return success();
  }
};

} // namespace

static void populateDecomposeTuplesTestPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertMakeTupleOp,
      ConvertGetTupleElementOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

/// Creates a sequence of `test.get_tuple_element` ops for all elements of a
/// given tuple value. If some tuple elements are, in turn, tuples, the elements
/// of those are extracted recursively such that the returned values have the
/// same types as `resultTypes.getFlattenedTypes()`.
///
/// This function has been copied (with small adaptions) from
/// TestDecomposeCallGraphTypes.cpp.
static std::optional<SmallVector<Value>>
buildGetTupleElementOps(OpBuilder &builder, TypeRange resultTypes, Value input,
                        Location loc) {
  TupleType inputType = input.getType().dyn_cast<TupleType>();
  if (!inputType)
    return {};

  SmallVector<Value> values;
  for (auto [idx, elementType] : llvm::enumerate(inputType.getTypes())) {
    Value element = builder.create<::test::GetTupleElementOp>(
        loc, elementType, input, builder.getI32IntegerAttr(idx));
    if (auto nestedTupleType = elementType.dyn_cast<TupleType>()) {
      // Recurse if the current element is also a tuple.
      SmallVector<Type> flatRecursiveTypes;
      nestedTupleType.getFlattenedTypes(flatRecursiveTypes);
      std::optional<SmallVector<Value>> resursiveValues =
          buildGetTupleElementOps(builder, flatRecursiveTypes, element, loc);
      if (!resursiveValues.has_value())
        return {};
      values.append(resursiveValues.value());
    } else {
      values.push_back(element);
    }
  }
  return values;
}

/// Creates a `test.make_tuple` op out of the given inputs building a tuple of
/// type `resultType`. If that type is nested, each nested tuple is built
/// recursively with another `test.make_tuple` op.
///
/// This function has been copied (with small adaptions) from
/// TestDecomposeCallGraphTypes.cpp.
static std::optional<Value> buildMakeTupleOp(OpBuilder &builder,
                                             TupleType resultType,
                                             ValueRange inputs, Location loc) {
  // Build one value for each element at this nesting level.
  SmallVector<Value> elements;
  elements.reserve(resultType.getTypes().size());
  ValueRange::iterator inputIt = inputs.begin();
  for (Type elementType : resultType.getTypes()) {
    if (auto nestedTupleType = elementType.dyn_cast<TupleType>()) {
      // Determine how many input values are needed for the nested elements of
      // the nested TupleType and advance inputIt by that number.
      // TODO: We only need the *number* of nested types, not the types itself.
      //       Maybe it's worth adding a more efficient overload?
      SmallVector<Type> nestedFlattenedTypes;
      nestedTupleType.getFlattenedTypes(nestedFlattenedTypes);
      size_t numNestedFlattenedTypes = nestedFlattenedTypes.size();
      ValueRange nestedFlattenedelements(inputIt,
                                         inputIt + numNestedFlattenedTypes);
      inputIt += numNestedFlattenedTypes;

      // Recurse on the values for the nested TupleType.
      std::optional<Value> res = buildMakeTupleOp(builder, nestedTupleType,
                                                  nestedFlattenedelements, loc);
      if (!res.has_value())
        return {};

      // The tuple constructed by the conversion is the element value.
      elements.push_back(res.value());
    } else {
      // Base case: take one input as is.
      elements.push_back(*inputIt++);
    }
  }

  // Assemble the tuple from the elements.
  return builder.create<::test::MakeTupleOp>(loc, resultType, elements);
}

void TestOneToNTypeConversionPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto *context = &getContext();

  // Assemble type converter.
  OneToNTypeConverter typeConverter;

  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](TupleType tupleType, SmallVectorImpl<Type> &types) {
        tupleType.getFlattenedTypes(types);
        return success();
      });

  typeConverter.addArgumentMaterialization(buildMakeTupleOp);
  typeConverter.addSourceMaterialization(buildMakeTupleOp);
  typeConverter.addTargetMaterialization(buildGetTupleElementOps);

  // Assemble patterns.
  RewritePatternSet patterns(context);
  if (convertTupleOps)
    populateDecomposeTuplesTestPatterns(typeConverter, patterns);
  if (convertFuncOps)
    populateFuncTypeConversionPatterns(typeConverter, patterns);

  // Run conversion.
  if (failed(applyPartialOneToNConversion(module, typeConverter,
                                          std::move(patterns))))
    return signalPassFailure();
}
