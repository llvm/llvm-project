//===- TestOneToNTypeConversionPass.cpp - Test pass 1:N type conv. utils --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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

  Option<bool> convertSCFOps{*this, "convert-scf-ops",
                             llvm::cl::desc("Enable conversion on scf ops"),
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

  LogicalResult
  matchAndRewrite(::test::MakeTupleOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Simply replace the current op with the converted operands.
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());
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

  LogicalResult
  matchAndRewrite(::test::GetTupleElementOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Construct mapping for tuple element types.
    auto stateType = cast<TupleType>(op->getOperand(0).getType());
    TypeRange originalElementTypes = stateType.getTypes();
    OneToNTypeMapping elementMapping(originalElementTypes);
    if (failed(typeConverter->convertSignatureArgs(originalElementTypes,
                                                   elementMapping)))
      return failure();

    // Compute converted operands corresponding to original input tuple.
    assert(adaptor.getOperands().size() == 1 &&
           "expected 'get_tuple_element' to have one operand");
    ValueRange convertedTuple = adaptor.getOperands()[0];

    // Got those converted operands that correspond to the index-th element ofq
    // the original input tuple.
    size_t index = op.getIndex();
    ValueRange extractedElement =
        elementMapping.getConvertedValues(convertedTuple, index);

    rewriter.replaceOp(op, extractedElement, adaptor.getResultMapping());

    return success();
  }
};

} // namespace

static void
populateDecomposeTuplesTestPatterns(const TypeConverter &typeConverter,
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
static SmallVector<Value> buildGetTupleElementOps(OpBuilder &builder,
                                                  TypeRange resultTypes,
                                                  ValueRange inputs,
                                                  Location loc) {
  if (inputs.size() != 1)
    return {};
  Value input = inputs.front();

  TupleType inputType = dyn_cast<TupleType>(input.getType());
  if (!inputType)
    return {};

  SmallVector<Value> values;
  for (auto [idx, elementType] : llvm::enumerate(inputType.getTypes())) {
    Value element = builder.create<::test::GetTupleElementOp>(
        loc, elementType, input, builder.getI32IntegerAttr(idx));
    if (auto nestedTupleType = dyn_cast<TupleType>(elementType)) {
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
static Value buildMakeTupleOp(OpBuilder &builder, TupleType resultType,
                              ValueRange inputs, Location loc) {
  // Build one value for each element at this nesting level.
  SmallVector<Value> elements;
  elements.reserve(resultType.getTypes().size());
  ValueRange::iterator inputIt = inputs.begin();
  for (Type elementType : resultType.getTypes()) {
    if (auto nestedTupleType = dyn_cast<TupleType>(elementType)) {
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
      Value res = buildMakeTupleOp(builder, nestedTupleType,
                                   nestedFlattenedelements, loc);
      if (!res)
        return Value();

      // The tuple constructed by the conversion is the element value.
      elements.push_back(res);
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
  TypeConverter typeConverter;

  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](TupleType tupleType, SmallVectorImpl<Type> &types) {
        tupleType.getFlattenedTypes(types);
        return success();
      });

  typeConverter.addArgumentMaterialization(buildMakeTupleOp);
  typeConverter.addSourceMaterialization(buildMakeTupleOp);
  typeConverter.addTargetMaterialization(buildGetTupleElementOps);
  // Test the other target materialization variant that takes the original type
  // as additional argument. This materialization function always fails.
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
         Location loc, Type originalType) -> SmallVector<Value> { return {}; });

  // Assemble patterns.
  RewritePatternSet patterns(context);
  if (convertTupleOps)
    populateDecomposeTuplesTestPatterns(typeConverter, patterns);
  if (convertFuncOps)
    populateFuncTypeConversionPatterns(typeConverter, patterns);
  if (convertSCFOps)
    scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);

  // Run conversion.
  if (failed(applyPartialOneToNConversion(module, typeConverter,
                                          std::move(patterns))))
    return signalPassFailure();
}
