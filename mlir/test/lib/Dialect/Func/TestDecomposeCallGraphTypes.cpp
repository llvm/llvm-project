//===- TestDecomposeCallGraphTypes.cpp - Test CG type decomposition -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// Creates a sequence of `test.get_tuple_element` ops for all elements of a
/// given tuple value. If some tuple elements are, in turn, tuples, the elements
/// of those are extracted recursively such that the returned values have the
/// same types as `resultTypes.getFlattenedTypes()`.
static LogicalResult buildDecomposeTuple(OpBuilder &builder, Location loc,
                                         TupleType resultType, Value value,
                                         SmallVectorImpl<Value> &values) {
  for (unsigned i = 0, e = resultType.size(); i < e; ++i) {
    Type elementType = resultType.getType(i);
    Value element = builder.create<test::GetTupleElementOp>(
        loc, elementType, value, builder.getI32IntegerAttr(i));
    if (auto nestedTupleType = dyn_cast<TupleType>(elementType)) {
      // Recurse if the current element is also a tuple.
      if (failed(buildDecomposeTuple(builder, loc, nestedTupleType, element,
                                     values)))
        return failure();
    } else {
      values.push_back(element);
    }
  }
  return success();
}

/// Creates a `test.make_tuple` op out of the given inputs building a tuple of
/// type `resultType`. If that type is nested, each nested tuple is built
/// recursively with another `test.make_tuple` op.
static std::optional<Value> buildMakeTupleOp(OpBuilder &builder,
                                             TupleType resultType,
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
  return builder.create<test::MakeTupleOp>(loc, resultType, elements);
}

/// A pass for testing call graph type decomposition.
///
/// This instantiates the patterns with a TypeConverter and ValueDecomposer
/// that splits tuple types into their respective element types.
/// For example, `tuple<T1, T2, T3> --> T1, T2, T3`.
struct TestDecomposeCallGraphTypes
    : public PassWrapper<TestDecomposeCallGraphTypes, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDecomposeCallGraphTypes)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
  }
  StringRef getArgument() const final {
    return "test-decompose-call-graph-types";
  }
  StringRef getDescription() const final {
    return "Decomposes types at call graph boundaries.";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *context = &getContext();
    TypeConverter typeConverter;
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    target.addLegalDialect<test::TestDialect>();

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](TupleType tupleType, SmallVectorImpl<Type> &types) {
          tupleType.getFlattenedTypes(types);
          return success();
        });
    typeConverter.addArgumentMaterialization(buildMakeTupleOp);

    ValueDecomposer decomposer;
    decomposer.addDecomposeValueConversion(buildDecomposeTuple);

    populateDecomposeCallGraphTypesPatterns(context, typeConverter, decomposer,
                                            patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestDecomposeCallGraphTypes() {
  PassRegistration<TestDecomposeCallGraphTypes>();
}
} // namespace test
} // namespace mlir
