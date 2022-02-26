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
/// A pass for testing call graph type decomposition.
///
/// This instantiates the patterns with a TypeConverter and ValueDecomposer
/// that splits tuple types into their respective element types.
/// For example, `tuple<T1, T2, T3> --> T1, T2, T3`.
struct TestDecomposeCallGraphTypes
    : public PassWrapper<TestDecomposeCallGraphTypes, OperationPass<ModuleOp>> {

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
    ValueDecomposer decomposer;
    RewritePatternSet patterns(context);

    target.addLegalDialect<test::TestDialect>();

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](TupleType tupleType, SmallVectorImpl<Type> &types) {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    decomposer.addDecomposeValueConversion([](OpBuilder &builder, Location loc,
                                              TupleType resultType, Value value,
                                              SmallVectorImpl<Value> &values) {
      for (unsigned i = 0, e = resultType.size(); i < e; ++i) {
        Value res = builder.create<test::GetTupleElementOp>(
            loc, resultType.getType(i), value, builder.getI32IntegerAttr(i));
        values.push_back(res);
      }
      return success();
    });

    typeConverter.addArgumentMaterialization(
        [](OpBuilder &builder, TupleType resultType, ValueRange inputs,
           Location loc) -> Optional<Value> {
          if (inputs.size() == 1)
            return llvm::None;
          TupleType tuple = builder.getTupleType(inputs.getTypes());
          Value value = builder.create<test::MakeTupleOp>(loc, tuple, inputs);
          return value;
        });

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
