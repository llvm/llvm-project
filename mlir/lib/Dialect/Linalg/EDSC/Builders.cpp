//===- Builders.cpp - MLIR Declarative Linalg Builders --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::scf;

Operation *mlir::edsc::makeGenericLinalgOp(
    ArrayRef<IteratorType> iteratorTypes, ArrayRef<StructuredIndexed> inputs,
    ArrayRef<StructuredIndexed> outputBuffers, ArrayRef<Value> initTensors,
    ArrayRef<StructuredIndexed> resultTensorTypes,
    function_ref<void(ValueRange)> regionBuilder, ArrayRef<Value> otherValues,
    ArrayRef<Attribute> otherAttributes) {
  OpBuilder &builder = edsc::ScopedContext::getBuilderRef();

  // Build maps
  SmallVector<SmallVector<AffineExpr, 4>, 4> exprsList;
  exprsList.reserve(inputs.size() + outputBuffers.size() + initTensors.size());
  for (auto container : {inputs, outputBuffers, resultTensorTypes})
    for (const StructuredIndexed &s : container)
      exprsList.emplace_back(s.getExprs().begin(), s.getExprs().end());
  auto maps = AffineMap::inferFromExprList(exprsList);

  SmallVector<Type, 4> types;
  assert(llvm::all_of(resultTensorTypes, [](const StructuredIndexed &s) {
    return !s.hasValue();
  }));
  std::copy(resultTensorTypes.begin(), resultTensorTypes.end(),
            std::back_inserter(types));

  SmallVector<Value, 4> inputValues, outputBufferValues, initTensorValues;
  inputValues.reserve(inputs.size());
  outputBufferValues.reserve(outputBuffers.size());
  initTensorValues.reserve(initTensors.size());
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputValues));
  std::copy(outputBuffers.begin(), outputBuffers.end(),
            std::back_inserter(outputBufferValues));
  std::copy(initTensors.begin(), initTensors.end(),
            std::back_inserter(initTensorValues));

  auto iteratorStrTypes =
      llvm::to_vector<8>(llvm::map_range(iteratorTypes, toString));
  // clang-format off
  auto *op =
      edsc::ScopedContext::getBuilderRef()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(),
              types,
              inputValues,
              outputBufferValues,
              initTensorValues,
              builder.getAffineMapArrayAttr(maps),
              builder.getStrArrayAttr(iteratorStrTypes),
              StringAttr() /*doc*/,
              StringAttr() /*library_call*/,
              ArrayAttr() /*sparse*/,
              IntegerAttr() /*symbol_source*/
              /* TODO: other attributes in op */
              )
          .getOperation();
  // clang-format on

  using namespace edsc;
  SmallVector<Type, 4> blockTypes;
  blockTypes.reserve(inputs.size() + outputBuffers.size() + initTensors.size());
  for (auto container : {inputs, outputBuffers})
    for (const StructuredIndexed &s : container)
      blockTypes.push_back(getElementTypeOrSelf(s.getType()));
  for (Value v : initTensors)
    blockTypes.push_back(getElementTypeOrSelf(v.getType()));

  assert(op->getNumRegions() == 1);
  assert(op->getRegion(0).empty());
  OpBuilder opBuilder(op);
  ScopedContext scope(opBuilder, op->getLoc());
  buildInNewBlock(op->getRegion(0), blockTypes, regionBuilder);
  assert(llvm::hasSingleElement(op->getRegion(0)));
  return op;
}

void mlir::edsc::ops::mulRegionBuilder(ValueRange args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 2 && "expected 2 block arguments");
  Value a(args[0]), b(args[1]);
  linalg_yield(a * b);
}

void mlir::edsc::ops::macRegionBuilder(ValueRange args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 3 && "expected 3 block arguments");
  Value a(args[0]), b(args[1]), c(args[2]);
  linalg_yield(c + a * b);
}

Operation *mlir::edsc::ops::linalg_generic_pointwise(
    UnaryPointwiseOpBuilder unaryOp, StructuredIndexed I, StructuredIndexed O) {
  SmallVector<IteratorType, 4> iterTypes(O.getExprs().size(),
                                         IteratorType::Parallel);
  auto fun = [&unaryOp](ValueRange args) {
    assert(!args.empty() && "expected >= 1 block arguments");
    Value a(args[0]);
    linalg_yield(unaryOp(a));
  };
  if (O.getType().isa<RankedTensorType>())
    return makeGenericLinalgOp(iterTypes, /*inputs=*/{I}, /*outputBuffers=*/{},
                               /*initTensors=*/{}, /*resultTensorTypes=*/{O},
                               fun);
  return makeGenericLinalgOp(iterTypes, /*inputs=*/{I}, /*outputBuffers=*/{O},
                             /*initTensors=*/{}, /*resultTensorTypes=*/{}, fun);
}

Operation *mlir::edsc::ops::linalg_generic_pointwise_tanh(StructuredIndexed I,
                                                          StructuredIndexed O) {
  UnaryPointwiseOpBuilder unOp([](Value a) -> Value { return std_tanh(a); });
  return linalg_generic_pointwise(unOp, I, O);
}

/// Binary pointwise operation (with broadcast) entry point.
Operation *mlir::edsc::ops::linalg_generic_pointwise(
    BinaryPointwiseOpBuilder binaryOp, StructuredIndexed I1,
    StructuredIndexed I2, StructuredIndexed O) {
  SmallVector<IteratorType, 4> iterTypes(O.getExprs().size(),
                                         IteratorType::Parallel);
  auto fun = [&binaryOp](ValueRange args) {
    assert(args.size() >= 2 && "expected >= 2 block arguments");
    Value a(args[0]), b(args[1]);
    linalg_yield(binaryOp(a, b));
  };
  if (O.getType().isa<RankedTensorType>())
    return makeGenericLinalgOp(
        iterTypes, /*inputs=*/{I1, I2}, /*outputBuffers=*/{},
        /*initTensors=*/{}, /*resultTensorTypes=*/{O}, fun);
  return makeGenericLinalgOp(iterTypes, /*inputs=*/{I1, I2},
                             /*outputBuffers=*/{O},
                             /*initTensors=*/{}, /*resultTensorTypes=*/{}, fun);
}

Operation *mlir::edsc::ops::linalg_generic_pointwise_add(StructuredIndexed I1,
                                                         StructuredIndexed I2,
                                                         StructuredIndexed O) {
  using edsc::op::operator+;
  BinaryPointwiseOpBuilder binOp(
      [](Value a, Value b) -> Value { return a + b; });
  return linalg_generic_pointwise(binOp, I1, I2, O);
}

Operation *mlir::edsc::ops::linalg_generic_pointwise_max(StructuredIndexed I1,
                                                         StructuredIndexed I2,
                                                         StructuredIndexed O) {
  BinaryPointwiseOpBuilder binOp([](Value a, Value b) -> Value {
    using edsc::op::sgt;
    return std_select(sgt(a, b), a, b);
  });
  return linalg_generic_pointwise(binOp, I1, I2, O);
}

Operation *
mlir::edsc::ops::linalg_generic_matmul(Value vA, Value vB, Value vC,
                                       MatmulRegionBuilder regionBuilder) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(vC);
  return makeGenericLinalgOp(
    {IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction},
    /*inputs=*/{A({m, k}), B({k, n})},
    /*outputBuffers=*/{C({m, n})},
    /*initTensors=*/{},
    /*resultTensorTypes=*/{},
    regionBuilder);
  // clang-format on
}

Operation *
mlir::edsc::ops::linalg_generic_matmul(Value vA, Value vB, Value vC,
                                       RankedTensorType tD,
                                       MatmulRegionBuilder regionBuilder) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(vC), D(tD);
  return makeGenericLinalgOp(
    {IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction},
    /*inputs=*/{A({m, k}), B({k, n})},
    /*outputBuffers=*/{},
    /*initTensors=*/{C({m, n})},
    /*resultTensorTypes=*/{D({m, n})},
    regionBuilder);
  // clang-format on
}

Operation *mlir::edsc::ops::linalg_generic_conv_nhwc(Value vI, Value vW,
                                                     Value vO,
                                                     ArrayRef<int> strides,
                                                     ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO: some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IteratorType::Parallel;
  auto red = IteratorType::Reduction;
  auto s = strides;
  auto d = dilations;

  AffineExpr b, f, h, w, kh, kw, c;
  bindDims(ctx, b, f, h, w, kh, kw, c);
  unsigned numDims = c.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);
  // clang-format off
  return makeGenericLinalgOp(
    {par, par, par, par, red, red, red},
    /*inputs=*/{
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, f}) },
    /*outputBuffers=*/{ O({b, h, w, f}) },
    /*initTensors=*/{},
    /*resultTensorTypes=*/{},
    macRegionBuilder);
  // clang-format on
}

Operation *mlir::edsc::ops::linalg_generic_dilated_conv_nhwc(
    Value vI, Value vW, Value vO, int depth_multiplier, ArrayRef<int> strides,
    ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO: some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IteratorType::Parallel;
  auto red = IteratorType::Reduction;
  auto s = strides;
  auto d = dilations;

  // clang-format off
  AffineExpr b, dm, c, h, w, kh, kw;
  bindDims(ctx, b, dm, c, h, w, kh, kw);
  unsigned numDims = kw.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);
  return makeGenericLinalgOp(
    {par, par, par, par, par, red, red},
    /*inputs=*/{
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, dm})},
    /*outputBuffers=*/{
      O({b, h, w, simplifyAffineExpr(c * depth_multiplier + dm, numDims, 0)})},
    /*initTensors=*/{},
    /*resultTensorTypes=*/{},
    macRegionBuilder);
  // clang-format on
}
