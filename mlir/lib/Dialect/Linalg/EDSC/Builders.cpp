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
    ArrayRef<StructuredIndexed> outputs,
    function_ref<void(ValueRange)> regionBuilder, ArrayRef<Value> otherValues,
    ArrayRef<Attribute> otherAttributes) {
  for (unsigned i = 0, e = outputs.size(); i + 1 < e; ++i)
    assert(!(outputs[i].getType().isa<RankedTensorType>() &&
             outputs[i + 1].getType().isa<MemRefType>()) &&
           "output tensors must be passed after output buffers");
  auto &builder = edsc::ScopedContext::getBuilderRef();
  auto *ctx = builder.getContext();
  unsigned nInputs = inputs.size();
  unsigned nOutputs = outputs.size();

  SmallVector<SmallVector<AffineExpr, 4>, 4> exprsList;
  exprsList.reserve(nInputs + nOutputs);
  for (auto structuredIndexed : inputs)
    exprsList.emplace_back(structuredIndexed.getExprs().begin(),
                           structuredIndexed.getExprs().end());
  for (auto structuredIndexed : outputs)
    exprsList.emplace_back(structuredIndexed.getExprs().begin(),
                           structuredIndexed.getExprs().end());
  auto maps = AffineMap::inferFromExprList(exprsList);

  unsigned nViews = nInputs + nOutputs;
  SmallVector<Value, 4> values;
  values.reserve(nViews);
  values.append(inputs.begin(), inputs.end());
  std::copy_if(outputs.begin(), outputs.end(), std::back_inserter(values),
               [](StructuredIndexed s) { return s.hasValue(); });
  SmallVector<Type, 4> types;
  std::copy_if(outputs.begin(), outputs.end(), std::back_inserter(types),
               [](StructuredIndexed s) { return !s.hasValue(); });

  auto iteratorStrTypes =
      llvm::to_vector<8>(llvm::map_range(iteratorTypes, toString));
  // clang-format off
  auto *op =
      edsc::ScopedContext::getBuilderRef()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(),
              types,
              values,
              IntegerAttr::get(IntegerType::get(64, ctx), nInputs),
              IntegerAttr::get(IntegerType::get(64, ctx), nOutputs),
              builder.getAffineMapArrayAttr(maps),
              builder.getStrArrayAttr(iteratorStrTypes),
              StringAttr() /*doc*/,
              StringAttr() /*library_call*/
              /* TODO: other attributes in op */
              )
          .getOperation();
  // clang-format on

  using namespace edsc;
  SmallVector<Type, 4> blockTypes;
  blockTypes.reserve(values.size());
  for (auto it : llvm::enumerate(values))
    blockTypes.push_back((it.index() < nViews)
                             ? getElementTypeOrSelf(it.value())
                             : it.value().getType());

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
  if (O.getType().isa<RankedTensorType>()) {
    auto fun = [&unaryOp](ValueRange args) {
      assert(args.size() == 1 && "expected 1 block arguments");
      Value a(args[0]);
      linalg_yield(unaryOp(a));
    };
    return makeGenericLinalgOp(iterTypes, {I}, {O}, fun);
  }
  auto fun = [&unaryOp](ValueRange args) {
    assert(args.size() == 2 && "expected 2 block arguments");
    Value a(args[0]);
    linalg_yield(unaryOp(a));
  };
  return makeGenericLinalgOp(iterTypes, {I}, {O}, fun);
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
  if (O.getType().isa<RankedTensorType>()) {
    auto fun = [&binaryOp](ValueRange args) {
      assert(args.size() == 2 && "expected 2 block arguments");
      Value a(args[0]), b(args[1]);
      linalg_yield(binaryOp(a, b));
    };
    return makeGenericLinalgOp(iterTypes, {I1, I2}, {O}, fun);
  }
  auto fun = [&binaryOp](ValueRange args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    Value a(args[0]), b(args[1]);
    linalg_yield(binaryOp(a, b));
  };
  return makeGenericLinalgOp(iterTypes, {I1, I2}, {O}, fun);
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
    {A({m, k}), B({k, n})},
    {C({m, n})},
    regionBuilder);
  // clang-format on
}

Operation *
mlir::edsc::ops::linalg_generic_matmul(Value vA, Value vB, RankedTensorType tC,
                                       MatmulRegionBuilder regionBuilder) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(tC);
  return makeGenericLinalgOp(
    {IteratorType::Parallel, IteratorType::Parallel, IteratorType::Reduction},
    {A({m, k}), B({k, n})},
    {C({m, n})},
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
    {A({m, k}), B({k, n}), C({m, n})},
    {D({m, n})},
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
    {par, par, par, par, red, red, red}, {
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, f})}, {
      O({b, h, w, f})},
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
    {par, par, par, par, par, red, red}, {
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, dm})}, {
      O({b, h, w, simplifyAffineExpr(c * depth_multiplier + dm, numDims, 0)})},
    macRegionBuilder);
  // clang-format on
}
