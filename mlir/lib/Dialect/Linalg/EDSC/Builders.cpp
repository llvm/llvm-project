//===- Builders.cpp - MLIR Declarative Linalg Builders --------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/Functional.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::edsc::ops;
using namespace mlir::linalg;
using namespace mlir::loop;

mlir::edsc::LoopRangeBuilder::LoopRangeBuilder(ValueHandle *iv,
                                               ValueHandle range) {
  assert(range.getType() && "expected !linalg.range type");
  assert(range.getValue().getDefiningOp() &&
         "need operations to extract range parts");
  auto rangeOp = cast<RangeOp>(range.getValue().getDefiningOp());
  auto lb = rangeOp.min();
  auto ub = rangeOp.max();
  auto step = rangeOp.step();
  auto forOp = OperationHandle::createOp<ForOp>(lb, ub, step);
  *iv = ValueHandle(forOp.getInductionVar());
  auto *body = forOp.getBody();
  enter(body, /*prev=*/1);
}

mlir::edsc::LoopRangeBuilder::LoopRangeBuilder(ValueHandle *iv,
                                               SubViewOp::Range range) {
  auto forOp =
      OperationHandle::createOp<ForOp>(range.offset, range.size, range.stride);
  *iv = ValueHandle(forOp.getInductionVar());
  auto *body = forOp.getBody();
  enter(body, /*prev=*/1);
}

ValueHandle mlir::edsc::LoopRangeBuilder::
operator()(std::function<void(void)> fun) {
  if (fun)
    fun();
  exit();
  return ValueHandle::null();
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<SubViewOp::Range> ranges) {
  loops.reserve(ranges.size());
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    loops.emplace_back(ivs[i], ranges[i]);
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> ranges) {
  loops.reserve(ranges.size());
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    loops.emplace_back(ivs[i], ranges[i]);
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<Value> ranges)
    : LoopNestRangeBuilder(
          ivs, SmallVector<ValueHandle, 4>(ranges.begin(), ranges.end())) {}

ValueHandle LoopNestRangeBuilder::LoopNestRangeBuilder::
operator()(std::function<void(void)> fun) {
  if (fun)
    fun();
  for (auto &lit : reverse(loops)) {
    lit({});
  }
  return ValueHandle::null();
}

namespace mlir {
namespace edsc {
template <>
GenericLoopNestRangeBuilder<loop::ForOp>::GenericLoopNestRangeBuilder(
    ArrayRef<edsc::ValueHandle *> ivs, ArrayRef<Value> ranges) {
  builder = std::make_unique<LoopNestRangeBuilder>(ivs, ranges);
}

template <>
GenericLoopNestRangeBuilder<AffineForOp>::GenericLoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<Value> ranges) {
  SmallVector<ValueHandle, 4> lbs;
  SmallVector<ValueHandle, 4> ubs;
  SmallVector<int64_t, 4> steps;
  for (Value range : ranges) {
    assert(range.getType() && "expected linalg.range type");
    assert(range.getDefiningOp() && "need operations to extract range parts");
    RangeOp rangeOp = cast<RangeOp>(range.getDefiningOp());
    lbs.emplace_back(ValueHandle(rangeOp.min()));
    ubs.emplace_back(ValueHandle(rangeOp.max()));
    steps.emplace_back(ValueHandle(rangeOp.step()));
  }
  builder = std::make_unique<AffineLoopNestBuilder>(ivs, lbs, ubs, steps);
}
} // namespace edsc
} // namespace mlir

static void getMaxDimIndex(ArrayRef<StructuredIndexed> structuredIndices,
                           unsigned &pos) {
  for (auto sidx : structuredIndices) {
    for (auto expr : sidx.getExprs()) {
      expr.walk([&pos](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineDimExpr>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
}

Operation *mlir::edsc::makeGenericLinalgOp(
    ArrayRef<IterType> iteratorTypes, ArrayRef<StructuredIndexed> inputs,
    ArrayRef<StructuredIndexed> outputBuffers, ArrayRef<Type> resultTensorTypes,
    function_ref<void(ArrayRef<BlockArgument>)> regionBuilder,
    ArrayRef<Value> otherValues, ArrayRef<Attribute> otherAttributes) {
  assert(
      llvm::all_of(llvm::make_range(outputBuffers.begin(), outputBuffers.end()),
                   [](Value v) { return v.getType().isa<MemRefType>(); }) &&
      "output operands must all be buffers.");
  auto &builder = edsc::ScopedContext::getBuilder();
  auto *ctx = builder.getContext();
  unsigned nInputs = inputs.size();
  unsigned nOutputs = outputBuffers.size() + resultTensorTypes.size();
  unsigned maxPos = 0;
  getMaxDimIndex(inputs, maxPos);
  getMaxDimIndex(outputBuffers, maxPos);
  // maxPos is 0 indexed, need to turn this into a count (i.e. +1)
  unsigned nDims = maxPos + 1;

  SmallVector<AffineMap, 4> maps;
  maps.reserve(nInputs + nOutputs);
  for (auto in : inputs)
    maps.push_back(
        AffineMap::get(/*dimCount=*/nDims, /*symbolCount=*/0, in.getExprs()));
  for (auto out : outputBuffers)
    maps.push_back(
        AffineMap::get(/*dimCount=*/nDims, /*symbolCount=*/0, out.getExprs()));

  unsigned nViews = nInputs + nOutputs;
  SmallVector<Value, 4> values;
  values.reserve(nViews);
  values.append(inputs.begin(), inputs.end());
  values.append(outputBuffers.begin(), outputBuffers.end());

  auto iteratorStrTypes = functional::map(toString, iteratorTypes);
  // clang-format off
  auto *op =
      edsc::ScopedContext::getBuilder()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(),
              resultTensorTypes,
              values,
              IntegerAttr::get(IntegerType::get(64, ctx), nInputs),
              IntegerAttr::get(IntegerType::get(64, ctx), nOutputs),
              builder.getAffineMapArrayAttr(maps),
              builder.getStrArrayAttr(iteratorStrTypes),
              StringAttr() /*doc*/,
              FlatSymbolRefAttr() /*fun*/,
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
  BlockHandle b;
  auto handles = makeValueHandles(blockTypes);
  BlockBuilder(&b, op->getRegion(0),
               makeHandlePointers(MutableArrayRef<ValueHandle>(handles)))(
      [&] { regionBuilder(b.getBlock()->getArguments()); });
  assert(op->getRegion(0).getBlocks().size() == 1);
  return op;
}

void mlir::edsc::ops::macRegionBuilder(ArrayRef<BlockArgument> args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 3 && "expected 3 block arguments");
  ValueHandle a(args[0]), b(args[1]), c(args[2]);
  linalg_yield((c + a * b).getValue());
}

Operation *mlir::edsc::ops::linalg_pointwise(UnaryPointwiseOpBuilder unaryOp,
                                             StructuredIndexed I,
                                             StructuredIndexed O,
                                             ArrayRef<Type> resultTensorTypes) {
  SmallVector<edsc::IterType, 4> iterTypes(O.getExprs().size(),
                                           edsc::IterType::Parallel);
  auto fun = [&unaryOp](ArrayRef<BlockArgument> args) {
    assert(args.size() == 2 && "expected 2 block arguments");
    ValueHandle a(args[0]);
    linalg_yield(unaryOp(a));
  };

  // Distinguish between tensor and buffer semantics.
  if (O.getType().isa<MemRefType>()) {
    assert(resultTensorTypes.empty());
    return makeGenericLinalgOp(iterTypes, {I}, {O}, {}, fun);
  }
  return makeGenericLinalgOp(iterTypes, {I, O}, {}, resultTensorTypes, fun);
}

Operation *
mlir::edsc::ops::linalg_pointwise_tanh(StructuredIndexed I, StructuredIndexed O,
                                       ArrayRef<Type> resultTensorTypes) {
  ;
  using edsc::intrinsics::tanh;
  UnaryPointwiseOpBuilder unOp([](ValueHandle a) -> Value { return tanh(a); });
  return linalg_pointwise(unOp, I, O, resultTensorTypes);
}

/// Binary pointwise operation (with broadcast) entry point.
Operation *mlir::edsc::ops::linalg_pointwise(BinaryPointwiseOpBuilder binaryOp,
                                             StructuredIndexed I1,
                                             StructuredIndexed I2,
                                             StructuredIndexed O,
                                             ArrayRef<Type> resultTensorTypes) {
  SmallVector<edsc::IterType, 4> iterTypes(O.getExprs().size(),
                                           edsc::IterType::Parallel);
  auto fun = [&binaryOp](ArrayRef<BlockArgument> args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    ValueHandle a(args[0]), b(args[1]);
    linalg_yield(binaryOp(a, b));
  };
  // Distinguish between tensor and buffer semantics.
  if (O.getType().isa<MemRefType>()) {
    assert(resultTensorTypes.empty());
    return makeGenericLinalgOp(iterTypes, {I1, I2}, {O}, {}, fun);
  }
  return makeGenericLinalgOp(iterTypes, {I1, I2, O}, {}, resultTensorTypes,
                             fun);
}

Operation *
mlir::edsc::ops::linalg_pointwise_add(StructuredIndexed I1,
                                      StructuredIndexed I2, StructuredIndexed O,
                                      ArrayRef<Type> resultTensorTypes) {
  using edsc::op::operator+;
  BinaryPointwiseOpBuilder binOp(
      [](ValueHandle a, ValueHandle b) -> Value { return a + b; });
  return linalg_pointwise(binOp, I1, I2, O, resultTensorTypes);
}

Operation *
mlir::edsc::ops::linalg_pointwise_max(StructuredIndexed I1,
                                      StructuredIndexed I2, StructuredIndexed O,
                                      ArrayRef<Type> resultTensorTypes) {
  BinaryPointwiseOpBuilder binOp([](ValueHandle a, ValueHandle b) -> Value {
    using edsc::intrinsics::select;
    using edsc::op::operator>;
    return select(a > b, a, b).getValue();
  });
  return linalg_pointwise(binOp, I1, I2, O, resultTensorTypes);
}

Operation *mlir::edsc::ops::linalg_matmul(ValueHandle vA, ValueHandle vB,
                                          ValueHandle vC,
                                          ArrayRef<Type> resultTensorTypes) {
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(vC);

  assert(!C.getType().isa<MemRefType>() || resultTensorTypes.empty());
  StructuredIndexed allIndexed[3]{A({m, k}), B({k, n}), C({m, n})};
  ArrayRef<StructuredIndexed> inputs =
      (C.getType().isa<MemRefType>())
          ? ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 2}
          : ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 3};
  ArrayRef<StructuredIndexed> outputs =
      (C.getType().isa<MemRefType>())
          ? ArrayRef<StructuredIndexed>{allIndexed + 2, allIndexed + 3}
          : ArrayRef<StructuredIndexed>{};
  return makeGenericLinalgOp(
      {IterType::Parallel, IterType::Parallel, IterType::Reduction}, inputs,
      outputs, resultTensorTypes, macRegionBuilder);
}

Operation *mlir::edsc::ops::linalg_conv_nhwc(ValueHandle vI, ValueHandle vW,
                                             ValueHandle vO,
                                             ArrayRef<Type> resultTensorTypes,
                                             ArrayRef<int> strides,
                                             ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO(ntv) some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IterType::Parallel;
  auto red = IterType::Reduction;
  auto s = strides;
  auto d = dilations;

  AffineExpr b, f, h, w, kh, kw, c;
  bindDims(ctx, b, f, h, w, kh, kw, c);
  unsigned numDims = c.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);

  assert(!O.getType().isa<MemRefType>() || resultTensorTypes.empty());
  // Roundtrip to flattened form to serve as canonicalization and ensure
  // consistent ordering of subexpressions.
  // clang-format off
  StructuredIndexed allIndexed[3] = {
      I({b,
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, f}),
      O({b, h, w, f})};
  // clang-format on
  auto inputs = (O.getType().isa<MemRefType>())
                    ? ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 2}
                    : ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 3};
  ArrayRef<StructuredIndexed> outputs =
      (O.getType().isa<MemRefType>())
          ? ArrayRef<StructuredIndexed>{allIndexed + 2, allIndexed + 3}
          : ArrayRef<StructuredIndexed>{};
  return makeGenericLinalgOp({par, par, par, par, red, red, red}, inputs,
                             outputs, resultTensorTypes, macRegionBuilder);
}

Operation *mlir::edsc::ops::linalg_dilated_conv_nhwc(
    ValueHandle vI, ValueHandle vW, ValueHandle vO,
    ArrayRef<Type> resultTensorTypes, int depthMultiplier,
    ArrayRef<int> strides, ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO(ntv) some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IterType::Parallel;
  auto red = IterType::Reduction;
  auto s = strides;
  auto d = dilations;

  // clang-format off
  AffineExpr b, dm, c, h, w, kh, kw;
  bindDims(ctx, b, dm, c, h, w, kh, kw);
  unsigned numDims = kw.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);
  // Roundtrip to flattened form to serve as canonicalization and ensure
  // consistent ordering of subexpressions.
  // clang-format off
  StructuredIndexed allIndexed[3] = {
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, dm}),
      O({b, h, w, simplifyAffineExpr(c * depthMultiplier + dm, numDims, 0)})};
  // clang-format on
  auto inputs = (O.getType().isa<MemRefType>())
                    ? ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 2}
                    : ArrayRef<StructuredIndexed>{allIndexed, allIndexed + 3};
  ArrayRef<StructuredIndexed> outputs =
      (O.getType().isa<MemRefType>())
          ? ArrayRef<StructuredIndexed>{allIndexed + 2, allIndexed + 3}
          : ArrayRef<StructuredIndexed>{};
  return makeGenericLinalgOp({par, par, par, par, par, red, red}, inputs,
                             outputs, resultTensorTypes, macRegionBuilder);
}
