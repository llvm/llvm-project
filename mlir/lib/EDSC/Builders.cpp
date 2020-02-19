//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/Optional.h"

using namespace mlir;
using namespace mlir::edsc;

mlir::edsc::ScopedContext::ScopedContext(OpBuilder &builder, Location location)
    : builder(builder), location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()),
      nestedBuilder(nullptr) {
  getCurrentScopedContext() = this;
}

/// Sets the insertion point of the builder to 'newInsertPt' for the duration
/// of the scope. The existing insertion point of the builder is restored on
/// destruction.
mlir::edsc::ScopedContext::ScopedContext(OpBuilder &builder,
                                         OpBuilder::InsertPoint newInsertPt,
                                         Location location)
    : builder(builder), prevBuilderInsertPoint(builder.saveInsertionPoint()),
      location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()),
      nestedBuilder(nullptr) {
  getCurrentScopedContext() = this;
  builder.restoreInsertionPoint(newInsertPt);
}

mlir::edsc::ScopedContext::~ScopedContext() {
  assert(!nestedBuilder &&
         "Active NestedBuilder must have been exited at this point!");
  if (prevBuilderInsertPoint)
    builder.restoreInsertionPoint(*prevBuilderInsertPoint);
  getCurrentScopedContext() = enclosingScopedContext;
}

ScopedContext *&mlir::edsc::ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

OpBuilder &mlir::edsc::ScopedContext::getBuilder() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return ScopedContext::getCurrentScopedContext()->builder;
}

Location mlir::edsc::ScopedContext::getLocation() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return ScopedContext::getCurrentScopedContext()->location;
}

MLIRContext *mlir::edsc::ScopedContext::getContext() {
  return getBuilder().getContext();
}

ValueHandle &mlir::edsc::ValueHandle::operator=(const ValueHandle &other) {
  assert(t == other.t && "Wrong type capture");
  assert(!v && "ValueHandle has already been captured, use a new name!");
  v = other.v;
  return *this;
}

ValueHandle ValueHandle::create(StringRef name, ArrayRef<ValueHandle> operands,
                                ArrayRef<Type> resultTypes,
                                ArrayRef<NamedAttribute> attributes) {
  Operation *op =
      OperationHandle::create(name, operands, resultTypes, attributes);
  if (op->getNumResults() == 1)
    return ValueHandle(op->getResult(0));
  llvm_unreachable("unsupported operation, use an OperationHandle instead");
}

OperationHandle OperationHandle::create(StringRef name,
                                        ArrayRef<ValueHandle> operands,
                                        ArrayRef<Type> resultTypes,
                                        ArrayRef<NamedAttribute> attributes) {
  OperationState state(ScopedContext::getLocation(), name);
  SmallVector<Value, 4> ops(operands.begin(), operands.end());
  state.addOperands(ops);
  state.addTypes(resultTypes);
  for (const auto &attr : attributes) {
    state.addAttribute(attr.first, attr.second);
  }
  return OperationHandle(ScopedContext::getBuilder().createOperation(state));
}

BlockHandle mlir::edsc::BlockHandle::create(ArrayRef<Type> argTypes) {
  auto &currentB = ScopedContext::getBuilder();
  auto *ib = currentB.getInsertionBlock();
  auto ip = currentB.getInsertionPoint();
  BlockHandle res;
  res.block = ScopedContext::getBuilder().createBlock(ib->getParent());
  // createBlock sets the insertion point inside the block.
  // We do not want this behavior when using declarative builders with nesting.
  currentB.setInsertionPoint(ib, ip);
  for (auto t : argTypes) {
    res.block->addArgument(t);
  }
  return res;
}

BlockHandle mlir::edsc::BlockHandle::createInRegion(Region &region,
                                                    ArrayRef<Type> argTypes) {
  auto &currentB = ScopedContext::getBuilder();
  BlockHandle res;
  region.push_back(new Block);
  res.block = &region.back();
  // createBlock sets the insertion point inside the block.
  // We do not want this behavior when using declarative builders with nesting.
  OpBuilder::InsertionGuard g(currentB);
  currentB.setInsertionPoint(res.block, res.block->begin());
  for (auto t : argTypes) {
    res.block->addArgument(t);
  }
  return res;
}

void mlir::edsc::LoopBuilder::operator()(function_ref<void(void)> fun) {
  // Call to `exit` must be explicit and asymmetric (cannot happen in the
  // destructor) because of ordering wrt comma operator.
  /// The particular use case concerns nested blocks:
  ///
  /// ```c++
  ///    For (&i, lb, ub, 1)({
  ///      /--- destructor for this `For` is not always called before ...
  ///      V
  ///      For (&j1, lb, ub, 1)({
  ///        some_op_1,
  ///      }),
  ///      /--- ... this scope is entered, resulting in improperly nested IR.
  ///      V
  ///      For (&j2, lb, ub, 1)({
  ///        some_op_2,
  ///      }),
  ///    });
  /// ```
  if (fun)
    fun();
  exit();
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle bh, Append) {
  assert(bh && "Expected already captured BlockHandle");
  enter(bh.getBlock());
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle *bh,
                                       ArrayRef<ValueHandle *> args) {
  assert(!*bh && "BlockHandle already captures a block, use "
                 "the explicit BockBuilder(bh, Append())({}) syntax instead.");
  SmallVector<Type, 8> types;
  for (auto *a : args) {
    assert(!a->hasValue() &&
           "Expected delayed ValueHandle that has not yet captured.");
    types.push_back(a->getType());
  }
  *bh = BlockHandle::create(types);
  for (auto it : llvm::zip(args, bh->getBlock()->getArguments())) {
    *(std::get<0>(it)) = ValueHandle(std::get<1>(it));
  }
  enter(bh->getBlock());
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle *bh, Region &region,
                                       ArrayRef<ValueHandle *> args) {
  assert(!*bh && "BlockHandle already captures a block, use "
                 "the explicit BockBuilder(bh, Append())({}) syntax instead.");
  SmallVector<Type, 8> types;
  for (auto *a : args) {
    assert(!a->hasValue() &&
           "Expected delayed ValueHandle that has not yet captured.");
    types.push_back(a->getType());
  }
  *bh = BlockHandle::createInRegion(region, types);
  for (auto it : llvm::zip(args, bh->getBlock()->getArguments())) {
    *(std::get<0>(it)) = ValueHandle(std::get<1>(it));
  }
  enter(bh->getBlock());
}

/// Only serves as an ordering point between entering nested block and creating
/// stmts.
void mlir::edsc::BlockBuilder::operator()(function_ref<void(void)> fun) {
  // Call to `exit` must be explicit and asymmetric (cannot happen in the
  // destructor) because of ordering wrt comma operator.
  if (fun)
    fun();
  exit();
}
