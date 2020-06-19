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

mlir::edsc::ScopedContext::ScopedContext(OpBuilder &b, Location location)
    : builder(b), guard(builder), location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()) {
  getCurrentScopedContext() = this;
}

/// Sets the insertion point of the builder to 'newInsertPt' for the duration
/// of the scope. The existing insertion point of the builder is restored on
/// destruction.
mlir::edsc::ScopedContext::ScopedContext(OpBuilder &b,
                                         OpBuilder::InsertPoint newInsertPt,
                                         Location location)
    : builder(b), guard(builder), location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()) {
  getCurrentScopedContext() = this;
  builder.restoreInsertionPoint(newInsertPt);
}

mlir::edsc::ScopedContext::~ScopedContext() {
  getCurrentScopedContext() = enclosingScopedContext;
}

ScopedContext *&mlir::edsc::ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

OpBuilder &mlir::edsc::ScopedContext::getBuilderRef() {
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
  return getBuilderRef().getContext();
}

BlockHandle mlir::edsc::BlockHandle::create(ArrayRef<Type> argTypes) {
  auto &currentB = ScopedContext::getBuilderRef();
  auto *ib = currentB.getInsertionBlock();
  auto ip = currentB.getInsertionPoint();
  BlockHandle res;
  res.block = ScopedContext::getBuilderRef().createBlock(ib->getParent());
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
  BlockHandle res;
  region.push_back(new Block);
  res.block = &region.back();
  // createBlock sets the insertion point inside the block.
  // We do not want this behavior when using declarative builders with nesting.
  OpBuilder::InsertionGuard g(ScopedContext::getBuilderRef());
  ScopedContext::getBuilderRef().setInsertionPoint(res.block,
                                                   res.block->begin());
  res.block->addArguments(argTypes);
  return res;
}

Block *mlir::edsc::createBlock(TypeRange argTypes) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Block *block = builder.getInsertionBlock();
  assert(block != nullptr &&
         "insertion point not set up in the builder within ScopedContext");

  return createBlockInRegion(*block->getParent(), argTypes);
}

Block *mlir::edsc::createBlockInRegion(Region &region, TypeRange argTypes) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  OpBuilder::InsertionGuard guard(builder);
  return builder.createBlock(&region, {}, argTypes);
}

void mlir::edsc::appendToBlock(Block *block,
                               function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  OpBuilder::InsertionGuard guard(builder);
  if (block->empty() || block->back().isKnownNonTerminator())
    builder.setInsertionPointToEnd(block);
  else
    builder.setInsertionPoint(&block->back());
  builderFn(block->getArguments());
}

Block *mlir::edsc::buildInNewBlock(TypeRange argTypes,
                                   function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Block *block = builder.getInsertionBlock();
  assert(block != nullptr &&
         "insertion point not set up in the builder within ScopedContext");
  return buildInNewBlock(*block->getParent(), argTypes, builderFn);
}

Block *mlir::edsc::buildInNewBlock(Region &region, TypeRange argTypes,
                                   function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  Block *block = createBlockInRegion(region, argTypes);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builderFn(block->getArguments());
  return block;
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle bh, Append) {
  assert(bh && "Expected already captured BlockHandle");
  enter(bh.getBlock());
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle *bh, ArrayRef<Type> types,
                                       MutableArrayRef<Value> args) {
  assert(!*bh && "BlockHandle already captures a block, use "
                 "the explicit BockBuilder(bh, Append())({}) syntax instead.");
  assert((args.empty() || args.size() == types.size()) &&
         "if args captures are specified, their number must match the number "
         "of types");
  *bh = BlockHandle::create(types);
  if (!args.empty())
    for (auto it : llvm::zip(args, bh->getBlock()->getArguments()))
      std::get<0>(it) = Value(std::get<1>(it));
  enter(bh->getBlock());
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle *bh, Region &region,
                                       ArrayRef<Type> types,
                                       MutableArrayRef<Value> args) {
  assert(!*bh && "BlockHandle already captures a block, use "
                 "the explicit BockBuilder(bh, Append())({}) syntax instead.");
  assert((args.empty() || args.size() == types.size()) &&
         "if args captures are specified, their number must match the number "
         "of types");
  *bh = BlockHandle::createInRegion(region, types);
  if (!args.empty())
    for (auto it : llvm::zip(args, bh->getBlock()->getArguments()))
      std::get<0>(it) = Value(std::get<1>(it));
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
