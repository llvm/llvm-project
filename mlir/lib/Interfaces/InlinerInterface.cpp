//===- InlinerInterface.cpp ---- Interface for inlining -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the inlining interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InlinerInterface.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

bool InlinerInterface::isLegalToInline(Operation *call, Operation *callable,
                                       bool wouldBeCloned) const {
  if (auto *handler = getInterfaceFor(call))
    return handler->isLegalToInline(call, callable, wouldBeCloned);
  return false;
}

bool InlinerInterface::isLegalToInline(Region *dest, Region *src,
                                       bool wouldBeCloned,
                                       IRMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(dest->getParentOp()))
    return handler->isLegalToInline(dest, src, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::isLegalToInline(Operation *op, Region *dest,
                                       bool wouldBeCloned,
                                       IRMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(op))
    return handler->isLegalToInline(op, dest, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::shouldAnalyzeRecursively(Operation *op) const {
  auto *handler = getInterfaceFor(op);
  return handler ? handler->shouldAnalyzeRecursively(op) : true;
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op, Block *newDest) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, newDest);
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op,
                                        ValueRange valuesToRepl) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, valuesToRepl);
}

Value InlinerInterface::handleArgument(OpBuilder &builder, Operation *call,
                                       Operation *callable, Value argument,
                                       DictionaryAttr argumentAttrs) const {
  auto *handler = getInterfaceFor(callable);
  assert(handler && "expected valid dialect handler");
  return handler->handleArgument(builder, call, callable, argument,
                                 argumentAttrs);
}

Value InlinerInterface::handleResult(OpBuilder &builder, Operation *call,
                                     Operation *callable, Value result,
                                     DictionaryAttr resultAttrs) const {
  auto *handler = getInterfaceFor(callable);
  assert(handler && "expected valid dialect handler");
  return handler->handleResult(builder, call, callable, result, resultAttrs);
}

void InlinerInterface::processInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) const {
  auto *handler = getInterfaceFor(call);
  assert(handler && "expected valid dialect handler");
  handler->processInlinedCallBlocks(call, inlinedBlocks);
}
