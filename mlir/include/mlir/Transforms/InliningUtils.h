//===- InliningUtils.h - Inliner utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines interfaces for various inlining utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_INLININGUTILS_H
#define MLIR_TRANSFORMS_INLININGUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include <optional>

namespace mlir {

class Block;
class IRMapping;
class CallableOpInterface;
class CallOpInterface;
class OpBuilder;
class Operation;
class Region;
class TypeRange;
class Value;
class ValueRange;
class DialectInlinerInterface;

/// This interface provides the hooks into the inlining interface.
/// Note: this class automatically collects 'DialectInlinerInterface' objects
/// registered to each dialect within the given context.
class InlinerInterface
    : public DialectInterfaceCollection<DialectInlinerInterface> {
public:
  using CloneCallbackSigTy = void(OpBuilder &builder, Region *src,
                                  Block *inlineBlock, Block *postInsertBlock,
                                  IRMapping &mapper,
                                  bool shouldCloneInlinedRegion);
  using CloneCallbackTy = std::function<CloneCallbackSigTy>;

  using Base::Base;

  /// Process a set of blocks that have been inlined. This callback is invoked
  /// *before* inlined terminator operations have been processed.
  virtual void
  processInlinedBlocks(iterator_range<Region::iterator> inlinedBlocks) {}

  /// These hooks mirror the hooks for the DialectInlinerInterface, with default
  /// implementations that call the hook on the handler for the dialect 'op' is
  /// registered to.

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  virtual bool isLegalToInline(Operation *call, Operation *callable,
                               bool wouldBeCloned) const;
  virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                               IRMapping &valueMapping) const;
  virtual bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                               IRMapping &valueMapping) const;
  virtual bool shouldAnalyzeRecursively(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  virtual void handleTerminator(Operation *op, Block *newDest) const;
  virtual void handleTerminator(Operation *op, ValueRange valuesToRepl) const;

  virtual Value handleArgument(OpBuilder &builder, Operation *call,
                               Operation *callable, Value argument,
                               DictionaryAttr argumentAttrs) const;
  virtual Value handleResult(OpBuilder &builder, Operation *call,
                             Operation *callable, Value result,
                             DictionaryAttr resultAttrs) const;

  virtual void processInlinedCallBlocks(
      Operation *call, iterator_range<Region::iterator> inlinedBlocks) const;

  virtual bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const;
};

//===----------------------------------------------------------------------===//
// Inline Methods.
//===----------------------------------------------------------------------===//

/// This function inlines a region, 'src', into another. This function returns
/// failure if it is not possible to inline this function. If the function
/// returned failure, then no changes to the module have been made.
///
/// The provided 'inlinePoint' must be within a region, and corresponds to the
/// location where the 'src' region should be inlined. 'mapping' contains any
/// remapped operands that are used within the region, and *must* include
/// remappings for the entry arguments to the region. 'resultsToReplace'
/// corresponds to any results that should be replaced by terminators within the
/// inlined region. 'regionResultTypes' specifies the expected return types of
/// the terminators in the region. 'inlineLoc' is an optional Location that, if
/// provided, will be used to update the inlined operations' location
/// information. 'shouldCloneInlinedRegion' corresponds to whether the source
/// region should be cloned into the 'inlinePoint' or spliced directly.
LogicalResult
inlineRegion(InlinerInterface &interface,
             function_ref<InlinerInterface::CloneCallbackSigTy> cloneCallback,
             Region *src, Operation *inlinePoint, IRMapping &mapper,
             ValueRange resultsToReplace, TypeRange regionResultTypes,
             std::optional<Location> inlineLoc = std::nullopt,
             bool shouldCloneInlinedRegion = true);
LogicalResult
inlineRegion(InlinerInterface &interface,
             function_ref<InlinerInterface::CloneCallbackSigTy> cloneCallback,
             Region *src, Block *inlineBlock, Block::iterator inlinePoint,
             IRMapping &mapper, ValueRange resultsToReplace,
             TypeRange regionResultTypes,
             std::optional<Location> inlineLoc = std::nullopt,
             bool shouldCloneInlinedRegion = true);

/// This function is an overload of the above 'inlineRegion' that allows for
/// providing the set of operands ('inlinedOperands') that should be used
/// in-favor of the region arguments when inlining.
LogicalResult
inlineRegion(InlinerInterface &interface,
             function_ref<InlinerInterface::CloneCallbackSigTy> cloneCallback,
             Region *src, Operation *inlinePoint, ValueRange inlinedOperands,
             ValueRange resultsToReplace,
             std::optional<Location> inlineLoc = std::nullopt,
             bool shouldCloneInlinedRegion = true);
LogicalResult
inlineRegion(InlinerInterface &interface,
             function_ref<InlinerInterface::CloneCallbackSigTy> cloneCallback,
             Region *src, Block *inlineBlock, Block::iterator inlinePoint,
             ValueRange inlinedOperands, ValueRange resultsToReplace,
             std::optional<Location> inlineLoc = std::nullopt,
             bool shouldCloneInlinedRegion = true);

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult
inlineCall(InlinerInterface &interface,
           function_ref<InlinerInterface::CloneCallbackSigTy> cloneCallback,
           CallOpInterface call, CallableOpInterface callable, Region *src,
           bool shouldCloneInlinedRegion = true);

} // namespace mlir

#include "mlir/Transforms/DialectInlinerInterface.h.inc"

#endif // MLIR_TRANSFORMS_INLININGUTILS_H
