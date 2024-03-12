//===- InlinerInterface.h - Inliner interface -------------------*- C++ -*-===//
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

#ifndef MLIR_INTERFACES_INLINERINTERFACE_H
#define MLIR_INTERFACES_INLINERINTERFACE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"

namespace mlir {

class IRMapping;
class OpBuilder;

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

/// This is the interface that must be implemented by the dialects of operations
/// to be inlined. This interface should only handle the operations of the
/// given dialect.
class DialectInlinerInterface
    : public DialectInterface::Base<DialectInlinerInterface> {
public:
  DialectInlinerInterface(Dialect *dialect) : Base(dialect) {}

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Returns true if the given operation 'callable', that implements the
  /// 'CallableOpInterface', can be inlined into the position given call
  /// operation 'call', that is registered to the current dialect and implements
  /// the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  /// given 'callable' is set to be cloned during the inlining process, or false
  /// if the region is set to be moved in-place(i.e. no duplicates would be
  /// created).
  virtual bool isLegalToInline(Operation *call, Operation *callable,
                               bool wouldBeCloned) const {
    return false;
  }

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'wouldBeCloned' is set to true if the given 'src' region is set to be
  /// cloned during the inlining process, or false if the region is set to be
  /// moved in-place(i.e. no duplicates would be created). 'valueMapping'
  /// contains any remapped values from within the 'src' region. This can be
  /// used to examine what values will replace entry arguments into the 'src'
  /// region for example.
  virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                               IRMapping &valueMapping) const {
    return false;
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the given region, false otherwise.
  /// 'wouldBeCloned' is set to true if the given 'op' is set to be cloned
  /// during the inlining process, or false if the operation is set to be moved
  /// in-place(i.e. no duplicates would be created). 'valueMapping' contains any
  /// remapped values from within the 'src' region. This can be used to examine
  /// what values may potentially replace the operands to 'op'.
  virtual bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                               IRMapping &valueMapping) const {
    return false;
  }

  /// This hook is invoked on an operation that contains regions. It should
  /// return true if the analyzer should recurse within the regions of this
  /// operation when computing legality and cost, false otherwise. The default
  /// implementation returns true.
  virtual bool shouldAnalyzeRecursively(Operation *op) const { return true; }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary. This overload is called when the inlined region has more
  /// than one block. The 'newDest' block represents the new final branching
  /// destination of blocks within this region, i.e. operations that release
  /// control to the parent operation will likely now branch to this block.
  /// Its block arguments correspond to any values that need to be replaced by
  /// terminators within the inlined region.
  virtual void handleTerminator(Operation *op, Block *newDest) const {
    llvm_unreachable("must implement handleTerminator in the case of multiple "
                     "inlined blocks");
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary. This overload is called when the inlined region only
  /// contains one block. 'valuesToReplace' contains the previously returned
  /// values of the call site before inlining. These values must be replaced by
  /// this callback if they had any users (for example for traditional function
  /// calls, these are directly replaced with the operands of the `return`
  /// operation). The given 'op' will be removed by the caller, after this
  /// function has been called.
  virtual void handleTerminator(Operation *op,
                                ValueRange valuesToReplace) const {
    llvm_unreachable(
        "must implement handleTerminator in the case of one inlined block");
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned. For example, this hook may be invoked in the following
  /// scenarios:
  ///   func @foo(i32) -> i32 { ... }
  ///
  ///   // Mismatched input operand
  ///   ... = foo.call @foo(%input : i16) -> i32
  ///
  ///   // Mismatched result type.
  ///   ... = foo.call @foo(%input : i32) -> i16
  ///
  /// NOTE: This hook may be invoked before the 'isLegal' checks above.
  virtual Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                               Type resultType,
                                               Location conversionLoc) const {
    return nullptr;
  }

  /// Hook to transform the call arguments before using them to replace the
  /// callee arguments. Returns a value of the same type or the `argument`
  /// itself if nothing changed. The `argumentAttrs` dictionary is non-null even
  /// if no attribute is present. The hook is called after converting the
  /// callsite argument types using the materializeCallConversion callback, and
  /// right before inlining the callee region. Any operations created using the
  /// provided `builder` are inserted right before the inlined callee region. An
  /// example use case is the insertion of copies for by value arguments.
  virtual Value handleArgument(OpBuilder &builder, Operation *call,
                               Operation *callable, Value argument,
                               DictionaryAttr argumentAttrs) const {
    return argument;
  }

  /// Hook to transform the callee results before using them to replace the call
  /// results. Returns a value of the same type or the `result` itself if
  /// nothing changed. The `resultAttrs` dictionary is non-null even if no
  /// attribute is present. The hook is called right before handling
  /// terminators, and obtains the callee result before converting its type
  /// using the `materializeCallConversion` callback. Any operations created
  /// using the provided `builder` are inserted right after the inlined callee
  /// region. An example use case is the insertion of copies for by value
  /// results. NOTE: This hook is invoked after inlining the `callable` region.
  virtual Value handleResult(OpBuilder &builder, Operation *call,
                             Operation *callable, Value result,
                             DictionaryAttr resultAttrs) const {
    return result;
  }

  /// Process a set of blocks that have been inlined for a call. This callback
  /// is invoked before inlined terminator operations have been processed.
  virtual void processInlinedCallBlocks(
      Operation *call, iterator_range<Region::iterator> inlinedBlocks) const {}
};

/// This interface provides the hooks into the inlining interface.
/// Note: this class automatically collects 'DialectInlinerInterface' objects
/// registered to each dialect within the given context.
class InlinerInterface
    : public DialectInterfaceCollection<DialectInlinerInterface> {
public:
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
};

} // namespace mlir

#endif // MLIR_INTERFACES_INLINERINTERFACE_H
