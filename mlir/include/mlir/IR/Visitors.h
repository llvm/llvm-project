//===- Visitors.h - Utilities for visiting operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for walking and visiting operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VISITORS_H
#define MLIR_IR_VISITORS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class Diagnostic;
class InFlightDiagnostic;
class Operation;
class Block;
class Region;

/// A utility result that is used to signal how to proceed with an ongoing walk:
///   * Interrupt: the walk will be interrupted and no more operations, regions
///   or blocks will be visited.
///   * Advance: the walk will continue.
///   * Skip: the walk of the current operation, region or block and their
///   nested elements that haven't been visited already will be skipped and will
///   continue with the next operation, region or block.
class WalkResult {
  enum ResultEnum { Interrupt, Advance, Skip } result;

public:
  WalkResult(ResultEnum result) : result(result) {}

  /// Allow LogicalResult to interrupt the walk on failure.
  WalkResult(LogicalResult result)
      : result(failed(result) ? Interrupt : Advance) {}

  /// Allow diagnostics to interrupt the walk.
  WalkResult(Diagnostic &&) : result(Interrupt) {}
  WalkResult(InFlightDiagnostic &&) : result(Interrupt) {}

  bool operator==(const WalkResult &rhs) const { return result == rhs.result; }

  static WalkResult interrupt() { return {Interrupt}; }
  static WalkResult advance() { return {Advance}; }
  static WalkResult skip() { return {Skip}; }

  /// Returns true if the walk was interrupted.
  bool wasInterrupted() const { return result == Interrupt; }

  /// Returns true if the walk was skipped.
  bool wasSkipped() const { return result == Skip; }
};

/// Traversal order for region, block and operation walk utilities.
enum class WalkOrder { PreOrder, PostOrder };

namespace detail {
/// Helper templates to deduce the first argument of a callback parameter.
template <typename Ret, typename Arg> Arg first_argument_type(Ret (*)(Arg));
template <typename Ret, typename F, typename Arg>
Arg first_argument_type(Ret (F::*)(Arg));
template <typename Ret, typename F, typename Arg>
Arg first_argument_type(Ret (F::*)(Arg) const);
template <typename F>
decltype(first_argument_type(&F::operator())) first_argument_type(F);

/// Type definition of the first argument to the given callable 'T'.
template <typename T>
using first_argument = decltype(first_argument_type(std::declval<T>()));

/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. Regions, blocks and operations at the same nesting
/// level are visited in lexicographical order. The walk order for enclosing
/// regions, blocks and operations with respect to their nested ones is
/// specified by 'order'. These methods are invoked for void-returning
/// callbacks. A callback on a block or operation is allowed to erase that block
/// or operation only if the walk is in post-order. See non-void method for
/// pre-order erasure.
void walk(Operation *op, function_ref<void(Region *)> callback,
          WalkOrder order);
void walk(Operation *op, function_ref<void(Block *)> callback, WalkOrder order);
void walk(Operation *op, function_ref<void(Operation *)> callback,
          WalkOrder order);
/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. Regions, blocks and operations at the same nesting
/// level are visited in lexicographical order. The walk order for enclosing
/// regions, blocks and operations with respect to their nested ones is
/// specified by 'order'. This method is invoked for skippable or interruptible
/// callbacks. A callback on a block or operation is allowed to erase that block
/// or operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
WalkResult walk(Operation *op, function_ref<WalkResult(Region *)> callback,
                WalkOrder order);
WalkResult walk(Operation *op, function_ref<WalkResult(Block *)> callback,
                WalkOrder order);
WalkResult walk(Operation *op, function_ref<WalkResult(Operation *)> callback,
                WalkOrder order);

// Below are a set of functions to walk nested operations. Users should favor
// the direct `walk` methods on the IR classes(Operation/Block/etc) over these
// methods. They are also templated to allow for statically dispatching based
// upon the type of the callback function.

/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. Regions, blocks and operations at the same nesting
/// level are visited in lexicographical order. The walk order for enclosing
/// regions, blocks and operations with respect to their nested ones is
/// specified by 'Order' (post-order by default). A callback on a block or
/// operation is allowed to erase that block or operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
/// This method is selected for callbacks that operate on Region*, Block*, and
/// Operation*.
///
/// Example:
///   op->walk([](Region *r) { ... });
///   op->walk([](Block *b) { ... });
///   op->walk([](Operation *op) { ... });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename FuncTy,
    typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<
    llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value, RetT>::type
walk(Operation *op, FuncTy &&callback) {
  return detail::walk(op, function_ref<RetT(ArgT)>(callback), Order);
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. Regions, blocks and operations at the same nesting
/// level are visited in lexicographical order. The walk order for enclosing
/// regions, blocks and operations with respect to their nested ones is
/// specified by 'order' (post-order by default). This method is selected for
/// void-returning callbacks that operate on a specific derived operation type.
/// A callback on an operation is allowed to erase that operation only if the
/// walk is in post-order. See non-void method for pre-order erasure.
///
/// Example:
///   op->walk([](ReturnOp op) { ... });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename FuncTy,
    typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<
    !llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value &&
        std::is_same<RetT, void>::value,
    RetT>::type
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      callback(derivedOp);
  };
  return detail::walk(op, function_ref<RetT(Operation *)>(wrapperFn), Order);
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. Regions, blocks and operations at the same nesting level
/// are visited in lexicographical order. The walk order for enclosing regions,
/// blocks and operations with respect to their nested ones is specified by
/// 'Order' (post-order by default). This method is selected for WalkReturn
/// returning skippable or interruptible callbacks that operate on a specific
/// derived operation type. A callback on an operation is allowed to erase that
/// operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
///
/// Example:
///   op->walk([](ReturnOp op) {
///     if (some_invariant)
///       return WalkResult::skip();
///     if (another_invariant)
///       return WalkResult::interrupt();
///     return WalkResult::advance();
///   });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename FuncTy,
    typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<
    !llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value &&
        std::is_same<RetT, WalkResult>::value,
    RetT>::type
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      return callback(derivedOp);
    return WalkResult::advance();
  };
  return detail::walk(op, function_ref<RetT(Operation *)>(wrapperFn), Order);
}

/// Utility to provide the return type of a templated walk method.
template <typename FnT>
using walkResultType = decltype(walk(nullptr, std::declval<FnT>()));
} // namespace detail

} // namespace mlir

#endif
