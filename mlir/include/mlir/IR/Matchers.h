//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's
// include/llvm/IR/PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MATCHERS_H
#define MLIR_IR_MATCHERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Query/Matcher/MatchersInternal.h"
#include "mlir/Query/Query.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

namespace detail {

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <
    typename AttrClass,
    // Require AttrClass to be a derived class from Attribute and get its
    // value type
    typename ValueType = typename std::enable_if_t<
        std::is_base_of<Attribute, AttrClass>::value, AttrClass>::ValueType,
    // Require the ValueType is not void
    typename = std::enable_if_t<!std::is_void<ValueType>::value>>
struct attr_value_binder {
  ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    if (auto intAttr = llvm::dyn_cast<AttrClass>(attr)) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches operations that have the `ConstantLike` trait.
struct constant_op_matcher {
  bool match(Operation *op) { return op->hasTrait<OpTrait::ConstantLike>(); }
};

/// The matcher that matches operations that have the specified op name.
struct NameOpMatcher {
  NameOpMatcher(StringRef name) : name(name) {}
  bool match(Operation *op) { return op->getName().getStringRef() == name; }

  std::string name;
};

/// The matcher that matches operations that have the specified attribute name.
struct AttrOpMatcher {
  AttrOpMatcher(StringRef attrName) : attrName(attrName) {}
  bool match(Operation *op) { return op->hasAttr(attrName); }

  std::string attrName;
};

/// The matcher that matches operations that have the `ConstantLike` trait, and
/// binds the folded attribute value.
template <typename AttrT>
struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  constant_op_binder() : bind_value(nullptr) {}

  bool match(Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>())
      return false;

    // Fold the constant to an attribute.
    SmallVector<OpFoldResult, 1> foldedOp;
    LogicalResult result = op->fold(/*operands=*/std::nullopt, foldedOp);
    (void)result;
    assert(succeeded(result) && "expected ConstantLike op to be foldable");

    if (auto attr = llvm::dyn_cast<AttrT>(cast<Attribute>(foldedOp.front()))) {
      if (bind_value)
        *bind_value = attr;
      return true;
    }
    return false;
  }
};

/// A matcher that matches operations that implement the
/// `InferIntRangeInterface` interface, and binds the inferred range.
struct infer_int_range_op_binder {
  IntegerValueRange *bind_value;

  explicit infer_int_range_op_binder(IntegerValueRange *bind_value)
      : bind_value(bind_value) {}

  bool match(Operation *op) {
    auto inferIntRangeOp = dyn_cast<InferIntRangeInterface>(op);
    if (!inferIntRangeOp)
      return false;

    // Set the range of all integer operands to the maximal range.
    SmallVector<IntegerValueRange> argRanges =
        llvm::map_to_vector(op->getOperands(), IntegerValueRange::getMaxRange);

    // Infer the result result range if possible.
    bool matched = false;
    auto setResultRanges = [&](Value value,
                               const IntegerValueRange &argRanges) {
      if (argRanges.isUninitialized())
        return;
      if (value != op->getResult(0))
        return;
      *bind_value = argRanges;
      matched = true;
    };
    inferIntRangeOp.inferResultRangesFromOptional(argRanges, setResultRanges);
    return matched;
  }
};

/// The matcher that matches operations that have the specified attribute
/// name, and binds the attribute value.
template <typename AttrT>
struct AttrOpBinder {
  /// Creates a matcher instance that binds the attribute value to
  /// bind_value if match succeeds.
  AttrOpBinder(StringRef attrName, AttrT *bindValue)
      : attrName(attrName), bindValue(bindValue) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  AttrOpBinder(StringRef attrName) : attrName(attrName), bindValue(nullptr) {}

  bool match(Operation *op) {
    if (auto attr = op->getAttrOfType<AttrT>(attrName)) {
      if (bindValue)
        *bindValue = attr;
      return true;
    }
    return false;
  }
  StringRef attrName;
  AttrT *bindValue;
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// float Attribute or Operation and binds the constant float value.
struct constant_float_value_binder {
  FloatAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_float_value_binder(FloatAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    attr_value_binder<FloatAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<Attribute>());

    return false;
  }

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<FloatType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat float value that fulfills a predicate.
struct constant_float_predicate_matcher {
  bool (*predicate)(const APFloat &);

  bool match(Attribute attr) {
    APFloat value(APFloat::Bogus());
    return constant_float_value_binder(&value).match(attr) && predicate(value);
  }

  bool match(Operation *op) {
    APFloat value(APFloat::Bogus());
    return constant_float_value_binder(&value).match(op) && predicate(value);
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer Attribute or Operation and binds the constant integer value.
struct constant_int_value_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_value_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    attr_value_binder<IntegerAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<Attribute>());

    return false;
  }

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<IntegerType, IndexType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat integer value that fulfills a predicate.
struct constant_int_predicate_matcher {
  bool (*predicate)(const APInt &);

  bool match(Attribute attr) {
    APInt value;
    return constant_int_value_binder(&value).match(attr) && predicate(value);
  }

  bool match(Operation *op) {
    APInt value;
    return constant_int_value_binder(&value).match(op) && predicate(value);
  }
};

/// A matcher that matches a given a constant scalar / vector splat / tensor
/// splat integer value or a constant integer range that fulfills a predicate.
struct constant_int_range_predicate_matcher {
  bool (*predicate)(const ConstantIntRanges &);

  bool match(Attribute attr) {
    APInt value;
    return constant_int_value_binder(&value).match(attr) &&
           predicate(ConstantIntRanges::constant(value));
  }

  bool match(Operation *op) {
    // Try to match a constant integer value first.
    APInt value;
    if (constant_int_value_binder(&value).match(op))
      return predicate(ConstantIntRanges::constant(value));

    // Otherwise, try to match an operation that implements the
    // `InferIntRangeInterface` interface.
    IntegerValueRange range;
    return infer_int_range_op_binder(&range).match(op) &&
           predicate(range.getValue());
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass>
struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

/// Trait to check whether T provides a 'match' method with type
/// `MatchTarget` (Value, Operation, or Attribute).
template <typename T, typename MatchTarget>
using has_compatible_matcher_t =
    decltype(std::declval<T>().match(std::declval<MatchTarget>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_compatible_matcher_t,
                                   MatcherClass, Value>::value,
                 bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_compatible_matcher_t,
                                   MatcherClass, Operation *>::value,
                 bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  if (auto *defOp = op->getOperand(idx).getDefiningOp())
    return matcher.match(defOp);
  return false;
}

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(Value op) const { return true; }
};

/// Terminal matcher, always returns true.
struct AnyCapturedValueMatcher {
  Value *what;
  AnyCapturedValueMatcher(Value *what) : what(what) {}
  bool match(Value op) const {
    *what = op;
    return true;
  }
};

/// Binds to a specific value and matches it.
struct PatternMatcherValue {
  PatternMatcherValue(Value val) : value(val) {}
  bool match(Value val) const { return val == value; }
  Value value;
};

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {

  (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
   ...);
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  detail::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                        std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher {
  RecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (!isa<OpType>(op) || op->getNumOperands() != sizeof...(OperandMatchers))
      return false;
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

/// Fills `backwardSlice` with the computed backward slice (i.e.
/// all the transitive defs of op)
///
/// The implementation traverses the def chains in postorder traversal for
/// efficiency reasons: if an operation is already in `backwardSlice`, no
/// need to traverse its definitions again. Since use-def chains form a DAG,
/// this terminates.
///
/// Upon return to the root call, `backwardSlice` is filled with a
/// postorder list of defs. This happens to be a topological order, from the
/// point of view of the use-def chains.
///
/// Example starting from node 8
/// ============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
///    {1, 2, 5, 3, 4, 6}
///

class BackwardSliceMatcher {
public:
  BackwardSliceMatcher(mlir::query::matcher::DynMatcher &&innerMatcher,
                       int64_t maxDepth)
      : innerMatcher(std::move(innerMatcher)), maxDepth(maxDepth) {}

  bool match(Operation *op, SetVector<Operation *> &backwardSlice,
             mlir::query::QueryOptions &options) {

    if (innerMatcher.match(op) &&
        matches(op, backwardSlice, options, maxDepth)) {
      if (!options.inclusive) {
        // Don't insert the top level operation, we just queried on it and don't
        // want it in the results.
        backwardSlice.remove(op);
      }
      return true;
    }
    return false;
  }

private:
  bool matches(Operation *op, SetVector<Operation *> &backwardSlice,
               mlir::query::QueryOptions &options, int64_t remainingDepth) {

    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return false;
    }

    auto processValue = [&](Value value) {
      // We need to check the current depth level;
      // if we have reached level 0, we stop further traversing
      if (remainingDepth == 0) {
        return;
      }
      if (auto *definingOp = value.getDefiningOp()) {
        // We omit traversing the same operations
        if (backwardSlice.count(definingOp) == 0)
          matches(definingOp, backwardSlice, options, remainingDepth - 1);
      } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (options.omitBlockArguments)
          return;
        Block *block = blockArg.getOwner();

        Operation *parentOp = block->getParentOp();
        // TODO: determine whether we want to recurse backward into the other
        // blocks of parentOp, which are not technically backward unless they
        // flow into us. For now, just bail.
        if (parentOp && backwardSlice.count(parentOp) == 0) {
          if (parentOp->getNumRegions() != 1 &&
              parentOp->getRegion(0).getBlocks().size() != 1) {
            llvm::errs()
                << "Error: Expected parentOp to have exactly one region and "
                << "exactly one block, but found " << parentOp->getNumRegions()
                << " regions and "
                << (parentOp->getRegion(0).getBlocks().size()) << " blocks.\n";
          };
          matches(parentOp, backwardSlice, options, remainingDepth - 1);
        }
      } else {
        llvm_unreachable("No definingOp and not a block argument\n");
        return;
      }
    };

    if (!options.omitUsesFromAbove) {
      llvm::for_each(op->getRegions(), [&](Region &region) {
        // Walk this region recursively to collect the regions that descend from
        // this op's nested regions (inclusive).
        SmallPtrSet<Region *, 4> descendents;
        region.walk(
            [&](Region *childRegion) { descendents.insert(childRegion); });
        region.walk([&](Operation *op) {
          for (OpOperand &operand : op->getOpOperands()) {
            if (!descendents.contains(operand.get().getParentRegion()))
              processValue(operand.get());
          }
        });
      });
    }

    llvm::for_each(op->getOperands(), processValue);
    backwardSlice.insert(op);
    return true;
  }

private:
  // The outer matcher (e.g., BackwardSliceMatcher) relies on the innerMatcher
  // to determine whether we want to traverse the DAG or not. For example, we
  // want to explore the DAG only if the top-level operation name is
  // "arith.addf".
  mlir::query::matcher::DynMatcher innerMatcher;

  // maxDepth specifies the maximum depth that the matcher can traverse in the
  // DAG. For example, if maxDepth is 2, the matcher will explore the defining
  // operations of the top-level op up to 2 levels.
  int64_t maxDepth;
};

/// Fills `forwardSlice` with the computed forward slice (i.e. all
/// the transitive uses of op)
///
///
/// The implementation traverses the use chains in postorder traversal for
/// efficiency reasons: if an operation is already in `forwardSlice`, no
/// need to traverse its uses again. Since use-def chains form a DAG, this
/// terminates.
///
/// Upon return to the root call, `forwardSlice` is filled with a
/// postorder list of uses (i.e. a reverse topological order). To get a proper
/// topological order, we just reverse the order in `forwardSlice` before
/// returning.
///
/// Example starting from node 0
/// ============================
///
///               0
///    ___________|___________
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
/// 1. after getting back to the root getForwardSlice, `forwardSlice` may
///    contain:
///      {9, 7, 8, 5, 1, 2, 6, 3, 4}
/// 2. reversing the result of 1. gives:
///      {4, 3, 6, 2, 1, 5, 8, 7, 9}
///
class ForwardSliceMatcher {
public:
  ForwardSliceMatcher(mlir::query::matcher::DynMatcher &&innerMatcher,
                      int64_t maxDepth)
      : innerMatcher(std::move(innerMatcher)), maxDepth(maxDepth) {}

  bool match(Operation *op, SetVector<Operation *> &forwardSlice,
             mlir::query::QueryOptions &options) {
    if (innerMatcher.match(op) &&
        matches(op, forwardSlice, options, maxDepth)) {
      if (!options.inclusive) {
        // Don't insert the top level operation, we just queried on it and don't
        // want it in the results.
        forwardSlice.remove(op);
      }
      // Reverse to get back the actual topological order.
      // std::reverse does not work out of the box on SetVector and I want an
      // in-place swap based thing (the real std::reverse, not the LLVM
      // adapter).
      SmallVector<Operation *, 0> v(forwardSlice.takeVector());
      forwardSlice.insert(v.rbegin(), v.rend());
      return true;
    }
    return false;
  }

private:
  bool matches(Operation *op, SetVector<Operation *> &forwardSlice,
               mlir::query::QueryOptions &options, int64_t remainingDepth) {

    // We need to check the current depth level;
    // if we have reached level 0, we stop further traversing and insert
    // the last user in def-use chain
    if (remainingDepth == 0) {
      forwardSlice.insert(op);
      return true;
    }

    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &blockOp : block)
          if (forwardSlice.count(&blockOp) == 0)
            matches(&blockOp, forwardSlice, options, remainingDepth - 1);
    for (Value result : op->getResults()) {
      for (Operation *userOp : result.getUsers())
        // We omit traversing the same operations
        if (forwardSlice.count(userOp) == 0)
          matches(userOp, forwardSlice, options, remainingDepth - 1);
    }

    forwardSlice.insert(op);
    return true;
  }

private:
  // The outer matcher e.g (ForwardSliceMatcher) relies on the innerMatcher to
  // determine whether we want to traverse the graph or not. E.g: we want to
  // explore the DAG only if the top level operation name is "arith.addf"
  mlir::query::matcher::DynMatcher innerMatcher;

  // maxDepth specifies the maximum depth that the matcher can traverse the
  // graph E.g: if maxDepth is 2, the matcher will explore the user
  // operations of the top level op up to 2 levels
  int64_t maxDepth;
};

} // namespace detail

// Matches transitive defs of a top level operation up to 1 level
inline detail::BackwardSliceMatcher
m_DefinedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::BackwardSliceMatcher(std::move(innerMatcher), 1);
}

// Matches transitive defs of a top level operation up to N levels
inline detail::BackwardSliceMatcher
m_GetDefinitions(mlir::query::matcher::DynMatcher innerMatcher,
                 int64_t maxDepth) {
  assert(maxDepth >= 0 && "maxDepth must be non-negative");
  return detail::BackwardSliceMatcher(std::move(innerMatcher), maxDepth);
}

// Matches uses of a top level operation up to 1 level
inline detail::ForwardSliceMatcher
m_UsedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::ForwardSliceMatcher(std::move(innerMatcher), 1);
}

// Matches uses of a top level operation up to N  levels
inline detail::ForwardSliceMatcher
m_GetUses(mlir::query::matcher::DynMatcher innerMatcher, int64_t maxDepth) {
  assert(maxDepth >= 0 && "maxDepth must be non-negative");
  return detail::ForwardSliceMatcher(std::move(innerMatcher), maxDepth);
}

/// Matches a constant foldable operation.
inline detail::constant_op_matcher m_Constant() {
  return detail::constant_op_matcher();
}

/// Matches a named attribute operation.
inline detail::AttrOpMatcher m_Attr(StringRef attrName) {
  return detail::AttrOpMatcher(attrName);
}

/// Matches a named operation.
inline detail::NameOpMatcher m_Op(StringRef opName) {
  return detail::NameOpMatcher(opName);
}

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return detail::constant_op_binder<AttrT>(bind_value);
}

/// Matches a named attribute operation and writes the value to bind_value.
template <typename AttrT>
inline detail::AttrOpBinder<AttrT> m_Attr(StringRef attrName,
                                          AttrT *bindValue) {
  return detail::AttrOpBinder<AttrT>(attrName, bindValue);
}

/// Matches a constant scalar / vector splat / tensor splat float (both positive
/// and negative) zero.
inline detail::constant_float_predicate_matcher m_AnyZeroFloat() {
  return {[](const APFloat &value) { return value.isZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive zero.
inline detail::constant_float_predicate_matcher m_PosZeroFloat() {
  return {[](const APFloat &value) { return value.isPosZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative zero.
inline detail::constant_float_predicate_matcher m_NegZeroFloat() {
  return {[](const APFloat &value) { return value.isNegZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float ones.
inline detail::constant_float_predicate_matcher m_OneFloat() {
  return {[](const APFloat &value) {
    return APFloat(value.getSemantics(), 1) == value;
  }};
}

/// Matches a constant scalar / vector splat / tensor splat float ones.
inline detail::constant_float_predicate_matcher m_NaNFloat() {
  return {[](const APFloat &value) { return value.isNaN(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive
/// infinity.
inline detail::constant_float_predicate_matcher m_PosInfFloat() {
  return {[](const APFloat &value) {
    return !value.isNegative() && value.isInfinity();
  }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative
/// infinity.
inline detail::constant_float_predicate_matcher m_NegInfFloat() {
  return {[](const APFloat &value) {
    return value.isNegative() && value.isInfinity();
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
inline detail::constant_int_predicate_matcher m_Zero() {
  return {[](const APInt &value) { return 0 == value; }};
}

/// Matches a constant scalar / vector splat / tensor splat integer that is any
/// non-zero value.
inline detail::constant_int_predicate_matcher m_NonZero() {
  return {[](const APInt &value) { return 0 != value; }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// unsigned integer range that does not contain zero. Note that this matcher
/// interprets the target value as an unsigned integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutZeroU() {
  return {[](const ConstantIntRanges &range) { return range.umin().ugt(0); }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// signed integer range that does not contain zero. Note that this matcher
/// interprets the target value as a signed integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutZeroS() {
  return {[](const ConstantIntRanges &range) {
    return range.smin().sgt(0) || range.smax().slt(0);
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// signed integer range that does not contain minus one. Note
/// that this matcher interprets the target value as a signed integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutNegOneS() {
  return {[](const ConstantIntRanges &range) {
    return range.smin().sgt(-1) || range.smax().slt(-1);
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
inline detail::constant_int_predicate_matcher m_One() {
  return {[](const APInt &value) { return 1 == value; }};
}

/// Matches the given OpClass.
template <typename OpClass>
inline detail::op_matcher<OpClass> m_Op() {
  return detail::op_matcher<OpClass>();
}

/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(Value value, const Pattern &pattern) {
  assert(value);
  // TODO: handle other cases
  if (auto *op = value.getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  assert(op);
  return const_cast<Pattern &>(pattern).match(op);
}

/// Entry point for matching a pattern over an Attribute. Returns `false`
/// when `attr` is null.
template <typename Pattern>
inline bool matchPattern(Attribute attr, const Pattern &pattern) {
  static_assert(llvm::is_detected<detail::has_compatible_matcher_t, Pattern,
                                  Attribute>::value,
                "Pattern does not support matching Attributes");
  if (!attr)
    return false;
  return const_cast<Pattern &>(pattern).match(attr);
}

/// Matches a constant holding a scalar/vector/tensor float (splat) and
/// writes the float value to bind_value.
inline detail::constant_float_value_binder
m_ConstantFloat(FloatAttr::ValueType *bind_value) {
  return detail::constant_float_value_binder(bind_value);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_value_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_value_binder(bind_value);
}

template <typename OpType, typename... Matchers>
auto m_Op(Matchers... matchers) {
  return detail::RecursivePatternMatcher<OpType, Matchers...>(matchers...);
}

namespace matchers {
inline auto m_Any() { return detail::AnyValueMatcher(); }
inline auto m_Any(Value *val) { return detail::AnyCapturedValueMatcher(val); }
inline auto m_Val(Value v) { return detail::PatternMatcherValue(v); }
} // namespace matchers

} // namespace mlir

#endif // MLIR_IR_MATCHERS_H
