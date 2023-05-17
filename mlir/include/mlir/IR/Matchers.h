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

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

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

  bool match(const Attribute &attr) {
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

  StringRef name;
};

/// The matcher that matches operations that have the specified attribute name.
struct AttrOpMatcher {
  AttrOpMatcher(StringRef attrName) : attrName(attrName) {}
  bool match(Operation *op) { return op->hasAttr(attrName); }

  StringRef attrName;
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

    if (auto attr = llvm::dyn_cast<AttrT>(foldedOp.front().get<Attribute>())) {
      if (bind_value)
        *bind_value = attr;
      return true;
    }
    return false;
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
/// float operation and binds the constant float value.
struct constant_float_op_binder {
  FloatAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_float_op_binder(FloatAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (llvm::isa<FloatType>(type))
      return attr_value_binder<FloatAttr>(bind_value).match(attr);
    if (llvm::isa<VectorType, RankedTensorType>(type)) {
      if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(attr)) {
        return attr_value_binder<FloatAttr>(bind_value)
            .match(splatAttr.getSplatValue<Attribute>());
      }
    }
    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat float value that fulfills a predicate.
struct constant_float_predicate_matcher {
  bool (*predicate)(const APFloat &);

  bool match(Operation *op) {
    APFloat value(APFloat::Bogus());
    return constant_float_op_binder(&value).match(op) && predicate(value);
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer operation and binds the constant integer value.
struct constant_int_op_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_op_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (llvm::isa<IntegerType, IndexType>(type))
      return attr_value_binder<IntegerAttr>(bind_value).match(attr);
    if (llvm::isa<VectorType, RankedTensorType>(type)) {
      if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(attr)) {
        return attr_value_binder<IntegerAttr>(bind_value)
            .match(splatAttr.getSplatValue<Attribute>());
      }
    }
    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat integer value that fulfills a predicate.
struct constant_int_predicate_matcher {
  bool (*predicate)(const APInt &);

  bool match(Operation *op) {
    APInt value;
    return constant_int_op_binder(&value).match(op) && predicate(value);
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass>
struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

/// Trait to check whether T provides a 'match' method with type
/// `OperationOrValue`.
template <typename T, typename OperationOrValue>
using has_operation_or_value_matcher_t =
    decltype(std::declval<T>().match(std::declval<OperationOrValue>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_operation_or_value_matcher_t,
                                   MatcherClass, Value>::value,
                 bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_operation_or_value_matcher_t,
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

} // namespace detail

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
  // TODO: handle other cases
  if (auto *op = value.getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  return const_cast<Pattern &>(pattern).match(op);
}

/// Matches a constant holding a scalar/vector/tensor float (splat) and
/// writes the float value to bind_value.
inline detail::constant_float_op_binder
m_ConstantFloat(FloatAttr::ValueType *bind_value) {
  return detail::constant_float_op_binder(bind_value);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_op_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_op_binder(bind_value);
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
