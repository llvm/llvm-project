//===- CppTranslationUtils.h - Helpers to used in C++ emitter ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common helper functions used across different dialects
// during the Cpp translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_H
#define MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Target/Cpp/TranslateToCpp.h"

using namespace mlir;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

/// Determine whether expression \p expressionOp should be emitted inline, i.e.
/// as part of its user. This function recommends inlining of any expressions
/// that can be inlined unless it is used by another expression, under the
/// assumption that  any expression fusion/re-materialization was taken care of
/// by transformations run by the backend.
bool shouldBeInlined(emitc::ExpressionOp expressionOp);

LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation,
                              Attribute value);

LogicalResult printBinaryOperation(CppEmitter &emitter, Operation *operation,
                                   StringRef binaryOperator);

LogicalResult printUnaryOperation(CppEmitter &emitter, Operation *operation,
                                  StringRef unaryOperator);

LogicalResult printCallOperation(CppEmitter &emitter, Operation *callOp,
                                 StringRef callee);

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *functionOp,
                                ArrayRef<Type> arguments);

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *functionOp,
                                Region::BlockArgListType arguments);

LogicalResult printFunctionBody(CppEmitter &emitter, Operation *functionOp,
                                Region::BlockListType &blocks);

#endif // MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_H
