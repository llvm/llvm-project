//===--- MatchersInternal.cpp----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/MatchersInternal.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::query::matcher {

namespace internal {

bool allOfVariadicOperator(Operation *op, SetVector<Operation *> *matchedOps,
                           ArrayRef<DynMatcher> innerMatchers) {
  return llvm::all_of(innerMatchers, [&](const DynMatcher &matcher) {
    if (matchedOps)
      return matcher.match(op, *matchedOps);
    return matcher.match(op);
  });
}
bool anyOfVariadicOperator(Operation *op, SetVector<Operation *> *matchedOps,
                           ArrayRef<DynMatcher> innerMatchers) {
  return llvm::any_of(innerMatchers, [&](const DynMatcher &matcher) {
    if (matchedOps)
      return matcher.match(op, *matchedOps);
    return matcher.match(op);
  });
}
} // namespace internal
} // namespace mlir::query::matcher
