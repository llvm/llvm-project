//===- PredicateExpanderDag.h - Composable predicate dag lowering ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared walker that lowers a composable predicate dag of the form
//
//   (all_of / any_of / not <leaf> ...)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_BASIC_PREDICATEEXPANDERDAG_H
#define LLVM_UTILS_TABLEGEN_BASIC_PREDICATEEXPANDERDAG_H

#include "llvm/ADT/STLFunctionalExtras.h"

namespace llvm {

class Init;
class Record;
class raw_ostream;

/// Walk a composable predicate dag rooted at \p Val and emit the combined
/// boolean expression to \p OS.
///
/// The structural operators handled are the dags `(not X)` and `(all_of ...)` /
/// `(any_of ...)`. Any non-dag operand (and the base leaf) is delegated to \p
/// EmitLeaf, which emits the leaf test and returns true on error.
///
/// If \p ParenIfBinOp is true, a surrounding pair of parentheses is emitted
/// when \p Val lowers to a binary (`&&` / `||`) expression. A `(not X)` always
/// parenthesizes its operand.
bool emitPredicateDag(const Record *Owner, const Init &Val, bool ParenIfBinOp,
                      raw_ostream &OS,
                      function_ref<bool(const Init &, raw_ostream &)> EmitLeaf);

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_BASIC_PREDICATEEXPANDERDAG_H
