//===- GIMatchDagEdge.h - Represent node shared operand lists ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGEDGE_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAGEDGE_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
class GIMatchDagInstr;
class GIMatchDagOperand;

/// Represents an edge that connects two instructions together via a pair of
/// operands. For example:
///     %a = FOO ...
///     %0 = BAR %a
///     %1 = BAZ %a
/// would have two edges for %a like so:
///     BAR:Op#1 --[a]----> Op#0:FOO
///                         ^
///     BAZ:Op#1 --[a]------/
/// Ideally, all edges in the DAG are from a use to a def as this is a many
/// to one edge but edges from defs to uses are supported too.
class GIMatchDagEdge {
  /// The name of the edge. For example,
  ///     (FOO $a, $b, $c)
  ///     (BAR $d, $e, $a)
  /// will create an edge named 'a' to connect FOO to BAR. Although the name
  /// refers to the edge, the canonical value of 'a' is the operand that defines
  /// it.
  StringRef Name;
  const GIMatchDagInstr *FromMI;
  const GIMatchDagOperand *FromMO;
  const GIMatchDagInstr *ToMI;
  const GIMatchDagOperand *ToMO;

public:
  GIMatchDagEdge(StringRef Name, const GIMatchDagInstr *FromMI, const GIMatchDagOperand *FromMO,
            const GIMatchDagInstr *ToMI, const GIMatchDagOperand *ToMO)
      : Name(Name), FromMI(FromMI), FromMO(FromMO), ToMI(ToMI), ToMO(ToMO) {}

  StringRef getName() const { return Name; }
  const GIMatchDagInstr *getFromMI() const { return FromMI; }
  const GIMatchDagOperand *getFromMO() const { return FromMO; }
  const GIMatchDagInstr *getToMI() const { return ToMI; }
  const GIMatchDagOperand *getToMO() const { return ToMO; }

  /// Flip the direction of the edge.
  void reverse();

  /// Does this edge run from a def to (one of many) uses?
  bool isDefToUse() const;

  LLVM_DUMP_METHOD void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

raw_ostream &operator<<(raw_ostream &OS, const GIMatchDagEdge &E);

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGEDGE_H
