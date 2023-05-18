//===- Successor.h - TableGen successor definitions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_SUCCESSOR_H_
#define MLIR_TABLEGEN_SUCCESSOR_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"

namespace mlir {
namespace tblgen {

// Wrapper class providing helper methods for accessing Successor defined in
// TableGen.
class Successor : public Constraint {
public:
  using Constraint::Constraint;

  static bool classof(const Constraint *c) {
    return c->getKind() == CK_Successor;
  }

  // Returns true if this successor is variadic.
  bool isVariadic() const;
};

// A struct bundling a successor's constraint and its name.
struct NamedSuccessor {
  // Returns true if this successor is variadic.
  bool isVariadic() const { return constraint.isVariadic(); }

  StringRef name;
  Successor constraint;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_SUCCESSOR_H_
