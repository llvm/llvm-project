//===- Type.h - Type class --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type wrapper to simplify using TableGen Record defining a MLIR Type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_TYPE_H_
#define MLIR_TABLEGEN_TYPE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Dialect.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing Type constraints defined in
// TableGen.
class TypeConstraint : public Constraint {
public:
  explicit TypeConstraint(const llvm::Record *record);
  explicit TypeConstraint(const llvm::DefInit *init);

  static bool classof(const Constraint *c) { return c->getKind() == CK_Type; }

  // Returns true if this is an optional type constraint.
  bool isOptional() const;

  // Returns true if this is a variadic type constraint.
  bool isVariadic() const;

  // Returns true if this is a nested variadic type constraint.
  bool isVariadicOfVariadic() const;

  // Return the segment size attribute used if this is a variadic of variadic
  // constraint. Asserts isVariadicOfVariadic() is true.
  StringRef getVariadicOfVariadicSegmentSizeAttr() const;

  // Returns true if this is a variable length type constraint. This is either
  // variadic or optional.
  bool isVariableLength() const { return isOptional() || isVariadic(); }

  // Returns the builder call for this constraint if this is a buildable type,
  // returns None otherwise.
  Optional<StringRef> getBuilderCall() const;

  // Return the C++ class name for this type (which may just be ::mlir::Type).
  std::string getCPPClassName() const;
};

// Wrapper class with helper methods for accessing Types defined in TableGen.
class Type : public TypeConstraint {
public:
  explicit Type(const llvm::Record *record);

  // Returns the description of the type.
  StringRef getDescription() const;

  // Returns the dialect for the type if defined.
  Dialect getDialect() const;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPE_H_
