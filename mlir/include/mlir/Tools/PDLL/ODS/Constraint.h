//===- Constraint.h - MLIR PDLL ODS Constraints -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a PDLL description of ODS constraints. These are used to
// support the import of constraints defined outside of PDLL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_
#define MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace pdll {
namespace ods {

//===----------------------------------------------------------------------===//
// Constraint
//===----------------------------------------------------------------------===//

/// This class represents a generic ODS constraint.
class Constraint {
public:
  /// Return the unique name of this constraint.
  StringRef getName() const { return name; }

  /// Return the demangled name of this constraint. This tries to strip out bits
  /// of the name that are purely for uniquing, and show the underlying name. As
  /// such, this name does guarantee uniqueness and should only be used for
  /// logging or other lossy friendly "pretty" output.
  StringRef getDemangledName() const;

  /// Return the summary of this constraint.
  StringRef getSummary() const { return summary; }

protected:
  Constraint(StringRef name, StringRef summary)
      : name(name.str()), summary(summary.str()) {}
  Constraint(const Constraint &) = delete;

private:
  /// The name of the constraint.
  std::string name;
  /// A summary of the constraint.
  std::string summary;
};

//===----------------------------------------------------------------------===//
// AttributeConstraint
//===----------------------------------------------------------------------===//

/// This class represents a generic ODS Attribute constraint.
class AttributeConstraint : public Constraint {
public:
  /// Return the name of the underlying c++ class of this constraint.
  StringRef getCppClass() const { return cppClassName; }

private:
  AttributeConstraint(StringRef name, StringRef summary, StringRef cppClassName)
      : Constraint(name, summary), cppClassName(cppClassName.str()) {}

  /// The c++ class of the constraint.
  std::string cppClassName;

  /// Allow access to the constructor.
  friend class Context;
};

//===----------------------------------------------------------------------===//
// TypeConstraint
//===----------------------------------------------------------------------===//

/// This class represents a generic ODS Type constraint.
class TypeConstraint : public Constraint {
public:
  /// Return the name of the underlying c++ class of this constraint.
  StringRef getCppClass() const { return cppClassName; }

private:
  TypeConstraint(StringRef name, StringRef summary, StringRef cppClassName)
      : Constraint(name, summary), cppClassName(cppClassName.str()) {}

  /// The c++ class of the constraint.
  std::string cppClassName;

  /// Allow access to the constructor.
  friend class Context;
};

} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_
