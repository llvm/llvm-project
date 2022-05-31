//===- Operation.h - MLIR PDLL ODS Operation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_OPERATION_H_
#define MLIR_TOOLS_PDLL_ODS_OPERATION_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
namespace pdll {
namespace ods {
class AttributeConstraint;
class TypeConstraint;

//===----------------------------------------------------------------------===//
// VariableLengthKind
//===----------------------------------------------------------------------===//

enum VariableLengthKind { Single, Optional, Variadic };

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// This class provides an ODS representation of a specific operation attribute.
/// This includes the name, optionality, and more.
class Attribute {
public:
  /// Return the name of this operand.
  StringRef getName() const { return name; }

  /// Return true if this attribute is optional.
  bool isOptional() const { return optional; }

  /// Return the constraint of this attribute.
  const AttributeConstraint &getConstraint() const { return constraint; }

private:
  Attribute(StringRef name, bool optional,
            const AttributeConstraint &constraint)
      : name(name.str()), optional(optional), constraint(constraint) {}

  /// The ODS name of the attribute.
  std::string name;

  /// A flag indicating if the attribute is optional.
  bool optional;

  /// The ODS constraint of this attribute.
  const AttributeConstraint &constraint;

  /// Allow access to the private constructor.
  friend class Operation;
};

//===----------------------------------------------------------------------===//
// OperandOrResult
//===----------------------------------------------------------------------===//

/// This class provides an ODS representation of a specific operation operand or
/// result. This includes the name, variable length flags, and more.
class OperandOrResult {
public:
  /// Return the name of this value.
  StringRef getName() const { return name; }

  /// Returns true if this value is variable length, i.e. if it is Variadic or
  /// Optional.
  bool isVariableLength() const {
    return variableLengthKind != VariableLengthKind::Single;
  }

  /// Returns true if this value is variadic (Note this is false if the value is
  /// Optional).
  bool isVariadic() const {
    return variableLengthKind == VariableLengthKind::Variadic;
  }

  /// Returns the variable length kind of this value.
  VariableLengthKind getVariableLengthKind() const {
    return variableLengthKind;
  }

  /// Return the constraint of this value.
  const TypeConstraint &getConstraint() const { return constraint; }

private:
  OperandOrResult(StringRef name, VariableLengthKind variableLengthKind,
                  const TypeConstraint &constraint)
      : name(name.str()), variableLengthKind(variableLengthKind),
        constraint(constraint) {}

  /// The ODS name of this value.
  std::string name;

  /// The variable length kind of this value.
  VariableLengthKind variableLengthKind;

  /// The ODS constraint of this value.
  const TypeConstraint &constraint;

  /// Allow access to the private constructor.
  friend class Operation;
};

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

/// This class provides an ODS representation of a specific operation. This
/// includes all of the information necessary for use by the PDL frontend for
/// generating code for a pattern rewrite.
class Operation {
public:
  /// Return the source location of this operation.
  SMRange getLoc() const { return location; }

  /// Append an attribute to this operation.
  void appendAttribute(StringRef name, bool optional,
                       const AttributeConstraint &constraint) {
    attributes.emplace_back(Attribute(name, optional, constraint));
  }

  /// Append an operand to this operation.
  void appendOperand(StringRef name, VariableLengthKind variableLengthKind,
                     const TypeConstraint &constraint) {
    operands.emplace_back(
        OperandOrResult(name, variableLengthKind, constraint));
  }

  /// Append a result to this operation.
  void appendResult(StringRef name, VariableLengthKind variableLengthKind,
                    const TypeConstraint &constraint) {
    results.emplace_back(OperandOrResult(name, variableLengthKind, constraint));
  }

  /// Returns the name of the operation.
  StringRef getName() const { return name; }

  /// Returns the summary of the operation.
  StringRef getSummary() const { return summary; }

  /// Returns the description of the operation.
  StringRef getDescription() const { return description; }

  /// Returns the attributes of this operation.
  ArrayRef<Attribute> getAttributes() const { return attributes; }

  /// Returns the operands of this operation.
  ArrayRef<OperandOrResult> getOperands() const { return operands; }

  /// Returns the results of this operation.
  ArrayRef<OperandOrResult> getResults() const { return results; }

  /// Return if the operation is known to support result type inferrence.
  bool hasResultTypeInferrence() const { return supportsTypeInferrence; }

private:
  Operation(StringRef name, StringRef summary, StringRef desc,
            bool supportsTypeInferrence, SMLoc loc);

  /// The name of the operation.
  std::string name;

  /// The documentation of the operation.
  std::string summary;
  std::string description;

  /// Flag indicating if the operation is known to support type inferrence.
  bool supportsTypeInferrence;

  /// The source location of this operation.
  SMRange location;

  /// The operands of the operation.
  SmallVector<OperandOrResult> operands;

  /// The results of the operation.
  SmallVector<OperandOrResult> results;

  /// The attributes of the operation.
  SmallVector<Attribute> attributes;

  /// Allow access to the private constructor.
  friend class Dialect;
};
} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_ODS_OPERATION_H_
