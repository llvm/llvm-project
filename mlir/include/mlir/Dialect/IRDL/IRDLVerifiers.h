//===- IRDLVerifiers.h - IRDL verifiers --------------------------- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifiers for objects declared by IRDL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IRDLVERIFIERS_H
#define MLIR_DIALECT_IRDL_IRDLVERIFIERS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
class InFlightDiagnostic;
class DynamicAttrDefinition;
class DynamicTypeDefinition;
} // namespace mlir

namespace mlir {
namespace irdl {

class AttributeOp;
class Constraint;
class OperationOp;
class TypeOp;

/// Provides context to the verification of constraints.
/// It contains the assignment of variables to attributes, and the assignment
/// of variables to constraints.
class ConstraintVerifier {
public:
  ConstraintVerifier(ArrayRef<std::unique_ptr<Constraint>> constraints);

  /// Check that a constraint is satisfied by an attribute.
  ///
  /// Constraints may call other constraint verifiers. If that is the case,
  /// the constraint verifier will check if the variable is already assigned,
  /// and if so, check that the attribute is the same as the one assigned.
  /// If the variable is not assigned, the constraint verifier will
  /// assign the attribute to the variable, and check that the constraint
  /// is satisfied.
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr, unsigned variable);

private:
  /// The constraints that can be used for verification.
  ArrayRef<std::unique_ptr<Constraint>> constraints;

  /// The assignment of variables to attributes. Variables that are not assigned
  /// are represented by nullopt. Null attributes needs to be supported here as
  /// some attributes or types might use the null attribute to represent
  /// optional parameters.
  SmallVector<std::optional<Attribute>> assigned;
};

/// Once turned into IRDL verifiers, all constraints are
/// attribute constraints. Type constraints are represented
/// as `TypeAttr` attribute constraints to simplify verification.
/// Verification that a type constraint must yield a
/// `TypeAttr` attribute happens before conversion, at the MLIR level.
class Constraint {
public:
  virtual ~Constraint() = default;

  /// Check that an attribute is satisfying the constraint.
  ///
  /// Constraints may call other constraint verifiers. If that is the case,
  /// the constraint verifier will check if the variable is already assigned,
  /// and if so, check that the attribute is the same as the one assigned.
  /// If the variable is not assigned, the constraint verifier will
  /// assign the attribute to the variable, and check that the constraint
  /// is satisfied.
  virtual LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                               Attribute attr,
                               ConstraintVerifier &context) const = 0;
};

/// A constraint that checks that an attribute is equal to a given attribute.
class IsConstraint : public Constraint {
public:
  IsConstraint(Attribute expectedAttribute)
      : expectedAttribute(expectedAttribute) {}

  virtual ~IsConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  Attribute expectedAttribute;
};

/// A constraint that checks that an attribute is of a given attribute base
/// (e.g. IntegerAttr).
class BaseAttrConstraint : public Constraint {
public:
  BaseAttrConstraint(TypeID baseTypeID, StringRef baseName)
      : baseTypeID(baseTypeID), baseName(baseName) {}

  virtual ~BaseAttrConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  /// The expected base attribute typeID.
  TypeID baseTypeID;

  /// The base attribute name, only used for error reporting.
  StringRef baseName;
};

/// A constraint that checks that a type is of a given type base (e.g.
/// IntegerType).
class BaseTypeConstraint : public Constraint {
public:
  BaseTypeConstraint(TypeID baseTypeID, StringRef baseName)
      : baseTypeID(baseTypeID), baseName(baseName) {}

  virtual ~BaseTypeConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  /// The expected base type typeID.
  TypeID baseTypeID;

  /// The base type name, only used for error reporting.
  StringRef baseName;
};

/// A constraint that checks that an attribute is of a
/// specific dynamic attribute definition, and that all of its parameters
/// satisfy the given constraints.
class DynParametricAttrConstraint : public Constraint {
public:
  DynParametricAttrConstraint(DynamicAttrDefinition *attrDef,
                              SmallVector<unsigned> constraints)
      : attrDef(attrDef), constraints(std::move(constraints)) {}

  virtual ~DynParametricAttrConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  DynamicAttrDefinition *attrDef;
  SmallVector<unsigned> constraints;
};

/// A constraint that checks that a type is of a specific dynamic type
/// definition, and that all of its parameters satisfy the given constraints.
class DynParametricTypeConstraint : public Constraint {
public:
  DynParametricTypeConstraint(DynamicTypeDefinition *typeDef,
                              SmallVector<unsigned> constraints)
      : typeDef(typeDef), constraints(std::move(constraints)) {}

  virtual ~DynParametricTypeConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  DynamicTypeDefinition *typeDef;
  SmallVector<unsigned> constraints;
};

/// A constraint checking that one of the given constraints is satisfied.
class AnyOfConstraint : public Constraint {
public:
  AnyOfConstraint(SmallVector<unsigned> constraints)
      : constraints(std::move(constraints)) {}

  virtual ~AnyOfConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  SmallVector<unsigned> constraints;
};

/// A constraint checking that all of the given constraints are satisfied.
class AllOfConstraint : public Constraint {
public:
  AllOfConstraint(SmallVector<unsigned> constraints)
      : constraints(std::move(constraints)) {}

  virtual ~AllOfConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;

private:
  SmallVector<unsigned> constraints;
};

/// A constraint that is always satisfied.
class AnyAttributeConstraint : public Constraint {
public:
  virtual ~AnyAttributeConstraint() = default;

  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute attr,
                       ConstraintVerifier &context) const override;
};

/// A constraint checking that a region satisfies `irdl.region` requirements
struct RegionConstraint {
  /// The constructor accepts constrained entities from the `irdl.region`
  /// operation, such as slots of constraints for the region's arguments and the
  /// block count.

  // Both entities are optional, which means if an entity is not present, then
  // it is not constrained.
  RegionConstraint(std::optional<SmallVector<unsigned>> argumentConstraints,
                   std::optional<size_t> blockCount)
      : argumentConstraints(std::move(argumentConstraints)),
        blockCount(blockCount) {}

  /// Check that the `region` satisfies the constraint.
  ///
  /// `constraintContext` is needed to verify the region's arguments
  /// constraints.
  LogicalResult verify(mlir::Region &region,
                       ConstraintVerifier &constraintContext);

private:
  std::optional<SmallVector<unsigned>> argumentConstraints;
  std::optional<size_t> blockCount;
};

/// Generate an op verifier function from the given IRDL operation definition.
llvm::unique_function<LogicalResult(Operation *) const> createVerifier(
    OperationOp operation,
    const DenseMap<irdl::TypeOp, std::unique_ptr<DynamicTypeDefinition>>
        &typeDefs,
    const DenseMap<irdl::AttributeOp, std::unique_ptr<DynamicAttrDefinition>>
        &attrDefs);
} // namespace irdl
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_IRDLVERIFIERS_H
