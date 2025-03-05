//===- IRDLVerifiers.cpp - IRDL verifiers ------------------------- C++ -*-===//
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

#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::irdl;

ConstraintVerifier::ConstraintVerifier(
    ArrayRef<std::unique_ptr<Constraint>> constraints)
    : constraints(constraints), assigned() {
  assigned.resize(this->constraints.size());
}

LogicalResult
ConstraintVerifier::verify(function_ref<InFlightDiagnostic()> emitError,
                           Attribute attr, unsigned variable) {

  assert(variable < constraints.size() && "invalid constraint variable");

  // If the variable is already assigned, check that the attribute is the same.
  if (assigned[variable].has_value()) {
    if (attr == assigned[variable].value()) {
      return success();
    }
    if (emitError)
      return emitError() << "expected '" << assigned[variable].value()
                         << "' but got '" << attr << "'";
    return failure();
  }

  // Otherwise, check the constraint and assign the attribute to the variable.
  LogicalResult result = constraints[variable]->verify(emitError, attr, *this);
  if (succeeded(result))
    assigned[variable] = attr;

  return result;
}

LogicalResult IsConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                                   Attribute attr,
                                   ConstraintVerifier &context) const {
  if (attr == expectedAttribute)
    return success();

  if (emitError)
    return emitError() << "expected '" << expectedAttribute << "' but got '"
                       << attr << "'";
  return failure();
}

LogicalResult
BaseAttrConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                           Attribute attr, ConstraintVerifier &context) const {
  if (attr.getTypeID() == baseTypeID)
    return success();

  if (emitError)
    return emitError() << "expected base attribute '" << baseName
                       << "' but got '" << attr.getAbstractAttribute().getName()
                       << "'";
  return failure();
}

LogicalResult
BaseTypeConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                           Attribute attr, ConstraintVerifier &context) const {
  auto typeAttr = dyn_cast<TypeAttr>(attr);
  if (!typeAttr) {
    if (emitError)
      return emitError() << "expected type, got attribute '" << attr;
    return failure();
  }

  Type type = typeAttr.getValue();
  if (type.getTypeID() == baseTypeID)
    return success();

  if (emitError)
    return emitError() << "expected base type '" << baseName << "' but got '"
                       << type.getAbstractType().getName() << "'";
  return failure();
}

LogicalResult DynParametricAttrConstraint::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute attr,
    ConstraintVerifier &context) const {

  // Check that the base is the expected one.
  auto dynAttr = dyn_cast<DynamicAttr>(attr);
  if (!dynAttr || dynAttr.getAttrDef() != attrDef) {
    if (emitError) {
      StringRef dialectName = attrDef->getDialect()->getNamespace();
      StringRef attrName = attrDef->getName();
      return emitError() << "expected base attribute '" << attrName << '.'
                         << dialectName << "' but got '" << attr << "'";
    }
    return failure();
  }

  // Check that the parameters satisfy the constraints.
  ArrayRef<Attribute> params = dynAttr.getParams();
  if (params.size() != constraints.size()) {
    if (emitError) {
      StringRef dialectName = attrDef->getDialect()->getNamespace();
      StringRef attrName = attrDef->getName();
      emitError() << "attribute '" << dialectName << "." << attrName
                  << "' expects " << params.size() << " parameters but got "
                  << constraints.size();
    }
    return failure();
  }

  for (size_t i = 0, s = params.size(); i < s; i++)
    if (failed(context.verify(emitError, params[i], constraints[i])))
      return failure();

  return success();
}

LogicalResult DynParametricTypeConstraint::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute attr,
    ConstraintVerifier &context) const {
  // Check that the base is a TypeAttr.
  auto typeAttr = dyn_cast<TypeAttr>(attr);
  if (!typeAttr) {
    if (emitError)
      return emitError() << "expected type, got attribute '" << attr;
    return failure();
  }

  // Check that the type base is the expected one.
  auto dynType = dyn_cast<DynamicType>(typeAttr.getValue());
  if (!dynType || dynType.getTypeDef() != typeDef) {
    if (emitError) {
      StringRef dialectName = typeDef->getDialect()->getNamespace();
      StringRef attrName = typeDef->getName();
      return emitError() << "expected base type '" << dialectName << '.'
                         << attrName << "' but got '" << attr << "'";
    }
    return failure();
  }

  // Check that the parameters satisfy the constraints.
  ArrayRef<Attribute> params = dynType.getParams();
  if (params.size() != constraints.size()) {
    if (emitError) {
      StringRef dialectName = typeDef->getDialect()->getNamespace();
      StringRef attrName = typeDef->getName();
      emitError() << "attribute '" << dialectName << "." << attrName
                  << "' expects " << params.size() << " parameters but got "
                  << constraints.size();
    }
    return failure();
  }

  for (size_t i = 0, s = params.size(); i < s; i++)
    if (failed(context.verify(emitError, params[i], constraints[i])))
      return failure();

  return success();
}

LogicalResult
AnyOfConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                        Attribute attr, ConstraintVerifier &context) const {
  for (unsigned constr : constraints) {
    // We do not pass the `emitError` here, since we want to emit an error
    // only if none of the constraints are satisfied.
    if (succeeded(context.verify({}, attr, constr))) {
      return success();
    }
  }

  if (emitError)
    return emitError() << "'" << attr << "' does not satisfy the constraint";
  return failure();
}

LogicalResult
AllOfConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                        Attribute attr, ConstraintVerifier &context) const {
  for (unsigned constr : constraints) {
    if (failed(context.verify(emitError, attr, constr))) {
      return failure();
    }
  }

  return success();
}

LogicalResult
AnyAttributeConstraint::verify(function_ref<InFlightDiagnostic()> emitError,
                               Attribute attr,
                               ConstraintVerifier &context) const {
  return success();
}

LogicalResult RegionConstraint::verify(mlir::Region &region,
                                       ConstraintVerifier &constraintContext) {
  const auto emitError = [parentOp = region.getParentOp()](mlir::Location loc) {
    return [loc, parentOp] {
      InFlightDiagnostic diag = mlir::emitError(loc);
      // If we already have been given location of the parent operation, which
      // might happen when the region location is passed, we do not want to
      // produce the note on the same location
      if (loc != parentOp->getLoc())
        diag.attachNote(parentOp->getLoc()).append("see the operation");
      return diag;
    };
  };

  if (blockCount.has_value() && *blockCount != region.getBlocks().size()) {
    return emitError(region.getLoc())()
           << "expected region " << region.getRegionNumber() << " to have "
           << *blockCount << " block(s) but got " << region.getBlocks().size();
  }

  if (argumentConstraints.has_value()) {
    auto actualArgs = region.getArguments();
    if (actualArgs.size() != argumentConstraints->size()) {
      const mlir::Location firstArgLoc =
          actualArgs.empty() ? region.getLoc() : actualArgs.front().getLoc();
      return emitError(firstArgLoc)()
             << "expected region " << region.getRegionNumber() << " to have "
             << argumentConstraints->size() << " arguments but got "
             << actualArgs.size();
    }

    for (auto [arg, constraint] : llvm::zip(actualArgs, *argumentConstraints)) {
      mlir::Attribute type = TypeAttr::get(arg.getType());
      if (failed(constraintContext.verify(emitError(arg.getLoc()), type,
                                          constraint))) {
        return failure();
      }
    }
  }
  return success();
}
