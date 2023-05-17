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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Support/LogicalResult.h"

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
    } else {
      if (emitError)
        return emitError() << "expected '" << assigned[variable].value()
                           << "' but got '" << attr << "'";
      return failure();
    }
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
