//===- IRDLLoading.cpp - IRDL dialect loading --------------------- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the loading of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace mlir::irdl;

/// Verify that the given list of parameters satisfy the given constraints.
/// This encodes the logic of the verification method for attributes and types
/// defined with IRDL.
static LogicalResult
irdlAttrOrTypeVerifier(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> params,
                       ArrayRef<std::unique_ptr<Constraint>> constraints,
                       ArrayRef<size_t> paramConstraints) {
  if (params.size() != paramConstraints.size()) {
    emitError() << "expected " << paramConstraints.size()
                << " type arguments, but had " << params.size();
    return failure();
  }

  ConstraintVerifier verifier(constraints);

  // Check that each parameter satisfies its constraint.
  for (auto [i, param] : enumerate(params))
    if (failed(verifier.verify(emitError, param, paramConstraints[i])))
      return failure();

  return success();
}

/// Verify that the given operation satisfies the given constraints.
/// This encodes the logic of the verification method for operations defined
/// with IRDL.
static LogicalResult
irdlOpVerifier(Operation *op, ArrayRef<std::unique_ptr<Constraint>> constraints,
               ArrayRef<size_t> operandConstrs, ArrayRef<size_t> resultConstrs,
               const DenseMap<StringAttr, size_t> &attributeConstrs) {
  /// Check that we have the right number of operands.
  unsigned numOperands = op->getNumOperands();
  size_t numExpectedOperands = operandConstrs.size();
  if (numOperands != numExpectedOperands)
    return op->emitOpError() << numExpectedOperands
                             << " operands expected, but got " << numOperands;

  /// Check that we have the right number of results.
  unsigned numResults = op->getNumResults();
  size_t numExpectedResults = resultConstrs.size();
  if (numResults != numExpectedResults)
    return op->emitOpError()
           << numExpectedResults << " results expected, but got " << numResults;

  auto emitError = [op] { return op->emitError(); };

  ConstraintVerifier verifier(constraints);

  /// Сheck that we have all needed attributes passed
  /// and they satisfy the constraints.
  DictionaryAttr actualAttrs = op->getAttrDictionary();

  for (auto [name, constraint] : attributeConstrs) {
    /// First, check if the attribute actually passed.
    std::optional<NamedAttribute> actual = actualAttrs.getNamed(name);
    if (!actual.has_value())
      return op->emitOpError()
             << "attribute " << name << " is expected but not provided";

    /// Then, check if the attribute value satisfies the constraint.
    if (failed(verifier.verify({emitError}, actual->getValue(), constraint)))
      return failure();
  }

  /// Check that all operands satisfy the constraints.
  for (auto [i, operandType] : enumerate(op->getOperandTypes()))
    if (failed(verifier.verify({emitError}, TypeAttr::get(operandType),
                               operandConstrs[i])))
      return failure();

  /// Check that all results satisfy the constraints.
  for (auto [i, resultType] : enumerate(op->getResultTypes()))
    if (failed(verifier.verify({emitError}, TypeAttr::get(resultType),
                               resultConstrs[i])))
      return failure();

  return success();
}

/// Define and load an operation represented by a `irdl.operation`
/// operation.
static WalkResult loadOperation(
    OperationOp op, ExtensibleDialect *dialect,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> &attrs) {
  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (Operation &op : op->getRegion(0).getOps()) {
    if (isa<VerifyConstraintInterface>(op)) {
      if (op.getNumResults() != 1)
        return op.emitError()
               << "IRDL constraint operations must have exactly one result";
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<Constraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        cast<VerifyConstraintInterface>(v.getDefiningOp());
    std::unique_ptr<Constraint> verifier =
        op.getVerifier(constrToValue, types, attrs);
    if (!verifier)
      return WalkResult::interrupt();
    constraints.push_back(std::move(verifier));
  }

  SmallVector<size_t> operandConstraints;
  SmallVector<size_t> resultConstraints;

  // Gather which constraint slots correspond to operand constraints
  auto operandsOp = op.getOp<OperandsOp>();
  if (operandsOp.has_value()) {
    operandConstraints.reserve(operandsOp->getArgs().size());
    for (Value operand : operandsOp->getArgs()) {
      for (auto [i, constr] : enumerate(constrToValue)) {
        if (constr == operand) {
          operandConstraints.push_back(i);
          break;
        }
      }
    }
  }

  // Gather which constraint slots correspond to result constraints
  auto resultsOp = op.getOp<ResultsOp>();
  if (resultsOp.has_value()) {
    resultConstraints.reserve(resultsOp->getArgs().size());
    for (Value result : resultsOp->getArgs()) {
      for (auto [i, constr] : enumerate(constrToValue)) {
        if (constr == result) {
          resultConstraints.push_back(i);
          break;
        }
      }
    }
  }

  // Gather which constraint slots correspond to attributes constraints
  DenseMap<StringAttr, size_t> attributesContraints;
  auto attributesOp = op.getOp<AttributesOp>();
  if (attributesOp.has_value()) {
    const Operation::operand_range values = attributesOp->getAttributeValues();
    const ArrayAttr names = attributesOp->getAttributeValueNames();

    for (const auto &[name, value] : llvm::zip(names, values)) {
      for (auto [i, constr] : enumerate(constrToValue)) {
        if (constr == value) {
          attributesContraints[name.cast<StringAttr>()] = i;
          break;
        }
      }
    }
  }

  // IRDL does not support defining custom parsers or printers.
  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  auto verifier =
      [constraints{std::move(constraints)},
       operandConstraints{std::move(operandConstraints)},
       resultConstraints{std::move(resultConstraints)},
       attributesContraints{std::move(attributesContraints)}](Operation *op) {
        return irdlOpVerifier(op, constraints, operandConstraints,
                              resultConstraints, attributesContraints);
      };

  // IRDL does not support defining regions.
  auto regionVerifier = [](Operation *op) { return success(); };

  auto opDef = DynamicOpDefinition::get(
      op.getName(), dialect, std::move(verifier), std::move(regionVerifier),
      std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(opDef));

  return WalkResult::advance();
}

/// Get the verifier of a type or attribute definition.
/// Return nullptr if the definition is invalid.
static DynamicAttrDefinition::VerifierFn getAttrOrTypeVerifier(
    Operation *attrOrTypeDef, ExtensibleDialect *dialect,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> &attrs) {
  assert((isa<AttributeOp>(attrOrTypeDef) || isa<TypeOp>(attrOrTypeDef)) &&
         "Expected an attribute or type definition");

  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (Operation &op : attrOrTypeDef->getRegion(0).getOps()) {
    if (isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1 &&
             "IRDL constraint operations must have exactly one result");
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<Constraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        cast<VerifyConstraintInterface>(v.getDefiningOp());
    std::unique_ptr<Constraint> verifier =
        op.getVerifier(constrToValue, types, attrs);
    if (!verifier)
      return {};
    constraints.push_back(std::move(verifier));
  }

  // Get the parameter definitions.
  std::optional<ParametersOp> params;
  if (auto attr = dyn_cast<AttributeOp>(attrOrTypeDef))
    params = attr.getOp<ParametersOp>();
  else if (auto type = dyn_cast<TypeOp>(attrOrTypeDef))
    params = type.getOp<ParametersOp>();

  // Gather which constraint slots correspond to parameter constraints
  SmallVector<size_t> paramConstraints;
  if (params.has_value()) {
    paramConstraints.reserve(params->getArgs().size());
    for (Value param : params->getArgs()) {
      for (auto [i, constr] : enumerate(constrToValue)) {
        if (constr == param) {
          paramConstraints.push_back(i);
          break;
        }
      }
    }
  }

  auto verifier = [paramConstraints{std::move(paramConstraints)},
                   constraints{std::move(constraints)}](
                      function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Attribute> params) {
    return irdlAttrOrTypeVerifier(emitError, params, constraints,
                                  paramConstraints);
  };

  // While the `std::move` is not required, not adding it triggers a bug in
  // clang-10.
  return std::move(verifier);
}

/// Get the possible bases of a constraint. Return `true` if all bases can
/// potentially be matched.
/// A base is a type or an attribute definition. For instance, the base of
/// `irdl.parametric "!builtin.complex"(...)` is `builtin.complex`.
/// This function returns the following information through arguments:
/// - `paramIds`: the set of type or attribute IDs that are used as bases.
/// - `paramIrdlOps`: the set of IRDL operations that are used as bases.
/// - `isIds`: the set of type or attribute IDs that are used in `irdl.is`
///   constraints.
static bool getBases(Operation *op, SmallPtrSet<TypeID, 4> &paramIds,
                     SmallPtrSet<Operation *, 4> &paramIrdlOps,
                     SmallPtrSet<TypeID, 4> &isIds) {
  // For `irdl.any_of`, we get the bases from all its arguments.
  if (auto anyOf = dyn_cast<AnyOfOp>(op)) {
    bool has_any = false;
    for (Value arg : anyOf.getArgs())
      has_any &= getBases(arg.getDefiningOp(), paramIds, paramIrdlOps, isIds);
    return has_any;
  }

  // For `irdl.all_of`, we get the bases from the first argument.
  // This is restrictive, but we can relax it later if needed.
  if (auto allOf = dyn_cast<AllOfOp>(op))
    return getBases(allOf.getArgs()[0].getDefiningOp(), paramIds, paramIrdlOps,
                    isIds);

  // For `irdl.parametric`, we get directly the base from the operation.
  if (auto params = dyn_cast<ParametricOp>(op)) {
    SymbolRefAttr symRef = params.getBaseType();
    Operation *defOp = SymbolTable::lookupNearestSymbolFrom(op, symRef);
    assert(defOp && "symbol reference should refer to an existing operation");
    paramIrdlOps.insert(defOp);
    return false;
  }

  // For `irdl.is`, we get the base TypeID directly.
  if (auto is = dyn_cast<IsOp>(op)) {
    Attribute expected = is.getExpected();
    isIds.insert(expected.getTypeID());
    return false;
  }

  // For `irdl.any`, we return `false` since we can match any type or attribute
  // base.
  if (auto isA = dyn_cast<AnyOp>(op))
    return true;

  llvm_unreachable("unknown IRDL constraint");
}

/// Check that an any_of is in the subset IRDL can handle.
/// IRDL uses a greedy algorithm to match constraints. This means that if we
/// encounter an `any_of` with multiple constraints, we will match the first
/// constraint that is satisfied. Thus, the order of constraints matter in
/// `any_of` with our current algorithm.
/// In order to make the order of constraints irrelevant, we require that
/// all `any_of` constraint parameters are disjoint. For this, we check that
/// the base parameters are all disjoints between `parametric` operations, and
/// that they are disjoint between `parametric` and `is` operations.
/// This restriction will be relaxed in the future, when we will change our
/// algorithm to be non-greedy.
static LogicalResult checkCorrectAnyOf(AnyOfOp anyOf) {
  SmallPtrSet<TypeID, 4> paramIds;
  SmallPtrSet<Operation *, 4> paramIrdlOps;
  SmallPtrSet<TypeID, 4> isIds;

  for (Value arg : anyOf.getArgs()) {
    Operation *argOp = arg.getDefiningOp();
    SmallPtrSet<TypeID, 4> argParamIds;
    SmallPtrSet<Operation *, 4> argParamIrdlOps;
    SmallPtrSet<TypeID, 4> argIsIds;

    // Get the bases of this argument. If it can match any type or attribute,
    // then our `any_of` should not be allowed.
    if (getBases(argOp, argParamIds, argParamIrdlOps, argIsIds))
      return failure();

    // We check that the base parameters are all disjoints between `parametric`
    // operations, and that they are disjoint between `parametric` and `is`
    // operations.
    for (TypeID id : argParamIds) {
      if (isIds.count(id))
        return failure();
      bool inserted = paramIds.insert(id).second;
      if (!inserted)
        return failure();
    }

    // We check that the base parameters are all disjoints with `irdl.is`
    // operations.
    for (TypeID id : isIds) {
      if (paramIds.count(id))
        return failure();
      isIds.insert(id);
    }

    // We check that all `parametric` operations are disjoint. We do not
    // need to check that they are disjoint with `is` operations, since
    // `is` operations cannot refer to attributes defined with `irdl.parametric`
    // operations.
    for (Operation *op : argParamIrdlOps) {
      bool inserted = paramIrdlOps.insert(op).second;
      if (!inserted)
        return failure();
    }
  }

  return success();
}

/// Load all dialects in the given module, without loading any operation, type
/// or attribute definitions.
static DenseMap<DialectOp, ExtensibleDialect *> loadEmptyDialects(ModuleOp op) {
  DenseMap<DialectOp, ExtensibleDialect *> dialects;
  op.walk([&](DialectOp dialectOp) {
    MLIRContext *ctx = dialectOp.getContext();
    StringRef dialectName = dialectOp.getName();

    DynamicDialect *dialect = ctx->getOrLoadDynamicDialect(
        dialectName, [](DynamicDialect *dialect) {});

    dialects.insert({dialectOp, dialect});
  });
  return dialects;
}

/// Preallocate type definitions objects with empty verifiers.
/// This in particular allocates a TypeID for each type definition.
static DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>>
preallocateTypeDefs(ModuleOp op,
                    DenseMap<DialectOp, ExtensibleDialect *> dialects) {
  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> typeDefs;
  op.walk([&](TypeOp typeOp) {
    ExtensibleDialect *dialect = dialects[typeOp.getParentOp()];
    auto typeDef = DynamicTypeDefinition::get(
        typeOp.getName(), dialect,
        [](function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) {
          return success();
        });
    typeDefs.try_emplace(typeOp, std::move(typeDef));
  });
  return typeDefs;
}

/// Preallocate attribute definitions objects with empty verifiers.
/// This in particular allocates a TypeID for each attribute definition.
static DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>>
preallocateAttrDefs(ModuleOp op,
                    DenseMap<DialectOp, ExtensibleDialect *> dialects) {
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrDefs;
  op.walk([&](AttributeOp attrOp) {
    ExtensibleDialect *dialect = dialects[attrOp.getParentOp()];
    auto attrDef = DynamicAttrDefinition::get(
        attrOp.getName(), dialect,
        [](function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) {
          return success();
        });
    attrDefs.try_emplace(attrOp, std::move(attrDef));
  });
  return attrDefs;
}

LogicalResult mlir::irdl::loadDialects(ModuleOp op) {
  // First, check that all any_of constraints are in a correct form.
  // This is to ensure we can do the verification correctly.
  WalkResult anyOfCorrects = op.walk(
      [](AnyOfOp anyOf) { return (WalkResult)checkCorrectAnyOf(anyOf); });
  if (anyOfCorrects.wasInterrupted())
    return op.emitError("any_of constraints are not in the correct form");

  // Preallocate all dialects, and type and attribute definitions.
  // In particular, this allocates TypeIDs so type and attributes can have
  // verifiers that refer to each other.
  DenseMap<DialectOp, ExtensibleDialect *> dialects = loadEmptyDialects(op);
  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> types =
      preallocateTypeDefs(op, dialects);
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrs =
      preallocateAttrDefs(op, dialects);

  // Set the verifier for types.
  WalkResult res = op.walk([&](TypeOp typeOp) {
    DynamicAttrDefinition::VerifierFn verifier = getAttrOrTypeVerifier(
        typeOp, dialects[typeOp.getParentOp()], types, attrs);
    if (!verifier)
      return WalkResult::interrupt();
    types[typeOp]->setVerifyFn(std::move(verifier));
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return failure();

  // Set the verifier for attributes.
  res = op.walk([&](AttributeOp attrOp) {
    DynamicAttrDefinition::VerifierFn verifier = getAttrOrTypeVerifier(
        attrOp, dialects[attrOp.getParentOp()], types, attrs);
    if (!verifier)
      return WalkResult::interrupt();
    attrs[attrOp]->setVerifyFn(std::move(verifier));
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return failure();

  // Define and load all operations.
  res = op.walk([&](OperationOp opOp) {
    return loadOperation(opOp, dialects[opOp.getParentOp()], types, attrs);
  });
  if (res.wasInterrupted())
    return failure();

  // Load all types in their dialects.
  for (auto &pair : types) {
    ExtensibleDialect *dialect = dialects[pair.first.getParentOp()];
    dialect->registerDynamicType(std::move(pair.second));
  }

  // Load all attributes in their dialects.
  for (auto &pair : attrs) {
    ExtensibleDialect *dialect = dialects[pair.first.getParentOp()];
    dialect->registerDynamicAttr(std::move(pair.second));
  }

  return success();
}
