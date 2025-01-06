//===- CodeGenHelpers.cpp - MLIR op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate C++
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pattern.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// Generate a unique label based on the current file name to prevent name
/// collisions if multiple generated files are included at once.
static std::string getUniqueOutputLabel(const RecordKeeper &records,
                                        StringRef tag) {
  // Use the input file name when generating a unique name.
  std::string inputFilename = records.getInputFilename();

  // Drop all but the base filename.
  StringRef nameRef = sys::path::filename(inputFilename);
  nameRef.consume_back(".td");

  // Sanitize any invalid characters.
  std::string uniqueName(tag);
  for (char c : nameRef) {
    if (isAlnum(c) || c == '_')
      uniqueName.push_back(c);
    else
      uniqueName.append(utohexstr((unsigned char)c));
  }
  return uniqueName;
}

StaticVerifierFunctionEmitter::StaticVerifierFunctionEmitter(
    raw_ostream &os, const RecordKeeper &records, StringRef tag)
    : os(os), uniqueOutputLabel(getUniqueOutputLabel(records, tag)) {}

void StaticVerifierFunctionEmitter::emitOpConstraints(
    ArrayRef<const Record *> opDefs) {
  NamespaceEmitter namespaceEmitter(os, Operator(*opDefs[0]).getCppNamespace());
  emitTypeConstraints();
  emitAttrConstraints();
  emitSuccessorConstraints();
  emitRegionConstraints();
}

void StaticVerifierFunctionEmitter::emitPatternConstraints(
    const ArrayRef<DagLeaf> constraints) {
  collectPatternConstraints(constraints);
  emitPatternConstraints();
}

//===----------------------------------------------------------------------===//
// Constraint Getters

StringRef StaticVerifierFunctionEmitter::getTypeConstraintFn(
    const Constraint &constraint) const {
  const auto *it = typeConstraints.find(constraint);
  assert(it != typeConstraints.end() && "expected to find a type constraint");
  return it->second;
}

// Find a uniqued attribute constraint. Since not all attribute constraints can
// be uniqued, return std::nullopt if one was not found.
std::optional<StringRef> StaticVerifierFunctionEmitter::getAttrConstraintFn(
    const Constraint &constraint) const {
  const auto *it = attrConstraints.find(constraint);
  return it == attrConstraints.end() ? std::optional<StringRef>()
                                     : StringRef(it->second);
}

StringRef StaticVerifierFunctionEmitter::getSuccessorConstraintFn(
    const Constraint &constraint) const {
  const auto *it = successorConstraints.find(constraint);
  assert(it != successorConstraints.end() &&
         "expected to find a sucessor constraint");
  return it->second;
}

StringRef StaticVerifierFunctionEmitter::getRegionConstraintFn(
    const Constraint &constraint) const {
  const auto *it = regionConstraints.find(constraint);
  assert(it != regionConstraints.end() &&
         "expected to find a region constraint");
  return it->second;
}

//===----------------------------------------------------------------------===//
// Constraint Emission

/// Code templates for emitting type, attribute, successor, and region
/// constraints. Each of these templates require the following arguments:
///
/// {0}: The unique constraint name.
/// {1}: The constraint code.
/// {2}: The constraint description.

/// Code for a type constraint. These may be called on the type of either
/// operands or results.
static const char *const typeConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!({1})) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be {2}, but got " << type;
  }
  return ::mlir::success();
}
)";

/// Code for an attribute constraint. These may be called from ops only.
/// Attribute constraints cannot reference anything other than `$_self` and
/// `$_op`.
///
/// TODO: Unique constraints for adaptors. However, most Adaptor::verify
/// functions are stripped anyways.
static const char *const attrConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {{
  if (attr && !({1}))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: {2}";
  return ::mlir::success();
}
static ::llvm::LogicalResult {0}(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {{
  return {0}(attr, attrName, [op]() {{
    return op->emitOpError();
  });
}
)";

/// Code for a successor constraint.
static const char *const successorConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::Operation *op, ::mlir::Block *successor,
    ::llvm::StringRef successorName, unsigned successorIndex) {
  if (!({1})) {
    return op->emitOpError("successor #") << successorIndex << " ('"
        << successorName << ")' failed to verify constraint: {2}";
  }
  return ::mlir::success();
}
)";

/// Code for a region constraint. Callers will need to pass in the region's name
/// for emitting an error message.
static const char *const regionConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!({1})) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: {2}";
  }
  return ::mlir::success();
}
)";

/// Code for a pattern type or attribute constraint.
///
/// {3}: "Type type" or "Attribute attr".
static const char *const patternAttrOrTypeConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::{3},
    ::llvm::StringRef failureStr) {
  if (!({1})) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": {2}";
    });
  }
  return ::mlir::success();
}
)";

void StaticVerifierFunctionEmitter::emitConstraints(
    const ConstraintMap &constraints, StringRef selfName,
    const char *const codeTemplate) {
  FmtContext ctx;
  ctx.addSubst("_op", "*op").withSelf(selfName);
  for (auto &it : constraints) {
    os << formatv(codeTemplate, it.second,
                  tgfmt(it.first.getConditionTemplate(), &ctx),
                  escapeString(it.first.getSummary()));
  }
}

void StaticVerifierFunctionEmitter::emitTypeConstraints() {
  emitConstraints(typeConstraints, "type", typeConstraintCode);
}

void StaticVerifierFunctionEmitter::emitAttrConstraints() {
  emitConstraints(attrConstraints, "attr", attrConstraintCode);
}

void StaticVerifierFunctionEmitter::emitSuccessorConstraints() {
  emitConstraints(successorConstraints, "successor", successorConstraintCode);
}

void StaticVerifierFunctionEmitter::emitRegionConstraints() {
  emitConstraints(regionConstraints, "region", regionConstraintCode);
}

void StaticVerifierFunctionEmitter::emitPatternConstraints() {
  FmtContext ctx;
  ctx.addSubst("_op", "*op").withBuilder("rewriter").withSelf("type");
  for (auto &it : typeConstraints) {
    os << formatv(patternAttrOrTypeConstraintCode, it.second,
                  tgfmt(it.first.getConditionTemplate(), &ctx),
                  escapeString(it.first.getSummary()), "Type type");
  }
  ctx.withSelf("attr");
  for (auto &it : attrConstraints) {
    os << formatv(patternAttrOrTypeConstraintCode, it.second,
                  tgfmt(it.first.getConditionTemplate(), &ctx),
                  escapeString(it.first.getSummary()), "Attribute attr");
  }
}

//===----------------------------------------------------------------------===//
// Constraint Uniquing

/// An attribute constraint that references anything other than itself and the
/// current op cannot be generically extracted into a function. Most
/// prohibitive are operands and results, which require calls to
/// `getODSOperands` or `getODSResults`. Attribute references are tricky too
/// because ops use cached identifiers.
static bool canUniqueAttrConstraint(Attribute attr) {
  FmtContext ctx;
  auto test = tgfmt(attr.getConditionTemplate(),
                    &ctx.withSelf("attr").addSubst("_op", "*op"))
                  .str();
  return !StringRef(test).contains("<no-subst-found>");
}

std::string StaticVerifierFunctionEmitter::getUniqueName(StringRef kind,
                                                         unsigned index) {
  return ("__mlir_ods_local_" + kind + "_constraint_" + uniqueOutputLabel +
          Twine(index))
      .str();
}

void StaticVerifierFunctionEmitter::collectConstraint(ConstraintMap &map,
                                                      StringRef kind,
                                                      Constraint constraint) {
  auto [it, inserted] = map.try_emplace(constraint);
  if (inserted)
    it->second = getUniqueName(kind, map.size());
}

void StaticVerifierFunctionEmitter::collectOpConstraints(
    ArrayRef<const Record *> opDefs) {
  const auto collectTypeConstraints = [&](Operator::const_value_range values) {
    for (const NamedTypeConstraint &value : values)
      if (value.hasPredicate())
        collectConstraint(typeConstraints, "type", value.constraint);
  };

  for (const Record *def : opDefs) {
    Operator op(*def);
    /// Collect type constraints.
    collectTypeConstraints(op.getOperands());
    collectTypeConstraints(op.getResults());
    /// Collect attribute constraints.
    for (const NamedAttribute &namedAttr : op.getAttributes()) {
      if (!namedAttr.attr.getPredicate().isNull() &&
          !namedAttr.attr.isDerivedAttr() &&
          canUniqueAttrConstraint(namedAttr.attr))
        collectConstraint(attrConstraints, "attr", namedAttr.attr);
    }
    /// Collect successor constraints.
    for (const NamedSuccessor &successor : op.getSuccessors()) {
      if (!successor.constraint.getPredicate().isNull()) {
        collectConstraint(successorConstraints, "successor",
                          successor.constraint);
      }
    }
    /// Collect region constraints.
    for (const NamedRegion &region : op.getRegions())
      if (!region.constraint.getPredicate().isNull())
        collectConstraint(regionConstraints, "region", region.constraint);
  }
}

void StaticVerifierFunctionEmitter::collectPatternConstraints(
    const ArrayRef<DagLeaf> constraints) {
  for (auto &leaf : constraints) {
    assert(leaf.isOperandMatcher() || leaf.isAttrMatcher());
    collectConstraint(
        leaf.isOperandMatcher() ? typeConstraints : attrConstraints,
        leaf.isOperandMatcher() ? "type" : "attr", leaf.getAsConstraint());
  }
}

//===----------------------------------------------------------------------===//
// Public Utility Functions
//===----------------------------------------------------------------------===//

std::string mlir::tblgen::escapeString(StringRef value) {
  std::string ret;
  raw_string_ostream os(ret);
  os.write_escaped(value);
  return ret;
}
