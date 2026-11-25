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
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pattern.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/Successor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <cassert>
#include <optional>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

void mlir::tblgen::reportFatalFmtError(llvm::Error error,
                                       const FmtOrigin &callerOrigin) {
  if (!error)
    PrintFatalError("internal formatting error: expected a non-success error");
  SmallVector<FmtDiagnostic> messages;
  llvm::handleAllErrors(
      addFmtOrigin(std::move(error), callerOrigin),
      [&](const FmtError &failure) {
        llvm::append_range(messages, getFmtDiagnostics(failure));
      },
      [&](const llvm::ErrorInfoBase &failure) {
        const FmtSite &site = callerOrigin.getDefinition();
        messages.push_back({true, site.locations, failure.message()});
      });
  if (messages.empty())
    PrintFatalError("internal formatting error: no diagnostic messages");
  // Use location-free overloads for explanatory messages, avoiding an
  // artificial <unknown>:0 prefix on template and availability notes.
  for (const FmtDiagnostic &message : llvm::drop_end(messages)) {
    if (message.isError) {
      if (message.locations.empty())
        PrintError(message.message);
      else
        PrintError(message.locations, message.message);
    } else {
      if (message.locations.empty())
        PrintNote(message.message);
      else
        PrintNote(message.locations, message.message);
    }
  }
  const FmtDiagnostic &last = messages.back();
  if (last.isError) {
    if (last.locations.empty())
      PrintFatalError(last.message);
    PrintFatalError(last.locations, last.message);
  }
  if (last.locations.empty())
    PrintFatalNote(last.message);
  PrintFatalNote(last.locations, last.message);
}

/// Generate a unique label based on the current file name to prevent name
/// collisions if multiple generated files are included at once.
static std::string getUniqueOutputLabel(const RecordKeeper &records,
                                        StringRef tag) {
  // Use the input file name when generating a unique name.
  StringRef inputFilename = records.getInputFilename();

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

void StaticVerifierFunctionEmitter::emitOpConstraints() {
  emitTypeConstraints();
  emitAttrConstraints();
  emitPropConstraints();
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
//===----------------------------------------------------------------------===//

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

// Find a uniqued property constraint. Since not all property constraints can
// be uniqued, return std::nullopt if one was not found.
std::optional<StringRef> StaticVerifierFunctionEmitter::getPropConstraintFn(
    const Constraint &constraint) const {
  const auto *it = propConstraints.find(constraint);
  return it == propConstraints.end() ? std::optional<StringRef>()
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
//===----------------------------------------------------------------------===//

/// Point at the record that supplies the template and retain its operation use.
FmtOrigin mlir::tblgen::getConstraintFmtOrigin(const Constraint &constraint,
                                               const Operator &op,
                                               StringRef purpose) {
  return FmtOrigin::at(constraint.getDef().getLoc(), purpose)
      .withUse(op.getLoc(), "operation '" + op.getOperationName() + "'");
}

// Builtin segment-size properties are synthesized without a TableGen record.
FmtOrigin mlir::tblgen::getConstraintFmtOrigin(const Property &property,
                                               const Operator &op,
                                               StringRef purpose) {
  if (!property.hasDef())
    return FmtOrigin::at(op.getLoc(), purpose);
  return getConstraintFmtOrigin(static_cast<const Constraint &>(property), op,
                                purpose);
}

// The legacy summary helper retains marker behavior during migration.
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH
/// Helper to generate a C++ string expression from a given message.
/// Message can contain '{{...}}' placeholders that are substituted with
/// C-expressions via tgfmt.
static llvm::Expected<std::string>
buildErrorStreamingStringImpl(StringRef message, const FmtContext &ctx,
                              ErrorStreamType errorStreamType, bool checked) {
  std::string result;
  raw_string_ostream os(result);

  std::string msgStr = escapeString(message);
  StringRef msg = msgStr;

  // Split the message by '{{' and '}}' and build a streaming expression.
  auto split = msg.split("{{");
  os << split.first;
  if (split.second.empty()) {
    return msgStr;
  }

  if (errorStreamType == ErrorStreamType::InsideOpError)
    os << "\")";
  else
    os << '"';

  msg = split.second;
  while (!msg.empty()) {
    split = msg.split("}}");
    StringRef var = split.first;
    StringRef rest = split.second;

    auto expression =
        checked
            ? tgfmtChecked(var, ctx)
            : llvm::Expected<std::string>(FmtStrVecObject(var, &ctx, {}).str());
    if (!expression)
      return addFmtOrigin(expression.takeError(),
                          FmtOrigin::describe("escaped summary expression"));
    os << " << " << *expression;

    if (errorStreamType == ErrorStreamType::InsideOpError) {
      // To enable having part of string post, this adds a parenthesis before
      // the last string segment to match the existing one.
      os << " << (\"";
    } else {
      os << " << \"";
    }

    split = rest.split("{{");
    os << split.first;
    msg = split.second;
  }

  return os.str();
}

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP

std::string mlir::tblgen::buildErrorStreamingString(
    StringRef message, const FmtContext &ctx, ErrorStreamType errorStreamType) {
  return llvm::cantFail(
      buildErrorStreamingStringImpl(message, ctx, errorStreamType, false));
}

llvm::Expected<std::string> mlir::tblgen::buildErrorStreamingStringChecked(
    StringRef message, const FmtContext &ctx, ErrorStreamType errorStreamType) {
  return buildErrorStreamingStringImpl(message, ctx, errorStreamType, true);
}

std::string mlir::tblgen::buildErrorStreamingStringOrFatal(
    const FmtOrigin &origin, StringRef message, const FmtContext &ctx,
    ErrorStreamType errorStreamType) {
  auto result = buildErrorStreamingStringChecked(message, ctx, errorStreamType);
  if (!result)
    reportFatalFmtError(result.takeError(), origin);
  return std::move(*result);
}

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

/// Code for a property constraint. These may be called from ops only.
/// Property constraints cannot reference anything other than `$_self` and
/// `$_op`. {3} is the interface type of the property.
static const char *const propConstraintCode = R"(
  static ::llvm::LogicalResult {0}(
      {3} prop, ::llvm::StringRef propName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {{
    if (!({1}))
      return emitError() << "property '" << propName
          << "' failed to satisfy constraint: {2}";
    return ::mlir::success();
  }
  static ::llvm::LogicalResult {0}(
      ::mlir::Operation *op, {3} prop, ::llvm::StringRef propName) {{
    return {0}(prop, propName, [op]() {{
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
        << successorName << "') failed to verify constraint: {2}";
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
/// {0}: name of function
/// {1}: Condition template
/// {2}: Constraint summary
/// {3}: "::mlir::Type type" or "::mlirAttribute attr" or "propType prop".
/// Can be "T prop" for generic property constraints.
static const char *const patternConstraintCode = R"(
static ::llvm::LogicalResult {0}(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, {3},
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
    const char *const codeTemplate, ErrorStreamType errorStreamType) {
  FmtContext ctx;
  ctx.addSubst("_op", "*op").withSelf(selfName);

  for (auto &it : constraints) {
    FmtOrigin origin = FmtOrigin::at(it.first.getDef().getLoc(),
                                     "shared " + selfName + " constraint");
    os << formatv(codeTemplate, it.second,
                  tgfmtOrFatal(origin, it.first.getConditionTemplate(), ctx),
                  buildErrorStreamingStringOrFatal(
                      origin, it.first.getSummary(), ctx, errorStreamType));
  }
}
void StaticVerifierFunctionEmitter::emitTypeConstraints() {
  emitConstraints(typeConstraints, "type", typeConstraintCode,
                  ErrorStreamType::InString);
}

void StaticVerifierFunctionEmitter::emitAttrConstraints() {
  emitConstraints(attrConstraints, "attr", attrConstraintCode,
                  ErrorStreamType::InString);
}

/// Unlike with the other helpers, this one has to substitute in the interface
/// type of the property, so we can't just use the generic function.
void StaticVerifierFunctionEmitter::emitPropConstraints() {
  FmtContext ctx;
  ctx.addSubst("_op", "*op").withSelf("prop");
  for (auto &it : propConstraints) {
    FmtOrigin origin =
        FmtOrigin::at(it.first.getDef().getLoc(), "shared property constraint");
    auto propConstraint = cast<PropConstraint>(it.first);
    os << formatv(
        propConstraintCode, it.second,
        tgfmtOrFatal(origin, propConstraint.getConditionTemplate(), ctx),
        buildErrorStreamingStringOrFatal(origin, it.first.getSummary(), ctx),
        propConstraint.getInterfaceType());
  }
}

void StaticVerifierFunctionEmitter::emitSuccessorConstraints() {
  emitConstraints(successorConstraints, "successor", successorConstraintCode,
                  ErrorStreamType::InString);
}

void StaticVerifierFunctionEmitter::emitRegionConstraints() {
  emitConstraints(regionConstraints, "region", regionConstraintCode,
                  ErrorStreamType::InString);
}

void StaticVerifierFunctionEmitter::emitPatternConstraints() {
  FmtContext ctx;
  ctx.addSubst("_op", "*op").withBuilder("rewriter").withSelf("type");
  for (auto &it : typeConstraints) {
    FmtOrigin origin = FmtOrigin::at(it.first.getDef().getLoc(),
                                     "shared pattern type constraint");
    os << formatv(
        patternConstraintCode, it.second,
        tgfmtOrFatal(origin, it.first.getConditionTemplate(), ctx),
        buildErrorStreamingStringOrFatal(origin, it.first.getSummary(), ctx),
        "::mlir::Type type");
  }
  ctx.withSelf("attr");
  for (auto &it : attrConstraints) {
    FmtOrigin origin = FmtOrigin::at(it.first.getDef().getLoc(),
                                     "shared pattern attribute constraint");
    os << formatv(
        patternConstraintCode, it.second,
        tgfmtOrFatal(origin, it.first.getConditionTemplate(), ctx),
        buildErrorStreamingStringOrFatal(origin, it.first.getSummary(), ctx),
        "::mlir::Attribute attr");
  }
  ctx.withSelf("prop");
  for (auto &it : propConstraints) {
    FmtOrigin origin = FmtOrigin::at(it.first.getDef().getLoc(),
                                     "shared pattern property constraint");
    PropConstraint propConstraint = cast<PropConstraint>(it.first);
    StringRef interfaceType = propConstraint.getInterfaceType();
    // Constraints that are generic over multiple interface types are
    // templatized under the assumption that they'll be used correctly.
    if (interfaceType.empty()) {
      interfaceType = "T";
      os << "template <typename T>";
    }
    os << formatv(
        patternConstraintCode, it.second,
        tgfmtOrFatal(origin, propConstraint.getConditionTemplate(), ctx),
        buildErrorStreamingStringOrFatal(origin, propConstraint.getSummary(),
                                         ctx),
        Twine(interfaceType) + " prop");
  }
}

//===----------------------------------------------------------------------===//
// Constraint Uniquing
//===----------------------------------------------------------------------===//

/// An attribute constraint that references anything other than itself and the
/// current op cannot be generically extracted into a function. Most
/// prohibitive are operands and results, which require calls to
/// `getODSOperands` or `getODSResults`. Attribute references are tricky too
/// because ops use cached identifiers.
static bool canUniqueAttrConstraint(Attribute attr) {
  FmtContext ctx;
  auto test = tgfmtChecked(attr.getConditionTemplate(),
                           ctx.withSelf("attr").addSubst("_op", "*op"));
  if (!test) {
    llvm::consumeError(test.takeError());
    return false;
  }
  auto summary = buildErrorStreamingStringChecked(attr.getSummary(), ctx);
  if (!summary) {
    llvm::consumeError(summary.takeError());
    return false;
  }
  return true;
}

/// A property constraint that references anything other than itself and the
/// current op cannot be generically extracted into a function, just as with
/// canUnequePropConstraint(). Additionally, property constraints without
/// an interface type specified can't be uniqued, and ones that are a literal
/// "true" shouldn't be constrained.
static bool canUniquePropConstraint(Property prop) {
  FmtContext ctx;
  auto test = tgfmtChecked(prop.getConditionTemplate(),
                           ctx.withSelf("prop").addSubst("_op", "*op"));
  if (!test) {
    llvm::consumeError(test.takeError());
    return false;
  }
  auto summary = buildErrorStreamingStringChecked(prop.getSummary(), ctx);
  if (!summary) {
    llvm::consumeError(summary.takeError());
    return false;
  }
  return *test != "true" && !prop.getInterfaceType().empty();
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
    /// Collect non-trivial property constraints.
    for (const NamedProperty &namedProp : op.getProperties()) {
      if (!namedProp.prop.getPredicate().isNull() &&
          canUniquePropConstraint(namedProp.prop)) {
        collectConstraint(propConstraints, "prop", namedProp.prop);
      }
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
    assert(leaf.isOperandMatcher() || leaf.isAttrMatcher() ||
           leaf.isPropMatcher());
    Constraint constraint = leaf.getAsConstraint();
    if (leaf.isOperandMatcher())
      collectConstraint(typeConstraints, "type", constraint);
    else if (leaf.isAttrMatcher())
      collectConstraint(attrConstraints, "attr", constraint);
    else if (leaf.isPropMatcher())
      collectConstraint(propConstraints, "prop", constraint);
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
