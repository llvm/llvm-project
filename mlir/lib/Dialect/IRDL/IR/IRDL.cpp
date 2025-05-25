//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLSymbols.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing/Printing/Verifying
//===----------------------------------------------------------------------===//

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.has_value() && failed(regionParseRes.value()))
    return failure();

  // If the region is empty, add a single empty block.
  if (region.empty())
    region.push_back(new Block());

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty())
    p.printRegion(region);
}
static llvm::LogicalResult isValidName(llvm::StringRef in, mlir::Operation *loc,
                                       const Twine &label) {
  if (in.empty())
    return loc->emitError("name of ") << label << " is empty";

  bool allowUnderscore = false;
  for (auto &elem : in) {
    if (elem == '_') {
      if (!allowUnderscore)
        return loc->emitError("name of ")
               << label << " should not contain leading or double underscores";
    } else {
      if (!isalnum(elem))
        return loc->emitError("name of ")
               << label
               << " must contain only lowercase letters, digits and "
                  "underscores";

      if (llvm::isUpper(elem))
        return loc->emitError("name of ")
               << label << " should not contain uppercase letters";
    }

    allowUnderscore = elem != '_';
  }

  return success();
}

LogicalResult DialectOp::verify() {
  if (!Dialect::isValidNamespace(getName()))
    return emitOpError("invalid dialect name");
  if (failed(isValidName(getSymName(), getOperation(), "dialect")))
    return failure();

  return success();
}

LogicalResult OperationOp::verify() {
  return isValidName(getSymName(), getOperation(), "operation");
}

LogicalResult TypeOp::verify() {
  auto symName = getSymName();
  if (symName.front() == '!')
    symName = symName.substr(1);
  return isValidName(symName, getOperation(), "type");
}

LogicalResult AttributeOp::verify() {
  auto symName = getSymName();
  if (symName.front() == '#')
    symName = symName.substr(1);
  return isValidName(symName, getOperation(), "attribute");
}

LogicalResult OperationOp::verifyRegions() {
  // Stores pairs of value kinds and the list of names of values of this kind in
  // the operation.
  SmallVector<std::tuple<StringRef, llvm::SmallDenseSet<StringRef>>> valueNames;

  auto insertNames = [&](StringRef kind, ArrayAttr names) {
    llvm::SmallDenseSet<StringRef> nameSet;
    nameSet.reserve(names.size());
    for (auto name : names)
      nameSet.insert(llvm::cast<StringAttr>(name).getValue());
    valueNames.emplace_back(kind, std::move(nameSet));
  };

  for (Operation &op : getBody().getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case<OperandsOp>(
            [&](OperandsOp op) { insertNames("operands", op.getNames()); })
        .Case<ResultsOp>(
            [&](ResultsOp op) { insertNames("results", op.getNames()); })
        .Case<RegionsOp>(
            [&](RegionsOp op) { insertNames("regions", op.getNames()); });
  }

  // Verify that no two operand, result or region share the same name.
  // The absence of duplicates within each value kind is checked by the
  // associated operation's verifier.
  for (size_t i : llvm::seq(valueNames.size())) {
    for (size_t j : llvm::seq(i + 1, valueNames.size())) {
      auto [lhs, lhsSet] = valueNames[i];
      auto &[rhs, rhsSet] = valueNames[j];
      llvm::set_intersect(lhsSet, rhsSet);
      if (!lhsSet.empty())
        return emitOpError("contains a value named '")
               << *lhsSet.begin() << "' for both its " << lhs << " and " << rhs;
    }
  }

  return success();
}

static LogicalResult verifyNames(Operation *op, StringRef kindName,
                                 ArrayAttr names, size_t numOperands) {
  if (numOperands != names.size())
    return op->emitOpError()
           << "the number of " << kindName
           << "s and their names must be "
              "the same, but got "
           << numOperands << " and " << names.size() << " respectively";

  DenseMap<StringRef, size_t> nameMap;
  for (auto [i, name] : llvm::enumerate(names)) {
    StringRef nameRef = llvm::cast<StringAttr>(name).getValue();

    if (failed(isValidName(nameRef, op, Twine(kindName) + " #" + Twine(i))))
      return failure();

    if (nameMap.contains(nameRef))
      return op->emitOpError() << "name of " << kindName << " #" << i
                               << " is a duplicate of the name of " << kindName
                               << " #" << nameMap[nameRef];
    nameMap.insert({nameRef, i});
  }

  return success();
}

LogicalResult ParametersOp::verify() {
  return verifyNames(*this, "parameter", getNames(), getNumOperands());
}

template <typename ValueListOp>
static LogicalResult verifyOperandsResultsCommon(ValueListOp op,
                                                 StringRef kindName) {
  size_t numVariadicities = op.getVariadicity().size();
  size_t numOperands = op.getNumOperands();

  if (numOperands != numVariadicities)
    return op.emitOpError()
           << "the number of " << kindName
           << "s and their variadicities must be "
              "the same, but got "
           << numOperands << " and " << numVariadicities << " respectively";

  return verifyNames(op, kindName, op.getNames(), numOperands);
}

LogicalResult OperandsOp::verify() {
  return verifyOperandsResultsCommon(*this, "operand");
}

LogicalResult ResultsOp::verify() {
  return verifyOperandsResultsCommon(*this, "result");
}

LogicalResult AttributesOp::verify() {
  size_t namesSize = getAttributeValueNames().size();
  size_t valuesSize = getAttributeValues().size();

  if (namesSize != valuesSize)
    return emitOpError()
           << "the number of attribute names and their constraints must be "
              "the same but got "
           << namesSize << " and " << valuesSize << " respectively";

  return success();
}

LogicalResult BaseOp::verify() {
  std::optional<StringRef> baseName = getBaseName();
  std::optional<SymbolRefAttr> baseRef = getBaseRef();
  if (baseName.has_value() == baseRef.has_value())
    return emitOpError() << "the base type or attribute should be specified by "
                            "either a name or a reference";

  if (baseName &&
      (baseName->empty() || ((*baseName)[0] != '!' && (*baseName)[0] != '#')))
    return emitOpError() << "the base type or attribute name should start with "
                            "'!' or '#'";

  return success();
}

/// Finds whether the provided symbol is an IRDL type or attribute definition.
/// The source operation must be within a DialectOp.
static LogicalResult
checkSymbolIsTypeOrAttribute(SymbolTableCollection &symbolTable,
                             Operation *source, SymbolRefAttr symbol) {
  Operation *targetOp =
      irdl::lookupSymbolNearDialect(symbolTable, source, symbol);

  if (!targetOp)
    return source->emitOpError() << "symbol '" << symbol << "' not found";

  if (!isa<TypeOp, AttributeOp>(targetOp))
    return source->emitOpError() << "symbol '" << symbol
                                 << "' does not refer to a type or attribute "
                                    "definition (refers to '"
                                 << targetOp->getName() << "')";

  return success();
}

LogicalResult BaseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  std::optional<SymbolRefAttr> baseRef = getBaseRef();
  if (!baseRef)
    return success();

  return checkSymbolIsTypeOrAttribute(symbolTable, *this, *baseRef);
}

LogicalResult
ParametricOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  std::optional<SymbolRefAttr> baseRef = getBaseType();
  if (!baseRef)
    return success();

  return checkSymbolIsTypeOrAttribute(symbolTable, *this, *baseRef);
}

/// Parse a value with its variadicity first. By default, the variadicity is
/// single.
///
/// value-with-variadicity ::= ("single" | "optional" | "variadic")? ssa-value
static ParseResult
parseValueWithVariadicity(OpAsmParser &p,
                          OpAsmParser::UnresolvedOperand &operand,
                          VariadicityAttr &variadicityAttr) {
  MLIRContext *ctx = p.getBuilder().getContext();

  // Parse the variadicity, if present
  if (p.parseOptionalKeyword("single").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::single);
  } else if (p.parseOptionalKeyword("optional").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::optional);
  } else if (p.parseOptionalKeyword("variadic").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::variadic);
  } else {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::single);
  }

  // Parse the value
  if (p.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseNamedValueListImpl(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    ArrayAttr &valueNamesAttr, VariadicityArrayAttr *variadicityAttr) {
  Builder &builder = p.getBuilder();
  MLIRContext *ctx = builder.getContext();
  SmallVector<Attribute> valueNames;
  SmallVector<VariadicityAttr> variadicities;

  // Parse a single value with its variadicity
  auto parseOne = [&] {
    StringRef name;
    OpAsmParser::UnresolvedOperand operand;
    VariadicityAttr variadicity;
    if (p.parseKeyword(&name) || p.parseColon())
      return failure();

    if (variadicityAttr) {
      if (parseValueWithVariadicity(p, operand, variadicity))
        return failure();
      variadicities.push_back(variadicity);
    } else {
      if (p.parseOperand(operand))
        return failure();
    }

    valueNames.push_back(StringAttr::get(ctx, name));
    operands.push_back(operand);
    return success();
  };

  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseOne))
    return failure();
  valueNamesAttr = ArrayAttr::get(ctx, valueNames);
  if (variadicityAttr)
    *variadicityAttr = VariadicityArrayAttr::get(ctx, variadicities);
  return success();
}

/// Parse a list of named values.
///
/// values ::=
///   `(` (named-value (`,` named-value)*)? `)`
/// named-value := bare-id `:` ssa-value
static ParseResult
parseNamedValueList(OpAsmParser &p,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                    ArrayAttr &valueNamesAttr) {
  return parseNamedValueListImpl(p, operands, valueNamesAttr, nullptr);
}

/// Parse a list of named values with their variadicities first. By default, the
/// variadicity is single.
///
/// values-with-variadicity ::=
///   `(` (value-with-variadicity (`,` value-with-variadicity)*)? `)`
/// value-with-variadicity
///   ::= bare-id `:` ("single" | "optional" | "variadic")? ssa-value
static ParseResult parseNamedValueListWithVariadicity(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    ArrayAttr &valueNamesAttr, VariadicityArrayAttr &variadicityAttr) {
  return parseNamedValueListImpl(p, operands, valueNamesAttr, &variadicityAttr);
}

static void printNamedValueListImpl(OpAsmPrinter &p, Operation *op,
                                    OperandRange operands,
                                    ArrayAttr valueNamesAttr,
                                    VariadicityArrayAttr variadicityAttr) {
  p << "(";
  interleaveComma(llvm::seq<int>(0, operands.size()), p, [&](int i) {
    p << llvm::cast<StringAttr>(valueNamesAttr[i]).getValue() << ": ";
    if (variadicityAttr) {
      Variadicity variadicity = variadicityAttr[i].getValue();
      if (variadicity != Variadicity::single) {
        p << stringifyVariadicity(variadicity) << " ";
      }
    }
    p << operands[i];
  });
  p << ")";
}

/// Print a list of named values.
///
/// values ::=
///   `(` (named-value (`,` named-value)*)? `)`
/// named-value := bare-id `:` ssa-value
static void printNamedValueList(OpAsmPrinter &p, Operation *op,
                                OperandRange operands,
                                ArrayAttr valueNamesAttr) {
  printNamedValueListImpl(p, op, operands, valueNamesAttr, nullptr);
}

/// Print a list of named values with their variadicities first. By default, the
/// variadicity is single.
///
/// values-with-variadicity ::=
///   `(` (value-with-variadicity (`,` value-with-variadicity)*)? `)`
/// value-with-variadicity ::=
///   bare-id `:` ("single" | "optional" | "variadic")? ssa-value
static void printNamedValueListWithVariadicity(
    OpAsmPrinter &p, Operation *op, OperandRange operands,
    ArrayAttr valueNamesAttr, VariadicityArrayAttr variadicityAttr) {
  printNamedValueListImpl(p, op, operands, valueNamesAttr, variadicityAttr);
}

static ParseResult
parseAttributesOp(OpAsmParser &p,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &attrOperands,
                  ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    auto parseOperands = [&]() {
      if (p.parseAttribute(attrNames.emplace_back()) || p.parseEqual() ||
          p.parseOperand(attrOperands.emplace_back()))
        return failure();
      return success();
    };
    if (p.parseCommaSeparatedList(parseOperands) || p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  return success();
}

static void printAttributesOp(OpAsmPrinter &p, AttributesOp op,
                              OperandRange attrArgs, ArrayAttr attrNames) {
  if (attrNames.empty())
    return;
  p << "{";
  interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                  [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
  p << '}';
}

LogicalResult RegionOp::verify() {
  if (IntegerAttr numberOfBlocks = getNumberOfBlocksAttr())
    if (int64_t number = numberOfBlocks.getInt(); number <= 0) {
      return emitOpError("the number of blocks is expected to be >= 1 but got ")
             << number;
    }
  return success();
}

LogicalResult RegionsOp::verify() {
  return verifyNames(*this, "region", getNames(), getNumOperands());
}

#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
