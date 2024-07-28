//===- TestFormatUtils.cpp - MLIR Test Dialect Assembly Format Utilities --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFormatUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// CustomDirectiveOperands
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveOperands(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varOperands) {
  if (parser.parseOperand(operand))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand))
      return failure();
  }
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseOperandList(varOperands) || parser.parseRParen())
    return failure();
  return success();
}

void test::printCustomDirectiveOperands(OpAsmPrinter &printer, Operation *,
                                        Value operand, Value optOperand,
                                        OperandRange varOperands) {
  printer << operand;
  if (optOperand)
    printer << ", " << optOperand;
  printer << " -> (" << varOperands << ")";
}

//===----------------------------------------------------------------------===//
// CustomDirectiveResults
//===----------------------------------------------------------------------===//

ParseResult
test::parseCustomDirectiveResults(OpAsmParser &parser, Type &operandType,
                                  Type &optOperandType,
                                  SmallVectorImpl<Type> &varOperandTypes) {
  if (parser.parseColon())
    return failure();

  if (parser.parseType(operandType))
    return failure();
  if (succeeded(parser.parseOptionalComma()))
    if (parser.parseType(optOperandType))
      return failure();
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseTypeList(varOperandTypes) || parser.parseRParen())
    return failure();
  return success();
}

void test::printCustomDirectiveResults(OpAsmPrinter &printer, Operation *,
                                       Type operandType, Type optOperandType,
                                       TypeRange varOperandTypes) {
  printer << " : " << operandType;
  if (optOperandType)
    printer << ", " << optOperandType;
  printer << " -> (" << varOperandTypes << ")";
}

//===----------------------------------------------------------------------===//
// CustomDirectiveWithTypeRefs
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveWithTypeRefs(
    OpAsmParser &parser, Type operandType, Type optOperandType,
    const SmallVectorImpl<Type> &varOperandTypes) {
  if (parser.parseKeyword("type_refs_capture"))
    return failure();

  Type operandType2, optOperandType2;
  SmallVector<Type, 1> varOperandTypes2;
  if (parseCustomDirectiveResults(parser, operandType2, optOperandType2,
                                  varOperandTypes2))
    return failure();

  if (operandType != operandType2 || optOperandType != optOperandType2 ||
      varOperandTypes != varOperandTypes2)
    return failure();

  return success();
}

void test::printCustomDirectiveWithTypeRefs(OpAsmPrinter &printer,
                                            Operation *op, Type operandType,
                                            Type optOperandType,
                                            TypeRange varOperandTypes) {
  printer << " type_refs_capture ";
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}

//===----------------------------------------------------------------------===//
// CustomDirectiveOperandsAndTypes
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveOperandsAndTypes(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varOperands,
    Type &operandType, Type &optOperandType,
    SmallVectorImpl<Type> &varOperandTypes) {
  if (parseCustomDirectiveOperands(parser, operand, optOperand, varOperands) ||
      parseCustomDirectiveResults(parser, operandType, optOperandType,
                                  varOperandTypes))
    return failure();
  return success();
}

void test::printCustomDirectiveOperandsAndTypes(
    OpAsmPrinter &printer, Operation *op, Value operand, Value optOperand,
    OperandRange varOperands, Type operandType, Type optOperandType,
    TypeRange varOperandTypes) {
  printCustomDirectiveOperands(printer, op, operand, optOperand, varOperands);
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}

//===----------------------------------------------------------------------===//
// CustomDirectiveRegions
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveRegions(
    OpAsmParser &parser, Region &region,
    SmallVectorImpl<std::unique_ptr<Region>> &varRegions) {
  if (parser.parseRegion(region))
    return failure();
  if (failed(parser.parseOptionalComma()))
    return success();
  std::unique_ptr<Region> varRegion = std::make_unique<Region>();
  if (parser.parseRegion(*varRegion))
    return failure();
  varRegions.emplace_back(std::move(varRegion));
  return success();
}

void test::printCustomDirectiveRegions(OpAsmPrinter &printer, Operation *,
                                       Region &region,
                                       MutableArrayRef<Region> varRegions) {
  printer.printRegion(region);
  if (!varRegions.empty()) {
    printer << ", ";
    for (Region &region : varRegions)
      printer.printRegion(region);
  }
}

//===----------------------------------------------------------------------===//
// CustomDirectiveSuccessors
//===----------------------------------------------------------------------===//

ParseResult
test::parseCustomDirectiveSuccessors(OpAsmParser &parser, Block *&successor,
                                     SmallVectorImpl<Block *> &varSuccessors) {
  if (parser.parseSuccessor(successor))
    return failure();
  if (failed(parser.parseOptionalComma()))
    return success();
  Block *varSuccessor;
  if (parser.parseSuccessor(varSuccessor))
    return failure();
  varSuccessors.append(2, varSuccessor);
  return success();
}

void test::printCustomDirectiveSuccessors(OpAsmPrinter &printer, Operation *,
                                          Block *successor,
                                          SuccessorRange varSuccessors) {
  printer << successor;
  if (!varSuccessors.empty())
    printer << ", " << varSuccessors.front();
}

//===----------------------------------------------------------------------===//
// CustomDirectiveAttributes
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveAttributes(OpAsmParser &parser,
                                                 IntegerAttr &attr,
                                                 IntegerAttr &optAttr) {
  if (parser.parseAttribute(attr))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(optAttr))
      return failure();
  }
  return success();
}

void test::printCustomDirectiveAttributes(OpAsmPrinter &printer, Operation *,
                                          Attribute attribute,
                                          Attribute optAttribute) {
  printer << attribute;
  if (optAttribute)
    printer << ", " << optAttribute;
}

//===----------------------------------------------------------------------===//
// CustomDirectiveAttrDict
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveAttrDict(OpAsmParser &parser,
                                               NamedAttrList &attrs) {
  return parser.parseOptionalAttrDict(attrs);
}

void test::printCustomDirectiveAttrDict(OpAsmPrinter &printer, Operation *op,
                                        DictionaryAttr attrs) {
  printer.printOptionalAttrDict(attrs.getValue());
}

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperandRef
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomDirectiveOptionalOperandRef(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  int64_t operandCount = 0;
  if (parser.parseInteger(operandCount))
    return failure();
  bool expectedOptionalOperand = operandCount == 0;
  return success(expectedOptionalOperand != !!optOperand);
}

void test::printCustomDirectiveOptionalOperandRef(OpAsmPrinter &printer,
                                                  Operation *op,
                                                  Value optOperand) {
  printer << (optOperand ? "1" : "0");
}

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperand
//===----------------------------------------------------------------------===//

ParseResult test::parseCustomOptionalOperand(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  if (succeeded(parser.parseOptionalLParen())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand) || parser.parseRParen())
      return failure();
  }
  return success();
}

void test::printCustomOptionalOperand(OpAsmPrinter &printer, Operation *,
                                      Value optOperand) {
  if (optOperand)
    printer << "(" << optOperand << ") ";
}

//===----------------------------------------------------------------------===//
// CustomDirectiveSwitchCases
//===----------------------------------------------------------------------===//

ParseResult
test::parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                       SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (p.parseInteger(value) || p.parseRegion(region, /*arguments=*/{}))
      return failure();
    caseValues.push_back(value);
  }
  cases = p.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

void test::printSwitchCases(OpAsmPrinter &p, Operation *op,
                            DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
}

//===----------------------------------------------------------------------===//
// CustomUsingPropertyInCustom
//===----------------------------------------------------------------------===//

bool test::parseUsingPropertyInCustom(OpAsmParser &parser,
                                      SmallVector<int64_t> &value) {
  auto elemParser = [&]() {
    int64_t v = 0;
    if (failed(parser.parseInteger(v)))
      return failure();
    value.push_back(v);
    return success();
  };
  return failed(parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                               elemParser));
}

void test::printUsingPropertyInCustom(OpAsmPrinter &printer, Operation *op,
                                      ArrayRef<int64_t> value) {
  printer << '[' << value << ']';
}

//===----------------------------------------------------------------------===//
// CustomDirectiveIntProperty
//===----------------------------------------------------------------------===//

bool test::parseIntProperty(OpAsmParser &parser, int64_t &value) {
  return failed(parser.parseInteger(value));
}

void test::printIntProperty(OpAsmPrinter &printer, Operation *op,
                            int64_t value) {
  printer << value;
}

//===----------------------------------------------------------------------===//
// CustomDirectiveSumProperty
//===----------------------------------------------------------------------===//

bool test::parseSumProperty(OpAsmParser &parser, int64_t &second,
                            int64_t first) {
  int64_t sum;
  auto loc = parser.getCurrentLocation();
  if (parser.parseInteger(second) || parser.parseEqual() ||
      parser.parseInteger(sum))
    return true;
  if (sum != second + first) {
    parser.emitError(loc, "Expected sum to equal first + second");
    return true;
  }
  return false;
}

void test::printSumProperty(OpAsmPrinter &printer, Operation *op,
                            int64_t second, int64_t first) {
  printer << second << " = " << (second + first);
}

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalCustomParser
//===----------------------------------------------------------------------===//

OptionalParseResult test::parseOptionalCustomParser(AsmParser &p,
                                                    IntegerAttr &result) {
  if (succeeded(p.parseOptionalKeyword("foo")))
    return p.parseAttribute(result);
  return {};
}

void test::printOptionalCustomParser(AsmPrinter &p, Operation *,
                                     IntegerAttr result) {
  p << "foo ";
  p.printAttribute(result);
}

//===----------------------------------------------------------------------===//
// CustomDirectiveAttrElideType
//===----------------------------------------------------------------------===//

ParseResult test::parseAttrElideType(AsmParser &parser, TypeAttr type,
                                     Attribute &attr) {
  return parser.parseAttribute(attr, type.getValue());
}

void test::printAttrElideType(AsmPrinter &printer, Operation *op, TypeAttr type,
                              Attribute attr) {
  printer.printAttributeWithoutType(attr);
}
