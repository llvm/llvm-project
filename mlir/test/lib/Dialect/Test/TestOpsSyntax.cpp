//===- TestOpsSyntax.cpp - Operations for testing syntax ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsSyntax.h"
#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Base64.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// Test Format* operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parsing

static ParseResult parseCustomOptionalOperand(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  if (succeeded(parser.parseOptionalLParen())) {
    optOperand.emplace();
    if (parser.parseOperand(*optOperand) || parser.parseRParen())
      return failure();
  }
  return success();
}

static ParseResult parseCustomDirectiveOperands(
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
static ParseResult
parseCustomDirectiveResults(OpAsmParser &parser, Type &operandType,
                            Type &optOperandType,
                            SmallVectorImpl<Type> &varOperandTypes) {
  if (parser.parseColon())
    return failure();

  if (parser.parseType(operandType))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(optOperandType))
      return failure();
  }
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseTypeList(varOperandTypes) || parser.parseRParen())
    return failure();
  return success();
}
static ParseResult
parseCustomDirectiveWithTypeRefs(OpAsmParser &parser, Type operandType,
                                 Type optOperandType,
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
static ParseResult parseCustomDirectiveOperandsAndTypes(
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
static ParseResult parseCustomDirectiveRegions(
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
static ParseResult
parseCustomDirectiveSuccessors(OpAsmParser &parser, Block *&successor,
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
static ParseResult parseCustomDirectiveAttributes(OpAsmParser &parser,
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
static ParseResult parseCustomDirectiveSpacing(OpAsmParser &parser,
                                               mlir::StringAttr &attr) {
  return parser.parseAttribute(attr);
}
static ParseResult parseCustomDirectiveAttrDict(OpAsmParser &parser,
                                                NamedAttrList &attrs) {
  return parser.parseOptionalAttrDict(attrs);
}
static ParseResult parseCustomDirectiveOptionalOperandRef(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &optOperand) {
  int64_t operandCount = 0;
  if (parser.parseInteger(operandCount))
    return failure();
  bool expectedOptionalOperand = operandCount == 0;
  return success(expectedOptionalOperand != optOperand.has_value());
}

//===----------------------------------------------------------------------===//
// Printing

static void printCustomOptionalOperand(OpAsmPrinter &printer, Operation *,
                                       Value optOperand) {
  if (optOperand)
    printer << "(" << optOperand << ") ";
}

static void printCustomDirectiveOperands(OpAsmPrinter &printer, Operation *,
                                         Value operand, Value optOperand,
                                         OperandRange varOperands) {
  printer << operand;
  if (optOperand)
    printer << ", " << optOperand;
  printer << " -> (" << varOperands << ")";
}
static void printCustomDirectiveResults(OpAsmPrinter &printer, Operation *,
                                        Type operandType, Type optOperandType,
                                        TypeRange varOperandTypes) {
  printer << " : " << operandType;
  if (optOperandType)
    printer << ", " << optOperandType;
  printer << " -> (" << varOperandTypes << ")";
}
static void printCustomDirectiveWithTypeRefs(OpAsmPrinter &printer,
                                             Operation *op, Type operandType,
                                             Type optOperandType,
                                             TypeRange varOperandTypes) {
  printer << " type_refs_capture ";
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}
static void printCustomDirectiveOperandsAndTypes(
    OpAsmPrinter &printer, Operation *op, Value operand, Value optOperand,
    OperandRange varOperands, Type operandType, Type optOperandType,
    TypeRange varOperandTypes) {
  printCustomDirectiveOperands(printer, op, operand, optOperand, varOperands);
  printCustomDirectiveResults(printer, op, operandType, optOperandType,
                              varOperandTypes);
}
static void printCustomDirectiveRegions(OpAsmPrinter &printer, Operation *,
                                        Region &region,
                                        MutableArrayRef<Region> varRegions) {
  printer.printRegion(region);
  if (!varRegions.empty()) {
    printer << ", ";
    for (Region &region : varRegions)
      printer.printRegion(region);
  }
}
static void printCustomDirectiveSuccessors(OpAsmPrinter &printer, Operation *,
                                           Block *successor,
                                           SuccessorRange varSuccessors) {
  printer << successor;
  if (!varSuccessors.empty())
    printer << ", " << varSuccessors.front();
}
static void printCustomDirectiveAttributes(OpAsmPrinter &printer, Operation *,
                                           Attribute attribute,
                                           Attribute optAttribute) {
  printer << attribute;
  if (optAttribute)
    printer << ", " << optAttribute;
}
static void printCustomDirectiveSpacing(OpAsmPrinter &printer, Operation *op,
                                        Attribute attribute) {
  printer << attribute;
}
static void printCustomDirectiveAttrDict(OpAsmPrinter &printer, Operation *op,
                                         DictionaryAttr attrs) {
  printer.printOptionalAttrDict(attrs.getValue());
}

static void printCustomDirectiveOptionalOperandRef(OpAsmPrinter &printer,
                                                   Operation *op,
                                                   Value optOperand) {
  printer << (optOperand ? "1" : "0");
}
//===----------------------------------------------------------------------===//
// Test parser.
//===----------------------------------------------------------------------===//

ParseResult ParseIntegerLiteralOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalColon())
    return success();
  uint64_t numResults;
  if (parser.parseInteger(numResults))
    return failure();

  IndexType type = parser.getBuilder().getIndexType();
  for (unsigned i = 0; i < numResults; ++i)
    result.addTypes(type);
  return success();
}

void ParseIntegerLiteralOp::print(OpAsmPrinter &p) {
  if (unsigned numResults = getNumResults())
    p << " : " << numResults;
}

ParseResult ParseWrappedKeywordOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  result.addAttribute("keyword", parser.getBuilder().getStringAttr(keyword));
  return success();
}

void ParseWrappedKeywordOp::print(OpAsmPrinter &p) { p << " " << getKeyword(); }

ParseResult ParseB64BytesOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  std::vector<char> bytes;
  if (parser.parseBase64Bytes(&bytes))
    return failure();
  result.addAttribute("b64", parser.getBuilder().getStringAttr(
                                 StringRef(&bytes.front(), bytes.size())));
  return success();
}

void ParseB64BytesOp::print(OpAsmPrinter &p) {
  p << " \"" << llvm::encodeBase64(getB64()) << "\"";
}

::mlir::LogicalResult FormatInferType2Op::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.assign({::mlir::IntegerType::get(context, 16)});
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Test WrapRegionOp - wrapping op exercising `parseGenericOperation()`.

ParseResult WrappingRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseKeyword("wraps"))
    return failure();

  // Parse the wrapped op in a region
  Region &body = *result.addRegion();
  body.push_back(new Block);
  Block &block = body.back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());
  if (!wrappedOp)
    return failure();

  // Create a return terminator in the inner region, pass as operand to the
  // terminator the returned values from the wrapped operation.
  SmallVector<Value, 8> returnOperands(wrappedOp->getResults());
  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<TestReturnOp>(wrappedOp->getLoc(), returnOperands);

  // Get the results type for the wrapping op from the terminator operands.
  Operation &returnOp = body.back().back();
  result.types.append(returnOp.operand_type_begin(),
                      returnOp.operand_type_end());

  // Use the location of the wrapped op for the "test.wrapping_region" op.
  result.location = wrappedOp->getLoc();

  return success();
}

void WrappingRegionOp::print(OpAsmPrinter &p) {
  p << " wraps ";
  p.printGenericOp(&getRegion().front().front());
}

//===----------------------------------------------------------------------===//
// Test PrettyPrintedRegionOp -  exercising the following parser APIs
//   parseGenericOperationAfterOpName
//   parseCustomOperationName
//===----------------------------------------------------------------------===//

ParseResult PrettyPrintedRegionOp::parse(OpAsmParser &parser,
                                         OperationState &result) {

  SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // Check if we are parsing the pretty-printed version
  //  test.pretty_printed_region start <inner-op> end : <functional-type>
  // Else fallback to parsing the "non pretty-printed" version.
  if (!succeeded(parser.parseOptionalKeyword("start")))
    return parser.parseGenericOperationAfterOpName(result,
                                                   llvm::ArrayRef(operands));

  FailureOr<OperationName> parseOpNameInfo = parser.parseCustomOperationName();
  if (failed(parseOpNameInfo))
    return failure();

  StringAttr innerOpName = parseOpNameInfo->getIdentifier();

  FunctionType opFntype;
  std::optional<Location> explicitLoc;
  if (parser.parseKeyword("end") || parser.parseColon() ||
      parser.parseType(opFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  // If location of the op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location opLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for op's operands.
  if (parser.resolveOperands(operands, opFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Add a region for op.
  Region &region = *result.addRegion();

  // Create a basic-block inside op's region.
  Block &block = region.emplaceBlock();

  // Create and insert an "inner-op" operation in the block.
  // Just for testing purposes, we can assume that inner op is a binary op with
  // result and operand types all same as the test-op's first operand.
  Type innerOpType = opFntype.getInput(0);
  Value lhs = block.addArgument(innerOpType, opLoc);
  Value rhs = block.addArgument(innerOpType, opLoc);

  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  Operation *innerOp =
      builder.create(opLoc, innerOpName, /*operands=*/{lhs, rhs}, innerOpType);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<TestReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the op operation-state with result-type and location.
  result.addTypes(opFntype.getResults());
  result.location = innerOp->getLoc();

  return success();
}

void PrettyPrintedRegionOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());

  Operation &innerOp = getRegion().front().front();
  // Assuming that region has a single non-terminator inner-op, if the inner-op
  // meets some criteria (which in this case is a simple one  based on the name
  // of inner-op), then we can print the entire region in a succinct way.
  // Here we assume that the prototype of "test.special.op" can be trivially
  // derived while parsing it back.
  if (innerOp.getName().getStringRef() == "test.special.op") {
    p << " start test.special.op end";
  } else {
    p << " (";
    p.printRegion(getRegion());
    p << ")";
  }

  p << " : ";
  p.printFunctionalType(*this);
}

//===----------------------------------------------------------------------===//
// Test PolyForOp - parse list of region arguments.
//===----------------------------------------------------------------------===//

ParseResult PolyForOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> ivsInfo;
  // Parse list of region arguments without a delimiter.
  if (parser.parseArgumentList(ivsInfo, OpAsmParser::Delimiter::None))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  for (auto &iv : ivsInfo)
    iv.type = parser.getBuilder().getIndexType();
  return parser.parseRegion(*body, ivsInfo);
}

void PolyForOp::print(OpAsmPrinter &p) {
  p << " ";
  llvm::interleaveComma(getRegion().getArguments(), p, [&](auto arg) {
    p.printRegionArgument(arg, /*argAttrs =*/{}, /*omitType=*/true);
  });
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

void PolyForOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  auto arrayAttr = getOperation()->getAttrOfType<ArrayAttr>("arg_names");
  if (!arrayAttr)
    return;
  auto args = getRegion().front().getArguments();
  auto e = std::min(arrayAttr.size(), args.size());
  for (unsigned i = 0; i < e; ++i) {
    if (auto strAttr = dyn_cast<StringAttr>(arrayAttr[i]))
      setNameFn(args[i], strAttr.getValue());
  }
}

//===----------------------------------------------------------------------===//
// TestAttrWithLoc - parse/printOptionalLocationSpecifier
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalLoc(OpAsmParser &p, Attribute &loc) {
  std::optional<Location> result;
  SMLoc sourceLoc = p.getCurrentLocation();
  if (p.parseOptionalLocationSpecifier(result))
    return failure();
  if (result)
    loc = *result;
  else
    loc = p.getEncodedSourceLoc(sourceLoc);
  return success();
}

static void printOptionalLoc(OpAsmPrinter &p, Operation *op, Attribute loc) {
  p.printOptionalLocationSpecifier(cast<LocationAttr>(loc));
}

#define GET_OP_CLASSES
#include "TestOpsSyntax.cpp.inc"

void TestDialect::registerOpsSyntax() {
  addOperations<
#define GET_OP_LIST
#include "TestOpsSyntax.cpp.inc"
      >();
}
