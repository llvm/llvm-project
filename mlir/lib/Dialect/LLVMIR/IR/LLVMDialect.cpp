//===- LLVMDialect.cpp - LLVM IR Ops and Dialect registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the LLVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

static constexpr const char kVolatileAttrName[] = "volatile_";
static constexpr const char kNonTemporalAttrName[] = "nontemporal";

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CmpOp.
//===----------------------------------------------------------------------===//
static void printICmpOp(OpAsmPrinter &p, ICmpOp &op) {
  p << op.getOperationName() << " \"" << stringifyICmpPredicate(op.predicate())
    << "\" " << op.getOperand(0) << ", " << op.getOperand(1);
  p.printOptionalAttrDict(op.getAttrs(), {"predicate"});
  p << " : " << op.lhs().getType();
}

static void printFCmpOp(OpAsmPrinter &p, FCmpOp &op) {
  p << op.getOperationName() << " \"" << stringifyFCmpPredicate(op.predicate())
    << "\" " << op.getOperand(0) << ", " << op.getOperand(1);
  p.printOptionalAttrDict(op.getAttrs(), {"predicate"});
  p << " : " << op.lhs().getType();
}

// <operation> ::= `llvm.icmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
// <operation> ::= `llvm.fcmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
template <typename CmpPredicateType>
static ParseResult parseCmpOp(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  StringAttr predicateAttr;
  OpAsmParser::OperandType lhs, rhs;
  Type type;
  llvm::SMLoc predicateLoc, trailingTypeLoc;
  if (parser.getCurrentLocation(&predicateLoc) ||
      parser.parseAttribute(predicateAttr, "predicate", result.attributes) ||
      parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type) ||
      parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands))
    return failure();

  // Replace the string attribute `predicate` with an integer attribute.
  int64_t predicateValue = 0;
  if (std::is_same<CmpPredicateType, ICmpPredicate>()) {
    Optional<ICmpPredicate> predicate =
        symbolizeICmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  } else {
    Optional<FCmpPredicate> predicate =
        symbolizeFCmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  }

  result.attributes.set("predicate",
                        parser.getBuilder().getI64IntegerAttr(predicateValue));

  // The result type is either i1 or a vector type <? x i1> if the inputs are
  // vectors.
  auto resultType = LLVMType::getInt1Ty(builder.getContext());
  auto argType = type.dyn_cast<LLVM::LLVMType>();
  if (!argType)
    return parser.emitError(trailingTypeLoc, "expected LLVM IR dialect type");
  if (argType.isVectorTy())
    resultType =
        LLVMType::getVectorTy(resultType, argType.getVectorNumElements());

  result.addTypes({resultType});
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

static void printAllocaOp(OpAsmPrinter &p, AllocaOp &op) {
  auto elemTy = op.getType().cast<LLVM::LLVMType>().getPointerElementTy();

  auto funcTy = FunctionType::get({op.arraySize().getType()}, {op.getType()},
                                  op.getContext());

  p << op.getOperationName() << ' ' << op.arraySize() << " x " << elemTy;
  if (op.alignment().hasValue() && *op.alignment() != 0)
    p.printOptionalAttrDict(op.getAttrs());
  else
    p.printOptionalAttrDict(op.getAttrs(), {"alignment"});
  p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` ssa-use `x` type attribute-dict?
//                 `:` type `,` type
static ParseResult parseAllocaOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType arraySize;
  Type type, elemType;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseOperand(arraySize) || parser.parseKeyword("x") ||
      parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  Optional<NamedAttribute> alignmentAttr =
      result.attributes.getNamed("alignment");
  if (alignmentAttr.hasValue()) {
    auto alignmentInt = alignmentAttr.getValue().second.dyn_cast<IntegerAttr>();
    if (!alignmentInt)
      return parser.emitError(parser.getNameLoc(),
                              "expected integer alignment");
    if (alignmentInt.getValue().isNullValue())
      result.attributes.erase("alignment");
  }

  // Extract the result type from the trailing function type.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser.emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser.resolveOperand(arraySize, funcType.getInput(0), result.operands))
    return failure();

  result.addTypes({funcType.getResult(0)});
  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::BrOp
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange>
BrOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

//===----------------------------------------------------------------------===//
// LLVM::CondBrOp
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange>
CondBrOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == 0 ? trueDestOperandsMutable() : falseDestOperandsMutable();
}

//===----------------------------------------------------------------------===//
// Builder, printer and parser for for LLVM::LoadOp.
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &builder, OperationState &result, Type t,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal) {
  result.addOperands(addr);
  result.addTypes(t);
  if (isVolatile)
    result.addAttribute(kVolatileAttrName, builder.getUnitAttr());
  if (isNonTemporal)
    result.addAttribute(kNonTemporalAttrName, builder.getUnitAttr());
  if (alignment != 0)
    result.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
}

static void printLoadOp(OpAsmPrinter &p, LoadOp &op) {
  p << op.getOperationName() << ' ';
  if (op.volatile_())
    p << "volatile ";
  p << op.addr();
  p.printOptionalAttrDict(op.getAttrs(), {kVolatileAttrName});
  p << " : " << op.addr().getType();
}

// Extract the pointee type from the LLVM pointer type wrapped in MLIR.  Return
// the resulting type wrapped in MLIR, or nullptr on error.
static Type getLoadStoreElementType(OpAsmParser &parser, Type type,
                                    llvm::SMLoc trailingTypeLoc) {
  auto llvmTy = type.dyn_cast<LLVM::LLVMType>();
  if (!llvmTy)
    return parser.emitError(trailingTypeLoc, "expected LLVM IR dialect type"),
           nullptr;
  if (!llvmTy.isPointerTy())
    return parser.emitError(trailingTypeLoc, "expected LLVM pointer type"),
           nullptr;
  return llvmTy.getPointerElementTy();
}

// <operation> ::= `llvm.load` `volatile` ssa-use attribute-dict? `:` type
static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType addr;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (succeeded(parser.parseOptionalKeyword("volatile")))
    result.addAttribute(kVolatileAttrName, parser.getBuilder().getUnitAttr());

  if (parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);

  result.addTypes(elemTy);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and parser for LLVM::StoreOp.
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &builder, OperationState &result, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal) {
  result.addOperands({value, addr});
  result.addTypes({});
  if (isVolatile)
    result.addAttribute(kVolatileAttrName, builder.getUnitAttr());
  if (isNonTemporal)
    result.addAttribute(kNonTemporalAttrName, builder.getUnitAttr());
  if (alignment != 0)
    result.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
}

static void printStoreOp(OpAsmPrinter &p, StoreOp &op) {
  p << op.getOperationName() << ' ';
  if (op.volatile_())
    p << "volatile ";
  p << op.value() << ", " << op.addr();
  p.printOptionalAttrDict(op.getAttrs(), {kVolatileAttrName});
  p << " : " << op.addr().getType();
}

// <operation> ::= `llvm.store` `volatile` ssa-use `,` ssa-use
//                 attribute-dict? `:` type
static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType addr, value;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (succeeded(parser.parseOptionalKeyword("volatile")))
    result.addAttribute(kVolatileAttrName, parser.getBuilder().getUnitAttr());

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);
  if (!elemTy)
    return failure();

  if (parser.resolveOperand(value, elemTy, result.operands) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  return success();
}

///===---------------------------------------------------------------------===//
/// LLVM::InvokeOp
///===---------------------------------------------------------------------===//

Optional<MutableOperandRange>
InvokeOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == 0 ? normalDestOperandsMutable() : unwindDestOperandsMutable();
}

static LogicalResult verify(InvokeOp op) {
  if (op.getNumResults() > 1)
    return op.emitOpError("must have 0 or 1 result");

  Block *unwindDest = op.unwindDest();
  if (unwindDest->empty())
    return op.emitError(
        "must have at least one operation in unwind destination");

  // In unwind destination, first operation must be LandingpadOp
  if (!isa<LandingpadOp>(unwindDest->front()))
    return op.emitError("first operation in unwind destination should be a "
                        "llvm.landingpad operation");

  return success();
}

static void printInvokeOp(OpAsmPrinter &p, InvokeOp op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();

  p << op.getOperationName() << ' ';

  // Either function name or pointer
  if (isDirect)
    p.printSymbolName(callee.getValue());
  else
    p << op.getOperand(0);

  p << '(' << op.getOperands().drop_front(isDirect ? 0 : 1) << ')';
  p << " to ";
  p.printSuccessorAndUseList(op.normalDest(), op.normalDestOperands());
  p << " unwind ";
  p.printSuccessorAndUseList(op.unwindDest(), op.unwindDestOperands());

  p.printOptionalAttrDict(op.getAttrs(),
                          {InvokeOp::getOperandSegmentSizeAttr(), "callee"});
  p << " : ";
  p.printFunctionalType(
      llvm::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1),
      op.getResultTypes());
}

/// <operation> ::= `llvm.invoke` (function-id | ssa-use) `(` ssa-use-list `)`
///                  `to` bb-id (`[` ssa-use-and-type-list `]`)?
///                  `unwind` bb-id (`[` ssa-use-and-type-list `]`)?
///                  attribute-dict? `:` function-type
static ParseResult parseInvokeOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> operands;
  FunctionType funcType;
  SymbolRefAttr funcAttr;
  llvm::SMLoc trailingTypeLoc;
  Block *normalDest, *unwindDest;
  SmallVector<Value, 4> normalOperands, unwindOperands;
  Builder &builder = parser.getBuilder();

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser.parseOperandList(operands))
    return failure();
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect && parser.parseAttribute(funcAttr, "callee", result.attributes))
    return failure();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseSuccessorAndUseList(normalDest, normalOperands) ||
      parser.parseKeyword("unwind") ||
      parser.parseSuccessorAndUseList(unwindDest, unwindOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(funcType))
    return failure();

  if (isDirect) {
    // Make sure types match.
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();
    result.addTypes(funcType.getResults());
  } else {
    // Construct the LLVM IR Dialect function type that the first operand
    // should match.
    if (funcType.getNumResults() > 1)
      return parser.emitError(trailingTypeLoc,
                              "expected function with 0 or 1 result");

    LLVM::LLVMType llvmResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = LLVM::LLVMType::getVoidTy(builder.getContext());
    } else {
      llvmResultType = funcType.getResult(0).dyn_cast<LLVM::LLVMType>();
      if (!llvmResultType)
        return parser.emitError(trailingTypeLoc,
                                "expected result to have LLVM type");
    }

    SmallVector<LLVM::LLVMType, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (Type ty : funcType.getInputs()) {
      if (auto argType = ty.dyn_cast<LLVM::LLVMType>())
        argTypes.push_back(argType);
      else
        return parser.emitError(trailingTypeLoc,
                                "expected LLVM types as inputs");
    }

    auto llvmFuncType = LLVM::LLVMType::getFunctionTy(llvmResultType, argTypes,
                                                      /*isVarArg=*/false);
    auto wrappedFuncType = llvmFuncType.getPointerTo();

    auto funcArguments = llvm::makeArrayRef(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser.resolveOperand(operands[0], wrappedFuncType, result.operands) ||
        parser.resolveOperands(funcArguments, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    result.addTypes(llvmResultType);
  }
  result.addSuccessors({normalDest, unwindDest});
  result.addOperands(normalOperands);
  result.addOperands(unwindOperands);

  result.addAttribute(
      InvokeOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(operands.size()),
                                static_cast<int32_t>(normalOperands.size()),
                                static_cast<int32_t>(unwindOperands.size())}));
  return success();
}

///===----------------------------------------------------------------------===//
/// Verifying/Printing/Parsing for LLVM::LandingpadOp.
///===----------------------------------------------------------------------===//

static LogicalResult verify(LandingpadOp op) {
  Value value;
  if (LLVMFuncOp func = op.getParentOfType<LLVMFuncOp>()) {
    if (!func.personality().hasValue())
      return op.emitError(
          "llvm.landingpad needs to be in a function with a personality");
  }

  if (!op.cleanup() && op.getOperands().empty())
    return op.emitError("landingpad instruction expects at least one clause or "
                        "cleanup attribute");

  for (unsigned idx = 0, ie = op.getNumOperands(); idx < ie; idx++) {
    value = op.getOperand(idx);
    bool isFilter = value.getType().cast<LLVMType>().isArrayTy();
    if (isFilter) {
      // FIXME: Verify filter clauses when arrays are appropriately handled
    } else {
      // catch - global addresses only.
      // Bitcast ops should have global addresses as their args.
      if (auto bcOp = value.getDefiningOp<BitcastOp>()) {
        if (auto addrOp = bcOp.arg().getDefiningOp<AddressOfOp>())
          continue;
        return op.emitError("constant clauses expected")
                   .attachNote(bcOp.getLoc())
               << "global addresses expected as operand to "
                  "bitcast used in clauses for landingpad";
      }
      // NullOp and AddressOfOp allowed
      if (value.getDefiningOp<NullOp>())
        continue;
      if (value.getDefiningOp<AddressOfOp>())
        continue;
      return op.emitError("clause #")
             << idx << " is not a known constant - null, addressof, bitcast";
    }
  }
  return success();
}

static void printLandingpadOp(OpAsmPrinter &p, LandingpadOp &op) {
  p << op.getOperationName() << (op.cleanup() ? " cleanup " : " ");

  // Clauses
  for (auto value : op.getOperands()) {
    // Similar to llvm - if clause is an array type then it is filter
    // clause else catch clause
    bool isArrayTy = value.getType().cast<LLVMType>().isArrayTy();
    p << '(' << (isArrayTy ? "filter " : "catch ") << value << " : "
      << value.getType() << ") ";
  }

  p.printOptionalAttrDict(op.getAttrs(), {"cleanup"});

  p << ": " << op.getType();
}

/// <operation> ::= `llvm.landingpad` `cleanup`?
///                 ((`catch` | `filter`) operand-type ssa-use)* attribute-dict?
static ParseResult parseLandingpadOp(OpAsmParser &parser,
                                     OperationState &result) {
  // Check for cleanup
  if (succeeded(parser.parseOptionalKeyword("cleanup")))
    result.addAttribute("cleanup", parser.getBuilder().getUnitAttr());

  // Parse clauses with types
  while (succeeded(parser.parseOptionalLParen()) &&
         (succeeded(parser.parseOptionalKeyword("filter")) ||
          succeeded(parser.parseOptionalKeyword("catch")))) {
    OpAsmParser::OperandType operand;
    Type ty;
    if (parser.parseOperand(operand) || parser.parseColon() ||
        parser.parseType(ty) ||
        parser.resolveOperand(operand, ty, result.operands) ||
        parser.parseRParen())
      return failure();
  }

  Type type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifying/Printing/parsing for LLVM::CallOp.
//===----------------------------------------------------------------------===//

static LogicalResult verify(CallOp &op) {
  if (op.getNumResults() > 1)
    return op.emitOpError("must have 0 or 1 result");

  // Type for the callee, we'll get it differently depending if it is a direct
  // or indirect call.
  LLVMType fnType;

  bool isIndirect = false;

  // If this is an indirect call, the callee attribute is missing.
  Optional<StringRef> calleeName = op.callee();
  if (!calleeName) {
    isIndirect = true;
    if (!op.getNumOperands())
      return op.emitOpError(
          "must have either a `callee` attribute or at least an operand");
    fnType = op.getOperand(0).getType().dyn_cast<LLVMType>();
    if (!fnType)
      return op.emitOpError("indirect call to a non-llvm type: ")
             << op.getOperand(0).getType();
    auto ptrType = fnType.dyn_cast<LLVMPointerType>();
    if (!ptrType)
      return op.emitOpError("indirect call expects a pointer as callee: ")
             << fnType;
    fnType = ptrType.getElementType();
  } else {
    Operation *callee = SymbolTable::lookupNearestSymbolFrom(op, *calleeName);
    if (!callee)
      return op.emitOpError()
             << "'" << *calleeName
             << "' does not reference a symbol in the current scope";
    auto fn = dyn_cast<LLVMFuncOp>(callee);
    if (!fn)
      return op.emitOpError() << "'" << *calleeName
                              << "' does not reference a valid LLVM function";

    fnType = fn.getType();
  }
  if (!fnType.isFunctionTy())
    return op.emitOpError("callee does not have a functional type: ") << fnType;

  // Verify that the operand and result types match the callee.

  if (!fnType.isFunctionVarArg() &&
      fnType.getFunctionNumParams() != (op.getNumOperands() - isIndirect))
    return op.emitOpError()
           << "incorrect number of operands ("
           << (op.getNumOperands() - isIndirect)
           << ") for callee (expecting: " << fnType.getFunctionNumParams()
           << ")";

  if (fnType.getFunctionNumParams() > (op.getNumOperands() - isIndirect))
    return op.emitOpError() << "incorrect number of operands ("
                            << (op.getNumOperands() - isIndirect)
                            << ") for varargs callee (expecting at least: "
                            << fnType.getFunctionNumParams() << ")";

  for (unsigned i = 0, e = fnType.getFunctionNumParams(); i != e; ++i)
    if (op.getOperand(i + isIndirect).getType() !=
        fnType.getFunctionParamType(i))
      return op.emitOpError() << "operand type mismatch for operand " << i
                              << ": " << op.getOperand(i + isIndirect).getType()
                              << " != " << fnType.getFunctionParamType(i);

  if (op.getNumResults() &&
      op.getResult(0).getType() != fnType.getFunctionResultType())
    return op.emitOpError()
           << "result type mismatch: " << op.getResult(0).getType()
           << " != " << fnType.getFunctionResultType();

  return success();
}

static void printCallOp(OpAsmPrinter &p, CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  p << op.getOperationName() << ' ';
  if (isDirect)
    p.printSymbolName(callee.getValue());
  else
    p << op.getOperand(0);

  auto args = op.getOperands().drop_front(isDirect ? 0 : 1);
  p << '(' << args << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});

  // Reconstruct the function MLIR function type from operand and result types.
  p << " : "
    << FunctionType::get(args.getTypes(), op.getResultTypes(), op.getContext());
}

// <operation> ::= `llvm.call` (function-id | ssa-use) `(` ssa-use-list `)`
//                 attribute-dict? `:` function-type
static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> operands;
  Type type;
  SymbolRefAttr funcAttr;
  llvm::SMLoc trailingTypeLoc;

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser.parseOperandList(operands))
    return failure();
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", result.attributes))
      return failure();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType)
    return parser.emitError(trailingTypeLoc, "expected function type");
  if (isDirect) {
    // Make sure types match.
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();
    result.addTypes(funcType.getResults());
  } else {
    // Construct the LLVM IR Dialect function type that the first operand
    // should match.
    if (funcType.getNumResults() > 1)
      return parser.emitError(trailingTypeLoc,
                              "expected function with 0 or 1 result");

    Builder &builder = parser.getBuilder();
    LLVM::LLVMType llvmResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = LLVM::LLVMType::getVoidTy(builder.getContext());
    } else {
      llvmResultType = funcType.getResult(0).dyn_cast<LLVM::LLVMType>();
      if (!llvmResultType)
        return parser.emitError(trailingTypeLoc,
                                "expected result to have LLVM type");
    }

    SmallVector<LLVM::LLVMType, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      auto argType = funcType.getInput(i).dyn_cast<LLVM::LLVMType>();
      if (!argType)
        return parser.emitError(trailingTypeLoc,
                                "expected LLVM types as inputs");
      argTypes.push_back(argType);
    }
    auto llvmFuncType = LLVM::LLVMType::getFunctionTy(llvmResultType, argTypes,
                                                      /*isVarArg=*/false);
    auto wrappedFuncType = llvmFuncType.getPointerTo();

    auto funcArguments =
        ArrayRef<OpAsmParser::OperandType>(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser.resolveOperand(operands[0], wrappedFuncType, result.operands) ||
        parser.resolveOperands(funcArguments, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    result.addTypes(llvmResultType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractElementOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ExtractElementOp::build(OpBuilder &b, OperationState &result,
                                   Value vector, Value position,
                                   ArrayRef<NamedAttribute> attrs) {
  auto wrappedVectorType = vector.getType().cast<LLVM::LLVMType>();
  auto llvmType = wrappedVectorType.getVectorElementType();
  build(b, result, llvmType, vector, position);
  result.addAttributes(attrs);
}

static void printExtractElementOp(OpAsmPrinter &p, ExtractElementOp &op) {
  p << op.getOperationName() << ' ' << op.vector() << "[" << op.position()
    << " : " << op.position().getType() << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector().getType();
}

// <operation> ::= `llvm.extractelement` ssa-use `, ` ssa-use
//                 attribute-dict? `:` type
static ParseResult parseExtractElementOp(OpAsmParser &parser,
                                         OperationState &result) {
  llvm::SMLoc loc;
  OpAsmParser::OperandType vector, position;
  Type type, positionType;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(vector) ||
      parser.parseLSquare() || parser.parseOperand(position) ||
      parser.parseColonType(positionType) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(vector, type, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands))
    return failure();
  auto wrappedVectorType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedVectorType || !wrappedVectorType.isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  result.addTypes(wrappedVectorType.getVectorElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractValueOp.
//===----------------------------------------------------------------------===//

static void printExtractValueOp(OpAsmPrinter &p, ExtractValueOp &op) {
  p << op.getOperationName() << ' ' << op.container() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.container().getType();
}

// Extract the type at `position` in the wrapped LLVM IR aggregate type
// `containerType`.  Position is an integer array attribute where each value
// is a zero-based position of the element in the aggregate type.  Return the
// resulting type wrapped in MLIR, or nullptr on error.
static LLVM::LLVMType getInsertExtractValueElementType(OpAsmParser &parser,
                                                       Type containerType,
                                                       ArrayAttr positionAttr,
                                                       llvm::SMLoc attributeLoc,
                                                       llvm::SMLoc typeLoc) {
  auto wrappedContainerType = containerType.dyn_cast<LLVM::LLVMType>();
  if (!wrappedContainerType)
    return parser.emitError(typeLoc, "expected LLVM IR Dialect type"), nullptr;

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // structures.  Check the position index before accessing, it is supposed to
  // be in bounds.
  for (Attribute subAttr : positionAttr) {
    auto positionElementAttr = subAttr.dyn_cast<IntegerAttr>();
    if (!positionElementAttr)
      return parser.emitError(attributeLoc,
                              "expected an array of integer literals"),
             nullptr;
    int position = positionElementAttr.getInt();
    if (wrappedContainerType.isArrayTy()) {
      if (position < 0 || static_cast<unsigned>(position) >=
                              wrappedContainerType.getArrayNumElements())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      wrappedContainerType = wrappedContainerType.getArrayElementType();
    } else if (wrappedContainerType.isStructTy()) {
      if (position < 0 || static_cast<unsigned>(position) >=
                              wrappedContainerType.getStructNumElements())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      wrappedContainerType =
          wrappedContainerType.getStructElementType(position);
    } else {
      return parser.emitError(typeLoc,
                              "expected wrapped LLVM IR structure/array type"),
             nullptr;
    }
  }
  return wrappedContainerType;
}

// <operation> ::= `llvm.extractvalue` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseExtractValueOp(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::OperandType container;
  Type containerType;
  ArrayAttr positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType) ||
      parser.resolveOperand(container, containerType, result.operands))
    return failure();

  auto elementType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!elementType)
    return failure();

  result.addTypes(elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertElementOp.
//===----------------------------------------------------------------------===//

static void printInsertElementOp(OpAsmPrinter &p, InsertElementOp &op) {
  p << op.getOperationName() << ' ' << op.value() << ", " << op.vector() << "["
    << op.position() << " : " << op.position().getType() << "]";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector().getType();
}

// <operation> ::= `llvm.insertelement` ssa-use `,` ssa-use `,` ssa-use
//                 attribute-dict? `:` type
static ParseResult parseInsertElementOp(OpAsmParser &parser,
                                        OperationState &result) {
  llvm::SMLoc loc;
  OpAsmParser::OperandType vector, value, position;
  Type vectorType, positionType;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(value) ||
      parser.parseComma() || parser.parseOperand(vector) ||
      parser.parseLSquare() || parser.parseOperand(position) ||
      parser.parseColonType(positionType) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(vectorType))
    return failure();

  auto wrappedVectorType = vectorType.dyn_cast<LLVM::LLVMType>();
  if (!wrappedVectorType || !wrappedVectorType.isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  auto valueType = wrappedVectorType.getVectorElementType();
  if (!valueType)
    return failure();

  if (parser.resolveOperand(vector, vectorType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands))
    return failure();

  result.addTypes(vectorType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertValueOp.
//===----------------------------------------------------------------------===//

static void printInsertValueOp(OpAsmPrinter &p, InsertValueOp &op) {
  p << op.getOperationName() << ' ' << op.value() << ", " << op.container()
    << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.container().getType();
}

// <operation> ::= `llvm.insertvaluevalue` ssa-use `,` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseInsertValueOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType container, value;
  Type containerType;
  ArrayAttr positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType))
    return failure();

  auto valueType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!valueType)
    return failure();

  if (parser.resolveOperand(container, containerType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands))
    return failure();

  result.addTypes(containerType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ReturnOp.
//===----------------------------------------------------------------------===//

static void printReturnOp(OpAsmPrinter &p, ReturnOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  assert(op.getNumOperands() <= 1);

  if (op.getNumOperands() == 0)
    return;

  p << ' ' << op.getOperand(0) << " : " << op.getOperand(0).getType();
}

// <operation> ::= `llvm.return` ssa-use-list attribute-dict? `:`
//                 type-list-no-parens
static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 1> operands;
  Type type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (operands.empty())
    return success();

  if (parser.parseColonType(type) ||
      parser.resolveOperand(operands[0], type, result.operands))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for LLVM::AddressOfOp.
//===----------------------------------------------------------------------===//

template <typename OpTy>
static OpTy lookupSymbolInModule(Operation *parent, StringRef name) {
  Operation *module = parent;
  while (module && !satisfiesLLVMModule(module))
    module = module->getParentOp();
  assert(module && "unexpected operation outside of a module");
  return dyn_cast_or_null<OpTy>(
      mlir::SymbolTable::lookupSymbolIn(module, name));
}

GlobalOp AddressOfOp::getGlobal() {
  return lookupSymbolInModule<LLVM::GlobalOp>(getParentOp(), global_name());
}

LLVMFuncOp AddressOfOp::getFunction() {
  return lookupSymbolInModule<LLVM::LLVMFuncOp>(getParentOp(), global_name());
}

static LogicalResult verify(AddressOfOp op) {
  auto global = op.getGlobal();
  auto function = op.getFunction();
  if (!global && !function)
    return op.emitOpError(
        "must reference a global defined by 'llvm.mlir.global' or 'llvm.func'");

  if (global && global.getType().getPointerTo(global.addr_space()) !=
                    op.getResult().getType())
    return op.emitOpError(
        "the type must be a pointer to the type of the referenced global");

  if (function && function.getType().getPointerTo() != op.getResult().getType())
    return op.emitOpError(
        "the type must be a pointer to the type of the referenced function");

  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::GlobalOp.
//===----------------------------------------------------------------------===//

/// Returns the name used for the linkage attribute. This *must* correspond to
/// the name of the attribute in ODS.
static StringRef getLinkageAttrName() { return "linkage"; }

void GlobalOp::build(OpBuilder &builder, OperationState &result, LLVMType type,
                     bool isConstant, Linkage linkage, StringRef name,
                     Attribute value, unsigned addrSpace,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  if (isConstant)
    result.addAttribute("constant", builder.getUnitAttr());
  if (value)
    result.addAttribute("value", value);
  result.addAttribute(getLinkageAttrName(),
                      builder.getI64IntegerAttr(static_cast<int64_t>(linkage)));
  if (addrSpace != 0)
    result.addAttribute("addr_space", builder.getI32IntegerAttr(addrSpace));
  result.attributes.append(attrs.begin(), attrs.end());
  result.addRegion();
}

static void printGlobalOp(OpAsmPrinter &p, GlobalOp op) {
  p << op.getOperationName() << ' ' << stringifyLinkage(op.linkage()) << ' ';
  if (op.constant())
    p << "constant ";
  p.printSymbolName(op.sym_name());
  p << '(';
  if (auto value = op.getValueOrNull())
    p.printAttribute(value);
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(),
                          {SymbolTable::getSymbolAttrName(), "type", "constant",
                           "value", getLinkageAttrName()});

  // Print the trailing type unless it's a string global.
  if (op.getValueOrNull().dyn_cast_or_null<StringAttr>())
    return;
  p << " : " << op.type();

  Region &initializer = op.getInitializerRegion();
  if (!initializer.empty())
    p.printRegion(initializer, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// Verifier for LLVM::DialectCastOp.
//===----------------------------------------------------------------------===//

static LogicalResult verify(DialectCastOp op) {
  auto verifyMLIRCastType = [&op](Type type) -> LogicalResult {
    if (auto llvmType = type.dyn_cast<LLVM::LLVMType>()) {
      if (llvmType.isVectorTy())
        llvmType = llvmType.getVectorElementType();
      if (llvmType.isIntegerTy() || llvmType.isBFloatTy() ||
          llvmType.isHalfTy() || llvmType.isFloatTy() ||
          llvmType.isDoubleTy()) {
        return success();
      }
      return op.emitOpError("type must be non-index integer types, float "
                            "types, or vector of mentioned types.");
    }
    if (auto vectorType = type.dyn_cast<VectorType>()) {
      if (vectorType.getShape().size() > 1)
        return op.emitOpError("only 1-d vector is allowed");
      type = vectorType.getElementType();
    }
    if (type.isSignlessIntOrFloat())
      return success();
    // Note that memrefs are not supported. We currently don't have a use case
    // for it, but even if we do, there are challenges:
    // * if we allow memrefs to cast from/to memref descriptors, then the
    // semantics of the cast op depends on the implementation detail of the
    // descriptor.
    // * if we allow memrefs to cast from/to bare pointers, some users might
    // alternatively want metadata that only present in the descriptor.
    //
    // TODO: re-evaluate the memref cast design when it's needed.
    return op.emitOpError("type must be non-index integer types, float types, "
                          "or vector of mentioned types.");
  };
  return failure(failed(verifyMLIRCastType(op.in().getType())) ||
                 failed(verifyMLIRCastType(op.getType())));
}

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
static int parseOptionalKeywordAlternative(OpAsmParser &parser,
                                           ArrayRef<StringRef> keywords) {
  for (auto en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

namespace {
template <typename Ty> struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <> struct EnumTraits<Ty> {                                          \
    static StringRef stringify(Ty value) { return stringify##Ty(value); }      \
    static unsigned getMaxEnumVal() { return getMaxEnumValFor##Ty(); }         \
  }

REGISTER_ENUM_TYPE(Linkage);
} // end namespace

template <typename EnumTy>
static ParseResult parseOptionalLLVMKeyword(OpAsmParser &parser,
                                            OperationState &result,
                                            StringRef name) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = getMaxEnumValForLinkage(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return failure();
  result.addAttribute(name, parser.getBuilder().getI64IntegerAttr(index));
  return success();
}

// operation ::= `llvm.mlir.global` linkage? `constant`? `@` identifier
//               `(` attribute? `)` attribute-list? (`:` type)? region?
//
// The type can be omitted for string attributes, in which case it will be
// inferred from the value of the string as [strlen(value) x i8].
static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  if (failed(parseOptionalLLVMKeyword<Linkage>(parser, result,
                                               getLinkageAttrName())))
    result.addAttribute(getLinkageAttrName(),
                        parser.getBuilder().getI64IntegerAttr(
                            static_cast<int64_t>(LLVM::Linkage::External)));

  if (succeeded(parser.parseOptionalKeyword("constant")))
    result.addAttribute("constant", parser.getBuilder().getUnitAttr());

  StringAttr name;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLParen())
    return failure();

  Attribute value;
  if (parser.parseOptionalRParen()) {
    if (parser.parseAttribute(value, "value", result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (types.empty()) {
    if (auto strAttr = value.dyn_cast_or_null<StringAttr>()) {
      MLIRContext *context = parser.getBuilder().getContext();
      auto arrayType = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(context), strAttr.getValue().size());
      types.push_back(arrayType);
    } else {
      return parser.emitError(parser.getNameLoc(),
                              "type can only be omitted for string globals");
    }
  } else if (parser.parseOptionalRegion(initRegion, /*arguments=*/{},
                                        /*argTypes=*/{})) {
    return failure();
  }

  result.addAttribute("type", TypeAttr::get(types[0]));
  return success();
}

static LogicalResult verify(GlobalOp op) {
  if (!LLVMPointerType::isValidElementType(op.getType()))
    return op.emitOpError(
        "expects type to be a valid element type for an LLVM pointer");
  if (op.getParentOp() && !satisfiesLLVMModule(op.getParentOp()))
    return op.emitOpError("must appear at the module level");

  if (auto strAttr = op.getValueOrNull().dyn_cast_or_null<StringAttr>()) {
    auto type = op.getType();
    if (!type.isArrayTy() || !type.getArrayElementType().isIntegerTy(8) ||
        type.getArrayNumElements() != strAttr.getValue().size())
      return op.emitOpError(
          "requires an i8 array type of the length equal to that of the string "
          "attribute");
  }

  if (Block *b = op.getInitializerBlock()) {
    ReturnOp ret = cast<ReturnOp>(b->getTerminator());
    if (ret.operand_type_begin() == ret.operand_type_end())
      return op.emitOpError("initializer region cannot return void");
    if (*ret.operand_type_begin() != op.getType())
      return op.emitOpError("initializer region type ")
             << *ret.operand_type_begin() << " does not match global type "
             << op.getType();

    if (op.getValueOrNull())
      return op.emitOpError("cannot have both initializer value and region");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ShuffleVectorOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ShuffleVectorOp::build(OpBuilder &b, OperationState &result,
                                  Value v1, Value v2, ArrayAttr mask,
                                  ArrayRef<NamedAttribute> attrs) {
  auto wrappedContainerType1 = v1.getType().cast<LLVM::LLVMType>();
  auto vType = LLVMType::getVectorTy(
      wrappedContainerType1.getVectorElementType(), mask.size());
  build(b, result, vType, v1, v2, mask);
  result.addAttributes(attrs);
}

static void printShuffleVectorOp(OpAsmPrinter &p, ShuffleVectorOp &op) {
  p << op.getOperationName() << ' ' << op.v1() << ", " << op.v2() << " "
    << op.mask();
  p.printOptionalAttrDict(op.getAttrs(), {"mask"});
  p << " : " << op.v1().getType() << ", " << op.v2().getType();
}

// <operation> ::= `llvm.shufflevector` ssa-use `, ` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseShuffleVectorOp(OpAsmParser &parser,
                                        OperationState &result) {
  llvm::SMLoc loc;
  OpAsmParser::OperandType v1, v2;
  ArrayAttr maskAttr;
  Type typeV1, typeV2;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(v1) ||
      parser.parseComma() || parser.parseOperand(v2) ||
      parser.parseAttribute(maskAttr, "mask", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(typeV1) || parser.parseComma() ||
      parser.parseType(typeV2) ||
      parser.resolveOperand(v1, typeV1, result.operands) ||
      parser.resolveOperand(v2, typeV2, result.operands))
    return failure();
  auto wrappedContainerType1 = typeV1.dyn_cast<LLVM::LLVMType>();
  if (!wrappedContainerType1 || !wrappedContainerType1.isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  auto vType = LLVMType::getVectorTy(
      wrappedContainerType1.getVectorElementType(), maskAttr.size());
  result.addTypes(vType);
  return success();
}

//===----------------------------------------------------------------------===//
// Implementations for LLVM::LLVMFuncOp.
//===----------------------------------------------------------------------===//

// Add the entry block to the function.
Block *LLVMFuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  assert(!isVarArg() && "unimplemented: non-external variadic functions");

  auto *entry = new Block;
  push_back(entry);

  LLVMType type = getType();
  for (unsigned i = 0, e = type.getFunctionNumParams(); i < e; ++i)
    entry->addArgument(type.getFunctionParamType(i));
  return entry;
}

void LLVMFuncOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, LLVMType type, LLVM::Linkage linkage,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<MutableDictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.addAttribute(getLinkageAttrName(),
                      builder.getI64IntegerAttr(static_cast<int64_t>(linkage)));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty())
    return;

  unsigned numInputs = type.getFunctionNumParams();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0; i < numInputs; ++i)
    if (auto argDict = argAttrs[i].getDictionary(builder.getContext()))
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

// Builds an LLVM function type from the given lists of input and output types.
// Returns a null type if any of the types provided are non-LLVM types, or if
// there is more than one output type.
static Type buildLLVMFunctionType(OpAsmParser &parser, llvm::SMLoc loc,
                                  ArrayRef<Type> inputs, ArrayRef<Type> outputs,
                                  impl::VariadicFlag variadicFlag) {
  Builder &b = parser.getBuilder();
  if (outputs.size() > 1) {
    parser.emitError(loc, "failed to construct function type: expected zero or "
                          "one function result");
    return {};
  }

  // Convert inputs to LLVM types, exit early on error.
  SmallVector<LLVMType, 4> llvmInputs;
  for (auto t : inputs) {
    auto llvmTy = t.dyn_cast<LLVMType>();
    if (!llvmTy) {
      parser.emitError(loc, "failed to construct function type: expected LLVM "
                            "type for function arguments");
      return {};
    }
    llvmInputs.push_back(llvmTy);
  }

  // No output is denoted as "void" in LLVM type system.
  LLVMType llvmOutput = outputs.empty() ? LLVMType::getVoidTy(b.getContext())
                                        : outputs.front().dyn_cast<LLVMType>();
  if (!llvmOutput) {
    parser.emitError(loc, "failed to construct function type: expected LLVM "
                          "type for function results");
    return {};
  }
  return LLVMType::getFunctionTy(llvmOutput, llvmInputs,
                                 variadicFlag.isVariadic());
}

// Parses an LLVM function.
//
// operation ::= `llvm.func` linkage? function-signature function-attributes?
//               function-body
//
static ParseResult parseLLVMFuncOp(OpAsmParser &parser,
                                   OperationState &result) {
  // Default to external linkage if no keyword is provided.
  if (failed(parseOptionalLLVMKeyword<Linkage>(parser, result,
                                               getLinkageAttrName())))
    result.addAttribute(getLinkageAttrName(),
                        parser.getBuilder().getI64IntegerAttr(
                            static_cast<int64_t>(LLVM::Linkage::External)));

  StringAttr nameAttr;
  SmallVector<OpAsmParser::OperandType, 8> entryArgs;
  SmallVector<NamedAttrList, 1> argAttrs;
  SmallVector<NamedAttrList, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  bool isVariadic;

  auto signatureLocation = parser.getCurrentLocation();
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      impl::parseFunctionSignature(parser, /*allowVariadic=*/true, entryArgs,
                                   argTypes, argAttrs, isVariadic, resultTypes,
                                   resultAttrs))
    return failure();

  auto type =
      buildLLVMFunctionType(parser, signatureLocation, argTypes, resultTypes,
                            impl::VariadicFlag(isVariadic));
  if (!type)
    return failure();
  result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(type));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  impl::addArgAndResultAttrs(parser.getBuilder(), result, argAttrs,
                             resultAttrs);

  auto *body = result.addRegion();
  return parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes);
}

// Print the LLVMFuncOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back. Skips
// the external linkage since it is the default value.
static void printLLVMFuncOp(OpAsmPrinter &p, LLVMFuncOp op) {
  p << op.getOperationName() << ' ';
  if (op.linkage() != LLVM::Linkage::External)
    p << stringifyLinkage(op.linkage()) << ' ';
  p.printSymbolName(op.getName());

  LLVMType fnType = op.getType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 1> resTypes;
  argTypes.reserve(fnType.getFunctionNumParams());
  for (unsigned i = 0, e = fnType.getFunctionNumParams(); i < e; ++i)
    argTypes.push_back(fnType.getFunctionParamType(i));

  LLVMType returnType = fnType.getFunctionResultType();
  if (!returnType.isVoidTy())
    resTypes.push_back(returnType);

  impl::printFunctionSignature(p, op, argTypes, op.isVarArg(), resTypes);
  impl::printFunctionAttributes(p, op, argTypes.size(), resTypes.size(),
                                {getLinkageAttrName()});

  // Print the body if this is not an external function.
  Region &body = op.body();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present.  This can check for preconditions of the
// getNumArguments hook not failing.
LogicalResult LLVMFuncOp::verifyType() {
  auto llvmType = getTypeAttr().getValue().dyn_cast_or_null<LLVMType>();
  if (!llvmType || !llvmType.isFunctionTy())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of wrapped LLVM function type");

  return success();
}

// Hook for OpTrait::FunctionLike, returns the number of function arguments.
// Depends on the type attribute being correct as checked by verifyType
unsigned LLVMFuncOp::getNumFuncArguments() {
  return getType().getFunctionNumParams();
}

// Hook for OpTrait::FunctionLike, returns the number of function results.
// Depends on the type attribute being correct as checked by verifyType
unsigned LLVMFuncOp::getNumFuncResults() {
  // We model LLVM functions that return void as having zero results,
  // and all others as having one result.
  // If we modeled a void return as one result, then it would be possible to
  // attach an MLIR result attribute to it, and it isn't clear what semantics we
  // would assign to that.
  if (getType().getFunctionResultType().isVoidTy())
    return 0;
  return 1;
}

// Verifies LLVM- and implementation-specific properties of the LLVM func Op:
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage;
// - vararg is (currently) only supported for external functions;
// - entry block arguments are of LLVM types and match the function signature.
static LogicalResult verify(LLVMFuncOp op) {
  if (op.linkage() == LLVM::Linkage::Common)
    return op.emitOpError()
           << "functions cannot have '"
           << stringifyLinkage(LLVM::Linkage::Common) << "' linkage";

  if (op.isExternal()) {
    if (op.linkage() != LLVM::Linkage::External &&
        op.linkage() != LLVM::Linkage::ExternWeak)
      return op.emitOpError()
             << "external functions must have '"
             << stringifyLinkage(LLVM::Linkage::External) << "' or '"
             << stringifyLinkage(LLVM::Linkage::ExternWeak) << "' linkage";
    return success();
  }

  if (op.isVarArg())
    return op.emitOpError("only external functions can be variadic");

  unsigned numArguments = op.getType().getFunctionNumParams();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i < numArguments; ++i) {
    Type argType = entryBlock.getArgument(i).getType();
    auto argLLVMType = argType.dyn_cast<LLVMType>();
    if (!argLLVMType)
      return op.emitOpError("entry block argument #")
             << i << " is not of LLVM type";
    if (op.getType().getFunctionParamType(i) != argLLVMType)
      return op.emitOpError("the type of entry block argument #")
             << i << " does not match the function signature";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Verification for LLVM::NullOp.
//===----------------------------------------------------------------------===//

// Only LLVM pointer types are supported.
static LogicalResult verify(LLVM::NullOp op) {
  auto llvmType = op.getType().dyn_cast<LLVM::LLVMType>();
  if (!llvmType || !llvmType.isPointerTy())
    return op.emitOpError("expected LLVM IR pointer type");
  return success();
}

//===----------------------------------------------------------------------===//
// Verification for LLVM::ConstantOp.
//===----------------------------------------------------------------------===//

static LogicalResult verify(LLVM::ConstantOp op) {
  if (!(op.value().isa<IntegerAttr>() || op.value().isa<FloatAttr>() ||
        op.value().isa<ElementsAttr>() || op.value().isa<StringAttr>()))
    return op.emitOpError()
           << "only supports integer, float, string or elements attributes";
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions for parsing atomic ops
//===----------------------------------------------------------------------===//

// Helper function to parse a keyword into the specified attribute named by
// `attrName`. The keyword must match one of the string values defined by the
// AtomicBinOp enum. The resulting I64 attribute is added to the `result`
// state.
static ParseResult parseAtomicBinOp(OpAsmParser &parser, OperationState &result,
                                    StringRef attrName) {
  llvm::SMLoc loc;
  StringRef keyword;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
    return failure();

  // Replace the keyword `keyword` with an integer attribute.
  auto kind = symbolizeAtomicBinOp(keyword);
  if (!kind) {
    return parser.emitError(loc)
           << "'" << keyword << "' is an incorrect value of the '" << attrName
           << "' attribute";
  }

  auto value = static_cast<int64_t>(kind.getValue());
  auto attr = parser.getBuilder().getI64IntegerAttr(value);
  result.addAttribute(attrName, attr);

  return success();
}

// Helper function to parse a keyword into the specified attribute named by
// `attrName`. The keyword must match one of the string values defined by the
// AtomicOrdering enum. The resulting I64 attribute is added to the `result`
// state.
static ParseResult parseAtomicOrdering(OpAsmParser &parser,
                                       OperationState &result,
                                       StringRef attrName) {
  llvm::SMLoc loc;
  StringRef ordering;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&ordering))
    return failure();

  // Replace the keyword `ordering` with an integer attribute.
  auto kind = symbolizeAtomicOrdering(ordering);
  if (!kind) {
    return parser.emitError(loc)
           << "'" << ordering << "' is an incorrect value of the '" << attrName
           << "' attribute";
  }

  auto value = static_cast<int64_t>(kind.getValue());
  auto attr = parser.getBuilder().getI64IntegerAttr(value);
  result.addAttribute(attrName, attr);

  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::AtomicRMWOp.
//===----------------------------------------------------------------------===//

static void printAtomicRMWOp(OpAsmPrinter &p, AtomicRMWOp &op) {
  p << op.getOperationName() << ' ' << stringifyAtomicBinOp(op.bin_op()) << ' '
    << op.ptr() << ", " << op.val() << ' '
    << stringifyAtomicOrdering(op.ordering()) << ' ';
  p.printOptionalAttrDict(op.getAttrs(), {"bin_op", "ordering"});
  p << " : " << op.res().getType();
}

// <operation> ::= `llvm.atomicrmw` keyword ssa-use `,` ssa-use keyword
//                 attribute-dict? `:` type
static ParseResult parseAtomicRMWOp(OpAsmParser &parser,
                                    OperationState &result) {
  LLVMType type;
  OpAsmParser::OperandType ptr, val;
  if (parseAtomicBinOp(parser, result, "bin_op") || parser.parseOperand(ptr) ||
      parser.parseComma() || parser.parseOperand(val) ||
      parseAtomicOrdering(parser, result, "ordering") ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptr, type.getPointerTo(), result.operands) ||
      parser.resolveOperand(val, type, result.operands))
    return failure();

  result.addTypes(type);
  return success();
}

static LogicalResult verify(AtomicRMWOp op) {
  auto ptrType = op.ptr().getType().cast<LLVM::LLVMType>();
  auto valType = op.val().getType().cast<LLVM::LLVMType>();
  if (valType != ptrType.getPointerElementTy())
    return op.emitOpError("expected LLVM IR element type for operand #0 to "
                          "match type for operand #1");
  auto resType = op.res().getType().cast<LLVM::LLVMType>();
  if (resType != valType)
    return op.emitOpError(
        "expected LLVM IR result type to match type for operand #1");
  if (op.bin_op() == AtomicBinOp::fadd || op.bin_op() == AtomicBinOp::fsub) {
    if (!valType.isFloatingPointTy())
      return op.emitOpError("expected LLVM IR floating point type");
  } else if (op.bin_op() == AtomicBinOp::xchg) {
    if (!valType.isIntegerTy(8) && !valType.isIntegerTy(16) &&
        !valType.isIntegerTy(32) && !valType.isIntegerTy(64) &&
        !valType.isBFloatTy() && !valType.isHalfTy() && !valType.isFloatTy() &&
        !valType.isDoubleTy())
      return op.emitOpError("unexpected LLVM IR type for 'xchg' bin_op");
  } else {
    if (!valType.isIntegerTy(8) && !valType.isIntegerTy(16) &&
        !valType.isIntegerTy(32) && !valType.isIntegerTy(64))
      return op.emitOpError("expected LLVM IR integer type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::AtomicCmpXchgOp.
//===----------------------------------------------------------------------===//

static void printAtomicCmpXchgOp(OpAsmPrinter &p, AtomicCmpXchgOp &op) {
  p << op.getOperationName() << ' ' << op.ptr() << ", " << op.cmp() << ", "
    << op.val() << ' ' << stringifyAtomicOrdering(op.success_ordering()) << ' '
    << stringifyAtomicOrdering(op.failure_ordering());
  p.printOptionalAttrDict(op.getAttrs(),
                          {"success_ordering", "failure_ordering"});
  p << " : " << op.val().getType();
}

// <operation> ::= `llvm.cmpxchg` ssa-use `,` ssa-use `,` ssa-use
//                 keyword keyword attribute-dict? `:` type
static ParseResult parseAtomicCmpXchgOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto &builder = parser.getBuilder();
  LLVMType type;
  OpAsmParser::OperandType ptr, cmp, val;
  if (parser.parseOperand(ptr) || parser.parseComma() ||
      parser.parseOperand(cmp) || parser.parseComma() ||
      parser.parseOperand(val) ||
      parseAtomicOrdering(parser, result, "success_ordering") ||
      parseAtomicOrdering(parser, result, "failure_ordering") ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptr, type.getPointerTo(), result.operands) ||
      parser.resolveOperand(cmp, type, result.operands) ||
      parser.resolveOperand(val, type, result.operands))
    return failure();

  auto boolType = LLVMType::getInt1Ty(builder.getContext());
  auto resultType = LLVMType::getStructTy(type, boolType);
  result.addTypes(resultType);

  return success();
}

static LogicalResult verify(AtomicCmpXchgOp op) {
  auto ptrType = op.ptr().getType().cast<LLVM::LLVMType>();
  if (!ptrType.isPointerTy())
    return op.emitOpError("expected LLVM IR pointer type for operand #0");
  auto cmpType = op.cmp().getType().cast<LLVM::LLVMType>();
  auto valType = op.val().getType().cast<LLVM::LLVMType>();
  if (cmpType != ptrType.getPointerElementTy() || cmpType != valType)
    return op.emitOpError("expected LLVM IR element type for operand #0 to "
                          "match type for all other operands");
  if (!valType.isPointerTy() && !valType.isIntegerTy(8) &&
      !valType.isIntegerTy(16) && !valType.isIntegerTy(32) &&
      !valType.isIntegerTy(64) && !valType.isBFloatTy() &&
      !valType.isHalfTy() && !valType.isFloatTy() && !valType.isDoubleTy())
    return op.emitOpError("unexpected LLVM IR type");
  if (op.success_ordering() < AtomicOrdering::monotonic ||
      op.failure_ordering() < AtomicOrdering::monotonic)
    return op.emitOpError("ordering must be at least 'monotonic'");
  if (op.failure_ordering() == AtomicOrdering::release ||
      op.failure_ordering() == AtomicOrdering::acq_rel)
    return op.emitOpError("failure ordering cannot be 'release' or 'acq_rel'");
  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::FenceOp.
//===----------------------------------------------------------------------===//

// <operation> ::= `llvm.fence` (`syncscope(`strAttr`)`)? keyword
// attribute-dict?
static ParseResult parseFenceOp(OpAsmParser &parser, OperationState &result) {
  StringAttr sScope;
  StringRef syncscopeKeyword = "syncscope";
  if (!failed(parser.parseOptionalKeyword(syncscopeKeyword))) {
    if (parser.parseLParen() ||
        parser.parseAttribute(sScope, syncscopeKeyword, result.attributes) ||
        parser.parseRParen())
      return failure();
  } else {
    result.addAttribute(syncscopeKeyword,
                        parser.getBuilder().getStringAttr(""));
  }
  if (parseAtomicOrdering(parser, result, "ordering") ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void printFenceOp(OpAsmPrinter &p, FenceOp &op) {
  StringRef syncscopeKeyword = "syncscope";
  p << op.getOperationName() << ' ';
  if (!op.getAttr(syncscopeKeyword).cast<StringAttr>().getValue().empty())
    p << "syncscope(" << op.getAttr(syncscopeKeyword) << ") ";
  p << stringifyAtomicOrdering(op.ordering());
}

static LogicalResult verify(FenceOp &op) {
  if (op.ordering() == AtomicOrdering::not_atomic ||
      op.ordering() == AtomicOrdering::unordered ||
      op.ordering() == AtomicOrdering::monotonic)
    return op.emitOpError("can be given only acquire, release, acq_rel, "
                          "and seq_cst orderings");
  return success();
}

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void LLVMDialect::initialize() {
  // clang-format off
  addTypes<LLVMVoidType,
           LLVMHalfType,
           LLVMBFloatType,
           LLVMFloatType,
           LLVMDoubleType,
           LLVMFP128Type,
           LLVMX86FP80Type,
           LLVMPPCFP128Type,
           LLVMX86MMXType,
           LLVMTokenType,
           LLVMLabelType,
           LLVMMetadataType,
           LLVMFunctionType,
           LLVMIntegerType,
           LLVMPointerType,
           LLVMFixedVectorType,
           LLVMScalableVectorType,
           LLVMArrayType,
           LLVMStructType>();
  // clang-format on
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"
      >();

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"

/// Parse a type registered to this dialect.
Type LLVMDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}

/// Print a type registered to this dialect.
void LLVMDialect::printType(Type type, DialectAsmPrinter &os) const {
  return detail::printType(type.cast<LLVMType>(), os);
}

LogicalResult LLVMDialect::verifyDataLayoutString(
    StringRef descr, llvm::function_ref<void(const Twine &)> reportError) {
  llvm::Expected<llvm::DataLayout> maybeDataLayout =
      llvm::DataLayout::parse(descr);
  if (maybeDataLayout)
    return success();

  std::string message;
  llvm::raw_string_ostream messageStream(message);
  llvm::logAllUnhandledErrors(maybeDataLayout.takeError(), messageStream);
  reportError("invalid data layout descriptor: " + messageStream.str());
  return failure();
}

/// Verify LLVM dialect attributes.
LogicalResult LLVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // If the data layout attribute is present, it must use the LLVM data layout
  // syntax. Try parsing it and report errors in case of failure. Users of this
  // attribute may assume it is well-formed and can pass it to the (asserting)
  // llvm::DataLayout constructor.
  if (attr.first.strref() != LLVM::LLVMDialect::getDataLayoutAttrName())
    return success();
  if (auto stringAttr = attr.second.dyn_cast<StringAttr>())
    return verifyDataLayoutString(
        stringAttr.getValue(),
        [op](const Twine &message) { op->emitOpError() << message.str(); });

  return op->emitOpError() << "expected '"
                           << LLVM::LLVMDialect::getDataLayoutAttrName()
                           << "' to be a string attribute";
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIdx,
                                                    unsigned argIdx,
                                                    NamedAttribute argAttr) {
  // Check that llvm.noalias is a boolean attribute.
  if (argAttr.first == LLVMDialect::getNoAliasAttrName() &&
      !argAttr.second.isa<BoolAttr>())
    return op->emitError()
           << "llvm.noalias argument attribute of non boolean type";
  // Check that llvm.align is an integer attribute.
  if (argAttr.first == LLVMDialect::getAlignAttrName() &&
      !argAttr.second.isa<IntegerAttr>())
    return op->emitError()
           << "llvm.align argument attribute of non integer type";
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

Value mlir::LLVM::createGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     LLVM::Linkage linkage) {
  assert(builder.getInsertionBlock() &&
         builder.getInsertionBlock()->getParentOp() &&
         "expected builder to point to a block constrained in an op");
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(module && "builder points to an op outside of a module");

  // Create the global at the entry of the module.
  OpBuilder moduleBuilder(module.getBodyRegion());
  MLIRContext *ctx = builder.getContext();
  auto type =
      LLVM::LLVMType::getArrayTy(LLVM::LLVMType::getInt8Ty(ctx), value.size());
  auto global = moduleBuilder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, linkage, name,
      builder.getStringAttr(value));

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(ctx),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMType::getInt8PtrTy(ctx),
                                     globalPtr, ValueRange{cst0, cst0});
}

bool mlir::LLVM::satisfiesLLVMModule(Operation *op) {
  return op->hasTrait<OpTrait::SymbolTable>() &&
         op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}
