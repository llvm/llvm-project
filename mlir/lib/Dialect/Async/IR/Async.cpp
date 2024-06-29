//===- Async.cpp - MLIR Async Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::async;

#include "mlir/Dialect/Async/IR/AsyncOpsDialect.cpp.inc"

constexpr StringRef AsyncDialect::kAllowedToBlockAttrName;

void AsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Async/IR/AsyncOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
/// ExecuteOp
//===----------------------------------------------------------------------===//

constexpr char kOperandSegmentSizesAttr[] = "operandSegmentSizes";

OperandRange ExecuteOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getBodyRegion() && "invalid region index");
  return getBodyOperands();
}

bool ExecuteOp::areTypesCompatible(Type lhs, Type rhs) {
  const auto getValueOrTokenType = [](Type type) {
    if (auto value = llvm::dyn_cast<ValueType>(type))
      return value.getValueType();
    return type;
  };
  return getValueOrTokenType(lhs) == getValueOrTokenType(rhs);
}

void ExecuteOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `body` region branch back to the parent operation.
  if (point == getBodyRegion()) {
    regions.push_back(RegionSuccessor(getBodyResults()));
    return;
  }

  // Otherwise the successor is the body region.
  regions.push_back(
      RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
}

void ExecuteOp::build(OpBuilder &builder, OperationState &result,
                      TypeRange resultTypes, ValueRange dependencies,
                      ValueRange operands, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operandSegmentSizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes =
      builder.getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `async.value`.
  result.addTypes({TokenType::get(result.getContext())});
  for (Type type : resultTypes)
    result.addTypes(ValueType::get(type));

  // Add a body region with block arguments as unwrapped async value operands.
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  for (Value operand : operands) {
    auto valueType = llvm::dyn_cast<ValueType>(operand.getType());
    bodyBlock->addArgument(valueType ? valueType.getValueType()
                                     : operand.getType(),
                           operand.getLoc());
  }

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (resultTypes.empty() && !bodyBuilder) {
    builder.create<async::YieldOp>(result.location, ValueRange());
  } else if (bodyBuilder) {
    bodyBuilder(builder, result.location, bodyBlock->getArguments());
  }
}

void ExecuteOp::print(OpAsmPrinter &p) {
  // [%tokens,...]
  if (!getDependencies().empty())
    p << " [" << getDependencies() << "]";

  // (%value as %unwrapped: !async.value<!arg.type>, ...)
  if (!getBodyOperands().empty()) {
    p << " (";
    Block *entry = getBodyRegion().empty() ? nullptr : &getBodyRegion().front();
    llvm::interleaveComma(
        getBodyOperands(), p, [&, n = 0](Value operand) mutable {
          Value argument = entry ? entry->getArgument(n++) : Value();
          p << operand << " as " << argument << ": " << operand.getType();
        });
    p << ")";
  }

  // -> (!async.value<!return.type>, ...)
  p.printOptionalArrowTypeList(llvm::drop_begin(getResultTypes()));
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {kOperandSegmentSizesAttr});
  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Sizes of parsed variadic operands, will be updated below after parsing.
  int32_t numDependencies = 0;

  auto tokenTy = TokenType::get(ctx);

  // Parse dependency tokens.
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> tokenArgs;
    if (parser.parseOperandList(tokenArgs) ||
        parser.resolveOperands(tokenArgs, tokenTy, result.operands) ||
        parser.parseRSquare())
      return failure();

    numDependencies = tokenArgs.size();
  }

  // Parse async value operands (%value as %unwrapped : !async.value<!type>).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> valueArgs;
  SmallVector<OpAsmParser::Argument, 4> unwrappedArgs;
  SmallVector<Type, 4> valueTypes;

  // Parse a single instance of `%value as %unwrapped : !async.value<!type>`.
  auto parseAsyncValueArg = [&]() -> ParseResult {
    if (parser.parseOperand(valueArgs.emplace_back()) ||
        parser.parseKeyword("as") ||
        parser.parseArgument(unwrappedArgs.emplace_back()) ||
        parser.parseColonType(valueTypes.emplace_back()))
      return failure();

    auto valueTy = llvm::dyn_cast<ValueType>(valueTypes.back());
    unwrappedArgs.back().type = valueTy ? valueTy.getValueType() : Type();
    return success();
  };

  auto argsLoc = parser.getCurrentLocation();
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::OptionalParen,
                                     parseAsyncValueArg) ||
      parser.resolveOperands(valueArgs, valueTypes, argsLoc, result.operands))
    return failure();

  int32_t numOperands = valueArgs.size();

  // Add derived `operandSegmentSizes` attribute based on parsed operands.
  auto operandSegmentSizes =
      parser.getBuilder().getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // Parse the types of results returned from the async execute op.
  SmallVector<Type, 4> resultTypes;
  NamedAttrList attrs;
  if (parser.parseOptionalArrowTypeList(resultTypes) ||
      // Async execute first result is always a completion token.
      parser.addTypeToList(tokenTy, result.types) ||
      parser.addTypesToList(resultTypes, result.types) ||
      // Parse operation attributes.
      parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();

  result.addAttributes(attrs);

  // Parse asynchronous region.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, /*arguments=*/unwrappedArgs);
}

LogicalResult ExecuteOp::verifyRegions() {
  // Unwrap async.execute value operands types.
  auto unwrappedTypes = llvm::map_range(getBodyOperands(), [](Value operand) {
    return llvm::cast<ValueType>(operand.getType()).getValueType();
  });

  // Verify that unwrapped argument types matches the body region arguments.
  if (getBodyRegion().getArgumentTypes() != unwrappedTypes)
    return emitOpError("async body region argument types do not match the "
                       "execute operation arguments types");

  return success();
}

//===----------------------------------------------------------------------===//
/// CreateGroupOp
//===----------------------------------------------------------------------===//

LogicalResult CreateGroupOp::canonicalize(CreateGroupOp op,
                                          PatternRewriter &rewriter) {
  // Find all `await_all` users of the group.
  llvm::SmallVector<AwaitAllOp> awaitAllUsers;

  auto isAwaitAll = [&](Operation *op) -> bool {
    if (AwaitAllOp awaitAll = dyn_cast<AwaitAllOp>(op)) {
      awaitAllUsers.push_back(awaitAll);
      return true;
    }
    return false;
  };

  // Check if all users of the group are `await_all` operations.
  if (!llvm::all_of(op->getUsers(), isAwaitAll))
    return failure();

  // If group is only awaited without adding anything to it, we can safely erase
  // the create operation and all users.
  for (AwaitAllOp awaitAll : awaitAllUsers)
    rewriter.eraseOp(awaitAll);
  rewriter.eraseOp(op);

  return success();
}

//===----------------------------------------------------------------------===//
/// AwaitOp
//===----------------------------------------------------------------------===//

void AwaitOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    ArrayRef<NamedAttribute> attrs) {
  result.addOperands({operand});
  result.attributes.append(attrs.begin(), attrs.end());

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = llvm::dyn_cast<ValueType>(operand.getType()))
    result.addTypes(valueType.getValueType());
}

static ParseResult parseAwaitResultType(OpAsmParser &parser, Type &operandType,
                                        Type &resultType) {
  if (parser.parseType(operandType))
    return failure();

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = llvm::dyn_cast<ValueType>(operandType))
    resultType = valueType.getValueType();

  return success();
}

static void printAwaitResultType(OpAsmPrinter &p, Operation *op,
                                 Type operandType, Type resultType) {
  p << operandType;
}

LogicalResult AwaitOp::verify() {
  Type argType = getOperand().getType();

  // Awaiting on a token does not have any results.
  if (llvm::isa<TokenType>(argType) && !getResultTypes().empty())
    return emitOpError("awaiting on a token must have empty result");

  // Awaiting on a value unwraps the async value type.
  if (auto value = llvm::dyn_cast<ValueType>(argType)) {
    if (*getResultType() != value.getValueType())
      return emitOpError() << "result type " << *getResultType()
                           << " does not match async value type "
                           << value.getValueType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Check that the result type of async.func is not void and must be
/// some async token or async values.
LogicalResult FuncOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();

  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i) {
    auto type = resultTypes[i];
    if (!llvm::isa<TokenType>(type) && !llvm::isa<ValueType>(type))
      return emitOpError() << "result type must be async value type or async "
                              "token type, but got "
                           << type;
    // We only allow AsyncToken appear as the first return value
    if (llvm::isa<TokenType>(type) && i != 0) {
      return emitOpError()
             << " results' (optional) async token type is expected "
                "to appear as the 1st return value, but got "
             << i + 1;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
/// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid async function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
/// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = (*this)->getParentOfType<FuncOp>();
  ArrayRef<Type> resultTypes = funcOp.isStateful()
                                   ? funcOp.getResultTypes().drop_front()
                                   : funcOp.getResultTypes();
  // Get the underlying value types from async types returned from the
  // parent `async.func` operation.
  auto types = llvm::map_range(resultTypes, [](const Type &result) {
    return llvm::cast<ValueType>(result).getValueType();
  });

  if (getOperandTypes() != types)
    return emitOpError("operand types do not match the types returned from "
                       "the parent FuncOp");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOpsTypes.cpp.inc"

void ValueType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getValueType());
  printer << '>';
}

Type ValueType::parse(mlir::AsmParser &parser) {
  Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "failed to parse async value type");
    return Type();
  }
  return ValueType::get(ty);
}
