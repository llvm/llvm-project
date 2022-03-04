//===- Async.cpp - MLIR Async Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/IR/DialectImplementation.h"
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
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // Get the underlying value types from async values returned from the
  // parent `async.execute` operation.
  auto executeOp = (*this)->getParentOfType<ExecuteOp>();
  auto types = llvm::map_range(executeOp.results(), [](const OpResult &result) {
    return result.getType().cast<ValueType>().getValueType();
  });

  if (getOperandTypes() != types)
    return emitOpError("operand types do not match the types returned from "
                       "the parent ExecuteOp");

  return success();
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(Optional<unsigned> index) {
  return operandsMutable();
}

//===----------------------------------------------------------------------===//
/// ExecuteOp
//===----------------------------------------------------------------------===//

constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

OperandRange ExecuteOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");
  return operands();
}

bool ExecuteOp::areTypesCompatible(Type lhs, Type rhs) {
  const auto getValueOrTokenType = [](Type type) {
    if (auto value = type.dyn_cast<ValueType>())
      return value.getValueType();
    return type;
  };
  return getValueOrTokenType(lhs) == getValueOrTokenType(rhs);
}

void ExecuteOp::getSuccessorRegions(Optional<unsigned> index,
                                    ArrayRef<Attribute>,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `body` region branch back to the parent operation.
  if (index.hasValue()) {
    assert(*index == 0 && "invalid region index");
    regions.push_back(RegionSuccessor(results()));
    return;
  }

  // Otherwise the successor is the body region.
  regions.push_back(RegionSuccessor(&body(), body().getArguments()));
}

void ExecuteOp::build(OpBuilder &builder, OperationState &result,
                      TypeRange resultTypes, ValueRange dependencies,
                      ValueRange operands, BodyBuilderFn bodyBuilder) {

  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes = DenseIntElementsAttr::get(
      VectorType::get({2}, builder.getIntegerType(32)),
      {numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `async.value`.
  result.addTypes({TokenType::get(result.getContext())});
  for (Type type : resultTypes)
    result.addTypes(ValueType::get(type));

  // Add a body region with block arguments as unwrapped async value operands.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value operand : operands) {
    auto valueType = operand.getType().dyn_cast<ValueType>();
    bodyBlock.addArgument(valueType ? valueType.getValueType()
                                    : operand.getType(),
                          operand.getLoc());
  }

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (resultTypes.empty() && !bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<async::YieldOp>(result.location, ValueRange());
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

void ExecuteOp::print(OpAsmPrinter &p) {
  // [%tokens,...]
  if (!dependencies().empty())
    p << " [" << dependencies() << "]";

  // (%value as %unwrapped: !async.value<!arg.type>, ...)
  if (!operands().empty()) {
    p << " (";
    Block *entry = body().empty() ? nullptr : &body().front();
    llvm::interleaveComma(operands(), p, [&, n = 0](Value operand) mutable {
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
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Sizes of parsed variadic operands, will be updated below after parsing.
  int32_t numDependencies = 0;

  auto tokenTy = TokenType::get(ctx);

  // Parse dependency tokens.
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<OpAsmParser::OperandType, 4> tokenArgs;
    if (parser.parseOperandList(tokenArgs) ||
        parser.resolveOperands(tokenArgs, tokenTy, result.operands) ||
        parser.parseRSquare())
      return failure();

    numDependencies = tokenArgs.size();
  }

  // Parse async value operands (%value as %unwrapped : !async.value<!type>).
  SmallVector<OpAsmParser::OperandType, 4> valueArgs;
  SmallVector<OpAsmParser::OperandType, 4> unwrappedArgs;
  SmallVector<Type, 4> valueTypes;
  SmallVector<Type, 4> unwrappedTypes;

  // Parse a single instance of `%value as %unwrapped : !async.value<!type>`.
  auto parseAsyncValueArg = [&]() -> ParseResult {
    if (parser.parseOperand(valueArgs.emplace_back()) ||
        parser.parseKeyword("as") ||
        parser.parseOperand(unwrappedArgs.emplace_back()) ||
        parser.parseColonType(valueTypes.emplace_back()))
      return failure();

    auto valueTy = valueTypes.back().dyn_cast<ValueType>();
    unwrappedTypes.emplace_back(valueTy ? valueTy.getValueType() : Type());

    return success();
  };

  auto argsLoc = parser.getCurrentLocation();
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::OptionalParen,
                                     parseAsyncValueArg) ||
      parser.resolveOperands(valueArgs, valueTypes, argsLoc, result.operands))
    return failure();

  int32_t numOperands = valueArgs.size();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  auto operandSegmentSizes = DenseIntElementsAttr::get(
      VectorType::get({2}, parser.getBuilder().getI32Type()),
      {numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // Parse the types of results returned from the async execute op.
  SmallVector<Type, 4> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();

  // Async execute first result is always a completion token.
  parser.addTypeToList(tokenTy, result.types);
  parser.addTypesToList(resultTypes, result.types);

  // Parse operation attributes.
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();
  result.addAttributes(attrs);

  // Parse asynchronous region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{unwrappedArgs},
                         /*argTypes=*/{unwrappedTypes},
                         /*argLocations=*/{},
                         /*enableNameShadowing=*/false))
    return failure();

  return success();
}

LogicalResult ExecuteOp::verify() {
  // Unwrap async.execute value operands types.
  auto unwrappedTypes = llvm::map_range(operands(), [](Value operand) {
    return operand.getType().cast<ValueType>().getValueType();
  });

  // Verify that unwrapped argument types matches the body region arguments.
  if (body().getArgumentTypes() != unwrappedTypes)
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
  if (auto valueType = operand.getType().dyn_cast<ValueType>())
    result.addTypes(valueType.getValueType());
}

static ParseResult parseAwaitResultType(OpAsmParser &parser, Type &operandType,
                                        Type &resultType) {
  if (parser.parseType(operandType))
    return failure();

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = operandType.dyn_cast<ValueType>())
    resultType = valueType.getValueType();

  return success();
}

static void printAwaitResultType(OpAsmPrinter &p, Operation *op,
                                 Type operandType, Type resultType) {
  p << operandType;
}

LogicalResult AwaitOp::verify() {
  Type argType = operand().getType();

  // Awaiting on a token does not have any results.
  if (argType.isa<TokenType>() && !getResultTypes().empty())
    return emitOpError("awaiting on a token must have empty result");

  // Awaiting on a value unwraps the async value type.
  if (auto value = argType.dyn_cast<ValueType>()) {
    if (*getResultType() != value.getValueType())
      return emitOpError() << "result type " << *getResultType()
                           << " does not match async value type "
                           << value.getValueType();
  }

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
