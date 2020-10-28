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

void AsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"
      >();
  addTypes<TokenType>();
  addTypes<ValueType>();
}

/// Parse a type registered to this dialect.
Type AsyncDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "token")
    return TokenType::get(getContext());

  if (keyword == "value") {
    Type ty;
    if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "failed to parse async value type");
      return Type();
    }
    return ValueType::get(ty);
  }

  parser.emitError(parser.getNameLoc(), "unknown async type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void AsyncDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<TokenType>([&](TokenType) { os << "token"; })
      .Case<ValueType>([&](ValueType valueTy) {
        os << "value<";
        os.printType(valueTy.getValueType());
        os << '>';
      })
      .Default([](Type) { llvm_unreachable("unexpected 'async' type kind"); });
}

//===----------------------------------------------------------------------===//
/// ValueType
//===----------------------------------------------------------------------===//

namespace mlir {
namespace async {
namespace detail {

// Storage for `async.value<T>` type, the only member is the wrapped type.
struct ValueTypeStorage : public TypeStorage {
  ValueTypeStorage(Type valueType) : valueType(valueType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == valueType; }

  /// Construction.
  static ValueTypeStorage *construct(TypeStorageAllocator &allocator,
                                     Type valueType) {
    return new (allocator.allocate<ValueTypeStorage>())
        ValueTypeStorage(valueType);
  }

  Type valueType;
};

} // namespace detail
} // namespace async
} // namespace mlir

ValueType ValueType::get(Type valueType) {
  return Base::get(valueType.getContext(), valueType);
}

Type ValueType::getValueType() { return getImpl()->valueType; }

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(YieldOp op) {
  // Get the underlying value types from async values returned from the
  // parent `async.execute` operation.
  auto executeOp = op.getParentOfType<ExecuteOp>();
  auto types = llvm::map_range(executeOp.results(), [](const OpResult &result) {
    return result.getType().cast<ValueType>().getValueType();
  });

  if (op.getOperandTypes() != types)
    return op.emitOpError("operand types do not match the types returned from "
                          "the parent ExecuteOp");

  return success();
}

//===----------------------------------------------------------------------===//
/// ExecuteOp
//===----------------------------------------------------------------------===//

constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << op.getOperationName();

  // [%tokens,...]
  if (!op.dependencies().empty())
    p << " [" << op.dependencies() << "]";

  // (%value as %unwrapped: !async.value<!arg.type>, ...)
  if (!op.operands().empty()) {
    p << " (";
    llvm::interleaveComma(op.operands(), p, [&, n = 0](Value operand) mutable {
      p << operand << " as " << op.body().front().getArgument(n++) << ": "
        << operand.getType();
    });
    p << ")";
  }

  // -> (!async.value<!return.type>, ...)
  p.printOptionalArrowTypeList(op.getResultTypes().drop_front(1));
  p.printOptionalAttrDictWithKeyword(op.getAttrs(), {kOperandSegmentSizesAttr});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Sizes of parsed variadic operands, will be updated below after parsing.
  int32_t numDependencies = 0;
  int32_t numOperands = 0;

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

  if (succeeded(parser.parseOptionalLParen())) {
    auto argsLoc = parser.getCurrentLocation();

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

    // If the next token is `)` skip async value arguments parsing.
    if (failed(parser.parseOptionalRParen())) {
      do {
        if (parseAsyncValueArg())
          return failure();
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseRParen() ||
          parser.resolveOperands(valueArgs, valueTypes, argsLoc,
                                 result.operands))
        return failure();
    }

    numOperands = valueArgs.size();
  }

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
                         /*enableNameShadowing=*/false))
    return failure();

  return success();
}

static LogicalResult verify(ExecuteOp op) {
  // Unwrap async.execute value operands types.
  auto unwrappedTypes = llvm::map_range(op.operands(), [](Value operand) {
    return operand.getType().cast<ValueType>().getValueType();
  });

  // Verify that unwrapped argument types matches the body region arguments.
  if (op.body().getArgumentTypes() != unwrappedTypes)
    return op.emitOpError("async body region argument types do not match the "
                          "execute operation arguments types");

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

static LogicalResult verify(AwaitOp op) {
  Type argType = op.operand().getType();

  // Awaiting on a token does not have any results.
  if (argType.isa<TokenType>() && !op.getResultTypes().empty())
    return op.emitOpError("awaiting on a token must have empty result");

  // Awaiting on a value unwraps the async value type.
  if (auto value = argType.dyn_cast<ValueType>()) {
    if (*op.getResultType() != value.getValueType())
      return op.emitOpError()
             << "result type " << *op.getResultType()
             << " does not match async value type " << value.getValueType();
  }

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"
