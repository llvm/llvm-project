//===- Async.cpp - MLIR Async Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace async {

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
  auto types = llvm::map_range(executeOp.values(), [](const OpResult &result) {
    return result.getType().cast<ValueType>().getValueType();
  });

  if (!std::equal(types.begin(), types.end(), op.getOperandTypes().begin()))
    return op.emitOpError("Operand types do not match the types returned from "
                          "the parent ExecuteOp");

  return success();
}

//===----------------------------------------------------------------------===//
/// ExecuteOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << "async.execute ";
  p.printRegion(op.body());
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.done().getType());
  if (!op.values().empty())
    p << ", ";
  llvm::interleaveComma(op.values(), p, [&](const OpResult &result) {
    p.printType(result.getType());
  });
}

static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Parse asynchronous region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{},
                         /*enableNameShadowing=*/false))
    return failure();

  // Parse operation attributes.
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  result.addAttributes(attrs);

  // Parse result types.
  SmallVector<Type, 4> resultTypes;
  if (parser.parseColonTypeList(resultTypes))
    return failure();

  // First result type must be an async token type.
  if (resultTypes.empty() || resultTypes.front() != TokenType::get(ctx))
    return failure();
  parser.addTypesToList(resultTypes, result.types);

  return success();
}

} // namespace async
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"
