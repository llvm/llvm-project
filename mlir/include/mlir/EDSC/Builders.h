//===- Builders.h - MLIR Declarative Builder Classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_BUILDERS_H_
#define MLIR_EDSC_BUILDERS_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
class OperationFolder;

namespace edsc {
/// Helper class to transparently handle builder insertion points by RAII.
/// As its name indicates, a ScopedContext is means to be used locally in a
/// scoped fashion. This abstracts away all the boilerplate related to
/// checking proper usage of captures, NestedBuilders as well as handling the
/// setting and restoring of insertion points.
class ScopedContext {
public:
  ScopedContext(OpBuilder &b);
  ScopedContext(OpBuilder &b, Location location);

  /// Sets the insertion point of the builder to 'newInsertPt' for the duration
  /// of the scope. The existing insertion point of the builder is restored on
  /// destruction.
  ScopedContext(OpBuilder &b, OpBuilder::InsertPoint newInsertPt,
                Location location);
  ~ScopedContext();

  static MLIRContext *getContext();
  static OpBuilder &getBuilderRef();
  static Location getLocation();

private:
  /// Only NestedBuilder (which is used to create an operation with a body)
  /// may access private members in order to implement scoping.
  friend class NestedBuilder;

  ScopedContext() = delete;
  ScopedContext(const ScopedContext &) = delete;
  ScopedContext &operator=(const ScopedContext &) = delete;

  static ScopedContext *&getCurrentScopedContext();

  /// Top level OpBuilder.
  OpBuilder &builder;
  /// Guard to the previous insertion point.
  OpBuilder::InsertionGuard guard;
  /// Current location.
  Location location;
  /// Parent context we return into.
  ScopedContext *enclosingScopedContext;
};

template <typename Op>
struct ValueBuilder {
  template <typename... Args>
  ValueBuilder(Args... args) {
    value = ScopedContext::getBuilderRef()
                .create<Op>(ScopedContext::getLocation(), args...)
                .getResult();
  }
  operator Value() { return value; }
  Value value;
};

template <typename Op>
struct OperationBuilder {
  template <typename... Args>
  OperationBuilder(Args... args) {
    op = ScopedContext::getBuilderRef().create<Op>(ScopedContext::getLocation(),
                                                   args...);
  }
  operator Op() { return op; }
  operator Operation *() { return op.getOperation(); }
  Op op;
};

/// Creates a block in the region that contains the insertion block of the
/// OpBuilder currently at the top of ScopedContext stack (appends the block to
/// the region). Be aware that this will NOT update the insertion point of the
/// builder to insert into the newly constructed block.
Block *createBlock(TypeRange argTypes = llvm::None);

/// Creates a block in the specified region using OpBuilder at the top of
/// ScopedContext stack (appends the block to the region). Be aware that this
/// will NOT update the insertion point of the builder to insert into the newly
/// constructed block.
Block *createBlockInRegion(Region &region, TypeRange argTypes = llvm::None);

/// Calls "builderFn" with ScopedContext reconfigured to insert into "block" and
/// passes in the block arguments. If the block has a terminator, the operations
/// are inserted before the terminator, otherwise appended to the block.
void appendToBlock(Block *block, function_ref<void(ValueRange)> builderFn);

/// Creates a block in the region that contains the insertion block of the
/// OpBuilder currently at the top of ScopedContext stack, and calls "builderFn"
/// to populate the body of the block while passing it the block arguments.
Block *buildInNewBlock(TypeRange argTypes,
                       function_ref<void(ValueRange)> builderFn);

/// Creates a block in the specified region using OpBuilder at the top of
/// ScopedContext stack, and calls "builderFn" to populate the body of the block
/// while passing it the block arguments.
Block *buildInNewBlock(Region &region, TypeRange argTypes,
                       function_ref<void(ValueRange)> builderFn);

/// A StructuredIndexed represents an indexable quantity that is either:
/// 1. a captured value, which is suitable for buffer and tensor operands, or;
/// 2. a captured type, which is suitable for tensor return values.
///
/// A StructuredIndexed itself is indexed and passed to `makeGenericLinalgOp`.
/// It enable an idiomatic syntax for index expressions such as:
///
/// ```
///      StructuredIndexed A(buffer_or_tensor_value), B(buffer_or_tensor_value),
///        C(buffer_value_or_tensor_type);
///      makeGenericLinalgOp({A({m, n}), B({k, n})}, {C({m, n})}, ... );
/// ```
struct StructuredIndexed {
  StructuredIndexed(Value v) : value(v) {}
  StructuredIndexed(Type t) : type(t) {}
  StructuredIndexed operator()(ArrayRef<AffineExpr> indexings) {
    return value ? StructuredIndexed(value, indexings)
                 : StructuredIndexed(type, indexings);
  }

  StructuredIndexed(Value v, ArrayRef<AffineExpr> indexings)
      : value(v), exprs(indexings.begin(), indexings.end()) {
    assert((v.getType().isa<MemRefType, RankedTensorType, VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }
  StructuredIndexed(Type t, ArrayRef<AffineExpr> indexings)
      : type(t), exprs(indexings.begin(), indexings.end()) {
    assert((t.isa<MemRefType, RankedTensorType, VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }

  bool hasValue() const { return (bool)value; }
  Value getValue() const {
    assert(value && "StructuredIndexed Value not set.");
    return value;
  }
  Type getType() const {
    assert((value || type) && "StructuredIndexed Value and Type not set.");
    return value ? value.getType() : type;
  }
  ArrayRef<AffineExpr> getExprs() const { return exprs; }
  operator Value() const { return getValue(); }
  operator Type() const { return getType(); }

private:
  // Only one of Value or type may be set.
  Type type;
  Value value;
  SmallVector<AffineExpr, 4> exprs;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_BUILDERS_H_
