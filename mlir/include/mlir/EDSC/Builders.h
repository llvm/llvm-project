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
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
class OperationFolder;

namespace edsc {
class BlockHandle;
class NestedBuilder;

/// Helper class to transparently handle builder insertion points by RAII.
/// As its name indicates, a ScopedContext is means to be used locally in a
/// scoped fashion. This abstracts away all the boilerplate related to
/// checking proper usage of captures, NestedBuilders as well as handling the
/// setting and restoring of insertion points.
class ScopedContext {
public:
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

/// A NestedBuilder is a scoping abstraction to create an idiomatic syntax
/// embedded in C++ that serves the purpose of building nested MLIR.
/// Nesting and compositionality is obtained by using the strict ordering that
/// exists between object construction and method invocation on said object (in
/// our case, the call to `operator()`).
/// This ordering allows implementing an abstraction that decouples definition
/// from declaration (in a PL sense) on placeholders.
class NestedBuilder {
protected:
  NestedBuilder() = default;
  NestedBuilder(const NestedBuilder &) = delete;
  NestedBuilder(NestedBuilder &&other) : bodyScope(other.bodyScope) {
    other.bodyScope = nullptr;
  }

  NestedBuilder &operator=(const NestedBuilder &) = delete;
  NestedBuilder &operator=(NestedBuilder &&other) {
    std::swap(bodyScope, other.bodyScope);
    return *this;
  }

  /// Enter an mlir::Block and setup a ScopedContext to insert operations at
  /// the end of it. Since we cannot use c++ language-level scoping to implement
  /// scoping itself, we use enter/exit pairs of operations.
  /// As a consequence we must allocate a new OpBuilder + ScopedContext and
  /// let the escape.
  void enter(mlir::Block *block) {
    bodyScope = new ScopedContext(ScopedContext::getBuilderRef(),
                                  OpBuilder::InsertPoint(block, block->end()),
                                  ScopedContext::getLocation());
    if (!block->empty()) {
      auto &termOp = block->back();
      if (termOp.isKnownTerminator())
        ScopedContext::getBuilderRef().setInsertionPoint(&termOp);
    }
  }

  /// Exit the current mlir::Block by explicitly deleting the dynamically
  /// allocated OpBuilder and ScopedContext.
  void exit() {
    delete bodyScope;
    bodyScope = nullptr;
  }

  /// Custom destructor does nothing because we already destroyed bodyScope
  /// manually in `exit`. Insert an assertion to defensively guard against
  /// improper usage of scoping.
  ~NestedBuilder() {
    assert(!bodyScope &&
           "Illegal use of NestedBuilder; must have called exit()");
  }

private:
  ScopedContext *bodyScope = nullptr;
};

/// A LoopBuilder is a generic NestedBuilder for loop-like MLIR operations.
/// More specifically it is meant to be used as a temporary object for
/// representing any nested MLIR construct that is "related to" an mlir::Value
/// (for now an induction variable).
/// This is extensible and will evolve in the future as MLIR evolves, hence
/// the name LoopBuilder (as opposed to say ForBuilder or AffineForBuilder).
class LoopBuilder : public NestedBuilder {
public:
  LoopBuilder(const LoopBuilder &) = delete;
  LoopBuilder(LoopBuilder &&) = default;

  LoopBuilder &operator=(const LoopBuilder &) = delete;
  LoopBuilder &operator=(LoopBuilder &&) = default;

  /// The only purpose of this operator is to serve as a sequence point so that
  /// the evaluation of `fun` (which build IR snippets in a scoped fashion) is
  /// scoped within a LoopBuilder.
  void operator()(function_ref<void(void)> fun = nullptr);
  void setOp(Operation *op) { this->op = op; }
  Operation *getOp() { return op; }

private:
  LoopBuilder() = default;

  friend LoopBuilder makeAffineLoopBuilder(Value *iv, ArrayRef<Value> lbs,
                                           ArrayRef<Value> ubs, int64_t step);
  friend LoopBuilder makeParallelLoopBuilder(MutableArrayRef<Value> ivs,
                                             ArrayRef<Value> lbs,
                                             ArrayRef<Value> ubs,
                                             ArrayRef<Value> steps);
  friend LoopBuilder makeLoopBuilder(Value *iv, Value lb, Value ub, Value step,
                                     MutableArrayRef<Value> iterArgsHandles,
                                     ValueRange iterArgsInitValues);
  Operation *op;
};

// This class exists solely to handle the C++ vexing parse case when
// trying to enter a Block that has already been constructed.
class Append {};

/// A BlockBuilder is a NestedBuilder for mlir::Block*.
/// This exists by opposition to LoopBuilder which is not related to an
/// mlir::Block* but to a mlir::Value.
/// It is meant to be used as a temporary object for representing any nested
/// MLIR construct that is "related to" an mlir::Block*.
class BlockBuilder : public NestedBuilder {
public:
  /// Enters the mlir::Block* previously captured by `bh` and sets the insertion
  /// point to its end. If the block already contains a terminator, set the
  /// insertion point before the terminator.
  BlockBuilder(BlockHandle bh, Append);

  /// Constructs a new mlir::Block with argument types derived from `args`.
  /// Captures the new block in `bh` and its arguments into `args`.
  /// Enters the new mlir::Block* and sets the insertion point to its end.
  ///
  /// Prerequisites:
  ///   The Value `args` are typed delayed Values; i.e. they are
  ///   not yet bound to mlir::Value.
  BlockBuilder(BlockHandle *bh) : BlockBuilder(bh, {}, {}) {}
  BlockBuilder(BlockHandle *bh, ArrayRef<Type> types,
               MutableArrayRef<Value> args);

  /// Constructs a new mlir::Block with argument types derived from `args` and
  /// appends it as the last block in the region.
  /// Captures the new block in `bh` and its arguments into `args`.
  /// Enters the new mlir::Block* and sets the insertion point to its end.
  ///
  /// Prerequisites:
  ///   The Value `args` are typed delayed Values; i.e. they are
  ///   not yet bound to mlir::Value.
  BlockBuilder(BlockHandle *bh, Region &region, ArrayRef<Type> types,
               MutableArrayRef<Value> args);

  /// The only purpose of this operator is to serve as a sequence point so that
  /// the evaluation of `fun` (which build IR snippets in a scoped fashion) is
  /// scoped within a BlockBuilder.
  void operator()(function_ref<void(void)> fun = nullptr);

private:
  BlockBuilder(BlockBuilder &) = delete;
  BlockBuilder &operator=(BlockBuilder &other) = delete;
};

/// A BlockHandle represents a (potentially "delayed") Block abstraction.
/// This extra abstraction is necessary because an mlir::Block is not an
/// mlir::Value.
/// A BlockHandle should be captured by pointer but otherwise passed by Value
/// everywhere.
class BlockHandle {
public:
  /// A BlockHandle constructed without an mlir::Block* represents a "delayed"
  /// Block. A delayed Block represents the declaration (in the PL sense) of a
  /// placeholder for an mlir::Block* that will be constructed and captured at
  /// some later point in the program.
  BlockHandle() : block(nullptr) {}

  /// A BlockHandle constructed with an mlir::Block* represents an "eager"
  /// Block. An eager Block represents both the declaration and the definition
  /// (in the PL sense) of a placeholder for an mlir::Block* that has already
  /// been constructed in the past and that is captured "now" in the program.
  BlockHandle(mlir::Block *block) : block(block) {}

  /// BlockHandle is a value type, use the default copy constructor and
  /// assignment operator.
  BlockHandle(const BlockHandle &) = default;
  BlockHandle &operator=(const BlockHandle &) = default;

  /// Delegates block creation to MLIR and wrap the resulting mlir::Block.
  static BlockHandle create(ArrayRef<Type> argTypes);

  /// Delegates block creation to MLIR and wrap the resulting mlir::Block.
  static BlockHandle createInRegion(Region &region, ArrayRef<Type> argTypes);

  operator bool() { return block != nullptr; }
  operator mlir::Block *() { return block; }
  mlir::Block *getBlock() { return block; }

private:
  mlir::Block *block;
};

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
    assert((v.getType().isa<MemRefType>() ||
            v.getType().isa<RankedTensorType>() ||
            v.getType().isa<VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }
  StructuredIndexed(Type t, ArrayRef<AffineExpr> indexings)
      : type(t), exprs(indexings.begin(), indexings.end()) {
    assert((t.isa<MemRefType>() || t.isa<RankedTensorType>() ||
            t.isa<VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }

  bool hasValue() const { return value; }
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

/// A TemplatedIndexedValue brings an index notation over the template Load and
/// Store parameters. Assigning to an IndexedValue emits an actual `Store`
/// operation, while converting an IndexedValue to a Value emits an actual
/// `Load` operation.
template <typename Load, typename Store>
class TemplatedIndexedValue {
public:
  explicit TemplatedIndexedValue(Value v) : value(v) {}

  TemplatedIndexedValue(const TemplatedIndexedValue &rhs) = default;

  TemplatedIndexedValue operator()() { return *this; }
  /// Returns a new `TemplatedIndexedValue`.
  TemplatedIndexedValue operator()(Value index) {
    TemplatedIndexedValue res(value);
    res.indices.push_back(index);
    return res;
  }
  template <typename... Args>
  TemplatedIndexedValue operator()(Value index, Args... indices) {
    return TemplatedIndexedValue(value, index).append(indices...);
  }
  TemplatedIndexedValue operator()(ArrayRef<Value> indices) {
    return TemplatedIndexedValue(value, indices);
  }

  /// Emits a `store`.
  Store operator=(const TemplatedIndexedValue &rhs) {
    return Store(rhs, value, indices);
  }
  Store operator=(Value rhs) { return Store(rhs, value, indices); }

  /// Emits a `load` when converting to a Value.
  operator Value() const { return Load(value, indices); }

  /// Returns the base memref.
  Value getBase() const { return value; }

  /// Returns the underlying memref.
  MemRefType getMemRefType() const {
    return value.getType().template cast<MemRefType>();
  }

  /// Returns the underlying MemRef elemental type cast as `T`.
  template <typename T>
  T getElementalTypeAs() const {
    return value.getType()
        .template cast<MemRefType>()
        .getElementType()
        .template cast<T>();
  }

  /// Arithmetic operator overloadings.
  Value operator+(Value e);
  Value operator-(Value e);
  Value operator*(Value e);
  Value operator/(Value e);
  Value operator%(Value e);
  Value operator^(Value e);
  Value operator+(TemplatedIndexedValue e) {
    return *this + static_cast<Value>(e);
  }
  Value operator-(TemplatedIndexedValue e) {
    return *this - static_cast<Value>(e);
  }
  Value operator*(TemplatedIndexedValue e) {
    return *this * static_cast<Value>(e);
  }
  Value operator/(TemplatedIndexedValue e) {
    return *this / static_cast<Value>(e);
  }
  Value operator%(TemplatedIndexedValue e) {
    return *this % static_cast<Value>(e);
  }
  Value operator^(TemplatedIndexedValue e) {
    return *this ^ static_cast<Value>(e);
  }

  /// Assignment-arithmetic operator overloadings.
  Store operator+=(Value e);
  Store operator-=(Value e);
  Store operator*=(Value e);
  Store operator/=(Value e);
  Store operator%=(Value e);
  Store operator^=(Value e);
  Store operator+=(TemplatedIndexedValue e) {
    return this->operator+=(static_cast<Value>(e));
  }
  Store operator-=(TemplatedIndexedValue e) {
    return this->operator-=(static_cast<Value>(e));
  }
  Store operator*=(TemplatedIndexedValue e) {
    return this->operator*=(static_cast<Value>(e));
  }
  Store operator/=(TemplatedIndexedValue e) {
    return this->operator/=(static_cast<Value>(e));
  }
  Store operator%=(TemplatedIndexedValue e) {
    return this->operator%=(static_cast<Value>(e));
  }
  Store operator^=(TemplatedIndexedValue e) {
    return this->operator^=(static_cast<Value>(e));
  }

  /// Logical operator overloadings.
  Value operator&&(Value e);
  Value operator||(Value e);
  Value operator&&(TemplatedIndexedValue e) {
    return *this && static_cast<Value>(e);
  }
  Value operator||(TemplatedIndexedValue e) {
    return *this || static_cast<Value>(e);
  }

  /// Comparison operator overloadings.
  Value eq(Value e);
  Value ne(Value e);
  Value operator<(Value e);
  Value operator<=(Value e);
  Value operator>(Value e);
  Value operator>=(Value e);
  Value operator<(TemplatedIndexedValue e) {
    return *this < static_cast<Value>(e);
  }
  Value operator<=(TemplatedIndexedValue e) {
    return *this <= static_cast<Value>(e);
  }
  Value operator>(TemplatedIndexedValue e) {
    return *this > static_cast<Value>(e);
  }
  Value operator>=(TemplatedIndexedValue e) {
    return *this >= static_cast<Value>(e);
  }

private:
  TemplatedIndexedValue(Value value, ArrayRef<Value> indices)
      : value(value), indices(indices.begin(), indices.end()) {}

  TemplatedIndexedValue &append() { return *this; }

  template <typename T, typename... Args>
  TemplatedIndexedValue &append(T index, Args... indices) {
    this->indices.push_back(static_cast<Value>(index));
    append(indices...);
    return *this;
  }
  Value value;
  SmallVector<Value, 8> indices;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_BUILDERS_H_
