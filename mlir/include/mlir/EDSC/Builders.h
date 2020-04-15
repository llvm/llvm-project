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
class CapturableHandle;
class NestedBuilder;
class ValueHandle;

/// Helper class to transparently handle builder insertion points by RAII.
/// As its name indicates, a ScopedContext is means to be used locally in a
/// scoped fashion. This abstracts away all the boilerplate related to
/// checking proper usage of captures, NestedBuilders as well as handling the
/// setting and restoring of insertion points.
class ScopedContext {
public:
  ScopedContext(OpBuilder &builder, Location location);

  /// Sets the insertion point of the builder to 'newInsertPt' for the duration
  /// of the scope. The existing insertion point of the builder is restored on
  /// destruction.
  ScopedContext(OpBuilder &builder, OpBuilder::InsertPoint newInsertPt,
                Location location);
  ~ScopedContext();

  static MLIRContext *getContext();
  static OpBuilder &getBuilder();
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
  /// The previous insertion point of the builder.
  Optional<OpBuilder::InsertPoint> prevBuilderInsertPoint;
  /// Current location.
  Location location;
  /// Parent context we return into.
  ScopedContext *enclosingScopedContext;
  /// Defensively keeps track of the current NestedBuilder to ensure proper
  /// scoping usage.
  NestedBuilder *nestedBuilder;

  // TODO: Implement scoping of ValueHandles. To do this we need a proper data
  // structure to hold ValueHandle objects. We can emulate one but there should
  // already be something available in LLVM for this purpose.
};

/// A NestedBuilder is a scoping abstraction to create an idiomatic syntax
/// embedded in C++ that serves the purpose of building nested MLIR.
/// Nesting and compositionality is obtained by using the strict ordering that
/// exists between object construction and method invocation on said object (in
/// our case, the call to `operator()`).
/// This ordering allows implementing an abstraction that decouples definition
/// from declaration (in a PL sense) on placeholders of type ValueHandle and
/// BlockHandle.
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
  /// Step back "prev" times from the end of the block to set up the insertion
  /// point, which is useful for non-empty blocks.
  void enter(mlir::Block *block, int prev = 0) {
    bodyScope = new ScopedContext(
        ScopedContext::getBuilder(),
        OpBuilder::InsertPoint(block, std::prev(block->end(), prev)),
        ScopedContext::getLocation());
    bodyScope->nestedBuilder = this;
  }

  /// Exit the current mlir::Block by explicitly deleting the dynamically
  /// allocated OpBuilder and ScopedContext.
  void exit() {
    // Reclaim now to exit the scope.
    bodyScope->nestedBuilder = nullptr;
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

  friend LoopBuilder makeAffineLoopBuilder(ValueHandle *iv,
                                           ArrayRef<ValueHandle> lbHandles,
                                           ArrayRef<ValueHandle> ubHandles,
                                           int64_t step);
  friend LoopBuilder makeParallelLoopBuilder(ArrayRef<ValueHandle *> ivs,
                                             ArrayRef<ValueHandle> lbHandles,
                                             ArrayRef<ValueHandle> ubHandles,
                                             ArrayRef<ValueHandle> steps);
  friend LoopBuilder makeLoopBuilder(ValueHandle *iv, ValueHandle lbHandle,
                                     ValueHandle ubHandle,
                                     ValueHandle stepHandle,
                                     ArrayRef<ValueHandle *> iter_args_handles,
                                     ValueRange iter_args_init_values);
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
  /// point to its end.
  BlockBuilder(BlockHandle bh, Append);

  /// Constructs a new mlir::Block with argument types derived from `args`.
  /// Captures the new block in `bh` and its arguments into `args`.
  /// Enters the new mlir::Block* and sets the insertion point to its end.
  ///
  /// Prerequisites:
  ///   The ValueHandle `args` are typed delayed ValueHandles; i.e. they are
  ///   not yet bound to mlir::Value.
  BlockBuilder(BlockHandle *bh, ArrayRef<ValueHandle *> args);

  /// Constructs a new mlir::Block with argument types derived from `args` and
  /// appends it as the last block in the region.
  /// Captures the new block in `bh` and its arguments into `args`.
  /// Enters the new mlir::Block* and sets the insertion point to its end.
  ///
  /// Prerequisites:
  ///   The ValueHandle `args` are typed delayed ValueHandles; i.e. they are
  ///   not yet bound to mlir::Value.
  BlockBuilder(BlockHandle *bh, Region &region, ArrayRef<ValueHandle *> args);

  /// The only purpose of this operator is to serve as a sequence point so that
  /// the evaluation of `fun` (which build IR snippets in a scoped fashion) is
  /// scoped within a BlockBuilder.
  void operator()(function_ref<void(void)> fun = nullptr);

private:
  BlockBuilder(BlockBuilder &) = delete;
  BlockBuilder &operator=(BlockBuilder &other) = delete;
};

/// Base class for ValueHandle, OperationHandle and BlockHandle.
/// Not meant to be used outside of these classes.
class CapturableHandle {
protected:
  CapturableHandle() = default;
};

/// ValueHandle implements a (potentially "delayed") typed Value abstraction.
/// ValueHandle should be captured by pointer but otherwise passed by Value
/// everywhere.
/// A ValueHandle can have 3 states:
///   1. null state (empty type and empty value), in which case it does not hold
///      a value and must never hold a Value (now or in the future). This is
///      used for MLIR operations with zero returns as well as the result of
///      calling a NestedBuilder::operator(). In both cases the objective is to
///      have an object that can be inserted in an ArrayRef<ValueHandle> to
///      implement nesting;
///   2. delayed state (empty value), in which case it represents an eagerly
///      typed "delayed" value that can be hold a Value in the future;
///   3. constructed state,in which case it holds a Value.
///
/// A ValueHandle is meant to capture a single Value and should be used for
/// operations that have a single result. For convenience of use, we also
/// include AffineForOp in this category although it does not return a value.
/// In the case of AffineForOp, the captured Value is the loop induction
/// variable.
class ValueHandle : public CapturableHandle {
public:
  /// A ValueHandle in a null state can never be captured;
  static ValueHandle null() { return ValueHandle(); }

  /// A ValueHandle that is constructed from a Type represents a typed "delayed"
  /// Value. A delayed Value can only capture Values of the specified type.
  /// Such a delayed value represents the declaration (in the PL sense) of a
  /// placeholder for an mlir::Value that will be constructed and captured at
  /// some later point in the program.
  explicit ValueHandle(Type t) : t(t), v(nullptr) {}

  /// A ValueHandle that is constructed from an mlir::Value is an "eager"
  /// Value. An eager Value represents both the declaration and the definition
  /// (in the PL sense) of a placeholder for an mlir::Value that has already
  /// been constructed in the past and that is captured "now" in the program.
  explicit ValueHandle(Value v) : t(v.getType()), v(v) {}

  /// ValueHandle is a value type, use the default copy constructor.
  ValueHandle(const ValueHandle &other) = default;

  /// ValueHandle is a value type, the assignment operator typechecks before
  /// assigning.
  ValueHandle &operator=(const ValueHandle &other);

  /// Provide a swap operator.
  void swap(ValueHandle &other) {
    if (this == &other)
      return;
    std::swap(t, other.t);
    std::swap(v, other.v);
  }

  /// Implicit conversion useful for automatic conversion to Container<Value>.
  operator Value() const { return getValue(); }
  operator Type() const { return getType(); }
  operator bool() const { return hasValue(); }

  /// Generic mlir::Op create. This is the key to being extensible to the whole
  /// of MLIR without duplicating the type system or the op definitions.
  template <typename Op, typename... Args>
  static ValueHandle create(Args... args);

  /// Generic mlir::Op create. This is the key to being extensible to the whole
  /// of MLIR without duplicating the type system or the op definitions.
  /// When non-null, the optional pointer `folder` is used to call into the
  /// `createAndFold` builder method. If `folder` is null, the regular `create`
  /// method is called.
  template <typename Op, typename... Args>
  static ValueHandle create(OperationFolder *folder, Args... args);

  /// Generic create for a named operation producing a single value.
  static ValueHandle create(StringRef name, ArrayRef<ValueHandle> operands,
                            ArrayRef<Type> resultTypes,
                            ArrayRef<NamedAttribute> attributes = {});

  bool hasValue() const { return v != nullptr; }
  Value getValue() const {
    assert(hasValue() && "Unexpected null value;");
    return v;
  }
  bool hasType() const { return t != Type(); }
  Type getType() const { return t; }

  Operation *getOperation() const {
    if (!v)
      return nullptr;
    return v.getDefiningOp();
  }

  // Return a vector of fresh ValueHandles that have not captured.
  static SmallVector<ValueHandle, 8> makeIndexHandles(unsigned count) {
    auto indexType = IndexType::get(ScopedContext::getContext());
    return SmallVector<ValueHandle, 8>(count, ValueHandle(indexType));
  }

protected:
  ValueHandle() : t(), v(nullptr) {}

  Type t;
  Value v;
};

/// An OperationHandle can be used in lieu of ValueHandle to capture the
/// operation in cases when one does not care about, or cannot extract, a
/// unique Value from the operation.
/// This can be used for capturing zero result operations as well as
/// multi-result operations that are not supported by ValueHandle.
/// We do not distinguish further between zero and multi-result operations at
/// this time.
struct OperationHandle : public CapturableHandle {
  OperationHandle() : op(nullptr) {}
  OperationHandle(Operation *op) : op(op) {}

  OperationHandle(const OperationHandle &) = default;
  OperationHandle &operator=(const OperationHandle &) = default;

  /// Generic mlir::Op create. This is the key to being extensible to the whole
  /// of MLIR without duplicating the type system or the op definitions.
  template <typename Op, typename... Args>
  static OperationHandle create(Args... args);
  template <typename Op, typename... Args>
  static Op createOp(Args... args);

  /// Generic create for a named operation.
  static OperationHandle create(StringRef name, ArrayRef<ValueHandle> operands,
                                ArrayRef<Type> resultTypes,
                                ArrayRef<NamedAttribute> attributes = {});

  operator Operation *() { return op; }
  Operation *getOperation() const { return op; }

private:
  Operation *op;
};

/// Simple wrapper to build a generic operation without successor blocks.
template <typename HandleType>
struct CustomOperation {
  CustomOperation(StringRef name) : name(name) {
    static_assert(std::is_same<HandleType, ValueHandle>() ||
                      std::is_same<HandleType, OperationHandle>(),
                  "Only CustomOperation<ValueHandle> or "
                  "CustomOperation<OperationHandle> can be constructed.");
  }
  HandleType operator()(ArrayRef<ValueHandle> operands = {},
                        ArrayRef<Type> resultTypes = {},
                        ArrayRef<NamedAttribute> attributes = {}) {
    return HandleType::create(name, operands, resultTypes, attributes);
  }
  std::string name;
};

/// A BlockHandle represents a (potentially "delayed") Block abstraction.
/// This extra abstraction is necessary because an mlir::Block is not an
/// mlir::Value.
/// A BlockHandle should be captured by pointer but otherwise passed by Value
/// everywhere.
class BlockHandle : public CapturableHandle {
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
struct StructuredIndexed : public ValueHandle {
  StructuredIndexed(Type type) : ValueHandle(type) {}
  StructuredIndexed(Value value) : ValueHandle(value) {}
  StructuredIndexed(ValueHandle valueHandle) : ValueHandle(valueHandle) {}
  StructuredIndexed operator()(ArrayRef<AffineExpr> indexings) {
    return this->hasValue() ? StructuredIndexed(this->getValue(), indexings)
                            : StructuredIndexed(this->getType(), indexings);
  }

  StructuredIndexed(Type t, ArrayRef<AffineExpr> indexings)
      : ValueHandle(t), exprs(indexings.begin(), indexings.end()) {
    assert(t.isa<RankedTensorType>() && "RankedTensor expected");
  }
  StructuredIndexed(Value v, ArrayRef<AffineExpr> indexings)
      : ValueHandle(v), exprs(indexings.begin(), indexings.end()) {
    assert((v.getType().isa<MemRefType>() ||
            v.getType().isa<RankedTensorType>() ||
            v.getType().isa<VectorType>()) &&
           "MemRef, RankedTensor or Vector expected");
  }
  StructuredIndexed(ValueHandle vh, ArrayRef<AffineExpr> indexings)
      : ValueHandle(vh), exprs(indexings.begin(), indexings.end()) {}

  ArrayRef<AffineExpr> getExprs() { return exprs; }

private:
  SmallVector<AffineExpr, 4> exprs;
};

template <typename Op, typename... Args>
OperationHandle OperationHandle::create(Args... args) {
  return OperationHandle(ScopedContext::getBuilder()
                             .create<Op>(ScopedContext::getLocation(), args...)
                             .getOperation());
}

template <typename Op, typename... Args>
Op OperationHandle::createOp(Args... args) {
  return cast<Op>(
      OperationHandle(ScopedContext::getBuilder()
                          .create<Op>(ScopedContext::getLocation(), args...)
                          .getOperation())
          .getOperation());
}

template <typename Op, typename... Args>
ValueHandle ValueHandle::create(Args... args) {
  Operation *op = ScopedContext::getBuilder()
                      .create<Op>(ScopedContext::getLocation(), args...)
                      .getOperation();
  if (op->getNumResults() == 1)
    return ValueHandle(op->getResult(0));
  llvm_unreachable("unsupported operation, use an OperationHandle instead");
}

/// Entry point to build multiple ValueHandle from a `Container` of Value or
/// Type.
template <typename Container>
inline SmallVector<ValueHandle, 8> makeValueHandles(Container values) {
  SmallVector<ValueHandle, 8> res;
  res.reserve(values.size());
  for (auto v : values)
    res.push_back(ValueHandle(v));
  return res;
}

/// A TemplatedIndexedValue brings an index notation over the template Load and
/// Store parameters. Assigning to an IndexedValue emits an actual `Store`
/// operation, while converting an IndexedValue to a ValueHandle emits an actual
/// `Load` operation.
template <typename Load, typename Store>
class TemplatedIndexedValue {
public:
  explicit TemplatedIndexedValue(Type t) : base(t) {}
  explicit TemplatedIndexedValue(Value v)
      : TemplatedIndexedValue(ValueHandle(v)) {}
  explicit TemplatedIndexedValue(ValueHandle v) : base(v) {}

  TemplatedIndexedValue(const TemplatedIndexedValue &rhs) = default;

  TemplatedIndexedValue operator()() { return *this; }
  /// Returns a new `TemplatedIndexedValue`.
  TemplatedIndexedValue operator()(ValueHandle index) {
    TemplatedIndexedValue res(base);
    res.indices.push_back(index);
    return res;
  }
  template <typename... Args>
  TemplatedIndexedValue operator()(ValueHandle index, Args... indices) {
    return TemplatedIndexedValue(base, index).append(indices...);
  }
  TemplatedIndexedValue operator()(ArrayRef<ValueHandle> indices) {
    return TemplatedIndexedValue(base, indices);
  }

  /// Emits a `store`.
  OperationHandle operator=(const TemplatedIndexedValue &rhs) {
    ValueHandle rrhs(rhs);
    return Store(rrhs, getBase(), {indices.begin(), indices.end()});
  }
  OperationHandle operator=(ValueHandle rhs) {
    return Store(rhs, getBase(), {indices.begin(), indices.end()});
  }

  /// Emits a `load` when converting to a ValueHandle.
  operator ValueHandle() const {
    return Load(getBase(), {indices.begin(), indices.end()});
  }

  /// Emits a `load` when converting to a Value.
  Value operator*(void)const {
    return Load(getBase(), {indices.begin(), indices.end()}).getValue();
  }

  ValueHandle getBase() const { return base; }

  /// Arithmetic operator overloadings.
  ValueHandle operator+(ValueHandle e);
  ValueHandle operator-(ValueHandle e);
  ValueHandle operator*(ValueHandle e);
  ValueHandle operator/(ValueHandle e);
  ValueHandle operator%(ValueHandle e);
  ValueHandle operator^(ValueHandle e);
  ValueHandle operator+(TemplatedIndexedValue e) {
    return *this + static_cast<ValueHandle>(e);
  }
  ValueHandle operator-(TemplatedIndexedValue e) {
    return *this - static_cast<ValueHandle>(e);
  }
  ValueHandle operator*(TemplatedIndexedValue e) {
    return *this * static_cast<ValueHandle>(e);
  }
  ValueHandle operator/(TemplatedIndexedValue e) {
    return *this / static_cast<ValueHandle>(e);
  }
  ValueHandle operator%(TemplatedIndexedValue e) {
    return *this % static_cast<ValueHandle>(e);
  }
  ValueHandle operator^(TemplatedIndexedValue e) {
    return *this ^ static_cast<ValueHandle>(e);
  }

  /// Assignment-arithmetic operator overloadings.
  OperationHandle operator+=(ValueHandle e);
  OperationHandle operator-=(ValueHandle e);
  OperationHandle operator*=(ValueHandle e);
  OperationHandle operator/=(ValueHandle e);
  OperationHandle operator%=(ValueHandle e);
  OperationHandle operator^=(ValueHandle e);
  OperationHandle operator+=(TemplatedIndexedValue e) {
    return this->operator+=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator-=(TemplatedIndexedValue e) {
    return this->operator-=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator*=(TemplatedIndexedValue e) {
    return this->operator*=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator/=(TemplatedIndexedValue e) {
    return this->operator/=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator%=(TemplatedIndexedValue e) {
    return this->operator%=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator^=(TemplatedIndexedValue e) {
    return this->operator^=(static_cast<ValueHandle>(e));
  }

  /// Logical operator overloadings.
  ValueHandle operator&&(ValueHandle e);
  ValueHandle operator||(ValueHandle e);
  ValueHandle operator&&(TemplatedIndexedValue e) {
    return *this && static_cast<ValueHandle>(e);
  }
  ValueHandle operator||(TemplatedIndexedValue e) {
    return *this || static_cast<ValueHandle>(e);
  }

  /// Comparison operator overloadings.
  ValueHandle operator==(ValueHandle e);
  ValueHandle operator!=(ValueHandle e);
  ValueHandle operator<(ValueHandle e);
  ValueHandle operator<=(ValueHandle e);
  ValueHandle operator>(ValueHandle e);
  ValueHandle operator>=(ValueHandle e);
  ValueHandle operator==(TemplatedIndexedValue e) {
    return *this == static_cast<ValueHandle>(e);
  }
  ValueHandle operator!=(TemplatedIndexedValue e) {
    return *this != static_cast<ValueHandle>(e);
  }
  ValueHandle operator<(TemplatedIndexedValue e) {
    return *this < static_cast<ValueHandle>(e);
  }
  ValueHandle operator<=(TemplatedIndexedValue e) {
    return *this <= static_cast<ValueHandle>(e);
  }
  ValueHandle operator>(TemplatedIndexedValue e) {
    return *this > static_cast<ValueHandle>(e);
  }
  ValueHandle operator>=(TemplatedIndexedValue e) {
    return *this >= static_cast<ValueHandle>(e);
  }

private:
  TemplatedIndexedValue(ValueHandle base, ArrayRef<ValueHandle> indices)
      : base(base), indices(indices.begin(), indices.end()) {}

  TemplatedIndexedValue &append() { return *this; }

  template <typename T, typename... Args>
  TemplatedIndexedValue &append(T index, Args... indices) {
    this->indices.push_back(static_cast<ValueHandle>(index));
    append(indices...);
    return *this;
  }
  ValueHandle base;
  SmallVector<ValueHandle, 8> indices;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_BUILDERS_H_
