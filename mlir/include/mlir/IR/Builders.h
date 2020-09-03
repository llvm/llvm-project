//===- Builders.h - Helpers for constructing MLIR Classes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILDERS_H
#define MLIR_IR_BUILDERS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {

class AffineExpr;
class BlockAndValueMapping;
class ModuleOp;
class UnknownLoc;
class FileLineColLoc;
class Type;
class PrimitiveType;
class IntegerType;
class FunctionType;
class MemRefType;
class VectorType;
class RankedTensorType;
class UnrankedTensorType;
class TupleType;
class NoneType;
class BoolAttr;
class IntegerAttr;
class FloatAttr;
class StringAttr;
class TypeAttr;
class ArrayAttr;
class SymbolRefAttr;
class ElementsAttr;
class DenseElementsAttr;
class DenseIntElementsAttr;
class AffineMapAttr;
class AffineMap;
class UnitAttr;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(ModuleOp module);

  MLIRContext *getContext() const { return context; }

  Identifier getIdentifier(StringRef str);

  // Locations.
  Location getUnknownLoc();
  Location getFileLineColLoc(Identifier filename, unsigned line,
                             unsigned column);
  Location getFusedLoc(ArrayRef<Location> locs,
                       Attribute metadata = Attribute());

  // Types.
  FloatType getBF16Type();
  FloatType getF16Type();
  FloatType getF32Type();
  FloatType getF64Type();

  IndexType getIndexType();

  IntegerType getI1Type();
  IntegerType getI32Type();
  IntegerType getI64Type();
  IntegerType getIntegerType(unsigned width);
  IntegerType getIntegerType(unsigned width, bool isSigned);
  FunctionType getFunctionType(ArrayRef<Type> inputs, ArrayRef<Type> results);
  TupleType getTupleType(ArrayRef<Type> elementTypes);
  NoneType getNoneType();

  /// Get or construct an instance of the type 'ty' with provided arguments.
  template <typename Ty, typename... Args> Ty getType(Args... args) {
    return Ty::get(context, args...);
  }

  // Attributes.
  NamedAttribute getNamedAttr(StringRef name, Attribute val);

  UnitAttr getUnitAttr();
  BoolAttr getBoolAttr(bool value);
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> value);
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  IntegerAttr getIntegerAttr(Type type, const APInt &value);
  FloatAttr getFloatAttr(Type type, double value);
  FloatAttr getFloatAttr(Type type, const APFloat &value);
  StringAttr getStringAttr(StringRef bytes);
  ArrayAttr getArrayAttr(ArrayRef<Attribute> value);
  FlatSymbolRefAttr getSymbolRefAttr(Operation *value);
  FlatSymbolRefAttr getSymbolRefAttr(StringRef value);
  SymbolRefAttr getSymbolRefAttr(StringRef value,
                                 ArrayRef<FlatSymbolRefAttr> nestedReferences);

  // Returns a 0-valued attribute of the given `type`. This function only
  // supports boolean, integer, and 16-/32-/64-bit float types, and vector or
  // ranked tensor of them. Returns null attribute otherwise.
  Attribute getZeroAttr(Type type);

  // Convenience methods for fixed types.
  FloatAttr getF16FloatAttr(float value);
  FloatAttr getF32FloatAttr(float value);
  FloatAttr getF64FloatAttr(double value);

  IntegerAttr getI8IntegerAttr(int8_t value);
  IntegerAttr getI16IntegerAttr(int16_t value);
  IntegerAttr getI32IntegerAttr(int32_t value);
  IntegerAttr getI64IntegerAttr(int64_t value);
  IntegerAttr getIndexAttr(int64_t value);

  /// Signed and unsigned integer attribute getters.
  IntegerAttr getSI32IntegerAttr(int32_t value);
  IntegerAttr getUI32IntegerAttr(uint32_t value);

  /// Vector-typed DenseIntElementsAttr getters. `values` must not be empty.
  DenseIntElementsAttr getBoolVectorAttr(ArrayRef<bool> values);
  DenseIntElementsAttr getI32VectorAttr(ArrayRef<int32_t> values);
  DenseIntElementsAttr getI64VectorAttr(ArrayRef<int64_t> values);

  /// Tensor-typed DenseIntElementsAttr getters. `values` can be empty.
  /// These are generally preferable for representing general lists of integers
  /// as attributes.
  DenseIntElementsAttr getI32TensorAttr(ArrayRef<int32_t> values);
  DenseIntElementsAttr getI64TensorAttr(ArrayRef<int64_t> values);
  DenseIntElementsAttr getIndexTensorAttr(ArrayRef<int64_t> values);

  ArrayAttr getAffineMapArrayAttr(ArrayRef<AffineMap> values);
  ArrayAttr getBoolArrayAttr(ArrayRef<bool> values);
  ArrayAttr getI32ArrayAttr(ArrayRef<int32_t> values);
  ArrayAttr getI64ArrayAttr(ArrayRef<int64_t> values);
  ArrayAttr getIndexArrayAttr(ArrayRef<int64_t> values);
  ArrayAttr getF32ArrayAttr(ArrayRef<float> values);
  ArrayAttr getF64ArrayAttr(ArrayRef<double> values);
  ArrayAttr getStrArrayAttr(ArrayRef<StringRef> values);

  // Affine expressions and affine maps.
  AffineExpr getAffineDimExpr(unsigned position);
  AffineExpr getAffineSymbolExpr(unsigned position);
  AffineExpr getAffineConstantExpr(int64_t constant);

  // Special cases of affine maps and integer sets
  /// Returns a zero result affine map with no dimensions or symbols: () -> ().
  AffineMap getEmptyAffineMap();
  /// Returns a single constant result affine map with 0 dimensions and 0
  /// symbols.  One constant result: () -> (val).
  AffineMap getConstantAffineMap(int64_t val);
  // One dimension id identity map: (i) -> (i).
  AffineMap getDimIdentityMap();
  // Multi-dimensional identity map: (d0, d1, d2) -> (d0, d1, d2).
  AffineMap getMultiDimIdentityMap(unsigned rank);
  // One symbol identity map: ()[s] -> (s).
  AffineMap getSymbolIdentityMap();

  /// Returns a map that shifts its (single) input dimension by 'shift'.
  /// (d0) -> (d0 + shift)
  AffineMap getSingleDimShiftAffineMap(int64_t shift);

  /// Returns an affine map that is a translation (shift) of all result
  /// expressions in 'map' by 'shift'.
  /// Eg: input: (d0, d1)[s0] -> (d0, d1 + s0), shift = 2
  ///   returns:    (d0, d1)[s0] -> (d0 + 2, d1 + s0 + 2)
  AffineMap getShiftedAffineMap(AffineMap map, int64_t shift);

protected:
  MLIRContext *context;
};

/// This class helps build Operations. Operations that are created are
/// automatically inserted at an insertion point. The builder is copyable.
class OpBuilder : public Builder {
public:
  struct Listener;

  /// Create a builder with the given context.
  explicit OpBuilder(MLIRContext *ctx, Listener *listener = nullptr)
      : Builder(ctx), listener(listener) {}

  /// Create a builder and set the insertion point to the start of the region.
  explicit OpBuilder(Region *region, Listener *listener = nullptr)
      : OpBuilder(region->getContext(), listener) {
    if (!region->empty())
      setInsertionPoint(&region->front(), region->front().begin());
  }
  explicit OpBuilder(Region &region, Listener *listener = nullptr)
      : OpBuilder(&region, listener) {}

  /// Create a builder and set insertion point to the given operation, which
  /// will cause subsequent insertions to go right before it.
  explicit OpBuilder(Operation *op, Listener *listener = nullptr)
      : OpBuilder(op->getContext(), listener) {
    setInsertionPoint(op);
  }

  OpBuilder(Block *block, Block::iterator insertPoint,
            Listener *listener = nullptr)
      : OpBuilder(block->getParent()->getContext(), listener) {
    setInsertionPoint(block, insertPoint);
  }

  /// Create a builder and set the insertion point to before the first operation
  /// in the block but still inside the block.
  static OpBuilder atBlockBegin(Block *block, Listener *listener = nullptr) {
    return OpBuilder(block, block->begin(), listener);
  }

  /// Create a builder and set the insertion point to after the last operation
  /// in the block but still inside the block.
  static OpBuilder atBlockEnd(Block *block, Listener *listener = nullptr) {
    return OpBuilder(block, block->end(), listener);
  }

  /// Create a builder and set the insertion point to before the block
  /// terminator.
  static OpBuilder atBlockTerminator(Block *block,
                                     Listener *listener = nullptr) {
    auto *terminator = block->getTerminator();
    assert(terminator != nullptr && "the block has no terminator");
    return OpBuilder(block, Block::iterator(terminator), listener);
  }

  //===--------------------------------------------------------------------===//
  // Listeners
  //===--------------------------------------------------------------------===//

  /// This class represents a listener that may be used to hook into various
  /// actions within an OpBuilder.
  struct Listener {
    virtual ~Listener();

    /// Notification handler for when an operation is inserted into the builder.
    /// `op` is the operation that was inserted.
    virtual void notifyOperationInserted(Operation *op) {}

    /// Notification handler for when a block is created using the builder.
    /// `block` is the block that was created.
    virtual void notifyBlockCreated(Block *block) {}
  };

  /// Sets the listener of this builder to the one provided.
  void setListener(Listener *newListener) { listener = newListener; }

  /// Returns the current listener of this builder, or nullptr if this builder
  /// doesn't have a listener.
  Listener *getListener() const { return listener; }

  //===--------------------------------------------------------------------===//
  // Insertion Point Management
  //===--------------------------------------------------------------------===//

  /// This class represents a saved insertion point.
  class InsertPoint {
  public:
    /// Creates a new insertion point which doesn't point to anything.
    InsertPoint() = default;

    /// Creates a new insertion point at the given location.
    InsertPoint(Block *insertBlock, Block::iterator insertPt)
        : block(insertBlock), point(insertPt) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return (block != nullptr); }

    Block *getBlock() const { return block; }
    Block::iterator getPoint() const { return point; }

  private:
    Block *block = nullptr;
    Block::iterator point;
  };

  /// RAII guard to reset the insertion point of the builder when destroyed.
  class InsertionGuard {
  public:
    InsertionGuard(OpBuilder &builder)
        : builder(builder), ip(builder.saveInsertionPoint()) {}
    ~InsertionGuard() { builder.restoreInsertionPoint(ip); }

  private:
    OpBuilder &builder;
    OpBuilder::InsertPoint ip;
  };

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = Block::iterator();
  }

  /// Return a saved insertion point.
  InsertPoint saveInsertionPoint() const {
    return InsertPoint(getInsertionBlock(), getInsertionPoint());
  }

  /// Restore the insert point to a previously saved point.
  void restoreInsertionPoint(InsertPoint ip) {
    if (ip.isSet())
      setInsertionPoint(ip.getBlock(), ip.getPoint());
    else
      clearInsertionPoint();
  }

  /// Set the insertion point to the specified location.
  void setInsertionPoint(Block *block, Block::iterator insertPoint) {
    // TODO: check that insertPoint is in this rather than some other block.
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Sets the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void setInsertionPoint(Operation *op) {
    setInsertionPoint(op->getBlock(), Block::iterator(op));
  }

  /// Sets the insertion point to the node after the specified operation, which
  /// will cause subsequent insertions to go right after it.
  void setInsertionPointAfter(Operation *op) {
    setInsertionPoint(op->getBlock(), ++Block::iterator(op));
  }

  /// Sets the insertion point to the start of the specified block.
  void setInsertionPointToStart(Block *block) {
    setInsertionPoint(block, block->begin());
  }

  /// Sets the insertion point to the end of the specified block.
  void setInsertionPointToEnd(Block *block) {
    setInsertionPoint(block, block->end());
  }

  /// Return the block the current insertion point belongs to.  Note that the
  /// the insertion point is not necessarily the end of the block.
  Block *getInsertionBlock() const { return block; }

  /// Returns the current insertion point of the builder.
  Block::iterator getInsertionPoint() const { return insertPoint; }

  /// Returns the current block of the builder.
  Block *getBlock() const { return block; }

  //===--------------------------------------------------------------------===//
  // Block Creation
  //===--------------------------------------------------------------------===//

  /// Add new block with 'argTypes' arguments and set the insertion point to the
  /// end of it. The block is inserted at the provided insertion point of
  /// 'parent'.
  Block *createBlock(Region *parent, Region::iterator insertPt = {},
                     TypeRange argTypes = llvm::None);

  /// Add new block with 'argTypes' arguments and set the insertion point to the
  /// end of it. The block is placed before 'insertBefore'.
  Block *createBlock(Block *insertBefore, TypeRange argTypes = llvm::None);

  //===--------------------------------------------------------------------===//
  // Operation Creation
  //===--------------------------------------------------------------------===//

  /// Insert the given operation at the current insertion point and return it.
  Operation *insert(Operation *op);

  /// Creates an operation given the fields represented as an OperationState.
  Operation *createOperation(const OperationState &state);

  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&... args) {
    OperationState state(location, OpTy::getOperationName());
    if (!state.name.getAbstractOperation())
      llvm::report_fatal_error("Building op `" +
                               state.name.getStringRef().str() +
                               "` but it isn't registered in this MLIRContext");
    OpTy::build(*this, state, std::forward<Args>(args)...);
    auto *op = createOperation(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "builder didn't return the right type");
    return result;
  }

  /// Create an operation of specific op type at the current insertion point,
  /// and immediately try to fold it. This functions populates 'results' with
  /// the results after folding the operation.
  template <typename OpTy, typename... Args>
  void createOrFold(SmallVectorImpl<Value> &results, Location location,
                    Args &&... args) {
    // Create the operation without using 'createOperation' as we don't want to
    // insert it yet.
    OperationState state(location, OpTy::getOperationName());
    if (!state.name.getAbstractOperation())
      llvm::report_fatal_error("Building op `" +
                               state.name.getStringRef().str() +
                               "` but it isn't registered in this MLIRContext");
    OpTy::build(*this, state, std::forward<Args>(args)...);
    Operation *op = Operation::create(state);

    // Fold the operation. If successful destroy it, otherwise insert it.
    if (succeeded(tryFold(op, results)))
      op->destroy();
    else
      insert(op);
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<OpTrait::OneResult>(),
                          Value>::type
  createOrFold(Location location, Args &&... args) {
    SmallVector<Value, 1> results;
    createOrFold<OpTy>(results, location, std::forward<Args>(args)...);
    return results.front();
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<OpTrait::ZeroResult>(),
                          OpTy>::type
  createOrFold(Location location, Args &&... args) {
    auto op = create<OpTy>(location, std::forward<Args>(args)...);
    SmallVector<Value, 0> unused;
    tryFold(op.getOperation(), unused);

    // Folding cannot remove a zero-result operation, so for convenience we
    // continue to return it.
    return op;
  }

  /// Attempts to fold the given operation and places new results within
  /// 'results'. Returns success if the operation was folded, failure otherwise.
  /// Note: This function does not erase the operation on a successful fold.
  LogicalResult tryFold(Operation *op, SmallVectorImpl<Value> &results);

  /// Creates a deep copy of the specified operation, remapping any operands
  /// that use values outside of the operation using the map that is provided
  /// ( leaving them alone if no entry is present).  Replaces references to
  /// cloned sub-operations to the corresponding operation that is copied,
  /// and adds those mappings to the map.
  Operation *clone(Operation &op, BlockAndValueMapping &mapper) {
    return insert(op.clone(mapper));
  }
  Operation *clone(Operation &op) { return insert(op.clone()); }

  /// Creates a deep copy of this operation but keep the operation regions
  /// empty. Operands are remapped using `mapper` (if present), and `mapper` is
  /// updated to contain the results.
  Operation *cloneWithoutRegions(Operation &op, BlockAndValueMapping &mapper) {
    return insert(op.cloneWithoutRegions(mapper));
  }
  Operation *cloneWithoutRegions(Operation &op) {
    return insert(op.cloneWithoutRegions());
  }
  template <typename OpT> OpT cloneWithoutRegions(OpT op) {
    return cast<OpT>(cloneWithoutRegions(*op.getOperation()));
  }

private:
  /// The current block this builder is inserting into.
  Block *block = nullptr;
  /// The insertion point within the block that this builder is inserting
  /// before.
  Block::iterator insertPoint;
  /// The optional listener for events of this builder.
  Listener *listener;
};

} // namespace mlir

#endif
