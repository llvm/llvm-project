//===- Operation.h - MLIR Operation Class -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Operation class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATION_H
#define MLIR_IR_OPERATION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
/// Operation is a basic unit of execution within MLIR. Operations can
/// be nested within `Region`s held by other operations effectively forming a
/// tree. Child operations are organized into operation blocks represented by a
/// 'Block' class.
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, BlockOperand, Region,
                                    detail::OperandStorage> {
public:
  /// Create a new Operation with the specific fields.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           ArrayRef<NamedAttribute> attributes,
                           BlockRange successors, unsigned numRegions);

  /// Overload of create that takes an existing DictionaryAttr to avoid
  /// unnecessarily uniquing a list of attributes.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           DictionaryAttr attributes, BlockRange successors,
                           unsigned numRegions);

  /// Create a new Operation from the fields stored in `state`.
  static Operation *create(const OperationState &state);

  /// Create a new Operation with the specific fields.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           DictionaryAttr attributes,
                           BlockRange successors = {},
                           RegionRange regions = {});

  /// The name of an operation is the key identifier for it.
  OperationName getName() { return name; }

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() {
    return getName().getAbstractOperation();
  }

  /// Returns true if this operation has a registered operation description,
  /// otherwise false.
  bool isRegistered() { return getAbstractOperation(); }

  /// Remove this operation from its parent block and delete it.
  void erase();

  /// Remove the operation from its parent block, but don't delete it.
  void remove();

  /// Create a deep copy of this operation, remapping any operands that use
  /// values outside of the operation using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-operations to the corresponding operation that is copied, and adds
  /// those mappings to the map.
  Operation *clone(BlockAndValueMapping &mapper);
  Operation *clone();

  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  /// Operands are remapped using `mapper` (if present), and `mapper` is updated
  /// to contain the results.
  Operation *cloneWithoutRegions(BlockAndValueMapping &mapper);

  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  Operation *cloneWithoutRegions();

  /// Returns the operation block that contains this operation.
  Block *getBlock() { return block; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext();

  /// Return the dialect this operation is associated with, or nullptr if the
  /// associated dialect is not registered.
  Dialect *getDialect();

  /// The source location the operation was defined or derived from.
  Location getLoc() { return location; }

  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Returns the region to which the instruction belongs. Returns nullptr if
  /// the instruction is unlinked.
  Region *getParentRegion();

  /// Returns the closest surrounding operation that contains this operation
  /// or nullptr if this is a top-level operation.
  Operation *getParentOp();

  /// Return the closest surrounding parent operation that is of type 'OpTy'.
  template <typename OpTy> OpTy getParentOfType() {
    auto *op = this;
    while ((op = op->getParentOp()))
      if (auto parentOp = dyn_cast<OpTy>(op))
        return parentOp;
    return OpTy();
  }

  /// Returns the closest surrounding parent operation with trait `Trait`.
  template <template <typename T> class Trait>
  Operation *getParentWithTrait() {
    Operation *op = this;
    while ((op = op->getParentOp()))
      if (op->hasTrait<Trait>())
        return op;
    return nullptr;
  }

  /// Return true if this operation is a proper ancestor of the `other`
  /// operation.
  bool isProperAncestor(Operation *other);

  /// Return true if this operation is an ancestor of the `other` operation. An
  /// operation is considered as its own ancestor, use `isProperAncestor` to
  /// avoid this.
  bool isAncestor(Operation *other) {
    return this == other || isProperAncestor(other);
  }

  /// Replace any uses of 'from' with 'to' within this operation.
  void replaceUsesOfWith(Value from, Value to);

  /// Replace all uses of results of this operation with the provided 'values'.
  template <typename ValuesT>
  std::enable_if_t<!std::is_convertible<ValuesT, Operation *>::value>
  replaceAllUsesWith(ValuesT &&values) {
    assert(std::distance(values.begin(), values.end()) == getNumResults() &&
           "expected 'values' to correspond 1-1 with the number of results");

    auto valueIt = values.begin();
    for (unsigned i = 0, e = getNumResults(); i != e; ++i)
      getResult(i).replaceAllUsesWith(*(valueIt++));
  }

  /// Replace all uses of results of this operation with results of 'op'.
  void replaceAllUsesWith(Operation *op) {
    assert(getNumResults() == op->getNumResults());
    for (unsigned i = 0, e = getNumResults(); i != e; ++i)
      getResult(i).replaceAllUsesWith(op->getResult(i));
  }

  /// Destroys this operation and its subclass data.
  void destroy();

  /// This drops all operand uses from this operation, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Drop uses of all values defined by this operation or its nested regions.
  void dropAllDefinedValueUses();

  /// Unlink this operation from its current block and insert it right before
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveBefore(Operation *existingOp);

  /// Unlink this operation from its current block and insert it right before
  /// `iterator` in the specified block.
  void moveBefore(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// Unlink this operation from its current block and insert it right after
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveAfter(Operation *existingOp);

  /// Unlink this operation from its current block and insert it right after
  /// `iterator` in the specified block.
  void moveAfter(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// Given an operation 'other' that is within the same parent block, return
  /// whether the current operation is before 'other' in the operation list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of operations within the parent block.
  bool isBeforeInBlock(Operation *other);

  void print(raw_ostream &os, OpPrintingFlags flags = llvm::None);
  void print(raw_ostream &os, AsmState &state,
             OpPrintingFlags flags = llvm::None);
  void dump();

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Replace the current operands of this operation with the ones provided in
  /// 'operands'.
  void setOperands(ValueRange operands);

  /// Replace the operands beginning at 'start' and ending at 'start' + 'length'
  /// with the ones provided in 'operands'. 'operands' may be smaller or larger
  /// than the range pointed to by 'start'+'length'.
  void setOperands(unsigned start, unsigned length, ValueRange operands);

  /// Insert the given operands into the operand list at the given 'index'.
  void insertOperands(unsigned index, ValueRange operands);

  unsigned getNumOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().size() : 0;
  }

  Value getOperand(unsigned idx) { return getOpOperand(idx).get(); }
  void setOperand(unsigned idx, Value value) {
    return getOpOperand(idx).set(value);
  }

  /// Erase the operand at position `idx`.
  void eraseOperand(unsigned idx) { eraseOperands(idx); }

  /// Erase the operands starting at position `idx` and ending at position
  /// 'idx'+'length'.
  void eraseOperands(unsigned idx, unsigned length = 1) {
    getOperandStorage().eraseOperands(idx, length);
  }

  /// Erases the operands that have their corresponding bit set in
  /// `eraseIndices` and removes them from the operand list.
  void eraseOperands(const llvm::BitVector &eraseIndices) {
    getOperandStorage().eraseOperands(eraseIndices);
  }

  // Support operand iteration.
  using operand_range = OperandRange;
  using operand_iterator = operand_range::iterator;

  operand_iterator operand_begin() { return getOperands().begin(); }
  operand_iterator operand_end() { return getOperands().end(); }

  /// Returns an iterator on the underlying Value's.
  operand_range getOperands() { return operand_range(this); }

  MutableArrayRef<OpOperand> getOpOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().getOperands()
                                          : MutableArrayRef<OpOperand>();
  }

  OpOperand &getOpOperand(unsigned idx) { return getOpOperands()[idx]; }

  // Support operand type iteration.
  using operand_type_iterator = operand_range::type_iterator;
  using operand_type_range = operand_range::type_range;
  operand_type_iterator operand_type_begin() { return operand_begin(); }
  operand_type_iterator operand_type_end() { return operand_end(); }
  operand_type_range getOperandTypes() { return getOperands().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  /// Return the number of results held by this operation.
  unsigned getNumResults() { return numResults; }

  /// Get the 'idx'th result of this operation.
  OpResult getResult(unsigned idx) { return OpResult(getOpResultImpl(idx)); }

  /// Support result iteration.
  using result_range = ResultRange;
  using result_iterator = result_range::iterator;

  result_iterator result_begin() { return getResults().begin(); }
  result_iterator result_end() { return getResults().end(); }
  result_range getResults() {
    return numResults == 0 ? result_range(nullptr, 0)
                           : result_range(getInlineOpResult(0), numResults);
  }

  result_range getOpResults() { return getResults(); }
  OpResult getOpResult(unsigned idx) { return getResult(idx); }

  /// Support result type iteration.
  using result_type_iterator = result_range::type_iterator;
  using result_type_range = result_range::type_range;
  result_type_iterator result_type_begin() { return getResultTypes().begin(); }
  result_type_iterator result_type_end() { return getResultTypes().end(); }
  result_type_range getResultTypes() { return getResults().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Operations may optionally carry a list of attributes that associate
  // constants to names.  Attributes may be dynamically added and removed over
  // the lifetime of an operation.

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() { return attrs.getValue(); }

  /// Return all of the attributes on this operation as a DictionaryAttr.
  DictionaryAttr getAttrDictionary() { return attrs; }

  /// Set the attribute dictionary on this operation.
  void setAttrs(DictionaryAttr newAttrs) {
    assert(newAttrs && "expected valid attribute dictionary");
    attrs = newAttrs;
  }
  void setAttrs(ArrayRef<NamedAttribute> newAttrs) {
    setAttrs(DictionaryAttr::get(getContext(), newAttrs));
  }

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) { return attrs.get(name); }
  Attribute getAttr(StringRef name) { return attrs.get(name); }

  template <typename AttrClass> AttrClass getAttrOfType(Identifier name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }
  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  /// Return true if the operation has an attribute with the provided name,
  /// false otherwise.
  bool hasAttr(Identifier name) { return static_cast<bool>(getAttr(name)); }
  bool hasAttr(StringRef name) { return static_cast<bool>(getAttr(name)); }
  template <typename AttrClass, typename NameT>
  bool hasAttrOfType(NameT &&name) {
    return static_cast<bool>(
        getAttrOfType<AttrClass>(std::forward<NameT>(name)));
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute value) {
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(Identifier::get(name, getContext()), value);
  }

  /// Remove the attribute with the specified name if it exists. Return the
  /// attribute that was erased, or nullptr if there was no attribute with such
  /// name.
  Attribute removeAttr(Identifier name) {
    NamedAttrList attributes(attrs);
    Attribute removedAttr = attributes.erase(name);
    if (removedAttr)
      attrs = attributes.getDictionary(getContext());
    return removedAttr;
  }
  Attribute removeAttr(StringRef name) {
    return removeAttr(Identifier::get(name, getContext()));
  }

  /// A utility iterator that filters out non-dialect attributes.
  class dialect_attr_iterator
      : public llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                     bool (*)(NamedAttribute)> {
    static bool filter(NamedAttribute attr) {
      // Dialect attributes are prefixed by the dialect name, like operations.
      return attr.first.strref().count('.');
    }

    explicit dialect_attr_iterator(ArrayRef<NamedAttribute>::iterator it,
                                   ArrayRef<NamedAttribute>::iterator end)
        : llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                bool (*)(NamedAttribute)>(it, end, &filter) {}

    // Allow access to the constructor.
    friend Operation;
  };
  using dialect_attr_range = iterator_range<dialect_attr_iterator>;

  /// Return a range corresponding to the dialect attributes for this operation.
  dialect_attr_range getDialectAttrs() {
    auto attrs = getAttrs();
    return {dialect_attr_iterator(attrs.begin(), attrs.end()),
            dialect_attr_iterator(attrs.end(), attrs.end())};
  }
  dialect_attr_iterator dialect_attr_begin() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.begin(), attrs.end());
  }
  dialect_attr_iterator dialect_attr_end() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.end(), attrs.end());
  }

  /// Set the dialect attributes for this operation, and preserve all dependent.
  template <typename DialectAttrT>
  void setDialectAttrs(DialectAttrT &&dialectAttrs) {
    NamedAttrList attrs;
    attrs.append(std::begin(dialectAttrs), std::end(dialectAttrs));
    for (auto attr : getAttrs())
      if (!attr.first.strref().contains('.'))
        attrs.push_back(attr);
    setAttrs(attrs.getDictionary(getContext()));
  }

  //===--------------------------------------------------------------------===//
  // Blocks
  //===--------------------------------------------------------------------===//

  /// Returns the number of regions held by this operation.
  unsigned getNumRegions() { return numRegions; }

  /// Returns the regions held by this operation.
  MutableArrayRef<Region> getRegions() {
    auto *regions = getTrailingObjects<Region>();
    return {regions, numRegions};
  }

  /// Returns the region held by this operation at position 'index'.
  Region &getRegion(unsigned index) {
    assert(index < numRegions && "invalid region index");
    return getRegions()[index];
  }

  //===--------------------------------------------------------------------===//
  // Successors
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BlockOperand> getBlockOperands() {
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }

  // Successor iteration.
  using succ_iterator = SuccessorRange::iterator;
  succ_iterator successor_begin() { return getSuccessors().begin(); }
  succ_iterator successor_end() { return getSuccessors().end(); }
  SuccessorRange getSuccessors() { return SuccessorRange(this); }

  bool hasSuccessors() { return numSuccs != 0; }
  unsigned getNumSuccessors() { return numSuccs; }

  Block *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  void setSuccessor(Block *block, unsigned index);

  //===--------------------------------------------------------------------===//
  // Accessors for various properties of operations
  //===--------------------------------------------------------------------===//

  /// Attempt to fold this operation with the specified constant operand values
  /// - the elements in "operands" will correspond directly to the operands of
  /// the operation, but may be null if non-constant. If folding is successful,
  /// this fills in the `results` vector. If not, `results` is unspecified.
  LogicalResult fold(ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results);

  /// Returns true if the operation was registered with a particular trait, e.g.
  /// hasTrait<OperandsAreSignlessIntegerLike>().
  template <template <typename T> class Trait> bool hasTrait() {
    const AbstractOperation *abstractOp = getAbstractOperation();
    return abstractOp ? abstractOp->hasTrait<Trait>() : false;
  }

  /// Returns true if the operation is *might* have the provided trait. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide trait.
  template <template <typename T> class Trait> bool mightHaveTrait() {
    const AbstractOperation *abstractOp = getAbstractOperation();
    return abstractOp ? abstractOp->hasTrait<Trait>() : true;
  }

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operation by calling the callback for each nested operation
  /// (including this one), block or region, depending on the callback provided.
  /// Regions, blocks and operations at the same nesting level are visited in
  /// lexicographical order. The walk order for enclosing regions, blocks and
  /// operations with respect to their nested ones is specified by 'Order'
  /// (post-order by default). A callback on a block or operation is allowed to
  /// erase that block or operation if either:
  ///   * the walk is in post-order, or
  ///   * the walk is in pre-order and the walk is skipped after the erasure.
  ///
  /// The callback method can take any of the following forms:
  ///   void(Operation*) : Walk all operations opaquely.
  ///     * op->walk([](Operation *nestedOp) { ...});
  ///   void(OpT) : Walk all operations of the given derived type.
  ///     * op->walk([](ReturnOp returnOp) { ...});
  ///   WalkResult(Operation*|OpT) : Walk operations, but allow for
  ///                                interruption/skipping.
  ///     * op->walk([](... op) {
  ///         // Skip the walk of this op based on some invariant.
  ///         if (some_invariant)
  ///           return WalkResult::skip();
  ///         // Interrupt, i.e cancel, the walk based on some invariant.
  ///         if (another_invariant)
  ///           return WalkResult::interrupt();
  ///         return WalkResult::advance();
  ///       });
  template <WalkOrder Order = WalkOrder::PostOrder, typename FnT,
            typename RetT = detail::walkResultType<FnT>>
  RetT walk(FnT &&callback) {
    return detail::walk<Order>(this, std::forward<FnT>(callback));
  }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  /// Drop all uses of results of this operation.
  void dropAllUses() {
    for (OpResult result : getOpResults())
      result.dropAllUses();
  }

  /// This class implements a use iterator for the Operation. This iterates over
  /// all uses of all results.
  class UseIterator final
      : public llvm::iterator_facade_base<
            UseIterator, std::forward_iterator_tag, OpOperand> {
  public:
    /// Initialize UseIterator for op, specify end to return iterator to last
    /// use.
    explicit UseIterator(Operation *op, bool end = false);

    using llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                     OpOperand>::operator++;
    UseIterator &operator++();
    OpOperand *operator->() const { return use.getOperand(); }
    OpOperand &operator*() const { return *use.getOperand(); }

    bool operator==(const UseIterator &rhs) const { return use == rhs.use; }
    bool operator!=(const UseIterator &rhs) const { return !(*this == rhs); }

  private:
    void skipOverResultsWithNoUsers();

    /// The operation whose uses are being iterated over.
    Operation *op;
    /// The result of op who's uses are being iterated over.
    Operation::result_iterator res;
    /// The use of the result.
    Value::use_iterator use;
  };
  using use_iterator = UseIterator;
  using use_range = iterator_range<use_iterator>;

  use_iterator use_begin() { return use_iterator(this); }
  use_iterator use_end() { return use_iterator(this, /*end=*/true); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() { return {use_begin(), use_end()}; }

  /// Returns true if this operation has exactly one use.
  bool hasOneUse() { return llvm::hasSingleElement(getUses()); }

  /// Returns true if this operation has no uses.
  bool use_empty() {
    return llvm::all_of(getOpResults(),
                        [](OpResult result) { return result.use_empty(); });
  }

  /// Returns true if the results of this operation are used outside of the
  /// given block.
  bool isUsedOutsideOfBlock(Block *block) {
    return llvm::any_of(getOpResults(), [block](OpResult result) {
      return result.isUsedOutsideOfBlock(block);
    });
  }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() { return user_iterator(use_begin()); }
  user_iterator user_end() { return user_iterator(use_end()); }

  /// Returns a range of all users.
  user_range getUsers() { return {user_begin(), user_end()}; }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.
  InFlightDiagnostic emitOpError(const Twine &message = {});

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.
  InFlightDiagnostic emitError(const Twine &message = {});

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitWarning(const Twine &message = {});

  /// Emit a remark about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitRemark(const Twine &message = {});

private:
  //===--------------------------------------------------------------------===//
  // Ordering
  //===--------------------------------------------------------------------===//

  /// This value represents an invalid index ordering for an operation within a
  /// block.
  static constexpr unsigned kInvalidOrderIdx = -1;

  /// This value represents the stride to use when computing a new order for an
  /// operation.
  static constexpr unsigned kOrderStride = 5;

  /// Update the order index of this operation of this operation if necessary,
  /// potentially recomputing the order of the parent block.
  void updateOrderIfNecessary();

  /// Returns true if this operation has a valid order.
  bool hasValidOrder() { return orderIndex != kInvalidOrderIdx; }

private:
  Operation(Location location, OperationName name, unsigned numResults,
            unsigned numSuccessors, unsigned numRegions,
            DictionaryAttr attributes, bool hasOperandStorage);

  // Operations are deleted through the destroy() member because they are
  // allocated with malloc.
  ~Operation();

  /// Returns the additional size necessary for allocating the given objects
  /// before an Operation in-memory.
  static size_t prefixAllocSize(unsigned numOutOfLineResults,
                                unsigned numInlineResults) {
    return sizeof(detail::OutOfLineOpResult) * numOutOfLineResults +
           sizeof(detail::InlineOpResult) * numInlineResults;
  }
  /// Returns the additional size allocated before this Operation in-memory.
  size_t prefixAllocSize() {
    unsigned numResults = getNumResults();
    unsigned numOutOfLineResults = OpResult::getNumTrailing(numResults);
    unsigned numInlineResults = OpResult::getNumInline(numResults);
    return prefixAllocSize(numOutOfLineResults, numInlineResults);
  }

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    assert(hasOperandStorage && "expected operation to have operand storage");
    return *getTrailingObjects<detail::OperandStorage>();
  }

  /// Returns a pointer to the use list for the given out-of-line result.
  detail::OutOfLineOpResult *getOutOfLineOpResult(unsigned resultNumber) {
    // Out-of-line results are stored in reverse order after (before in memory)
    // the inline results.
    return reinterpret_cast<detail::OutOfLineOpResult *>(getInlineOpResult(
               detail::OpResultImpl::getMaxInlineResults() - 1)) -
           ++resultNumber;
  }

  /// Returns a pointer to the use list for the given inline result.
  detail::InlineOpResult *getInlineOpResult(unsigned resultNumber) {
    // Inline results are stored in reverse order before the operation in
    // memory.
    return reinterpret_cast<detail::InlineOpResult *>(this) - ++resultNumber;
  }

  /// Returns a pointer to the use list for the given result, which may be
  /// either inline or out-of-line.
  detail::OpResultImpl *getOpResultImpl(unsigned resultNumber) {
    unsigned maxInlineResults = detail::OpResultImpl::getMaxInlineResults();
    if (resultNumber < maxInlineResults)
      return getInlineOpResult(resultNumber);
    return getOutOfLineOpResult(resultNumber - maxInlineResults);
  }

  /// Provide a 'getParent' method for ilist_node_with_parent methods.
  /// We mark it as a const function because ilist_node_with_parent specifically
  /// requires a 'getParent() const' method. Once ilist_node removes this
  /// constraint, we should drop the const to fit the rest of the MLIR const
  /// model.
  Block *getParent() const { return block; }

  /// The operation block that contains this operation.
  Block *block = nullptr;

  /// This holds information about the source location the operation was defined
  /// or derived from.
  Location location;

  /// Relative order of this operation in its parent block. Used for
  /// O(1) local dominance checks between operations.
  mutable unsigned orderIndex = 0;

  const unsigned numResults;
  const unsigned numSuccs;
  const unsigned numRegions : 31;

  /// This bit signals whether this operation has an operand storage or not. The
  /// operand storage may be elided for operations that are known to never have
  /// operands.
  bool hasOperandStorage : 1;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  DictionaryAttr attrs;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Operation>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow value to access the 'ResultStorage' methods.
  friend class Value;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Operation, Block>;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Operation, BlockOperand, Region,
                               detail::OperandStorage>;
  size_t numTrailingObjects(OverloadToken<BlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<Region>) const { return numRegions; }
};

inline raw_ostream &operator<<(raw_ostream &os, Operation &op) {
  op.print(os, OpPrintingFlags().useLocalScope());
  return os;
}

} // end namespace mlir

namespace llvm {
/// Provide isa functionality for operation casts.
template <typename T> struct isa_impl<T, ::mlir::Operation> {
  static inline bool doit(const ::mlir::Operation &op) {
    return T::classof(const_cast<::mlir::Operation *>(&op));
  }
};

/// Provide specializations for operation casts as the resulting T is value
/// typed.
template <typename T> struct cast_retty_impl<T, ::mlir::Operation *> {
  using ret_type = T;
};
template <typename T> struct cast_retty_impl<T, ::mlir::Operation> {
  using ret_type = T;
};
template <class T>
struct cast_convert_val<T, ::mlir::Operation, ::mlir::Operation> {
  static T doit(::mlir::Operation &val) { return T(&val); }
};
template <class T>
struct cast_convert_val<T, ::mlir::Operation *, ::mlir::Operation *> {
  static T doit(::mlir::Operation *val) { return T(val); }
};
} // end namespace llvm

#endif // MLIR_IR_OPERATION_H
