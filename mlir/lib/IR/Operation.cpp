//===- Operation.cpp - Operation support code -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include <numeric>

using namespace mlir;

OpAsmParser::~OpAsmParser() {}

//===----------------------------------------------------------------------===//
// OperationName
//===----------------------------------------------------------------------===//

/// Form the OperationName for an op with the specified string.  This either is
/// a reference to an AbstractOperation if one is known, or a uniqued Identifier
/// if not.
OperationName::OperationName(StringRef name, MLIRContext *context) {
  if (auto *op = AbstractOperation::lookup(name, context))
    representation = op;
  else
    representation = Identifier::get(name, context);
}

/// Return the name of the dialect this operation is registered to.
StringRef OperationName::getDialectNamespace() const {
  if (Dialect *dialect = getDialect())
    return dialect->getNamespace();
  return representation.get<Identifier>().strref().split('.').first;
}

/// Return the operation name with dialect name stripped, if it has one.
StringRef OperationName::stripDialect() const {
  auto splitName = getStringRef().split(".");
  return splitName.second.empty() ? splitName.first : splitName.second;
}

/// Return the name of this operation. This always succeeds.
StringRef OperationName::getStringRef() const {
  return getIdentifier().strref();
}

/// Return the name of this operation as an identifier. This always succeeds.
Identifier OperationName::getIdentifier() const {
  if (auto *op = representation.dyn_cast<const AbstractOperation *>())
    return op->name;
  return representation.get<Identifier>();
}

OperationName OperationName::getFromOpaquePointer(const void *pointer) {
  return OperationName(
      RepresentationUnion::getFromOpaqueValue(const_cast<void *>(pointer)));
}

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

/// Create a new Operation with the specific fields.
Operation *Operation::create(Location location, OperationName name,
                             TypeRange resultTypes, ValueRange operands,
                             ArrayRef<NamedAttribute> attributes,
                             BlockRange successors, unsigned numRegions) {
  return create(location, name, resultTypes, operands,
                DictionaryAttr::get(location.getContext(), attributes),
                successors, numRegions);
}

/// Create a new Operation from operation state.
Operation *Operation::create(const OperationState &state) {
  return create(state.location, state.name, state.types, state.operands,
                state.attributes.getDictionary(state.getContext()),
                state.successors, state.regions);
}

/// Create a new Operation with the specific fields.
Operation *Operation::create(Location location, OperationName name,
                             TypeRange resultTypes, ValueRange operands,
                             DictionaryAttr attributes, BlockRange successors,
                             RegionRange regions) {
  unsigned numRegions = regions.size();
  Operation *op = create(location, name, resultTypes, operands, attributes,
                         successors, numRegions);
  for (unsigned i = 0; i < numRegions; ++i)
    if (regions[i])
      op->getRegion(i).takeBody(*regions[i]);
  return op;
}

/// Overload of create that takes an existing DictionaryAttr to avoid
/// unnecessarily uniquing a list of attributes.
Operation *Operation::create(Location location, OperationName name,
                             TypeRange resultTypes, ValueRange operands,
                             DictionaryAttr attributes, BlockRange successors,
                             unsigned numRegions) {
  // We only need to allocate additional memory for a subset of results.
  unsigned numTrailingResults = OpResult::getNumTrailing(resultTypes.size());
  unsigned numInlineResults = OpResult::getNumInline(resultTypes.size());
  unsigned numSuccessors = successors.size();
  unsigned numOperands = operands.size();

  // If the operation is known to have no operands, don't allocate an operand
  // storage.
  bool needsOperandStorage = true;
  if (operands.empty()) {
    if (const AbstractOperation *abstractOp = name.getAbstractOperation())
      needsOperandStorage = !abstractOp->hasTrait<OpTrait::ZeroOperands>();
  }

  // Compute the byte size for the operation and the operand storage. This takes
  // into account the size of the operation, its trailing objects, and its
  // prefixed objects.
  size_t byteSize =
      totalSizeToAlloc<BlockOperand, Region, Type, detail::OperandStorage>(
          numSuccessors, numRegions,
          // Result type storage only needed if there is not 0 or 1 results.
          resultTypes.size() == 1 ? 0 : resultTypes.size(),
          needsOperandStorage ? 1 : 0) +
      detail::OperandStorage::additionalAllocSize(numOperands);
  size_t prefixByteSize = llvm::alignTo(
      Operation::prefixAllocSize(numTrailingResults, numInlineResults),
      alignof(Operation));
  char *mallocMem = reinterpret_cast<char *>(malloc(byteSize + prefixByteSize));
  void *rawMem = mallocMem + prefixByteSize;

  // Create the new Operation.
  Operation *op =
      ::new (rawMem) Operation(location, name, resultTypes, numSuccessors,
                               numRegions, attributes, needsOperandStorage);

  assert((numSuccessors == 0 || op->mightHaveTrait<OpTrait::IsTerminator>()) &&
         "unexpected successors in a non-terminator operation");

  // Initialize the results.
  for (unsigned i = 0; i < numInlineResults; ++i)
    new (op->getInlineResult(i)) detail::InLineOpResult();
  for (unsigned i = 0; i < numTrailingResults; ++i)
    new (op->getTrailingResult(i)) detail::TrailingOpResult(i);

  // Initialize the regions.
  for (unsigned i = 0; i != numRegions; ++i)
    new (&op->getRegion(i)) Region(op);

  // Initialize the operands.
  if (needsOperandStorage)
    new (&op->getOperandStorage()) detail::OperandStorage(op, operands);

  // Initialize the successors.
  auto blockOperands = op->getBlockOperands();
  for (unsigned i = 0; i != numSuccessors; ++i)
    new (&blockOperands[i]) BlockOperand(op, successors[i]);

  return op;
}

Operation::Operation(Location location, OperationName name,
                     TypeRange resultTypes, unsigned numSuccessors,
                     unsigned numRegions, DictionaryAttr attributes,
                     bool hasOperandStorage)
    : location(location), numSuccs(numSuccessors), numRegions(numRegions),
      hasOperandStorage(hasOperandStorage), hasSingleResult(false), name(name),
      attrs(attributes) {
  assert(attributes && "unexpected null attribute dictionary");
  assert(llvm::all_of(resultTypes, [](Type t) { return t; }) &&
         "unexpected null result type");
  if (resultTypes.empty()) {
    resultTypeOrSize.size = 0;
  } else {
    // If there is a single result it is stored in-place, otherwise use trailing
    // type storage.
    hasSingleResult = resultTypes.size() == 1;
    if (hasSingleResult) {
      resultTypeOrSize.type = resultTypes.front();
    } else {
      resultTypeOrSize.size = resultTypes.size();
      llvm::copy(resultTypes, getTrailingObjects<Type>());
    }
  }
}

// Operations are deleted through the destroy() member because they are
// allocated via malloc.
Operation::~Operation() {
  assert(block == nullptr && "operation destroyed but still in a block");

  // Explicitly run the destructors for the operands.
  if (hasOperandStorage)
    getOperandStorage().~OperandStorage();

  // Explicitly run the destructors for the successors.
  for (auto &successor : getBlockOperands())
    successor.~BlockOperand();

  // Explicitly destroy the regions.
  for (auto &region : getRegions())
    region.~Region();
}

/// Destroy this operation or one of its subclasses.
void Operation::destroy() {
  // Operations may have additional prefixed allocation, which needs to be
  // accounted for here when computing the address to free.
  char *rawMem = reinterpret_cast<char *>(this) -
                 llvm::alignTo(prefixAllocSize(), alignof(Operation));
  this->~Operation();
  free(rawMem);
}

/// Return the context this operation is associated with.
MLIRContext *Operation::getContext() { return location->getContext(); }

/// Return the dialect this operation is associated with, or nullptr if the
/// associated dialect is not registered.
Dialect *Operation::getDialect() { return getName().getDialect(); }

Region *Operation::getParentRegion() {
  return block ? block->getParent() : nullptr;
}

Operation *Operation::getParentOp() {
  return block ? block->getParentOp() : nullptr;
}

/// Return true if this operation is a proper ancestor of the `other`
/// operation.
bool Operation::isProperAncestor(Operation *other) {
  while ((other = other->getParentOp()))
    if (this == other)
      return true;
  return false;
}

/// Replace any uses of 'from' with 'to' within this operation.
void Operation::replaceUsesOfWith(Value from, Value to) {
  if (from == to)
    return;
  for (auto &operand : getOpOperands())
    if (operand.get() == from)
      operand.set(to);
}

/// Replace the current operands of this operation with the ones provided in
/// 'operands'.
void Operation::setOperands(ValueRange operands) {
  if (LLVM_LIKELY(hasOperandStorage))
    return getOperandStorage().setOperands(this, operands);
  assert(operands.empty() && "setting operands without an operand storage");
}

/// Replace the operands beginning at 'start' and ending at 'start' + 'length'
/// with the ones provided in 'operands'. 'operands' may be smaller or larger
/// than the range pointed to by 'start'+'length'.
void Operation::setOperands(unsigned start, unsigned length,
                            ValueRange operands) {
  assert((start + length) <= getNumOperands() &&
         "invalid operand range specified");
  if (LLVM_LIKELY(hasOperandStorage))
    return getOperandStorage().setOperands(this, start, length, operands);
  assert(operands.empty() && "setting operands without an operand storage");
}

/// Insert the given operands into the operand list at the given 'index'.
void Operation::insertOperands(unsigned index, ValueRange operands) {
  if (LLVM_LIKELY(hasOperandStorage))
    return setOperands(index, /*length=*/0, operands);
  assert(operands.empty() && "inserting operands without an operand storage");
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.
InFlightDiagnostic Operation::emitError(const Twine &message) {
  InFlightDiagnostic diag = mlir::emitError(getLoc(), message);
  if (getContext()->shouldPrintOpOnDiagnostic()) {
    // Print out the operation explicitly here so that we can print the generic
    // form.
    // TODO: It would be nice if we could instead provide the
    // specific printing flags when adding the operation as an argument to the
    // diagnostic.
    std::string printedOp;
    {
      llvm::raw_string_ostream os(printedOp);
      print(os, OpPrintingFlags().printGenericOpForm().useLocalScope());
    }
    diag.attachNote(getLoc()) << "see current operation: " << printedOp;
  }
  return diag;
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Operation::emitWarning(const Twine &message) {
  InFlightDiagnostic diag = mlir::emitWarning(getLoc(), message);
  if (getContext()->shouldPrintOpOnDiagnostic())
    diag.attachNote(getLoc()) << "see current operation: " << *this;
  return diag;
}

/// Emit a remark about this operation, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Operation::emitRemark(const Twine &message) {
  InFlightDiagnostic diag = mlir::emitRemark(getLoc(), message);
  if (getContext()->shouldPrintOpOnDiagnostic())
    diag.attachNote(getLoc()) << "see current operation: " << *this;
  return diag;
}

//===----------------------------------------------------------------------===//
// Operation Ordering
//===----------------------------------------------------------------------===//

constexpr unsigned Operation::kInvalidOrderIdx;
constexpr unsigned Operation::kOrderStride;

/// Given an operation 'other' that is within the same parent block, return
/// whether the current operation is before 'other' in the operation list
/// of the parent block.
/// Note: This function has an average complexity of O(1), but worst case may
/// take O(N) where N is the number of operations within the parent block.
bool Operation::isBeforeInBlock(Operation *other) {
  assert(block && "Operations without parent blocks have no order.");
  assert(other && other->block == block &&
         "Expected other operation to have the same parent block.");
  // If the order of the block is already invalid, directly recompute the
  // parent.
  if (!block->isOpOrderValid()) {
    block->recomputeOpOrder();
  } else {
    // Update the order either operation if necessary.
    updateOrderIfNecessary();
    other->updateOrderIfNecessary();
  }

  return orderIndex < other->orderIndex;
}

/// Update the order index of this operation of this operation if necessary,
/// potentially recomputing the order of the parent block.
void Operation::updateOrderIfNecessary() {
  assert(block && "expected valid parent");

  // If the order is valid for this operation there is nothing to do.
  if (hasValidOrder())
    return;
  Operation *blockFront = &block->front();
  Operation *blockBack = &block->back();

  // This method is expected to only be invoked on blocks with more than one
  // operation.
  assert(blockFront != blockBack && "expected more than one operation");

  // If the operation is at the end of the block.
  if (this == blockBack) {
    Operation *prevNode = getPrevNode();
    if (!prevNode->hasValidOrder())
      return block->recomputeOpOrder();

    // Add the stride to the previous operation.
    orderIndex = prevNode->orderIndex + kOrderStride;
    return;
  }

  // If this is the first operation try to use the next operation to compute the
  // ordering.
  if (this == blockFront) {
    Operation *nextNode = getNextNode();
    if (!nextNode->hasValidOrder())
      return block->recomputeOpOrder();
    // There is no order to give this operation.
    if (nextNode->orderIndex == 0)
      return block->recomputeOpOrder();

    // If we can't use the stride, just take the middle value left. This is safe
    // because we know there is at least one valid index to assign to.
    if (nextNode->orderIndex <= kOrderStride)
      orderIndex = (nextNode->orderIndex / 2);
    else
      orderIndex = kOrderStride;
    return;
  }

  // Otherwise, this operation is between two others. Place this operation in
  // the middle of the previous and next if possible.
  Operation *prevNode = getPrevNode(), *nextNode = getNextNode();
  if (!prevNode->hasValidOrder() || !nextNode->hasValidOrder())
    return block->recomputeOpOrder();
  unsigned prevOrder = prevNode->orderIndex, nextOrder = nextNode->orderIndex;

  // Check to see if there is a valid order between the two.
  if (prevOrder + 1 == nextOrder)
    return block->recomputeOpOrder();
  orderIndex = prevOrder + ((nextOrder - prevOrder) / 2);
}

//===----------------------------------------------------------------------===//
// ilist_traits for Operation
//===----------------------------------------------------------------------===//

auto llvm::ilist_detail::SpecificNodeAccess<
    typename llvm::ilist_detail::compute_node_options<
        ::mlir::Operation>::type>::getNodePtr(pointer N) -> node_type * {
  return NodeAccess::getNodePtr<OptionsT>(N);
}

auto llvm::ilist_detail::SpecificNodeAccess<
    typename llvm::ilist_detail::compute_node_options<
        ::mlir::Operation>::type>::getNodePtr(const_pointer N)
    -> const node_type * {
  return NodeAccess::getNodePtr<OptionsT>(N);
}

auto llvm::ilist_detail::SpecificNodeAccess<
    typename llvm::ilist_detail::compute_node_options<
        ::mlir::Operation>::type>::getValuePtr(node_type *N) -> pointer {
  return NodeAccess::getValuePtr<OptionsT>(N);
}

auto llvm::ilist_detail::SpecificNodeAccess<
    typename llvm::ilist_detail::compute_node_options<
        ::mlir::Operation>::type>::getValuePtr(const node_type *N)
    -> const_pointer {
  return NodeAccess::getValuePtr<OptionsT>(N);
}

void llvm::ilist_traits<::mlir::Operation>::deleteNode(Operation *op) {
  op->destroy();
}

Block *llvm::ilist_traits<::mlir::Operation>::getContainingBlock() {
  size_t Offset(size_t(&((Block *)nullptr->*Block::getSublistAccess(nullptr))));
  iplist<Operation> *Anchor(static_cast<iplist<Operation> *>(this));
  return reinterpret_cast<Block *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when an operation is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Operation>::addNodeToList(Operation *op) {
  assert(!op->getBlock() && "already in an operation block!");
  op->block = getContainingBlock();

  // Invalidate the order on the operation.
  op->orderIndex = Operation::kInvalidOrderIdx;
}

/// This is a trait method invoked when an operation is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Operation>::removeNodeFromList(Operation *op) {
  assert(op->block && "not already in an operation block!");
  op->block = nullptr;
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Operation>::transferNodesFromList(
    ilist_traits<Operation> &otherList, op_iterator first, op_iterator last) {
  Block *curParent = getContainingBlock();

  // Invalidate the ordering of the parent block.
  curParent->invalidateOpOrder();

  // If we are transferring operations within the same block, the block
  // pointer doesn't need to be updated.
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each operation.
  for (; first != last; ++first)
    first->block = curParent;
}

/// Remove this operation (and its descendants) from its Block and delete
/// all of them.
void Operation::erase() {
  if (auto *parent = getBlock())
    parent->getOperations().erase(this);
  else
    destroy();
}

/// Remove the operation from its parent block, but don't delete it.
void Operation::remove() {
  if (Block *parent = getBlock())
    parent->getOperations().remove(this);
}

/// Unlink this operation from its current block and insert it right before
/// `existingOp` which may be in the same or another block in the same
/// function.
void Operation::moveBefore(Operation *existingOp) {
  moveBefore(existingOp->getBlock(), existingOp->getIterator());
}

/// Unlink this operation from its current basic block and insert it right
/// before `iterator` in the specified basic block.
void Operation::moveBefore(Block *block,
                           llvm::iplist<Operation>::iterator iterator) {
  block->getOperations().splice(iterator, getBlock()->getOperations(),
                                getIterator());
}

/// Unlink this operation from its current block and insert it right after
/// `existingOp` which may be in the same or another block in the same function.
void Operation::moveAfter(Operation *existingOp) {
  moveAfter(existingOp->getBlock(), existingOp->getIterator());
}

/// Unlink this operation from its current block and insert it right after
/// `iterator` in the specified block.
void Operation::moveAfter(Block *block,
                          llvm::iplist<Operation>::iterator iterator) {
  assert(iterator != block->end() && "cannot move after end of block");
  moveBefore(&*std::next(iterator));
}

/// This drops all operand uses from this operation, which is an essential
/// step in breaking cyclic dependences between references when they are to
/// be deleted.
void Operation::dropAllReferences() {
  for (auto &op : getOpOperands())
    op.drop();

  for (auto &region : getRegions())
    region.dropAllReferences();

  for (auto &dest : getBlockOperands())
    dest.drop();
}

/// This drops all uses of any values defined by this operation or its nested
/// regions, wherever they are located.
void Operation::dropAllDefinedValueUses() {
  dropAllUses();

  for (auto &region : getRegions())
    for (auto &block : region)
      block.dropAllDefinedValueUses();
}

/// Return the number of results held by this operation.
unsigned Operation::getNumResults() {
  if (hasSingleResult)
    return 1;
  return resultTypeOrSize.size;
}

auto Operation::getResultTypes() -> result_type_range {
  if (hasSingleResult)
    return resultTypeOrSize.type;
  return ArrayRef<Type>(getTrailingObjects<Type>(), resultTypeOrSize.size);
}

void Operation::setSuccessor(Block *block, unsigned index) {
  assert(index < getNumSuccessors());
  getBlockOperands()[index].set(block);
}

/// Attempt to fold this operation using the Op's registered foldHook.
LogicalResult Operation::fold(ArrayRef<Attribute> operands,
                              SmallVectorImpl<OpFoldResult> &results) {
  // If we have a registered operation definition matching this one, use it to
  // try to constant fold the operation.
  auto *abstractOp = getAbstractOperation();
  if (abstractOp && succeeded(abstractOp->foldHook(this, operands, results)))
    return success();

  // Otherwise, fall back on the dialect hook to handle it.
  Dialect *dialect = getDialect();
  if (!dialect)
    return failure();

  auto *interface = dialect->getRegisteredInterface<DialectFoldInterface>();
  if (!interface)
    return failure();

  return interface->fold(this, operands, results);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
InFlightDiagnostic Operation::emitOpError(const Twine &message) {
  return emitError() << "'" << getName() << "' op " << message;
}

//===----------------------------------------------------------------------===//
// Operation Cloning
//===----------------------------------------------------------------------===//

/// Create a deep copy of this operation but keep the operation regions empty.
/// Operands are remapped using `mapper` (if present), and `mapper` is updated
/// to contain the results.
Operation *Operation::cloneWithoutRegions(BlockAndValueMapping &mapper) {
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;

  // Remap the operands.
  operands.reserve(getNumOperands());
  for (auto opValue : getOperands())
    operands.push_back(mapper.lookupOrDefault(opValue));

  // Remap the successors.
  successors.reserve(getNumSuccessors());
  for (Block *successor : getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto *newOp = create(getLoc(), getName(), getResultTypes(), operands, attrs,
                       successors, getNumRegions());

  // Remember the mapping of any results.
  for (unsigned i = 0, e = getNumResults(); i != e; ++i)
    mapper.map(getResult(i), newOp->getResult(i));

  return newOp;
}

Operation *Operation::cloneWithoutRegions() {
  BlockAndValueMapping mapper;
  return cloneWithoutRegions(mapper);
}

/// Create a deep copy of this operation, remapping any operands that use
/// values outside of the operation using the map that is provided (leaving
/// them alone if no entry is present).  Replaces references to cloned
/// sub-operations to the corresponding operation that is copied, and adds
/// those mappings to the map.
Operation *Operation::clone(BlockAndValueMapping &mapper) {
  auto *newOp = cloneWithoutRegions(mapper);

  // Clone the regions.
  for (unsigned i = 0; i != numRegions; ++i)
    getRegion(i).cloneInto(&newOp->getRegion(i), mapper);

  return newOp;
}

Operation *Operation::clone() {
  BlockAndValueMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
// OpState trait class.
//===----------------------------------------------------------------------===//

// The fallback for the parser is to reject the custom assembly form.
ParseResult OpState::parse(OpAsmParser &parser, OperationState &result) {
  return parser.emitError(parser.getNameLoc(), "has no custom assembly form");
}

// The fallback for the printer is to print in the generic assembly form.
void OpState::print(Operation *op, OpAsmPrinter &p) { p.printGenericOp(op); }

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.
InFlightDiagnostic OpState::emitError(const Twine &message) {
  return getOperation()->emitError(message);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
InFlightDiagnostic OpState::emitOpError(const Twine &message) {
  return getOperation()->emitOpError(message);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic OpState::emitWarning(const Twine &message) {
  return getOperation()->emitWarning(message);
}

/// Emit a remark about this operation, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic OpState::emitRemark(const Twine &message) {
  return getOperation()->emitRemark(message);
}

//===----------------------------------------------------------------------===//
// Op Trait implementations
//===----------------------------------------------------------------------===//

OpFoldResult OpTrait::impl::foldIdempotent(Operation *op) {
  auto *argumentOp = op->getOperand(0).getDefiningOp();
  if (argumentOp && op->getName() == argumentOp->getName()) {
    // Replace the outer operation output with the inner operation.
    return op->getOperand(0);
  }

  return {};
}

OpFoldResult OpTrait::impl::foldInvolution(Operation *op) {
  auto *argumentOp = op->getOperand(0).getDefiningOp();
  if (argumentOp && op->getName() == argumentOp->getName()) {
    // Replace the outer involutions output with inner's input.
    return argumentOp->getOperand(0);
  }

  return {};
}

LogicalResult OpTrait::impl::verifyZeroOperands(Operation *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError() << "requires zero operands";
  return success();
}

LogicalResult OpTrait::impl::verifyOneOperand(Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError() << "requires a single operand";
  return success();
}

LogicalResult OpTrait::impl::verifyNOperands(Operation *op,
                                             unsigned numOperands) {
  if (op->getNumOperands() != numOperands) {
    return op->emitOpError() << "expected " << numOperands
                             << " operands, but found " << op->getNumOperands();
  }
  return success();
}

LogicalResult OpTrait::impl::verifyAtLeastNOperands(Operation *op,
                                                    unsigned numOperands) {
  if (op->getNumOperands() < numOperands)
    return op->emitOpError()
           << "expected " << numOperands << " or more operands";
  return success();
}

/// If this is a vector type, or a tensor type, return the scalar element type
/// that it is built around, otherwise return the type unmodified.
static Type getTensorOrVectorElementType(Type type) {
  if (auto vec = type.dyn_cast<VectorType>())
    return vec.getElementType();

  // Look through tensor<vector<...>> to find the underlying element type.
  if (auto tensor = type.dyn_cast<TensorType>())
    return getTensorOrVectorElementType(tensor.getElementType());
  return type;
}

LogicalResult OpTrait::impl::verifyIsIdempotent(Operation *op) {
  // FIXME: Add back check for no side effects on operation.
  // Currently adding it would cause the shared library build
  // to fail since there would be a dependency of IR on SideEffectInterfaces
  // which is cyclical.
  return success();
}

LogicalResult OpTrait::impl::verifyIsInvolution(Operation *op) {
  // FIXME: Add back check for no side effects on operation.
  // Currently adding it would cause the shared library build
  // to fail since there would be a dependency of IR on SideEffectInterfaces
  // which is cyclical.
  return success();
}

LogicalResult
OpTrait::impl::verifyOperandsAreSignlessIntegerLike(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    auto type = getTensorOrVectorElementType(opType);
    if (!type.isSignlessIntOrIndex())
      return op->emitOpError() << "requires an integer or index type";
  }
  return success();
}

LogicalResult OpTrait::impl::verifyOperandsAreFloatLike(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    auto type = getTensorOrVectorElementType(opType);
    if (!type.isa<FloatType>())
      return op->emitOpError("requires a float type");
  }
  return success();
}

LogicalResult OpTrait::impl::verifySameTypeOperands(Operation *op) {
  // Zero or one operand always have the "same" type.
  unsigned nOperands = op->getNumOperands();
  if (nOperands < 2)
    return success();

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1))
    if (opType != type)
      return op->emitOpError() << "requires all operands to have the same type";
  return success();
}

LogicalResult OpTrait::impl::verifyZeroRegion(Operation *op) {
  if (op->getNumRegions() != 0)
    return op->emitOpError() << "requires zero regions";
  return success();
}

LogicalResult OpTrait::impl::verifyOneRegion(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitOpError() << "requires one region";
  return success();
}

LogicalResult OpTrait::impl::verifyNRegions(Operation *op,
                                            unsigned numRegions) {
  if (op->getNumRegions() != numRegions)
    return op->emitOpError() << "expected " << numRegions << " regions";
  return success();
}

LogicalResult OpTrait::impl::verifyAtLeastNRegions(Operation *op,
                                                   unsigned numRegions) {
  if (op->getNumRegions() < numRegions)
    return op->emitOpError() << "expected " << numRegions << " or more regions";
  return success();
}

LogicalResult OpTrait::impl::verifyZeroResult(Operation *op) {
  if (op->getNumResults() != 0)
    return op->emitOpError() << "requires zero results";
  return success();
}

LogicalResult OpTrait::impl::verifyOneResult(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError() << "requires one result";
  return success();
}

LogicalResult OpTrait::impl::verifyNResults(Operation *op,
                                            unsigned numOperands) {
  if (op->getNumResults() != numOperands)
    return op->emitOpError() << "expected " << numOperands << " results";
  return success();
}

LogicalResult OpTrait::impl::verifyAtLeastNResults(Operation *op,
                                                   unsigned numOperands) {
  if (op->getNumResults() < numOperands)
    return op->emitOpError()
           << "expected " << numOperands << " or more results";
  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1)) {
    if (failed(verifyCompatibleShape(opType, type)))
      return op->emitOpError() << "requires the same shape for all operands";
  }
  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultShape(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto type = op->getOperand(0).getType();
  for (auto resultType : op->getResultTypes()) {
    if (failed(verifyCompatibleShape(resultType, type)))
      return op->emitOpError()
             << "requires the same shape for all operands and results";
  }
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1)) {
    if (failed(verifyCompatibleShape(opType, type)))
      return op->emitOpError()
             << "requires the same shape for all operands and results";
  }
  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsElementType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();
  auto elementType = getElementTypeOrSelf(op->getOperand(0));

  for (auto operand : llvm::drop_begin(op->getOperands(), 1)) {
    if (getElementTypeOrSelf(operand) != elementType)
      return op->emitOpError("requires the same element type for all operands");
  }

  return success();
}

LogicalResult
OpTrait::impl::verifySameOperandsAndResultElementType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementTypeOrSelf(op->getResult(0));

  // Verify result element type matches first result's element type.
  for (auto result : llvm::drop_begin(op->getResults(), 1)) {
    if (getElementTypeOrSelf(result) != elementType)
      return op->emitOpError(
          "requires the same element type for all operands and results");
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->getOperands()) {
    if (getElementTypeOrSelf(operand) != elementType)
      return op->emitOpError(
          "requires the same element type for all operands and results");
  }

  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto type = op->getResult(0).getType();
  auto elementType = getElementTypeOrSelf(type);
  for (auto resultType : op->getResultTypes().drop_front(1)) {
    if (getElementTypeOrSelf(resultType) != elementType ||
        failed(verifyCompatibleShape(resultType, type)))
      return op->emitOpError()
             << "requires the same type for all operands and results";
  }
  for (auto opType : op->getOperandTypes()) {
    if (getElementTypeOrSelf(opType) != elementType ||
        failed(verifyCompatibleShape(opType, type)))
      return op->emitOpError()
             << "requires the same type for all operands and results";
  }
  return success();
}

LogicalResult OpTrait::impl::verifyIsTerminator(Operation *op) {
  Block *block = op->getBlock();
  // Verify that the operation is at the end of the respective parent block.
  if (!block || &block->back() != op)
    return op->emitOpError("must be the last operation in the parent block");
  return success();
}

static LogicalResult verifyTerminatorSuccessors(Operation *op) {
  auto *parent = op->getParentRegion();

  // Verify that the operands lines up with the BB arguments in the successor.
  for (Block *succ : op->getSuccessors())
    if (succ->getParent() != parent)
      return op->emitError("reference to block defined in another region");
  return success();
}

LogicalResult OpTrait::impl::verifyZeroSuccessor(Operation *op) {
  if (op->getNumSuccessors() != 0) {
    return op->emitOpError("requires 0 successors but found ")
           << op->getNumSuccessors();
  }
  return success();
}

LogicalResult OpTrait::impl::verifyOneSuccessor(Operation *op) {
  if (op->getNumSuccessors() != 1) {
    return op->emitOpError("requires 1 successor but found ")
           << op->getNumSuccessors();
  }
  return verifyTerminatorSuccessors(op);
}
LogicalResult OpTrait::impl::verifyNSuccessors(Operation *op,
                                               unsigned numSuccessors) {
  if (op->getNumSuccessors() != numSuccessors) {
    return op->emitOpError("requires ")
           << numSuccessors << " successors but found "
           << op->getNumSuccessors();
  }
  return verifyTerminatorSuccessors(op);
}
LogicalResult OpTrait::impl::verifyAtLeastNSuccessors(Operation *op,
                                                      unsigned numSuccessors) {
  if (op->getNumSuccessors() < numSuccessors) {
    return op->emitOpError("requires at least ")
           << numSuccessors << " successors but found "
           << op->getNumSuccessors();
  }
  return verifyTerminatorSuccessors(op);
}

LogicalResult OpTrait::impl::verifyResultsAreBoolLike(Operation *op) {
  for (auto resultType : op->getResultTypes()) {
    auto elementType = getTensorOrVectorElementType(resultType);
    bool isBoolType = elementType.isInteger(1);
    if (!isBoolType)
      return op->emitOpError() << "requires a bool result type";
  }

  return success();
}

LogicalResult OpTrait::impl::verifyResultsAreFloatLike(Operation *op) {
  for (auto resultType : op->getResultTypes())
    if (!getTensorOrVectorElementType(resultType).isa<FloatType>())
      return op->emitOpError() << "requires a floating point type";

  return success();
}

LogicalResult
OpTrait::impl::verifyResultsAreSignlessIntegerLike(Operation *op) {
  for (auto resultType : op->getResultTypes())
    if (!getTensorOrVectorElementType(resultType).isSignlessIntOrIndex())
      return op->emitOpError() << "requires an integer or index type";
  return success();
}

static LogicalResult verifyValueSizeAttr(Operation *op, StringRef attrName,
                                         bool isOperand) {
  auto sizeAttr = op->getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return op->emitOpError("requires 1D vector attribute '") << attrName << "'";

  auto sizeAttrType = sizeAttr.getType().dyn_cast<VectorType>();
  if (!sizeAttrType || sizeAttrType.getRank() != 1)
    return op->emitOpError("requires 1D vector attribute '") << attrName << "'";

  if (llvm::any_of(sizeAttr.getIntValues(), [](const APInt &element) {
        return !element.isNonNegative();
      }))
    return op->emitOpError("'")
           << attrName << "' attribute cannot have negative elements";

  size_t totalCount = std::accumulate(
      sizeAttr.begin(), sizeAttr.end(), 0,
      [](unsigned all, APInt one) { return all + one.getZExtValue(); });

  if (isOperand && totalCount != op->getNumOperands())
    return op->emitOpError("operand count (")
           << op->getNumOperands() << ") does not match with the total size ("
           << totalCount << ") specified in attribute '" << attrName << "'";
  else if (!isOperand && totalCount != op->getNumResults())
    return op->emitOpError("result count (")
           << op->getNumResults() << ") does not match with the total size ("
           << totalCount << ") specified in attribute '" << attrName << "'";
  return success();
}

LogicalResult OpTrait::impl::verifyOperandSizeAttr(Operation *op,
                                                   StringRef attrName) {
  return verifyValueSizeAttr(op, attrName, /*isOperand=*/true);
}

LogicalResult OpTrait::impl::verifyResultSizeAttr(Operation *op,
                                                  StringRef attrName) {
  return verifyValueSizeAttr(op, attrName, /*isOperand=*/false);
}

LogicalResult OpTrait::impl::verifyNoRegionArguments(Operation *op) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;

    if (region.getNumArguments() != 0) {
      if (op->getNumRegions() > 1)
        return op->emitOpError("region #")
               << region.getRegionNumber() << " should have no arguments";
      else
        return op->emitOpError("region should have no arguments");
    }
  }
  return success();
}

/// Checks if two ShapedTypes are the same, ignoring the element type.
static bool areSameShapedTypeIgnoringElementType(ShapedType a, ShapedType b) {
  if (a.getTypeID() != b.getTypeID())
    return false;
  if (!a.hasRank())
    return !b.hasRank();
  return a.getShape() == b.getShape();
}

LogicalResult OpTrait::impl::verifyElementwiseMappable(Operation *op) {
  auto isMappableType = [](Type type) {
    return type.isa<VectorType, TensorType>();
  };
  auto resultMappableTypes = llvm::to_vector<1>(
      llvm::make_filter_range(op->getResultTypes(), isMappableType));
  auto operandMappableTypes = llvm::to_vector<2>(
      llvm::make_filter_range(op->getOperandTypes(), isMappableType));

  // If the op only has scalar operand/result types, then we have nothing to
  // check.
  if (resultMappableTypes.empty() && operandMappableTypes.empty())
    return success();

  if (!resultMappableTypes.empty() && operandMappableTypes.empty())
    return op->emitOpError("if a result is non-scalar, then at least one "
                           "operand must be non-scalar");

  assert(!operandMappableTypes.empty());

  if (resultMappableTypes.empty())
    return op->emitOpError("if an operand is non-scalar, then there must be at "
                           "least one non-scalar result");

  if (resultMappableTypes.size() != op->getNumResults())
    return op->emitOpError(
        "if an operand is non-scalar, then all results must be non-scalar");

  auto mustMatchType = operandMappableTypes[0].cast<ShapedType>();
  for (auto type :
       llvm::concat<Type>(resultMappableTypes, operandMappableTypes)) {
    if (!areSameShapedTypeIgnoringElementType(type.cast<ShapedType>(),
                                              mustMatchType)) {
      return op->emitOpError() << "all non-scalar operands/results must have "
                                  "the same shape and base type: found "
                               << type << " and " << mustMatchType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BinaryOp implementation
//===----------------------------------------------------------------------===//

// These functions are out-of-line implementations of the methods in BinaryOp,
// which avoids them being template instantiated/duplicated.

void impl::buildBinaryOp(OpBuilder &builder, OperationState &result, Value lhs,
                         Value rhs) {
  assert(lhs.getType() == rhs.getType());
  result.addOperands({lhs, rhs});
  result.types.push_back(lhs.getType());
}

ParseResult impl::parseOneResultSameOperandTypeOp(OpAsmParser &parser,
                                                  OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(ops, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

void impl::printOneResultOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumResults() == 1 && "op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type type) { return type != resultType; })) {
    p.printGenericOp(op);
    return;
  }

  p << op->getName() << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  // Now we can output only one type for all operands and the result.
  p << " : " << resultType;
}

//===----------------------------------------------------------------------===//
// CastOp implementation
//===----------------------------------------------------------------------===//

/// Attempt to fold the given cast operation.
LogicalResult
impl::foldCastInterfaceOp(Operation *op, ArrayRef<Attribute> attrOperands,
                          SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = op->getOperands();
  if (operands.empty())
    return failure();
  ResultRange results = op->getResults();

  // Check for the case where the input and output types match 1-1.
  if (operands.getTypes() == results.getTypes()) {
    foldResults.append(operands.begin(), operands.end());
    return success();
  }

  return failure();
}

/// Attempt to verify the given cast operation.
LogicalResult impl::verifyCastInterfaceOp(
    Operation *op, function_ref<bool(TypeRange, TypeRange)> areCastCompatible) {
  auto resultTypes = op->getResultTypes();
  if (llvm::empty(resultTypes))
    return op->emitOpError()
           << "expected at least one result for cast operation";

  auto operandTypes = op->getOperandTypes();
  if (!areCastCompatible(operandTypes, resultTypes)) {
    InFlightDiagnostic diag = op->emitOpError("operand type");
    if (llvm::empty(operandTypes))
      diag << "s []";
    else if (llvm::size(operandTypes) == 1)
      diag << " " << *operandTypes.begin();
    else
      diag << "s " << operandTypes;
    return diag << " and result type" << (resultTypes.size() == 1 ? " " : "s ")
                << resultTypes << " are cast incompatible";
  }

  return success();
}

void impl::buildCastOp(OpBuilder &builder, OperationState &result, Value source,
                       Type destType) {
  result.addOperands(source);
  result.addTypes(destType);
}

ParseResult impl::parseCastOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  Type srcType, dstType;
  return failure(parser.parseOperand(srcInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(srcType) ||
                 parser.resolveOperand(srcInfo, srcType, result.operands) ||
                 parser.parseKeywordType("to", dstType) ||
                 parser.addTypeToList(dstType, result.types));
}

void impl::printCastOp(Operation *op, OpAsmPrinter &p) {
  p << op->getName() << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0).getType() << " to "
    << op->getResult(0).getType();
}

Value impl::foldCastOp(Operation *op) {
  // Identity cast
  if (op->getOperand(0).getType() == op->getResult(0).getType())
    return op->getOperand(0);
  return nullptr;
}

LogicalResult
impl::verifyCastOp(Operation *op,
                   function_ref<bool(Type, Type)> areCastCompatible) {
  auto opType = op->getOperand(0).getType();
  auto resType = op->getResult(0).getType();
  if (!areCastCompatible(opType, resType))
    return op->emitError("operand type ")
           << opType << " and result type " << resType
           << " are cast incompatible";

  return success();
}

//===----------------------------------------------------------------------===//
// Misc. utils
//===----------------------------------------------------------------------===//

/// Insert an operation, generated by `buildTerminatorOp`, at the end of the
/// region's only block if it does not have a terminator already. If the region
/// is empty, insert a new block first. `buildTerminatorOp` should return the
/// terminator operation to insert.
void impl::ensureRegionTerminator(
    Region &region, OpBuilder &builder, Location loc,
    function_ref<Operation *(OpBuilder &, Location)> buildTerminatorOp) {
  OpBuilder::InsertionGuard guard(builder);
  if (region.empty())
    builder.createBlock(&region);

  Block &block = region.back();
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>())
    return;

  builder.setInsertionPointToEnd(&block);
  builder.insert(buildTerminatorOp(builder, loc));
}

/// Create a simple OpBuilder and forward to the OpBuilder version of this
/// function.
void impl::ensureRegionTerminator(
    Region &region, Builder &builder, Location loc,
    function_ref<Operation *(OpBuilder &, Location)> buildTerminatorOp) {
  OpBuilder opBuilder(builder.getContext());
  ensureRegionTerminator(region, opBuilder, loc, buildTerminatorOp);
}

//===----------------------------------------------------------------------===//
// UseIterator
//===----------------------------------------------------------------------===//

Operation::UseIterator::UseIterator(Operation *op, bool end)
    : op(op), res(end ? op->result_end() : op->result_begin()) {
  // Only initialize current use if there are results/can be uses.
  if (op->getNumResults())
    skipOverResultsWithNoUsers();
}

Operation::UseIterator &Operation::UseIterator::operator++() {
  // We increment over uses, if we reach the last use then move to next
  // result.
  if (use != (*res).use_end())
    ++use;
  if (use == (*res).use_end()) {
    ++res;
    skipOverResultsWithNoUsers();
  }
  return *this;
}

void Operation::UseIterator::skipOverResultsWithNoUsers() {
  while (res != op->result_end() && (*res).use_empty())
    ++res;

  // If we are at the last result, then set use to first use of
  // first result (sentinel value used for end).
  if (res == op->result_end())
    use = {};
  else
    use = (*res).use_begin();
}
