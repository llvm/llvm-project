//===- OperationSupport.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains out-of-line implementations of the support types that
// Operation and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/SHA1.h"
#include <numeric>

using namespace mlir;

//===----------------------------------------------------------------------===//
// NamedAttrList
//===----------------------------------------------------------------------===//

NamedAttrList::NamedAttrList(ArrayRef<NamedAttribute> attributes) {
  assign(attributes.begin(), attributes.end());
}

NamedAttrList::NamedAttrList(DictionaryAttr attributes)
    : NamedAttrList(attributes ? attributes.getValue()
                               : ArrayRef<NamedAttribute>()) {
  dictionarySorted.setPointerAndInt(attributes, true);
}

NamedAttrList::NamedAttrList(const_iterator inStart, const_iterator inEnd) {
  assign(inStart, inEnd);
}

ArrayRef<NamedAttribute> NamedAttrList::getAttrs() const { return attrs; }

Optional<NamedAttribute> NamedAttrList::findDuplicate() const {
  Optional<NamedAttribute> duplicate =
      DictionaryAttr::findDuplicate(attrs, isSorted());
  // DictionaryAttr::findDuplicate will sort the list, so reset the sorted
  // state.
  if (!isSorted())
    dictionarySorted.setPointerAndInt(nullptr, true);
  return duplicate;
}

DictionaryAttr NamedAttrList::getDictionary(MLIRContext *context) const {
  if (!isSorted()) {
    DictionaryAttr::sortInPlace(attrs);
    dictionarySorted.setPointerAndInt(nullptr, true);
  }
  if (!dictionarySorted.getPointer())
    dictionarySorted.setPointer(DictionaryAttr::getWithSorted(context, attrs));
  return dictionarySorted.getPointer().cast<DictionaryAttr>();
}

/// Add an attribute with the specified name.
void NamedAttrList::append(StringRef name, Attribute attr) {
  append(StringAttr::get(attr.getContext(), name), attr);
}

/// Replaces the attributes with new list of attributes.
void NamedAttrList::assign(const_iterator inStart, const_iterator inEnd) {
  DictionaryAttr::sort(ArrayRef<NamedAttribute>{inStart, inEnd}, attrs);
  dictionarySorted.setPointerAndInt(nullptr, true);
}

void NamedAttrList::push_back(NamedAttribute newAttribute) {
  if (isSorted())
    dictionarySorted.setInt(attrs.empty() || attrs.back() < newAttribute);
  dictionarySorted.setPointer(nullptr);
  attrs.push_back(newAttribute);
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttrList::get(StringRef name) const {
  auto it = findAttr(*this, name);
  return it.second ? it.first->getValue() : Attribute();
}
Attribute NamedAttrList::get(StringAttr name) const {
  auto it = findAttr(*this, name);
  return it.second ? it.first->getValue() : Attribute();
}

/// Return the specified named attribute if present, std::nullopt otherwise.
Optional<NamedAttribute> NamedAttrList::getNamed(StringRef name) const {
  auto it = findAttr(*this, name);
  return it.second ? *it.first : Optional<NamedAttribute>();
}
Optional<NamedAttribute> NamedAttrList::getNamed(StringAttr name) const {
  auto it = findAttr(*this, name);
  return it.second ? *it.first : Optional<NamedAttribute>();
}

/// If the an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
Attribute NamedAttrList::set(StringAttr name, Attribute value) {
  assert(value && "attributes may never be null");

  // Look for an existing attribute with the given name, and set its value
  // in-place. Return the previous value of the attribute, if there was one.
  auto it = findAttr(*this, name);
  if (it.second) {
    // Update the existing attribute by swapping out the old value for the new
    // value. Return the old value.
    Attribute oldValue = it.first->getValue();
    if (it.first->getValue() != value) {
      it.first->setValue(value);

      // If the attributes have changed, the dictionary is invalidated.
      dictionarySorted.setPointer(nullptr);
    }
    return oldValue;
  }
  // Perform a string lookup to insert the new attribute into its sorted
  // position.
  if (isSorted())
    it = findAttr(*this, name.strref());
  attrs.insert(it.first, {name, value});
  // Invalidate the dictionary. Return null as there was no previous value.
  dictionarySorted.setPointer(nullptr);
  return Attribute();
}

Attribute NamedAttrList::set(StringRef name, Attribute value) {
  assert(value && "attributes may never be null");
  return set(mlir::StringAttr::get(value.getContext(), name), value);
}

Attribute
NamedAttrList::eraseImpl(SmallVectorImpl<NamedAttribute>::iterator it) {
  // Erasing does not affect the sorted property.
  Attribute attr = it->getValue();
  attrs.erase(it);
  dictionarySorted.setPointer(nullptr);
  return attr;
}

Attribute NamedAttrList::erase(StringAttr name) {
  auto it = findAttr(*this, name);
  return it.second ? eraseImpl(it.first) : Attribute();
}

Attribute NamedAttrList::erase(StringRef name) {
  auto it = findAttr(*this, name);
  return it.second ? eraseImpl(it.first) : Attribute();
}

NamedAttrList &
NamedAttrList::operator=(const SmallVectorImpl<NamedAttribute> &rhs) {
  assign(rhs.begin(), rhs.end());
  return *this;
}

NamedAttrList::operator ArrayRef<NamedAttribute>() const { return attrs; }

//===----------------------------------------------------------------------===//
// OperationState
//===----------------------------------------------------------------------===//

OperationState::OperationState(Location location, StringRef name)
    : location(location), name(name, location->getContext()) {}

OperationState::OperationState(Location location, OperationName name)
    : location(location), name(name) {}

OperationState::OperationState(Location location, OperationName name,
                               ValueRange operands, TypeRange types,
                               ArrayRef<NamedAttribute> attributes,
                               BlockRange successors,
                               MutableArrayRef<std::unique_ptr<Region>> regions)
    : location(location), name(name),
      operands(operands.begin(), operands.end()),
      types(types.begin(), types.end()),
      attributes(attributes.begin(), attributes.end()),
      successors(successors.begin(), successors.end()) {
  for (std::unique_ptr<Region> &r : regions)
    this->regions.push_back(std::move(r));
}
OperationState::OperationState(Location location, StringRef name,
                               ValueRange operands, TypeRange types,
                               ArrayRef<NamedAttribute> attributes,
                               BlockRange successors,
                               MutableArrayRef<std::unique_ptr<Region>> regions)
    : OperationState(location, OperationName(name, location.getContext()),
                     operands, types, attributes, successors, regions) {}

void OperationState::addOperands(ValueRange newOperands) {
  operands.append(newOperands.begin(), newOperands.end());
}

void OperationState::addSuccessors(BlockRange newSuccessors) {
  successors.append(newSuccessors.begin(), newSuccessors.end());
}

Region *OperationState::addRegion() {
  regions.emplace_back(new Region);
  return regions.back().get();
}

void OperationState::addRegion(std::unique_ptr<Region> &&region) {
  regions.push_back(std::move(region));
}

void OperationState::addRegions(
    MutableArrayRef<std::unique_ptr<Region>> regions) {
  for (std::unique_ptr<Region> &region : regions)
    addRegion(std::move(region));
}

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

detail::OperandStorage::OperandStorage(Operation *owner,
                                       OpOperand *trailingOperands,
                                       ValueRange values)
    : isStorageDynamic(false), operandStorage(trailingOperands) {
  numOperands = capacity = values.size();
  for (unsigned i = 0; i < numOperands; ++i)
    new (&operandStorage[i]) OpOperand(owner, values[i]);
}

detail::OperandStorage::~OperandStorage() {
  for (auto &operand : getOperands())
    operand.~OpOperand();

  // If the storage is dynamic, deallocate it.
  if (isStorageDynamic)
    free(operandStorage);
}

/// Replace the operands contained in the storage with the ones provided in
/// 'values'.
void detail::OperandStorage::setOperands(Operation *owner, ValueRange values) {
  MutableArrayRef<OpOperand> storageOperands = resize(owner, values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i)
    storageOperands[i].set(values[i]);
}

/// Replace the operands beginning at 'start' and ending at 'start' + 'length'
/// with the ones provided in 'operands'. 'operands' may be smaller or larger
/// than the range pointed to by 'start'+'length'.
void detail::OperandStorage::setOperands(Operation *owner, unsigned start,
                                         unsigned length, ValueRange operands) {
  // If the new size is the same, we can update inplace.
  unsigned newSize = operands.size();
  if (newSize == length) {
    MutableArrayRef<OpOperand> storageOperands = getOperands();
    for (unsigned i = 0, e = length; i != e; ++i)
      storageOperands[start + i].set(operands[i]);
    return;
  }
  // If the new size is greater, remove the extra operands and set the rest
  // inplace.
  if (newSize < length) {
    eraseOperands(start + operands.size(), length - newSize);
    setOperands(owner, start, newSize, operands);
    return;
  }
  // Otherwise, the new size is greater so we need to grow the storage.
  auto storageOperands = resize(owner, size() + (newSize - length));

  // Shift operands to the right to make space for the new operands.
  unsigned rotateSize = storageOperands.size() - (start + length);
  auto rbegin = storageOperands.rbegin();
  std::rotate(rbegin, std::next(rbegin, newSize - length), rbegin + rotateSize);

  // Update the operands inplace.
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    storageOperands[start + i].set(operands[i]);
}

/// Erase an operand held by the storage.
void detail::OperandStorage::eraseOperands(unsigned start, unsigned length) {
  MutableArrayRef<OpOperand> operands = getOperands();
  assert((start + length) <= operands.size());
  numOperands -= length;

  // Shift all operands down if the operand to remove is not at the end.
  if (start != numOperands) {
    auto *indexIt = std::next(operands.begin(), start);
    std::rotate(indexIt, std::next(indexIt, length), operands.end());
  }
  for (unsigned i = 0; i != length; ++i)
    operands[numOperands + i].~OpOperand();
}

void detail::OperandStorage::eraseOperands(const BitVector &eraseIndices) {
  MutableArrayRef<OpOperand> operands = getOperands();
  assert(eraseIndices.size() == operands.size());

  // Check that at least one operand is erased.
  int firstErasedIndice = eraseIndices.find_first();
  if (firstErasedIndice == -1)
    return;

  // Shift all of the removed operands to the end, and destroy them.
  numOperands = firstErasedIndice;
  for (unsigned i = firstErasedIndice + 1, e = operands.size(); i < e; ++i)
    if (!eraseIndices.test(i))
      operands[numOperands++] = std::move(operands[i]);
  for (OpOperand &operand : operands.drop_front(numOperands))
    operand.~OpOperand();
}

/// Resize the storage to the given size. Returns the array containing the new
/// operands.
MutableArrayRef<OpOperand> detail::OperandStorage::resize(Operation *owner,
                                                          unsigned newSize) {
  // If the number of operands is less than or equal to the current amount, we
  // can just update in place.
  MutableArrayRef<OpOperand> origOperands = getOperands();
  if (newSize <= numOperands) {
    // If the number of new size is less than the current, remove any extra
    // operands.
    for (unsigned i = newSize; i != numOperands; ++i)
      origOperands[i].~OpOperand();
    numOperands = newSize;
    return origOperands.take_front(newSize);
  }

  // If the new size is within the original inline capacity, grow inplace.
  if (newSize <= capacity) {
    OpOperand *opBegin = origOperands.data();
    for (unsigned e = newSize; numOperands != e; ++numOperands)
      new (&opBegin[numOperands]) OpOperand(owner);
    return MutableArrayRef<OpOperand>(opBegin, newSize);
  }

  // Otherwise, we need to allocate a new storage.
  unsigned newCapacity =
      std::max(unsigned(llvm::NextPowerOf2(capacity + 2)), newSize);
  OpOperand *newOperandStorage =
      reinterpret_cast<OpOperand *>(malloc(sizeof(OpOperand) * newCapacity));

  // Move the current operands to the new storage.
  MutableArrayRef<OpOperand> newOperands(newOperandStorage, newSize);
  std::uninitialized_move(origOperands.begin(), origOperands.end(),
                          newOperands.begin());

  // Destroy the original operands.
  for (auto &operand : origOperands)
    operand.~OpOperand();

  // Initialize any new operands.
  for (unsigned e = newSize; numOperands != e; ++numOperands)
    new (&newOperands[numOperands]) OpOperand(owner);

  // If the current storage is dynamic, free it.
  if (isStorageDynamic)
    free(operandStorage);

  // Update the storage representation to use the new dynamic storage.
  operandStorage = newOperandStorage;
  capacity = newCapacity;
  isStorageDynamic = true;
  return newOperands;
}

//===----------------------------------------------------------------------===//
// Operation Value-Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperandRange

unsigned OperandRange::getBeginOperandIndex() const {
  assert(!empty() && "range must not be empty");
  return base->getOperandNumber();
}

OperandRangeRange OperandRange::split(DenseI32ArrayAttr segmentSizes) const {
  return OperandRangeRange(*this, segmentSizes);
}

//===----------------------------------------------------------------------===//
// OperandRangeRange

OperandRangeRange::OperandRangeRange(OperandRange operands,
                                     Attribute operandSegments)
    : OperandRangeRange(OwnerT(operands.getBase(), operandSegments), 0,
                        operandSegments.cast<DenseI32ArrayAttr>().size()) {}

OperandRange OperandRangeRange::join() const {
  const OwnerT &owner = getBase();
  ArrayRef<int32_t> sizeData = owner.second.cast<DenseI32ArrayAttr>();
  return OperandRange(owner.first,
                      std::accumulate(sizeData.begin(), sizeData.end(), 0));
}

OperandRange OperandRangeRange::dereference(const OwnerT &object,
                                            ptrdiff_t index) {
  ArrayRef<int32_t> sizeData = object.second.cast<DenseI32ArrayAttr>();
  uint32_t startIndex =
      std::accumulate(sizeData.begin(), sizeData.begin() + index, 0);
  return OperandRange(object.first + startIndex, *(sizeData.begin() + index));
}

//===----------------------------------------------------------------------===//
// MutableOperandRange

/// Construct a new mutable range from the given operand, operand start index,
/// and range length.
MutableOperandRange::MutableOperandRange(
    Operation *owner, unsigned start, unsigned length,
    ArrayRef<OperandSegment> operandSegments)
    : owner(owner), start(start), length(length),
      operandSegments(operandSegments.begin(), operandSegments.end()) {
  assert((start + length) <= owner->getNumOperands() && "invalid range");
}
MutableOperandRange::MutableOperandRange(Operation *owner)
    : MutableOperandRange(owner, /*start=*/0, owner->getNumOperands()) {}

/// Slice this range into a sub range, with the additional operand segment.
MutableOperandRange
MutableOperandRange::slice(unsigned subStart, unsigned subLen,
                           Optional<OperandSegment> segment) const {
  assert((subStart + subLen) <= length && "invalid sub-range");
  MutableOperandRange subSlice(owner, start + subStart, subLen,
                               operandSegments);
  if (segment)
    subSlice.operandSegments.push_back(*segment);
  return subSlice;
}

/// Append the given values to the range.
void MutableOperandRange::append(ValueRange values) {
  if (values.empty())
    return;
  owner->insertOperands(start + length, values);
  updateLength(length + values.size());
}

/// Assign this range to the given values.
void MutableOperandRange::assign(ValueRange values) {
  owner->setOperands(start, length, values);
  if (length != values.size())
    updateLength(/*newLength=*/values.size());
}

/// Assign the range to the given value.
void MutableOperandRange::assign(Value value) {
  if (length == 1) {
    owner->setOperand(start, value);
  } else {
    owner->setOperands(start, length, value);
    updateLength(/*newLength=*/1);
  }
}

/// Erase the operands within the given sub-range.
void MutableOperandRange::erase(unsigned subStart, unsigned subLen) {
  assert((subStart + subLen) <= length && "invalid sub-range");
  if (length == 0)
    return;
  owner->eraseOperands(start + subStart, subLen);
  updateLength(length - subLen);
}

/// Clear this range and erase all of the operands.
void MutableOperandRange::clear() {
  if (length != 0) {
    owner->eraseOperands(start, length);
    updateLength(/*newLength=*/0);
  }
}

/// Allow implicit conversion to an OperandRange.
MutableOperandRange::operator OperandRange() const {
  return owner->getOperands().slice(start, length);
}

MutableOperandRangeRange
MutableOperandRange::split(NamedAttribute segmentSizes) const {
  return MutableOperandRangeRange(*this, segmentSizes);
}

/// Update the length of this range to the one provided.
void MutableOperandRange::updateLength(unsigned newLength) {
  int32_t diff = int32_t(newLength) - int32_t(length);
  length = newLength;

  // Update any of the provided segment attributes.
  for (OperandSegment &segment : operandSegments) {
    auto attr = segment.second.getValue().cast<DenseI32ArrayAttr>();
    SmallVector<int32_t, 8> segments(attr.asArrayRef());
    segments[segment.first] += diff;
    segment.second.setValue(
        DenseI32ArrayAttr::get(attr.getContext(), segments));
    owner->setAttr(segment.second.getName(), segment.second.getValue());
  }
}

//===----------------------------------------------------------------------===//
// MutableOperandRangeRange

MutableOperandRangeRange::MutableOperandRangeRange(
    const MutableOperandRange &operands, NamedAttribute operandSegmentAttr)
    : MutableOperandRangeRange(
          OwnerT(operands, operandSegmentAttr), 0,
          operandSegmentAttr.getValue().cast<DenseI32ArrayAttr>().size()) {}

MutableOperandRange MutableOperandRangeRange::join() const {
  return getBase().first;
}

MutableOperandRangeRange::operator OperandRangeRange() const {
  return OperandRangeRange(getBase().first, getBase().second.getValue());
}

MutableOperandRange MutableOperandRangeRange::dereference(const OwnerT &object,
                                                          ptrdiff_t index) {
  ArrayRef<int32_t> sizeData =
      object.second.getValue().cast<DenseI32ArrayAttr>();
  uint32_t startIndex =
      std::accumulate(sizeData.begin(), sizeData.begin() + index, 0);
  return object.first.slice(
      startIndex, *(sizeData.begin() + index),
      MutableOperandRange::OperandSegment(index, object.second));
}

//===----------------------------------------------------------------------===//
// ResultRange

ResultRange::ResultRange(OpResult result)
    : ResultRange(static_cast<detail::OpResultImpl *>(Value(result).getImpl()),
                  1) {}

ResultRange::use_range ResultRange::getUses() const {
  return {use_begin(), use_end()};
}
ResultRange::use_iterator ResultRange::use_begin() const {
  return use_iterator(*this);
}
ResultRange::use_iterator ResultRange::use_end() const {
  return use_iterator(*this, /*end=*/true);
}
ResultRange::user_range ResultRange::getUsers() {
  return {user_begin(), user_end()};
}
ResultRange::user_iterator ResultRange::user_begin() {
  return user_iterator(use_begin());
}
ResultRange::user_iterator ResultRange::user_end() {
  return user_iterator(use_end());
}

ResultRange::UseIterator::UseIterator(ResultRange results, bool end)
    : it(end ? results.end() : results.begin()), endIt(results.end()) {
  // Only initialize current use if there are results/can be uses.
  if (it != endIt)
    skipOverResultsWithNoUsers();
}

ResultRange::UseIterator &ResultRange::UseIterator::operator++() {
  // We increment over uses, if we reach the last use then move to next
  // result.
  if (use != (*it).use_end())
    ++use;
  if (use == (*it).use_end()) {
    ++it;
    skipOverResultsWithNoUsers();
  }
  return *this;
}

void ResultRange::UseIterator::skipOverResultsWithNoUsers() {
  while (it != endIt && (*it).use_empty())
    ++it;

  // If we are at the last result, then set use to first use of
  // first result (sentinel value used for end).
  if (it == endIt)
    use = {};
  else
    use = (*it).use_begin();
}

void ResultRange::replaceAllUsesWith(Operation *op) {
  replaceAllUsesWith(op->getResults());
}

//===----------------------------------------------------------------------===//
// ValueRange

ValueRange::ValueRange(ArrayRef<Value> values)
    : ValueRange(values.data(), values.size()) {}
ValueRange::ValueRange(OperandRange values)
    : ValueRange(values.begin().getBase(), values.size()) {}
ValueRange::ValueRange(ResultRange values)
    : ValueRange(values.getBase(), values.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
ValueRange::OwnerT ValueRange::offset_base(const OwnerT &owner,
                                           ptrdiff_t index) {
  if (const auto *value = owner.dyn_cast<const Value *>())
    return {value + index};
  if (auto *operand = owner.dyn_cast<OpOperand *>())
    return {operand + index};
  return owner.get<detail::OpResultImpl *>()->getNextResultAtOffset(index);
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Value ValueRange::dereference_iterator(const OwnerT &owner, ptrdiff_t index) {
  if (const auto *value = owner.dyn_cast<const Value *>())
    return value[index];
  if (auto *operand = owner.dyn_cast<OpOperand *>())
    return operand[index].get();
  return owner.get<detail::OpResultImpl *>()->getNextResultAtOffset(index);
}

//===----------------------------------------------------------------------===//
// Operation Equivalency
//===----------------------------------------------------------------------===//

llvm::hash_code OperationEquivalence::computeHash(
    Operation *op, function_ref<llvm::hash_code(Value)> hashOperands,
    function_ref<llvm::hash_code(Value)> hashResults, Flags flags) {
  // Hash operations based upon their:
  //   - Operation Name
  //   - Attributes
  //   - Result Types
  llvm::hash_code hash = llvm::hash_combine(
      op->getName(), op->getAttrDictionary(), op->getResultTypes());

  //   - Operands
  ValueRange operands = op->getOperands();
  SmallVector<Value> operandStorage;
  if (op->hasTrait<mlir::OpTrait::IsCommutative>()) {
    operandStorage.append(operands.begin(), operands.end());
    llvm::sort(operandStorage, [](Value a, Value b) -> bool {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });
    operands = operandStorage;
  }
  for (Value operand : operands)
    hash = llvm::hash_combine(hash, hashOperands(operand));

  //   - Operands
  for (Value result : op->getResults())
    hash = llvm::hash_combine(hash, hashResults(result));
  return hash;
}

static bool
isRegionEquivalentTo(Region *lhs, Region *rhs,
                     function_ref<LogicalResult(Value, Value)> mapOperands,
                     function_ref<LogicalResult(Value, Value)> mapResults,
                     OperationEquivalence::Flags flags) {
  DenseMap<Block *, Block *> blocksMap;
  auto blocksEquivalent = [&](Block &lBlock, Block &rBlock) {
    // Check block arguments.
    if (lBlock.getNumArguments() != rBlock.getNumArguments())
      return false;

    // Map the two blocks.
    auto insertion = blocksMap.insert({&lBlock, &rBlock});
    if (insertion.first->getSecond() != &rBlock)
      return false;

    for (auto argPair :
         llvm::zip(lBlock.getArguments(), rBlock.getArguments())) {
      Value curArg = std::get<0>(argPair);
      Value otherArg = std::get<1>(argPair);
      if (curArg.getType() != otherArg.getType())
        return false;
      if (!(flags & OperationEquivalence::IgnoreLocations) &&
          curArg.getLoc() != otherArg.getLoc())
        return false;
      // Check if this value was already mapped to another value.
      if (failed(mapOperands(curArg, otherArg)))
        return false;
    }

    auto opsEquivalent = [&](Operation &lOp, Operation &rOp) {
      // Check for op equality (recursively).
      if (!OperationEquivalence::isEquivalentTo(&lOp, &rOp, mapOperands,
                                                mapResults, flags))
        return false;
      // Check successor mapping.
      for (auto successorsPair :
           llvm::zip(lOp.getSuccessors(), rOp.getSuccessors())) {
        Block *curSuccessor = std::get<0>(successorsPair);
        Block *otherSuccessor = std::get<1>(successorsPair);
        auto insertion = blocksMap.insert({curSuccessor, otherSuccessor});
        if (insertion.first->getSecond() != otherSuccessor)
          return false;
      }
      return true;
    };
    return llvm::all_of_zip(lBlock, rBlock, opsEquivalent);
  };
  return llvm::all_of_zip(*lhs, *rhs, blocksEquivalent);
}

bool OperationEquivalence::isEquivalentTo(
    Operation *lhs, Operation *rhs,
    function_ref<LogicalResult(Value, Value)> mapOperands,
    function_ref<LogicalResult(Value, Value)> mapResults, Flags flags) {
  if (lhs == rhs)
    return true;

  // Compare the operation properties.
  if (lhs->getName() != rhs->getName() ||
      lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
      lhs->getNumRegions() != rhs->getNumRegions() ||
      lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults())
    return false;
  if (!(flags & IgnoreLocations) && lhs->getLoc() != rhs->getLoc())
    return false;

  ValueRange lhsOperands = lhs->getOperands(), rhsOperands = rhs->getOperands();
  SmallVector<Value> lhsOperandStorage, rhsOperandStorage;
  if (lhs->hasTrait<mlir::OpTrait::IsCommutative>()) {
    auto sortValues = [](ValueRange values) {
      SmallVector<Value> sortedValues = llvm::to_vector(values);
      llvm::sort(sortedValues, [](Value a, Value b) {
        auto aArg = a.dyn_cast<BlockArgument>();
        auto bArg = b.dyn_cast<BlockArgument>();

        // Case 1. Both `a` and `b` are `BlockArgument`s.
        if (aArg && bArg) {
          if (aArg.getParentBlock() == bArg.getParentBlock())
            return aArg.getArgNumber() < bArg.getArgNumber();
          return aArg.getParentBlock() < bArg.getParentBlock();
        }

        // Case 2. One of then is a `BlockArgument` and other is not. Treat
        // `BlockArgument` as lesser.
        if (aArg && !bArg)
          return true;
        if (bArg && !aArg)
          return false;

        // Case 3. Both are values.
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      return sortedValues;
    };
    lhsOperandStorage = sortValues(lhsOperands);
    lhsOperands = lhsOperandStorage;
    rhsOperandStorage = sortValues(rhsOperands);
    rhsOperands = rhsOperandStorage;
  }
  auto checkValueRangeMapping =
      [](ValueRange lhs, ValueRange rhs,
         function_ref<LogicalResult(Value, Value)> mapValues) {
        for (auto operandPair : llvm::zip(lhs, rhs)) {
          Value curArg = std::get<0>(operandPair);
          Value otherArg = std::get<1>(operandPair);
          if (curArg.getType() != otherArg.getType())
            return false;
          if (failed(mapValues(curArg, otherArg)))
            return false;
        }
        return true;
      };
  // Check mapping of operands and results.
  if (!checkValueRangeMapping(lhsOperands, rhsOperands, mapOperands))
    return false;
  if (!checkValueRangeMapping(lhs->getResults(), rhs->getResults(), mapResults))
    return false;
  for (auto regionPair : llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!isRegionEquivalentTo(&std::get<0>(regionPair),
                              &std::get<1>(regionPair), mapOperands, mapResults,
                              flags))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// OperationFingerPrint
//===----------------------------------------------------------------------===//

template <typename T>
static void addDataToHash(llvm::SHA1 &hasher, const T &data) {
  hasher.update(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&data), sizeof(T)));
}

OperationFingerPrint::OperationFingerPrint(Operation *topOp) {
  llvm::SHA1 hasher;

  // Hash each of the operations based upon their mutable bits:
  topOp->walk([&](Operation *op) {
    //   - Operation pointer
    addDataToHash(hasher, op);
    //   - Attributes
    addDataToHash(hasher, op->getAttrDictionary());
    //   - Blocks in Regions
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        addDataToHash(hasher, &block);
        for (BlockArgument arg : block.getArguments())
          addDataToHash(hasher, arg);
      }
    }
    //   - Location
    addDataToHash(hasher, op->getLoc().getAsOpaquePointer());
    //   - Operands
    for (Value operand : op->getOperands())
      addDataToHash(hasher, operand);
    //   - Successors
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i)
      addDataToHash(hasher, op->getSuccessor(i));
  });
  hash = hasher.result();
}
