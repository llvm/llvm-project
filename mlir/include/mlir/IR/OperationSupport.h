//===- OperationSupport.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of support types that Operation and related
// classes build on top of.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATION_SUPPORT_H
#define MLIR_IR_OPERATION_SUPPORT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/InterfaceSupport.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TrailingObjects.h"
#include <memory>

namespace mlir {
class Dialect;
class Operation;
struct OperationState;
class OpAsmParser;
class OpAsmParserResult;
class OpAsmPrinter;
class OperandRange;
class OpFoldResult;
class ParseResult;
class Pattern;
class Region;
class ResultRange;
class RewritePattern;
class Type;
class Value;
class ValueRange;
template <typename ValueRangeT> class ValueTypeRange;

class OwningRewritePatternList;

//===----------------------------------------------------------------------===//
// AbstractOperation
//===----------------------------------------------------------------------===//

enum class OperationProperty {
  /// This bit is set for an operation if it is a commutative
  /// operation: that is an operator where order of operands does not
  /// change the result of the operation.  For example, in a binary
  /// commutative operation, "a op b" and "b op a" produce the same
  /// results.
  Commutative = 0x1,

  /// This bit is set for an operation if it is a terminator: that means
  /// an operation at the end of a block.
  Terminator = 0x2,

  /// This bit is set for operations that are completely isolated from above.
  /// This is used for operations whose regions are explicit capture only, i.e.
  /// they are never allowed to implicitly reference values defined above the
  /// parent operation.
  IsolatedFromAbove = 0x4,
};

/// This is a "type erased" representation of a registered operation.  This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks.  Most user code should use
/// the concrete operation types.
class AbstractOperation {
public:
  using OperationProperties = uint32_t;

  /// This is the name of the operation.
  const Identifier name;

  /// This is the dialect that this operation belongs to.
  Dialect &dialect;

  /// The unique identifier of the derived Op class.
  TypeID typeID;

  /// Use the specified object to parse this ops custom assembly format.
  ParseResult (&parseAssembly)(OpAsmParser &parser, OperationState &result);

  /// This hook implements the AsmPrinter for this operation.
  void (&printAssembly)(Operation *op, OpAsmPrinter &p);

  /// This hook implements the verifier for this operation.  It should emits an
  /// error message and returns failure if a problem is detected, or returns
  /// success if everything is ok.
  LogicalResult (&verifyInvariants)(Operation *op);

  /// This hook implements a generalized folder for this operation.  Operations
  /// can implement this to provide simplifications rules that are applied by
  /// the Builder::createOrFold API and the canonicalization pass.
  ///
  /// This is an intentionally limited interface - implementations of this hook
  /// can only perform the following changes to the operation:
  ///
  ///  1. They can leave the operation alone and without changing the IR, and
  ///     return failure.
  ///  2. They can mutate the operation in place, without changing anything else
  ///     in the IR.  In this case, return success.
  ///  3. They can return a list of existing values that can be used instead of
  ///     the operation.  In this case, fill in the results list and return
  ///     success.  The caller will remove the operation and use those results
  ///     instead.
  ///
  /// This allows expression of some simple in-place canonicalizations (e.g.
  /// "x+0 -> x", "min(x,y,x,z) -> min(x,y,z)", "x+y-x -> y", etc), as well as
  /// generalized constant folding.
  LogicalResult (&foldHook)(Operation *op, ArrayRef<Attribute> operands,
                            SmallVectorImpl<OpFoldResult> &results);

  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  void (&getCanonicalizationPatterns)(OwningRewritePatternList &results,
                                      MLIRContext *context);

  /// Returns whether the operation has a particular property.
  bool hasProperty(OperationProperty property) const {
    return opProperties & static_cast<OperationProperties>(property);
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this operation, null otherwise. This should not be used
  /// directly.
  template <typename T> typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  /// Returns true if the operation has a particular trait.
  template <template <typename T> class Trait> bool hasTrait() const {
    return hasRawTrait(TypeID::get<Trait>());
  }

  /// Look up the specified operation in the specified MLIRContext and return a
  /// pointer to it if present.  Otherwise, return a null pointer.
  static const AbstractOperation *lookup(StringRef opName,
                                         MLIRContext *context);

  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T> static AbstractOperation get(Dialect &dialect) {
    return AbstractOperation(
        T::getOperationName(), dialect, T::getOperationProperties(),
        TypeID::get<T>(), T::parseAssembly, T::printAssembly,
        T::verifyInvariants, T::foldHook, T::getCanonicalizationPatterns,
        T::getInterfaceMap(), T::hasTrait);
  }

private:
  AbstractOperation(
      StringRef name, Dialect &dialect, OperationProperties opProperties,
      TypeID typeID,
      ParseResult (&parseAssembly)(OpAsmParser &parser, OperationState &result),
      void (&printAssembly)(Operation *op, OpAsmPrinter &p),
      LogicalResult (&verifyInvariants)(Operation *op),
      LogicalResult (&foldHook)(Operation *op, ArrayRef<Attribute> operands,
                                SmallVectorImpl<OpFoldResult> &results),
      void (&getCanonicalizationPatterns)(OwningRewritePatternList &results,
                                          MLIRContext *context),
      detail::InterfaceMap &&interfaceMap, bool (&hasTrait)(TypeID traitID));

  /// The properties of the operation.
  const OperationProperties opProperties;

  /// A map of interfaces that were registered to this operation.
  detail::InterfaceMap interfaceMap;

  /// This hook returns if the operation contains the trait corresponding
  /// to the given TypeID.
  bool (&hasRawTrait)(TypeID traitID);
};

//===----------------------------------------------------------------------===//
// NamedAttrList
//===----------------------------------------------------------------------===//

/// NamedAttrList is array of NamedAttributes that tracks whether it is sorted
/// and does some basic work to remain sorted.
class NamedAttrList {
public:
  using const_iterator = SmallVectorImpl<NamedAttribute>::const_iterator;
  using const_reference = const NamedAttribute &;
  using reference = NamedAttribute &;
  using size_type = size_t;

  NamedAttrList() : dictionarySorted({}, true) {}
  NamedAttrList(ArrayRef<NamedAttribute> attributes);
  NamedAttrList(const_iterator in_start, const_iterator in_end);

  bool operator!=(const NamedAttrList &other) const {
    return !(*this == other);
  }
  bool operator==(const NamedAttrList &other) const {
    return attrs == other.attrs;
  }

  /// Add an attribute with the specified name.
  void append(StringRef name, Attribute attr);

  /// Add an attribute with the specified name.
  void append(Identifier name, Attribute attr);

  /// Add an array of named attributes.
  void append(ArrayRef<NamedAttribute> newAttributes);

  /// Add a range of named attributes.
  void append(const_iterator in_start, const_iterator in_end);

  /// Replaces the attributes with new list of attributes.
  void assign(const_iterator in_start, const_iterator in_end);

  /// Replaces the attributes with new list of attributes.
  void assign(ArrayRef<NamedAttribute> range) {
    append(range.begin(), range.end());
  }

  bool empty() const { return attrs.empty(); }

  void reserve(size_type N) { attrs.reserve(N); }

  /// Add an attribute with the specified name.
  void push_back(NamedAttribute newAttribute);

  /// Pop last element from list.
  void pop_back() { attrs.pop_back(); }

  /// Return a dictionary attribute for the underlying dictionary. This will
  /// return an empty dictionary attribute if empty rather than null.
  DictionaryAttr getDictionary(MLIRContext *context) const;

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() const;

  /// Return the specified attribute if present, null otherwise.
  Attribute get(Identifier name) const;
  Attribute get(StringRef name) const;

  /// Return the specified named attribute if present, None otherwise.
  Optional<NamedAttribute> getNamed(StringRef name) const;
  Optional<NamedAttribute> getNamed(Identifier name) const;

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void set(Identifier name, Attribute value);
  void set(StringRef name, Attribute value);

  const_iterator begin() const { return attrs.begin(); }
  const_iterator end() const { return attrs.end(); }

  NamedAttrList &operator=(const SmallVectorImpl<NamedAttribute> &rhs);
  operator ArrayRef<NamedAttribute>() const;
  operator MutableDictionaryAttr() const;

private:
  /// Return whether the attributes are sorted.
  bool isSorted() const { return dictionarySorted.getInt(); }

  // These are marked mutable as they may be modified (e.g., sorted)
  mutable SmallVector<NamedAttribute, 4> attrs;
  // Pair with cached DictionaryAttr and status of whether attrs is sorted.
  // Note: just because sorted does not mean a DictionaryAttr has been created
  // but the case where there is a DictionaryAttr but attrs isn't sorted should
  // not occur.
  mutable llvm::PointerIntPair<Attribute, 1, bool> dictionarySorted;
};

//===----------------------------------------------------------------------===//
// OperationName
//===----------------------------------------------------------------------===//

class OperationName {
public:
  using RepresentationUnion =
      PointerUnion<Identifier, const AbstractOperation *>;

  OperationName(AbstractOperation *op) : representation(op) {}
  OperationName(StringRef name, MLIRContext *context);

  /// Return the name of the dialect this operation is registered to.
  StringRef getDialect() const;

  /// Return the operation name with dialect name stripped, if it has one.
  StringRef stripDialect() const;

  /// Return the name of this operation. This always succeeds.
  StringRef getStringRef() const;

  /// Return the name of this operation as an identifier. This always succeeds.
  Identifier getIdentifier() const;

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const;

  void print(raw_ostream &os) const;
  void dump() const;

  void *getAsOpaquePointer() const {
    return static_cast<void *>(representation.getOpaqueValue());
  }
  static OperationName getFromOpaquePointer(void *pointer);

private:
  RepresentationUnion representation;
  OperationName(RepresentationUnion representation)
      : representation(representation) {}
};

inline raw_ostream &operator<<(raw_ostream &os, OperationName identifier) {
  identifier.print(os);
  return os;
}

inline bool operator==(OperationName lhs, OperationName rhs) {
  return lhs.getAsOpaquePointer() == rhs.getAsOpaquePointer();
}

inline bool operator!=(OperationName lhs, OperationName rhs) {
  return lhs.getAsOpaquePointer() != rhs.getAsOpaquePointer();
}

// Make operation names hashable.
inline llvm::hash_code hash_value(OperationName arg) {
  return llvm::hash_value(arg.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// OperationState
//===----------------------------------------------------------------------===//

/// This represents an operation in an abstracted form, suitable for use with
/// the builder APIs.  This object is a large and heavy weight object meant to
/// be used as a temporary object on the stack.  It is generally unwise to put
/// this in a collection.
struct OperationState {
  Location location;
  OperationName name;
  SmallVector<Value, 4> operands;
  /// Types of the results of this operation.
  SmallVector<Type, 4> types;
  NamedAttrList attributes;
  /// Successors of this operation and their respective operands.
  SmallVector<Block *, 1> successors;
  /// Regions that the op will hold.
  SmallVector<std::unique_ptr<Region>, 1> regions;

public:
  OperationState(Location location, StringRef name);

  OperationState(Location location, OperationName name);

  OperationState(Location location, StringRef name, ValueRange operands,
                 TypeRange types, ArrayRef<NamedAttribute> attributes,
                 BlockRange successors = {},
                 MutableArrayRef<std::unique_ptr<Region>> regions = {});

  void addOperands(ValueRange newOperands);

  void addTypes(ArrayRef<Type> newTypes) {
    types.append(newTypes.begin(), newTypes.end());
  }
  template <typename RangeT>
  std::enable_if_t<!std::is_convertible<RangeT, ArrayRef<Type>>::value>
  addTypes(RangeT &&newTypes) {
    types.append(newTypes.begin(), newTypes.end());
  }

  /// Add an attribute with the specified name.
  void addAttribute(StringRef name, Attribute attr) {
    addAttribute(Identifier::get(name, getContext()), attr);
  }

  /// Add an attribute with the specified name.
  void addAttribute(Identifier name, Attribute attr) {
    attributes.append(name, attr);
  }

  /// Add an array of named attributes.
  void addAttributes(ArrayRef<NamedAttribute> newAttributes) {
    attributes.append(newAttributes);
  }

  void addSuccessors(Block *successor) { successors.push_back(successor); }
  void addSuccessors(BlockRange newSuccessors);

  /// Create a region that should be attached to the operation.  These regions
  /// can be filled in immediately without waiting for Operation to be
  /// created.  When it is, the region bodies will be transferred.
  Region *addRegion();

  /// Take a region that should be attached to the Operation.  The body of the
  /// region will be transferred when the Operation is constructed.  If the
  /// region is null, a new empty region will be attached to the Operation.
  void addRegion(std::unique_ptr<Region> &&region);

  /// Take ownership of a set of regions that should be attached to the
  /// Operation.
  void addRegions(MutableArrayRef<std::unique_ptr<Region>> regions);

  /// Get the context held by this operation state.
  MLIRContext *getContext() const { return location->getContext(); }
};

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

namespace detail {
/// This class contains the information for a trailing operand storage.
struct TrailingOperandStorage final
    : public llvm::TrailingObjects<TrailingOperandStorage, OpOperand> {
  ~TrailingOperandStorage() {
    for (auto &operand : getOperands())
      operand.~OpOperand();
  }

  /// Return the operands held by this storage.
  MutableArrayRef<OpOperand> getOperands() {
    return {getTrailingObjects<OpOperand>(), numOperands};
  }

  /// The number of operands within the storage.
  unsigned numOperands;
  /// The total capacity number of operands that the storage can hold.
  unsigned capacity : 31;
  /// We reserve a range of bits for use by the operand storage.
  unsigned reserved : 1;
};

/// This class handles the management of operation operands. Operands are
/// stored either in a trailing array, or a dynamically resizable vector.
class OperandStorage final
    : private llvm::TrailingObjects<OperandStorage, OpOperand> {
public:
  OperandStorage(Operation *owner, ValueRange values);
  ~OperandStorage();

  /// Replace the operands contained in the storage with the ones provided in
  /// 'values'.
  void setOperands(Operation *owner, ValueRange values);

  /// Replace the operands beginning at 'start' and ending at 'start' + 'length'
  /// with the ones provided in 'operands'. 'operands' may be smaller or larger
  /// than the range pointed to by 'start'+'length'.
  void setOperands(Operation *owner, unsigned start, unsigned length,
                   ValueRange operands);

  /// Erase the operands held by the storage within the given range.
  void eraseOperands(unsigned start, unsigned length);

  /// Get the operation operands held by the storage.
  MutableArrayRef<OpOperand> getOperands() {
    return getStorage().getOperands();
  }

  /// Return the number of operands held in the storage.
  unsigned size() { return getStorage().numOperands; }

  /// Returns the additional size necessary for allocating this object.
  static size_t additionalAllocSize(unsigned numOperands) {
    return additionalSizeToAlloc<OpOperand>(numOperands);
  }

private:
  enum : uint64_t {
    /// The bit used to mark the storage as dynamic.
    DynamicStorageBit = 1ull << 63ull
  };

  /// Resize the storage to the given size. Returns the array containing the new
  /// operands.
  MutableArrayRef<OpOperand> resize(Operation *owner, unsigned newSize);

  /// Returns the current internal storage instance.
  TrailingOperandStorage &getStorage() {
    return LLVM_UNLIKELY(isDynamicStorage()) ? getDynamicStorage()
                                             : getInlineStorage();
  }

  /// Returns the storage container if the storage is inline.
  TrailingOperandStorage &getInlineStorage() {
    assert(!isDynamicStorage() && "expected storage to be inline");
    static_assert(sizeof(TrailingOperandStorage) == sizeof(uint64_t),
                  "inline storage representation must match the opaque "
                  "representation");
    return inlineStorage;
  }

  /// Returns the storage container if this storage is dynamic.
  TrailingOperandStorage &getDynamicStorage() {
    assert(isDynamicStorage() && "expected dynamic storage");
    uint64_t maskedRepresentation = representation & ~DynamicStorageBit;
    return *reinterpret_cast<TrailingOperandStorage *>(maskedRepresentation);
  }

  /// Returns true if the storage is currently dynamic.
  bool isDynamicStorage() const { return representation & DynamicStorageBit; }

  /// The current representation of the storage. This is either a
  /// InlineOperandStorage, or a pointer to a InlineOperandStorage.
  union {
    TrailingOperandStorage inlineStorage;
    uint64_t representation;
  };

  /// This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperandStorage, OpOperand>;
};
} // end namespace detail

//===----------------------------------------------------------------------===//
// ResultStorage
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides the implementation for an in-line operation result. This
/// is an operation result whose number can be stored inline inside of the bits
/// of an Operation*.
struct InLineOpResult : public IRObjectWithUseList<OpOperand> {};
/// This class provides the implementation for an out-of-line operation result.
/// This is an operation result whose number cannot be stored inline inside of
/// the bits of an Operation*.
struct TrailingOpResult : public IRObjectWithUseList<OpOperand> {
  TrailingOpResult(uint64_t trailingResultNumber)
      : trailingResultNumber(trailingResultNumber) {}

  /// Returns the parent operation of this trailing result.
  Operation *getOwner();

  /// Return the proper result number of this op result.
  unsigned getResultNumber() {
    return trailingResultNumber + OpResult::getMaxInlineResults();
  }

  /// The trailing result number, or the offset from the beginning of the
  /// trailing array.
  uint64_t trailingResultNumber;
};
} // end namespace detail

//===----------------------------------------------------------------------===//
// OpPrintingFlags
//===----------------------------------------------------------------------===//

/// Set of flags used to control the behavior of the various IR print methods
/// (e.g. Operation::Print).
class OpPrintingFlags {
public:
  OpPrintingFlags();
  OpPrintingFlags(llvm::NoneType) : OpPrintingFlags() {}

  /// Enable the elision of large elements attributes, by printing a '...'
  /// instead of the element data. Note: The IR generated with this option is
  /// not parsable. `largeElementLimit` is used to configure what is considered
  /// to be a "large" ElementsAttr by providing an upper limit to the number of
  /// elements.
  OpPrintingFlags &elideLargeElementsAttrs(int64_t largeElementLimit = 16);

  /// Enable printing of debug information. If 'prettyForm' is set to true,
  /// debug information is printed in a more readable 'pretty' form. Note: The
  /// IR generated with 'prettyForm' is not parsable.
  OpPrintingFlags &enableDebugInfo(bool prettyForm = false);

  /// Always print operations in the generic form.
  OpPrintingFlags &printGenericOpForm();

  /// Use local scope when printing the operation. This allows for using the
  /// printer in a more localized and thread-safe setting, but may not
  /// necessarily be identical to what the IR will look like when dumping
  /// the full module.
  OpPrintingFlags &useLocalScope();

  /// Return if the given ElementsAttr should be elided.
  bool shouldElideElementsAttr(ElementsAttr attr) const;

  /// Return the size limit for printing large ElementsAttr.
  Optional<int64_t> getLargeElementsAttrLimit() const;

  /// Return if debug information should be printed.
  bool shouldPrintDebugInfo() const;

  /// Return if debug information should be printed in the pretty form.
  bool shouldPrintDebugInfoPrettyForm() const;

  /// Return if operations should be printed in the generic form.
  bool shouldPrintGenericOpForm() const;

  /// Return if the printer should use local scope when dumping the IR.
  bool shouldUseLocalScope() const;

private:
  /// Elide large elements attributes if the number of elements is larger than
  /// the upper limit.
  Optional<int64_t> elementsAttrElementLimit;

  /// Print debug information.
  bool printDebugInfoFlag : 1;
  bool printDebugInfoPrettyFormFlag : 1;

  /// Print operations in the generic form.
  bool printGenericOpFormFlag : 1;

  /// Print operations with numberings local to the current operation.
  bool printLocalScope : 1;
};

//===----------------------------------------------------------------------===//
// Operation Value-Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperandRange

/// This class implements the operand iterators for the Operation class.
class OperandRange final : public llvm::detail::indexed_accessor_range_base<
                               OperandRange, OpOperand *, Value, Value, Value> {
public:
  using RangeBaseT::RangeBaseT;
  OperandRange(Operation *op);

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<OperandRange>;
  type_range getTypes() const { return {begin(), end()}; }
  auto getType() const { return getTypes(); }

  /// Return the operand index of the first element of this range. The range
  /// must not be empty.
  unsigned getBeginOperandIndex() const;

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OpOperand *offset_base(OpOperand *object, ptrdiff_t index) {
    return object + index;
  }
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Value dereference_iterator(OpOperand *object, ptrdiff_t index) {
    return object[index].get();
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// MutableOperandRange

/// This class provides a mutable adaptor for a range of operands. It allows for
/// setting, inserting, and erasing operands from the given range.
class MutableOperandRange {
public:
  /// A pair of a named attribute corresponding to an operand segment attribute,
  /// and the index within that attribute. The attribute should correspond to an
  /// i32 DenseElementsAttr.
  using OperandSegment = std::pair<unsigned, NamedAttribute>;

  /// Construct a new mutable range from the given operand, operand start index,
  /// and range length. `operandSegments` is an optional set of operand segments
  /// to be updated when mutating the operand list.
  MutableOperandRange(Operation *owner, unsigned start, unsigned length,
                      ArrayRef<OperandSegment> operandSegments = llvm::None);
  MutableOperandRange(Operation *owner);

  /// Slice this range into a sub range, with the additional operand segment.
  MutableOperandRange slice(unsigned subStart, unsigned subLen,
                            Optional<OperandSegment> segment = llvm::None);

  /// Append the given values to the range.
  void append(ValueRange values);

  /// Assign this range to the given values.
  void assign(ValueRange values);

  /// Assign the range to the given value.
  void assign(Value value);

  /// Erase the operands within the given sub-range.
  void erase(unsigned subStart, unsigned subLen = 1);

  /// Clear this range and erase all of the operands.
  void clear();

  /// Returns the current size of the range.
  unsigned size() const { return length; }

  /// Allow implicit conversion to an OperandRange.
  operator OperandRange() const;

  /// Returns the owning operation.
  Operation *getOwner() const { return owner; }

private:
  /// Update the length of this range to the one provided.
  void updateLength(unsigned newLength);

  /// The owning operation of this range.
  Operation *owner;

  /// The start index of the operand range within the owner operand list, and
  /// the length starting from `start`.
  unsigned start, length;

  /// Optional set of operand segments that should be updated when mutating the
  /// length of this range.
  SmallVector<std::pair<unsigned, NamedAttribute>, 1> operandSegments;
};

//===----------------------------------------------------------------------===//
// ResultRange

/// This class implements the result iterators for the Operation class.
class ResultRange final
    : public llvm::indexed_accessor_range<ResultRange, Operation *, OpResult,
                                          OpResult, OpResult> {
public:
  using indexed_accessor_range<ResultRange, Operation *, OpResult, OpResult,
                               OpResult>::indexed_accessor_range;
  ResultRange(Operation *op);

  /// Returns the types of the values within this range.
  using type_iterator = ArrayRef<Type>::iterator;
  using type_range = ArrayRef<Type>;
  type_range getTypes() const;
  auto getType() const { return getTypes(); }

private:
  /// See `llvm::indexed_accessor_range` for details.
  static OpResult dereference(Operation *op, ptrdiff_t index);

  /// Allow access to `dereference_iterator`.
  friend llvm::indexed_accessor_range<ResultRange, Operation *, OpResult,
                                      OpResult, OpResult>;
};

//===----------------------------------------------------------------------===//
// ValueRange

namespace detail {
/// The type representing the owner of a ValueRange. This is either a list of
/// values, operands, or an Operation+start index for results.
struct ValueRangeOwner {
  ValueRangeOwner(const Value *owner) : ptr(owner), startIndex(0) {}
  ValueRangeOwner(OpOperand *owner) : ptr(owner), startIndex(0) {}
  ValueRangeOwner(Operation *owner, unsigned startIndex)
      : ptr(owner), startIndex(startIndex) {}
  bool operator==(const ValueRangeOwner &rhs) const { return ptr == rhs.ptr; }

  /// The owner pointer of the range. The owner has represents three distinct
  /// states:
  /// const Value *: The owner is the base to a contiguous array of Value.
  /// OpOperand *  : The owner is the base to a contiguous array of operands.
  /// void*        : This owner is an Operation*. It is marked as void* here
  ///                because the definition of Operation is not visible here.
  PointerUnion<const Value *, OpOperand *, void *> ptr;

  /// Ths start index into the range. This is only used for Operation* owners.
  unsigned startIndex;
};
} // end namespace detail

/// This class provides an abstraction over the different types of ranges over
/// Values. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class ValueRange final
    : public llvm::detail::indexed_accessor_range_base<
          ValueRange, detail::ValueRangeOwner, Value, Value, Value> {
public:
  using RangeBaseT::RangeBaseT;

  template <typename Arg,
            typename = typename std::enable_if_t<
                std::is_constructible<ArrayRef<Value>, Arg>::value &&
                !std::is_convertible<Arg, Value>::value>>
  ValueRange(Arg &&arg) : ValueRange(ArrayRef<Value>(std::forward<Arg>(arg))) {}
  ValueRange(const Value &value) : ValueRange(&value, /*count=*/1) {}
  ValueRange(const std::initializer_list<Value> &values)
      : ValueRange(ArrayRef<Value>(values)) {}
  ValueRange(iterator_range<OperandRange::iterator> values)
      : ValueRange(OperandRange(values)) {}
  ValueRange(iterator_range<ResultRange::iterator> values)
      : ValueRange(ResultRange(values)) {}
  ValueRange(ArrayRef<BlockArgument> values)
      : ValueRange(ArrayRef<Value>(values.data(), values.size())) {}
  ValueRange(ArrayRef<Value> values = llvm::None);
  ValueRange(OperandRange values);
  ValueRange(ResultRange values);

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<ValueRange>;
  type_range getTypes() const { return {begin(), end()}; }
  auto getType() const { return getTypes(); }

private:
  using OwnerT = detail::ValueRangeOwner;

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(const OwnerT &owner, ptrdiff_t index);
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Value dereference_iterator(const OwnerT &owner, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// Operation Equivalency
//===----------------------------------------------------------------------===//

/// This class provides utilities for computing if two operations are
/// equivalent.
struct OperationEquivalence {
  enum Flags {
    None = 0,

    /// This flag signals that operands should not be considered when checking
    /// for equivalence. This allows for users to implement there own
    /// equivalence schemes for operand values. The number of operands are still
    /// checked, just not the operands themselves.
    IgnoreOperands = 1,

    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ IgnoreOperands)
  };

  /// Compute a hash for the given operation.
  static llvm::hash_code computeHash(Operation *op, Flags flags = Flags::None);

  /// Compare two operations and return if they are equivalent.
  static bool isEquivalentTo(Operation *lhs, Operation *rhs,
                             Flags flags = Flags::None);
};

/// Enable Bitmask enums for OperationEquivalence::Flags.
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // end namespace mlir

namespace llvm {
// Identifiers hash just like pointers, there is no need to hash the bytes.
template <> struct DenseMapInfo<mlir::OperationName> {
  static mlir::OperationName getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static mlir::OperationName getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::OperationName Val) {
    return DenseMapInfo<void *>::getHashValue(Val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::OperationName LHS, mlir::OperationName RHS) {
    return LHS == RHS;
  }
};

/// The pointer inside of an identifier comes from a StringMap, so its alignment
/// is always at least 4 and probably 8 (on 64-bit machines).  Allow LLVM to
/// steal the low bits.
template <> struct PointerLikeTypeTraits<mlir::OperationName> {
public:
  static inline void *getAsVoidPointer(mlir::OperationName I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::OperationName getFromVoidPointer(void *P) {
    return mlir::OperationName::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable = PointerLikeTypeTraits<
      mlir::OperationName::RepresentationUnion>::NumLowBitsAvailable;
};

} // end namespace llvm

#endif
