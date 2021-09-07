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

namespace llvm {
class BitVector;
} // end namespace llvm

namespace mlir {
class Dialect;
class DictionaryAttr;
class ElementsAttr;
class MutableOperandRangeRange;
class Operation;
struct OperationState;
class OpAsmParser;
class OpAsmParserResult;
class OpAsmPrinter;
class OperandRange;
class OperandRangeRange;
class OpFoldResult;
class ParseResult;
class Pattern;
class Region;
class ResultRange;
class RewritePattern;
class Type;
class Value;
class ValueRange;
template <typename ValueRangeT>
class ValueTypeRange;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

//===----------------------------------------------------------------------===//
// AbstractOperation
//===----------------------------------------------------------------------===//

/// This is a "type erased" representation of a registered operation.  This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks.  Most user code should use
/// the concrete operation types.
class AbstractOperation {
public:
  using GetCanonicalizationPatternsFn =
      llvm::unique_function<void(RewritePatternSet &, MLIRContext *) const>;
  using FoldHookFn = llvm::unique_function<LogicalResult(
      Operation *, ArrayRef<Attribute>, SmallVectorImpl<OpFoldResult> &) const>;
  using HasTraitFn = llvm::unique_function<bool(TypeID) const>;
  using ParseAssemblyFn =
      llvm::unique_function<ParseResult(OpAsmParser &, OperationState &) const>;
  using PrintAssemblyFn =
      llvm::unique_function<void(Operation *, OpAsmPrinter &, StringRef) const>;
  using VerifyInvariantsFn =
      llvm::unique_function<LogicalResult(Operation *) const>;

  /// This is the name of the operation.
  const Identifier name;

  /// This is the dialect that this operation belongs to.
  Dialect &dialect;

  /// The unique identifier of the derived Op class.
  TypeID typeID;

  /// Use the specified object to parse this ops custom assembly format.
  ParseResult parseAssembly(OpAsmParser &parser, OperationState &result) const;

  /// Return the static hook for parsing this operation assembly.
  const ParseAssemblyFn &getParseAssemblyFn() const { return parseAssemblyFn; }

  /// This hook implements the AsmPrinter for this operation.
  void printAssembly(Operation *op, OpAsmPrinter &p,
                     StringRef defaultDialect) const {
    return printAssemblyFn(op, p, defaultDialect);
  }

  /// This hook implements the verifier for this operation.  It should emits an
  /// error message and returns failure if a problem is detected, or returns
  /// success if everything is ok.
  LogicalResult verifyInvariants(Operation *op) const {
    return verifyInvariantsFn(op);
  }

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
  LogicalResult foldHook(Operation *op, ArrayRef<Attribute> operands,
                         SmallVectorImpl<OpFoldResult> &results) const {
    return foldHookFn(op, operands, results);
  }

  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  void getCanonicalizationPatterns(RewritePatternSet &results,
                                   MLIRContext *context) const {
    return getCanonicalizationPatternsFn(results, context);
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this operation, null otherwise. This should not be used
  /// directly.
  template <typename T>
  typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  /// Returns true if this operation has the given interface registered to it.
  bool hasInterface(TypeID interfaceID) const {
    return interfaceMap.contains(interfaceID);
  }

  /// Returns true if the operation has a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() const {
    return hasTraitFn(TypeID::get<Trait>());
  }

  /// Returns true if the operation has a particular trait.
  bool hasTrait(TypeID traitID) const { return hasTraitFn(traitID); }

  /// Look up the specified operation in the specified MLIRContext and return a
  /// pointer to it if present.  Otherwise, return a null pointer.
  static const AbstractOperation *lookup(StringRef opName,
                                         MLIRContext *context) {
    return lookupMutable(opName, context);
  }

  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T>
  static void insert(Dialect &dialect) {
    insert(T::getOperationName(), dialect, TypeID::get<T>(),
           T::getParseAssemblyFn(), T::getPrintAssemblyFn(),
           T::getVerifyInvariantsFn(), T::getFoldHookFn(),
           T::getGetCanonicalizationPatternsFn(), T::getInterfaceMap(),
           T::getHasTraitFn(), T::getAttributeNames());
  }

  /// Register a new operation in a Dialect object.
  /// The use of this method is in general discouraged in favor of
  /// 'insert<CustomOp>(dialect)'.
  static void
  insert(StringRef name, Dialect &dialect, TypeID typeID,
         ParseAssemblyFn &&parseAssembly, PrintAssemblyFn &&printAssembly,
         VerifyInvariantsFn &&verifyInvariants, FoldHookFn &&foldHook,
         GetCanonicalizationPatternsFn &&getCanonicalizationPatterns,
         detail::InterfaceMap &&interfaceMap, HasTraitFn &&hasTrait,
         ArrayRef<StringRef> attrNames);

  /// Return the list of cached attribute names registered to this operation.
  /// The order of attributes cached here is unique to each type of operation,
  /// and the interpretation of this attribute list should generally be driven
  /// by the respective operation. In many cases, this caching removes the need
  /// to use the raw string name of a known attribute.
  ///
  /// For example the ODS generator, with an op defining the following
  /// attributes:
  ///
  ///   let arguments = (ins I32Attr:$attr1, I32Attr:$attr2);
  ///
  /// ... may produce an order here of ["attr1", "attr2"]. This allows for the
  /// ODS generator to directly access the cached name for a known attribute,
  /// greatly simplifying the cost and complexity of attribute usage produced by
  /// the generator.
  ///
  ArrayRef<Identifier> getAttributeNames() const { return attributeNames; }

private:
  AbstractOperation(StringRef name, Dialect &dialect, TypeID typeID,
                    ParseAssemblyFn &&parseAssembly,
                    PrintAssemblyFn &&printAssembly,
                    VerifyInvariantsFn &&verifyInvariants,
                    FoldHookFn &&foldHook,
                    GetCanonicalizationPatternsFn &&getCanonicalizationPatterns,
                    detail::InterfaceMap &&interfaceMap, HasTraitFn &&hasTrait,
                    ArrayRef<Identifier> attrNames);

  /// Give Op access to lookupMutable.
  template <typename ConcreteType, template <typename T> class... Traits>
  friend class Op;

  /// Look up the specified operation in the specified MLIRContext and return a
  /// pointer to it if present.  Otherwise, return a null pointer.
  static AbstractOperation *lookupMutable(StringRef opName,
                                          MLIRContext *context);

  /// A map of interfaces that were registered to this operation.
  detail::InterfaceMap interfaceMap;

  /// Internal callback hooks provided by the op implementation.
  FoldHookFn foldHookFn;
  GetCanonicalizationPatternsFn getCanonicalizationPatternsFn;
  HasTraitFn hasTraitFn;
  ParseAssemblyFn parseAssemblyFn;
  PrintAssemblyFn printAssemblyFn;
  VerifyInvariantsFn verifyInvariantsFn;

  /// A list of attribute names registered to this operation in identifier form.
  /// This allows for operation classes to use identifiers for attribute
  /// lookup/creation/etc., as opposed to strings.
  ArrayRef<Identifier> attributeNames;
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
  NamedAttrList(DictionaryAttr attributes);
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
  void append(Identifier name, Attribute attr) {
    append(NamedAttribute(name, attr));
  }

  /// Append the given named attribute.
  void append(NamedAttribute attr) { push_back(attr); }

  /// Add an array of named attributes.
  template <typename RangeT>
  void append(RangeT &&newAttributes) {
    append(std::begin(newAttributes), std::end(newAttributes));
  }

  /// Add a range of named attributes.
  template <typename IteratorT>
  void append(IteratorT in_start, IteratorT in_end) {
    // TODO: expand to handle case where values appended are in order & after
    // end of current list.
    dictionarySorted.setPointerAndInt(nullptr, false);
    attrs.append(in_start, in_end);
  }

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

  /// Returns an entry with a duplicate name the list, if it exists, else
  /// returns llvm::None.
  Optional<NamedAttribute> findDuplicate() const;

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
  /// value. Otherwise, add a new attribute with the specified name/value.
  /// Returns the previous attribute value of `name`, or null if no
  /// attribute previously existed with `name`.
  Attribute set(Identifier name, Attribute value);
  Attribute set(StringRef name, Attribute value);

  /// Erase the attribute with the given name from the list. Return the
  /// attribute that was erased, or nullptr if there was no attribute with such
  /// name.
  Attribute erase(Identifier name);
  Attribute erase(StringRef name);

  const_iterator begin() const { return attrs.begin(); }
  const_iterator end() const { return attrs.end(); }

  NamedAttrList &operator=(const SmallVectorImpl<NamedAttribute> &rhs);
  operator ArrayRef<NamedAttribute>() const;

private:
  /// Return whether the attributes are sorted.
  bool isSorted() const { return dictionarySorted.getInt(); }

  /// Erase the attribute at the given iterator position.
  Attribute eraseImpl(SmallVectorImpl<NamedAttribute>::iterator it);

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
  StringRef getDialectNamespace() const;

  /// Return the Dialect this operation is registered to if it is loaded in the
  /// context, or nullptr if the dialect isn't loaded.
  Dialect *getDialect() const {
    if (const auto *abstractOp = getAbstractOperation())
      return &abstractOp->dialect;
    return representation.get<Identifier>().getDialect();
  }

  /// Return the operation name with dialect name stripped, if it has one.
  StringRef stripDialect() const;

  /// Return the name of this operation. This always succeeds.
  StringRef getStringRef() const;

  /// Return the name of this operation as an identifier. This always succeeds.
  Identifier getIdentifier() const;

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const {
    return representation.dyn_cast<const AbstractOperation *>();
  }

  void print(raw_ostream &os) const;
  void dump() const;

  void *getAsOpaquePointer() const {
    return static_cast<void *>(representation.getOpaqueValue());
  }
  static OperationName getFromOpaquePointer(const void *pointer);

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
#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && (BYTE_ORDER == BIG_ENDIAN)
  TrailingOperandStorage() : numOperands(0), capacity(0), reserved(0) {}
#else
  TrailingOperandStorage() : reserved(0), capacity(0), numOperands(0) {}
#endif
  ~TrailingOperandStorage() {
    for (auto &operand : getOperands())
      operand.~OpOperand();
  }

  /// Return the operands held by this storage.
  MutableArrayRef<OpOperand> getOperands() {
    return {getTrailingObjects<OpOperand>(), numOperands};
  }

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && (BYTE_ORDER == BIG_ENDIAN)
  /// The number of operands within the storage.
  unsigned numOperands;
  /// The total capacity number of operands that the storage can hold.
  unsigned capacity : 31;
  /// We reserve a range of bits for use by the operand storage.
  unsigned reserved : 1;
#else
  /// We reserve a range of bits for use by the operand storage.
  unsigned reserved : 1;
  /// The total capacity number of operands that the storage can hold.
  unsigned capacity : 31;
  /// The number of operands within the storage.
  unsigned numOperands;
#endif
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

  /// Erase the operands held by the storage that have their corresponding bit
  /// set in `eraseIndices`.
  void eraseOperands(const llvm::BitVector &eraseIndices);

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
  /// Pointer type traits for the storage pointer that ensures that we use the
  /// lowest bit for the storage pointer.
  struct StoragePointerLikeTypeTraits
      : llvm::PointerLikeTypeTraits<TrailingOperandStorage *> {
    static constexpr int NumLowBitsAvailable = 1;
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
    return inlineStorage;
  }

  /// Returns the storage container if this storage is dynamic.
  TrailingOperandStorage &getDynamicStorage() {
    assert(isDynamicStorage() && "expected dynamic storage");
    return *dynamicStorage.getPointer();
  }

  /// Returns true if the storage is currently dynamic.
  bool isDynamicStorage() const { return dynamicStorage.getInt(); }

  /// The current representation of the storage. This is either a
  /// InlineOperandStorage, or a pointer to a InlineOperandStorage.
  union {
    TrailingOperandStorage inlineStorage;
    llvm::PointerIntPair<TrailingOperandStorage *, 1, bool,
                         StoragePointerLikeTypeTraits>
        dynamicStorage;
  };

  /// This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperandStorage, OpOperand>;
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

  /// Enables the elision of large elements attributes by printing a lexically
  /// valid but otherwise meaningless form instead of the element data. The
  /// `largeElementLimit` is used to configure what is considered to be a
  /// "large" ElementsAttr by providing an upper limit to the number of
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

  /// Split this range into a set of contiguous subranges using the given
  /// elements attribute, which contains the sizes of the sub ranges.
  OperandRangeRange split(ElementsAttr segmentSizes) const;

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
// OperandRangeRange

/// This class represents a contiguous range of operand ranges, e.g. from a
/// VariadicOfVariadic operand group.
class OperandRangeRange final
    : public llvm::indexed_accessor_range<
          OperandRangeRange, std::pair<OpOperand *, Attribute>, OperandRange,
          OperandRange, OperandRange> {
  using OwnerT = std::pair<OpOperand *, Attribute>;
  using RangeBaseT =
      llvm::indexed_accessor_range<OperandRangeRange, OwnerT, OperandRange,
                                   OperandRange, OperandRange>;

public:
  using RangeBaseT::RangeBaseT;

  /// Returns the range of types of the values within this range.
  TypeRangeRange getTypes() const { return TypeRangeRange(*this); }
  auto getType() const { return getTypes(); }

  /// Construct a range given a parent set of operands, and an I32 elements
  /// attribute containing the sizes of the sub ranges.
  OperandRangeRange(OperandRange operands, Attribute operandSegments);

  /// Flatten all of the sub ranges into a single contiguous operand range.
  OperandRange join() const;

private:
  /// See `llvm::indexed_accessor_range` for details.
  static OperandRange dereference(const OwnerT &object, ptrdiff_t index);

  /// Allow access to `dereference_iterator`.
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
  MutableOperandRange
  slice(unsigned subStart, unsigned subLen,
        Optional<OperandSegment> segment = llvm::None) const;

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

  /// Returns if the current range is empty.
  bool empty() const { return size() == 0; }

  /// Allow implicit conversion to an OperandRange.
  operator OperandRange() const;

  /// Returns the owning operation.
  Operation *getOwner() const { return owner; }

  /// Split this range into a set of contiguous subranges using the given
  /// elements attribute, which contains the sizes of the sub ranges.
  MutableOperandRangeRange split(NamedAttribute segmentSizes) const;

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
  SmallVector<OperandSegment, 1> operandSegments;
};

//===----------------------------------------------------------------------===//
// MutableOperandRangeRange

/// This class represents a contiguous range of mutable operand ranges, e.g.
/// from a VariadicOfVariadic operand group.
class MutableOperandRangeRange final
    : public llvm::indexed_accessor_range<
          MutableOperandRangeRange,
          std::pair<MutableOperandRange, NamedAttribute>, MutableOperandRange,
          MutableOperandRange, MutableOperandRange> {
  using OwnerT = std::pair<MutableOperandRange, NamedAttribute>;
  using RangeBaseT =
      llvm::indexed_accessor_range<MutableOperandRangeRange, OwnerT,
                                   MutableOperandRange, MutableOperandRange,
                                   MutableOperandRange>;

public:
  using RangeBaseT::RangeBaseT;

  /// Construct a range given a parent set of operands, and an I32 tensor
  /// elements attribute containing the sizes of the sub ranges.
  MutableOperandRangeRange(const MutableOperandRange &operands,
                           NamedAttribute operandSegmentAttr);

  /// Flatten all of the sub ranges into a single contiguous mutable operand
  /// range.
  MutableOperandRange join() const;

  /// Allow implicit conversion to an OperandRangeRange.
  operator OperandRangeRange() const;

private:
  /// See `llvm::indexed_accessor_range` for details.
  static MutableOperandRange dereference(const OwnerT &object, ptrdiff_t index);

  /// Allow access to `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// ResultRange

/// This class implements the result iterators for the Operation class.
class ResultRange final
    : public llvm::detail::indexed_accessor_range_base<
          ResultRange, detail::OpResultImpl *, OpResult, OpResult, OpResult> {
public:
  using RangeBaseT::RangeBaseT;

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<ResultRange>;
  type_range getTypes() const { return {begin(), end()}; }
  auto getType() const { return getTypes(); }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  class UseIterator;
  using use_iterator = UseIterator;
  using use_range = iterator_range<use_iterator>;

  /// Returns a range of all uses of results within this range, which is useful
  /// for iterating over all uses.
  use_range getUses() const;
  use_iterator use_begin() const;
  use_iterator use_end() const;

  /// Returns true if no results in this range have uses.
  bool use_empty() const {
    return llvm::all_of(*this,
                        [](OpResult result) { return result.use_empty(); });
  }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  /// Returns a range of all users.
  user_range getUsers();
  user_iterator user_begin();
  user_iterator user_end();

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static detail::OpResultImpl *offset_base(detail::OpResultImpl *object,
                                           ptrdiff_t index) {
    return object->getNextResultAtOffset(index);
  }
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OpResult dereference_iterator(detail::OpResultImpl *object,
                                       ptrdiff_t index) {
    return offset_base(object, index);
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

/// This class implements a use iterator for a range of operation results.
/// This iterates over all uses of all results within the given result range.
class ResultRange::UseIterator final
    : public llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                        OpOperand> {
public:
  /// Initialize the UseIterator. Specify `end` to return iterator to last
  /// use, otherwise this is an iterator to the first use.
  explicit UseIterator(ResultRange results, bool end = false);

  using llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                   OpOperand>::operator++;
  UseIterator &operator++();
  OpOperand *operator->() const { return use.getOperand(); }
  OpOperand &operator*() const { return *use.getOperand(); }

  bool operator==(const UseIterator &rhs) const { return use == rhs.use; }
  bool operator!=(const UseIterator &rhs) const { return !(*this == rhs); }

private:
  void skipOverResultsWithNoUsers();

  /// The range of results being iterated over.
  ResultRange::iterator it, endIt;
  /// The use of the result.
  Value::use_iterator use;
};

//===----------------------------------------------------------------------===//
// ValueRange

/// This class provides an abstraction over the different types of ranges over
/// Values. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class ValueRange final
    : public llvm::detail::indexed_accessor_range_base<
          ValueRange,
          PointerUnion<const Value *, OpOperand *, detail::OpResultImpl *>,
          Value, Value, Value> {
public:
  /// The type representing the owner of a ValueRange. This is either a list of
  /// values, operands, or results.
  using OwnerT =
      PointerUnion<const Value *, OpOperand *, detail::OpResultImpl *>;

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

    // When provided, the location attached to the operation are ignored.
    IgnoreLocations = 1,

    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ IgnoreLocations)
  };

  /// Compute a hash for the given operation.
  /// The `hashOperands` and `hashResults` callbacks are expected to return a
  /// unique hash_code for a given Value.
  static llvm::hash_code computeHash(
      Operation *op,
      function_ref<llvm::hash_code(Value)> hashOperands =
          [](Value v) { return hash_value(v); },
      function_ref<llvm::hash_code(Value)> hashResults =
          [](Value v) { return hash_value(v); },
      Flags flags = Flags::None);

  /// Helper that can be used with `computeHash` above to ignore operation
  /// operands/result mapping.
  static llvm::hash_code ignoreHashValue(Value) { return llvm::hash_code{}; }
  /// Helper that can be used with `computeHash` above to ignore operation
  /// operands/result mapping.
  static llvm::hash_code directHashValue(Value v) { return hash_value(v); }

  /// Compare two operations and return if they are equivalent.
  /// `mapOperands` and `mapResults` are optional callbacks that allows the
  /// caller to check the mapping of SSA value between the lhs and rhs
  /// operations. It is expected to return success if the mapping is valid and
  /// failure if it conflicts with a previous mapping.
  static bool
  isEquivalentTo(Operation *lhs, Operation *rhs,
                 function_ref<LogicalResult(Value, Value)> mapOperands,
                 function_ref<LogicalResult(Value, Value)> mapResults,
                 Flags flags = Flags::None);

  /// Helper that can be used with `isEquivalentTo` above to ignore operation
  /// operands/result mapping.
  static LogicalResult ignoreValueEquivalence(Value lhs, Value rhs) {
    return success();
  }
  /// Helper that can be used with `isEquivalentTo` above to ignore operation
  /// operands/result mapping.
  static LogicalResult exactValueMatch(Value lhs, Value rhs) {
    return success(lhs == rhs);
  }
};

/// Enable Bitmask enums for OperationEquivalence::Flags.
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // end namespace mlir

namespace llvm {
// Identifiers hash just like pointers, there is no need to hash the bytes.
template <>
struct DenseMapInfo<mlir::OperationName> {
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
template <>
struct PointerLikeTypeTraits<mlir::OperationName> {
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
