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

#ifndef MLIR_IR_OPERATIONSUPPORT_H
#define MLIR_IR_OPERATIONSUPPORT_H

#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
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
} // namespace llvm

namespace mlir {
class Dialect;
class DictionaryAttr;
class ElementsAttr;
class MutableOperandRangeRange;
class NamedAttrList;
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
class RewritePatternSet;
class Type;
class Value;
class ValueRange;
template <typename ValueRangeT>
class ValueTypeRange;

//===----------------------------------------------------------------------===//
// OperationName
//===----------------------------------------------------------------------===//

class OperationName {
public:
  using GetCanonicalizationPatternsFn =
      llvm::unique_function<void(RewritePatternSet &, MLIRContext *) const>;
  using FoldHookFn = llvm::unique_function<LogicalResult(
      Operation *, ArrayRef<Attribute>, SmallVectorImpl<OpFoldResult> &) const>;
  using HasTraitFn = llvm::unique_function<bool(TypeID) const>;
  using ParseAssemblyFn =
      llvm::unique_function<ParseResult(OpAsmParser &, OperationState &) const>;
  // Note: RegisteredOperationName is passed as reference here as the derived
  // class is defined below.
  using PopulateDefaultAttrsFn = llvm::unique_function<void(
      const RegisteredOperationName &, NamedAttrList &) const>;
  using PrintAssemblyFn =
      llvm::unique_function<void(Operation *, OpAsmPrinter &, StringRef) const>;
  using VerifyInvariantsFn =
      llvm::unique_function<LogicalResult(Operation *) const>;
  using VerifyRegionInvariantsFn =
      llvm::unique_function<LogicalResult(Operation *) const>;

protected:
  /// This class represents a type erased version of an operation. It contains
  /// all of the components necessary for opaquely interacting with an
  /// operation. If the operation is not registered, some of these components
  /// may not be populated.
  struct Impl {
    Impl(StringAttr name)
        : name(name), dialect(nullptr), interfaceMap(llvm::None) {}

    /// The name of the operation.
    StringAttr name;

    //===------------------------------------------------------------------===//
    // Registered Operation Info

    /// The following fields are only populated when the operation is
    /// registered.

    /// Returns true if the operation has been registered, i.e. if the
    /// registration info has been populated.
    bool isRegistered() const { return dialect; }

    /// This is the dialect that this operation belongs to.
    Dialect *dialect;

    /// The unique identifier of the derived Op class.
    TypeID typeID;

    /// A map of interfaces that were registered to this operation.
    detail::InterfaceMap interfaceMap;

    /// Internal callback hooks provided by the op implementation.
    FoldHookFn foldHookFn;
    GetCanonicalizationPatternsFn getCanonicalizationPatternsFn;
    HasTraitFn hasTraitFn;
    ParseAssemblyFn parseAssemblyFn;
    PopulateDefaultAttrsFn populateDefaultAttrsFn;
    PrintAssemblyFn printAssemblyFn;
    VerifyInvariantsFn verifyInvariantsFn;
    VerifyRegionInvariantsFn verifyRegionInvariantsFn;

    /// A list of attribute names registered to this operation in StringAttr
    /// form. This allows for operation classes to use StringAttr for attribute
    /// lookup/creation/etc., as opposed to raw strings.
    ArrayRef<StringAttr> attributeNames;
  };

public:
  OperationName(StringRef name, MLIRContext *context);

  /// Return if this operation is registered.
  bool isRegistered() const { return impl->isRegistered(); }

  /// If this operation is registered, returns the registered information, None
  /// otherwise.
  Optional<RegisteredOperationName> getRegisteredInfo() const;

  /// Returns true if the operation was registered with a particular trait, e.g.
  /// hasTrait<OperandsAreSignlessIntegerLike>(). Returns false if the operation
  /// is unregistered.
  template <template <typename T> class Trait>
  bool hasTrait() const {
    return hasTrait(TypeID::get<Trait>());
  }
  bool hasTrait(TypeID traitID) const {
    return isRegistered() && impl->hasTraitFn(traitID);
  }

  /// Returns true if the operation *might* have the provided trait. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide trait.
  template <template <typename T> class Trait>
  bool mightHaveTrait() const {
    return mightHaveTrait(TypeID::get<Trait>());
  }
  bool mightHaveTrait(TypeID traitID) const {
    return !isRegistered() || impl->hasTraitFn(traitID);
  }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this operation, null otherwise. This should not be used
  /// directly.
  template <typename T>
  typename T::Concept *getInterface() const {
    return impl->interfaceMap.lookup<T>();
  }

  /// Returns true if this operation has the given interface registered to it.
  template <typename T>
  bool hasInterface() const {
    return hasInterface(TypeID::get<T>());
  }
  bool hasInterface(TypeID interfaceID) const {
    return impl->interfaceMap.contains(interfaceID);
  }

  /// Returns true if the operation *might* have the provided interface. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide interface.
  template <typename T>
  bool mightHaveInterface() const {
    return mightHaveInterface(TypeID::get<T>());
  }
  bool mightHaveInterface(TypeID interfaceID) const {
    return !isRegistered() || hasInterface(interfaceID);
  }

  /// Return the dialect this operation is registered to if the dialect is
  /// loaded in the context, or nullptr if the dialect isn't loaded.
  Dialect *getDialect() const {
    return isRegistered() ? impl->dialect : impl->name.getReferencedDialect();
  }

  /// Return the name of the dialect this operation is registered to.
  StringRef getDialectNamespace() const;

  /// Return the operation name with dialect name stripped, if it has one.
  StringRef stripDialect() const { return getStringRef().split('.').second; }

  /// Return the name of this operation. This always succeeds.
  StringRef getStringRef() const { return getIdentifier(); }

  /// Return the name of this operation as a StringAttr.
  StringAttr getIdentifier() const { return impl->name; }

  void print(raw_ostream &os) const;
  void dump() const;

  /// Represent the operation name as an opaque pointer. (Used to support
  /// PointerLikeTypeTraits).
  void *getAsOpaquePointer() const { return const_cast<Impl *>(impl); }
  static OperationName getFromOpaquePointer(const void *pointer) {
    return OperationName(
        const_cast<Impl *>(reinterpret_cast<const Impl *>(pointer)));
  }

  bool operator==(const OperationName &rhs) const { return impl == rhs.impl; }
  bool operator!=(const OperationName &rhs) const { return !(*this == rhs); }

protected:
  OperationName(Impl *impl) : impl(impl) {}

  /// The internal implementation of the operation name.
  Impl *impl;

  /// Allow access to the Impl struct.
  friend MLIRContextImpl;
};

inline raw_ostream &operator<<(raw_ostream &os, OperationName info) {
  info.print(os);
  return os;
}

// Make operation names hashable.
inline llvm::hash_code hash_value(OperationName arg) {
  return llvm::hash_value(arg.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// RegisteredOperationName
//===----------------------------------------------------------------------===//

/// This is a "type erased" representation of a registered operation. This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks. Most user code should use
/// the concrete operation types.
class RegisteredOperationName : public OperationName {
public:
  /// Lookup the registered operation information for the given operation.
  /// Returns None if the operation isn't registered.
  static Optional<RegisteredOperationName> lookup(StringRef name,
                                                  MLIRContext *ctx);

  /// Register a new operation in a Dialect object.
  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T>
  static void insert(Dialect &dialect) {
    insert(T::getOperationName(), dialect, TypeID::get<T>(),
           T::getParseAssemblyFn(), T::getPrintAssemblyFn(),
           T::getVerifyInvariantsFn(), T::getVerifyRegionInvariantsFn(),
           T::getFoldHookFn(), T::getGetCanonicalizationPatternsFn(),
           T::getInterfaceMap(), T::getHasTraitFn(), T::getAttributeNames(),
           T::getPopulateDefaultAttrsFn());
  }
  /// The use of this method is in general discouraged in favor of
  /// 'insert<CustomOp>(dialect)'.
  static void
  insert(StringRef name, Dialect &dialect, TypeID typeID,
         ParseAssemblyFn &&parseAssembly, PrintAssemblyFn &&printAssembly,
         VerifyInvariantsFn &&verifyInvariants,
         VerifyRegionInvariantsFn &&verifyRegionInvariants,
         FoldHookFn &&foldHook,
         GetCanonicalizationPatternsFn &&getCanonicalizationPatterns,
         detail::InterfaceMap &&interfaceMap, HasTraitFn &&hasTrait,
         ArrayRef<StringRef> attrNames,
         PopulateDefaultAttrsFn &&populateDefaultAttrs);

  /// Return the dialect this operation is registered to.
  Dialect &getDialect() const { return *impl->dialect; }

  /// Return the unique identifier of the derived Op class.
  TypeID getTypeID() const { return impl->typeID; }

  /// Use the specified object to parse this ops custom assembly format.
  ParseResult parseAssembly(OpAsmParser &parser, OperationState &result) const;

  /// Return the static hook for parsing this operation assembly.
  const ParseAssemblyFn &getParseAssemblyFn() const {
    return impl->parseAssemblyFn;
  }

  /// This hook implements the AsmPrinter for this operation.
  void printAssembly(Operation *op, OpAsmPrinter &p,
                     StringRef defaultDialect) const {
    return impl->printAssemblyFn(op, p, defaultDialect);
  }

  /// These hooks implement the verifiers for this operation.  It should emits
  /// an error message and returns failure if a problem is detected, or returns
  /// success if everything is ok.
  LogicalResult verifyInvariants(Operation *op) const {
    return impl->verifyInvariantsFn(op);
  }
  LogicalResult verifyRegionInvariants(Operation *op) const {
    return impl->verifyRegionInvariantsFn(op);
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
    return impl->foldHookFn(op, operands, results);
  }

  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  void getCanonicalizationPatterns(RewritePatternSet &results,
                                   MLIRContext *context) const {
    return impl->getCanonicalizationPatternsFn(results, context);
  }

  /// Attach the given models as implementations of the corresponding interfaces
  /// for the concrete operation.
  template <typename... Models>
  void attachInterface() {
    impl->interfaceMap.insert<Models...>();
  }

  /// Returns true if the operation has a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() const {
    return hasTrait(TypeID::get<Trait>());
  }

  /// Returns true if the operation has a particular trait.
  bool hasTrait(TypeID traitID) const { return impl->hasTraitFn(traitID); }

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
  ArrayRef<StringAttr> getAttributeNames() const {
    return impl->attributeNames;
  }

  /// This hook implements the method to populate defaults attributes that are
  /// unset.
  void populateDefaultAttrs(NamedAttrList &attrs) const;

  /// Represent the operation name as an opaque pointer. (Used to support
  /// PointerLikeTypeTraits).
  static RegisteredOperationName getFromOpaquePointer(const void *pointer) {
    return RegisteredOperationName(
        const_cast<Impl *>(reinterpret_cast<const Impl *>(pointer)));
  }

private:
  RegisteredOperationName(Impl *impl) : OperationName(impl) {}

  /// Allow access to the constructor.
  friend OperationName;
};

inline Optional<RegisteredOperationName>
OperationName::getRegisteredInfo() const {
  return isRegistered() ? RegisteredOperationName(impl)
                        : Optional<RegisteredOperationName>();
}

//===----------------------------------------------------------------------===//
// Attribute Dictionary-Like Interface
//===----------------------------------------------------------------------===//

/// Attribute collections provide a dictionary-like interface. Define common
/// lookup functions.
namespace impl {

/// Unsorted string search or identifier lookups are linear scans.
template <typename IteratorT, typename NameT>
std::pair<IteratorT, bool> findAttrUnsorted(IteratorT first, IteratorT last,
                                            NameT name) {
  for (auto it = first; it != last; ++it)
    if (it->getName() == name)
      return {it, true};
  return {last, false};
}

/// Using llvm::lower_bound requires an extra string comparison to check whether
/// the returned iterator points to the found element or whether it indicates
/// the lower bound. Skip this redundant comparison by checking if `compare ==
/// 0` during the binary search.
template <typename IteratorT>
std::pair<IteratorT, bool> findAttrSorted(IteratorT first, IteratorT last,
                                          StringRef name) {
  ptrdiff_t length = std::distance(first, last);

  while (length > 0) {
    ptrdiff_t half = length / 2;
    IteratorT mid = first + half;
    int compare = mid->getName().strref().compare(name);
    if (compare < 0) {
      first = mid + 1;
      length = length - half - 1;
    } else if (compare > 0) {
      length = half;
    } else {
      return {mid, true};
    }
  }
  return {first, false};
}

/// StringAttr lookups on large attribute lists will switch to string binary
/// search. String binary searches become significantly faster than linear scans
/// with the identifier when the attribute list becomes very large.
template <typename IteratorT>
std::pair<IteratorT, bool> findAttrSorted(IteratorT first, IteratorT last,
                                          StringAttr name) {
  constexpr unsigned kSmallAttributeList = 16;
  if (std::distance(first, last) > kSmallAttributeList)
    return findAttrSorted(first, last, name.strref());
  return findAttrUnsorted(first, last, name);
}

/// Get an attribute from a sorted range of named attributes. Returns null if
/// the attribute was not found.
template <typename IteratorT, typename NameT>
Attribute getAttrFromSortedRange(IteratorT first, IteratorT last, NameT name) {
  std::pair<IteratorT, bool> result = findAttrSorted(first, last, name);
  return result.second ? result.first->getValue() : Attribute();
}

/// Get an attribute from a sorted range of named attributes. Returns None if
/// the attribute was not found.
template <typename IteratorT, typename NameT>
Optional<NamedAttribute>
getNamedAttrFromSortedRange(IteratorT first, IteratorT last, NameT name) {
  std::pair<IteratorT, bool> result = findAttrSorted(first, last, name);
  return result.second ? *result.first : Optional<NamedAttribute>();
}

} // namespace impl

//===----------------------------------------------------------------------===//
// NamedAttrList
//===----------------------------------------------------------------------===//

/// NamedAttrList is array of NamedAttributes that tracks whether it is sorted
/// and does some basic work to remain sorted.
class NamedAttrList {
public:
  using iterator = SmallVectorImpl<NamedAttribute>::iterator;
  using const_iterator = SmallVectorImpl<NamedAttribute>::const_iterator;
  using reference = NamedAttribute &;
  using const_reference = const NamedAttribute &;
  using size_type = size_t;

  NamedAttrList() : dictionarySorted({}, true) {}
  NamedAttrList(llvm::NoneType none) : NamedAttrList() {}
  NamedAttrList(ArrayRef<NamedAttribute> attributes);
  NamedAttrList(DictionaryAttr attributes);
  NamedAttrList(const_iterator inStart, const_iterator inEnd);

  template <typename Container>
  NamedAttrList(const Container &vec)
      : NamedAttrList(ArrayRef<NamedAttribute>(vec)) {}

  bool operator!=(const NamedAttrList &other) const {
    return !(*this == other);
  }
  bool operator==(const NamedAttrList &other) const {
    return attrs == other.attrs;
  }

  /// Add an attribute with the specified name.
  void append(StringRef name, Attribute attr);

  /// Add an attribute with the specified name.
  void append(StringAttr name, Attribute attr) {
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
  template <typename IteratorT,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<IteratorT>::iterator_category,
                std::input_iterator_tag>::value>>
  void append(IteratorT inStart, IteratorT inEnd) {
    // TODO: expand to handle case where values appended are in order & after
    // end of current list.
    dictionarySorted.setPointerAndInt(nullptr, false);
    attrs.append(inStart, inEnd);
  }

  /// Replaces the attributes with new list of attributes.
  void assign(const_iterator inStart, const_iterator inEnd);

  /// Replaces the attributes with new list of attributes.
  void assign(ArrayRef<NamedAttribute> range) {
    assign(range.begin(), range.end());
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
  Attribute get(StringAttr name) const;
  Attribute get(StringRef name) const;

  /// Return the specified named attribute if present, None otherwise.
  Optional<NamedAttribute> getNamed(StringRef name) const;
  Optional<NamedAttribute> getNamed(StringAttr name) const;

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  /// Returns the previous attribute value of `name`, or null if no
  /// attribute previously existed with `name`.
  Attribute set(StringAttr name, Attribute value);
  Attribute set(StringRef name, Attribute value);

  /// Erase the attribute with the given name from the list. Return the
  /// attribute that was erased, or nullptr if there was no attribute with such
  /// name.
  Attribute erase(StringAttr name);
  Attribute erase(StringRef name);

  iterator begin() { return attrs.begin(); }
  iterator end() { return attrs.end(); }
  const_iterator begin() const { return attrs.begin(); }
  const_iterator end() const { return attrs.end(); }

  NamedAttrList &operator=(const SmallVectorImpl<NamedAttribute> &rhs);
  operator ArrayRef<NamedAttribute>() const;

private:
  /// Return whether the attributes are sorted.
  bool isSorted() const { return dictionarySorted.getInt(); }

  /// Erase the attribute at the given iterator position.
  Attribute eraseImpl(SmallVectorImpl<NamedAttribute>::iterator it);

  /// Lookup an attribute in the list.
  template <typename AttrListT, typename NameT>
  static auto findAttr(AttrListT &attrs, NameT name) {
    return attrs.isSorted()
               ? impl::findAttrSorted(attrs.begin(), attrs.end(), name)
               : impl::findAttrUnsorted(attrs.begin(), attrs.end(), name);
  }

  // These are marked mutable as they may be modified (e.g., sorted)
  mutable SmallVector<NamedAttribute, 4> attrs;
  // Pair with cached DictionaryAttr and status of whether attrs is sorted.
  // Note: just because sorted does not mean a DictionaryAttr has been created
  // but the case where there is a DictionaryAttr but attrs isn't sorted should
  // not occur.
  mutable llvm::PointerIntPair<Attribute, 1, bool> dictionarySorted;
};

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

  OperationState(Location location, OperationName name, ValueRange operands,
                 TypeRange types, ArrayRef<NamedAttribute> attributes = {},
                 BlockRange successors = {},
                 MutableArrayRef<std::unique_ptr<Region>> regions = {});
  OperationState(Location location, StringRef name, ValueRange operands,
                 TypeRange types, ArrayRef<NamedAttribute> attributes = {},
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
    addAttribute(StringAttr::get(getContext(), name), attr);
  }

  /// Add an attribute with the specified name.
  void addAttribute(StringAttr name, Attribute attr) {
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
/// This class handles the management of operation operands. Operands are
/// stored either in a trailing array, or a dynamically resizable vector.
class alignas(8) OperandStorage {
public:
  OperandStorage(Operation *owner, OpOperand *trailingOperands,
                 ValueRange values);
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
  void eraseOperands(const BitVector &eraseIndices);

  /// Get the operation operands held by the storage.
  MutableArrayRef<OpOperand> getOperands() { return {operandStorage, size()}; }

  /// Return the number of operands held in the storage.
  unsigned size() { return numOperands; }

private:
  /// Resize the storage to the given size. Returns the array containing the new
  /// operands.
  MutableArrayRef<OpOperand> resize(Operation *owner, unsigned newSize);

  /// The total capacity number of operands that the storage can hold.
  unsigned capacity : 31;
  /// A flag indicating if the operand storage was dynamically allocated, as
  /// opposed to inlined into the owning operation.
  unsigned isStorageDynamic : 1;
  /// The number of operands within the storage.
  unsigned numOperands;
  /// A pointer to the operand storage.
  OpOperand *operandStorage;
};
} // namespace detail

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

  /// Do not verify the operation when using custom operation printers.
  OpPrintingFlags &assumeVerified();

  /// Use local scope when printing the operation. This allows for using the
  /// printer in a more localized and thread-safe setting, but may not
  /// necessarily be identical to what the IR will look like when dumping
  /// the full module.
  OpPrintingFlags &useLocalScope();

  /// Print users of values as comments.
  OpPrintingFlags &printValueUsers();

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

  /// Return if operation verification should be skipped.
  bool shouldAssumeVerified() const;

  /// Return if the printer should use local scope when dumping the IR.
  bool shouldUseLocalScope() const;

  /// Return if the printer should print users of values.
  bool shouldPrintValueUsers() const;

private:
  /// Elide large elements attributes if the number of elements is larger than
  /// the upper limit.
  Optional<int64_t> elementsAttrElementLimit;

  /// Print debug information.
  bool printDebugInfoFlag : 1;
  bool printDebugInfoPrettyFormFlag : 1;

  /// Print operations in the generic form.
  bool printGenericOpFormFlag : 1;

  /// Skip operation verification.
  bool assumeVerifiedFlag : 1;

  /// Print operations with numberings local to the current operation.
  bool printLocalScope : 1;

  /// Print users of values.
  bool printValueUsersFlag : 1;
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

} // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::OperationName> {
  static mlir::OperationName getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static mlir::OperationName getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::OperationName val) {
    return DenseMapInfo<void *>::getHashValue(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::OperationName lhs, mlir::OperationName rhs) {
    return lhs == rhs;
  }
};
template <>
struct DenseMapInfo<mlir::RegisteredOperationName>
    : public DenseMapInfo<mlir::OperationName> {
  static mlir::RegisteredOperationName getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::RegisteredOperationName::getFromOpaquePointer(pointer);
  }
  static mlir::RegisteredOperationName getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::RegisteredOperationName::getFromOpaquePointer(pointer);
  }
};

template <>
struct PointerLikeTypeTraits<mlir::OperationName> {
  static inline void *getAsVoidPointer(mlir::OperationName I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::OperationName getFromVoidPointer(void *P) {
    return mlir::OperationName::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable =
      PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
};
template <>
struct PointerLikeTypeTraits<mlir::RegisteredOperationName>
    : public PointerLikeTypeTraits<mlir::OperationName> {
  static inline mlir::RegisteredOperationName getFromVoidPointer(void *P) {
    return mlir::RegisteredOperationName::getFromOpaquePointer(P);
  }
};

} // namespace llvm

#endif
