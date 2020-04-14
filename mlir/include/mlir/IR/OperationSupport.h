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
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TrailingObjects.h"
#include <memory>

namespace mlir {
class Block;
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
class SuccessorRange;
class Type;
class Value;
class ValueRange;
template <typename ValueRangeT> class ValueTypeRange;

/// This is an adaptor from a list of values to named operands of OpTy.  In a
/// generic operation context, e.g., in dialect conversions, an ordered array of
/// `Value`s is treated as operands of `OpTy`.  This adaptor takes a reference
/// to the array and provides accessors with the same names as `OpTy` for
/// operands.  This makes possible to create function templates that operate on
/// either OpTy or OperandAdaptor<OpTy> seamlessly.
template <typename OpTy> using OperandAdaptor = typename OpTy::OperandAdaptor;

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
  const StringRef name;

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
    return reinterpret_cast<typename T::Concept *>(
        getRawInterface(T::getInterfaceID()));
  }

  /// Returns if the operation has a particular trait.
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
        T::getRawInterface, T::hasTrait);
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
      void *(&getRawInterface)(TypeID interfaceID),
      bool (&hasTrait)(TypeID traitID))
      : name(name), dialect(dialect), typeID(typeID),
        parseAssembly(parseAssembly), printAssembly(printAssembly),
        verifyInvariants(verifyInvariants), foldHook(foldHook),
        getCanonicalizationPatterns(getCanonicalizationPatterns),
        opProperties(opProperties), getRawInterface(getRawInterface),
        hasRawTrait(hasTrait) {}

  /// The properties of the operation.
  const OperationProperties opProperties;

  /// Returns a raw instance of the concept for the given interface id if it is
  /// registered to this operation, nullptr otherwise. This should not be used
  /// directly.
  void *(&getRawInterface)(TypeID interfaceID);

  /// This hook returns if the operation contains the trait corresponding
  /// to the given TypeID.
  bool (&hasRawTrait)(TypeID traitID);
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

  /// Return the name of this operation.  This always succeeds.
  StringRef getStringRef() const;

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
  SmallVector<NamedAttribute, 4> attributes;
  /// Successors of this operation and their respective operands.
  SmallVector<Block *, 1> successors;
  /// Regions that the op will hold.
  SmallVector<std::unique_ptr<Region>, 1> regions;
  /// If the operation has a resizable operand list.
  bool resizableOperandList = false;

public:
  OperationState(Location location, StringRef name);

  OperationState(Location location, OperationName name);

  OperationState(Location location, StringRef name, ValueRange operands,
                 ArrayRef<Type> types, ArrayRef<NamedAttribute> attributes,
                 ArrayRef<Block *> successors = {},
                 MutableArrayRef<std::unique_ptr<Region>> regions = {},
                 bool resizableOperandList = false);

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
    attributes.push_back({name, attr});
  }

  /// Add an array of named attributes.
  void addAttributes(ArrayRef<NamedAttribute> newAttributes) {
    attributes.append(newAttributes.begin(), newAttributes.end());
  }

  /// Add an array of successors.
  void addSuccessors(ArrayRef<Block *> newSuccessors) {
    successors.append(newSuccessors.begin(), newSuccessors.end());
  }
  void addSuccessors(Block *successor) { successors.push_back(successor); }
  void addSuccessors(SuccessorRange newSuccessors);

  /// Create a region that should be attached to the operation.  These regions
  /// can be filled in immediately without waiting for Operation to be
  /// created.  When it is, the region bodies will be transferred.
  Region *addRegion();

  /// Take a region that should be attached to the Operation.  The body of the
  /// region will be transferred when the Operation is constructed.  If the
  /// region is null, a new empty region will be attached to the Operation.
  void addRegion(std::unique_ptr<Region> &&region);

  /// Sets the operand list of the operation as resizable.
  void setOperandListToResizable(bool isResizable = true) {
    resizableOperandList = isResizable;
  }

  /// Get the context held by this operation state.
  MLIRContext *getContext() { return location->getContext(); }
};

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility class holding the information necessary to dynamically resize
/// operands.
struct ResizableStorage {
  ResizableStorage(OpOperand *opBegin, unsigned numOperands)
      : firstOpAndIsDynamic(opBegin, false), capacity(numOperands) {}

  ~ResizableStorage() { cleanupStorage(); }

  /// Cleanup any allocated storage.
  void cleanupStorage() {
    // If the storage is dynamic, then we need to free the storage.
    if (isStorageDynamic())
      free(firstOpAndIsDynamic.getPointer());
  }

  /// Sets the storage pointer to a new dynamically allocated block.
  void setDynamicStorage(OpOperand *opBegin) {
    /// Cleanup the old storage if necessary.
    cleanupStorage();
    firstOpAndIsDynamic.setPointerAndInt(opBegin, true);
  }

  /// Returns the current storage pointer.
  OpOperand *getPointer() { return firstOpAndIsDynamic.getPointer(); }

  /// Returns if the current storage of operands is in the trailing objects is
  /// in a dynamically allocated memory block.
  bool isStorageDynamic() const { return firstOpAndIsDynamic.getInt(); }

  /// A pointer to the first operand element. This is either to the trailing
  /// objects storage, or a dynamically allocated block of memory.
  llvm::PointerIntPair<OpOperand *, 1, bool> firstOpAndIsDynamic;

  // The maximum number of operands that can be currently held by the storage.
  unsigned capacity;
};

/// This class handles the management of operation operands. Operands are
/// stored similarly to the elements of a SmallVector except for two key
/// differences. The first is the inline storage, which is a trailing objects
/// array. The second is that being able to dynamically resize the operand list
/// is optional.
class OperandStorage final
    : private llvm::TrailingObjects<OperandStorage, ResizableStorage,
                                    OpOperand> {
public:
  OperandStorage(unsigned numOperands, bool resizable)
      : numOperands(numOperands), resizable(resizable) {
    // Initialize the resizable storage.
    if (resizable) {
      new (&getResizableStorage())
          ResizableStorage(getTrailingObjects<OpOperand>(), numOperands);
    }
  }

  ~OperandStorage() {
    // Manually destruct the operands.
    for (auto &operand : getOperands())
      operand.~OpOperand();

    // If the storage is resizable then destruct the utility.
    if (resizable)
      getResizableStorage().~ResizableStorage();
  }

  /// Replace the operands contained in the storage with the ones provided in
  /// 'operands'.
  void setOperands(Operation *owner, ValueRange operands);

  /// Erase an operand held by the storage.
  void eraseOperand(unsigned index);

  /// Get the operation operands held by the storage.
  MutableArrayRef<OpOperand> getOperands() {
    return {getRawOperands(), size()};
  }

  /// Return the number of operands held in the storage.
  unsigned size() const { return numOperands; }

  /// Returns the additional size necessary for allocating this object.
  static size_t additionalAllocSize(unsigned numOperands, bool resizable) {
    return additionalSizeToAlloc<ResizableStorage, OpOperand>(resizable ? 1 : 0,
                                                              numOperands);
  }

  /// Returns if this storage is resizable.
  bool isResizable() const { return resizable; }

private:
  /// Clear the storage and destroy the current operands held by the storage.
  void clear() { numOperands = 0; }

  /// Returns the current pointer for the raw operands array.
  OpOperand *getRawOperands() {
    return resizable ? getResizableStorage().getPointer()
                     : getTrailingObjects<OpOperand>();
  }

  /// Returns the resizable operand utility class.
  ResizableStorage &getResizableStorage() {
    assert(resizable);
    return *getTrailingObjects<ResizableStorage>();
  }

  /// Grow the internal resizable operand storage.
  void grow(ResizableStorage &resizeUtil, size_t minSize);

  /// The current number of operands, and the current max operand capacity.
  unsigned numOperands : 31;

  /// Whether this storage is resizable or not.
  bool resizable : 1;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperandStorage, ResizableStorage, OpOperand>;
  size_t numTrailingObjects(OverloadToken<ResizableStorage>) const {
    return resizable ? 1 : 0;
  }
};
} // end namespace detail

//===----------------------------------------------------------------------===//
// TrailingOpResult
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides the implementation for a trailing operation result.
struct TrailingOpResult {
  /// The only element is the trailing result number, or the offset from the
  /// beginning of the trailing array.
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
// TypeRange

/// This class provides an abstraction over the various different ranges of
/// value types. In many cases, this prevents the need to explicitly materialize
/// a SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class TypeRange
    : public llvm::detail::indexed_accessor_range_base<
          TypeRange,
          llvm::PointerUnion<const Value *, const Type *, OpOperand *>, Type,
          Type, Type> {
public:
  using RangeBaseT::RangeBaseT;
  TypeRange(ArrayRef<Type> types = llvm::None);
  explicit TypeRange(OperandRange values);
  explicit TypeRange(ResultRange values);
  explicit TypeRange(ValueRange values);
  explicit TypeRange(ArrayRef<Value> values);
  explicit TypeRange(ArrayRef<BlockArgument> values)
      : TypeRange(ArrayRef<Value>(values.data(), values.size())) {}
  template <typename ValueRangeT>
  TypeRange(ValueTypeRange<ValueRangeT> values)
      : TypeRange(ValueRangeT(values.begin().getCurrent(),
                              values.end().getCurrent())) {}
  template <typename Arg,
            typename = typename std::enable_if_t<
                std::is_constructible<ArrayRef<Type>, Arg>::value>>
  TypeRange(Arg &&arg) : TypeRange(ArrayRef<Type>(std::forward<Arg>(arg))) {}
  TypeRange(std::initializer_list<Type> types)
      : TypeRange(ArrayRef<Type>(types)) {}

private:
  /// The owner of the range is either:
  /// * A pointer to the first element of an array of values.
  /// * A pointer to the first element of an array of types.
  /// * A pointer to the first element of an array of operands.
  using OwnerT = llvm::PointerUnion<const Value *, const Type *, OpOperand *>;

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(OwnerT object, ptrdiff_t index);
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Type dereference_iterator(OwnerT object, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// ValueTypeRange

/// This class implements iteration on the types of a given range of values.
template <typename ValueIteratorT>
class ValueTypeIterator final
    : public llvm::mapped_iterator<ValueIteratorT, Type (*)(Value)> {
  static Type unwrap(Value value) { return value.getType(); }

public:
  using reference = Type;

  /// Provide a const dereference method.
  Type operator*() const { return unwrap(*this->I); }

  /// Initializes the type iterator to the specified value iterator.
  ValueTypeIterator(ValueIteratorT it)
      : llvm::mapped_iterator<ValueIteratorT, Type (*)(Value)>(it, &unwrap) {}
};

/// This class implements iteration on the types of a given range of values.
template <typename ValueRangeT>
class ValueTypeRange final
    : public llvm::iterator_range<
          ValueTypeIterator<typename ValueRangeT::iterator>> {
public:
  using llvm::iterator_range<
      ValueTypeIterator<typename ValueRangeT::iterator>>::iterator_range;
  template <typename Container>
  ValueTypeRange(Container &&c) : ValueTypeRange(c.begin(), c.end()) {}
};

template <typename RangeT>
inline bool operator==(ArrayRef<Type> lhs, const ValueTypeRange<RangeT> &rhs) {
  return lhs.size() == static_cast<size_t>(llvm::size(rhs)) &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

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
