//===- DataLayoutInterfaces.h - Data Layout Interface Decls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces for the data layout specification, operations to which
// they can be attached, types subject to data layout and dialects containing
// data layout entries.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_DATALAYOUTINTERFACES_H
#define MLIR_INTERFACES_DATALAYOUTINTERFACES_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class DataLayout;
class DataLayoutEntryInterface;
using DataLayoutEntryKey = llvm::PointerUnion<Type, Identifier>;
// Using explicit SmallVector size because we cannot infer the size from the
// forward declaration, and we need the typedef in the actual declaration.
using DataLayoutEntryList = llvm::SmallVector<DataLayoutEntryInterface, 4>;
using DataLayoutEntryListRef = llvm::ArrayRef<DataLayoutEntryInterface>;
class DataLayoutOpInterface;
class DataLayoutSpecInterface;
class ModuleOp;

namespace detail {
/// Default handler for the type size request. Computes results for built-in
/// types and dispatches to the DataLayoutTypeInterface for other types.
unsigned getDefaultTypeSize(Type type, const DataLayout &dataLayout,
                            DataLayoutEntryListRef params);

/// Default handler for the type size in bits request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
unsigned getDefaultTypeSizeInBits(Type type, const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params);

/// Default handler for the required alignemnt request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
unsigned getDefaultABIAlignment(Type type, const DataLayout &dataLayout,
                                ArrayRef<DataLayoutEntryInterface> params);

/// Default handler for the preferred alignemnt request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
unsigned
getDefaultPreferredAlignment(Type type, const DataLayout &dataLayout,
                             ArrayRef<DataLayoutEntryInterface> params);

/// Given a list of data layout entries, returns a new list containing the
/// entries with keys having the given type ID, i.e. belonging to the same type
/// class.
DataLayoutEntryList filterEntriesForType(DataLayoutEntryListRef entries,
                                         TypeID typeID);

/// Given a list of data layout entries, returns the entry that has the given
/// identifier as key, if such an entry exists in the list.
DataLayoutEntryInterface
filterEntryForIdentifier(DataLayoutEntryListRef entries, Identifier id);

/// Verifies that the operation implementing the data layout interface, or a
/// module operation, is valid. This calls the verifier of the spec attribute
/// and checks if the layout is compatible with specs attached to the enclosing
/// operations.
LogicalResult verifyDataLayoutOp(Operation *op);

/// Verifies that a data layout spec is valid. This dispatches to individual
/// entry verifiers, and then to the verifiers implemented by the relevant type
/// and dialect interfaces for type and identifier keys respectively.
LogicalResult verifyDataLayoutSpec(DataLayoutSpecInterface spec, Location loc);
} // namespace detail
} // namespace mlir

#include "mlir/Interfaces/DataLayoutAttrInterface.h.inc"
#include "mlir/Interfaces/DataLayoutOpInterface.h.inc"
#include "mlir/Interfaces/DataLayoutTypeInterface.h.inc"

namespace mlir {

//===----------------------------------------------------------------------===//
// DataLayoutDialectInterface
//===----------------------------------------------------------------------===//

/// An interface to be implemented by dialects that can have identifiers in the
/// data layout specification entries. Provides hooks for verifying the entry
/// validity and combining two entries.
class DataLayoutDialectInterface
    : public DialectInterface::Base<DataLayoutDialectInterface> {
public:
  DataLayoutDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Checks whether the given data layout entry is valid and reports any errors
  /// at the provided location. Derived classes should override this.
  virtual LogicalResult verifyEntry(DataLayoutEntryInterface entry,
                                    Location loc) const {
    return success();
  }

  /// Default implementation of entry combination that combines identical
  /// entries and returns null otherwise.
  static DataLayoutEntryInterface
  defaultCombine(DataLayoutEntryInterface outer,
                 DataLayoutEntryInterface inner) {
    if (!outer || outer == inner)
      return inner;
    return {};
  }

  /// Combines two entries with identifiers that belong to this dialect. Returns
  /// the combined entry or null if the entries are not compatible. Derived
  /// classes likely need to reimplement this.
  virtual DataLayoutEntryInterface
  combine(DataLayoutEntryInterface outer,
          DataLayoutEntryInterface inner) const {
    return defaultCombine(outer, inner);
  }
};

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

/// The main mechanism for performing data layout queries. Instances of this
/// class can be created for an operation implementing DataLayoutOpInterface.
/// Upon construction, a layout spec combining that of the given operation with
/// all its ancestors will be computed and used to handle further requests. For
/// efficiency, results to all requests will be cached in this object.
/// Therefore, if the data layout spec for the scoping operation, or any of the
/// enclosing operations, changes, the cache is no longer valid. The user is
/// responsible creating a new DataLayout object after any spec change. In debug
/// mode, the cache validity is being checked in every request.
class DataLayout {
public:
  explicit DataLayout(DataLayoutOpInterface op);
  explicit DataLayout(ModuleOp op);

  /// Returns the size of the given type in the current scope.
  unsigned getTypeSize(Type t) const;

  /// Returns the size in bits of the given type in the current scope.
  unsigned getTypeSizeInBits(Type t) const;

  /// Returns the required alignment of the given type in the current scope.
  unsigned getTypeABIAlignment(Type t) const;

  /// Returns the preferred of the given type in the current scope.
  unsigned getTypePreferredAlignment(Type t) const;

private:
  /// Combined layout spec at the given scope.
  const DataLayoutSpecInterface originalLayout;

#ifndef NDEBUG
  /// List of enclosing layout specs.
  SmallVector<DataLayoutSpecInterface, 2> layoutStack;
#endif

  /// Asserts that the cache is still valid. Expensive in debug mode. No-op in
  /// release mode.
  void checkValid() const;

  /// Operation defining the scope of requests.
  // TODO: this is mutable because the generated interface method are not const.
  // Update the generator to support const methods and change this to const.
  mutable Operation *scope;

  /// Caches for individual requests.
  mutable DenseMap<Type, unsigned> sizes;
  mutable DenseMap<Type, unsigned> bitsizes;
  mutable DenseMap<Type, unsigned> abiAlignments;
  mutable DenseMap<Type, unsigned> preferredAlignments;
};

} // namespace mlir

#endif // MLIR_INTERFACES_DATALAYOUTINTERFACES_H
