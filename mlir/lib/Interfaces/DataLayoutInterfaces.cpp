//===- DataLayoutInterfaces.cpp - Data Layout Interface Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Default implementations
//===----------------------------------------------------------------------===//

/// Reports that the given type is missing the data layout information and
/// exits.
static LLVM_ATTRIBUTE_NORETURN void reportMissingDataLayout(Type type) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "neither the scoping op nor the type class provide data layout "
        "information for "
     << type;
  llvm::report_fatal_error(os.str());
}

unsigned
mlir::detail::getDefaultTypeSize(Type type, const DataLayout &dataLayout,
                                 ArrayRef<DataLayoutEntryInterface> params) {
  if (type.isa<IntegerType, FloatType>())
    return llvm::divideCeil(type.getIntOrFloatBitWidth(), 8);

  // Sizes of vector types are rounded up to those of types with closest
  // power-of-two number of elements.
  // TODO: make this extensible.
  if (auto vecType = type.dyn_cast<VectorType>())
    return llvm::PowerOf2Ceil(vecType.getNumElements()) *
           dataLayout.getTypeSize(vecType.getElementType());

  if (auto typeInterface = type.dyn_cast<DataLayoutTypeInterface>())
    return typeInterface.getTypeSize(dataLayout, params);

  reportMissingDataLayout(type);
}

unsigned mlir::detail::getDefaultABIAlignment(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  // Natural alignment is the closest power-of-two number above.
  if (type.isa<FloatType, VectorType>())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(type));

  if (auto intType = type.dyn_cast<IntegerType>()) {
    return intType.getWidth() < 64
               ? llvm::PowerOf2Ceil(llvm::divideCeil(intType.getWidth(), 8))
               : 4;
  }

  if (auto typeInterface = type.dyn_cast<DataLayoutTypeInterface>())
    return typeInterface.getABIAlignment(dataLayout, params);

  reportMissingDataLayout(type);
}

unsigned mlir::detail::getDefaultPreferredAlignment(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  // Preferred alignment is same as natural for floats and vectors.
  if (type.isa<FloatType, VectorType>())
    return dataLayout.getTypeABIAlignment(type);

  // Preferred alignment is the cloest power-of-two number above for integers
  // (ABI alignment may be smaller).
  if (auto intType = type.dyn_cast<IntegerType>())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(type));

  if (auto typeInterface = type.dyn_cast<DataLayoutTypeInterface>())
    return typeInterface.getPreferredAlignment(dataLayout, params);

  reportMissingDataLayout(type);
}

DataLayoutEntryList
mlir::detail::filterEntriesForType(DataLayoutEntryListRef entries,
                                   TypeID typeID) {
  return llvm::to_vector<4>(llvm::make_filter_range(
      entries, [typeID](DataLayoutEntryInterface entry) {
        auto type = entry.getKey().dyn_cast<Type>();
        return type && type.getTypeID() == typeID;
      }));
}

DataLayoutEntryInterface
mlir::detail::filterEntryForIdentifier(DataLayoutEntryListRef entries,
                                       Identifier id) {
  const auto *it = llvm::find_if(entries, [id](DataLayoutEntryInterface entry) {
    if (!entry.getKey().is<Identifier>())
      return false;
    return entry.getKey().get<Identifier>() == id;
  });
  return it == entries.end() ? DataLayoutEntryInterface() : *it;
}

/// Populates `opsWithLayout` with the list of proper ancestors of `leaf` that
/// implement the `DataLayoutOpInterface`.
static void findProperAscendantsWithLayout(
    Operation *leaf, SmallVectorImpl<DataLayoutOpInterface> &opsWithLayout) {
  if (!leaf)
    return;

  while (auto opLayout = leaf->getParentOfType<DataLayoutOpInterface>()) {
    opsWithLayout.push_back(opLayout);
    leaf = opLayout;
  }
}

/// Returns a layout spec that is a combination of the layout specs attached
/// to the given operation and all its ancestors.
static DataLayoutSpecInterface
getCombinedDataLayout(DataLayoutOpInterface leaf) {
  if (!leaf)
    return {};

  SmallVector<DataLayoutOpInterface> opsWithLayout;
  findProperAscendantsWithLayout(leaf, opsWithLayout);

  // Fast track if there are no ancestors.
  if (opsWithLayout.empty())
    return leaf.getDataLayoutSpec();

  // Create the list of non-null specs (null/missing specs can be safely
  // ignored) from the outermost to the innermost.
  SmallVector<DataLayoutSpecInterface> specs;
  specs.reserve(opsWithLayout.size());
  for (DataLayoutOpInterface op : llvm::reverse(opsWithLayout))
    if (DataLayoutSpecInterface current = op.getDataLayoutSpec())
      specs.push_back(current);

  // Combine the specs using the innermost as anchor.
  if (DataLayoutSpecInterface current = leaf.getDataLayoutSpec())
    return current.combineWith(specs);
  if (specs.empty())
    return {};
  return specs.back().combineWith(llvm::makeArrayRef(specs).drop_back());
}

LogicalResult mlir::detail::verifyDataLayoutOp(DataLayoutOpInterface op) {
  DataLayoutSpecInterface spec = op.getDataLayoutSpec();
  // The layout specification may be missing and it's fine.
  if (!spec)
    return success();

  if (failed(spec.verifySpec(op.getLoc())))
    return failure();
  if (!getCombinedDataLayout(op)) {
    InFlightDiagnostic diag =
        op.emitError()
        << "data layout is not a refinement of the layouts in enclosing ops";
    SmallVector<DataLayoutOpInterface> opsWithLayout;
    findProperAscendantsWithLayout(op, opsWithLayout);
    for (DataLayoutOpInterface parent : opsWithLayout)
      diag.attachNote(parent.getLoc()) << "enclosing op with data layout";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

mlir::DataLayout::DataLayout(DataLayoutOpInterface op)
    : originalLayout(getCombinedDataLayout(op)), scope(op) {
  if (!originalLayout) {
    assert((!op || !op.getDataLayoutSpec()) &&
           "could not compute layout information for an op (failed to "
           "combine attributes?)");
  }

#ifndef NDEBUG
  SmallVector<DataLayoutOpInterface> opsWithLayout;
  findProperAscendantsWithLayout(op, opsWithLayout);
  layoutStack = llvm::to_vector<2>(
      llvm::map_range(opsWithLayout, [](DataLayoutOpInterface iface) {
        return iface.getDataLayoutSpec();
      }));
#endif
}

void mlir::DataLayout::checkValid() const {
#ifndef NDEBUG
  SmallVector<DataLayoutOpInterface> opsWithLayout;
  findProperAscendantsWithLayout(scope, opsWithLayout);
  assert(opsWithLayout.size() == layoutStack.size() &&
         "data layout object used, but no longer valid due to the change in "
         "number of nested layouts");
  for (auto pair : llvm::zip(opsWithLayout, layoutStack)) {
    Attribute newLayout = std::get<0>(pair).getDataLayoutSpec();
    Attribute origLayout = std::get<1>(pair);
    assert(newLayout == origLayout &&
           "data layout object used, but no longer valid "
           "due to the change in layout attributes");
  }
#endif
  assert(((!scope && !this->originalLayout) ||
          (scope && this->originalLayout == getCombinedDataLayout(scope))) &&
         "data layout object used, but no longer valid due to the change in "
         "layout spec");
}

/// Looks up the value for the given type key in the given cache. If there is no
/// such value in the cache, compute it using the given callback and put it in
/// the cache before returning.
static unsigned cachedLookup(Type t, DenseMap<Type, unsigned> &cache,
                             function_ref<unsigned(Type)> compute) {
  auto it = cache.find(t);
  if (it != cache.end())
    return it->second;

  auto result = cache.try_emplace(t, compute(t));
  return result.first->second;
}

unsigned mlir::DataLayout::getTypeSize(Type t) const {
  checkValid();
  return cachedLookup(t, sizes, [&](Type ty) {
    return (scope && originalLayout)
               ? scope.getTypeSize(
                     ty, *this, originalLayout.getSpecForType(ty.getTypeID()))
               : detail::getDefaultTypeSize(ty, *this, {});
  });
}

unsigned mlir::DataLayout::getTypeABIAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, abiAlignments, [&](Type ty) {
    return (scope && originalLayout)
               ? scope.getTypeABIAlignment(
                     ty, *this, originalLayout.getSpecForType(ty.getTypeID()))
               : detail::getDefaultABIAlignment(ty, *this, {});
  });
}

unsigned mlir::DataLayout::getTypePreferredAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, preferredAlignments, [&](Type ty) {
    return (scope && originalLayout)
               ? scope.getTypePreferredAlignment(
                     ty, *this, originalLayout.getSpecForType(ty.getTypeID()))
               : detail::getDefaultPreferredAlignment(ty, *this, {});
  });
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecInterface
//===----------------------------------------------------------------------===//

void DataLayoutSpecInterface::bucketEntriesByType(
    DenseMap<TypeID, DataLayoutEntryList> &types,
    DenseMap<Identifier, DataLayoutEntryInterface> &ids) {
  for (DataLayoutEntryInterface entry : getEntries()) {
    if (auto type = entry.getKey().dyn_cast<Type>())
      types[type.getTypeID()].push_back(entry);
    else
      ids[entry.getKey().get<Identifier>()] = entry;
  }
}

LogicalResult mlir::detail::verifyDataLayoutSpec(DataLayoutSpecInterface spec,
                                                 Location loc) {
  // First, verify individual entries.
  for (DataLayoutEntryInterface entry : spec.getEntries())
    if (failed(entry.verifyEntry(loc)))
      return failure();

  // Second, dispatch verifications of entry groups to types or dialects they
  // are are associated with.
  DenseMap<TypeID, DataLayoutEntryList> types;
  DenseMap<Identifier, DataLayoutEntryInterface> ids;
  spec.bucketEntriesByType(types, ids);

  for (const auto &kvp : types) {
    auto sampleType = kvp.second.front().getKey().get<Type>();
    if (isa<BuiltinDialect>(&sampleType.getDialect()))
      return emitError(loc) << "unexpected data layout for a built-in type";

    auto dlType = sampleType.dyn_cast<DataLayoutTypeInterface>();
    if (!dlType)
      return emitError(loc)
             << "data layout specified for a type that does not support it";
    if (failed(dlType.verifyEntries(kvp.second, loc)))
      return failure();
  }

  for (const auto &kvp : ids) {
    Identifier identifier = kvp.second.getKey().get<Identifier>();
    Dialect *dialect = identifier.getDialect();

    // Ignore attributes that belong to an unknown dialect, the dialect may
    // actually implement the relevant interface but we don't know about that.
    if (!dialect)
      continue;

    const auto *iface =
        dialect->getRegisteredInterface<DataLayoutDialectInterface>();
    if (failed(iface->verifyEntry(kvp.second, loc)))
      return failure();
  }

  return success();
}

#include "mlir/Interfaces/DataLayoutAttrInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutOpInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutTypeInterface.cpp.inc"
