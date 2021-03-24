//===- DataLayoutInterfaces.cpp - Data Layout Interface Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/TypeSwitch.h"

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
  unsigned bits = getDefaultTypeSizeInBits(type, dataLayout, params);
  return llvm::divideCeil(bits, 8);
}

unsigned mlir::detail::getDefaultTypeSizeInBits(Type type,
                                                const DataLayout &dataLayout,
                                                DataLayoutEntryListRef params) {
  if (type.isa<IntegerType, FloatType>())
    return type.getIntOrFloatBitWidth();

  // Sizes of vector types are rounded up to those of types with closest
  // power-of-two number of elements in the innermost dimension. We also assume
  // there is no bit-packing at the moment element sizes are taken in bytes and
  // multiplied with 8 bits.
  // TODO: make this extensible.
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getNumElements() / vecType.getShape().back() *
           llvm::PowerOf2Ceil(vecType.getShape().back()) *
           dataLayout.getTypeSize(vecType.getElementType()) * 8;

  if (auto typeInterface = type.dyn_cast<DataLayoutTypeInterface>())
    return typeInterface.getTypeSizeInBits(dataLayout, params);

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

static DataLayoutSpecInterface getSpec(Operation *operation) {
  return llvm::TypeSwitch<Operation *, DataLayoutSpecInterface>(operation)
      .Case<ModuleOp, DataLayoutOpInterface>(
          [&](auto op) { return op.getDataLayoutSpec(); })
      .Default([](Operation *) {
        llvm_unreachable("expected an op with data layout spec");
        return DataLayoutSpecInterface();
      });
}

/// Populates `opsWithLayout` with the list of proper ancestors of `leaf` that
/// are either modules or implement the `DataLayoutOpInterface`.
static void
collectParentLayouts(Operation *leaf,
                     SmallVectorImpl<DataLayoutSpecInterface> &specs,
                     SmallVectorImpl<Location> *opLocations = nullptr) {
  if (!leaf)
    return;

  for (Operation *parent = leaf->getParentOp(); parent != nullptr;
       parent = parent->getParentOp()) {
    llvm::TypeSwitch<Operation *>(parent)
        .Case<ModuleOp>([&](ModuleOp op) {
          // Skip top-level module op unless it has a layout. Top-level module
          // without layout is most likely the one implicitly added by the
          // parser and it doesn't have location. Top-level null specification
          // would have had the same effect as not having a specification at all
          // (using type defaults).
          if (!op->getParentOp() && !op.getDataLayoutSpec())
            return;
          specs.push_back(op.getDataLayoutSpec());
          if (opLocations)
            opLocations->push_back(op.getLoc());
        })
        .Case<DataLayoutOpInterface>([&](DataLayoutOpInterface op) {
          specs.push_back(op.getDataLayoutSpec());
          if (opLocations)
            opLocations->push_back(op.getLoc());
        });
  }
}

/// Returns a layout spec that is a combination of the layout specs attached
/// to the given operation and all its ancestors.
static DataLayoutSpecInterface getCombinedDataLayout(Operation *leaf) {
  if (!leaf)
    return {};

  assert((isa<ModuleOp, DataLayoutOpInterface>(leaf)) &&
         "expected an op with data layout spec");

  SmallVector<DataLayoutOpInterface> opsWithLayout;
  SmallVector<DataLayoutSpecInterface> specs;
  collectParentLayouts(leaf, specs);

  // Fast track if there are no ancestors.
  if (specs.empty())
    return getSpec(leaf);

  // Create the list of non-null specs (null/missing specs can be safely
  // ignored) from the outermost to the innermost.
  auto nonNullSpecs = llvm::to_vector<2>(llvm::make_filter_range(
      llvm::reverse(specs),
      [](DataLayoutSpecInterface iface) { return iface != nullptr; }));

  // Combine the specs using the innermost as anchor.
  if (DataLayoutSpecInterface current = getSpec(leaf))
    return current.combineWith(nonNullSpecs);
  if (nonNullSpecs.empty())
    return {};
  return nonNullSpecs.back().combineWith(
      llvm::makeArrayRef(nonNullSpecs).drop_back());
}

LogicalResult mlir::detail::verifyDataLayoutOp(Operation *op) {
  DataLayoutSpecInterface spec = getSpec(op);
  // The layout specification may be missing and it's fine.
  if (!spec)
    return success();

  if (failed(spec.verifySpec(op->getLoc())))
    return failure();
  if (!getCombinedDataLayout(op)) {
    InFlightDiagnostic diag =
        op->emitError()
        << "data layout does not combine with layouts of enclosing ops";
    SmallVector<DataLayoutSpecInterface> specs;
    SmallVector<Location> opLocations;
    collectParentLayouts(op, specs, &opLocations);
    for (Location loc : opLocations)
      diag.attachNote(loc) << "enclosing op with data layout";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

template <typename OpTy>
void checkMissingLayout(DataLayoutSpecInterface originalLayout, OpTy op) {
  if (!originalLayout) {
    assert((!op || !op.getDataLayoutSpec()) &&
           "could not compute layout information for an op (failed to "
           "combine attributes?)");
  }
}

mlir::DataLayout::DataLayout(DataLayoutOpInterface op)
    : originalLayout(getCombinedDataLayout(op)), scope(op) {
#ifndef NDEBUG
  checkMissingLayout(originalLayout, op);
  collectParentLayouts(op, layoutStack);
#endif
}

mlir::DataLayout::DataLayout(ModuleOp op)
    : originalLayout(getCombinedDataLayout(op)), scope(op) {
#ifndef NDEBUG
  checkMissingLayout(originalLayout, op);
  collectParentLayouts(op, layoutStack);
#endif
}

void mlir::DataLayout::checkValid() const {
#ifndef NDEBUG
  SmallVector<DataLayoutSpecInterface> specs;
  collectParentLayouts(scope, specs);
  assert(specs.size() == layoutStack.size() &&
         "data layout object used, but no longer valid due to the change in "
         "number of nested layouts");
  for (auto pair : llvm::zip(specs, layoutStack)) {
    Attribute newLayout = std::get<0>(pair);
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
    if (originalLayout) {
      DataLayoutEntryList list = originalLayout.getSpecForType(ty.getTypeID());
      if (auto iface = dyn_cast<DataLayoutOpInterface>(scope))
        return iface.getTypeSize(ty, *this, list);
      return detail::getDefaultTypeSize(ty, *this, list);
    }
    return detail::getDefaultTypeSize(ty, *this, {});
  });
}

unsigned mlir::DataLayout::getTypeSizeInBits(Type t) const {
  checkValid();
  return cachedLookup(t, bitsizes, [&](Type ty) {
    if (originalLayout) {
      DataLayoutEntryList list = originalLayout.getSpecForType(ty.getTypeID());
      if (auto iface = dyn_cast<DataLayoutOpInterface>(scope))
        return iface.getTypeSizeInBits(ty, *this, list);
      return detail::getDefaultTypeSizeInBits(ty, *this, list);
    }
    return detail::getDefaultTypeSizeInBits(ty, *this, {});
  });
}

unsigned mlir::DataLayout::getTypeABIAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, abiAlignments, [&](Type ty) {
    if (originalLayout) {
      DataLayoutEntryList list = originalLayout.getSpecForType(ty.getTypeID());
      if (auto iface = dyn_cast<DataLayoutOpInterface>(scope))
        return iface.getTypeABIAlignment(ty, *this, list);
      return detail::getDefaultABIAlignment(ty, *this, list);
    }
    return detail::getDefaultABIAlignment(ty, *this, {});
  });
}

unsigned mlir::DataLayout::getTypePreferredAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, preferredAlignments, [&](Type ty) {
    if (originalLayout) {
      DataLayoutEntryList list = originalLayout.getSpecForType(ty.getTypeID());
      if (auto iface = dyn_cast<DataLayoutOpInterface>(scope))
        return iface.getTypePreferredAlignment(ty, *this, list);
      return detail::getDefaultPreferredAlignment(ty, *this, list);
    }
    return detail::getDefaultPreferredAlignment(ty, *this, {});
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
