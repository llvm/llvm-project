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
#include "llvm/Support/MathExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Default implementations
//===----------------------------------------------------------------------===//

/// Reports that the given type is missing the data layout information and
/// exits.
[[noreturn]] static void reportMissingDataLayout(Type type) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "neither the scoping op nor the type class provide data layout "
        "information for "
     << type;
  llvm::report_fatal_error(Twine(os.str()));
}

/// Returns the bitwidth of the index type if specified in the param list.
/// Assumes 64-bit index otherwise.
static unsigned getIndexBitwidth(DataLayoutEntryListRef params) {
  if (params.empty())
    return 64;
  auto attr = cast<IntegerAttr>(params.front().getValue());
  return attr.getValue().getZExtValue();
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
  if (isa<IntegerType, FloatType>(type))
    return type.getIntOrFloatBitWidth();

  if (auto ctype = dyn_cast<ComplexType>(type)) {
    auto et = ctype.getElementType();
    auto innerAlignment =
        getDefaultPreferredAlignment(et, dataLayout, params) * 8;
    auto innerSize = getDefaultTypeSizeInBits(et, dataLayout, params);

    // Include padding required to align the imaginary value in the complex
    // type.
    return llvm::alignTo(innerSize, innerAlignment) + innerSize;
  }

  // Index is an integer of some bitwidth.
  if (isa<IndexType>(type))
    return dataLayout.getTypeSizeInBits(
        IntegerType::get(type.getContext(), getIndexBitwidth(params)));

  // Sizes of vector types are rounded up to those of types with closest
  // power-of-two number of elements in the innermost dimension. We also assume
  // there is no bit-packing at the moment element sizes are taken in bytes and
  // multiplied with 8 bits.
  // TODO: make this extensible.
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getNumElements() / vecType.getShape().back() *
           llvm::PowerOf2Ceil(vecType.getShape().back()) *
           dataLayout.getTypeSize(vecType.getElementType()) * 8;

  if (auto typeInterface = dyn_cast<DataLayoutTypeInterface>(type))
    return typeInterface.getTypeSizeInBits(dataLayout, params);

  reportMissingDataLayout(type);
}

static DataLayoutEntryInterface
findEntryForIntegerType(IntegerType intType,
                        ArrayRef<DataLayoutEntryInterface> params) {
  assert(!params.empty() && "expected non-empty parameter list");
  std::map<unsigned, DataLayoutEntryInterface> sortedParams;
  for (DataLayoutEntryInterface entry : params) {
    sortedParams.insert(std::make_pair(
        entry.getKey().get<Type>().getIntOrFloatBitWidth(), entry));
  }
  auto iter = sortedParams.lower_bound(intType.getWidth());
  if (iter == sortedParams.end())
    iter = std::prev(iter);

  return iter->second;
}

static unsigned extractABIAlignment(DataLayoutEntryInterface entry) {
  auto values =
      cast<DenseIntElementsAttr>(entry.getValue()).getValues<int32_t>();
  return *values.begin() / 8u;
}

static unsigned
getIntegerTypeABIAlignment(IntegerType intType,
                           ArrayRef<DataLayoutEntryInterface> params) {
  if (params.empty()) {
    return intType.getWidth() < 64
               ? llvm::PowerOf2Ceil(llvm::divideCeil(intType.getWidth(), 8))
               : 4;
  }

  return extractABIAlignment(findEntryForIntegerType(intType, params));
}

static unsigned
getFloatTypeABIAlignment(FloatType fltType, const DataLayout &dataLayout,
                         ArrayRef<DataLayoutEntryInterface> params) {
  assert(params.size() <= 1 && "at most one data layout entry is expected for "
                               "the singleton floating-point type");
  if (params.empty())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(fltType));
  return extractABIAlignment(params[0]);
}

unsigned mlir::detail::getDefaultABIAlignment(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  // Natural alignment is the closest power-of-two number above.
  if (isa<VectorType>(type))
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(type));

  if (auto fltType = dyn_cast<FloatType>(type))
    return getFloatTypeABIAlignment(fltType, dataLayout, params);

  // Index is an integer of some bitwidth.
  if (isa<IndexType>(type))
    return dataLayout.getTypeABIAlignment(
        IntegerType::get(type.getContext(), getIndexBitwidth(params)));

  if (auto intType = dyn_cast<IntegerType>(type))
    return getIntegerTypeABIAlignment(intType, params);

  if (auto ctype = dyn_cast<ComplexType>(type))
    return getDefaultABIAlignment(ctype.getElementType(), dataLayout, params);

  if (auto typeInterface = dyn_cast<DataLayoutTypeInterface>(type))
    return typeInterface.getABIAlignment(dataLayout, params);

  reportMissingDataLayout(type);
}

static unsigned extractPreferredAlignment(DataLayoutEntryInterface entry) {
  auto values =
      cast<DenseIntElementsAttr>(entry.getValue()).getValues<int32_t>();
  return *std::next(values.begin(), values.size() - 1) / 8u;
}

static unsigned
getIntegerTypePreferredAlignment(IntegerType intType,
                                 const DataLayout &dataLayout,
                                 ArrayRef<DataLayoutEntryInterface> params) {
  if (params.empty())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(intType));

  return extractPreferredAlignment(findEntryForIntegerType(intType, params));
}

static unsigned
getFloatTypePreferredAlignment(FloatType fltType, const DataLayout &dataLayout,
                               ArrayRef<DataLayoutEntryInterface> params) {
  assert(params.size() <= 1 && "at most one data layout entry is expected for "
                               "the singleton floating-point type");
  if (params.empty())
    return dataLayout.getTypeABIAlignment(fltType);
  return extractPreferredAlignment(params[0]);
}

unsigned mlir::detail::getDefaultPreferredAlignment(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  // Preferred alignment is same as natural for floats and vectors.
  if (isa<VectorType>(type))
    return dataLayout.getTypeABIAlignment(type);

  if (auto fltType = dyn_cast<FloatType>(type))
    return getFloatTypePreferredAlignment(fltType, dataLayout, params);

  // Preferred alignment is the closest power-of-two number above for integers
  // (ABI alignment may be smaller).
  if (auto intType = dyn_cast<IntegerType>(type))
    return getIntegerTypePreferredAlignment(intType, dataLayout, params);

  if (isa<IndexType>(type)) {
    return dataLayout.getTypePreferredAlignment(
        IntegerType::get(type.getContext(), getIndexBitwidth(params)));
  }

  if (auto ctype = dyn_cast<ComplexType>(type))
    return getDefaultPreferredAlignment(ctype.getElementType(), dataLayout,
                                        params);

  if (auto typeInterface = dyn_cast<DataLayoutTypeInterface>(type))
    return typeInterface.getPreferredAlignment(dataLayout, params);

  reportMissingDataLayout(type);
}

// Returns the memory space used for allocal operations if specified in the
// given entry. If the entry is empty the default memory space represented by
// an empty attribute is returned.
Attribute
mlir::detail::getDefaultAllocaMemorySpace(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface()) {
    return Attribute();
  }

  return entry.getValue();
}

// Returns the stack alignment if specified in the given entry. If the entry is
// empty the default alignment zero is returned.
unsigned
mlir::detail::getDefaultStackAlignment(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface())
    return 0;

  auto value = cast<IntegerAttr>(entry.getValue());
  return value.getValue().getZExtValue();
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
                                       StringAttr id) {
  const auto *it = llvm::find_if(entries, [id](DataLayoutEntryInterface entry) {
    if (!entry.getKey().is<StringAttr>())
      return false;
    return entry.getKey().get<StringAttr>() == id;
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
      llvm::ArrayRef(nonNullSpecs).drop_back());
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

mlir::DataLayout::DataLayout() : DataLayout(ModuleOp()) {}

mlir::DataLayout::DataLayout(DataLayoutOpInterface op)
    : originalLayout(getCombinedDataLayout(op)), scope(op),
      allocaMemorySpace(std::nullopt), stackAlignment(std::nullopt) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  checkMissingLayout(originalLayout, op);
  collectParentLayouts(op, layoutStack);
#endif
}

mlir::DataLayout::DataLayout(ModuleOp op)
    : originalLayout(getCombinedDataLayout(op)), scope(op),
      allocaMemorySpace(std::nullopt), stackAlignment(std::nullopt) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  checkMissingLayout(originalLayout, op);
  collectParentLayouts(op, layoutStack);
#endif
}

mlir::DataLayout mlir::DataLayout::closest(Operation *op) {
  // Search the closest parent either being a module operation or implementing
  // the data layout interface.
  while (op) {
    if (auto module = dyn_cast<ModuleOp>(op))
      return DataLayout(module);
    if (auto iface = dyn_cast<DataLayoutOpInterface>(op))
      return DataLayout(iface);
    op = op->getParentOp();
  }
  return DataLayout();
}

void mlir::DataLayout::checkValid() const {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
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
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeSize(ty, *this, list);
    return detail::getDefaultTypeSize(ty, *this, list);
  });
}

unsigned mlir::DataLayout::getTypeSizeInBits(Type t) const {
  checkValid();
  return cachedLookup(t, bitsizes, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeSizeInBits(ty, *this, list);
    return detail::getDefaultTypeSizeInBits(ty, *this, list);
  });
}

unsigned mlir::DataLayout::getTypeABIAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, abiAlignments, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeABIAlignment(ty, *this, list);
    return detail::getDefaultABIAlignment(ty, *this, list);
  });
}

unsigned mlir::DataLayout::getTypePreferredAlignment(Type t) const {
  checkValid();
  return cachedLookup(t, preferredAlignments, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypePreferredAlignment(ty, *this, list);
    return detail::getDefaultPreferredAlignment(ty, *this, list);
  });
}

mlir::Attribute mlir::DataLayout::getAllocaMemorySpace() const {
  checkValid();
  if (allocaMemorySpace)
    return *allocaMemorySpace;
  DataLayoutEntryInterface entry;
  if (originalLayout)
    entry = originalLayout.getSpecForIdentifier(
        originalLayout.getAllocaMemorySpaceIdentifier(
            originalLayout.getContext()));
  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    allocaMemorySpace = iface.getAllocaMemorySpace(entry);
  else
    allocaMemorySpace = detail::getDefaultAllocaMemorySpace(entry);
  return *allocaMemorySpace;
}

unsigned mlir::DataLayout::getStackAlignment() const {
  checkValid();
  if (stackAlignment)
    return *stackAlignment;
  DataLayoutEntryInterface entry;
  if (originalLayout)
    entry = originalLayout.getSpecForIdentifier(
        originalLayout.getStackAlignmentIdentifier(
            originalLayout.getContext()));
  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    stackAlignment = iface.getStackAlignment(entry);
  else
    stackAlignment = detail::getDefaultStackAlignment(entry);
  return *stackAlignment;
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecInterface
//===----------------------------------------------------------------------===//

void DataLayoutSpecInterface::bucketEntriesByType(
    DenseMap<TypeID, DataLayoutEntryList> &types,
    DenseMap<StringAttr, DataLayoutEntryInterface> &ids) {
  for (DataLayoutEntryInterface entry : getEntries()) {
    if (auto type = entry.getKey().dyn_cast<Type>())
      types[type.getTypeID()].push_back(entry);
    else
      ids[entry.getKey().get<StringAttr>()] = entry;
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
  DenseMap<StringAttr, DataLayoutEntryInterface> ids;
  spec.bucketEntriesByType(types, ids);

  for (const auto &kvp : types) {
    auto sampleType = kvp.second.front().getKey().get<Type>();
    if (isa<IndexType>(sampleType)) {
      assert(kvp.second.size() == 1 &&
             "expected one data layout entry for non-parametric 'index' type");
      if (!isa<IntegerAttr>(kvp.second.front().getValue()))
        return emitError(loc)
               << "expected integer attribute in the data layout entry for "
               << sampleType;
      continue;
    }

    if (isa<IntegerType, FloatType>(sampleType)) {
      for (DataLayoutEntryInterface entry : kvp.second) {
        auto value = dyn_cast<DenseIntElementsAttr>(entry.getValue());
        if (!value || !value.getElementType().isSignlessInteger(32)) {
          emitError(loc) << "expected a dense i32 elements attribute in the "
                            "data layout entry "
                         << entry;
          return failure();
        }

        auto elements = llvm::to_vector<2>(value.getValues<int32_t>());
        unsigned numElements = elements.size();
        if (numElements < 1 || numElements > 2) {
          emitError(loc) << "expected 1 or 2 elements in the data layout entry "
                         << entry;
          return failure();
        }

        int32_t abi = elements[0];
        int32_t preferred = numElements == 2 ? elements[1] : abi;
        if (preferred < abi) {
          emitError(loc)
              << "preferred alignment is expected to be greater than or equal "
                 "to the abi alignment in data layout entry "
              << entry;
          return failure();
        }
      }
      continue;
    }

    if (isa<BuiltinDialect>(&sampleType.getDialect()))
      return emitError(loc) << "unexpected data layout for a built-in type";

    auto dlType = dyn_cast<DataLayoutTypeInterface>(sampleType);
    if (!dlType)
      return emitError(loc)
             << "data layout specified for a type that does not support it";
    if (failed(dlType.verifyEntries(kvp.second, loc)))
      return failure();
  }

  for (const auto &kvp : ids) {
    StringAttr identifier = kvp.second.getKey().get<StringAttr>();
    Dialect *dialect = identifier.getReferencedDialect();

    // Ignore attributes that belong to an unknown dialect, the dialect may
    // actually implement the relevant interface but we don't know about that.
    if (!dialect)
      continue;

    const auto *iface = dyn_cast<DataLayoutDialectInterface>(dialect);
    if (!iface) {
      return emitError(loc)
             << "the '" << dialect->getNamespace()
             << "' dialect does not support identifier data layout entries";
    }
    if (failed(iface->verifyEntry(kvp.second, loc)))
      return failure();
  }

  return success();
}

#include "mlir/Interfaces/DataLayoutAttrInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutOpInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutTypeInterface.cpp.inc"
