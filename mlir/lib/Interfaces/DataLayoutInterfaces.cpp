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
  llvm::report_fatal_error(Twine(message));
}

/// Returns the bitwidth of the index type if specified in the param list.
/// Assumes 64-bit index otherwise.
static uint64_t getIndexBitwidth(DataLayoutEntryListRef params) {
  if (params.empty())
    return 64;
  auto attr = cast<IntegerAttr>(params.front().getValue());
  return attr.getValue().getZExtValue();
}

llvm::TypeSize
mlir::detail::getDefaultTypeSize(Type type, const DataLayout &dataLayout,
                                 ArrayRef<DataLayoutEntryInterface> params) {
  llvm::TypeSize bits = getDefaultTypeSizeInBits(type, dataLayout, params);
  return divideCeil(bits, 8);
}

llvm::TypeSize
mlir::detail::getDefaultTypeSizeInBits(Type type, const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) {
  if (isa<IntegerType, FloatType>(type))
    return llvm::TypeSize::getFixed(type.getIntOrFloatBitWidth());

  if (auto ctype = dyn_cast<ComplexType>(type)) {
    Type et = ctype.getElementType();
    uint64_t innerAlignment =
        getDefaultPreferredAlignment(et, dataLayout, params) * 8;
    llvm::TypeSize innerSize = getDefaultTypeSizeInBits(et, dataLayout, params);

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
  if (auto vecType = dyn_cast<VectorType>(type)) {
    uint64_t baseSize = vecType.getNumElements() / vecType.getShape().back() *
                        llvm::PowerOf2Ceil(vecType.getShape().back()) *
                        dataLayout.getTypeSize(vecType.getElementType()) * 8;
    return llvm::TypeSize::get(baseSize, vecType.isScalable());
  }

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

constexpr const static uint64_t kDefaultBitsInByte = 8u;

static uint64_t extractABIAlignment(DataLayoutEntryInterface entry) {
  auto values =
      cast<DenseIntElementsAttr>(entry.getValue()).getValues<uint64_t>();
  return static_cast<uint64_t>(*values.begin()) / kDefaultBitsInByte;
}

static uint64_t
getIntegerTypeABIAlignment(IntegerType intType,
                           ArrayRef<DataLayoutEntryInterface> params) {
  constexpr uint64_t kDefaultSmallIntAlignment = 4u;
  constexpr unsigned kSmallIntSize = 64;
  if (params.empty()) {
    return intType.getWidth() < kSmallIntSize
               ? llvm::PowerOf2Ceil(
                     llvm::divideCeil(intType.getWidth(), kDefaultBitsInByte))
               : kDefaultSmallIntAlignment;
  }

  return extractABIAlignment(findEntryForIntegerType(intType, params));
}

static uint64_t
getFloatTypeABIAlignment(FloatType fltType, const DataLayout &dataLayout,
                         ArrayRef<DataLayoutEntryInterface> params) {
  assert(params.size() <= 1 && "at most one data layout entry is expected for "
                               "the singleton floating-point type");
  if (params.empty())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(fltType).getFixedValue());
  return extractABIAlignment(params[0]);
}

uint64_t mlir::detail::getDefaultABIAlignment(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  // Natural alignment is the closest power-of-two number above. For scalable
  // vectors, aligning them to the same as the base vector is sufficient.
  if (isa<VectorType>(type))
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(type).getKnownMinValue());

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

static uint64_t extractPreferredAlignment(DataLayoutEntryInterface entry) {
  auto values =
      cast<DenseIntElementsAttr>(entry.getValue()).getValues<uint64_t>();
  return *std::next(values.begin(), values.size() - 1) / kDefaultBitsInByte;
}

static uint64_t
getIntegerTypePreferredAlignment(IntegerType intType,
                                 const DataLayout &dataLayout,
                                 ArrayRef<DataLayoutEntryInterface> params) {
  if (params.empty())
    return llvm::PowerOf2Ceil(dataLayout.getTypeSize(intType).getFixedValue());

  return extractPreferredAlignment(findEntryForIntegerType(intType, params));
}

static uint64_t
getFloatTypePreferredAlignment(FloatType fltType, const DataLayout &dataLayout,
                               ArrayRef<DataLayoutEntryInterface> params) {
  assert(params.size() <= 1 && "at most one data layout entry is expected for "
                               "the singleton floating-point type");
  if (params.empty())
    return dataLayout.getTypeABIAlignment(fltType);
  return extractPreferredAlignment(params[0]);
}

uint64_t mlir::detail::getDefaultPreferredAlignment(
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

std::optional<uint64_t> mlir::detail::getDefaultIndexBitwidth(
    Type type, const DataLayout &dataLayout,
    ArrayRef<DataLayoutEntryInterface> params) {
  if (isa<IndexType>(type))
    return getIndexBitwidth(params);

  if (auto typeInterface = dyn_cast<DataLayoutTypeInterface>(type))
    if (std::optional<uint64_t> indexBitwidth =
            typeInterface.getIndexBitwidth(dataLayout, params))
      return *indexBitwidth;

  // Return std::nullopt for all other types, which are assumed to be non
  // pointer-like types.
  return std::nullopt;
}

// Returns the endianness if specified in the given entry. If the entry is empty
// the default endianness represented by an empty attribute is returned.
Attribute mlir::detail::getDefaultEndianness(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface())
    return Attribute();

  return entry.getValue();
}

// Returns the memory space used for alloca operations if specified in the
// given entry. If the entry is empty the default memory space represented by
// an empty attribute is returned.
Attribute
mlir::detail::getDefaultAllocaMemorySpace(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface()) {
    return Attribute();
  }

  return entry.getValue();
}

// Returns the memory space used for the program memory space.  if
// specified in the given entry. If the entry is empty the default
// memory space represented by an empty attribute is returned.
Attribute
mlir::detail::getDefaultProgramMemorySpace(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface()) {
    return Attribute();
  }

  return entry.getValue();
}

// Returns the memory space used for global the global memory space. if
// specified in the given entry. If the entry is empty the default memory
// space represented by an empty attribute is returned.
Attribute
mlir::detail::getDefaultGlobalMemorySpace(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface()) {
    return Attribute();
  }

  return entry.getValue();
}

// Returns the stack alignment if specified in the given entry. If the entry is
// empty the default alignment zero is returned.
uint64_t
mlir::detail::getDefaultStackAlignment(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface())
    return 0;

  auto value = cast<IntegerAttr>(entry.getValue());
  return value.getValue().getZExtValue();
}

std::optional<Attribute>
mlir::detail::getDevicePropertyValue(DataLayoutEntryInterface entry) {
  if (entry == DataLayoutEntryInterface())
    return std::nullopt;

  return entry.getValue();
}

DataLayoutEntryList
mlir::detail::filterEntriesForType(DataLayoutEntryListRef entries,
                                   TypeID typeID) {
  return llvm::to_vector<4>(llvm::make_filter_range(
      entries, [typeID](DataLayoutEntryInterface entry) {
        auto type = llvm::dyn_cast_if_present<Type>(entry.getKey());
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

static TargetSystemSpecInterface getTargetSystemSpec(Operation *operation) {
  if (operation) {
    ModuleOp moduleOp = dyn_cast<ModuleOp>(operation);
    if (!moduleOp)
      moduleOp = operation->getParentOfType<ModuleOp>();
    return moduleOp.getTargetSystemSpec();
  }
  return TargetSystemSpecInterface();
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

llvm::TypeSize mlir::detail::divideCeil(llvm::TypeSize numerator,
                                        uint64_t denominator) {
  uint64_t divided =
      llvm::divideCeil(numerator.getKnownMinValue(), denominator);
  return llvm::TypeSize::get(divided, numerator.isScalable());
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
    : originalLayout(getCombinedDataLayout(op)),
      originalTargetSystemDesc(getTargetSystemSpec(op)), scope(op),
      allocaMemorySpace(std::nullopt), programMemorySpace(std::nullopt),
      globalMemorySpace(std::nullopt), stackAlignment(std::nullopt) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  checkMissingLayout(originalLayout, op);
  collectParentLayouts(op, layoutStack);
#endif
}

mlir::DataLayout::DataLayout(ModuleOp op)
    : originalLayout(getCombinedDataLayout(op)),
      originalTargetSystemDesc(getTargetSystemSpec(op)), scope(op),
      allocaMemorySpace(std::nullopt), programMemorySpace(std::nullopt),
      globalMemorySpace(std::nullopt), stackAlignment(std::nullopt) {
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
template <typename T>
static T cachedLookup(Type t, DenseMap<Type, T> &cache,
                      function_ref<T(Type)> compute) {
  auto it = cache.find(t);
  if (it != cache.end())
    return it->second;

  auto result = cache.try_emplace(t, compute(t));
  return result.first->second;
}

llvm::TypeSize mlir::DataLayout::getTypeSize(Type t) const {
  checkValid();
  return cachedLookup<llvm::TypeSize>(t, sizes, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeSize(ty, *this, list);
    return detail::getDefaultTypeSize(ty, *this, list);
  });
}

llvm::TypeSize mlir::DataLayout::getTypeSizeInBits(Type t) const {
  checkValid();
  return cachedLookup<llvm::TypeSize>(t, bitsizes, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeSizeInBits(ty, *this, list);
    return detail::getDefaultTypeSizeInBits(ty, *this, list);
  });
}

uint64_t mlir::DataLayout::getTypeABIAlignment(Type t) const {
  checkValid();
  return cachedLookup<uint64_t>(t, abiAlignments, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypeABIAlignment(ty, *this, list);
    return detail::getDefaultABIAlignment(ty, *this, list);
  });
}

uint64_t mlir::DataLayout::getTypePreferredAlignment(Type t) const {
  checkValid();
  return cachedLookup<uint64_t>(t, preferredAlignments, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getTypePreferredAlignment(ty, *this, list);
    return detail::getDefaultPreferredAlignment(ty, *this, list);
  });
}

std::optional<uint64_t> mlir::DataLayout::getTypeIndexBitwidth(Type t) const {
  checkValid();
  return cachedLookup<std::optional<uint64_t>>(t, indexBitwidths, [&](Type ty) {
    DataLayoutEntryList list;
    if (originalLayout)
      list = originalLayout.getSpecForType(ty.getTypeID());
    if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
      return iface.getIndexBitwidth(ty, *this, list);
    return detail::getDefaultIndexBitwidth(ty, *this, list);
  });
}

mlir::Attribute mlir::DataLayout::getEndianness() const {
  checkValid();
  if (endianness)
    return *endianness;
  DataLayoutEntryInterface entry;
  if (originalLayout)
    entry = originalLayout.getSpecForIdentifier(
        originalLayout.getEndiannessIdentifier(originalLayout.getContext()));

  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    endianness = iface.getEndianness(entry);
  else
    endianness = detail::getDefaultEndianness(entry);
  return *endianness;
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

mlir::Attribute mlir::DataLayout::getProgramMemorySpace() const {
  checkValid();
  if (programMemorySpace)
    return *programMemorySpace;
  DataLayoutEntryInterface entry;
  if (originalLayout)
    entry = originalLayout.getSpecForIdentifier(
        originalLayout.getProgramMemorySpaceIdentifier(
            originalLayout.getContext()));
  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    programMemorySpace = iface.getProgramMemorySpace(entry);
  else
    programMemorySpace = detail::getDefaultProgramMemorySpace(entry);
  return *programMemorySpace;
}

mlir::Attribute mlir::DataLayout::getGlobalMemorySpace() const {
  checkValid();
  if (globalMemorySpace)
    return *globalMemorySpace;
  DataLayoutEntryInterface entry;
  if (originalLayout)
    entry = originalLayout.getSpecForIdentifier(
        originalLayout.getGlobalMemorySpaceIdentifier(
            originalLayout.getContext()));
  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    globalMemorySpace = iface.getGlobalMemorySpace(entry);
  else
    globalMemorySpace = detail::getDefaultGlobalMemorySpace(entry);
  return *globalMemorySpace;
}

uint64_t mlir::DataLayout::getStackAlignment() const {
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

std::optional<Attribute> mlir::DataLayout::getDevicePropertyValue(
    TargetSystemSpecInterface::DeviceID deviceID,
    StringAttr propertyName) const {
  checkValid();
  DataLayoutEntryInterface entry;
  if (originalTargetSystemDesc) {
    if (std::optional<TargetDeviceSpecInterface> device =
            originalTargetSystemDesc.getDeviceSpecForDeviceID(deviceID))
      entry = device->getSpecForIdentifier(propertyName);
  }
  // Currently I am not caching the results because we do not return
  // default values of these properties. Instead if the property is
  // missing, we return std::nullopt so that the users can resort to
  // the default value however they want.
  if (auto iface = dyn_cast_or_null<DataLayoutOpInterface>(scope))
    return iface.getDevicePropertyValue(entry);
  else
    return detail::getDevicePropertyValue(entry);
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecInterface
//===----------------------------------------------------------------------===//

void DataLayoutSpecInterface::bucketEntriesByType(
    DenseMap<TypeID, DataLayoutEntryList> &types,
    DenseMap<StringAttr, DataLayoutEntryInterface> &ids) {
  for (DataLayoutEntryInterface entry : getEntries()) {
    if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey()))
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
  // are associated with.
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
        if (!value || !value.getElementType().isSignlessInteger(64)) {
          emitError(loc) << "expected a dense i64 elements attribute in the "
                            "data layout entry "
                         << entry;
          return failure();
        }

        auto elements = llvm::to_vector<2>(value.getValues<uint64_t>());
        unsigned numElements = elements.size();
        if (numElements < 1 || numElements > 2) {
          emitError(loc) << "expected 1 or 2 elements in the data layout entry "
                         << entry;
          return failure();
        }

        uint64_t abi = elements[0];
        uint64_t preferred = numElements == 2 ? elements[1] : abi;
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

LogicalResult
mlir::detail::verifyTargetSystemSpec(TargetSystemSpecInterface spec,
                                     Location loc) {
  DenseMap<StringAttr, DataLayoutEntryInterface> deviceDescKeys;
  DenseSet<TargetSystemSpecInterface::DeviceID> deviceIDs;
  for (const auto &entry : spec.getEntries()) {
    TargetDeviceSpecInterface targetDeviceSpec = entry.second;
    // First, verify individual target device desc specs.
    if (failed(targetDeviceSpec.verifyEntry(loc)))
      return failure();

    // Check that device IDs are unique across all entries.
    TargetSystemSpecInterface::DeviceID deviceID = entry.first;
    if (!deviceIDs.insert(deviceID).second) {
      return failure();
    }

    // collect all the keys used by all the target device specs.
    for (DataLayoutEntryInterface entry : targetDeviceSpec.getEntries()) {
      if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
        // targetDeviceSpec does not support Type as a key.
        return failure();
      } else {
        deviceDescKeys[entry.getKey().get<StringAttr>()] = entry;
      }
    }
  }

  for (const auto &[keyName, keyVal] : deviceDescKeys) {
    Dialect *dialect = keyName.getReferencedDialect();

    // Ignore attributes that belong to an unknown dialect, the dialect may
    // actually implement the relevant interface but we don't know about that.
    if (!dialect)
      return failure();

    const auto *iface = dyn_cast<DataLayoutDialectInterface>(dialect);
    if (!iface) {
      return emitError(loc)
             << "the '" << dialect->getNamespace()
             << "' dialect does not support identifier data layout entries";
    }
    if (failed(iface->verifyEntry(keyVal, loc)))
      return failure();
  }

  return success();
}

#include "mlir/Interfaces/DataLayoutAttrInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutOpInterface.cpp.inc"
#include "mlir/Interfaces/DataLayoutTypeInterface.cpp.inc"
