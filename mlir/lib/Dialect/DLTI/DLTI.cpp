//===- DLTI.cpp - Data Layout And Target Info MLIR Dialect Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

#include "mlir/Dialect/DLTI/DLTIDialect.cpp.inc"

#define DEBUG_TYPE "dlti"

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//
//
constexpr const StringLiteral mlir::DataLayoutEntryAttr::kAttrKeyword;

namespace mlir {
namespace impl {
class DataLayoutEntryStorage : public AttributeStorage {
public:
  using KeyTy = std::pair<DataLayoutEntryKey, Attribute>;

  DataLayoutEntryStorage(DataLayoutEntryKey entryKey, Attribute value)
      : entryKey(entryKey), value(value) {}

  static DataLayoutEntryStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<DataLayoutEntryStorage>())
        DataLayoutEntryStorage(key.first, key.second);
  }

  bool operator==(const KeyTy &other) const {
    return other.first == entryKey && other.second == value;
  }

  DataLayoutEntryKey entryKey;
  Attribute value;
};
} // namespace impl
} // namespace mlir

DataLayoutEntryAttr DataLayoutEntryAttr::get(StringAttr key, Attribute value) {
  return Base::get(key.getContext(), key, value);
}

DataLayoutEntryAttr DataLayoutEntryAttr::get(Type key, Attribute value) {
  return Base::get(key.getContext(), key, value);
}

DataLayoutEntryKey DataLayoutEntryAttr::getKey() const {
  return getImpl()->entryKey;
}

Attribute DataLayoutEntryAttr::getValue() const { return getImpl()->value; }

/// Parses an attribute with syntax:
///   attr ::= `#target.` `dl_entry` `<` (type | quoted-string) `,` attr `>`
DataLayoutEntryAttr DataLayoutEntryAttr::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  Type type = nullptr;
  std::string identifier;
  SMLoc idLoc = parser.getCurrentLocation();
  OptionalParseResult parsedType = parser.parseOptionalType(type);
  if (parsedType.has_value() && failed(parsedType.value()))
    return {};
  if (!parsedType.has_value()) {
    OptionalParseResult parsedString = parser.parseOptionalString(&identifier);
    if (!parsedString.has_value() || failed(parsedString.value())) {
      parser.emitError(idLoc) << "expected a type or a quoted string";
      return {};
    }
  }

  Attribute value;
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(value)) ||
      failed(parser.parseGreater()))
    return {};

  return type ? get(type, value)
              : get(parser.getBuilder().getStringAttr(identifier), value);
}

void DataLayoutEntryAttr::print(AsmPrinter &os) const {
  os << DataLayoutEntryAttr::kAttrKeyword << "<";
  if (auto type = llvm::dyn_cast_if_present<Type>(getKey()))
    os << type;
  else
    os << "\"" << getKey().get<StringAttr>().strref() << "\"";
  os << ", " << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//
//
constexpr const StringLiteral mlir::DataLayoutSpecAttr::kAttrKeyword;
constexpr const StringLiteral
    mlir::DLTIDialect::kDataLayoutAllocaMemorySpaceKey;
constexpr const StringLiteral
    mlir::DLTIDialect::kDataLayoutProgramMemorySpaceKey;
constexpr const StringLiteral
    mlir::DLTIDialect::kDataLayoutGlobalMemorySpaceKey;

constexpr const StringLiteral mlir::DLTIDialect::kDataLayoutStackAlignmentKey;

namespace mlir {
namespace impl {
class DataLayoutSpecStorage : public AttributeStorage {
public:
  using KeyTy = ArrayRef<DataLayoutEntryInterface>;

  DataLayoutSpecStorage(ArrayRef<DataLayoutEntryInterface> entries)
      : entries(entries) {}

  bool operator==(const KeyTy &key) const { return key == entries; }

  static DataLayoutSpecStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<DataLayoutSpecStorage>())
        DataLayoutSpecStorage(allocator.copyInto(key));
  }

  ArrayRef<DataLayoutEntryInterface> entries;
};
} // namespace impl
} // namespace mlir

DataLayoutSpecAttr
DataLayoutSpecAttr::get(MLIRContext *ctx,
                        ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::get(ctx, entries);
}

DataLayoutSpecAttr
DataLayoutSpecAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context,
                               ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

LogicalResult
DataLayoutSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<DataLayoutEntryInterface> entries) {
  DenseSet<Type> types;
  DenseSet<StringAttr> ids;
  for (DataLayoutEntryInterface entry : entries) {
    if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
      if (!types.insert(type).second)
        return emitError() << "repeated layout entry key: " << type;
    } else {
      auto id = entry.getKey().get<StringAttr>();
      if (!ids.insert(id).second)
        return emitError() << "repeated layout entry key: " << id.getValue();
    }
  }
  return success();
}

/// Given a list of old and a list of new entries, overwrites old entries with
/// new ones if they have matching keys, appends new entries to the old entry
/// list otherwise.
static void
overwriteDuplicateEntries(SmallVectorImpl<DataLayoutEntryInterface> &oldEntries,
                          ArrayRef<DataLayoutEntryInterface> newEntries) {
  unsigned oldEntriesSize = oldEntries.size();
  for (DataLayoutEntryInterface entry : newEntries) {
    // We expect a small (dozens) number of entries, so it is practically
    // cheaper to iterate over the list linearly rather than to create an
    // auxiliary hashmap to avoid duplication. Also note that we never need to
    // check for duplicate keys the values that were added from `newEntries`.
    bool replaced = false;
    for (unsigned i = 0; i < oldEntriesSize; ++i) {
      if (oldEntries[i].getKey() == entry.getKey()) {
        oldEntries[i] = entry;
        replaced = true;
        break;
      }
    }
    if (!replaced)
      oldEntries.push_back(entry);
  }
}

/// Combines a data layout spec into the given lists of entries organized by
/// type class and identifier, overwriting them if necessary. Fails to combine
/// if the two entries with identical keys are not compatible.
static LogicalResult
combineOneSpec(DataLayoutSpecInterface spec,
               DenseMap<TypeID, DataLayoutEntryList> &entriesForType,
               DenseMap<StringAttr, DataLayoutEntryInterface> &entriesForID) {
  // A missing spec should be fine.
  if (!spec)
    return success();

  DenseMap<TypeID, DataLayoutEntryList> newEntriesForType;
  DenseMap<StringAttr, DataLayoutEntryInterface> newEntriesForID;
  spec.bucketEntriesByType(newEntriesForType, newEntriesForID);

  // Try overwriting the old entries with the new ones.
  for (auto &kvp : newEntriesForType) {
    if (!entriesForType.count(kvp.first)) {
      entriesForType[kvp.first] = std::move(kvp.second);
      continue;
    }

    Type typeSample = kvp.second.front().getKey().get<Type>();
    assert(&typeSample.getDialect() !=
               typeSample.getContext()->getLoadedDialect<BuiltinDialect>() &&
           "unexpected data layout entry for built-in type");

    auto interface = llvm::cast<DataLayoutTypeInterface>(typeSample);
    if (!interface.areCompatible(entriesForType.lookup(kvp.first), kvp.second))
      return failure();

    overwriteDuplicateEntries(entriesForType[kvp.first], kvp.second);
  }

  for (const auto &kvp : newEntriesForID) {
    StringAttr id = kvp.second.getKey().get<StringAttr>();
    Dialect *dialect = id.getReferencedDialect();
    if (!entriesForID.count(id)) {
      entriesForID[id] = kvp.second;
      continue;
    }

    // Attempt to combine the enties using the dialect interface. If the
    // dialect is not loaded for some reason, use the default combinator
    // that conservatively accepts identical entries only.
    entriesForID[id] =
        dialect ? cast<DataLayoutDialectInterface>(dialect)->combine(
                      entriesForID[id], kvp.second)
                : DataLayoutDialectInterface::defaultCombine(entriesForID[id],
                                                             kvp.second);
    if (!entriesForID[id])
      return failure();
  }

  return success();
}

DataLayoutSpecAttr
DataLayoutSpecAttr::combineWith(ArrayRef<DataLayoutSpecInterface> specs) const {
  // Only combine with attributes of the same kind.
  // TODO: reconsider this when the need arises.
  if (llvm::any_of(specs, [](DataLayoutSpecInterface spec) {
        return !llvm::isa<DataLayoutSpecAttr>(spec);
      }))
    return {};

  // Combine all specs in order, with `this` being the last one.
  DenseMap<TypeID, DataLayoutEntryList> entriesForType;
  DenseMap<StringAttr, DataLayoutEntryInterface> entriesForID;
  for (DataLayoutSpecInterface spec : specs)
    if (failed(combineOneSpec(spec, entriesForType, entriesForID)))
      return nullptr;
  if (failed(combineOneSpec(*this, entriesForType, entriesForID)))
    return nullptr;

  // Rebuild the linear list of entries.
  SmallVector<DataLayoutEntryInterface> entries;
  llvm::append_range(entries, llvm::make_second_range(entriesForID));
  for (const auto &kvp : entriesForType)
    llvm::append_range(entries, kvp.getSecond());

  return DataLayoutSpecAttr::get(getContext(), entries);
}

DataLayoutEntryListRef DataLayoutSpecAttr::getEntries() const {
  return getImpl()->entries;
}

StringAttr
DataLayoutSpecAttr::getEndiannessIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(DLTIDialect::kDataLayoutEndiannessKey);
}

StringAttr
DataLayoutSpecAttr::getAllocaMemorySpaceIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutAllocaMemorySpaceKey);
}

StringAttr DataLayoutSpecAttr::getProgramMemorySpaceIdentifier(
    MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutProgramMemorySpaceKey);
}

StringAttr
DataLayoutSpecAttr::getGlobalMemorySpaceIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutGlobalMemorySpaceKey);
}
StringAttr
DataLayoutSpecAttr::getStackAlignmentIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutStackAlignmentKey);
}

/// Parses an attribute with syntax
///   attr ::= `#target.` `dl_spec` `<` attr-list? `>`
///   attr-list ::= attr
///               | attr `,` attr-list
DataLayoutSpecAttr DataLayoutSpecAttr::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  // Empty spec.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), {});

  SmallVector<DataLayoutEntryInterface> entries;
  if (parser.parseCommaSeparatedList(
          [&]() { return parser.parseAttribute(entries.emplace_back()); }) ||
      parser.parseGreater())
    return {};

  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), entries);
}

void DataLayoutSpecAttr::print(AsmPrinter &os) const {
  os << DataLayoutSpecAttr::kAttrKeyword << "<";
  llvm::interleaveComma(getEntries(), os);
  os << ">";
}

//===----------------------------------------------------------------------===//
// TargetDeviceDescSpecAttr
//===----------------------------------------------------------------------===//
constexpr const StringLiteral mlir::TargetDeviceDescSpecAttr::kAttrKeyword;

constexpr const StringLiteral mlir::DLTIDialect::kTargetDeviceIDKey;
constexpr const StringLiteral mlir::DLTIDialect::kTargetDeviceTypeKey;
constexpr const StringLiteral
    mlir::DLTIDialect::kTargetDeviceMaxVectorOpWidthKey;
constexpr const StringLiteral
    mlir::DLTIDialect::kTargetDeviceCanonicalizerMaxIterationsKey;
constexpr const StringLiteral
    mlir::DLTIDialect::kTargetDeviceCanonicalizerMaxNumRewritesKey;

namespace mlir {
namespace impl {
class TargetDeviceDescSpecAttrStorage : public AttributeStorage {
public:
  using KeyTy = ArrayRef<DataLayoutEntryInterface>;

  TargetDeviceDescSpecAttrStorage(KeyTy entries) : entries(entries) {}

  bool operator==(const KeyTy &key) const { return key == entries; }

  static TargetDeviceDescSpecAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetDeviceDescSpecAttrStorage>())
        TargetDeviceDescSpecAttrStorage(allocator.copyInto(key));
  }

  ArrayRef<DataLayoutEntryInterface> entries;
};
} // namespace impl
} // namespace mlir

TargetDeviceDescSpecAttr
TargetDeviceDescSpecAttr::get(MLIRContext *ctx,
                              ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::get(ctx, entries);
}

DataLayoutEntryListRef TargetDeviceDescSpecAttr::getEntries() const {
  return getImpl()->entries;
}

TargetDeviceDescSpecAttr TargetDeviceDescSpecAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

LogicalResult
TargetDeviceDescSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<DataLayoutEntryInterface> entries) {
  // Entries in tdd_spec can only have StringAttr as key. It does not support
  // type as a key. Hence not reusing DataLayoutEntryInterface::verify.
  bool targetDeviceIDKeyPresentAndValid = false;
  bool targetDeviceTypeKeyPresentAndValid = false;

  DenseSet<StringAttr> ids;
  for (DataLayoutEntryInterface entry : entries) {
    if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
      return emitError()
             << "dlti.tdd_spec attribute does not allow type as a key: "
             << type;
    } else {
      auto id = entry.getKey().get<StringAttr>();
      if (!ids.insert(id).second)
        return emitError() << "repeated layout entry key: " << id.getValue();
    }

    // check that Device ID and Device Type are present.
    StringRef entryName = entry.getKey().get<StringAttr>().strref();
    if (entryName == DLTIDialect::kTargetDeviceIDKey) {
      // Also check the type of the value.
      IntegerAttr value =
          llvm::dyn_cast_if_present<IntegerAttr>(entry.getValue());
      if (value && value.getType().isUnsignedInteger(32)) {
        targetDeviceIDKeyPresentAndValid = true;
      }
    } else if (entryName == DLTIDialect::kTargetDeviceTypeKey) {
      // Also check the type of the value.
      if (auto value = llvm::dyn_cast<StringAttr>(entry.getValue())) {
        targetDeviceTypeKeyPresentAndValid = true;
      }
    }
  }

  // check that both DeviceID and DeviceType are present
  // and are of correct type.
  if (!targetDeviceIDKeyPresentAndValid) {
    return emitError() << "tdd_spec requires key: "
                       << DLTIDialect::kTargetDeviceIDKey
                       << " and its value of ui32 type";
  }
  if (!targetDeviceTypeKeyPresentAndValid) {
    return emitError() << "tdd_spec requires key: "
                       << DLTIDialect::kTargetDeviceTypeKey
                       << " and its value of string type";
  }

  return success();
}

/// Parses an attribute with syntax
///   tdd_spec_attr ::= `#target.` `tdd_spec` `<` dl-entry-attr-list? `>`
///   dl-entry-attr-list ::= dl-entry-attr
///                         | dl-entry-attr `,` dl-entry-attr-list
TargetDeviceDescSpecAttr TargetDeviceDescSpecAttr::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  // Empty spec.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), {});

  SmallVector<DataLayoutEntryInterface> entries;
  if (parser.parseCommaSeparatedList(
          [&]() { return parser.parseAttribute(entries.emplace_back()); }) ||
      parser.parseGreater())
    return {};

  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), entries);
}

void TargetDeviceDescSpecAttr::print(AsmPrinter &os) const {
  os << TargetDeviceDescSpecAttr::kAttrKeyword << "<";
  llvm::interleaveComma(getEntries(), os);
  os << ">";
}

// ---------------------------------------------------------------------------//
//                      Support for specific keys
// ---------------------------------------------------------------------------//

StringAttr
TargetDeviceDescSpecAttr::getDeviceIDIdentifier(MLIRContext *context) {
  return Builder(context).getStringAttr(DLTIDialect::kTargetDeviceIDKey);
}

StringAttr
TargetDeviceDescSpecAttr::getDeviceTypeIdentifier(MLIRContext *context) {
  return Builder(context).getStringAttr(DLTIDialect::kTargetDeviceTypeKey);
}

StringAttr
TargetDeviceDescSpecAttr::getMaxVectorOpWidthIdentifier(MLIRContext *context) {
  return Builder(context).getStringAttr(
      DLTIDialect::kTargetDeviceMaxVectorOpWidthKey);
}

StringAttr TargetDeviceDescSpecAttr::getCanonicalizerMaxIterationsIdentifier(
    MLIRContext *context) {
  return Builder(context).getStringAttr(
      DLTIDialect::kTargetDeviceCanonicalizerMaxIterationsKey);
}

StringAttr TargetDeviceDescSpecAttr::getCanonicalizerMaxNumRewritesIdentifier(
    MLIRContext *context) {
  return Builder(context).getStringAttr(
      DLTIDialect::kTargetDeviceCanonicalizerMaxNumRewritesKey);
}

DataLayoutEntryInterface
TargetDeviceDescSpecAttr::getSpecForDeviceID(MLIRContext *context) {
  return getSpecForIdentifier(getDeviceIDIdentifier(context));
}

DataLayoutEntryInterface
TargetDeviceDescSpecAttr::getSpecForDeviceType(MLIRContext *context) {
  return getSpecForIdentifier(getDeviceTypeIdentifier(context));
}

DataLayoutEntryInterface
TargetDeviceDescSpecAttr::getSpecForMaxVectorOpWidth(MLIRContext *context) {
  return getSpecForIdentifier(getMaxVectorOpWidthIdentifier(context));
}

DataLayoutEntryInterface
TargetDeviceDescSpecAttr::getSpecForCanonicalizerMaxIterations(
    MLIRContext *context) {
  return getSpecForIdentifier(getCanonicalizerMaxIterationsIdentifier(context));
}

DataLayoutEntryInterface
TargetDeviceDescSpecAttr::getSpecForCanonicalizerMaxNumRewrites(
    MLIRContext *context) {
  return getSpecForIdentifier(
      getCanonicalizerMaxNumRewritesIdentifier(context));
}

uint32_t TargetDeviceDescSpecAttr::getDeviceID(MLIRContext *context) {
  DataLayoutEntryInterface entry = getSpecForDeviceID(context);
  return llvm::cast<IntegerAttr>(entry.getValue()).getValue().getZExtValue();
}

//===----------------------------------------------------------------------===//
// TargetSystemDescSpecAttr
//===----------------------------------------------------------------------===//

constexpr const StringLiteral mlir::TargetSystemDescSpecAttr::kAttrKeyword;

namespace mlir {
namespace impl {
class TargetSystemDescSpecAttrStorage : public AttributeStorage {
public:
  using KeyTy = ArrayRef<TargetDeviceDescSpecInterface>;

  TargetSystemDescSpecAttrStorage(KeyTy entries) : entries(entries) {}

  bool operator==(const KeyTy &key) const { return key == entries; }

  static TargetSystemDescSpecAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetSystemDescSpecAttrStorage>())
        TargetSystemDescSpecAttrStorage(allocator.copyInto(key));
  }

  // This could be a map of DeviceID to DeviceDesc for faster lookup.
  ArrayRef<TargetDeviceDescSpecInterface> entries;
};
} // namespace impl
} // namespace mlir

TargetSystemDescSpecAttr
TargetSystemDescSpecAttr::get(MLIRContext *context,
                              ArrayRef<TargetDeviceDescSpecInterface> entries) {
  return Base::get(context, entries);
}

TargetDeviceDescSpecListRef TargetSystemDescSpecAttr::getEntries() const {
  return getImpl()->entries;
}

TargetDeviceDescSpecInterface
TargetSystemDescSpecAttr::getDeviceDescForDeviceID(
    TargetDeviceDescSpecInterface::DeviceID DeviceID) {
  for (TargetDeviceDescSpecInterface entry : getEntries()) {
    if (entry.getDeviceID(getContext()) == DeviceID)
      return entry;
  }
  return TargetDeviceDescSpecInterface();
}

TargetSystemDescSpecAttr TargetSystemDescSpecAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<TargetDeviceDescSpecInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

LogicalResult TargetSystemDescSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<TargetDeviceDescSpecInterface> entries) {
  DenseSet<uint32_t> device_ids;

  for (TargetDeviceDescSpecInterface tdd_spec : entries) {
    // First verify that a target device desc spec is valid.
    if (failed(
            TargetDeviceDescSpecAttr::verify(emitError, tdd_spec.getEntries())))
      return failure();

    // Check that device IDs are unique across all entries.
    MLIRContext *context = tdd_spec.getContext();
    uint32_t device_id = tdd_spec.getDeviceID(context);
    if (!device_ids.insert(device_id).second) {
      return emitError() << "repeated Device ID in dlti.tsd_spec: "
                         << device_id;
    }
  }
  return success();
}

/// Parses an attribute with syntax
///   attr ::= `#target.` `tsd_spec` `<` tdd-spec-attr-list? `>`
///   tdd-spec-attr-list ::= tdd_spec
///                         | tdd_spec `,` tdd_spec_attr_list
TargetSystemDescSpecAttr TargetSystemDescSpecAttr::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  // Empty spec.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), {});

  SmallVector<TargetDeviceDescSpecInterface> entries;
  if (parser.parseCommaSeparatedList(
          [&]() { return parser.parseAttribute(entries.emplace_back()); }) ||
      parser.parseGreater())
    return {};

  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), entries);
}

void TargetSystemDescSpecAttr::print(AsmPrinter &os) const {
  os << TargetSystemDescSpecAttr::kAttrKeyword << "<";
  llvm::interleaveComma(getEntries(), os);
  os << ">";
}

//===----------------------------------------------------------------------===//
// DLTIDialect
//===----------------------------------------------------------------------===//

constexpr const StringLiteral mlir::DLTIDialect::kDataLayoutAttrName;
constexpr const StringLiteral mlir::DLTIDialect::kDataLayoutEndiannessKey;
constexpr const StringLiteral mlir::DLTIDialect::kDataLayoutEndiannessBig;
constexpr const StringLiteral mlir::DLTIDialect::kDataLayoutEndiannessLittle;

namespace {
class TargetDataLayoutInterface : public DataLayoutDialectInterface {
public:
  using DataLayoutDialectInterface::DataLayoutDialectInterface;

  LogicalResult verifyEntry(DataLayoutEntryInterface entry,
                            Location loc) const final {
    StringRef entryName = entry.getKey().get<StringAttr>().strref();
    if (entryName == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = llvm::dyn_cast<StringAttr>(entry.getValue());
      if (value &&
          (value.getValue() == DLTIDialect::kDataLayoutEndiannessBig ||
           value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle))
        return success();
      return emitError(loc) << "'" << entryName
                            << "' data layout entry is expected to be either '"
                            << DLTIDialect::kDataLayoutEndiannessBig << "' or '"
                            << DLTIDialect::kDataLayoutEndiannessLittle << "'";
    }
    if (entryName == DLTIDialect::kDataLayoutAllocaMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutProgramMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutGlobalMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutStackAlignmentKey)
      return success();
    return emitError(loc) << "unknown data layout entry name: " << entryName;
  }
};
} // namespace

namespace {
class SystemDescSpecInterface : public DataLayoutDialectInterface {
public:
  using DataLayoutDialectInterface::DataLayoutDialectInterface;

  LogicalResult verifyEntry(TargetDeviceDescSpecInterface entry,
                            Location loc) const final {

    for (DataLayoutEntryInterface dl_entry : entry.getEntries()) {
      StringRef entryName = dl_entry.getKey().get<StringAttr>().strref();
      // Check that the key name is known to us. Although, we may allow keys
      // unknown to us.
      if (entryName != DLTIDialect::kTargetDeviceIDKey &&
          entryName != DLTIDialect::kTargetDeviceTypeKey &&
          entryName != DLTIDialect::kTargetDeviceMaxVectorOpWidthKey &&
          entryName !=
              DLTIDialect::kTargetDeviceCanonicalizerMaxIterationsKey &&
          entryName != DLTIDialect::kTargetDeviceCanonicalizerMaxNumRewritesKey)
        return emitError(loc) << "unknown target desc key name: " << entryName;
    }
    return success();
  }
};
} // namespace

void DLTIDialect::initialize() {
  addAttributes<DataLayoutEntryAttr, DataLayoutSpecAttr,
                TargetSystemDescSpecAttr, TargetDeviceDescSpecAttr>();
  addInterfaces<TargetDataLayoutInterface, SystemDescSpecInterface>();
}

Attribute DLTIDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrKind;
  if (parser.parseKeyword(&attrKind))
    return {};

  if (attrKind == DataLayoutEntryAttr::kAttrKeyword)
    return DataLayoutEntryAttr::parse(parser);
  if (attrKind == DataLayoutSpecAttr::kAttrKeyword)
    return DataLayoutSpecAttr::parse(parser);
  if (attrKind == TargetSystemDescSpecAttr::kAttrKeyword)
    return TargetSystemDescSpecAttr::parse(parser);
  if (attrKind == TargetDeviceDescSpecAttr::kAttrKeyword)
    return TargetDeviceDescSpecAttr::parse(parser);

  parser.emitError(parser.getNameLoc(), "unknown attrribute type: ")
      << attrKind;
  return {};
}

void DLTIDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  llvm::TypeSwitch<Attribute>(attr)
      .Case<DataLayoutEntryAttr, DataLayoutSpecAttr, TargetSystemDescSpecAttr,
            TargetDeviceDescSpecAttr>([&](auto a) { a.print(os); })
      .Default([](Attribute) { llvm_unreachable("unknown attribute kind"); });
}

LogicalResult DLTIDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  if (attr.getName() == DLTIDialect::kDataLayoutAttrName) {
    if (!llvm::isa<DataLayoutSpecAttr>(attr.getValue())) {
      return op->emitError() << "'" << DLTIDialect::kDataLayoutAttrName
                             << "' is expected to be a #dlti.dl_spec attribute";
    }
    if (isa<ModuleOp>(op))
      return detail::verifyDataLayoutOp(op);
    return success();
  } else if (attr.getName() == DLTIDialect::kTargetSystemDescAttrName) {
    if (!llvm::isa<TargetSystemDescSpecAttr>(attr.getValue())) {
      return op->emitError()
             << "'" << DLTIDialect::kTargetSystemDescAttrName
             << "' is expected to be a #dlti.tsd_spec attribute";
    }
    return success();
  }

  return op->emitError() << "attribute '" << attr.getName().getValue()
                         << "' not supported by dialect";
}
