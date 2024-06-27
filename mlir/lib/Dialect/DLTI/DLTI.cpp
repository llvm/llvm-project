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

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/DLTI/DLTIAttrs.cpp.inc"

#define DEBUG_TYPE "dlti"

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
class DataLayoutEntryAttrStorage : public AttributeStorage {
public:
  using KeyTy = std::pair<DataLayoutEntryKey, Attribute>;

  DataLayoutEntryAttrStorage(DataLayoutEntryKey entryKey, Attribute value)
      : entryKey(entryKey), value(value) {}

  static DataLayoutEntryAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<DataLayoutEntryAttrStorage>())
        DataLayoutEntryAttrStorage(key.first, key.second);
  }

  bool operator==(const KeyTy &other) const {
    return other.first == entryKey && other.second == value;
  }

  DataLayoutEntryKey entryKey;
  Attribute value;
};
} // namespace detail
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
Attribute DataLayoutEntryAttr::parse(AsmParser &parser, Type ty) {
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
  os << "<";
  if (auto type = llvm::dyn_cast_if_present<Type>(getKey()))
    os << type;
  else
    os << "\"" << getKey().get<StringAttr>().strref() << "\"";
  os << ", " << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//

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
Attribute DataLayoutSpecAttr::parse(AsmParser &parser, Type type) {
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
  os << "<";
  llvm::interleaveComma(getEntries(), os);
  os << ">";
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

namespace mlir {
/// A FieldParser for key-value pairs of DeviceID-target device spec pairs that
/// make up a target system spec.
template <>
struct FieldParser<DeviceIDTargetDeviceSpecPair> {
  static FailureOr<DeviceIDTargetDeviceSpecPair> parse(AsmParser &parser) {
    std::string deviceID;

    if (failed(parser.parseString(&deviceID))) {
      parser.emitError(parser.getCurrentLocation())
          << "DeviceID is missing, or is not of string type";
      return failure();
    }

    if (failed(parser.parseColon())) {
      parser.emitError(parser.getCurrentLocation()) << "Missing colon";
      return failure();
    }

    auto target_device_spec =
        FieldParser<TargetDeviceSpecInterface>::parse(parser);
    if (failed(target_device_spec)) {
      parser.emitError(parser.getCurrentLocation())
          << "Error in parsing target device spec";
      return failure();
    }

    return std::make_pair(parser.getBuilder().getStringAttr(deviceID),
                          *target_device_spec);
  }
};

inline AsmPrinter &operator<<(AsmPrinter &printer,
                              DeviceIDTargetDeviceSpecPair param) {
  return printer << param.first << " : " << param.second;
}

} // namespace mlir

LogicalResult
TargetDeviceSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<DataLayoutEntryInterface> entries) {
  // Entries in a target device spec can only have StringAttr as key. It does
  // not support type as a key. Hence not reusing
  // DataLayoutEntryInterface::verify.
  DenseSet<StringAttr> ids;
  for (DataLayoutEntryInterface entry : entries) {
    if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
      return emitError()
             << "dlti.target_device_spec does not allow type as a key: "
             << type;
    } else {
      auto id = entry.getKey().get<StringAttr>();
      if (!ids.insert(id).second)
        return emitError() << "repeated layout entry key: " << id.getValue();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TargetSystemSpecAttr
//===----------------------------------------------------------------------===//

LogicalResult
TargetSystemSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<DeviceIDTargetDeviceSpecPair> entries) {
  DenseSet<TargetSystemSpecInterface::DeviceID> device_ids;

  for (const auto &entry : entries) {
    TargetDeviceSpecInterface target_device_spec = entry.second;

    // First verify that a target device spec is valid.
    if (failed(TargetDeviceSpecAttr::verify(emitError,
                                            target_device_spec.getEntries())))
      return failure();

    // Check that device IDs are unique across all entries.
    TargetSystemSpecInterface::DeviceID device_id = entry.first;
    if (!device_ids.insert(device_id).second) {
      return emitError() << "repeated Device ID in dlti.target_system_spec: "
                         << device_id;
    }
  }
  return success();
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

void DLTIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/DLTI/DLTIAttrs.cpp.inc"
      >();
  addInterfaces<TargetDataLayoutInterface>();
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
    if (!llvm::isa<TargetSystemSpecAttr>(attr.getValue())) {
      return op->emitError()
             << "'" << DLTIDialect::kTargetSystemDescAttrName
             << "' is expected to be a #dlti.target_system_spec attribute";
    }
    return success();
  }

  return op->emitError() << "attribute '" << attr.getName().getValue()
                         << "' not supported by dialect";
}
