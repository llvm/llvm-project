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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

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

DataLayoutEntryAttr DataLayoutEntryAttr::get(Identifier key, Attribute value) {
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
DataLayoutEntryAttr DataLayoutEntryAttr::parse(DialectAsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  Type type = nullptr;
  StringRef identifier;
  llvm::SMLoc idLoc = parser.getCurrentLocation();
  OptionalParseResult parsedType = parser.parseOptionalType(type);
  if (parsedType.hasValue() && failed(parsedType.getValue()))
    return {};
  if (!parsedType.hasValue()) {
    OptionalParseResult parsedString = parser.parseOptionalString(&identifier);
    if (!parsedString.hasValue() || failed(parsedString.getValue())) {
      parser.emitError(idLoc) << "expected a type or a quoted string";
      return {};
    }
  }

  Attribute value;
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(value)) ||
      failed(parser.parseGreater()))
    return {};

  return type ? get(type, value)
              : get(parser.getBuilder().getIdentifier(identifier), value);
}

void DataLayoutEntryAttr::print(DialectAsmPrinter &os) const {
  os << DataLayoutEntryAttr::kAttrKeyword << "<";
  if (auto type = getKey().dyn_cast<Type>())
    os << type;
  else
    os << "\"" << getKey().get<Identifier>().strref() << "\"";
  os << ", " << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//
//
constexpr const StringLiteral mlir::DataLayoutSpecAttr::kAttrKeyword;

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
  DenseSet<Identifier> ids;
  for (DataLayoutEntryInterface entry : entries) {
    if (auto type = entry.getKey().dyn_cast<Type>()) {
      if (!types.insert(type).second)
        return emitError() << "repeated layout entry key: " << type;
    } else {
      auto id = entry.getKey().get<Identifier>();
      if (!ids.insert(id).second)
        return emitError() << "repeated layout entry key: " << id;
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
               DenseMap<Identifier, DataLayoutEntryInterface> &entriesForID) {
  // A missing spec should be fine.
  if (!spec)
    return success();

  DenseMap<TypeID, DataLayoutEntryList> newEntriesForType;
  DenseMap<Identifier, DataLayoutEntryInterface> newEntriesForID;
  spec.bucketEntriesByType(newEntriesForType, newEntriesForID);

  // Try overwriting the old entries with the new ones.
  for (const auto &kvp : newEntriesForType) {
    if (!entriesForType.count(kvp.first)) {
      entriesForType[kvp.first] = std::move(kvp.second);
      continue;
    }

    Type typeSample = kvp.second.front().getKey().get<Type>();
    assert(&typeSample.getDialect() !=
               typeSample.getContext()->getLoadedDialect<BuiltinDialect>() &&
           "unexpected data layout entry for built-in type");

    auto interface = typeSample.cast<DataLayoutTypeInterface>();
    if (!interface.areCompatible(entriesForType.lookup(kvp.first), kvp.second))
      return failure();

    overwriteDuplicateEntries(entriesForType[kvp.first], kvp.second);
  }

  for (const auto &kvp : newEntriesForID) {
    Identifier id = kvp.second.getKey().get<Identifier>();
    Dialect *dialect = id.getDialect();
    if (!entriesForID.count(id)) {
      entriesForID[id] = kvp.second;
      continue;
    }

    // Attempt to combine the enties using the dialect interface. If the
    // dialect is not loaded for some reason, use the default combinator
    // that conservatively accepts identical entries only.
    entriesForID[id] =
        dialect ? dialect->getRegisteredInterface<DataLayoutDialectInterface>()
                      ->combine(entriesForID[id], kvp.second)
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
        return !spec.isa<DataLayoutSpecAttr>();
      }))
    return {};

  // Combine all specs in order, with `this` being the last one.
  DenseMap<TypeID, DataLayoutEntryList> entriesForType;
  DenseMap<Identifier, DataLayoutEntryInterface> entriesForID;
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

/// Parses an attribute with syntax
///   attr ::= `#target.` `dl_spec` `<` attr-list? `>`
///   attr-list ::= attr
///               | attr `,` attr-list
DataLayoutSpecAttr DataLayoutSpecAttr::parse(DialectAsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};

  // Empty spec.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getBuilder().getContext(), {});

  SmallVector<DataLayoutEntryInterface> entries;
  do {
    entries.emplace_back();
    if (failed(parser.parseAttribute(entries.back())))
      return {};
  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseGreater()))
    return {};
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getBuilder().getContext(), entries);
}

void DataLayoutSpecAttr::print(DialectAsmPrinter &os) const {
  os << DataLayoutSpecAttr::kAttrKeyword << "<";
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
    StringRef entryName = entry.getKey().get<Identifier>().strref();
    if (entryName == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = entry.getValue().dyn_cast<StringAttr>();
      if (value &&
          (value.getValue() == DLTIDialect::kDataLayoutEndiannessBig ||
           value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle))
        return success();
      return emitError(loc) << "'" << entryName
                            << "' data layout entry is expected to be either '"
                            << DLTIDialect::kDataLayoutEndiannessBig << "' or '"
                            << DLTIDialect::kDataLayoutEndiannessLittle << "'";
    }
    return emitError(loc) << "unknown data layout entry name: " << entryName;
  }
};
} // namespace

void DLTIDialect::initialize() {
  addAttributes<DataLayoutEntryAttr, DataLayoutSpecAttr>();
  addInterfaces<TargetDataLayoutInterface>();
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

  parser.emitError(parser.getNameLoc(), "unknown attrribute type: ")
      << attrKind;
  return {};
}

void DLTIDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  llvm::TypeSwitch<Attribute>(attr)
      .Case<DataLayoutEntryAttr, DataLayoutSpecAttr>(
          [&](auto a) { a.print(os); })
      .Default([](Attribute) { llvm_unreachable("unknown attribute kind"); });
}

LogicalResult DLTIDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  if (attr.first == DLTIDialect::kDataLayoutAttrName) {
    if (!attr.second.isa<DataLayoutSpecAttr>()) {
      return op->emitError() << "'" << DLTIDialect::kDataLayoutAttrName
                             << "' is expected to be a #dlti.dl_spec attribute";
    }
    if (isa<ModuleOp>(op))
      return detail::verifyDataLayoutOp(op);
    return success();
  }

  return op->emitError() << "attribute '" << attr.first
                         << "' not supported by dialect";
}
