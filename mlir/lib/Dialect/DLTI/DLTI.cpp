//===- DLTI.cpp - Data Layout And Target Info MLIR Dialect Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

#include "mlir/Dialect/DLTI/DLTIDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/DLTI/DLTIAttrs.cpp.inc"

#define DEBUG_TYPE "dlti"

//===----------------------------------------------------------------------===//
// Common parsing utility functions.
//===----------------------------------------------------------------------===//

/// Parse an entry which can either be of the form `key = value` or a
/// #dlti.dl_entry attribute. When `tryType=true` the key can be a type,
/// otherwise only quoted strings are allowed. The grammar is as follows:
///   entry ::= ((type | quoted-string) `=` attr) | dl-entry-attr
static ParseResult parseKeyValuePair(AsmParser &parser,
                                     DataLayoutEntryInterface &entry,
                                     bool tryType = false) {
  Attribute value;

  if (tryType) {
    Type type;
    OptionalParseResult parsedType = parser.parseOptionalType(type);
    if (parsedType.has_value()) {
      if (failed(parsedType.value()))
        return parser.emitError(parser.getCurrentLocation())
               << "error while parsing type DLTI key";

      if (failed(parser.parseEqual()) || failed(parser.parseAttribute(value)))
        return failure();

      entry = DataLayoutEntryAttr::get(type, value);
      return ParseResult::success();
    }
  }

  std::string ident;
  OptionalParseResult parsedStr = parser.parseOptionalString(&ident);
  if (parsedStr.has_value() && succeeded(parsedStr.value())) {
    if (failed(parser.parseEqual()) || failed(parser.parseAttribute(value)))
      return failure(); // Assume that an error has already been emitted.

    entry = DataLayoutEntryAttr::get(
        StringAttr::get(parser.getContext(), ident), value);
    return ParseResult::success();
  }

  OptionalParseResult parsedEntry = parser.parseAttribute(entry);
  if (parsedEntry.has_value()) {
    if (succeeded(parsedEntry.value()))
      return parsedEntry.value();
    return failure(); // Assume that an error has already been emitted.
  }
  return parser.emitError(parser.getCurrentLocation())
         << "failed to parse DLTI entry";
}

/// Construct a requested attribute by parsing list of entries occurring within
/// a pair of `<` and `>`, optionally allow types as keys and an empty list.
/// The grammar is as follows:
///   bracketed-entry-list ::=`<` entry-list `>`
///   entry-list ::= | entry | entry `,` entry-list
///   entry ::= ((type | quoted-string) `=` attr) | dl-entry-attr
template <class Attr>
static Attribute parseAngleBracketedEntries(AsmParser &parser, Type ty,
                                            bool tryType = false,
                                            bool allowEmpty = false) {
  SmallVector<DataLayoutEntryInterface> entries;
  if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() {
            return parseKeyValuePair(parser, entries.emplace_back(), tryType);
          })))
    return {};

  if (entries.empty() && !allowEmpty) {
    parser.emitError(parser.getNameLoc()) << "no DLTI entries provided";
    return {};
  }

  return Attr::getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                          parser.getContext(), ArrayRef(entries));
}

//===----------------------------------------------------------------------===//
// Common printing utility functions.
//===----------------------------------------------------------------------===//

/// Convert pointer-union keys to strings.
static std::string keyToStr(DataLayoutEntryKey key) {
  std::string buf;
  TypeSwitch<DataLayoutEntryKey>(key)
      .Case<StringAttr, Type>( // The only two kinds of key we know of.
          [&](auto key) { llvm::raw_string_ostream(buf) << key; });
  return buf;
}

/// Pretty-print entries, each in `key = value` format, separated by commas.
template <class T>
static void printAngleBracketedEntries(AsmPrinter &os, T &&entries) {
  os << "<";
  llvm::interleaveComma(std::forward<T>(entries), os, [&](auto entry) {
    os << keyToStr(entry.getKey()) << " = " << entry.getValue();
  });
  os << ">";
}

//===----------------------------------------------------------------------===//
// Common verifying utility functions.
//===----------------------------------------------------------------------===//

/// Verify entries, with the option to disallow types as keys.
static LogicalResult verifyEntries(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<DataLayoutEntryInterface> entries,
                                   bool allowTypes = true) {
  DenseSet<DataLayoutEntryKey> keys;
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry)
      return emitError() << "contained invalid DLTI entry";
    DataLayoutEntryKey key = entry.getKey();
    if (key.isNull())
      return emitError() << "contained invalid DLTI key";
    if (!allowTypes && dyn_cast<Type>(key))
      return emitError() << "type as DLTI key is not allowed";
    if (auto strKey = dyn_cast<StringAttr>(key))
      if (strKey.getValue().empty())
        return emitError() << "empty string as DLTI key is not allowed";
    if (!keys.insert(key).second)
      return emitError() << "repeated DLTI key: " << keyToStr(key);
    if (!entry.getValue())
      return emitError() << "value associated to DLTI key " << keyToStr(key)
                         << " is invalid";
  }
  return success();
}

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
///   dl-entry-attr ::= `#dlti.` `dl_entry` `<` (type | quoted-string) `,`
///     attr `>`
Attribute DataLayoutEntryAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  Type typeKey = nullptr;
  std::string identifier;
  SMLoc idLoc = parser.getCurrentLocation();
  OptionalParseResult parsedType = parser.parseOptionalType(typeKey);
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

  return typeKey ? get(typeKey, value)
                 : get(parser.getBuilder().getStringAttr(identifier), value);
}

void DataLayoutEntryAttr::print(AsmPrinter &printer) const {
  printer << "<" << keyToStr(getKey()) << ", " << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// DLTIMapAttr
//===----------------------------------------------------------------------===//

/// Parses an attribute with syntax:
///   map-attr ::= `#dlti.` `map` `<` entry-list `>`
///   entry-list ::= entry | entry `,` entry-list
///   entry ::= ((type | quoted-string) `=` attr) | dl-entry-attr
Attribute MapAttr::parse(AsmParser &parser, Type type) {
  return parseAngleBracketedEntries<MapAttr>(parser, type, /*tryType=*/true,
                                             /*allowEmpty=*/true);
}

void MapAttr::print(AsmPrinter &printer) const {
  printAngleBracketedEntries(printer, getEntries());
}

LogicalResult MapAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<DataLayoutEntryInterface> entries) {
  return verifyEntries(emitError, entries);
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//

LogicalResult
DataLayoutSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<DataLayoutEntryInterface> entries) {
  return verifyEntries(emitError, entries);
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
static LogicalResult combineOneSpec(
    DataLayoutSpecInterface spec,
    llvm::MapVector<TypeID, DataLayoutEntryList> &entriesForType,
    llvm::MapVector<StringAttr, DataLayoutEntryInterface> &entriesForID) {
  // A missing spec should be fine.
  if (!spec)
    return success();

  llvm::MapVector<TypeID, DataLayoutEntryList> newEntriesForType;
  llvm::MapVector<StringAttr, DataLayoutEntryInterface> newEntriesForID;
  spec.bucketEntriesByType(newEntriesForType, newEntriesForID);

  // Combine non-Type DL entries first so they are visible to the
  // `type.areCompatible` method, allowing to query global properties.
  for (const auto &kvp : newEntriesForID) {
    StringAttr id = cast<StringAttr>(kvp.second.getKey());
    Dialect *dialect = id.getReferencedDialect();
    if (!entriesForID.count(id)) {
      entriesForID[id] = kvp.second;
      continue;
    }

    // Attempt to combine the entries using the dialect interface. If the
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

  // Try overwriting the old entries with the new ones.
  for (auto &kvp : newEntriesForType) {
    if (!entriesForType.count(kvp.first)) {
      entriesForType[kvp.first] = std::move(kvp.second);
      continue;
    }

    Type typeSample = cast<Type>(kvp.second.front().getKey());
    assert(&typeSample.getDialect() !=
               typeSample.getContext()->getLoadedDialect<BuiltinDialect>() &&
           "unexpected data layout entry for built-in type");

    auto interface = cast<DataLayoutTypeInterface>(typeSample);
    // TODO: Revisit this method and call once
    // https://github.com/llvm/llvm-project/issues/130321 gets resolved.
    if (!interface.areCompatible(entriesForType.lookup(kvp.first), kvp.second,
                                 spec, entriesForID))
      return failure();

    overwriteDuplicateEntries(entriesForType[kvp.first], kvp.second);
  }

  return success();
}

DataLayoutSpecAttr
DataLayoutSpecAttr::combineWith(ArrayRef<DataLayoutSpecInterface> specs) const {
  // Only combine with attributes of the same kind.
  // TODO: reconsider this when the need arises.
  if (any_of(specs, [](DataLayoutSpecInterface spec) {
        return !llvm::isa<DataLayoutSpecAttr>(spec);
      }))
    return {};

  // Combine all specs in order, with `this` being the last one.
  llvm::MapVector<TypeID, DataLayoutEntryList> entriesForType;
  llvm::MapVector<StringAttr, DataLayoutEntryInterface> entriesForID;
  for (DataLayoutSpecInterface spec : specs)
    if (failed(combineOneSpec(spec, entriesForType, entriesForID)))
      return nullptr;
  if (failed(combineOneSpec(*this, entriesForType, entriesForID)))
    return nullptr;

  // Rebuild the linear list of entries.
  SmallVector<DataLayoutEntryInterface> entries;
  llvm::append_range(entries, llvm::make_second_range(entriesForID));
  for (const auto &kvp : entriesForType)
    llvm::append_range(entries, kvp.second);

  return DataLayoutSpecAttr::get(getContext(), entries);
}

StringAttr
DataLayoutSpecAttr::getEndiannessIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(DLTIDialect::kDataLayoutEndiannessKey);
}

StringAttr DataLayoutSpecAttr::getDefaultMemorySpaceIdentifier(
    MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutDefaultMemorySpaceKey);
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
DataLayoutSpecAttr::getManglingModeIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutManglingModeKey);
}

StringAttr
DataLayoutSpecAttr::getStackAlignmentIdentifier(MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutStackAlignmentKey);
}

StringAttr DataLayoutSpecAttr::getFunctionPointerAlignmentIdentifier(
    MLIRContext *context) const {
  return Builder(context).getStringAttr(
      DLTIDialect::kDataLayoutFunctionPointerAlignmentKey);
}

/// Parses an attribute with syntax:
///   dl-spec-attr ::= `#dlti.` `dl_spec` `<` entry-list `>`
///   entry-list ::= | entry | entry `,` entry-list
///   entry ::= ((type | quoted-string) = attr) | dl-entry-attr
Attribute DataLayoutSpecAttr::parse(AsmParser &parser, Type type) {
  return parseAngleBracketedEntries<DataLayoutSpecAttr>(parser, type,
                                                        /*tryType=*/true,
                                                        /*allowEmpty=*/true);
}

void DataLayoutSpecAttr::print(AsmPrinter &printer) const {
  printAngleBracketedEntries(printer, getEntries());
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

LogicalResult
TargetDeviceSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<DataLayoutEntryInterface> entries) {
  return verifyEntries(emitError, entries, /*allowTypes=*/false);
}

/// Parses an attribute with syntax:
///   dev-spec-attr ::= `#dlti.` `target_device_spec` `<` entry-list `>`
///   entry-list ::= entry | entry `,` entry-list
///   entry ::= (quoted-string `=` attr) | dl-entry-attr
Attribute TargetDeviceSpecAttr::parse(AsmParser &parser, Type type) {
  return parseAngleBracketedEntries<TargetDeviceSpecAttr>(parser, type);
}

void TargetDeviceSpecAttr::print(AsmPrinter &printer) const {
  printAngleBracketedEntries(printer, getEntries());
}

//===----------------------------------------------------------------------===//
// TargetSystemSpecAttr
//===----------------------------------------------------------------------===//

LogicalResult
TargetSystemSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<DataLayoutEntryInterface> entries) {
  DenseSet<TargetSystemSpecInterface::DeviceID> deviceIds;

  for (const auto &entry : entries) {
    auto deviceId =
        llvm::dyn_cast<TargetSystemSpecInterface::DeviceID>(entry.getKey());
    if (!deviceId)
      return emitError() << "non-string key of DLTI system spec";

    if (auto targetDeviceSpec =
            llvm::dyn_cast<TargetDeviceSpecInterface>(entry.getValue())) {
      if (failed(TargetDeviceSpecAttr::verify(emitError,
                                              targetDeviceSpec.getEntries())))
        return failure(); // Assume sub-verifier outputted error message.
    } else {
      return emitError() << "value associated with key " << deviceId
                         << " is not a DLTI device spec";
    }

    // Check that device IDs are unique across all entries.
    if (!deviceIds.insert(deviceId).second)
      return emitError() << "repeated device ID in dlti.target_system_spec: "
                         << deviceId;
  }

  return success();
}

/// Parses an attribute with syntax:
///   sys-spec-attr ::= `#dlti.` `target_system_spec` `<` entry-list `>`
///   entry-list ::= entry | entry `,` entry-list
///   entry ::= (quoted-string `=` dev-spec-attr) | dl-entry-attr
Attribute TargetSystemSpecAttr::parse(AsmParser &parser, Type type) {
  return parseAngleBracketedEntries<TargetSystemSpecAttr>(parser, type);
}

void TargetSystemSpecAttr::print(AsmPrinter &printer) const {
  printAngleBracketedEntries(printer, getEntries());
}

//===----------------------------------------------------------------------===//
// DLTIDialect
//===----------------------------------------------------------------------===//

/// Retrieve the first `DLTIQueryInterface`-implementing attribute that is
/// attached to `op` or such an attr on as close as possible an ancestor. The
/// op the attribute is attached to is returned as well.
static std::pair<DLTIQueryInterface, Operation *>
getClosestQueryable(Operation *op) {
  DLTIQueryInterface queryable = {};

  // Search op and its ancestors for the first attached DLTIQueryInterface attr.
  do {
    for (NamedAttribute attr : op->getAttrs())
      if ((queryable = dyn_cast<DLTIQueryInterface>(attr.getValue())))
        break;
  } while (!queryable && (op = op->getParentOp()));

  return std::pair(queryable, op);
}

FailureOr<Attribute>
dlti::query(Operation *op, ArrayRef<DataLayoutEntryKey> keys, bool emitError) {
  if (!op)
    return failure();

  if (keys.empty()) {
    if (emitError) {
      auto diag = op->emitError() << "target op of failed DLTI query";
      diag.attachNote(op->getLoc()) << "no keys provided to attempt query with";
    }
    return failure();
  }

  auto [queryable, queryOp] = getClosestQueryable(op);
  Operation *reportOp = (queryOp ? queryOp : op);

  if (!queryable) {
    if (emitError) {
      auto diag = op->emitError() << "target op of failed DLTI query";
      diag.attachNote(reportOp->getLoc())
          << "no DLTI-queryable attrs on target op or any of its ancestors";
    }
    return failure();
  }

  Attribute currentAttr = queryable;
  for (auto &&[idx, key] : llvm::enumerate(keys)) {
    if (auto map = dyn_cast<DLTIQueryInterface>(currentAttr)) {
      auto maybeAttr = map.query(key);
      if (failed(maybeAttr)) {
        if (emitError) {
          auto diag = op->emitError() << "target op of failed DLTI query";
          diag.attachNote(reportOp->getLoc())
              << "key " << keyToStr(key)
              << " has no DLTI-mapping per attr: " << map;
        }
        return failure();
      }
      currentAttr = *maybeAttr;
    } else {
      if (emitError) {
        std::string commaSeparatedKeys;
        llvm::interleave(
            keys.take_front(idx), // All prior keys.
            [&](auto key) { commaSeparatedKeys += keyToStr(key); },
            [&]() { commaSeparatedKeys += ","; });

        auto diag = op->emitError() << "target op of failed DLTI query";
        diag.attachNote(reportOp->getLoc())
            << "got non-DLTI-queryable attribute upon looking up keys ["
            << commaSeparatedKeys << "] at op";
      }
      return failure();
    }
  }

  return currentAttr;
}

FailureOr<Attribute> dlti::query(Operation *op, ArrayRef<StringRef> keys,
                                 bool emitError) {
  if (!op)
    return failure();

  MLIRContext *ctx = op->getContext();
  SmallVector<DataLayoutEntryKey> entryKeys;
  entryKeys.reserve(keys.size());
  for (StringRef key : keys)
    entryKeys.push_back(StringAttr::get(ctx, key));

  return dlti::query(op, entryKeys, emitError);
}

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
    StringRef entryName = cast<StringAttr>(entry.getKey()).strref();
    if (entryName == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = dyn_cast<StringAttr>(entry.getValue());
      if (value &&
          (value.getValue() == DLTIDialect::kDataLayoutEndiannessBig ||
           value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle))
        return success();
      return emitError(loc) << "'" << entryName
                            << "' data layout entry is expected to be either '"
                            << DLTIDialect::kDataLayoutEndiannessBig << "' or '"
                            << DLTIDialect::kDataLayoutEndiannessLittle << "'";
    }
    if (entryName == DLTIDialect::kDataLayoutDefaultMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutAllocaMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutProgramMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutGlobalMemorySpaceKey ||
        entryName == DLTIDialect::kDataLayoutStackAlignmentKey ||
        entryName == DLTIDialect::kDataLayoutFunctionPointerAlignmentKey ||
        entryName == DLTIDialect::kDataLayoutManglingModeKey)
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
  }

  if (attr.getName() == DLTIDialect::kTargetSystemDescAttrName) {
    if (!llvm::isa<TargetSystemSpecAttr>(attr.getValue())) {
      return op->emitError()
             << "'" << DLTIDialect::kTargetSystemDescAttrName
             << "' is expected to be a #dlti.target_system_spec attribute";
    }
    return success();
  }

  if (attr.getName() == DLTIDialect::kMapAttrName) {
    if (!llvm::isa<MapAttr>(attr.getValue())) {
      return op->emitError() << "'" << DLTIDialect::kMapAttrName
                             << "' is expected to be a #dlti.map attribute";
    }
    return success();
  }

  return op->emitError() << "attribute '" << attr.getName().getValue()
                         << "' not supported by dialect";
}
