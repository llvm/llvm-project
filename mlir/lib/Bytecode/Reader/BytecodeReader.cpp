//===- BytecodeReader.cpp - MLIR Bytecode Reader --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Support for big-endian architectures.
// TODO: Properly preserve use lists of values.

#include "mlir/Bytecode/BytecodeReader.h"
#include "../Encoding.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

#define DEBUG_TYPE "mlir-bytecode-reader"

using namespace mlir;

/// Stringify the given section ID.
static std::string toString(bytecode::Section::ID sectionID) {
  switch (sectionID) {
  case bytecode::Section::kString:
    return "String (0)";
  case bytecode::Section::kDialect:
    return "Dialect (1)";
  case bytecode::Section::kAttrType:
    return "AttrType (2)";
  case bytecode::Section::kAttrTypeOffset:
    return "AttrTypeOffset (3)";
  case bytecode::Section::kIR:
    return "IR (4)";
  case bytecode::Section::kResource:
    return "Resource (5)";
  case bytecode::Section::kResourceOffset:
    return "ResourceOffset (6)";
  default:
    return ("Unknown (" + Twine(static_cast<unsigned>(sectionID)) + ")").str();
  }
}

/// Returns true if the given top-level section ID is optional.
static bool isSectionOptional(bytecode::Section::ID sectionID) {
  switch (sectionID) {
  case bytecode::Section::kString:
  case bytecode::Section::kDialect:
  case bytecode::Section::kAttrType:
  case bytecode::Section::kAttrTypeOffset:
  case bytecode::Section::kIR:
    return false;
  case bytecode::Section::kResource:
  case bytecode::Section::kResourceOffset:
    return true;
  default:
    llvm_unreachable("unknown section ID");
  }
}

//===----------------------------------------------------------------------===//
// EncodingReader
//===----------------------------------------------------------------------===//

namespace {
class EncodingReader {
public:
  explicit EncodingReader(ArrayRef<uint8_t> contents, Location fileLoc)
      : dataIt(contents.data()), dataEnd(contents.end()), fileLoc(fileLoc) {}
  explicit EncodingReader(StringRef contents, Location fileLoc)
      : EncodingReader({reinterpret_cast<const uint8_t *>(contents.data()),
                        contents.size()},
                       fileLoc) {}

  /// Returns true if the entire section has been read.
  bool empty() const { return dataIt == dataEnd; }

  /// Returns the remaining size of the bytecode.
  size_t size() const { return dataEnd - dataIt; }

  /// Align the current reader position to the specified alignment.
  LogicalResult alignTo(unsigned alignment) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError("expected alignment to be a power-of-two");

    // Shift the reader position to the next alignment boundary.
    while (uintptr_t(dataIt) & (uintptr_t(alignment) - 1)) {
      uint8_t padding;
      if (failed(parseByte(padding)))
        return failure();
      if (padding != bytecode::kAlignmentByte) {
        return emitError("expected alignment byte (0xCB), but got: '0x" +
                         llvm::utohexstr(padding) + "'");
      }
    }

    // Ensure the data iterator is now aligned. This case is unlikely because we
    // *just* went through the effort to align the data iterator.
    if (LLVM_UNLIKELY(!llvm::isAddrAligned(llvm::Align(alignment), dataIt))) {
      return emitError("expected data iterator aligned to ", alignment,
                       ", but got pointer: '0x" +
                           llvm::utohexstr((uintptr_t)dataIt) + "'");
    }

    return success();
  }

  /// Emit an error using the given arguments.
  template <typename... Args>
  InFlightDiagnostic emitError(Args &&...args) const {
    return ::emitError(fileLoc).append(std::forward<Args>(args)...);
  }
  InFlightDiagnostic emitError() const { return ::emitError(fileLoc); }

  /// Parse a single byte from the stream.
  template <typename T>
  LogicalResult parseByte(T &value) {
    if (empty())
      return emitError("attempting to parse a byte at the end of the bytecode");
    value = static_cast<T>(*dataIt++);
    return success();
  }
  /// Parse a range of bytes of 'length' into the given result.
  LogicalResult parseBytes(size_t length, ArrayRef<uint8_t> &result) {
    if (length > size()) {
      return emitError("attempting to parse ", length, " bytes when only ",
                       size(), " remain");
    }
    result = {dataIt, length};
    dataIt += length;
    return success();
  }
  /// Parse a range of bytes of 'length' into the given result, which can be
  /// assumed to be large enough to hold `length`.
  LogicalResult parseBytes(size_t length, uint8_t *result) {
    if (length > size()) {
      return emitError("attempting to parse ", length, " bytes when only ",
                       size(), " remain");
    }
    memcpy(result, dataIt, length);
    dataIt += length;
    return success();
  }

  /// Parse an aligned blob of data, where the alignment was encoded alongside
  /// the data.
  LogicalResult parseBlobAndAlignment(ArrayRef<uint8_t> &data,
                                      uint64_t &alignment) {
    uint64_t dataSize;
    if (failed(parseVarInt(alignment)) || failed(parseVarInt(dataSize)) ||
        failed(alignTo(alignment)))
      return failure();
    return parseBytes(dataSize, data);
  }

  /// Parse a variable length encoded integer from the byte stream. The first
  /// encoded byte contains a prefix in the low bits indicating the encoded
  /// length of the value. This length prefix is a bit sequence of '0's followed
  /// by a '1'. The number of '0' bits indicate the number of _additional_ bytes
  /// (not including the prefix byte). All remaining bits in the first byte,
  /// along with all of the bits in additional bytes, provide the value of the
  /// integer encoded in little-endian order.
  LogicalResult parseVarInt(uint64_t &result) {
    // Parse the first byte of the encoding, which contains the length prefix.
    if (failed(parseByte(result)))
      return failure();

    // Handle the overwhelmingly common case where the value is stored in a
    // single byte. In this case, the first bit is the `1` marker bit.
    if (LLVM_LIKELY(result & 1)) {
      result >>= 1;
      return success();
    }

    // Handle the overwhelming uncommon case where the value required all 8
    // bytes (i.e. a really really big number). In this case, the marker byte is
    // all zeros: `00000000`.
    if (LLVM_UNLIKELY(result == 0))
      return parseBytes(sizeof(result), reinterpret_cast<uint8_t *>(&result));
    return parseMultiByteVarInt(result);
  }

  /// Parse a signed variable length encoded integer from the byte stream. A
  /// signed varint is encoded as a normal varint with zigzag encoding applied,
  /// i.e. the low bit of the value is used to indicate the sign.
  LogicalResult parseSignedVarInt(uint64_t &result) {
    if (failed(parseVarInt(result)))
      return failure();
    // Essentially (but using unsigned): (x >> 1) ^ -(x & 1)
    result = (result >> 1) ^ (~(result & 1) + 1);
    return success();
  }

  /// Parse a variable length encoded integer whose low bit is used to encode an
  /// unrelated flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
  LogicalResult parseVarIntWithFlag(uint64_t &result, bool &flag) {
    if (failed(parseVarInt(result)))
      return failure();
    flag = result & 1;
    result >>= 1;
    return success();
  }

  /// Skip the first `length` bytes within the reader.
  LogicalResult skipBytes(size_t length) {
    if (length > size()) {
      return emitError("attempting to skip ", length, " bytes when only ",
                       size(), " remain");
    }
    dataIt += length;
    return success();
  }

  /// Parse a null-terminated string into `result` (without including the NUL
  /// terminator).
  LogicalResult parseNullTerminatedString(StringRef &result) {
    const char *startIt = (const char *)dataIt;
    const char *nulIt = (const char *)memchr(startIt, 0, size());
    if (!nulIt)
      return emitError(
          "malformed null-terminated string, no null character found");

    result = StringRef(startIt, nulIt - startIt);
    dataIt = (const uint8_t *)nulIt + 1;
    return success();
  }

  /// Parse a section header, placing the kind of section in `sectionID` and the
  /// contents of the section in `sectionData`.
  LogicalResult parseSection(bytecode::Section::ID &sectionID,
                             ArrayRef<uint8_t> &sectionData) {
    uint8_t sectionIDAndHasAlignment;
    uint64_t length;
    if (failed(parseByte(sectionIDAndHasAlignment)) ||
        failed(parseVarInt(length)))
      return failure();

    // Extract the section ID and whether the section is aligned. The high bit
    // of the ID is the alignment flag.
    sectionID = static_cast<bytecode::Section::ID>(sectionIDAndHasAlignment &
                                                   0b01111111);
    bool hasAlignment = sectionIDAndHasAlignment & 0b10000000;

    // Check that the section is actually valid before trying to process its
    // data.
    if (sectionID >= bytecode::Section::kNumSections)
      return emitError("invalid section ID: ", unsigned(sectionID));

    // Process the section alignment if present.
    if (hasAlignment) {
      uint64_t alignment;
      if (failed(parseVarInt(alignment)) || failed(alignTo(alignment)))
        return failure();
    }

    // Parse the actual section data.
    return parseBytes(static_cast<size_t>(length), sectionData);
  }

private:
  /// Parse a variable length encoded integer from the byte stream. This method
  /// is a fallback when the number of bytes used to encode the value is greater
  /// than 1, but less than the max (9). The provided `result` value can be
  /// assumed to already contain the first byte of the value.
  /// NOTE: This method is marked noinline to avoid pessimizing the common case
  /// of single byte encoding.
  LLVM_ATTRIBUTE_NOINLINE LogicalResult parseMultiByteVarInt(uint64_t &result) {
    // Count the number of trailing zeros in the marker byte, this indicates the
    // number of trailing bytes that are part of the value. We use `uint32_t`
    // here because we only care about the first byte, and so that be actually
    // get ctz intrinsic calls when possible (the `uint8_t` overload uses a loop
    // implementation).
    uint32_t numBytes = llvm::countr_zero<uint32_t>(result);
    assert(numBytes > 0 && numBytes <= 7 &&
           "unexpected number of trailing zeros in varint encoding");

    // Parse in the remaining bytes of the value.
    if (failed(parseBytes(numBytes, reinterpret_cast<uint8_t *>(&result) + 1)))
      return failure();

    // Shift out the low-order bits that were used to mark how the value was
    // encoded.
    result >>= (numBytes + 1);
    return success();
  }

  /// The current data iterator, and an iterator to the end of the buffer.
  const uint8_t *dataIt, *dataEnd;

  /// A location for the bytecode used to report errors.
  Location fileLoc;
};
} // namespace

/// Resolve an index into the given entry list. `entry` may either be a
/// reference, in which case it is assigned to the corresponding value in
/// `entries`, or a pointer, in which case it is assigned to the address of the
/// element in `entries`.
template <typename RangeT, typename T>
static LogicalResult resolveEntry(EncodingReader &reader, RangeT &entries,
                                  uint64_t index, T &entry,
                                  StringRef entryStr) {
  if (index >= entries.size())
    return reader.emitError("invalid ", entryStr, " index: ", index);

  // If the provided entry is a pointer, resolve to the address of the entry.
  if constexpr (std::is_convertible_v<llvm::detail::ValueOfRange<RangeT>, T>)
    entry = entries[index];
  else
    entry = &entries[index];
  return success();
}

/// Parse and resolve an index into the given entry list.
template <typename RangeT, typename T>
static LogicalResult parseEntry(EncodingReader &reader, RangeT &entries,
                                T &entry, StringRef entryStr) {
  uint64_t entryIdx;
  if (failed(reader.parseVarInt(entryIdx)))
    return failure();
  return resolveEntry(reader, entries, entryIdx, entry, entryStr);
}

//===----------------------------------------------------------------------===//
// StringSectionReader
//===----------------------------------------------------------------------===//

namespace {
/// This class is used to read references to the string section from the
/// bytecode.
class StringSectionReader {
public:
  /// Initialize the string section reader with the given section data.
  LogicalResult initialize(Location fileLoc, ArrayRef<uint8_t> sectionData);

  /// Parse a shared string from the string section. The shared string is
  /// encoded using an index to a corresponding string in the string section.
  LogicalResult parseString(EncodingReader &reader, StringRef &result) {
    return parseEntry(reader, strings, result, "string");
  }

private:
  /// The table of strings referenced within the bytecode file.
  SmallVector<StringRef> strings;
};
} // namespace

LogicalResult StringSectionReader::initialize(Location fileLoc,
                                              ArrayRef<uint8_t> sectionData) {
  EncodingReader stringReader(sectionData, fileLoc);

  // Parse the number of strings in the section.
  uint64_t numStrings;
  if (failed(stringReader.parseVarInt(numStrings)))
    return failure();
  strings.resize(numStrings);

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = sectionData.size();
  for (StringRef &string : llvm::reverse(strings)) {
    uint64_t stringSize;
    if (failed(stringReader.parseVarInt(stringSize)))
      return failure();
    if (stringDataEndOffset < stringSize) {
      return stringReader.emitError(
          "string size exceeds the available data size");
    }

    // Extract the string from the data, dropping the null character.
    size_t stringOffset = stringDataEndOffset - stringSize;
    string = StringRef(
        reinterpret_cast<const char *>(sectionData.data() + stringOffset),
        stringSize - 1);
    stringDataEndOffset = stringOffset;
  }

  // Check that the only remaining data was for the strings, i.e. the reader
  // should be at the same offset as the first string.
  if ((sectionData.size() - stringReader.size()) != stringDataEndOffset) {
    return stringReader.emitError("unexpected trailing data between the "
                                  "offsets for strings and their data");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BytecodeDialect
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents a dialect entry within the bytecode.
struct BytecodeDialect {
  /// Load the dialect into the provided context if it hasn't been loaded yet.
  /// Returns failure if the dialect couldn't be loaded *and* the provided
  /// context does not allow unregistered dialects. The provided reader is used
  /// for error emission if necessary.
  LogicalResult load(EncodingReader &reader, MLIRContext *ctx) {
    if (dialect)
      return success();
    Dialect *loadedDialect = ctx->getOrLoadDialect(name);
    if (!loadedDialect && !ctx->allowsUnregisteredDialects()) {
      return reader.emitError(
          "dialect '", name,
          "' is unknown. If this is intended, please call "
          "allowUnregisteredDialects() on the MLIRContext, or use "
          "-allow-unregistered-dialect with the MLIR tool used.");
    }
    dialect = loadedDialect;

    // If the dialect was actually loaded, check to see if it has a bytecode
    // interface.
    if (loadedDialect)
      interface = dyn_cast<BytecodeDialectInterface>(loadedDialect);
    return success();
  }

  /// Return the loaded dialect, or nullptr if the dialect is unknown. This can
  /// only be called after `load`.
  Dialect *getLoadedDialect() const {
    assert(dialect &&
           "expected `load` to be invoked before `getLoadedDialect`");
    return *dialect;
  }

  /// The loaded dialect entry. This field is std::nullopt if we haven't
  /// attempted to load, nullptr if we failed to load, otherwise the loaded
  /// dialect.
  std::optional<Dialect *> dialect;

  /// The bytecode interface of the dialect, or nullptr if the dialect does not
  /// implement the bytecode interface. This field should only be checked if the
  /// `dialect` field is not std::nullopt.
  const BytecodeDialectInterface *interface = nullptr;

  /// The name of the dialect.
  StringRef name;
};

/// This struct represents an operation name entry within the bytecode.
struct BytecodeOperationName {
  BytecodeOperationName(BytecodeDialect *dialect, StringRef name)
      : dialect(dialect), name(name) {}

  /// The loaded operation name, or std::nullopt if it hasn't been processed
  /// yet.
  std::optional<OperationName> opName;

  /// The dialect that owns this operation name.
  BytecodeDialect *dialect;

  /// The name of the operation, without the dialect prefix.
  StringRef name;
};
} // namespace

/// Parse a single dialect group encoded in the byte stream.
static LogicalResult parseDialectGrouping(
    EncodingReader &reader, MutableArrayRef<BytecodeDialect> dialects,
    function_ref<LogicalResult(BytecodeDialect *)> entryCallback) {
  // Parse the dialect and the number of entries in the group.
  BytecodeDialect *dialect;
  if (failed(parseEntry(reader, dialects, dialect, "dialect")))
    return failure();
  uint64_t numEntries;
  if (failed(reader.parseVarInt(numEntries)))
    return failure();

  for (uint64_t i = 0; i < numEntries; ++i)
    if (failed(entryCallback(dialect)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ResourceSectionReader
//===----------------------------------------------------------------------===//

namespace {
/// This class is used to read the resource section from the bytecode.
class ResourceSectionReader {
public:
  /// Initialize the resource section reader with the given section data.
  LogicalResult
  initialize(Location fileLoc, const ParserConfig &config,
             MutableArrayRef<BytecodeDialect> dialects,
             StringSectionReader &stringReader, ArrayRef<uint8_t> sectionData,
             ArrayRef<uint8_t> offsetSectionData,
             const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef);

  /// Parse a dialect resource handle from the resource section.
  LogicalResult parseResourceHandle(EncodingReader &reader,
                                    AsmDialectResourceHandle &result) {
    return parseEntry(reader, dialectResources, result, "resource handle");
  }

private:
  /// The table of dialect resources within the bytecode file.
  SmallVector<AsmDialectResourceHandle> dialectResources;
};

class ParsedResourceEntry : public AsmParsedResourceEntry {
public:
  ParsedResourceEntry(StringRef key, AsmResourceEntryKind kind,
                      EncodingReader &reader, StringSectionReader &stringReader,
                      const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : key(key), kind(kind), reader(reader), stringReader(stringReader),
        bufferOwnerRef(bufferOwnerRef) {}
  ~ParsedResourceEntry() override = default;

  StringRef getKey() const final { return key; }

  InFlightDiagnostic emitError() const final { return reader.emitError(); }

  AsmResourceEntryKind getKind() const final { return kind; }

  FailureOr<bool> parseAsBool() const final {
    if (kind != AsmResourceEntryKind::Bool)
      return emitError() << "expected a bool resource entry, but found a "
                         << toString(kind) << " entry instead";

    bool value;
    if (failed(reader.parseByte(value)))
      return failure();
    return value;
  }
  FailureOr<std::string> parseAsString() const final {
    if (kind != AsmResourceEntryKind::String)
      return emitError() << "expected a string resource entry, but found a "
                         << toString(kind) << " entry instead";

    StringRef string;
    if (failed(stringReader.parseString(reader, string)))
      return failure();
    return string.str();
  }

  FailureOr<AsmResourceBlob>
  parseAsBlob(BlobAllocatorFn allocator) const final {
    if (kind != AsmResourceEntryKind::Blob)
      return emitError() << "expected a blob resource entry, but found a "
                         << toString(kind) << " entry instead";

    ArrayRef<uint8_t> data;
    uint64_t alignment;
    if (failed(reader.parseBlobAndAlignment(data, alignment)))
      return failure();

    // If we have an extendable reference to the buffer owner, we don't need to
    // allocate a new buffer for the data, and can use the data directly.
    if (bufferOwnerRef) {
      ArrayRef<char> charData(reinterpret_cast<const char *>(data.data()),
                              data.size());

      // Allocate an unmanager buffer which captures a reference to the owner.
      // For now we just mark this as immutable, but in the future we should
      // explore marking this as mutable when desired.
      return UnmanagedAsmResourceBlob::allocateWithAlign(
          charData, alignment,
          [bufferOwnerRef = bufferOwnerRef](void *, size_t, size_t) {});
    }

    // Allocate memory for the blob using the provided allocator and copy the
    // data into it.
    AsmResourceBlob blob = allocator(data.size(), alignment);
    assert(llvm::isAddrAligned(llvm::Align(alignment), blob.getData().data()) &&
           blob.isMutable() &&
           "blob allocator did not return a properly aligned address");
    memcpy(blob.getMutableData().data(), data.data(), data.size());
    return blob;
  }

private:
  StringRef key;
  AsmResourceEntryKind kind;
  EncodingReader &reader;
  StringSectionReader &stringReader;
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
};
} // namespace

template <typename T>
static LogicalResult
parseResourceGroup(Location fileLoc, bool allowEmpty,
                   EncodingReader &offsetReader, EncodingReader &resourceReader,
                   StringSectionReader &stringReader, T *handler,
                   const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef,
                   function_ref<LogicalResult(StringRef)> processKeyFn = {}) {
  uint64_t numResources;
  if (failed(offsetReader.parseVarInt(numResources)))
    return failure();

  for (uint64_t i = 0; i < numResources; ++i) {
    StringRef key;
    AsmResourceEntryKind kind;
    uint64_t resourceOffset;
    ArrayRef<uint8_t> data;
    if (failed(stringReader.parseString(offsetReader, key)) ||
        failed(offsetReader.parseVarInt(resourceOffset)) ||
        failed(offsetReader.parseByte(kind)) ||
        failed(resourceReader.parseBytes(resourceOffset, data)))
      return failure();

    // Process the resource key.
    if ((processKeyFn && failed(processKeyFn(key))))
      return failure();

    // If the resource data is empty and we allow it, don't error out when
    // parsing below, just skip it.
    if (allowEmpty && data.empty())
      continue;

    // Ignore the entry if we don't have a valid handler.
    if (!handler)
      continue;

    // Otherwise, parse the resource value.
    EncodingReader entryReader(data, fileLoc);
    ParsedResourceEntry entry(key, kind, entryReader, stringReader,
                              bufferOwnerRef);
    if (failed(handler->parseResource(entry)))
      return failure();
    if (!entryReader.empty()) {
      return entryReader.emitError(
          "unexpected trailing bytes in resource entry '", key, "'");
    }
  }
  return success();
}

LogicalResult ResourceSectionReader::initialize(
    Location fileLoc, const ParserConfig &config,
    MutableArrayRef<BytecodeDialect> dialects,
    StringSectionReader &stringReader, ArrayRef<uint8_t> sectionData,
    ArrayRef<uint8_t> offsetSectionData,
    const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef) {
  EncodingReader resourceReader(sectionData, fileLoc);
  EncodingReader offsetReader(offsetSectionData, fileLoc);

  // Read the number of external resource providers.
  uint64_t numExternalResourceGroups;
  if (failed(offsetReader.parseVarInt(numExternalResourceGroups)))
    return failure();

  // Utility functor that dispatches to `parseResourceGroup`, but implicitly
  // provides most of the arguments.
  auto parseGroup = [&](auto *handler, bool allowEmpty = false,
                        function_ref<LogicalResult(StringRef)> keyFn = {}) {
    return parseResourceGroup(fileLoc, allowEmpty, offsetReader, resourceReader,
                              stringReader, handler, bufferOwnerRef, keyFn);
  };

  // Read the external resources from the bytecode.
  for (uint64_t i = 0; i < numExternalResourceGroups; ++i) {
    StringRef key;
    if (failed(stringReader.parseString(offsetReader, key)))
      return failure();

    // Get the handler for these resources.
    // TODO: Should we require handling external resources in some scenarios?
    AsmResourceParser *handler = config.getResourceParser(key);
    if (!handler) {
      emitWarning(fileLoc) << "ignoring unknown external resources for '" << key
                           << "'";
    }

    if (failed(parseGroup(handler)))
      return failure();
  }

  // Read the dialect resources from the bytecode.
  MLIRContext *ctx = fileLoc->getContext();
  while (!offsetReader.empty()) {
    BytecodeDialect *dialect;
    if (failed(parseEntry(offsetReader, dialects, dialect, "dialect")) ||
        failed(dialect->load(resourceReader, ctx)))
      return failure();
    Dialect *loadedDialect = dialect->getLoadedDialect();
    if (!loadedDialect) {
      return resourceReader.emitError()
             << "dialect '" << dialect->name << "' is unknown";
    }
    const auto *handler = dyn_cast<OpAsmDialectInterface>(loadedDialect);
    if (!handler) {
      return resourceReader.emitError()
             << "unexpected resources for dialect '" << dialect->name << "'";
    }

    // Ensure that each resource is declared before being processed.
    auto processResourceKeyFn = [&](StringRef key) -> LogicalResult {
      FailureOr<AsmDialectResourceHandle> handle =
          handler->declareResource(key);
      if (failed(handle)) {
        return resourceReader.emitError()
               << "unknown 'resource' key '" << key << "' for dialect '"
               << dialect->name << "'";
      }
      dialectResources.push_back(*handle);
      return success();
    };

    // Parse the resources for this dialect. We allow empty resources because we
    // just treat these as declarations.
    if (failed(parseGroup(handler, /*allowEmpty=*/true, processResourceKeyFn)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Attribute/Type Reader
//===----------------------------------------------------------------------===//

namespace {
/// This class provides support for reading attribute and type entries from the
/// bytecode. Attribute and Type entries are read lazily on demand, so we use
/// this reader to manage when to actually parse them from the bytecode.
class AttrTypeReader {
  /// This class represents a single attribute or type entry.
  template <typename T>
  struct Entry {
    /// The entry, or null if it hasn't been resolved yet.
    T entry = {};
    /// The parent dialect of this entry.
    BytecodeDialect *dialect = nullptr;
    /// A flag indicating if the entry was encoded using a custom encoding,
    /// instead of using the textual assembly format.
    bool hasCustomEncoding = false;
    /// The raw data of this entry in the bytecode.
    ArrayRef<uint8_t> data;
  };
  using AttrEntry = Entry<Attribute>;
  using TypeEntry = Entry<Type>;

public:
  AttrTypeReader(StringSectionReader &stringReader,
                 ResourceSectionReader &resourceReader, Location fileLoc)
      : stringReader(stringReader), resourceReader(resourceReader),
        fileLoc(fileLoc) {}

  /// Initialize the attribute and type information within the reader.
  LogicalResult initialize(MutableArrayRef<BytecodeDialect> dialects,
                           ArrayRef<uint8_t> sectionData,
                           ArrayRef<uint8_t> offsetSectionData);

  /// Resolve the attribute or type at the given index. Returns nullptr on
  /// failure.
  Attribute resolveAttribute(size_t index) {
    return resolveEntry(attributes, index, "Attribute");
  }
  Type resolveType(size_t index) { return resolveEntry(types, index, "Type"); }

  /// Parse a reference to an attribute or type using the given reader.
  LogicalResult parseAttribute(EncodingReader &reader, Attribute &result) {
    uint64_t attrIdx;
    if (failed(reader.parseVarInt(attrIdx)))
      return failure();
    result = resolveAttribute(attrIdx);
    return success(!!result);
  }
  LogicalResult parseType(EncodingReader &reader, Type &result) {
    uint64_t typeIdx;
    if (failed(reader.parseVarInt(typeIdx)))
      return failure();
    result = resolveType(typeIdx);
    return success(!!result);
  }

  template <typename T>
  LogicalResult parseAttribute(EncodingReader &reader, T &result) {
    Attribute baseResult;
    if (failed(parseAttribute(reader, baseResult)))
      return failure();
    if ((result = baseResult.dyn_cast<T>()))
      return success();
    return reader.emitError("expected attribute of type: ",
                            llvm::getTypeName<T>(), ", but got: ", baseResult);
  }

private:
  /// Resolve the given entry at `index`.
  template <typename T>
  T resolveEntry(SmallVectorImpl<Entry<T>> &entries, size_t index,
                 StringRef entryType);

  /// Parse an entry using the given reader that was encoded using the textual
  /// assembly format.
  template <typename T>
  LogicalResult parseAsmEntry(T &result, EncodingReader &reader,
                              StringRef entryType);

  /// Parse an entry using the given reader that was encoded using a custom
  /// bytecode format.
  template <typename T>
  LogicalResult parseCustomEntry(Entry<T> &entry, EncodingReader &reader,
                                 StringRef entryType);

  /// The string section reader used to resolve string references when parsing
  /// custom encoded attribute/type entries.
  StringSectionReader &stringReader;

  /// The resource section reader used to resolve resource references when
  /// parsing custom encoded attribute/type entries.
  ResourceSectionReader &resourceReader;

  /// The set of attribute and type entries.
  SmallVector<AttrEntry> attributes;
  SmallVector<TypeEntry> types;

  /// A location used for error emission.
  Location fileLoc;
};

class DialectReader : public DialectBytecodeReader {
public:
  DialectReader(AttrTypeReader &attrTypeReader,
                StringSectionReader &stringReader,
                ResourceSectionReader &resourceReader, EncodingReader &reader)
      : attrTypeReader(attrTypeReader), stringReader(stringReader),
        resourceReader(resourceReader), reader(reader) {}

  InFlightDiagnostic emitError(const Twine &msg) override {
    return reader.emitError(msg);
  }

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  LogicalResult readAttribute(Attribute &result) override {
    return attrTypeReader.parseAttribute(reader, result);
  }

  LogicalResult readType(Type &result) override {
    return attrTypeReader.parseType(reader, result);
  }

  FailureOr<AsmDialectResourceHandle> readResourceHandle() override {
    AsmDialectResourceHandle handle;
    if (failed(resourceReader.parseResourceHandle(reader, handle)))
      return failure();
    return handle;
  }

  //===--------------------------------------------------------------------===//
  // Primitives
  //===--------------------------------------------------------------------===//

  LogicalResult readVarInt(uint64_t &result) override {
    return reader.parseVarInt(result);
  }

  LogicalResult readSignedVarInt(int64_t &result) override {
    uint64_t unsignedResult;
    if (failed(reader.parseSignedVarInt(unsignedResult)))
      return failure();
    result = static_cast<int64_t>(unsignedResult);
    return success();
  }

  FailureOr<APInt> readAPIntWithKnownWidth(unsigned bitWidth) override {
    // Small values are encoded using a single byte.
    if (bitWidth <= 8) {
      uint8_t value;
      if (failed(reader.parseByte(value)))
        return failure();
      return APInt(bitWidth, value);
    }

    // Large values up to 64 bits are encoded using a single varint.
    if (bitWidth <= 64) {
      uint64_t value;
      if (failed(reader.parseSignedVarInt(value)))
        return failure();
      return APInt(bitWidth, value);
    }

    // Otherwise, for really big values we encode the array of active words in
    // the value.
    uint64_t numActiveWords;
    if (failed(reader.parseVarInt(numActiveWords)))
      return failure();
    SmallVector<uint64_t, 4> words(numActiveWords);
    for (uint64_t i = 0; i < numActiveWords; ++i)
      if (failed(reader.parseSignedVarInt(words[i])))
        return failure();
    return APInt(bitWidth, words);
  }

  FailureOr<APFloat>
  readAPFloatWithKnownSemantics(const llvm::fltSemantics &semantics) override {
    FailureOr<APInt> intVal =
        readAPIntWithKnownWidth(APFloat::getSizeInBits(semantics));
    if (failed(intVal))
      return failure();
    return APFloat(semantics, *intVal);
  }

  LogicalResult readString(StringRef &result) override {
    return stringReader.parseString(reader, result);
  }

  LogicalResult readBlob(ArrayRef<char> &result) override {
    uint64_t dataSize;
    ArrayRef<uint8_t> data;
    if (failed(reader.parseVarInt(dataSize)) ||
        failed(reader.parseBytes(dataSize, data)))
      return failure();
    result = llvm::ArrayRef(reinterpret_cast<const char *>(data.data()),
                            data.size());
    return success();
  }

private:
  AttrTypeReader &attrTypeReader;
  StringSectionReader &stringReader;
  ResourceSectionReader &resourceReader;
  EncodingReader &reader;
};
} // namespace

LogicalResult
AttrTypeReader::initialize(MutableArrayRef<BytecodeDialect> dialects,
                           ArrayRef<uint8_t> sectionData,
                           ArrayRef<uint8_t> offsetSectionData) {
  EncodingReader offsetReader(offsetSectionData, fileLoc);

  // Parse the number of attribute and type entries.
  uint64_t numAttributes, numTypes;
  if (failed(offsetReader.parseVarInt(numAttributes)) ||
      failed(offsetReader.parseVarInt(numTypes)))
    return failure();
  attributes.resize(numAttributes);
  types.resize(numTypes);

  // A functor used to accumulate the offsets for the entries in the given
  // range.
  uint64_t currentOffset = 0;
  auto parseEntries = [&](auto &&range) {
    size_t currentIndex = 0, endIndex = range.size();

    // Parse an individual entry.
    auto parseEntryFn = [&](BytecodeDialect *dialect) -> LogicalResult {
      auto &entry = range[currentIndex++];

      uint64_t entrySize;
      if (failed(offsetReader.parseVarIntWithFlag(entrySize,
                                                  entry.hasCustomEncoding)))
        return failure();

      // Verify that the offset is actually valid.
      if (currentOffset + entrySize > sectionData.size()) {
        return offsetReader.emitError(
            "Attribute or Type entry offset points past the end of section");
      }

      entry.data = sectionData.slice(currentOffset, entrySize);
      entry.dialect = dialect;
      currentOffset += entrySize;
      return success();
    };
    while (currentIndex != endIndex)
      if (failed(parseDialectGrouping(offsetReader, dialects, parseEntryFn)))
        return failure();
    return success();
  };

  // Process each of the attributes, and then the types.
  if (failed(parseEntries(attributes)) || failed(parseEntries(types)))
    return failure();

  // Ensure that we read everything from the section.
  if (!offsetReader.empty()) {
    return offsetReader.emitError(
        "unexpected trailing data in the Attribute/Type offset section");
  }
  return success();
}

template <typename T>
T AttrTypeReader::resolveEntry(SmallVectorImpl<Entry<T>> &entries, size_t index,
                               StringRef entryType) {
  if (index >= entries.size()) {
    emitError(fileLoc) << "invalid " << entryType << " index: " << index;
    return {};
  }

  // If the entry has already been resolved, there is nothing left to do.
  Entry<T> &entry = entries[index];
  if (entry.entry)
    return entry.entry;

  // Parse the entry.
  EncodingReader reader(entry.data, fileLoc);

  // Parse based on how the entry was encoded.
  if (entry.hasCustomEncoding) {
    if (failed(parseCustomEntry(entry, reader, entryType)))
      return T();
  } else if (failed(parseAsmEntry(entry.entry, reader, entryType))) {
    return T();
  }

  if (!reader.empty()) {
    reader.emitError("unexpected trailing bytes after " + entryType + " entry");
    return T();
  }
  return entry.entry;
}

template <typename T>
LogicalResult AttrTypeReader::parseAsmEntry(T &result, EncodingReader &reader,
                                            StringRef entryType) {
  StringRef asmStr;
  if (failed(reader.parseNullTerminatedString(asmStr)))
    return failure();

  // Invoke the MLIR assembly parser to parse the entry text.
  size_t numRead = 0;
  MLIRContext *context = fileLoc->getContext();
  if constexpr (std::is_same_v<T, Type>)
    result = ::parseType(asmStr, context, numRead);
  else
    result = ::parseAttribute(asmStr, context, numRead);
  if (!result)
    return failure();

  // Ensure there weren't dangling characters after the entry.
  if (numRead != asmStr.size()) {
    return reader.emitError("trailing characters found after ", entryType,
                            " assembly format: ", asmStr.drop_front(numRead));
  }
  return success();
}

template <typename T>
LogicalResult AttrTypeReader::parseCustomEntry(Entry<T> &entry,
                                               EncodingReader &reader,
                                               StringRef entryType) {
  if (failed(entry.dialect->load(reader, fileLoc.getContext())))
    return failure();

  // Ensure that the dialect implements the bytecode interface.
  if (!entry.dialect->interface) {
    return reader.emitError("dialect '", entry.dialect->name,
                            "' does not implement the bytecode interface");
  }

  // Ask the dialect to parse the entry.
  DialectReader dialectReader(*this, stringReader, resourceReader, reader);
  if constexpr (std::is_same_v<T, Type>)
    entry.entry = entry.dialect->interface->readType(dialectReader);
  else
    entry.entry = entry.dialect->interface->readAttribute(dialectReader);
  return success(!!entry.entry);
}

//===----------------------------------------------------------------------===//
// Bytecode Reader
//===----------------------------------------------------------------------===//

namespace {
/// This class is used to read a bytecode buffer and translate it into MLIR.
class BytecodeReader {
public:
  BytecodeReader(Location fileLoc, const ParserConfig &config,
                 const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : config(config), fileLoc(fileLoc),
        attrTypeReader(stringReader, resourceReader, fileLoc),
        // Use the builtin unrealized conversion cast operation to represent
        // forward references to values that aren't yet defined.
        forwardRefOpState(UnknownLoc::get(config.getContext()),
                          "builtin.unrealized_conversion_cast", ValueRange(),
                          NoneType::get(config.getContext())),
        bufferOwnerRef(bufferOwnerRef) {}

  /// Read the bytecode defined within `buffer` into the given block.
  LogicalResult read(llvm::MemoryBufferRef buffer, Block *block);

private:
  /// Return the context for this config.
  MLIRContext *getContext() const { return config.getContext(); }

  /// Parse the bytecode version.
  LogicalResult parseVersion(EncodingReader &reader);

  //===--------------------------------------------------------------------===//
  // Dialect Section

  LogicalResult parseDialectSection(ArrayRef<uint8_t> sectionData);

  /// Parse an operation name reference using the given reader.
  FailureOr<OperationName> parseOpName(EncodingReader &reader);

  //===--------------------------------------------------------------------===//
  // Attribute/Type Section

  /// Parse an attribute or type using the given reader.
  template <typename T>
  LogicalResult parseAttribute(EncodingReader &reader, T &result) {
    return attrTypeReader.parseAttribute(reader, result);
  }
  LogicalResult parseType(EncodingReader &reader, Type &result) {
    return attrTypeReader.parseType(reader, result);
  }

  //===--------------------------------------------------------------------===//
  // Resource Section

  LogicalResult
  parseResourceSection(std::optional<ArrayRef<uint8_t>> resourceData,
                       std::optional<ArrayRef<uint8_t>> resourceOffsetData);

  //===--------------------------------------------------------------------===//
  // IR Section

  /// This struct represents the current read state of a range of regions. This
  /// struct is used to enable iterative parsing of regions.
  struct RegionReadState {
    RegionReadState(Operation *op, bool isIsolatedFromAbove)
        : RegionReadState(op->getRegions(), isIsolatedFromAbove) {}
    RegionReadState(MutableArrayRef<Region> regions, bool isIsolatedFromAbove)
        : curRegion(regions.begin()), endRegion(regions.end()),
          isIsolatedFromAbove(isIsolatedFromAbove) {}

    /// The current regions being read.
    MutableArrayRef<Region>::iterator curRegion, endRegion;

    /// The number of values defined immediately within this region.
    unsigned numValues = 0;

    /// The current blocks of the region being read.
    SmallVector<Block *> curBlocks;
    Region::iterator curBlock = {};

    /// The number of operations remaining to be read from the current block
    /// being read.
    uint64_t numOpsRemaining = 0;

    /// A flag indicating if the regions being read are isolated from above.
    bool isIsolatedFromAbove = false;
  };

  LogicalResult parseIRSection(ArrayRef<uint8_t> sectionData, Block *block);
  LogicalResult parseRegions(EncodingReader &reader,
                             std::vector<RegionReadState> &regionStack,
                             RegionReadState &readState);
  FailureOr<Operation *> parseOpWithoutRegions(EncodingReader &reader,
                                               RegionReadState &readState,
                                               bool &isIsolatedFromAbove);

  LogicalResult parseRegion(EncodingReader &reader, RegionReadState &readState);
  LogicalResult parseBlock(EncodingReader &reader, RegionReadState &readState);
  LogicalResult parseBlockArguments(EncodingReader &reader, Block *block);

  //===--------------------------------------------------------------------===//
  // Value Processing

  /// Parse an operand reference using the given reader. Returns nullptr in the
  /// case of failure.
  Value parseOperand(EncodingReader &reader);

  /// Sequentially define the given value range.
  LogicalResult defineValues(EncodingReader &reader, ValueRange values);

  /// Create a value to use for a forward reference.
  Value createForwardRef();

  //===--------------------------------------------------------------------===//
  // Fields

  /// This class represents a single value scope, in which a value scope is
  /// delimited by isolated from above regions.
  struct ValueScope {
    /// Push a new region state onto this scope, reserving enough values for
    /// those defined within the current region of the provided state.
    void push(RegionReadState &readState) {
      nextValueIDs.push_back(values.size());
      values.resize(values.size() + readState.numValues);
    }

    /// Pop the values defined for the current region within the provided region
    /// state.
    void pop(RegionReadState &readState) {
      values.resize(values.size() - readState.numValues);
      nextValueIDs.pop_back();
    }

    /// The set of values defined in this scope.
    std::vector<Value> values;

    /// The ID for the next defined value for each region current being
    /// processed in this scope.
    SmallVector<unsigned, 4> nextValueIDs;
  };

  /// The configuration of the parser.
  const ParserConfig &config;

  /// A location to use when emitting errors.
  Location fileLoc;

  /// The reader used to process attribute and types within the bytecode.
  AttrTypeReader attrTypeReader;

  /// The version of the bytecode being read.
  uint64_t version = 0;

  /// The producer of the bytecode being read.
  StringRef producer;

  /// The table of IR units referenced within the bytecode file.
  SmallVector<BytecodeDialect> dialects;
  SmallVector<BytecodeOperationName> opNames;

  /// The reader used to process resources within the bytecode.
  ResourceSectionReader resourceReader;

  /// The table of strings referenced within the bytecode file.
  StringSectionReader stringReader;

  /// The current set of available IR value scopes.
  std::vector<ValueScope> valueScopes;
  /// A block containing the set of operations defined to create forward
  /// references.
  Block forwardRefOps;
  /// A block containing previously created, and no longer used, forward
  /// reference operations.
  Block openForwardRefOps;
  /// An operation state used when instantiating forward references.
  OperationState forwardRefOpState;

  /// The optional owning source manager, which when present may be used to
  /// extend the lifetime of the input buffer.
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
};
} // namespace

LogicalResult BytecodeReader::read(llvm::MemoryBufferRef buffer, Block *block) {
  EncodingReader reader(buffer.getBuffer(), fileLoc);

  // Skip over the bytecode header, this should have already been checked.
  if (failed(reader.skipBytes(StringRef("ML\xefR").size())))
    return failure();
  // Parse the bytecode version and producer.
  if (failed(parseVersion(reader)) ||
      failed(reader.parseNullTerminatedString(producer)))
    return failure();

  // Add a diagnostic handler that attaches a note that includes the original
  // producer of the bytecode.
  ScopedDiagnosticHandler diagHandler(getContext(), [&](Diagnostic &diag) {
    diag.attachNote() << "in bytecode version " << version
                      << " produced by: " << producer;
    return failure();
  });

  // Parse the raw data for each of the top-level sections of the bytecode.
  std::optional<ArrayRef<uint8_t>>
      sectionDatas[bytecode::Section::kNumSections];
  while (!reader.empty()) {
    // Read the next section from the bytecode.
    bytecode::Section::ID sectionID;
    ArrayRef<uint8_t> sectionData;
    if (failed(reader.parseSection(sectionID, sectionData)))
      return failure();

    // Check for duplicate sections, we only expect one instance of each.
    if (sectionDatas[sectionID]) {
      return reader.emitError("duplicate top-level section: ",
                              toString(sectionID));
    }
    sectionDatas[sectionID] = sectionData;
  }
  // Check that all of the required sections were found.
  for (int i = 0; i < bytecode::Section::kNumSections; ++i) {
    bytecode::Section::ID sectionID = static_cast<bytecode::Section::ID>(i);
    if (!sectionDatas[i] && !isSectionOptional(sectionID)) {
      return reader.emitError("missing data for top-level section: ",
                              toString(sectionID));
    }
  }

  // Process the string section first.
  if (failed(stringReader.initialize(
          fileLoc, *sectionDatas[bytecode::Section::kString])))
    return failure();

  // Process the dialect section.
  if (failed(parseDialectSection(*sectionDatas[bytecode::Section::kDialect])))
    return failure();

  // Process the resource section if present.
  if (failed(parseResourceSection(
          sectionDatas[bytecode::Section::kResource],
          sectionDatas[bytecode::Section::kResourceOffset])))
    return failure();

  // Process the attribute and type section.
  if (failed(attrTypeReader.initialize(
          dialects, *sectionDatas[bytecode::Section::kAttrType],
          *sectionDatas[bytecode::Section::kAttrTypeOffset])))
    return failure();

  // Finally, process the IR section.
  return parseIRSection(*sectionDatas[bytecode::Section::kIR], block);
}

LogicalResult BytecodeReader::parseVersion(EncodingReader &reader) {
  if (failed(reader.parseVarInt(version)))
    return failure();

  // Validate the bytecode version.
  uint64_t currentVersion = bytecode::kVersion;
  if (version < currentVersion) {
    return reader.emitError("bytecode version ", version,
                            " is older than the current version of ",
                            currentVersion, ", and upgrade is not supported");
  }
  if (version > currentVersion) {
    return reader.emitError("bytecode version ", version,
                            " is newer than the current version ",
                            currentVersion);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect Section

LogicalResult
BytecodeReader::parseDialectSection(ArrayRef<uint8_t> sectionData) {
  EncodingReader sectionReader(sectionData, fileLoc);

  // Parse the number of dialects in the section.
  uint64_t numDialects;
  if (failed(sectionReader.parseVarInt(numDialects)))
    return failure();
  dialects.resize(numDialects);

  // Parse each of the dialects.
  for (uint64_t i = 0; i < numDialects; ++i)
    if (failed(stringReader.parseString(sectionReader, dialects[i].name)))
      return failure();

  // Parse the operation names, which are grouped by dialect.
  auto parseOpName = [&](BytecodeDialect *dialect) {
    StringRef opName;
    if (failed(stringReader.parseString(sectionReader, opName)))
      return failure();
    opNames.emplace_back(dialect, opName);
    return success();
  };
  while (!sectionReader.empty())
    if (failed(parseDialectGrouping(sectionReader, dialects, parseOpName)))
      return failure();
  return success();
}

FailureOr<OperationName> BytecodeReader::parseOpName(EncodingReader &reader) {
  BytecodeOperationName *opName = nullptr;
  if (failed(parseEntry(reader, opNames, opName, "operation name")))
    return failure();

  // Check to see if this operation name has already been resolved. If we
  // haven't, load the dialect and build the operation name.
  if (!opName->opName) {
    if (failed(opName->dialect->load(reader, getContext())))
      return failure();
    opName->opName.emplace((opName->dialect->name + "." + opName->name).str(),
                           getContext());
  }
  return *opName->opName;
}

//===----------------------------------------------------------------------===//
// Resource Section

LogicalResult BytecodeReader::parseResourceSection(
    std::optional<ArrayRef<uint8_t>> resourceData,
    std::optional<ArrayRef<uint8_t>> resourceOffsetData) {
  // Ensure both sections are either present or not.
  if (resourceData.has_value() != resourceOffsetData.has_value()) {
    if (resourceOffsetData)
      return emitError(fileLoc, "unexpected resource offset section when "
                                "resource section is not present");
    return emitError(
        fileLoc,
        "expected resource offset section when resource section is present");
  }

  // If the resource sections are absent, there is nothing to do.
  if (!resourceData)
    return success();

  // Initialize the resource reader with the resource sections.
  return resourceReader.initialize(fileLoc, config, dialects, stringReader,
                                   *resourceData, *resourceOffsetData,
                                   bufferOwnerRef);
}

//===----------------------------------------------------------------------===//
// IR Section

LogicalResult BytecodeReader::parseIRSection(ArrayRef<uint8_t> sectionData,
                                             Block *block) {
  EncodingReader reader(sectionData, fileLoc);

  // A stack of operation regions currently being read from the bytecode.
  std::vector<RegionReadState> regionStack;

  // Parse the top-level block using a temporary module operation.
  OwningOpRef<ModuleOp> moduleOp = ModuleOp::create(fileLoc);
  regionStack.emplace_back(*moduleOp, /*isIsolatedFromAbove=*/true);
  regionStack.back().curBlocks.push_back(moduleOp->getBody());
  regionStack.back().curBlock = regionStack.back().curRegion->begin();
  if (failed(parseBlock(reader, regionStack.back())))
    return failure();
  valueScopes.emplace_back();
  valueScopes.back().push(regionStack.back());

  // Iteratively parse regions until everything has been resolved.
  while (!regionStack.empty())
    if (failed(parseRegions(reader, regionStack, regionStack.back())))
      return failure();
  if (!forwardRefOps.empty()) {
    return reader.emitError(
        "not all forward unresolved forward operand references");
  }

  // Verify that the parsed operations are valid.
  if (config.shouldVerifyAfterParse() && failed(verify(*moduleOp)))
    return failure();

  // Splice the parsed operations over to the provided top-level block.
  auto &parsedOps = moduleOp->getBody()->getOperations();
  auto &destOps = block->getOperations();
  destOps.splice(destOps.end(), parsedOps, parsedOps.begin(), parsedOps.end());
  return success();
}

LogicalResult
BytecodeReader::parseRegions(EncodingReader &reader,
                             std::vector<RegionReadState> &regionStack,
                             RegionReadState &readState) {
  // Read the regions of this operation.
  for (; readState.curRegion != readState.endRegion; ++readState.curRegion) {
    // If the current block hasn't been setup yet, parse the header for this
    // region.
    if (readState.curBlock == Region::iterator()) {
      if (failed(parseRegion(reader, readState)))
        return failure();

      // If the region is empty, there is nothing to more to do.
      if (readState.curRegion->empty())
        continue;
    }

    // Parse the blocks within the region.
    do {
      while (readState.numOpsRemaining--) {
        // Read in the next operation. We don't read its regions directly, we
        // handle those afterwards as necessary.
        bool isIsolatedFromAbove = false;
        FailureOr<Operation *> op =
            parseOpWithoutRegions(reader, readState, isIsolatedFromAbove);
        if (failed(op))
          return failure();

        // If the op has regions, add it to the stack for processing.
        if ((*op)->getNumRegions()) {
          regionStack.emplace_back(*op, isIsolatedFromAbove);

          // If the op is isolated from above, push a new value scope.
          if (isIsolatedFromAbove)
            valueScopes.emplace_back();
          return success();
        }
      }

      // Move to the next block of the region.
      if (++readState.curBlock == readState.curRegion->end())
        break;
      if (failed(parseBlock(reader, readState)))
        return failure();
    } while (true);

    // Reset the current block and any values reserved for this region.
    readState.curBlock = {};
    valueScopes.back().pop(readState);
  }

  // When the regions have been fully parsed, pop them off of the read stack. If
  // the regions were isolated from above, we also pop the last value scope.
  if (readState.isIsolatedFromAbove)
    valueScopes.pop_back();
  regionStack.pop_back();
  return success();
}

FailureOr<Operation *>
BytecodeReader::parseOpWithoutRegions(EncodingReader &reader,
                                      RegionReadState &readState,
                                      bool &isIsolatedFromAbove) {
  // Parse the name of the operation.
  FailureOr<OperationName> opName = parseOpName(reader);
  if (failed(opName))
    return failure();

  // Parse the operation mask, which indicates which components of the operation
  // are present.
  uint8_t opMask;
  if (failed(reader.parseByte(opMask)))
    return failure();

  /// Parse the location.
  LocationAttr opLoc;
  if (failed(parseAttribute(reader, opLoc)))
    return failure();

  // With the location and name resolved, we can start building the operation
  // state.
  OperationState opState(opLoc, *opName);

  // Parse the attributes of the operation.
  if (opMask & bytecode::OpEncodingMask::kHasAttrs) {
    DictionaryAttr dictAttr;
    if (failed(parseAttribute(reader, dictAttr)))
      return failure();
    opState.attributes = dictAttr;
  }

  /// Parse the results of the operation.
  if (opMask & bytecode::OpEncodingMask::kHasResults) {
    uint64_t numResults;
    if (failed(reader.parseVarInt(numResults)))
      return failure();
    opState.types.resize(numResults);
    for (int i = 0, e = numResults; i < e; ++i)
      if (failed(parseType(reader, opState.types[i])))
        return failure();
  }

  /// Parse the operands of the operation.
  if (opMask & bytecode::OpEncodingMask::kHasOperands) {
    uint64_t numOperands;
    if (failed(reader.parseVarInt(numOperands)))
      return failure();
    opState.operands.resize(numOperands);
    for (int i = 0, e = numOperands; i < e; ++i)
      if (!(opState.operands[i] = parseOperand(reader)))
        return failure();
  }

  /// Parse the successors of the operation.
  if (opMask & bytecode::OpEncodingMask::kHasSuccessors) {
    uint64_t numSuccs;
    if (failed(reader.parseVarInt(numSuccs)))
      return failure();
    opState.successors.resize(numSuccs);
    for (int i = 0, e = numSuccs; i < e; ++i) {
      if (failed(parseEntry(reader, readState.curBlocks, opState.successors[i],
                            "successor")))
        return failure();
    }
  }

  /// Parse the regions of the operation.
  if (opMask & bytecode::OpEncodingMask::kHasInlineRegions) {
    uint64_t numRegions;
    if (failed(reader.parseVarIntWithFlag(numRegions, isIsolatedFromAbove)))
      return failure();

    opState.regions.reserve(numRegions);
    for (int i = 0, e = numRegions; i < e; ++i)
      opState.regions.push_back(std::make_unique<Region>());
  }

  // Create the operation at the back of the current block.
  Operation *op = Operation::create(opState);
  readState.curBlock->push_back(op);

  // If the operation had results, update the value references.
  if (op->getNumResults() && failed(defineValues(reader, op->getResults())))
    return failure();

  return op;
}

LogicalResult BytecodeReader::parseRegion(EncodingReader &reader,
                                          RegionReadState &readState) {
  // Parse the number of blocks in the region.
  uint64_t numBlocks;
  if (failed(reader.parseVarInt(numBlocks)))
    return failure();

  // If the region is empty, there is nothing else to do.
  if (numBlocks == 0)
    return success();

  // Parse the number of values defined in this region.
  uint64_t numValues;
  if (failed(reader.parseVarInt(numValues)))
    return failure();
  readState.numValues = numValues;

  // Create the blocks within this region. We do this before processing so that
  // we can rely on the blocks existing when creating operations.
  readState.curBlocks.clear();
  readState.curBlocks.reserve(numBlocks);
  for (uint64_t i = 0; i < numBlocks; ++i) {
    readState.curBlocks.push_back(new Block());
    readState.curRegion->push_back(readState.curBlocks.back());
  }

  // Prepare the current value scope for this region.
  valueScopes.back().push(readState);

  // Parse the entry block of the region.
  readState.curBlock = readState.curRegion->begin();
  return parseBlock(reader, readState);
}

LogicalResult BytecodeReader::parseBlock(EncodingReader &reader,
                                         RegionReadState &readState) {
  bool hasArgs;
  if (failed(reader.parseVarIntWithFlag(readState.numOpsRemaining, hasArgs)))
    return failure();

  // Parse the arguments of the block.
  if (hasArgs && failed(parseBlockArguments(reader, &*readState.curBlock)))
    return failure();

  // We don't parse the operations of the block here, that's done elsewhere.
  return success();
}

LogicalResult BytecodeReader::parseBlockArguments(EncodingReader &reader,
                                                  Block *block) {
  // Parse the value ID for the first argument, and the number of arguments.
  uint64_t numArgs;
  if (failed(reader.parseVarInt(numArgs)))
    return failure();

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  argTypes.reserve(numArgs);
  argLocs.reserve(numArgs);

  while (numArgs--) {
    Type argType;
    LocationAttr argLoc;
    if (failed(parseType(reader, argType)) ||
        failed(parseAttribute(reader, argLoc)))
      return failure();

    argTypes.push_back(argType);
    argLocs.push_back(argLoc);
  }
  block->addArguments(argTypes, argLocs);
  return defineValues(reader, block->getArguments());
}

//===----------------------------------------------------------------------===//
// Value Processing

Value BytecodeReader::parseOperand(EncodingReader &reader) {
  std::vector<Value> &values = valueScopes.back().values;
  Value *value = nullptr;
  if (failed(parseEntry(reader, values, value, "value")))
    return Value();

  // Create a new forward reference if necessary.
  if (!*value)
    *value = createForwardRef();
  return *value;
}

LogicalResult BytecodeReader::defineValues(EncodingReader &reader,
                                           ValueRange newValues) {
  ValueScope &valueScope = valueScopes.back();
  std::vector<Value> &values = valueScope.values;

  unsigned &valueID = valueScope.nextValueIDs.back();
  unsigned valueIDEnd = valueID + newValues.size();
  if (valueIDEnd > values.size()) {
    return reader.emitError(
        "value index range was outside of the expected range for "
        "the parent region, got [",
        valueID, ", ", valueIDEnd, "), but the maximum index was ",
        values.size() - 1);
  }

  // Assign the values and update any forward references.
  for (unsigned i = 0, e = newValues.size(); i != e; ++i, ++valueID) {
    Value newValue = newValues[i];

    // Check to see if a definition for this value already exists.
    if (Value oldValue = std::exchange(values[valueID], newValue)) {
      Operation *forwardRefOp = oldValue.getDefiningOp();

      // Assert that this is a forward reference operation. Given how we compute
      // definition ids (incrementally as we parse), it shouldn't be possible
      // for the value to be defined any other way.
      assert(forwardRefOp && forwardRefOp->getBlock() == &forwardRefOps &&
             "value index was already defined?");

      oldValue.replaceAllUsesWith(newValue);
      forwardRefOp->moveBefore(&openForwardRefOps, openForwardRefOps.end());
    }
  }
  return success();
}

Value BytecodeReader::createForwardRef() {
  // Check for an avaliable existing operation to use. Otherwise, create a new
  // fake operation to use for the reference.
  if (!openForwardRefOps.empty()) {
    Operation *op = &openForwardRefOps.back();
    op->moveBefore(&forwardRefOps, forwardRefOps.end());
  } else {
    forwardRefOps.push_back(Operation::create(forwardRefOpState));
  }
  return forwardRefOps.back().getResult(0);
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

bool mlir::isBytecode(llvm::MemoryBufferRef buffer) {
  return buffer.getBuffer().startswith("ML\xefR");
}

/// Read the bytecode from the provided memory buffer reference.
/// `bufferOwnerRef` if provided is the owning source manager for the buffer,
/// and may be used to extend the lifetime of the buffer.
static LogicalResult
readBytecodeFileImpl(llvm::MemoryBufferRef buffer, Block *block,
                     const ParserConfig &config,
                     const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef) {
  Location sourceFileLoc =
      FileLineColLoc::get(config.getContext(), buffer.getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);
  if (!isBytecode(buffer)) {
    return emitError(sourceFileLoc,
                     "input buffer is not an MLIR bytecode file");
  }

  BytecodeReader reader(sourceFileLoc, config, bufferOwnerRef);
  return reader.read(buffer, block);
}

LogicalResult mlir::readBytecodeFile(llvm::MemoryBufferRef buffer, Block *block,
                                     const ParserConfig &config) {
  return readBytecodeFileImpl(buffer, block, config, /*bufferOwnerRef=*/{});
}
LogicalResult
mlir::readBytecodeFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                       Block *block, const ParserConfig &config) {
  return readBytecodeFileImpl(
      *sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID()), block, config,
      sourceMgr);
}
