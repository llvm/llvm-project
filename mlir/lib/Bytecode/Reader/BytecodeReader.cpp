//===- BytecodeReader.cpp - MLIR Bytecode Reader --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"

#include <cstddef>
#include <list>
#include <memory>
#include <numeric>
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
  case bytecode::Section::kDialectVersions:
    return "DialectVersions (7)";
  case bytecode::Section::kProperties:
    return "Properties (8)";
  default:
    return ("Unknown (" + Twine(static_cast<unsigned>(sectionID)) + ")").str();
  }
}

/// Returns true if the given top-level section ID is optional.
static bool isSectionOptional(bytecode::Section::ID sectionID, int version) {
  switch (sectionID) {
  case bytecode::Section::kString:
  case bytecode::Section::kDialect:
  case bytecode::Section::kAttrType:
  case bytecode::Section::kAttrTypeOffset:
  case bytecode::Section::kIR:
    return false;
  case bytecode::Section::kResource:
  case bytecode::Section::kResourceOffset:
  case bytecode::Section::kDialectVersions:
    return true;
  case bytecode::Section::kProperties:
    return version < bytecode::kNativePropertiesEncoding;
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
      : buffer(contents), dataIt(buffer.begin()), fileLoc(fileLoc) {}
  explicit EncodingReader(StringRef contents, Location fileLoc)
      : EncodingReader({reinterpret_cast<const uint8_t *>(contents.data()),
                        contents.size()},
                       fileLoc) {}

  /// Returns true if the entire section has been read.
  bool empty() const { return dataIt == buffer.end(); }

  /// Returns the remaining size of the bytecode.
  size_t size() const { return buffer.end() - dataIt; }

  /// Align the current reader position to the specified alignment.
  LogicalResult alignTo(unsigned alignment) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError("expected alignment to be a power-of-two");

    auto isUnaligned = [&](const uint8_t *ptr) {
      return ((uintptr_t)ptr & (alignment - 1)) != 0;
    };

    // Ensure the data buffer was sufficiently aligned in the first place.
    if (LLVM_UNLIKELY(isUnaligned(buffer.begin()))) {
      return emitError("expected bytecode buffer to be aligned to ", alignment,
                       ", but got pointer: '0x" +
                           llvm::utohexstr((uintptr_t)buffer.begin()) + "'");
    }

    // Shift the reader position to the next alignment boundary.
    while (isUnaligned(dataIt)) {
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
    if (LLVM_UNLIKELY(isUnaligned(dataIt))) {
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
    if (LLVM_UNLIKELY(result == 0)) {
      llvm::support::ulittle64_t resultLE;
      if (failed(parseBytes(sizeof(resultLE),
                            reinterpret_cast<uint8_t *>(&resultLE))))
        return failure();
      result = resultLE;
      return success();
    }
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

  Location getLoc() const { return fileLoc; }

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
    llvm::support::ulittle64_t resultLE(result);
    if (failed(parseBytes(numBytes, reinterpret_cast<uint8_t *>(&resultLE) + 1)))
      return failure();

    // Shift out the low-order bits that were used to mark how the value was
    // encoded.
    result = resultLE >> (numBytes + 1);
    return success();
  }

  /// The bytecode buffer.
  ArrayRef<uint8_t> buffer;

  /// The current iterator within the 'buffer'.
  const uint8_t *dataIt;

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

  /// Parse a shared string from the string section. The shared string is
  /// encoded using an index to a corresponding string in the string section.
  /// This variant parses a flag compressed with the index.
  LogicalResult parseStringWithFlag(EncodingReader &reader, StringRef &result,
                                    bool &flag) {
    uint64_t entryIdx;
    if (failed(reader.parseVarIntWithFlag(entryIdx, flag)))
      return failure();
    return parseStringAtIndex(reader, entryIdx, result);
  }

  /// Parse a shared string from the string section. The shared string is
  /// encoded using an index to a corresponding string in the string section.
  LogicalResult parseStringAtIndex(EncodingReader &reader, uint64_t index,
                                   StringRef &result) {
    return resolveEntry(reader, strings, index, result, "string");
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
class DialectReader;

/// This struct represents a dialect entry within the bytecode.
struct BytecodeDialect {
  /// Load the dialect into the provided context if it hasn't been loaded yet.
  /// Returns failure if the dialect couldn't be loaded *and* the provided
  /// context does not allow unregistered dialects. The provided reader is used
  /// for error emission if necessary.
  LogicalResult load(const DialectReader &reader, MLIRContext *ctx);

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

  /// A buffer containing the encoding of the dialect version parsed.
  ArrayRef<uint8_t> versionBuffer;

  /// Lazy loaded dialect version from the handle above.
  std::unique_ptr<DialectVersion> loadedVersion;
};

/// This struct represents an operation name entry within the bytecode.
struct BytecodeOperationName {
  BytecodeOperationName(BytecodeDialect *dialect, StringRef name,
                        std::optional<bool> wasRegistered)
      : dialect(dialect), name(name), wasRegistered(wasRegistered) {}

  /// The loaded operation name, or std::nullopt if it hasn't been processed
  /// yet.
  std::optional<OperationName> opName;

  /// The dialect that owns this operation name.
  BytecodeDialect *dialect;

  /// The name of the operation, without the dialect prefix.
  StringRef name;

  /// Whether this operation was registered when the bytecode was produced.
  /// This flag is populated when bytecode version >=kNativePropertiesEncoding.
  std::optional<bool> wasRegistered;
};
} // namespace

/// Parse a single dialect group encoded in the byte stream.
static LogicalResult parseDialectGrouping(
    EncodingReader &reader,
    MutableArrayRef<std::unique_ptr<BytecodeDialect>> dialects,
    function_ref<LogicalResult(BytecodeDialect *)> entryCallback) {
  // Parse the dialect and the number of entries in the group.
  std::unique_ptr<BytecodeDialect> *dialect;
  if (failed(parseEntry(reader, dialects, dialect, "dialect")))
    return failure();
  uint64_t numEntries;
  if (failed(reader.parseVarInt(numEntries)))
    return failure();

  for (uint64_t i = 0; i < numEntries; ++i)
    if (failed(entryCallback(dialect->get())))
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
             MutableArrayRef<std::unique_ptr<BytecodeDialect>> dialects,
             StringSectionReader &stringReader, ArrayRef<uint8_t> sectionData,
             ArrayRef<uint8_t> offsetSectionData, DialectReader &dialectReader,
             const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef);

  /// Parse a dialect resource handle from the resource section.
  LogicalResult parseResourceHandle(EncodingReader &reader,
                                    AsmDialectResourceHandle &result) {
    return parseEntry(reader, dialectResources, result, "resource handle");
  }

private:
  /// The table of dialect resources within the bytecode file.
  SmallVector<AsmDialectResourceHandle> dialectResources;
  llvm::StringMap<std::string> dialectResourceHandleRenamingMap;
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
                   function_ref<StringRef(StringRef)> remapKey = {},
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
    key = remapKey(key);
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
    MutableArrayRef<std::unique_ptr<BytecodeDialect>> dialects,
    StringSectionReader &stringReader, ArrayRef<uint8_t> sectionData,
    ArrayRef<uint8_t> offsetSectionData, DialectReader &dialectReader,
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
    auto resolveKey = [&](StringRef key) -> StringRef {
      auto it = dialectResourceHandleRenamingMap.find(key);
      if (it == dialectResourceHandleRenamingMap.end())
        return "";
      return it->second;
    };

    return parseResourceGroup(fileLoc, allowEmpty, offsetReader, resourceReader,
                              stringReader, handler, bufferOwnerRef, resolveKey,
                              keyFn);
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
    std::unique_ptr<BytecodeDialect> *dialect;
    if (failed(parseEntry(offsetReader, dialects, dialect, "dialect")) ||
        failed((*dialect)->load(dialectReader, ctx)))
      return failure();
    Dialect *loadedDialect = (*dialect)->getLoadedDialect();
    if (!loadedDialect) {
      return resourceReader.emitError()
             << "dialect '" << (*dialect)->name << "' is unknown";
    }
    const auto *handler = dyn_cast<OpAsmDialectInterface>(loadedDialect);
    if (!handler) {
      return resourceReader.emitError()
             << "unexpected resources for dialect '" << (*dialect)->name << "'";
    }

    // Ensure that each resource is declared before being processed.
    auto processResourceKeyFn = [&](StringRef key) -> LogicalResult {
      FailureOr<AsmDialectResourceHandle> handle =
          handler->declareResource(key);
      if (failed(handle)) {
        return resourceReader.emitError()
               << "unknown 'resource' key '" << key << "' for dialect '"
               << (*dialect)->name << "'";
      }
      dialectResourceHandleRenamingMap[key] = handler->getResourceKey(*handle);
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
                 ResourceSectionReader &resourceReader,
                 const llvm::StringMap<BytecodeDialect *> &dialectsMap,
                 uint64_t &bytecodeVersion, Location fileLoc,
                 const ParserConfig &config)
      : stringReader(stringReader), resourceReader(resourceReader),
        dialectsMap(dialectsMap), fileLoc(fileLoc),
        bytecodeVersion(bytecodeVersion), parserConfig(config) {}

  /// Initialize the attribute and type information within the reader.
  LogicalResult
  initialize(MutableArrayRef<std::unique_ptr<BytecodeDialect>> dialects,
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
  LogicalResult parseOptionalAttribute(EncodingReader &reader,
                                       Attribute &result) {
    uint64_t attrIdx;
    bool flag;
    if (failed(reader.parseVarIntWithFlag(attrIdx, flag)))
      return failure();
    if (!flag)
      return success();
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
    if ((result = dyn_cast<T>(baseResult)))
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

  /// The map of the loaded dialects used to retrieve dialect information, such
  /// as the dialect version.
  const llvm::StringMap<BytecodeDialect *> &dialectsMap;

  /// The set of attribute and type entries.
  SmallVector<AttrEntry> attributes;
  SmallVector<TypeEntry> types;

  /// A location used for error emission.
  Location fileLoc;

  /// Current bytecode version being used.
  uint64_t &bytecodeVersion;

  /// Reference to the parser configuration.
  const ParserConfig &parserConfig;
};

class DialectReader : public DialectBytecodeReader {
public:
  DialectReader(AttrTypeReader &attrTypeReader,
                StringSectionReader &stringReader,
                ResourceSectionReader &resourceReader,
                const llvm::StringMap<BytecodeDialect *> &dialectsMap,
                EncodingReader &reader, uint64_t &bytecodeVersion)
      : attrTypeReader(attrTypeReader), stringReader(stringReader),
        resourceReader(resourceReader), dialectsMap(dialectsMap),
        reader(reader), bytecodeVersion(bytecodeVersion) {}

  InFlightDiagnostic emitError(const Twine &msg) const override {
    return reader.emitError(msg);
  }

  FailureOr<const DialectVersion *>
  getDialectVersion(StringRef dialectName) const override {
    // First check if the dialect is available in the map.
    auto dialectEntry = dialectsMap.find(dialectName);
    if (dialectEntry == dialectsMap.end())
      return failure();
    // If the dialect was found, try to load it. This will trigger reading the
    // bytecode version from the version buffer if it wasn't already processed.
    // Return failure if either of those two actions could not be completed.
    if (failed(dialectEntry->getValue()->load(*this, getLoc().getContext())) ||
        dialectEntry->getValue()->loadedVersion == nullptr)
      return failure();
    return dialectEntry->getValue()->loadedVersion.get();
  }

  MLIRContext *getContext() const override { return getLoc().getContext(); }

  uint64_t getBytecodeVersion() const override { return bytecodeVersion; }

  DialectReader withEncodingReader(EncodingReader &encReader) const {
    return DialectReader(attrTypeReader, stringReader, resourceReader,
                         dialectsMap, encReader, bytecodeVersion);
  }

  Location getLoc() const { return reader.getLoc(); }

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  LogicalResult readAttribute(Attribute &result) override {
    return attrTypeReader.parseAttribute(reader, result);
  }
  LogicalResult readOptionalAttribute(Attribute &result) override {
    return attrTypeReader.parseOptionalAttribute(reader, result);
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

  LogicalResult readBool(bool &result) override {
    return reader.parseByte(result);
  }

private:
  AttrTypeReader &attrTypeReader;
  StringSectionReader &stringReader;
  ResourceSectionReader &resourceReader;
  const llvm::StringMap<BytecodeDialect *> &dialectsMap;
  EncodingReader &reader;
  uint64_t &bytecodeVersion;
};

/// Wraps the properties section and handles reading properties out of it.
class PropertiesSectionReader {
public:
  /// Initialize the properties section reader with the given section data.
  LogicalResult initialize(Location fileLoc, ArrayRef<uint8_t> sectionData) {
    if (sectionData.empty())
      return success();
    EncodingReader propReader(sectionData, fileLoc);
    uint64_t count;
    if (failed(propReader.parseVarInt(count)))
      return failure();
    // Parse the raw properties buffer.
    if (failed(propReader.parseBytes(propReader.size(), propertiesBuffers)))
      return failure();

    EncodingReader offsetsReader(propertiesBuffers, fileLoc);
    offsetTable.reserve(count);
    for (auto idx : llvm::seq<int64_t>(0, count)) {
      (void)idx;
      offsetTable.push_back(propertiesBuffers.size() - offsetsReader.size());
      ArrayRef<uint8_t> rawProperties;
      uint64_t dataSize;
      if (failed(offsetsReader.parseVarInt(dataSize)) ||
          failed(offsetsReader.parseBytes(dataSize, rawProperties)))
        return failure();
    }
    if (!offsetsReader.empty())
      return offsetsReader.emitError()
             << "Broken properties section: didn't exhaust the offsets table";
    return success();
  }

  LogicalResult read(Location fileLoc, DialectReader &dialectReader,
                     OperationName *opName, OperationState &opState) {
    uint64_t propertiesIdx;
    if (failed(dialectReader.readVarInt(propertiesIdx)))
      return failure();
    if (propertiesIdx >= offsetTable.size())
      return dialectReader.emitError("Properties idx out-of-bound for ")
             << opName->getStringRef();
    size_t propertiesOffset = offsetTable[propertiesIdx];
    if (propertiesIdx >= propertiesBuffers.size())
      return dialectReader.emitError("Properties offset out-of-bound for ")
             << opName->getStringRef();

    // Acquire the sub-buffer that represent the requested properties.
    ArrayRef<char> rawProperties;
    {
      // "Seek" to the requested offset by getting a new reader with the right
      // sub-buffer.
      EncodingReader reader(propertiesBuffers.drop_front(propertiesOffset),
                            fileLoc);
      // Properties are stored as a sequence of {size + raw_data}.
      if (failed(
              dialectReader.withEncodingReader(reader).readBlob(rawProperties)))
        return failure();
    }
    // Setup a new reader to read from the `rawProperties` sub-buffer.
    EncodingReader reader(
        StringRef(rawProperties.begin(), rawProperties.size()), fileLoc);
    DialectReader propReader = dialectReader.withEncodingReader(reader);

    auto *iface = opName->getInterface<BytecodeOpInterface>();
    if (iface)
      return iface->readProperties(propReader, opState);
    if (opName->isRegistered())
      return propReader.emitError(
                 "has properties but missing BytecodeOpInterface for ")
             << opName->getStringRef();
    // Unregistered op are storing properties as an attribute.
    return propReader.readAttribute(opState.propertiesAttr);
  }

private:
  /// The properties buffer referenced within the bytecode file.
  ArrayRef<uint8_t> propertiesBuffers;

  /// Table of offset in the buffer above.
  SmallVector<int64_t> offsetTable;
};
} // namespace

LogicalResult AttrTypeReader::initialize(
    MutableArrayRef<std::unique_ptr<BytecodeDialect>> dialects,
    ArrayRef<uint8_t> sectionData, ArrayRef<uint8_t> offsetSectionData) {
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
    result =
        ::parseType(asmStr, context, &numRead, /*isKnownNullTerminated=*/true);
  else
    result = ::parseAttribute(asmStr, context, Type(), &numRead,
                              /*isKnownNullTerminated=*/true);
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
  DialectReader dialectReader(*this, stringReader, resourceReader, dialectsMap,
                              reader, bytecodeVersion);
  if (failed(entry.dialect->load(dialectReader, fileLoc.getContext())))
    return failure();

  if constexpr (std::is_same_v<T, Type>) {
    // Try parsing with callbacks first if available.
    for (const auto &callback :
         parserConfig.getBytecodeReaderConfig().getTypeCallbacks()) {
      if (failed(
              callback->read(dialectReader, entry.dialect->name, entry.entry)))
        return failure();
      // Early return if parsing was successful.
      if (!!entry.entry)
        return success();

      // Reset the reader if we failed to parse, so we can fall through the
      // other parsing functions.
      reader = EncodingReader(entry.data, reader.getLoc());
    }
  } else {
    // Try parsing with callbacks first if available.
    for (const auto &callback :
         parserConfig.getBytecodeReaderConfig().getAttributeCallbacks()) {
      if (failed(
              callback->read(dialectReader, entry.dialect->name, entry.entry)))
        return failure();
      // Early return if parsing was successful.
      if (!!entry.entry)
        return success();

      // Reset the reader if we failed to parse, so we can fall through the
      // other parsing functions.
      reader = EncodingReader(entry.data, reader.getLoc());
    }
  }

  // Ensure that the dialect implements the bytecode interface.
  if (!entry.dialect->interface) {
    return reader.emitError("dialect '", entry.dialect->name,
                            "' does not implement the bytecode interface");
  }

  if constexpr (std::is_same_v<T, Type>)
    entry.entry = entry.dialect->interface->readType(dialectReader);
  else
    entry.entry = entry.dialect->interface->readAttribute(dialectReader);

  return success(!!entry.entry);
}

//===----------------------------------------------------------------------===//
// Bytecode Reader
//===----------------------------------------------------------------------===//

/// This class is used to read a bytecode buffer and translate it into MLIR.
class mlir::BytecodeReader::Impl {
  struct RegionReadState;
  using LazyLoadableOpsInfo =
      std::list<std::pair<Operation *, RegionReadState>>;
  using LazyLoadableOpsMap =
      DenseMap<Operation *, LazyLoadableOpsInfo::iterator>;

public:
  Impl(Location fileLoc, const ParserConfig &config, bool lazyLoading,
       llvm::MemoryBufferRef buffer,
       const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : config(config), fileLoc(fileLoc), lazyLoading(lazyLoading),
        attrTypeReader(stringReader, resourceReader, dialectsMap, version,
                       fileLoc, config),
        // Use the builtin unrealized conversion cast operation to represent
        // forward references to values that aren't yet defined.
        forwardRefOpState(UnknownLoc::get(config.getContext()),
                          "builtin.unrealized_conversion_cast", ValueRange(),
                          NoneType::get(config.getContext())),
        buffer(buffer), bufferOwnerRef(bufferOwnerRef) {}

  /// Read the bytecode defined within `buffer` into the given block.
  LogicalResult read(Block *block,
                     llvm::function_ref<bool(Operation *)> lazyOps);

  /// Return the number of ops that haven't been materialized yet.
  int64_t getNumOpsToMaterialize() const { return lazyLoadableOpsMap.size(); }

  bool isMaterializable(Operation *op) { return lazyLoadableOpsMap.count(op); }

  /// Materialize the provided operation, invoke the lazyOpsCallback on every
  /// newly found lazy operation.
  LogicalResult
  materialize(Operation *op,
              llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
    this->lazyOpsCallback = lazyOpsCallback;
    auto resetlazyOpsCallback =
        llvm::make_scope_exit([&] { this->lazyOpsCallback = nullptr; });
    auto it = lazyLoadableOpsMap.find(op);
    assert(it != lazyLoadableOpsMap.end() &&
           "materialize called on non-materializable op");
    return materialize(it);
  }

  /// Materialize all operations.
  LogicalResult materializeAll() {
    while (!lazyLoadableOpsMap.empty()) {
      if (failed(materialize(lazyLoadableOpsMap.begin())))
        return failure();
    }
    return success();
  }

  /// Finalize the lazy-loading by calling back with every op that hasn't been
  /// materialized to let the client decide if the op should be deleted or
  /// materialized. The op is materialized if the callback returns true, deleted
  /// otherwise.
  LogicalResult finalize(function_ref<bool(Operation *)> shouldMaterialize) {
    while (!lazyLoadableOps.empty()) {
      Operation *op = lazyLoadableOps.begin()->first;
      if (shouldMaterialize(op)) {
        if (failed(materialize(lazyLoadableOpsMap.find(op))))
          return failure();
        continue;
      }
      op->dropAllReferences();
      op->erase();
      lazyLoadableOps.pop_front();
      lazyLoadableOpsMap.erase(op);
    }
    return success();
  }

private:
  LogicalResult materialize(LazyLoadableOpsMap::iterator it) {
    assert(it != lazyLoadableOpsMap.end() &&
           "materialize called on non-materializable op");
    valueScopes.emplace_back();
    std::vector<RegionReadState> regionStack;
    regionStack.push_back(std::move(it->getSecond()->second));
    lazyLoadableOps.erase(it->getSecond());
    lazyLoadableOpsMap.erase(it);

    while (!regionStack.empty())
      if (failed(parseRegions(regionStack, regionStack.back())))
        return failure();
    return success();
  }

  /// Return the context for this config.
  MLIRContext *getContext() const { return config.getContext(); }

  /// Parse the bytecode version.
  LogicalResult parseVersion(EncodingReader &reader);

  //===--------------------------------------------------------------------===//
  // Dialect Section

  LogicalResult parseDialectSection(ArrayRef<uint8_t> sectionData);

  /// Parse an operation name reference using the given reader, and set the
  /// `wasRegistered` flag that indicates if the bytecode was produced by a
  /// context where opName was registered.
  FailureOr<OperationName> parseOpName(EncodingReader &reader,
                                       std::optional<bool> &wasRegistered);

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
  parseResourceSection(EncodingReader &reader,
                       std::optional<ArrayRef<uint8_t>> resourceData,
                       std::optional<ArrayRef<uint8_t>> resourceOffsetData);

  //===--------------------------------------------------------------------===//
  // IR Section

  /// This struct represents the current read state of a range of regions. This
  /// struct is used to enable iterative parsing of regions.
  struct RegionReadState {
    RegionReadState(Operation *op, EncodingReader *reader,
                    bool isIsolatedFromAbove)
        : RegionReadState(op->getRegions(), reader, isIsolatedFromAbove) {}
    RegionReadState(MutableArrayRef<Region> regions, EncodingReader *reader,
                    bool isIsolatedFromAbove)
        : curRegion(regions.begin()), endRegion(regions.end()), reader(reader),
          isIsolatedFromAbove(isIsolatedFromAbove) {}

    /// The current regions being read.
    MutableArrayRef<Region>::iterator curRegion, endRegion;
    /// This is the reader to use for this region, this pointer is pointing to
    /// the parent region reader unless the current region is IsolatedFromAbove,
    /// in which case the pointer is pointing to the `owningReader` which is a
    /// section dedicated to the current region.
    EncodingReader *reader;
    std::unique_ptr<EncodingReader> owningReader;

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
  LogicalResult parseRegions(std::vector<RegionReadState> &regionStack,
                             RegionReadState &readState);
  FailureOr<Operation *> parseOpWithoutRegions(EncodingReader &reader,
                                               RegionReadState &readState,
                                               bool &isIsolatedFromAbove);

  LogicalResult parseRegion(RegionReadState &readState);
  LogicalResult parseBlockHeader(EncodingReader &reader,
                                 RegionReadState &readState);
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
  // Use-list order helpers

  /// This struct is a simple storage that contains information required to
  /// reorder the use-list of a value with respect to the pre-order traversal
  /// ordering.
  struct UseListOrderStorage {
    UseListOrderStorage(bool isIndexPairEncoding,
                        SmallVector<unsigned, 4> &&indices)
        : indices(std::move(indices)),
          isIndexPairEncoding(isIndexPairEncoding){};
    /// The vector containing the information required to reorder the
    /// use-list of a value.
    SmallVector<unsigned, 4> indices;

    /// Whether indices represent a pair of type `(src, dst)` or it is a direct
    /// indexing, such as `dst = order[src]`.
    bool isIndexPairEncoding;
  };

  /// Parse use-list order from bytecode for a range of values if available. The
  /// range is expected to be either a block argument or an op result range. On
  /// success, return a map of the position in the range and the use-list order
  /// encoding. The function assumes to know the size of the range it is
  /// processing.
  using UseListMapT = DenseMap<unsigned, UseListOrderStorage>;
  FailureOr<UseListMapT> parseUseListOrderForRange(EncodingReader &reader,
                                                   uint64_t rangeSize);

  /// Shuffle the use-chain according to the order parsed.
  LogicalResult sortUseListOrder(Value value);

  /// Recursively visit all the values defined within topLevelOp and sort the
  /// use-list orders according to the indices parsed.
  LogicalResult processUseLists(Operation *topLevelOp);

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

  /// Flag that indicates if lazyloading is enabled.
  bool lazyLoading;

  /// Keep track of operations that have been lazy loaded (their regions haven't
  /// been materialized), along with the `RegionReadState` that allows to
  /// lazy-load the regions nested under the operation.
  LazyLoadableOpsInfo lazyLoadableOps;
  LazyLoadableOpsMap lazyLoadableOpsMap;
  llvm::function_ref<bool(Operation *)> lazyOpsCallback;

  /// The reader used to process attribute and types within the bytecode.
  AttrTypeReader attrTypeReader;

  /// The version of the bytecode being read.
  uint64_t version = 0;

  /// The producer of the bytecode being read.
  StringRef producer;

  /// The table of IR units referenced within the bytecode file.
  SmallVector<std::unique_ptr<BytecodeDialect>> dialects;
  llvm::StringMap<BytecodeDialect *> dialectsMap;
  SmallVector<BytecodeOperationName> opNames;

  /// The reader used to process resources within the bytecode.
  ResourceSectionReader resourceReader;

  /// Worklist of values with custom use-list orders to process before the end
  /// of the parsing.
  DenseMap<void *, UseListOrderStorage> valueToUseListMap;

  /// The table of strings referenced within the bytecode file.
  StringSectionReader stringReader;

  /// The table of properties referenced by the operation in the bytecode file.
  PropertiesSectionReader propertiesReader;

  /// The current set of available IR value scopes.
  std::vector<ValueScope> valueScopes;

  /// The global pre-order operation ordering.
  DenseMap<Operation *, unsigned> operationIDs;

  /// A block containing the set of operations defined to create forward
  /// references.
  Block forwardRefOps;

  /// A block containing previously created, and no longer used, forward
  /// reference operations.
  Block openForwardRefOps;

  /// An operation state used when instantiating forward references.
  OperationState forwardRefOpState;

  /// Reference to the input buffer.
  llvm::MemoryBufferRef buffer;

  /// The optional owning source manager, which when present may be used to
  /// extend the lifetime of the input buffer.
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
};

LogicalResult BytecodeReader::Impl::read(
    Block *block, llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
  EncodingReader reader(buffer.getBuffer(), fileLoc);
  this->lazyOpsCallback = lazyOpsCallback;
  auto resetlazyOpsCallback =
      llvm::make_scope_exit([&] { this->lazyOpsCallback = nullptr; });

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
                              ::toString(sectionID));
    }
    sectionDatas[sectionID] = sectionData;
  }
  // Check that all of the required sections were found.
  for (int i = 0; i < bytecode::Section::kNumSections; ++i) {
    bytecode::Section::ID sectionID = static_cast<bytecode::Section::ID>(i);
    if (!sectionDatas[i] && !isSectionOptional(sectionID, version)) {
      return reader.emitError("missing data for top-level section: ",
                              ::toString(sectionID));
    }
  }

  // Process the string section first.
  if (failed(stringReader.initialize(
          fileLoc, *sectionDatas[bytecode::Section::kString])))
    return failure();

  // Process the properties section.
  if (sectionDatas[bytecode::Section::kProperties] &&
      failed(propertiesReader.initialize(
          fileLoc, *sectionDatas[bytecode::Section::kProperties])))
    return failure();

  // Process the dialect section.
  if (failed(parseDialectSection(*sectionDatas[bytecode::Section::kDialect])))
    return failure();

  // Process the resource section if present.
  if (failed(parseResourceSection(
          reader, sectionDatas[bytecode::Section::kResource],
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

LogicalResult BytecodeReader::Impl::parseVersion(EncodingReader &reader) {
  if (failed(reader.parseVarInt(version)))
    return failure();

  // Validate the bytecode version.
  uint64_t currentVersion = bytecode::kVersion;
  uint64_t minSupportedVersion = bytecode::kMinSupportedVersion;
  if (version < minSupportedVersion) {
    return reader.emitError("bytecode version ", version,
                            " is older than the current version of ",
                            currentVersion, ", and upgrade is not supported");
  }
  if (version > currentVersion) {
    return reader.emitError("bytecode version ", version,
                            " is newer than the current version ",
                            currentVersion);
  }
  // Override any request to lazy-load if the bytecode version is too old.
  if (version < bytecode::kLazyLoading)
    lazyLoading = false;
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect Section

LogicalResult BytecodeDialect::load(const DialectReader &reader,
                                    MLIRContext *ctx) {
  if (dialect)
    return success();
  Dialect *loadedDialect = ctx->getOrLoadDialect(name);
  if (!loadedDialect && !ctx->allowsUnregisteredDialects()) {
    return reader.emitError("dialect '")
           << name
           << "' is unknown. If this is intended, please call "
              "allowUnregisteredDialects() on the MLIRContext, or use "
              "-allow-unregistered-dialect with the MLIR tool used.";
  }
  dialect = loadedDialect;

  // If the dialect was actually loaded, check to see if it has a bytecode
  // interface.
  if (loadedDialect)
    interface = dyn_cast<BytecodeDialectInterface>(loadedDialect);
  if (!versionBuffer.empty()) {
    if (!interface)
      return reader.emitError("dialect '")
             << name
             << "' does not implement the bytecode interface, "
                "but found a version entry";
    EncodingReader encReader(versionBuffer, reader.getLoc());
    DialectReader versionReader = reader.withEncodingReader(encReader);
    loadedVersion = interface->readVersion(versionReader);
    if (!loadedVersion)
      return failure();
  }
  return success();
}

LogicalResult
BytecodeReader::Impl::parseDialectSection(ArrayRef<uint8_t> sectionData) {
  EncodingReader sectionReader(sectionData, fileLoc);

  // Parse the number of dialects in the section.
  uint64_t numDialects;
  if (failed(sectionReader.parseVarInt(numDialects)))
    return failure();
  dialects.resize(numDialects);

  // Parse each of the dialects.
  for (uint64_t i = 0; i < numDialects; ++i) {
    dialects[i] = std::make_unique<BytecodeDialect>();
    /// Before version kDialectVersioning, there wasn't any versioning available
    /// for dialects, and the entryIdx represent the string itself.
    if (version < bytecode::kDialectVersioning) {
      if (failed(stringReader.parseString(sectionReader, dialects[i]->name)))
        return failure();
      continue;
    }

    // Parse ID representing dialect and version.
    uint64_t dialectNameIdx;
    bool versionAvailable;
    if (failed(sectionReader.parseVarIntWithFlag(dialectNameIdx,
                                                 versionAvailable)))
      return failure();
    if (failed(stringReader.parseStringAtIndex(sectionReader, dialectNameIdx,
                                               dialects[i]->name)))
      return failure();
    if (versionAvailable) {
      bytecode::Section::ID sectionID;
      if (failed(sectionReader.parseSection(sectionID,
                                            dialects[i]->versionBuffer)))
        return failure();
      if (sectionID != bytecode::Section::kDialectVersions) {
        emitError(fileLoc, "expected dialect version section");
        return failure();
      }
    }
    dialectsMap[dialects[i]->name] = dialects[i].get();
  }

  // Parse the operation names, which are grouped by dialect.
  auto parseOpName = [&](BytecodeDialect *dialect) {
    StringRef opName;
    std::optional<bool> wasRegistered;
    // Prior to version kNativePropertiesEncoding, the information about wheter
    // an op was registered or not wasn't encoded.
    if (version < bytecode::kNativePropertiesEncoding) {
      if (failed(stringReader.parseString(sectionReader, opName)))
        return failure();
    } else {
      bool wasRegisteredFlag;
      if (failed(stringReader.parseStringWithFlag(sectionReader, opName,
                                                  wasRegisteredFlag)))
        return failure();
      wasRegistered = wasRegisteredFlag;
    }
    opNames.emplace_back(dialect, opName, wasRegistered);
    return success();
  };
  // Avoid re-allocation in bytecode version >=kElideUnknownBlockArgLocation
  // where the number of ops are known.
  if (version >= bytecode::kElideUnknownBlockArgLocation) {
    uint64_t numOps;
    if (failed(sectionReader.parseVarInt(numOps)))
      return failure();
    opNames.reserve(numOps);
  }
  while (!sectionReader.empty())
    if (failed(parseDialectGrouping(sectionReader, dialects, parseOpName)))
      return failure();
  return success();
}

FailureOr<OperationName>
BytecodeReader::Impl::parseOpName(EncodingReader &reader,
                                  std::optional<bool> &wasRegistered) {
  BytecodeOperationName *opName = nullptr;
  if (failed(parseEntry(reader, opNames, opName, "operation name")))
    return failure();
  wasRegistered = opName->wasRegistered;
  // Check to see if this operation name has already been resolved. If we
  // haven't, load the dialect and build the operation name.
  if (!opName->opName) {
    // Load the dialect and its version.
    DialectReader dialectReader(attrTypeReader, stringReader, resourceReader,
                                dialectsMap, reader, version);
    if (failed(opName->dialect->load(dialectReader, getContext())))
      return failure();
    // If the opName is empty, this is because we use to accept names such as
    // `foo` without any `.` separator. We shouldn't tolerate this in textual
    // format anymore but for now we'll be backward compatible. This can only
    // happen with unregistered dialects.
    if (opName->name.empty()) {
      if (opName->dialect->getLoadedDialect())
        return emitError(fileLoc) << "has an empty opname for dialect '"
                                  << opName->dialect->name << "'\n";

      opName->opName.emplace(opName->dialect->name, getContext());
    } else {
      opName->opName.emplace((opName->dialect->name + "." + opName->name).str(),
                             getContext());
    }
  }
  return *opName->opName;
}

//===----------------------------------------------------------------------===//
// Resource Section

LogicalResult BytecodeReader::Impl::parseResourceSection(
    EncodingReader &reader, std::optional<ArrayRef<uint8_t>> resourceData,
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
  DialectReader dialectReader(attrTypeReader, stringReader, resourceReader,
                              dialectsMap, reader, version);
  return resourceReader.initialize(fileLoc, config, dialects, stringReader,
                                   *resourceData, *resourceOffsetData,
                                   dialectReader, bufferOwnerRef);
}

//===----------------------------------------------------------------------===//
// UseListOrder Helpers

FailureOr<BytecodeReader::Impl::UseListMapT>
BytecodeReader::Impl::parseUseListOrderForRange(EncodingReader &reader,
                                                uint64_t numResults) {
  BytecodeReader::Impl::UseListMapT map;
  uint64_t numValuesToRead = 1;
  if (numResults > 1 && failed(reader.parseVarInt(numValuesToRead)))
    return failure();

  for (size_t valueIdx = 0; valueIdx < numValuesToRead; valueIdx++) {
    uint64_t resultIdx = 0;
    if (numResults > 1 && failed(reader.parseVarInt(resultIdx)))
      return failure();

    uint64_t numValues;
    bool indexPairEncoding;
    if (failed(reader.parseVarIntWithFlag(numValues, indexPairEncoding)))
      return failure();

    SmallVector<unsigned, 4> useListOrders;
    for (size_t idx = 0; idx < numValues; idx++) {
      uint64_t index;
      if (failed(reader.parseVarInt(index)))
        return failure();
      useListOrders.push_back(index);
    }

    // Store in a map the result index
    map.try_emplace(resultIdx, UseListOrderStorage(indexPairEncoding,
                                                   std::move(useListOrders)));
  }

  return map;
}

/// Sorts each use according to the order specified in the use-list parsed. If
/// the custom use-list is not found, this means that the order needs to be
/// consistent with the reverse pre-order walk of the IR. If multiple uses lie
/// on the same operation, the order will follow the reverse operand number
/// ordering.
LogicalResult BytecodeReader::Impl::sortUseListOrder(Value value) {
  // Early return for trivial use-lists.
  if (value.use_empty() || value.hasOneUse())
    return success();

  bool hasIncomingOrder =
      valueToUseListMap.contains(value.getAsOpaquePointer());

  // Compute the current order of the use-list with respect to the global
  // ordering. Detect if the order is already sorted while doing so.
  bool alreadySorted = true;
  auto &firstUse = *value.use_begin();
  uint64_t prevID =
      bytecode::getUseID(firstUse, operationIDs.at(firstUse.getOwner()));
  llvm::SmallVector<std::pair<unsigned, uint64_t>> currentOrder = {{0, prevID}};
  for (auto item : llvm::drop_begin(llvm::enumerate(value.getUses()))) {
    uint64_t currentID = bytecode::getUseID(
        item.value(), operationIDs.at(item.value().getOwner()));
    alreadySorted &= prevID > currentID;
    currentOrder.push_back({item.index(), currentID});
    prevID = currentID;
  }

  // If the order is already sorted, and there wasn't a custom order to apply
  // from the bytecode file, we are done.
  if (alreadySorted && !hasIncomingOrder)
    return success();

  // If not already sorted, sort the indices of the current order by descending
  // useIDs.
  if (!alreadySorted)
    std::sort(
        currentOrder.begin(), currentOrder.end(),
        [](auto elem1, auto elem2) { return elem1.second > elem2.second; });

  if (!hasIncomingOrder) {
    // If the bytecode file did not contain any custom use-list order, it means
    // that the order was descending useID. Hence, shuffle by the first index
    // of the `currentOrder` pair.
    SmallVector<unsigned> shuffle = SmallVector<unsigned>(
        llvm::map_range(currentOrder, [&](auto item) { return item.first; }));
    value.shuffleUseList(shuffle);
    return success();
  }

  // Pull the custom order info from the map.
  UseListOrderStorage customOrder =
      valueToUseListMap.at(value.getAsOpaquePointer());
  SmallVector<unsigned, 4> shuffle = std::move(customOrder.indices);
  uint64_t numUses =
      std::distance(value.getUses().begin(), value.getUses().end());

  // If the encoding was a pair of indices `(src, dst)` for every permutation,
  // reconstruct the shuffle vector for every use. Initialize the shuffle vector
  // as identity, and then apply the mapping encoded in the indices.
  if (customOrder.isIndexPairEncoding) {
    // Return failure if the number of indices was not representing pairs.
    if (shuffle.size() & 1)
      return failure();

    SmallVector<unsigned, 4> newShuffle(numUses);
    size_t idx = 0;
    std::iota(newShuffle.begin(), newShuffle.end(), idx);
    for (idx = 0; idx < shuffle.size(); idx += 2)
      newShuffle[shuffle[idx]] = shuffle[idx + 1];

    shuffle = std::move(newShuffle);
  }

  // Make sure that the indices represent a valid mapping. That is, the sum of
  // all the values needs to be equal to (numUses - 1) * numUses / 2, and no
  // duplicates are allowed in the list.
  DenseSet<unsigned> set;
  uint64_t accumulator = 0;
  for (const auto &elem : shuffle) {
    if (set.contains(elem))
      return failure();
    accumulator += elem;
    set.insert(elem);
  }
  if (numUses != shuffle.size() ||
      accumulator != (((numUses - 1) * numUses) >> 1))
    return failure();

  // Apply the current ordering map onto the shuffle vector to get the final
  // use-list sorting indices before shuffling.
  shuffle = SmallVector<unsigned, 4>(llvm::map_range(
      currentOrder, [&](auto item) { return shuffle[item.first]; }));
  value.shuffleUseList(shuffle);
  return success();
}

LogicalResult BytecodeReader::Impl::processUseLists(Operation *topLevelOp) {
  // Precompute operation IDs according to the pre-order walk of the IR. We
  // can't do this while parsing since parseRegions ordering is not strictly
  // equal to the pre-order walk.
  unsigned operationID = 0;
  topLevelOp->walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) { operationIDs.try_emplace(op, operationID++); });

  auto blockWalk = topLevelOp->walk([this](Block *block) {
    for (auto arg : block->getArguments())
      if (failed(sortUseListOrder(arg)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  auto resultWalk = topLevelOp->walk([this](Operation *op) {
    for (auto result : op->getResults())
      if (failed(sortUseListOrder(result)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return failure(blockWalk.wasInterrupted() || resultWalk.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// IR Section

LogicalResult
BytecodeReader::Impl::parseIRSection(ArrayRef<uint8_t> sectionData,
                                     Block *block) {
  EncodingReader reader(sectionData, fileLoc);

  // A stack of operation regions currently being read from the bytecode.
  std::vector<RegionReadState> regionStack;

  // Parse the top-level block using a temporary module operation.
  OwningOpRef<ModuleOp> moduleOp = ModuleOp::create(fileLoc);
  regionStack.emplace_back(*moduleOp, &reader, /*isIsolatedFromAbove=*/true);
  regionStack.back().curBlocks.push_back(moduleOp->getBody());
  regionStack.back().curBlock = regionStack.back().curRegion->begin();
  if (failed(parseBlockHeader(reader, regionStack.back())))
    return failure();
  valueScopes.emplace_back();
  valueScopes.back().push(regionStack.back());

  // Iteratively parse regions until everything has been resolved.
  while (!regionStack.empty())
    if (failed(parseRegions(regionStack, regionStack.back())))
      return failure();
  if (!forwardRefOps.empty()) {
    return reader.emitError(
        "not all forward unresolved forward operand references");
  }

  // Sort use-lists according to what specified in bytecode.
  if (failed(processUseLists(*moduleOp)))
    return reader.emitError(
        "parsed use-list orders were invalid and could not be applied");

  // Resolve dialect version.
  for (const std::unique_ptr<BytecodeDialect> &byteCodeDialect : dialects) {
    // Parsing is complete, give an opportunity to each dialect to visit the
    // IR and perform upgrades.
    if (!byteCodeDialect->loadedVersion)
      continue;
    if (byteCodeDialect->interface &&
        failed(byteCodeDialect->interface->upgradeFromVersion(
            *moduleOp, *byteCodeDialect->loadedVersion)))
      return failure();
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
BytecodeReader::Impl::parseRegions(std::vector<RegionReadState> &regionStack,
                                   RegionReadState &readState) {
  // Process regions, blocks, and operations until the end or if a nested
  // region is encountered. In this case we push a new state in regionStack and
  // return, the processing of the current region will resume afterward.
  for (; readState.curRegion != readState.endRegion; ++readState.curRegion) {
    // If the current block hasn't been setup yet, parse the header for this
    // region. The current block is already setup when this function was
    // interrupted to recurse down in a nested region and we resume the current
    // block after processing the nested region.
    if (readState.curBlock == Region::iterator()) {
      if (failed(parseRegion(readState)))
        return failure();

      // If the region is empty, there is nothing to more to do.
      if (readState.curRegion->empty())
        continue;
    }

    // Parse the blocks within the region.
    EncodingReader &reader = *readState.reader;
    do {
      while (readState.numOpsRemaining--) {
        // Read in the next operation. We don't read its regions directly, we
        // handle those afterwards as necessary.
        bool isIsolatedFromAbove = false;
        FailureOr<Operation *> op =
            parseOpWithoutRegions(reader, readState, isIsolatedFromAbove);
        if (failed(op))
          return failure();

        // If the op has regions, add it to the stack for processing and return:
        // we stop the processing of the current region and resume it after the
        // inner one is completed. Unless LazyLoading is activated in which case
        // nested region parsing is delayed.
        if ((*op)->getNumRegions()) {
          RegionReadState childState(*op, &reader, isIsolatedFromAbove);

          // Isolated regions are encoded as a section in version 2 and above.
          if (version >= bytecode::kLazyLoading && isIsolatedFromAbove) {
            bytecode::Section::ID sectionID;
            ArrayRef<uint8_t> sectionData;
            if (failed(reader.parseSection(sectionID, sectionData)))
              return failure();
            if (sectionID != bytecode::Section::kIR)
              return emitError(fileLoc, "expected IR section for region");
            childState.owningReader =
                std::make_unique<EncodingReader>(sectionData, fileLoc);
            childState.reader = childState.owningReader.get();

            // If the user has a callback set, they have the opportunity to
            // control lazyloading as we go.
            if (lazyLoading && (!lazyOpsCallback || !lazyOpsCallback(*op))) {
              lazyLoadableOps.emplace_back(*op, std::move(childState));
              lazyLoadableOpsMap.try_emplace(*op,
                                             std::prev(lazyLoadableOps.end()));
              continue;
            }
          }
          regionStack.push_back(std::move(childState));

          // If the op is isolated from above, push a new value scope.
          if (isIsolatedFromAbove)
            valueScopes.emplace_back();
          return success();
        }
      }

      // Move to the next block of the region.
      if (++readState.curBlock == readState.curRegion->end())
        break;
      if (failed(parseBlockHeader(reader, readState)))
        return failure();
    } while (true);

    // Reset the current block and any values reserved for this region.
    readState.curBlock = {};
    valueScopes.back().pop(readState);
  }

  // When the regions have been fully parsed, pop them off of the read stack. If
  // the regions were isolated from above, we also pop the last value scope.
  if (readState.isIsolatedFromAbove) {
    assert(!valueScopes.empty() && "Expect a valueScope after reading region");
    valueScopes.pop_back();
  }
  assert(!regionStack.empty() && "Expect a regionStack after reading region");
  regionStack.pop_back();
  return success();
}

FailureOr<Operation *>
BytecodeReader::Impl::parseOpWithoutRegions(EncodingReader &reader,
                                            RegionReadState &readState,
                                            bool &isIsolatedFromAbove) {
  // Parse the name of the operation.
  std::optional<bool> wasRegistered;
  FailureOr<OperationName> opName = parseOpName(reader, wasRegistered);
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

  if (opMask & bytecode::OpEncodingMask::kHasProperties) {
    // kHasProperties wasn't emitted in older bytecode, we should never get
    // there without also having the `wasRegistered` flag available.
    if (!wasRegistered)
      return emitError(fileLoc,
                       "Unexpected missing `wasRegistered` opname flag at "
                       "bytecode version ")
             << version << " with properties.";
    // When an operation is emitted without being registered, the properties are
    // stored as an attribute. Otherwise the op must implement the bytecode
    // interface and control the serialization.
    if (wasRegistered) {
      DialectReader dialectReader(attrTypeReader, stringReader, resourceReader,
                                  dialectsMap, reader, version);
      if (failed(
              propertiesReader.read(fileLoc, dialectReader, &*opName, opState)))
        return failure();
    } else {
      // If the operation wasn't registered when it was emitted, the properties
      // was serialized as an attribute.
      if (failed(parseAttribute(reader, opState.propertiesAttr)))
        return failure();
    }
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

  /// Parse the use-list orders for the results of the operation. Use-list
  /// orders are available since version 3 of the bytecode.
  std::optional<UseListMapT> resultIdxToUseListMap = std::nullopt;
  if (version >= bytecode::kUseListOrdering &&
      (opMask & bytecode::OpEncodingMask::kHasUseListOrders)) {
    size_t numResults = opState.types.size();
    auto parseResult = parseUseListOrderForRange(reader, numResults);
    if (failed(parseResult))
      return failure();
    resultIdxToUseListMap = std::move(*parseResult);
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

  /// Store a map for every value that received a custom use-list order from the
  /// bytecode file.
  if (resultIdxToUseListMap.has_value()) {
    for (size_t idx = 0; idx < op->getNumResults(); idx++) {
      if (resultIdxToUseListMap->contains(idx)) {
        valueToUseListMap.try_emplace(op->getResult(idx).getAsOpaquePointer(),
                                      resultIdxToUseListMap->at(idx));
      }
    }
  }
  return op;
}

LogicalResult BytecodeReader::Impl::parseRegion(RegionReadState &readState) {
  EncodingReader &reader = *readState.reader;

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
  return parseBlockHeader(reader, readState);
}

LogicalResult
BytecodeReader::Impl::parseBlockHeader(EncodingReader &reader,
                                       RegionReadState &readState) {
  bool hasArgs;
  if (failed(reader.parseVarIntWithFlag(readState.numOpsRemaining, hasArgs)))
    return failure();

  // Parse the arguments of the block.
  if (hasArgs && failed(parseBlockArguments(reader, &*readState.curBlock)))
    return failure();

  // Uselist orders are available since version 3 of the bytecode.
  if (version < bytecode::kUseListOrdering)
    return success();

  uint8_t hasUseListOrders = 0;
  if (hasArgs && failed(reader.parseByte(hasUseListOrders)))
    return failure();

  if (!hasUseListOrders)
    return success();

  Block &blk = *readState.curBlock;
  auto argIdxToUseListMap =
      parseUseListOrderForRange(reader, blk.getNumArguments());
  if (failed(argIdxToUseListMap) || argIdxToUseListMap->empty())
    return failure();

  for (size_t idx = 0; idx < blk.getNumArguments(); idx++)
    if (argIdxToUseListMap->contains(idx))
      valueToUseListMap.try_emplace(blk.getArgument(idx).getAsOpaquePointer(),
                                    argIdxToUseListMap->at(idx));

  // We don't parse the operations of the block here, that's done elsewhere.
  return success();
}

LogicalResult BytecodeReader::Impl::parseBlockArguments(EncodingReader &reader,
                                                        Block *block) {
  // Parse the value ID for the first argument, and the number of arguments.
  uint64_t numArgs;
  if (failed(reader.parseVarInt(numArgs)))
    return failure();

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  argTypes.reserve(numArgs);
  argLocs.reserve(numArgs);

  Location unknownLoc = UnknownLoc::get(config.getContext());
  while (numArgs--) {
    Type argType;
    LocationAttr argLoc = unknownLoc;
    if (version >= bytecode::kElideUnknownBlockArgLocation) {
      // Parse the type with hasLoc flag to determine if it has type.
      uint64_t typeIdx;
      bool hasLoc;
      if (failed(reader.parseVarIntWithFlag(typeIdx, hasLoc)) ||
          !(argType = attrTypeReader.resolveType(typeIdx)))
        return failure();
      if (hasLoc && failed(parseAttribute(reader, argLoc)))
        return failure();
    } else {
      // All args has type and location.
      if (failed(parseType(reader, argType)) ||
          failed(parseAttribute(reader, argLoc)))
        return failure();
    }
    argTypes.push_back(argType);
    argLocs.push_back(argLoc);
  }
  block->addArguments(argTypes, argLocs);
  return defineValues(reader, block->getArguments());
}

//===----------------------------------------------------------------------===//
// Value Processing

Value BytecodeReader::Impl::parseOperand(EncodingReader &reader) {
  std::vector<Value> &values = valueScopes.back().values;
  Value *value = nullptr;
  if (failed(parseEntry(reader, values, value, "value")))
    return Value();

  // Create a new forward reference if necessary.
  if (!*value)
    *value = createForwardRef();
  return *value;
}

LogicalResult BytecodeReader::Impl::defineValues(EncodingReader &reader,
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

Value BytecodeReader::Impl::createForwardRef() {
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

BytecodeReader::~BytecodeReader() { assert(getNumOpsToMaterialize() == 0); }

BytecodeReader::BytecodeReader(
    llvm::MemoryBufferRef buffer, const ParserConfig &config, bool lazyLoading,
    const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef) {
  Location sourceFileLoc =
      FileLineColLoc::get(config.getContext(), buffer.getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);
  impl = std::make_unique<Impl>(sourceFileLoc, config, lazyLoading, buffer,
                                bufferOwnerRef);
}

LogicalResult BytecodeReader::readTopLevel(
    Block *block, llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
  return impl->read(block, lazyOpsCallback);
}

int64_t BytecodeReader::getNumOpsToMaterialize() const {
  return impl->getNumOpsToMaterialize();
}

bool BytecodeReader::isMaterializable(Operation *op) {
  return impl->isMaterializable(op);
}

LogicalResult BytecodeReader::materialize(
    Operation *op, llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
  return impl->materialize(op, lazyOpsCallback);
}

LogicalResult
BytecodeReader::finalize(function_ref<bool(Operation *)> shouldMaterialize) {
  return impl->finalize(shouldMaterialize);
}

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

  BytecodeReader::Impl reader(sourceFileLoc, config, /*lazyLoading=*/false,
                              buffer, bufferOwnerRef);
  return reader.read(block, /*lazyOpsCallback=*/nullptr);
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
