//===- BytecodeWriter.cpp - MLIR Bytecode Writer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeWriter.h"
#include "IRNumbering.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "mlir-bytecode-writer"

using namespace mlir;
using namespace mlir::bytecode::detail;

//===----------------------------------------------------------------------===//
// BytecodeWriterConfig
//===----------------------------------------------------------------------===//

struct BytecodeWriterConfig::Impl {
  Impl(StringRef producer) : producer(producer) {}

  /// Version to use when writing.
  /// Note: This only differs from kVersion if a specific version is set.
  int64_t bytecodeVersion = bytecode::kVersion;

  /// A flag specifying whether to elide emission of resources into the bytecode
  /// file.
  bool shouldElideResourceData = false;

  /// A map containing dialect version information for each dialect to emit.
  llvm::StringMap<std::unique_ptr<DialectVersion>> dialectVersionMap;

  /// The producer of the bytecode.
  StringRef producer;

  /// Printer callbacks used to emit custom type and attribute encodings.
  llvm::SmallVector<std::unique_ptr<AttrTypeBytecodeWriter<Attribute>>>
      attributeWriterCallbacks;
  llvm::SmallVector<std::unique_ptr<AttrTypeBytecodeWriter<Type>>>
      typeWriterCallbacks;

  /// A collection of non-dialect resource printers.
  SmallVector<std::unique_ptr<AsmResourcePrinter>> externalResourcePrinters;
};

BytecodeWriterConfig::BytecodeWriterConfig(StringRef producer)
    : impl(std::make_unique<Impl>(producer)) {}
BytecodeWriterConfig::BytecodeWriterConfig(FallbackAsmResourceMap &map,
                                           StringRef producer)
    : BytecodeWriterConfig(producer) {
  attachFallbackResourcePrinter(map);
}
BytecodeWriterConfig::BytecodeWriterConfig(BytecodeWriterConfig &&config)
    : impl(std::move(config.impl)) {}

BytecodeWriterConfig::~BytecodeWriterConfig() = default;

ArrayRef<std::unique_ptr<AttrTypeBytecodeWriter<Attribute>>>
BytecodeWriterConfig::getAttributeWriterCallbacks() const {
  return impl->attributeWriterCallbacks;
}

ArrayRef<std::unique_ptr<AttrTypeBytecodeWriter<Type>>>
BytecodeWriterConfig::getTypeWriterCallbacks() const {
  return impl->typeWriterCallbacks;
}

void BytecodeWriterConfig::attachAttributeCallback(
    std::unique_ptr<AttrTypeBytecodeWriter<Attribute>> callback) {
  impl->attributeWriterCallbacks.emplace_back(std::move(callback));
}

void BytecodeWriterConfig::attachTypeCallback(
    std::unique_ptr<AttrTypeBytecodeWriter<Type>> callback) {
  impl->typeWriterCallbacks.emplace_back(std::move(callback));
}

void BytecodeWriterConfig::attachResourcePrinter(
    std::unique_ptr<AsmResourcePrinter> printer) {
  impl->externalResourcePrinters.emplace_back(std::move(printer));
}

void BytecodeWriterConfig::setElideResourceDataFlag(
    bool shouldElideResourceData) {
  impl->shouldElideResourceData = shouldElideResourceData;
}

void BytecodeWriterConfig::setDesiredBytecodeVersion(int64_t bytecodeVersion) {
  impl->bytecodeVersion = bytecodeVersion;
}

int64_t BytecodeWriterConfig::getDesiredBytecodeVersion() const {
  return impl->bytecodeVersion;
}

llvm::StringMap<std::unique_ptr<DialectVersion>> &
BytecodeWriterConfig::getDialectVersionMap() const {
  return impl->dialectVersionMap;
}

void BytecodeWriterConfig::setDialectVersion(
    llvm::StringRef dialectName,
    std::unique_ptr<DialectVersion> dialectVersion) const {
  assert(!impl->dialectVersionMap.contains(dialectName) &&
         "cannot override a previously set dialect version");
  impl->dialectVersionMap.insert({dialectName, std::move(dialectVersion)});
}

//===----------------------------------------------------------------------===//
// EncodingEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This class functions as the underlying encoding emitter for the bytecode
/// writer. This class is a bit different compared to other types of encoders;
/// it does not use a single buffer, but instead may contain several buffers
/// (some owned by the writer, and some not) that get concatted during the final
/// emission.
class EncodingEmitter {
public:
  EncodingEmitter() = default;
  EncodingEmitter(const EncodingEmitter &) = delete;
  EncodingEmitter &operator=(const EncodingEmitter &) = delete;

  /// Write the current contents to the provided stream.
  void writeTo(raw_ostream &os) const;

  /// Return the current size of the encoded buffer.
  size_t size() const { return prevResultSize + currentResult.size(); }

  //===--------------------------------------------------------------------===//
  // Emission
  //===--------------------------------------------------------------------===//

  /// Backpatch a byte in the result buffer at the given offset.
  void patchByte(uint64_t offset, uint8_t value, StringLiteral desc) {
    LDBG() << "patchByte(" << offset << ',' << uint64_t(value) << ")\t" << desc;
    assert(offset < size() && offset >= prevResultSize &&
           "cannot patch previously emitted data");
    currentResult[offset - prevResultSize] = value;
  }

  /// Emit the provided blob of data, which is owned by the caller and is
  /// guaranteed to not die before the end of the bytecode process.
  void emitOwnedBlob(ArrayRef<uint8_t> data, StringLiteral desc) {
    LDBG() << "emitOwnedBlob(" << data.size() << "b)\t" << desc;
    // Push the current buffer before adding the provided data.
    appendResult(std::move(currentResult));
    appendOwnedResult(data);
  }

  /// Emit the provided blob of data that has the given alignment, which is
  /// owned by the caller and is guaranteed to not die before the end of the
  /// bytecode process. The alignment value is also encoded, making it available
  /// on load.
  void emitOwnedBlobAndAlignment(ArrayRef<uint8_t> data, uint32_t alignment,
                                 StringLiteral desc) {
    emitVarInt(alignment, desc);
    emitVarInt(data.size(), desc);

    alignTo(alignment);
    emitOwnedBlob(data, desc);
  }
  void emitOwnedBlobAndAlignment(ArrayRef<char> data, uint32_t alignment,
                                 StringLiteral desc) {
    ArrayRef<uint8_t> castedData(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
    emitOwnedBlobAndAlignment(castedData, alignment, desc);
  }

  /// Align the emitter to the given alignment.
  void alignTo(unsigned alignment) {
    if (alignment < 2)
      return;
    assert(llvm::isPowerOf2_32(alignment) && "expected valid alignment");

    // Check to see if we need to emit any padding bytes to meet the desired
    // alignment.
    size_t curOffset = size();
    size_t paddingSize = llvm::alignTo(curOffset, alignment) - curOffset;
    while (paddingSize--)
      emitByte(bytecode::kAlignmentByte, "alignment byte");

    // Keep track of the maximum required alignment.
    requiredAlignment = std::max(requiredAlignment, alignment);
  }

  //===--------------------------------------------------------------------===//
  // Integer Emission

  /// Emit a single byte.
  template <typename T>
  void emitByte(T byte, StringLiteral desc) {
    LDBG() << "emitByte(" << uint64_t(byte) << ")\t" << desc;
    currentResult.push_back(static_cast<uint8_t>(byte));
  }

  /// Emit a range of bytes.
  void emitBytes(ArrayRef<uint8_t> bytes, StringLiteral desc) {
    LDBG() << "emitBytes(" << bytes.size() << "b)\t" << desc;
    llvm::append_range(currentResult, bytes);
  }

  /// Emit a variable length integer. The first encoded byte contains a prefix
  /// in the low bits indicating the encoded length of the value. This length
  /// prefix is a bit sequence of '0's followed by a '1'. The number of '0' bits
  /// indicate the number of _additional_ bytes (not including the prefix byte).
  /// All remaining bits in the first byte, along with all of the bits in
  /// additional bytes, provide the value of the integer encoded in
  /// little-endian order.
  void emitVarInt(uint64_t value, StringLiteral desc) {
    LDBG() << "emitVarInt(" << value << ")\t" << desc;

    // In the most common case, the value can be represented in a single byte.
    // Given how hot this case is, explicitly handle that here.
    if ((value >> 7) == 0)
      return emitByte((value << 1) | 0x1, desc);
    emitMultiByteVarInt(value, desc);
  }

  /// Emit a signed variable length integer. Signed varints are encoded using
  /// a varint with zigzag encoding, meaning that we use the low bit of the
  /// value to indicate the sign of the value. This allows for more efficient
  /// encoding of negative values by limiting the number of active bits
  void emitSignedVarInt(uint64_t value, StringLiteral desc) {
    emitVarInt((value << 1) ^ (uint64_t)((int64_t)value >> 63), desc);
  }

  /// Emit a variable length integer whose low bit is used to encode the
  /// provided flag, i.e. encoded as: (value << 1) | (flag ? 1 : 0).
  void emitVarIntWithFlag(uint64_t value, bool flag, StringLiteral desc) {
    emitVarInt((value << 1) | (flag ? 1 : 0), desc);
  }

  //===--------------------------------------------------------------------===//
  // String Emission

  /// Emit the given string as a nul terminated string.
  void emitNulTerminatedString(StringRef str, StringLiteral desc) {
    emitString(str, desc);
    emitByte(0, "null terminator");
  }

  /// Emit the given string without a nul terminator.
  void emitString(StringRef str, StringLiteral desc) {
    emitBytes({reinterpret_cast<const uint8_t *>(str.data()), str.size()},
              desc);
  }

  //===--------------------------------------------------------------------===//
  // Section Emission

  /// Emit a nested section of the given code, whose contents are encoded in the
  /// provided emitter.
  void emitSection(bytecode::Section::ID code, EncodingEmitter &&emitter) {
    // Emit the section code and length. The high bit of the code is used to
    // indicate whether the section alignment is present, so save an offset to
    // it.
    uint64_t codeOffset = currentResult.size();
    emitByte(code, "section code");
    emitVarInt(emitter.size(), "section size");

    // Integrate the alignment of the section into this emitter if necessary.
    unsigned emitterAlign = emitter.requiredAlignment;
    if (emitterAlign > 1) {
      if (size() & (emitterAlign - 1)) {
        emitVarInt(emitterAlign, "section alignment");
        alignTo(emitterAlign);

        // Indicate that we needed to align the section, the high bit of the
        // code field is used for this.
        currentResult[codeOffset] |= 0b10000000;
      } else {
        // Otherwise, if we happen to be at a compatible offset, we just
        // remember that we need this alignment.
        requiredAlignment = std::max(requiredAlignment, emitterAlign);
      }
    }

    // Push our current buffer and then merge the provided section body into
    // ours.
    appendResult(std::move(currentResult));
    for (std::vector<uint8_t> &result : emitter.prevResultStorage)
      prevResultStorage.push_back(std::move(result));
    llvm::append_range(prevResultList, emitter.prevResultList);
    prevResultSize += emitter.prevResultSize;
    appendResult(std::move(emitter.currentResult));
  }

private:
  /// Emit the given value using a variable width encoding. This method is a
  /// fallback when the number of bytes needed to encode the value is greater
  /// than 1. We mark it noinline here so that the single byte hot path isn't
  /// pessimized.
  LLVM_ATTRIBUTE_NOINLINE void emitMultiByteVarInt(uint64_t value,
                                                   StringLiteral desc);

  /// Append a new result buffer to the current contents.
  void appendResult(std::vector<uint8_t> &&result) {
    if (result.empty())
      return;
    prevResultStorage.emplace_back(std::move(result));
    appendOwnedResult(prevResultStorage.back());
  }
  void appendOwnedResult(ArrayRef<uint8_t> result) {
    if (result.empty())
      return;
    prevResultSize += result.size();
    prevResultList.emplace_back(result);
  }

  /// The result of the emitter currently being built. We refrain from building
  /// a single buffer to simplify emitting sections, large data, and more. The
  /// result is thus represented using multiple distinct buffers, some of which
  /// we own (via prevResultStorage), and some of which are just pointers into
  /// externally owned buffers.
  std::vector<uint8_t> currentResult;
  std::vector<ArrayRef<uint8_t>> prevResultList;
  std::vector<std::vector<uint8_t>> prevResultStorage;

  /// An up-to-date total size of all of the buffers within `prevResultList`.
  /// This enables O(1) size checks of the current encoding.
  size_t prevResultSize = 0;

  /// The highest required alignment for the start of this section.
  unsigned requiredAlignment = 1;
};

//===----------------------------------------------------------------------===//
// StringSectionBuilder
//===----------------------------------------------------------------------===//

namespace {
/// This class is used to simplify the process of emitting the string section.
class StringSectionBuilder {
public:
  /// Add the given string to the string section, and return the index of the
  /// string within the section.
  size_t insert(StringRef str) {
    auto it = strings.insert({llvm::CachedHashStringRef(str), strings.size()});
    return it.first->second;
  }

  /// Write the current set of strings to the given emitter.
  void write(EncodingEmitter &emitter) {
    emitter.emitVarInt(strings.size(), "string section size");

    // Emit the sizes in reverse order, so that we don't need to backpatch an
    // offset to the string data or have a separate section.
    for (const auto &it : llvm::reverse(strings))
      emitter.emitVarInt(it.first.size() + 1, "string size");
    // Emit the string data itself.
    for (const auto &it : strings)
      emitter.emitNulTerminatedString(it.first.val(), "string");
  }

private:
  /// A set of strings referenced within the bytecode. The value of the map is
  /// unused.
  llvm::MapVector<llvm::CachedHashStringRef, size_t> strings;
};
} // namespace

class DialectWriter : public DialectBytecodeWriter {
  using DialectVersionMapT = llvm::StringMap<std::unique_ptr<DialectVersion>>;

public:
  DialectWriter(int64_t bytecodeVersion, EncodingEmitter &emitter,
                IRNumberingState &numberingState,
                StringSectionBuilder &stringSection,
                const DialectVersionMapT &dialectVersionMap)
      : bytecodeVersion(bytecodeVersion), emitter(emitter),
        numberingState(numberingState), stringSection(stringSection),
        dialectVersionMap(dialectVersionMap) {}

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  void writeAttribute(Attribute attr) override {
    emitter.emitVarInt(numberingState.getNumber(attr), "dialect attr");
  }
  void writeOptionalAttribute(Attribute attr) override {
    if (!attr) {
      emitter.emitVarInt(0, "dialect optional attr none");
      return;
    }
    emitter.emitVarIntWithFlag(numberingState.getNumber(attr), true,
                               "dialect optional attr");
  }

  void writeType(Type type) override {
    emitter.emitVarInt(numberingState.getNumber(type), "dialect type");
  }

  void writeResourceHandle(const AsmDialectResourceHandle &resource) override {
    emitter.emitVarInt(numberingState.getNumber(resource), "dialect resource");
  }

  //===--------------------------------------------------------------------===//
  // Primitives
  //===--------------------------------------------------------------------===//

  void writeVarInt(uint64_t value) override {
    emitter.emitVarInt(value, "dialect writer");
  }

  void writeSignedVarInt(int64_t value) override {
    emitter.emitSignedVarInt(value, "dialect writer");
  }

  void writeAPIntWithKnownWidth(const APInt &value) override {
    size_t bitWidth = value.getBitWidth();

    // If the value is a single byte, just emit it directly without going
    // through a varint.
    if (bitWidth <= 8)
      return emitter.emitByte(value.getLimitedValue(), "dialect APInt");

    // If the value fits within a single varint, emit it directly.
    if (bitWidth <= 64)
      return emitter.emitSignedVarInt(value.getLimitedValue(), "dialect APInt");

    // Otherwise, we need to encode a variable number of active words. We use
    // active words instead of the number of total words under the observation
    // that smaller values will be more common.
    unsigned numActiveWords = value.getActiveWords();
    emitter.emitVarInt(numActiveWords, "dialect APInt word count");

    const uint64_t *rawValueData = value.getRawData();
    for (unsigned i = 0; i < numActiveWords; ++i)
      emitter.emitSignedVarInt(rawValueData[i], "dialect APInt word");
  }

  void writeAPFloatWithKnownSemantics(const APFloat &value) override {
    writeAPIntWithKnownWidth(value.bitcastToAPInt());
  }

  void writeOwnedString(StringRef str) override {
    emitter.emitVarInt(stringSection.insert(str), "dialect string");
  }

  void writeOwnedBlob(ArrayRef<char> blob) override {
    emitter.emitVarInt(blob.size(), "dialect blob");
    emitter.emitOwnedBlob(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(blob.data()),
                          blob.size()),
        "dialect blob");
  }

  void writeOwnedBool(bool value) override {
    emitter.emitByte(value, "dialect bool");
  }

  int64_t getBytecodeVersion() const override { return bytecodeVersion; }

  FailureOr<const DialectVersion *>
  getDialectVersion(StringRef dialectName) const override {
    auto dialectEntry = dialectVersionMap.find(dialectName);
    if (dialectEntry == dialectVersionMap.end())
      return failure();
    return dialectEntry->getValue().get();
  }

private:
  int64_t bytecodeVersion;
  EncodingEmitter &emitter;
  IRNumberingState &numberingState;
  StringSectionBuilder &stringSection;
  const DialectVersionMapT &dialectVersionMap;
};

namespace {
class PropertiesSectionBuilder {
public:
  PropertiesSectionBuilder(IRNumberingState &numberingState,
                           StringSectionBuilder &stringSection,
                           const BytecodeWriterConfig::Impl &config)
      : numberingState(numberingState), stringSection(stringSection),
        config(config) {}

  /// Emit the op properties in the properties section and return the index of
  /// the properties within the section. Return -1 if no properties was emitted.
  std::optional<ssize_t> emit(Operation *op) {
    EncodingEmitter propertiesEmitter;
    if (!op->getPropertiesStorageSize())
      return std::nullopt;
    if (!op->isRegistered()) {
      // Unregistered op are storing properties as an optional attribute.
      Attribute prop = *op->getPropertiesStorage().as<Attribute *>();
      if (!prop)
        return std::nullopt;
      EncodingEmitter sizeEmitter;
      sizeEmitter.emitVarInt(numberingState.getNumber(prop), "properties size");
      scratch.clear();
      llvm::raw_svector_ostream os(scratch);
      sizeEmitter.writeTo(os);
      return emit(scratch);
    }

    EncodingEmitter emitter;
    DialectWriter propertiesWriter(config.bytecodeVersion, emitter,
                                   numberingState, stringSection,
                                   config.dialectVersionMap);
    auto iface = cast<BytecodeOpInterface>(op);
    iface.writeProperties(propertiesWriter);
    scratch.clear();
    llvm::raw_svector_ostream os(scratch);
    emitter.writeTo(os);
    return emit(scratch);
  }

  /// Write the current set of properties to the given emitter.
  void write(EncodingEmitter &emitter) {
    emitter.emitVarInt(propertiesStorage.size(), "properties size");
    if (propertiesStorage.empty())
      return;
    for (const auto &storage : propertiesStorage) {
      if (storage.empty()) {
        emitter.emitBytes(ArrayRef<uint8_t>(), "empty properties");
        continue;
      }
      emitter.emitBytes(ArrayRef(reinterpret_cast<const uint8_t *>(&storage[0]),
                                 storage.size()),
                        "property");
    }
  }

  /// Returns true if the section is empty.
  bool empty() { return propertiesStorage.empty(); }

private:
  /// Emit raw data and returns the offset in the internal buffer.
  /// Data are deduplicated and will be copied in the internal buffer only if
  /// they don't exist there already.
  ssize_t emit(ArrayRef<char> rawProperties) {
    // Populate a scratch buffer with the properties size.
    SmallVector<char> sizeScratch;
    {
      EncodingEmitter sizeEmitter;
      sizeEmitter.emitVarInt(rawProperties.size(), "properties");
      llvm::raw_svector_ostream os(sizeScratch);
      sizeEmitter.writeTo(os);
    }
    // Append a new storage to the table now.
    size_t index = propertiesStorage.size();
    propertiesStorage.emplace_back();
    std::vector<char> &newStorage = propertiesStorage.back();
    size_t propertiesSize = sizeScratch.size() + rawProperties.size();
    newStorage.reserve(propertiesSize);
    llvm::append_range(newStorage, sizeScratch);
    llvm::append_range(newStorage, rawProperties);

    // Try to de-duplicate the new serialized properties.
    // If the properties is a duplicate, pop it back from the storage.
    auto inserted = propertiesUniquing.insert(
        std::make_pair(ArrayRef<char>(newStorage), index));
    if (!inserted.second)
      propertiesStorage.pop_back();
    return inserted.first->getSecond();
  }

  /// Storage for properties.
  std::vector<std::vector<char>> propertiesStorage;
  SmallVector<char> scratch;
  DenseMap<ArrayRef<char>, int64_t> propertiesUniquing;
  IRNumberingState &numberingState;
  StringSectionBuilder &stringSection;
  const BytecodeWriterConfig::Impl &config;
};
} // namespace

/// A simple raw_ostream wrapper around a EncodingEmitter. This removes the need
/// to go through an intermediate buffer when interacting with code that wants a
/// raw_ostream.
class RawEmitterOstream : public raw_ostream {
public:
  explicit RawEmitterOstream(EncodingEmitter &emitter) : emitter(emitter) {
    SetUnbuffered();
  }

private:
  void write_impl(const char *ptr, size_t size) override {
    emitter.emitBytes({reinterpret_cast<const uint8_t *>(ptr), size},
                      "raw emitter");
  }
  uint64_t current_pos() const override { return emitter.size(); }

  /// The section being emitted to.
  EncodingEmitter &emitter;
};
} // namespace

void EncodingEmitter::writeTo(raw_ostream &os) const {
  // Reserve space in the ostream for the encoded contents.
  os.reserveExtraSpace(size());

  for (auto &prevResult : prevResultList)
    os.write((const char *)prevResult.data(), prevResult.size());
  os.write((const char *)currentResult.data(), currentResult.size());
}

void EncodingEmitter::emitMultiByteVarInt(uint64_t value, StringLiteral desc) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint64_t it = value >> 7;
  for (size_t numBytes = 2; numBytes < 9; ++numBytes) {
    if (LLVM_LIKELY(it >>= 7) == 0) {
      uint64_t encodedValue = (value << 1) | 0x1;
      encodedValue <<= (numBytes - 1);
      llvm::support::ulittle64_t encodedValueLE(encodedValue);
      emitBytes({reinterpret_cast<uint8_t *>(&encodedValueLE), numBytes}, desc);
      return;
    }
  }

  // If the value is too large to encode in a single byte, emit a special all
  // zero marker byte and splat the value directly.
  emitByte(0, desc);
  llvm::support::ulittle64_t valueLE(value);
  emitBytes({reinterpret_cast<uint8_t *>(&valueLE), sizeof(valueLE)}, desc);
}

//===----------------------------------------------------------------------===//
// Bytecode Writer
//===----------------------------------------------------------------------===//

namespace {
class BytecodeWriter {
public:
  BytecodeWriter(Operation *op, const BytecodeWriterConfig &config)
      : numberingState(op, config), config(config.getImpl()),
        propertiesSection(numberingState, stringSection, config.getImpl()) {}

  /// Write the bytecode for the given root operation.
  LogicalResult write(Operation *rootOp, raw_ostream &os);

private:
  //===--------------------------------------------------------------------===//
  // Dialects

  void writeDialectSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Attributes and Types

  void writeAttrTypeSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Operations

  LogicalResult writeBlock(EncodingEmitter &emitter, Block *block);
  LogicalResult writeOp(EncodingEmitter &emitter, Operation *op);
  LogicalResult writeRegion(EncodingEmitter &emitter, Region *region);
  LogicalResult writeIRSection(EncodingEmitter &emitter, Operation *op);

  LogicalResult writeRegions(EncodingEmitter &emitter,
                             MutableArrayRef<Region> regions) {
    return success(llvm::all_of(regions, [&](Region &region) {
      return succeeded(writeRegion(emitter, &region));
    }));
  }

  //===--------------------------------------------------------------------===//
  // Resources

  void writeResourceSection(Operation *op, EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Strings

  void writeStringSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Properties

  void writePropertiesSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Helpers

  void writeUseListOrders(EncodingEmitter &emitter, uint8_t &opEncodingMask,
                          ValueRange range);

  //===--------------------------------------------------------------------===//
  // Fields

  /// The builder used for the string section.
  StringSectionBuilder stringSection;

  /// The IR numbering state generated for the root operation.
  IRNumberingState numberingState;

  /// Configuration dictating bytecode emission.
  const BytecodeWriterConfig::Impl &config;

  /// Storage for the properties section
  PropertiesSectionBuilder propertiesSection;
};
} // namespace

LogicalResult BytecodeWriter::write(Operation *rootOp, raw_ostream &os) {
  EncodingEmitter emitter;

  // Emit the bytecode file header. This is how we identify the output as a
  // bytecode file.
  emitter.emitString("ML\xefR", "bytecode header");

  // Emit the bytecode version.
  if (config.bytecodeVersion < bytecode::kMinSupportedVersion ||
      config.bytecodeVersion > bytecode::kVersion)
    return rootOp->emitError()
           << "unsupported version requested " << config.bytecodeVersion
           << ", must be in range ["
           << static_cast<int64_t>(bytecode::kMinSupportedVersion) << ", "
           << static_cast<int64_t>(bytecode::kVersion) << ']';
  emitter.emitVarInt(config.bytecodeVersion, "bytecode version");

  // Emit the producer.
  emitter.emitNulTerminatedString(config.producer, "bytecode producer");

  // Emit the dialect section.
  writeDialectSection(emitter);

  // Emit the attributes and types section.
  writeAttrTypeSection(emitter);

  // Emit the IR section.
  if (failed(writeIRSection(emitter, rootOp)))
    return failure();

  // Emit the resources section.
  writeResourceSection(rootOp, emitter);

  // Emit the string section.
  writeStringSection(emitter);

  // Emit the properties section.
  if (config.bytecodeVersion >= bytecode::kNativePropertiesEncoding)
    writePropertiesSection(emitter);
  else if (!propertiesSection.empty())
    return rootOp->emitError(
        "unexpected properties emitted incompatible with bytecode <5");

  // Write the generated bytecode to the provided output stream.
  emitter.writeTo(os);

  return success();
}

//===----------------------------------------------------------------------===//
// Dialects
//===----------------------------------------------------------------------===//

/// Write the given entries in contiguous groups with the same parent dialect.
/// Each dialect sub-group is encoded with the parent dialect and number of
/// elements, followed by the encoding for the entries. The given callback is
/// invoked to encode each individual entry.
template <typename EntriesT, typename EntryCallbackT>
static void writeDialectGrouping(EncodingEmitter &emitter, EntriesT &&entries,
                                 EntryCallbackT &&callback) {
  for (auto it = entries.begin(), e = entries.end(); it != e;) {
    auto groupStart = it++;

    // Find the end of the group that shares the same parent dialect.
    DialectNumbering *currentDialect = groupStart->dialect;
    it = std::find_if(it, e, [&](const auto &entry) {
      return entry.dialect != currentDialect;
    });

    // Emit the dialect and number of elements.
    emitter.emitVarInt(currentDialect->number, "dialect number");
    emitter.emitVarInt(std::distance(groupStart, it), "dialect offset");

    // Emit the entries within the group.
    for (auto &entry : llvm::make_range(groupStart, it))
      callback(entry);
  }
}

void BytecodeWriter::writeDialectSection(EncodingEmitter &emitter) {
  EncodingEmitter dialectEmitter;

  // Emit the referenced dialects.
  auto dialects = numberingState.getDialects();
  dialectEmitter.emitVarInt(llvm::size(dialects), "dialects count");
  for (DialectNumbering &dialect : dialects) {
    // Write the string section and get the ID.
    size_t nameID = stringSection.insert(dialect.name);

    if (config.bytecodeVersion < bytecode::kDialectVersioning) {
      dialectEmitter.emitVarInt(nameID, "dialect name ID");
      continue;
    }

    // Try writing the version to the versionEmitter.
    EncodingEmitter versionEmitter;
    if (dialect.interface) {
      // The writer used when emitting using a custom bytecode encoding.
      DialectWriter versionWriter(config.bytecodeVersion, versionEmitter,
                                  numberingState, stringSection,
                                  config.dialectVersionMap);
      dialect.interface->writeVersion(versionWriter);
    }

    // If the version emitter is empty, version is not available. We can encode
    // this in the dialect ID, so if there is no version, we don't write the
    // section.
    size_t versionAvailable = versionEmitter.size() > 0;
    dialectEmitter.emitVarIntWithFlag(nameID, versionAvailable,
                                      "dialect version");
    if (versionAvailable)
      dialectEmitter.emitSection(bytecode::Section::kDialectVersions,
                                 std::move(versionEmitter));
  }

  if (config.bytecodeVersion >= bytecode::kElideUnknownBlockArgLocation)
    dialectEmitter.emitVarInt(size(numberingState.getOpNames()),
                              "op names count");

  // Emit the referenced operation names grouped by dialect.
  auto emitOpName = [&](OpNameNumbering &name) {
    size_t stringId = stringSection.insert(name.name.stripDialect());
    if (config.bytecodeVersion < bytecode::kNativePropertiesEncoding)
      dialectEmitter.emitVarInt(stringId, "dialect op name");
    else
      dialectEmitter.emitVarIntWithFlag(stringId, name.name.isRegistered(),
                                        "dialect op name");
  };
  writeDialectGrouping(dialectEmitter, numberingState.getOpNames(), emitOpName);

  emitter.emitSection(bytecode::Section::kDialect, std::move(dialectEmitter));
}

//===----------------------------------------------------------------------===//
// Attributes and Types
//===----------------------------------------------------------------------===//

void BytecodeWriter::writeAttrTypeSection(EncodingEmitter &emitter) {
  EncodingEmitter attrTypeEmitter;
  EncodingEmitter offsetEmitter;
  offsetEmitter.emitVarInt(llvm::size(numberingState.getAttributes()),
                           "attributes count");
  offsetEmitter.emitVarInt(llvm::size(numberingState.getTypes()),
                           "types count");

  // A functor used to emit an attribute or type entry.
  uint64_t prevOffset = 0;
  auto emitAttrOrType = [&](auto &entry) {
    auto entryValue = entry.getValue();

    auto emitAttrOrTypeRawImpl = [&]() -> void {
      RawEmitterOstream(attrTypeEmitter) << entryValue;
      attrTypeEmitter.emitByte(0, "attr/type separator");
    };
    auto emitAttrOrTypeImpl = [&]() -> bool {
      // TODO: We don't currently support custom encoded mutable types and
      // attributes.
      if (entryValue.template hasTrait<TypeTrait::IsMutable>() ||
          entryValue.template hasTrait<AttributeTrait::IsMutable>()) {
        emitAttrOrTypeRawImpl();
        return false;
      }

      DialectWriter dialectWriter(config.bytecodeVersion, attrTypeEmitter,
                                  numberingState, stringSection,
                                  config.dialectVersionMap);
      if constexpr (std::is_same_v<std::decay_t<decltype(entryValue)>, Type>) {
        for (const auto &callback : config.typeWriterCallbacks) {
          if (succeeded(callback->write(entryValue, dialectWriter)))
            return true;
        }
        if (const BytecodeDialectInterface *interface =
                entry.dialect->interface) {
          if (succeeded(interface->writeType(entryValue, dialectWriter)))
            return true;
        }
      } else {
        for (const auto &callback : config.attributeWriterCallbacks) {
          if (succeeded(callback->write(entryValue, dialectWriter)))
            return true;
        }
        if (const BytecodeDialectInterface *interface =
                entry.dialect->interface) {
          if (succeeded(interface->writeAttribute(entryValue, dialectWriter)))
            return true;
        }
      }

      // If the entry was not emitted using a callback or a dialect interface,
      // emit it using the textual format.
      emitAttrOrTypeRawImpl();
      return false;
    };

    bool hasCustomEncoding = emitAttrOrTypeImpl();

    // Record the offset of this entry.
    uint64_t curOffset = attrTypeEmitter.size();
    offsetEmitter.emitVarIntWithFlag(curOffset - prevOffset, hasCustomEncoding,
                                     "attr/type offset");
    prevOffset = curOffset;
  };

  // Emit the attribute and type entries for each dialect.
  writeDialectGrouping(offsetEmitter, numberingState.getAttributes(),
                       emitAttrOrType);
  writeDialectGrouping(offsetEmitter, numberingState.getTypes(),
                       emitAttrOrType);

  // Emit the sections to the stream.
  emitter.emitSection(bytecode::Section::kAttrTypeOffset,
                      std::move(offsetEmitter));
  emitter.emitSection(bytecode::Section::kAttrType, std::move(attrTypeEmitter));
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

LogicalResult BytecodeWriter::writeBlock(EncodingEmitter &emitter,
                                         Block *block) {
  ArrayRef<BlockArgument> args = block->getArguments();
  bool hasArgs = !args.empty();

  // Emit the number of operations in this block, and if it has arguments. We
  // use the low bit of the operation count to indicate if the block has
  // arguments.
  unsigned numOps = numberingState.getOperationCount(block);
  emitter.emitVarIntWithFlag(numOps, hasArgs, "block num ops");

  // Emit the arguments of the block.
  if (hasArgs) {
    emitter.emitVarInt(args.size(), "block args count");
    for (BlockArgument arg : args) {
      Location argLoc = arg.getLoc();
      if (config.bytecodeVersion >= bytecode::kElideUnknownBlockArgLocation) {
        emitter.emitVarIntWithFlag(numberingState.getNumber(arg.getType()),
                                   !isa<UnknownLoc>(argLoc), "block arg type");
        if (!isa<UnknownLoc>(argLoc))
          emitter.emitVarInt(numberingState.getNumber(argLoc),
                             "block arg location");
      } else {
        emitter.emitVarInt(numberingState.getNumber(arg.getType()),
                           "block arg type");
        emitter.emitVarInt(numberingState.getNumber(argLoc),
                           "block arg location");
      }
    }
    if (config.bytecodeVersion >= bytecode::kUseListOrdering) {
      uint64_t maskOffset = emitter.size();
      uint8_t encodingMask = 0;
      emitter.emitByte(0, "use-list separator");
      writeUseListOrders(emitter, encodingMask, args);
      if (encodingMask)
        emitter.patchByte(maskOffset, encodingMask, "block patch encoding");
    }
  }

  // Emit the operations within the block.
  for (Operation &op : *block)
    if (failed(writeOp(emitter, &op)))
      return failure();
  return success();
}

LogicalResult BytecodeWriter::writeOp(EncodingEmitter &emitter, Operation *op) {
  emitter.emitVarInt(numberingState.getNumber(op->getName()), "op name ID");

  // Emit a mask for the operation components. We need to fill this in later
  // (when we actually know what needs to be emitted), so emit a placeholder for
  // now.
  uint64_t maskOffset = emitter.size();
  uint8_t opEncodingMask = 0;
  emitter.emitByte(0, "op separator");

  // Emit the location for this operation.
  emitter.emitVarInt(numberingState.getNumber(op->getLoc()), "op location");

  // Emit the attributes of this operation.
  DictionaryAttr attrs = op->getDiscardableAttrDictionary();
  // Allow deployment to version <kNativePropertiesEncoding by merging inherent
  // attribute with the discardable ones. We should fail if there are any
  // conflicts. When properties are not used by the op, also store everything as
  // attributes.
  if (config.bytecodeVersion < bytecode::kNativePropertiesEncoding ||
      !op->getPropertiesStorage()) {
    attrs = op->getAttrDictionary();
  }
  if (!attrs.empty()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasAttrs;
    emitter.emitVarInt(numberingState.getNumber(attrs), "op attrs count");
  }

  // Emit the properties of this operation, for now we still support deployment
  // to version <kNativePropertiesEncoding.
  if (config.bytecodeVersion >= bytecode::kNativePropertiesEncoding) {
    std::optional<ssize_t> propertiesId = propertiesSection.emit(op);
    if (propertiesId.has_value()) {
      opEncodingMask |= bytecode::OpEncodingMask::kHasProperties;
      emitter.emitVarInt(*propertiesId, "op properties ID");
    }
  }

  // Emit the result types of the operation.
  if (unsigned numResults = op->getNumResults()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasResults;
    emitter.emitVarInt(numResults, "op results count");
    for (Type type : op->getResultTypes())
      emitter.emitVarInt(numberingState.getNumber(type), "op result type");
  }

  // Emit the operands of the operation.
  if (unsigned numOperands = op->getNumOperands()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasOperands;
    emitter.emitVarInt(numOperands, "op operands count");
    for (Value operand : op->getOperands())
      emitter.emitVarInt(numberingState.getNumber(operand), "op operand types");
  }

  // Emit the successors of the operation.
  if (unsigned numSuccessors = op->getNumSuccessors()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasSuccessors;
    emitter.emitVarInt(numSuccessors, "op successors count");
    for (Block *successor : op->getSuccessors())
      emitter.emitVarInt(numberingState.getNumber(successor), "op successor");
  }

  // Emit the use-list orders to bytecode, so we can reconstruct the same order
  // at parsing.
  if (config.bytecodeVersion >= bytecode::kUseListOrdering)
    writeUseListOrders(emitter, opEncodingMask, ValueRange(op->getResults()));

  // Check for regions.
  unsigned numRegions = op->getNumRegions();
  if (numRegions)
    opEncodingMask |= bytecode::OpEncodingMask::kHasInlineRegions;

  // Update the mask for the operation.
  emitter.patchByte(maskOffset, opEncodingMask, "op encoding mask");

  // With the mask emitted, we can now emit the regions of the operation. We do
  // this after mask emission to avoid offset complications that may arise by
  // emitting the regions first (e.g. if the regions are huge, backpatching the
  // op encoding mask is more annoying).
  if (numRegions) {
    bool isIsolatedFromAbove = numberingState.isIsolatedFromAbove(op);
    emitter.emitVarIntWithFlag(numRegions, isIsolatedFromAbove,
                               "op regions count");

    // If the region is not isolated from above, or we are emitting bytecode
    // targeting version <kLazyLoading, we don't use a section.
    if (isIsolatedFromAbove &&
        config.bytecodeVersion >= bytecode::kLazyLoading) {
      EncodingEmitter regionEmitter;
      if (failed(writeRegions(regionEmitter, op->getRegions())))
        return failure();
      emitter.emitSection(bytecode::Section::kIR, std::move(regionEmitter));

    } else if (failed(writeRegions(emitter, op->getRegions()))) {
      return failure();
    }
  }
  return success();
}

void BytecodeWriter::writeUseListOrders(EncodingEmitter &emitter,
                                        uint8_t &opEncodingMask,
                                        ValueRange range) {
  // Loop over the results and store the use-list order per result index.
  DenseMap<unsigned, llvm::SmallVector<unsigned>> map;
  for (auto item : llvm::enumerate(range)) {
    auto value = item.value();
    // No need to store a custom use-list order if the result does not have
    // multiple uses.
    if (value.use_empty() || value.hasOneUse())
      continue;

    // For each result, assemble the list of pairs (use-list-index,
    // global-value-index). While doing so, detect if the global-value-index is
    // already ordered with respect to the use-list-index.
    bool alreadyOrdered = true;
    auto &firstUse = *value.use_begin();
    uint64_t prevID = bytecode::getUseID(
        firstUse, numberingState.getNumber(firstUse.getOwner()));
    llvm::SmallVector<std::pair<unsigned, uint64_t>> useListPairs(
        {{0, prevID}});

    for (auto use : llvm::drop_begin(llvm::enumerate(value.getUses()))) {
      uint64_t currentID = bytecode::getUseID(
          use.value(), numberingState.getNumber(use.value().getOwner()));
      // The use-list order achieved when building the IR at parsing always
      // pushes new uses on front. Hence, if the order by unique ID is
      // monotonically decreasing, a roundtrip to bytecode preserves such order.
      alreadyOrdered &= (prevID > currentID);
      useListPairs.push_back({use.index(), currentID});
      prevID = currentID;
    }

    // Do not emit if the order is already sorted.
    if (alreadyOrdered)
      continue;

    // Sort the use indices by the unique ID indices in descending order.
    std::sort(
        useListPairs.begin(), useListPairs.end(),
        [](auto elem1, auto elem2) { return elem1.second > elem2.second; });

    map.try_emplace(item.index(), llvm::map_range(useListPairs, [](auto elem) {
                      return elem.first;
                    }));
  }

  if (map.empty())
    return;

  opEncodingMask |= bytecode::OpEncodingMask::kHasUseListOrders;
  // Emit the number of results that have a custom use-list order if the number
  // of results is greater than one.
  if (range.size() != 1) {
    emitter.emitVarInt(map.size(), "custom use-list size");
  }

  for (const auto &item : map) {
    auto resultIdx = item.getFirst();
    auto useListOrder = item.getSecond();

    // Compute the number of uses that are actually shuffled. If those are less
    // than half of the total uses, encoding the index pair `(src, dst)` is more
    // space efficient.
    size_t shuffledElements =
        llvm::count_if(llvm::enumerate(useListOrder),
                       [](auto item) { return item.index() != item.value(); });
    bool indexPairEncoding = shuffledElements < (useListOrder.size() / 2);

    // For single result, we don't need to store the result index.
    if (range.size() != 1)
      emitter.emitVarInt(resultIdx, "use-list result index");

    if (indexPairEncoding) {
      emitter.emitVarIntWithFlag(shuffledElements * 2, indexPairEncoding,
                                 "use-list index pair size");
      for (auto pair : llvm::enumerate(useListOrder)) {
        if (pair.index() != pair.value()) {
          emitter.emitVarInt(pair.value(), "use-list index pair first");
          emitter.emitVarInt(pair.index(), "use-list index pair second");
        }
      }
    } else {
      emitter.emitVarIntWithFlag(useListOrder.size(), indexPairEncoding,
                                 "use-list size");
      for (const auto &index : useListOrder)
        emitter.emitVarInt(index, "use-list order");
    }
  }
}

LogicalResult BytecodeWriter::writeRegion(EncodingEmitter &emitter,
                                          Region *region) {
  // If the region is empty, we only need to emit the number of blocks (which is
  // zero).
  if (region->empty()) {
    emitter.emitVarInt(/*numBlocks*/ 0, "region block count empty");
    return success();
  }

  // Emit the number of blocks and values within the region.
  unsigned numBlocks, numValues;
  std::tie(numBlocks, numValues) = numberingState.getBlockValueCount(region);
  emitter.emitVarInt(numBlocks, "region block count");
  emitter.emitVarInt(numValues, "region value count");

  // Emit the blocks within the region.
  for (Block &block : *region)
    if (failed(writeBlock(emitter, &block)))
      return failure();
  return success();
}

LogicalResult BytecodeWriter::writeIRSection(EncodingEmitter &emitter,
                                             Operation *op) {
  EncodingEmitter irEmitter;

  // Write the IR section the same way as a block with no arguments. Note that
  // the low-bit of the operation count for a block is used to indicate if the
  // block has arguments, which in this case is always false.
  irEmitter.emitVarIntWithFlag(/*numOps*/ 1, /*hasArgs*/ false, "ir section");

  // Emit the operations.
  if (failed(writeOp(irEmitter, op)))
    return failure();

  emitter.emitSection(bytecode::Section::kIR, std::move(irEmitter));
  return success();
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a resource builder implementation for the MLIR
/// bytecode format.
class ResourceBuilder : public AsmResourceBuilder {
public:
  using PostProcessFn = function_ref<void(StringRef, AsmResourceEntryKind)>;

  ResourceBuilder(EncodingEmitter &emitter, StringSectionBuilder &stringSection,
                  PostProcessFn postProcessFn, bool shouldElideData)
      : emitter(emitter), stringSection(stringSection),
        postProcessFn(postProcessFn), shouldElideData(shouldElideData) {}
  ~ResourceBuilder() override = default;

  void buildBlob(StringRef key, ArrayRef<char> data,
                 uint32_t dataAlignment) final {
    if (!shouldElideData)
      emitter.emitOwnedBlobAndAlignment(data, dataAlignment, "resource blob");
    postProcessFn(key, AsmResourceEntryKind::Blob);
  }
  void buildBool(StringRef key, bool data) final {
    if (!shouldElideData)
      emitter.emitByte(data, "resource bool");
    postProcessFn(key, AsmResourceEntryKind::Bool);
  }
  void buildString(StringRef key, StringRef data) final {
    if (!shouldElideData)
      emitter.emitVarInt(stringSection.insert(data), "resource string");
    postProcessFn(key, AsmResourceEntryKind::String);
  }

private:
  EncodingEmitter &emitter;
  StringSectionBuilder &stringSection;
  PostProcessFn postProcessFn;
  bool shouldElideData = false;
};
} // namespace

void BytecodeWriter::writeResourceSection(Operation *op,
                                          EncodingEmitter &emitter) {
  EncodingEmitter resourceEmitter;
  EncodingEmitter resourceOffsetEmitter;
  uint64_t prevOffset = 0;
  SmallVector<std::tuple<StringRef, AsmResourceEntryKind, uint64_t>>
      curResourceEntries;

  // Functor used to process the offset for a resource of `kind` defined by
  // 'key'.
  auto appendResourceOffset = [&](StringRef key, AsmResourceEntryKind kind) {
    uint64_t curOffset = resourceEmitter.size();
    curResourceEntries.emplace_back(key, kind, curOffset - prevOffset);
    prevOffset = curOffset;
  };

  // Functor used to emit a resource group defined by 'key'.
  auto emitResourceGroup = [&](uint64_t key) {
    resourceOffsetEmitter.emitVarInt(key, "resource group key");
    resourceOffsetEmitter.emitVarInt(curResourceEntries.size(),
                                     "resource group size");
    for (auto [key, kind, size] : curResourceEntries) {
      resourceOffsetEmitter.emitVarInt(stringSection.insert(key),
                                       "resource key");
      resourceOffsetEmitter.emitVarInt(size, "resource size");
      resourceOffsetEmitter.emitByte(kind, "resource kind");
    }
  };

  // Builder used to emit resources.
  ResourceBuilder entryBuilder(resourceEmitter, stringSection,
                               appendResourceOffset,
                               config.shouldElideResourceData);

  // Emit the external resource entries.
  resourceOffsetEmitter.emitVarInt(config.externalResourcePrinters.size(),
                                   "external resource printer count");
  for (const auto &printer : config.externalResourcePrinters) {
    curResourceEntries.clear();
    printer->buildResources(op, entryBuilder);
    emitResourceGroup(stringSection.insert(printer->getName()));
  }

  // Emit the dialect resource entries.
  for (DialectNumbering &dialect : numberingState.getDialects()) {
    if (!dialect.asmInterface)
      continue;
    curResourceEntries.clear();
    dialect.asmInterface->buildResources(op, dialect.resources, entryBuilder);

    // Emit the declaration resources for this dialect, these didn't get emitted
    // by the interface. These resources don't have data attached, so just use a
    // "blob" kind as a placeholder.
    for (const auto &resource : dialect.resourceMap)
      if (resource.second->isDeclaration)
        appendResourceOffset(resource.first, AsmResourceEntryKind::Blob);

    // Emit the resource group for this dialect.
    if (!curResourceEntries.empty())
      emitResourceGroup(dialect.number);
  }

  // If we didn't emit any resource groups, elide the resource sections.
  if (resourceOffsetEmitter.size() == 0)
    return;

  emitter.emitSection(bytecode::Section::kResourceOffset,
                      std::move(resourceOffsetEmitter));
  emitter.emitSection(bytecode::Section::kResource, std::move(resourceEmitter));
}

//===----------------------------------------------------------------------===//
// Strings
//===----------------------------------------------------------------------===//

void BytecodeWriter::writeStringSection(EncodingEmitter &emitter) {
  EncodingEmitter stringEmitter;
  stringSection.write(stringEmitter);
  emitter.emitSection(bytecode::Section::kString, std::move(stringEmitter));
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

void BytecodeWriter::writePropertiesSection(EncodingEmitter &emitter) {
  EncodingEmitter propertiesEmitter;
  propertiesSection.write(propertiesEmitter);
  emitter.emitSection(bytecode::Section::kProperties,
                      std::move(propertiesEmitter));
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

LogicalResult mlir::writeBytecodeToFile(Operation *op, raw_ostream &os,
                                        const BytecodeWriterConfig &config) {
  BytecodeWriter writer(op, config);
  return writer.write(op, os);
}
