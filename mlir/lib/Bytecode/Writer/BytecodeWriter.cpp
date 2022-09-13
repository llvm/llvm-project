//===- BytecodeWriter.cpp - MLIR Bytecode Writer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeWriter.h"
#include "../Encoding.h"
#include "IRNumbering.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <random>

#define DEBUG_TYPE "mlir-bytecode-writer"

using namespace mlir;
using namespace mlir::bytecode::detail;

//===----------------------------------------------------------------------===//
// BytecodeWriterConfig
//===----------------------------------------------------------------------===//

struct BytecodeWriterConfig::Impl {
  Impl(StringRef producer) : producer(producer) {}

  /// The producer of the bytecode.
  StringRef producer;

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
BytecodeWriterConfig::~BytecodeWriterConfig() = default;

void BytecodeWriterConfig::attachResourcePrinter(
    std::unique_ptr<AsmResourcePrinter> printer) {
  impl->externalResourcePrinters.emplace_back(std::move(printer));
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
  void patchByte(uint64_t offset, uint8_t value) {
    assert(offset < size() && offset >= prevResultSize &&
           "cannot patch previously emitted data");
    currentResult[offset - prevResultSize] = value;
  }

  /// Emit the provided blob of data, which is owned by the caller and is
  /// guaranteed to not die before the end of the bytecode process.
  void emitOwnedBlob(ArrayRef<uint8_t> data) {
    // Push the current buffer before adding the provided data.
    appendResult(std::move(currentResult));
    appendOwnedResult(data);
  }

  /// Emit the provided blob of data that has the given alignment, which is
  /// owned by the caller and is guaranteed to not die before the end of the
  /// bytecode process. The alignment value is also encoded, making it available
  /// on load.
  void emitOwnedBlobAndAlignment(ArrayRef<uint8_t> data, uint32_t alignment) {
    emitVarInt(alignment);
    emitVarInt(data.size());

    alignTo(alignment);
    emitOwnedBlob(data);
  }
  void emitOwnedBlobAndAlignment(ArrayRef<char> data, uint32_t alignment) {
    ArrayRef<uint8_t> castedData(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
    emitOwnedBlobAndAlignment(castedData, alignment);
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
      emitByte(bytecode::kAlignmentByte);

    // Keep track of the maximum required alignment.
    requiredAlignment = std::max(requiredAlignment, alignment);
  }

  //===--------------------------------------------------------------------===//
  // Integer Emission

  /// Emit a single byte.
  template <typename T>
  void emitByte(T byte) {
    currentResult.push_back(static_cast<uint8_t>(byte));
  }

  /// Emit a range of bytes.
  void emitBytes(ArrayRef<uint8_t> bytes) {
    llvm::append_range(currentResult, bytes);
  }

  /// Emit a variable length integer. The first encoded byte contains a prefix
  /// in the low bits indicating the encoded length of the value. This length
  /// prefix is a bit sequence of '0's followed by a '1'. The number of '0' bits
  /// indicate the number of _additional_ bytes (not including the prefix byte).
  /// All remaining bits in the first byte, along with all of the bits in
  /// additional bytes, provide the value of the integer encoded in
  /// little-endian order.
  void emitVarInt(uint64_t value) {
    // In the most common case, the value can be represented in a single byte.
    // Given how hot this case is, explicitly handle that here.
    if ((value >> 7) == 0)
      return emitByte((value << 1) | 0x1);
    emitMultiByteVarInt(value);
  }

  /// Emit a signed variable length integer. Signed varints are encoded using
  /// a varint with zigzag encoding, meaning that we use the low bit of the
  /// value to indicate the sign of the value. This allows for more efficient
  /// encoding of negative values by limiting the number of active bits
  void emitSignedVarInt(uint64_t value) {
    emitVarInt((value << 1) ^ (uint64_t)((int64_t)value >> 63));
  }

  /// Emit a variable length integer whose low bit is used to encode the
  /// provided flag, i.e. encoded as: (value << 1) | (flag ? 1 : 0).
  void emitVarIntWithFlag(uint64_t value, bool flag) {
    emitVarInt((value << 1) | (flag ? 1 : 0));
  }

  //===--------------------------------------------------------------------===//
  // String Emission

  /// Emit the given string as a nul terminated string.
  void emitNulTerminatedString(StringRef str) {
    emitString(str);
    emitByte(0);
  }

  /// Emit the given string without a nul terminator.
  void emitString(StringRef str) {
    emitBytes({reinterpret_cast<const uint8_t *>(str.data()), str.size()});
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
    emitByte(code);
    emitVarInt(emitter.size());

    // Integrate the alignment of the section into this emitter if necessary.
    unsigned emitterAlign = emitter.requiredAlignment;
    if (emitterAlign > 1) {
      if (size() & (emitterAlign - 1)) {
        emitVarInt(emitterAlign);
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
  LLVM_ATTRIBUTE_NOINLINE void emitMultiByteVarInt(uint64_t value);

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
    emitter.emitBytes({reinterpret_cast<const uint8_t *>(ptr), size});
  }
  uint64_t current_pos() const override { return emitter.size(); }

  /// The section being emitted to.
  EncodingEmitter &emitter;
};
} // namespace

void EncodingEmitter::writeTo(raw_ostream &os) const {
  for (auto &prevResult : prevResultList)
    os.write((const char *)prevResult.data(), prevResult.size());
  os.write((const char *)currentResult.data(), currentResult.size());
}

void EncodingEmitter::emitMultiByteVarInt(uint64_t value) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint64_t it = value >> 7;
  for (size_t numBytes = 2; numBytes < 9; ++numBytes) {
    if (LLVM_LIKELY(it >>= 7) == 0) {
      uint64_t encodedValue = (value << 1) | 0x1;
      encodedValue <<= (numBytes - 1);
      emitBytes({reinterpret_cast<uint8_t *>(&encodedValue), numBytes});
      return;
    }
  }

  // If the value is too large to encode in a single byte, emit a special all
  // zero marker byte and splat the value directly.
  emitByte(0);
  emitBytes({reinterpret_cast<uint8_t *>(&value), sizeof(value)});
}

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
    emitter.emitVarInt(strings.size());

    // Emit the sizes in reverse order, so that we don't need to backpatch an
    // offset to the string data or have a separate section.
    for (const auto &it : llvm::reverse(strings))
      emitter.emitVarInt(it.first.size() + 1);
    // Emit the string data itself.
    for (const auto &it : strings)
      emitter.emitNulTerminatedString(it.first.val());
  }

private:
  /// A set of strings referenced within the bytecode. The value of the map is
  /// unused.
  llvm::MapVector<llvm::CachedHashStringRef, size_t> strings;
};
} // namespace

//===----------------------------------------------------------------------===//
// Bytecode Writer
//===----------------------------------------------------------------------===//

namespace {
class BytecodeWriter {
public:
  BytecodeWriter(Operation *op) : numberingState(op) {}

  /// Write the bytecode for the given root operation.
  void write(Operation *rootOp, raw_ostream &os,
             const BytecodeWriterConfig::Impl &config);

private:
  //===--------------------------------------------------------------------===//
  // Dialects

  void writeDialectSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Attributes and Types

  void writeAttrTypeSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Operations

  void writeBlock(EncodingEmitter &emitter, Block *block);
  void writeOp(EncodingEmitter &emitter, Operation *op);
  void writeRegion(EncodingEmitter &emitter, Region *region);
  void writeIRSection(EncodingEmitter &emitter, Operation *op);

  //===--------------------------------------------------------------------===//
  // Resources

  void writeResourceSection(Operation *op, EncodingEmitter &emitter,
                            const BytecodeWriterConfig::Impl &config);

  //===--------------------------------------------------------------------===//
  // Strings

  void writeStringSection(EncodingEmitter &emitter);

  //===--------------------------------------------------------------------===//
  // Fields

  /// The builder used for the string section.
  StringSectionBuilder stringSection;

  /// The IR numbering state generated for the root operation.
  IRNumberingState numberingState;
};
} // namespace

void BytecodeWriter::write(Operation *rootOp, raw_ostream &os,
                           const BytecodeWriterConfig::Impl &config) {
  EncodingEmitter emitter;

  // Emit the bytecode file header. This is how we identify the output as a
  // bytecode file.
  emitter.emitString("ML\xefR");

  // Emit the bytecode version.
  emitter.emitVarInt(bytecode::kVersion);

  // Emit the producer.
  emitter.emitNulTerminatedString(config.producer);

  // Emit the dialect section.
  writeDialectSection(emitter);

  // Emit the attributes and types section.
  writeAttrTypeSection(emitter);

  // Emit the IR section.
  writeIRSection(emitter, rootOp);

  // Emit the resources section.
  writeResourceSection(rootOp, emitter, config);

  // Emit the string section.
  writeStringSection(emitter);

  // Write the generated bytecode to the provided output stream.
  emitter.writeTo(os);
}

//===----------------------------------------------------------------------===//
// Dialects

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
    emitter.emitVarInt(currentDialect->number);
    emitter.emitVarInt(std::distance(groupStart, it));

    // Emit the entries within the group.
    for (auto &entry : llvm::make_range(groupStart, it))
      callback(entry);
  }
}

void BytecodeWriter::writeDialectSection(EncodingEmitter &emitter) {
  EncodingEmitter dialectEmitter;

  // Emit the referenced dialects.
  auto dialects = numberingState.getDialects();
  dialectEmitter.emitVarInt(llvm::size(dialects));
  for (DialectNumbering &dialect : dialects)
    dialectEmitter.emitVarInt(stringSection.insert(dialect.name));

  // Emit the referenced operation names grouped by dialect.
  auto emitOpName = [&](OpNameNumbering &name) {
    dialectEmitter.emitVarInt(stringSection.insert(name.name.stripDialect()));
  };
  writeDialectGrouping(dialectEmitter, numberingState.getOpNames(), emitOpName);

  emitter.emitSection(bytecode::Section::kDialect, std::move(dialectEmitter));
}

//===----------------------------------------------------------------------===//
// Attributes and Types

namespace {
class DialectWriter : public DialectBytecodeWriter {
public:
  DialectWriter(EncodingEmitter &emitter, IRNumberingState &numberingState,
                StringSectionBuilder &stringSection)
      : emitter(emitter), numberingState(numberingState),
        stringSection(stringSection) {}

  //===--------------------------------------------------------------------===//
  // IR
  //===--------------------------------------------------------------------===//

  void writeAttribute(Attribute attr) override {
    emitter.emitVarInt(numberingState.getNumber(attr));
  }
  void writeType(Type type) override {
    emitter.emitVarInt(numberingState.getNumber(type));
  }

  void writeResourceHandle(const AsmDialectResourceHandle &resource) override {
    emitter.emitVarInt(numberingState.getNumber(resource));
  }

  //===--------------------------------------------------------------------===//
  // Primitives
  //===--------------------------------------------------------------------===//

  void writeVarInt(uint64_t value) override { emitter.emitVarInt(value); }

  void writeSignedVarInt(int64_t value) override {
    emitter.emitSignedVarInt(value);
  }

  void writeAPIntWithKnownWidth(const APInt &value) override {
    size_t bitWidth = value.getBitWidth();

    // If the value is a single byte, just emit it directly without going
    // through a varint.
    if (bitWidth <= 8)
      return emitter.emitByte(value.getLimitedValue());

    // If the value fits within a single varint, emit it directly.
    if (bitWidth <= 64)
      return emitter.emitSignedVarInt(value.getLimitedValue());

    // Otherwise, we need to encode a variable number of active words. We use
    // active words instead of the number of total words under the observation
    // that smaller values will be more common.
    unsigned numActiveWords = value.getActiveWords();
    emitter.emitVarInt(numActiveWords);

    const uint64_t *rawValueData = value.getRawData();
    for (unsigned i = 0; i < numActiveWords; ++i)
      emitter.emitSignedVarInt(rawValueData[i]);
  }

  void writeAPFloatWithKnownSemantics(const APFloat &value) override {
    writeAPIntWithKnownWidth(value.bitcastToAPInt());
  }

  void writeOwnedString(StringRef str) override {
    emitter.emitVarInt(stringSection.insert(str));
  }

private:
  EncodingEmitter &emitter;
  IRNumberingState &numberingState;
  StringSectionBuilder &stringSection;
};
} // namespace

void BytecodeWriter::writeAttrTypeSection(EncodingEmitter &emitter) {
  EncodingEmitter attrTypeEmitter;
  EncodingEmitter offsetEmitter;
  offsetEmitter.emitVarInt(llvm::size(numberingState.getAttributes()));
  offsetEmitter.emitVarInt(llvm::size(numberingState.getTypes()));

  // A functor used to emit an attribute or type entry.
  uint64_t prevOffset = 0;
  auto emitAttrOrType = [&](auto &entry) {
    auto entryValue = entry.getValue();

    // First, try to emit this entry using the dialect bytecode interface.
    bool hasCustomEncoding = false;
    if (const BytecodeDialectInterface *interface = entry.dialect->interface) {
      // The writer used when emitting using a custom bytecode encoding.
      DialectWriter dialectWriter(attrTypeEmitter, numberingState,
                                  stringSection);

      if constexpr (std::is_same_v<std::decay_t<decltype(entryValue)>, Type>) {
        // TODO: We don't currently support custom encoded mutable types.
        hasCustomEncoding =
            !entryValue.template hasTrait<TypeTrait::IsMutable>() &&
            succeeded(interface->writeType(entryValue, dialectWriter));
      } else {
        // TODO: We don't currently support custom encoded mutable attributes.
        hasCustomEncoding =
            !entryValue.template hasTrait<AttributeTrait::IsMutable>() &&
            succeeded(interface->writeAttribute(entryValue, dialectWriter));
      }
    }

    // If the entry was not emitted using the dialect interface, emit it using
    // the textual format.
    if (!hasCustomEncoding) {
      RawEmitterOstream(attrTypeEmitter) << entryValue;
      attrTypeEmitter.emitByte(0);
    }

    // Record the offset of this entry.
    uint64_t curOffset = attrTypeEmitter.size();
    offsetEmitter.emitVarIntWithFlag(curOffset - prevOffset, hasCustomEncoding);
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

void BytecodeWriter::writeBlock(EncodingEmitter &emitter, Block *block) {
  ArrayRef<BlockArgument> args = block->getArguments();
  bool hasArgs = !args.empty();

  // Emit the number of operations in this block, and if it has arguments. We
  // use the low bit of the operation count to indicate if the block has
  // arguments.
  unsigned numOps = numberingState.getOperationCount(block);
  emitter.emitVarIntWithFlag(numOps, hasArgs);

  // Emit the arguments of the block.
  if (hasArgs) {
    emitter.emitVarInt(args.size());
    for (BlockArgument arg : args) {
      emitter.emitVarInt(numberingState.getNumber(arg.getType()));
      emitter.emitVarInt(numberingState.getNumber(arg.getLoc()));
    }
  }

  // Emit the operations within the block.
  for (Operation &op : *block)
    writeOp(emitter, &op);
}

void BytecodeWriter::writeOp(EncodingEmitter &emitter, Operation *op) {
  emitter.emitVarInt(numberingState.getNumber(op->getName()));

  // Emit a mask for the operation components. We need to fill this in later
  // (when we actually know what needs to be emitted), so emit a placeholder for
  // now.
  uint64_t maskOffset = emitter.size();
  uint8_t opEncodingMask = 0;
  emitter.emitByte(0);

  // Emit the location for this operation.
  emitter.emitVarInt(numberingState.getNumber(op->getLoc()));

  // Emit the attributes of this operation.
  DictionaryAttr attrs = op->getAttrDictionary();
  if (!attrs.empty()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasAttrs;
    emitter.emitVarInt(numberingState.getNumber(op->getAttrDictionary()));
  }

  // Emit the result types of the operation.
  if (unsigned numResults = op->getNumResults()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasResults;
    emitter.emitVarInt(numResults);
    for (Type type : op->getResultTypes())
      emitter.emitVarInt(numberingState.getNumber(type));
  }

  // Emit the operands of the operation.
  if (unsigned numOperands = op->getNumOperands()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasOperands;
    emitter.emitVarInt(numOperands);
    for (Value operand : op->getOperands())
      emitter.emitVarInt(numberingState.getNumber(operand));
  }

  // Emit the successors of the operation.
  if (unsigned numSuccessors = op->getNumSuccessors()) {
    opEncodingMask |= bytecode::OpEncodingMask::kHasSuccessors;
    emitter.emitVarInt(numSuccessors);
    for (Block *successor : op->getSuccessors())
      emitter.emitVarInt(numberingState.getNumber(successor));
  }

  // Check for regions.
  unsigned numRegions = op->getNumRegions();
  if (numRegions)
    opEncodingMask |= bytecode::OpEncodingMask::kHasInlineRegions;

  // Update the mask for the operation.
  emitter.patchByte(maskOffset, opEncodingMask);

  // With the mask emitted, we can now emit the regions of the operation. We do
  // this after mask emission to avoid offset complications that may arise by
  // emitting the regions first (e.g. if the regions are huge, backpatching the
  // op encoding mask is more annoying).
  if (numRegions) {
    bool isIsolatedFromAbove = op->hasTrait<OpTrait::IsIsolatedFromAbove>();
    emitter.emitVarIntWithFlag(numRegions, isIsolatedFromAbove);

    for (Region &region : op->getRegions())
      writeRegion(emitter, &region);
  }
}

void BytecodeWriter::writeRegion(EncodingEmitter &emitter, Region *region) {
  // If the region is empty, we only need to emit the number of blocks (which is
  // zero).
  if (region->empty())
    return emitter.emitVarInt(/*numBlocks*/ 0);

  // Emit the number of blocks and values within the region.
  unsigned numBlocks, numValues;
  std::tie(numBlocks, numValues) = numberingState.getBlockValueCount(region);
  emitter.emitVarInt(numBlocks);
  emitter.emitVarInt(numValues);

  // Emit the blocks within the region.
  for (Block &block : *region)
    writeBlock(emitter, &block);
}

void BytecodeWriter::writeIRSection(EncodingEmitter &emitter, Operation *op) {
  EncodingEmitter irEmitter;

  // Write the IR section the same way as a block with no arguments. Note that
  // the low-bit of the operation count for a block is used to indicate if the
  // block has arguments, which in this case is always false.
  irEmitter.emitVarIntWithFlag(/*numOps*/ 1, /*hasArgs*/ false);

  // Emit the operations.
  writeOp(irEmitter, op);

  emitter.emitSection(bytecode::Section::kIR, std::move(irEmitter));
}

//===----------------------------------------------------------------------===//
// Resources

namespace {
/// This class represents a resource builder implementation for the MLIR
/// bytecode format.
class ResourceBuilder : public AsmResourceBuilder {
public:
  using PostProcessFn = function_ref<void(StringRef, AsmResourceEntryKind)>;

  ResourceBuilder(EncodingEmitter &emitter, StringSectionBuilder &stringSection,
                  PostProcessFn postProcessFn)
      : emitter(emitter), stringSection(stringSection),
        postProcessFn(postProcessFn) {}
  ~ResourceBuilder() override = default;

  void buildBlob(StringRef key, ArrayRef<char> data,
                 uint32_t dataAlignment) final {
    emitter.emitOwnedBlobAndAlignment(data, dataAlignment);
    postProcessFn(key, AsmResourceEntryKind::Blob);
  }
  void buildBool(StringRef key, bool data) final {
    emitter.emitByte(data);
    postProcessFn(key, AsmResourceEntryKind::Bool);
  }
  void buildString(StringRef key, StringRef data) final {
    emitter.emitVarInt(stringSection.insert(data));
    postProcessFn(key, AsmResourceEntryKind::String);
  }

private:
  EncodingEmitter &emitter;
  StringSectionBuilder &stringSection;
  PostProcessFn postProcessFn;
};
} // namespace

void BytecodeWriter::writeResourceSection(
    Operation *op, EncodingEmitter &emitter,
    const BytecodeWriterConfig::Impl &config) {
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
    resourceOffsetEmitter.emitVarInt(key);
    resourceOffsetEmitter.emitVarInt(curResourceEntries.size());
    for (auto [key, kind, size] : curResourceEntries) {
      resourceOffsetEmitter.emitVarInt(stringSection.insert(key));
      resourceOffsetEmitter.emitVarInt(size);
      resourceOffsetEmitter.emitByte(kind);
    }
  };

  // Builder used to emit resources.
  ResourceBuilder entryBuilder(resourceEmitter, stringSection,
                               appendResourceOffset);

  // Emit the external resource entries.
  resourceOffsetEmitter.emitVarInt(config.externalResourcePrinters.size());
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

void BytecodeWriter::writeStringSection(EncodingEmitter &emitter) {
  EncodingEmitter stringEmitter;
  stringSection.write(stringEmitter);
  emitter.emitSection(bytecode::Section::kString, std::move(stringEmitter));
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

void mlir::writeBytecodeToFile(Operation *op, raw_ostream &os,
                               const BytecodeWriterConfig &config) {
  BytecodeWriter writer(op);
  writer.write(op, os, config.getImpl());
}
