//===- DbiModuleDescriptorBuilder.cpp - PDB Mod Info Creation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptorBuilder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <limits>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace llvm {
namespace codeview {
class DebugSubsection;
}
} // namespace llvm

static uint32_t calculateDiSymbolStreamSize(uint32_t SymbolByteSize,
                                            uint32_t C13Size) {
  uint32_t Size = sizeof(uint32_t);   // Signature
  Size += alignTo(SymbolByteSize, 4); // Symbol Data
  Size += 0;                          // TODO: Layout.C11Bytes
  Size += C13Size;                    // C13 Debug Info Size
  Size += sizeof(uint32_t);           // GlobalRefs substream size (always 0)
  Size += 0;                          // GlobalRefs substream bytes
  return Size;
}

namespace {

/// Forwards a module's symbol bytes while replacing string-table references
/// before their physical MSF blocks are emitted.  This avoids the backward
/// seeks used by normal MSF output, which an append-only MSFZ writer cannot
/// support.
class StringTableFixupStream : public WritableBinaryStream {
public:
  StringTableFixupStream(WritableBinaryStreamRef Stream,
                         ArrayRef<StringTableFixup> Fixups,
                         uint32_t SymbolStreamSize)
      : Stream(Stream), Fixups(Fixups.begin(), Fixups.end()),
        SymbolStreamSize(SymbolStreamSize) {
    llvm::sort(this->Fixups,
               [](const StringTableFixup &L, const StringTableFixup &R) {
                 return L.SymOffsetOfReference < R.SymOffsetOfReference;
               });
  }

  Error validate() const {
    uint64_t PreviousEnd = 0;
    for (const StringTableFixup &Fixup : Fixups) {
      uint64_t Offset = Fixup.SymOffsetOfReference;
      if (Offset < sizeof(uint32_t) ||
          Offset > SymbolStreamSize - sizeof(uint32_t) || Offset < PreviousEnd)
        return make_error<RawError>(raw_error_code::invalid_format,
                                    "Invalid string table fixup");
      PreviousEnd = Offset + sizeof(uint32_t);
    }
    return Error::success();
  }

  llvm::endianness getEndian() const override { return Stream.getEndian(); }

  Error readBytes(uint64_t Offset, uint64_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    return BinaryStreamRef(Stream).readBytes(Offset, Size, Buffer);
  }

  Error readLongestContiguousChunk(uint64_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    return BinaryStreamRef(Stream).readLongestContiguousChunk(Offset, Buffer);
  }

  uint64_t getLength() override { return Stream.getLength(); }

  Error writeBytes(uint64_t Offset, ArrayRef<uint8_t> Data) override {
    if (Offset != InputOffset)
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Non-forward symbol stream write");
    if (Data.size() > SymbolStreamSize - InputOffset)
      return make_error<RawError>(raw_error_code::stream_too_long);

    while (!Data.empty()) {
      if (InputOffset < SkippedInputEnd) {
        uint32_t Bytes = std::min(static_cast<uint32_t>(Data.size()),
                                  SkippedInputEnd - InputOffset);
        InputOffset += Bytes;
        Data = Data.drop_front(Bytes);
        continue;
      }

      if (NextFixup != Fixups.size() &&
          Fixups[NextFixup].SymOffsetOfReference < InputOffset)
        return make_error<RawError>(raw_error_code::invalid_format,
                                    "Overlapping string table fixup");
      if (NextFixup != Fixups.size() &&
          Fixups[NextFixup].SymOffsetOfReference == InputOffset) {
        support::ulittle32_t Value;
        Value = Fixups[NextFixup].StrTabOffset;
        if (Error E = Stream.writeBytes(ForwardedOffset,
                                        bytesOf(&Value, sizeof(Value))))
          return E;
        ForwardedOffset += sizeof(Value);
        SkippedInputEnd = InputOffset + sizeof(Value);
        ++NextFixup;
        continue;
      }

      uint32_t Bytes = static_cast<uint32_t>(Data.size());
      if (NextFixup != Fixups.size())
        Bytes = std::min(Bytes,
                         Fixups[NextFixup].SymOffsetOfReference - InputOffset);
      if (Error E = Stream.writeBytes(ForwardedOffset, Data.take_front(Bytes)))
        return E;
      InputOffset += Bytes;
      ForwardedOffset += Bytes;
      Data = Data.drop_front(Bytes);
    }
    return Error::success();
  }

  Error commit() override { return Stream.commit(); }

  Error finish() const {
    if (InputOffset != SymbolStreamSize || SkippedInputEnd > InputOffset ||
        NextFixup != Fixups.size() || ForwardedOffset != SymbolStreamSize)
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Incomplete string table fixups");
    return Error::success();
  }

private:
  static ArrayRef<uint8_t> bytesOf(const void *Data, size_t Size) {
    return {reinterpret_cast<const uint8_t *>(Data), Size};
  }

  WritableBinaryStreamRef Stream;
  std::vector<StringTableFixup> Fixups;
  uint32_t SymbolStreamSize;
  uint32_t InputOffset = 0;
  uint32_t ForwardedOffset = 0;
  uint32_t SkippedInputEnd = 0;
  size_t NextFixup = 0;
};

} // namespace

DbiModuleDescriptorBuilder::DbiModuleDescriptorBuilder(StringRef ModuleName,
                                                       uint32_t ModIndex,
                                                       msf::MSFBuilder &Msf)
    : MSF(Msf), ModuleName(std::string(ModuleName)) {
  ::memset(&Layout, 0, sizeof(Layout));
  Layout.Mod = ModIndex;
}

DbiModuleDescriptorBuilder::~DbiModuleDescriptorBuilder() = default;

uint16_t DbiModuleDescriptorBuilder::getStreamIndex() const {
  return Layout.ModDiStream;
}

void DbiModuleDescriptorBuilder::setObjFileName(StringRef Name) {
  ObjFileName = std::string(Name);
}

void DbiModuleDescriptorBuilder::setPdbFilePathNI(uint32_t NI) {
  PdbFilePathNI = NI;
}

void DbiModuleDescriptorBuilder::setFirstSectionContrib(
    const SectionContrib &SC) {
  Layout.SC = SC;
}

void DbiModuleDescriptorBuilder::addSymbol(CVSymbol Symbol) {
  // Defer to the bulk API. It does the same thing.
  addSymbolsInBulk(Symbol.data());
}

void DbiModuleDescriptorBuilder::addSymbolsInBulk(
    ArrayRef<uint8_t> BulkSymbols) {
  // Do nothing for empty runs of symbols.
  if (BulkSymbols.empty())
    return;

  Symbols.push_back(SymbolListWrapper(BulkSymbols));
  // Symbols written to a PDB file are required to be 4 byte aligned. The same
  // is not true of object files.
  assert(BulkSymbols.size() % alignOf(CodeViewContainer::Pdb) == 0 &&
         "Invalid Symbol alignment!");
  SymbolByteSize += BulkSymbols.size();
}

void DbiModuleDescriptorBuilder::addUnmergedSymbols(void *SymSrc,
                                                    uint32_t SymLength) {
  assert(SymLength > 0);
  Symbols.push_back(SymbolListWrapper(SymSrc, SymLength));

  // Symbols written to a PDB file are required to be 4 byte aligned. The same
  // is not true of object files.
  assert(SymLength % alignOf(CodeViewContainer::Pdb) == 0 &&
         "Invalid Symbol alignment!");
  SymbolByteSize += SymLength;
}

void DbiModuleDescriptorBuilder::addSourceFile(StringRef Path) {
  SourceFiles.push_back(std::string(Path));
}

uint32_t DbiModuleDescriptorBuilder::calculateC13DebugInfoSize() const {
  uint32_t Result = 0;
  for (const auto &Builder : C13Builders) {
    Result += Builder.calculateSerializedLength();
  }
  return Result;
}

uint32_t DbiModuleDescriptorBuilder::calculateSerializedLength() const {
  uint32_t L = sizeof(Layout);
  uint32_t M = ModuleName.size() + 1;
  uint32_t O = ObjFileName.size() + 1;
  return alignTo(L + M + O, sizeof(uint32_t));
}

void DbiModuleDescriptorBuilder::finalize() {
  Layout.FileNameOffs = 0; // TODO: Fix this
  Layout.Flags = 0;        // TODO: Fix this
  Layout.C11Bytes = 0;
  Layout.C13Bytes = calculateC13DebugInfoSize();
  (void)Layout.Mod;         // Set in constructor
  (void)Layout.ModDiStream; // Set in finalizeMsfLayout
  Layout.NumFiles = SourceFiles.size();
  Layout.PdbFilePathNI = PdbFilePathNI;
  Layout.SrcFileNameNI = 0;

  // This value includes both the signature field as well as the record bytes
  // from the symbol stream.
  Layout.SymBytes =
      Layout.ModDiStream == kInvalidStreamIndex ? 0 : getNextSymbolOffset();
}

Error DbiModuleDescriptorBuilder::finalizeMsfLayout() {
  this->Layout.ModDiStream = kInvalidStreamIndex;
  uint32_t C13Size = calculateC13DebugInfoSize();
  if (!C13Size && !SymbolByteSize)
    return Error::success();
  auto ExpectedSN =
      MSF.addStream(calculateDiSymbolStreamSize(SymbolByteSize, C13Size));
  if (!ExpectedSN)
    return ExpectedSN.takeError();
  Layout.ModDiStream = *ExpectedSN;
  return Error::success();
}

Error DbiModuleDescriptorBuilder::commit(BinaryStreamWriter &ModiWriter) {
  // We write the Modi record to the `ModiWriter`, but we additionally write its
  // symbol stream to a brand new stream.
  if (auto EC = ModiWriter.writeObject(Layout))
    return EC;
  if (auto EC = ModiWriter.writeCString(ModuleName))
    return EC;
  if (auto EC = ModiWriter.writeCString(ObjFileName))
    return EC;
  if (auto EC = ModiWriter.padToAlignment(sizeof(uint32_t)))
    return EC;
  return Error::success();
}

Error DbiModuleDescriptorBuilder::commitSymbolStream(
    const msf::MSFLayout &MsfLayout, WritableBinaryStreamRef MsfBuffer) {
  return commitSymbolStream(MsfLayout, MsfBuffer, /*ForwardOnly=*/false);
}

Error DbiModuleDescriptorBuilder::commitSymbolStream(
    const msf::MSFLayout &MsfLayout, WritableBinaryStreamRef MsfBuffer,
    bool ForwardOnly) {
  if (Layout.ModDiStream == kInvalidStreamIndex)
    return Error::success();

  auto NS = WritableMappedBlockStream::createIndexedStream(
      MsfLayout, MsfBuffer, Layout.ModDiStream, MSF.getAllocator());
  return commitSymbolStream(WritableBinaryStreamRef(*NS), ForwardOnly);
}

Error DbiModuleDescriptorBuilder::commitSymbolStream(
    WritableBinaryStreamRef Stream, bool ForwardOnly) {
  auto WriteSymbols = [&](BinaryStreamWriter &Writer) -> Error {
    if (auto EC = Writer.writeInteger<uint32_t>(COFF::DEBUG_SECTION_MAGIC))
      return EC;
    for (const SymbolListWrapper &Sym : Symbols) {
      if (Sym.NeedsToBeMerged) {
        assert(MergeSymsCallback);
        if (auto EC = MergeSymsCallback(MergeSymsCtx, Sym.SymPtr, Writer))
          return EC;
      } else {
        if (auto EC = Writer.writeBytes(Sym.asArray()))
          return EC;
      }
    }
    return Error::success();
  };

  uint32_t SymbolEnd = 0;
  if (ForwardOnly) {
    if (SymbolByteSize >
        std::numeric_limits<uint32_t>::max() - sizeof(uint32_t))
      return make_error<RawError>(raw_error_code::stream_too_long);
    uint32_t SymbolStreamSize = sizeof(uint32_t) + SymbolByteSize;
    StringTableFixupStream FixupStream(Stream, StringTableFixups,
                                       SymbolStreamSize);
    if (Error E = FixupStream.validate())
      return E;
    BinaryStreamWriter SymbolWriter(FixupStream);
    if (Error E = WriteSymbols(SymbolWriter))
      return E;
    if (Error E = FixupStream.finish())
      return E;
    SymbolEnd = SymbolWriter.getOffset();
  } else {
    BinaryStreamWriter SymbolWriter(Stream);
    if (Error E = WriteSymbols(SymbolWriter))
      return E;

    auto SavedOffset = SymbolWriter.getOffset();
    for (const StringTableFixup &Fixup : StringTableFixups) {
      SymbolWriter.setOffset(Fixup.SymOffsetOfReference);
      if (Error E = SymbolWriter.writeInteger<uint32_t>(Fixup.StrTabOffset))
        return E;
    }
    SymbolWriter.setOffset(SavedOffset);
    SymbolEnd = SymbolWriter.getOffset();
  }

  assert(SymbolEnd % alignOf(CodeViewContainer::Pdb) == 0 &&
         "Invalid debug section alignment!");
  // String-table fixups only apply to symbols; later substreams can write
  // directly at the completed symbol offset.
  BinaryStreamWriter RemainingWriter(Stream);
  RemainingWriter.setOffset(SymbolEnd);
  // TODO: Write C11 Line data
  for (const auto &Builder : C13Builders) {
    if (auto EC = Builder.commit(RemainingWriter, CodeViewContainer::Pdb))
      return EC;
  }

  // TODO: Figure out what GlobalRefs substream actually is and populate it.
  if (auto EC = RemainingWriter.writeInteger<uint32_t>(0))
    return EC;
  if (RemainingWriter.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::stream_too_long);

  return Error::success();
}

void DbiModuleDescriptorBuilder::addDebugSubsection(
    std::shared_ptr<DebugSubsection> Subsection) {
  assert(Subsection);
  C13Builders.push_back(DebugSubsectionRecordBuilder(std::move(Subsection)));
}

void DbiModuleDescriptorBuilder::addDebugSubsection(
    const DebugSubsectionRecord &SubsectionContents) {
  C13Builders.push_back(DebugSubsectionRecordBuilder(SubsectionContents));
}
