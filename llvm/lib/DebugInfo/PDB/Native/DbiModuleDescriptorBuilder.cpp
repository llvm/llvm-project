//===- DbiModuleDescriptorBuilder.cpp - PDB Mod Info Creation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptorBuilder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/COFF.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace llvm {
template <> struct BinaryItemTraits<CVSymbol> {
  static size_t length(const CVSymbol &Item) { return Item.RecordData.size(); }

  static ArrayRef<uint8_t> bytes(const CVSymbol &Item) {
    return Item.RecordData;
  }
};
}

static uint32_t calculateDiSymbolStreamSize(uint32_t SymbolByteSize,
                                            uint32_t C13Size) {
  uint32_t Size = sizeof(uint32_t); // Signature
  Size += SymbolByteSize;           // Symbol Data
  Size += 0;                        // TODO: Layout.C11Bytes
  Size += C13Size;                  // C13 Debug Info Size
  Size += sizeof(uint32_t);         // GlobalRefs substream size (always 0)
  Size += 0;                        // GlobalRefs substream bytes
  return Size;
}

DbiModuleDescriptorBuilder::DbiModuleDescriptorBuilder(StringRef ModuleName,
                                                       uint32_t ModIndex,
                                                       msf::MSFBuilder &Msf)
    : MSF(Msf), ModuleName(ModuleName) {
  Layout.Mod = ModIndex;
}

DbiModuleDescriptorBuilder::~DbiModuleDescriptorBuilder() {}

uint16_t DbiModuleDescriptorBuilder::getStreamIndex() const {
  return Layout.ModDiStream;
}

void DbiModuleDescriptorBuilder::setObjFileName(StringRef Name) {
  ObjFileName = Name;
}

void DbiModuleDescriptorBuilder::addSymbol(CVSymbol Symbol) {
  Symbols.push_back(Symbol);
  SymbolByteSize += Symbol.data().size();
}

void DbiModuleDescriptorBuilder::addSourceFile(StringRef Path) {
  SourceFiles.push_back(Path);
}

uint32_t DbiModuleDescriptorBuilder::calculateC13DebugInfoSize() const {
  uint32_t Result = 0;
  for (const auto &Builder : C13Builders) {
    assert(Builder && "Empty C13 Fragment Builder!");
    Result += Builder->calculateSerializedLength();
  }
  return Result;
}

uint32_t DbiModuleDescriptorBuilder::calculateSerializedLength() const {
  uint32_t L = sizeof(Layout);
  uint32_t M = ModuleName.size() + 1;
  uint32_t O = ObjFileName.size() + 1;
  return alignTo(L + M + O, sizeof(uint32_t));
}

template <typename T> struct Foo {
  explicit Foo(T &&Answer) : Answer(Answer) {}

  T Answer;
};

template <typename T> Foo<T> makeFoo(T &&t) { return Foo<T>(std::move(t)); }

void DbiModuleDescriptorBuilder::finalize() {
  Layout.FileNameOffs = 0; // TODO: Fix this
  Layout.Flags = 0;        // TODO: Fix this
  Layout.C11Bytes = 0;
  Layout.C13Bytes = calculateC13DebugInfoSize();
  (void)Layout.Mod;         // Set in constructor
  (void)Layout.ModDiStream; // Set in finalizeMsfLayout
  Layout.NumFiles = SourceFiles.size();
  Layout.PdbFilePathNI = 0;
  Layout.SrcFileNameNI = 0;

  // This value includes both the signature field as well as the record bytes
  // from the symbol stream.
  Layout.SymBytes = SymbolByteSize + sizeof(uint32_t);
}

Error DbiModuleDescriptorBuilder::finalizeMsfLayout() {
  this->Layout.ModDiStream = kInvalidStreamIndex;
  uint32_t C13Size = calculateC13DebugInfoSize();
  auto ExpectedSN =
      MSF.addStream(calculateDiSymbolStreamSize(SymbolByteSize, C13Size));
  if (!ExpectedSN)
    return ExpectedSN.takeError();
  Layout.ModDiStream = *ExpectedSN;
  return Error::success();
}

Error DbiModuleDescriptorBuilder::commit(BinaryStreamWriter &ModiWriter,
                                         const msf::MSFLayout &MsfLayout,
                                         WritableBinaryStreamRef MsfBuffer) {
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

  if (Layout.ModDiStream != kInvalidStreamIndex) {
    auto NS = WritableMappedBlockStream::createIndexedStream(
        MsfLayout, MsfBuffer, Layout.ModDiStream);
    WritableBinaryStreamRef Ref(*NS);
    BinaryStreamWriter SymbolWriter(Ref);
    // Write the symbols.
    if (auto EC =
            SymbolWriter.writeInteger<uint32_t>(COFF::DEBUG_SECTION_MAGIC))
      return EC;
    BinaryItemStream<CVSymbol> Records(llvm::support::endianness::little);
    Records.setItems(Symbols);
    BinaryStreamRef RecordsRef(Records);
    if (auto EC = SymbolWriter.writeStreamRef(RecordsRef))
      return EC;
    // TODO: Write C11 Line data

    for (const auto &Builder : C13Builders) {
      assert(Builder && "Empty C13 Fragment Builder!");
      if (auto EC = Builder->commit(SymbolWriter))
        return EC;
    }

    // TODO: Figure out what GlobalRefs substream actually is and populate it.
    if (auto EC = SymbolWriter.writeInteger<uint32_t>(0))
      return EC;
    if (SymbolWriter.bytesRemaining() > 0)
      return make_error<RawError>(raw_error_code::stream_too_long);
  }
  return Error::success();
}

void DbiModuleDescriptorBuilder::addC13Fragment(
    std::unique_ptr<DebugLinesSubsection> Lines) {
  DebugLinesSubsection &Frag = *Lines;

  // File Checksums have to come first, so push an empty entry on if this
  // is the first.
  if (C13Builders.empty())
    C13Builders.push_back(nullptr);

  this->LineInfo.push_back(std::move(Lines));
  C13Builders.push_back(
      llvm::make_unique<DebugSubsectionRecordBuilder>(Frag.kind(), Frag));
}

void DbiModuleDescriptorBuilder::addC13Fragment(
    std::unique_ptr<codeview::DebugInlineeLinesSubsection> Inlinees) {
  DebugInlineeLinesSubsection &Frag = *Inlinees;

  // File Checksums have to come first, so push an empty entry on if this
  // is the first.
  if (C13Builders.empty())
    C13Builders.push_back(nullptr);

  this->Inlinees.push_back(std::move(Inlinees));
  C13Builders.push_back(
      llvm::make_unique<DebugSubsectionRecordBuilder>(Frag.kind(), Frag));
}

void DbiModuleDescriptorBuilder::setC13FileChecksums(
    std::unique_ptr<DebugChecksumsSubsection> Checksums) {
  assert(!ChecksumInfo && "Can't have more than one checksum info!");

  if (C13Builders.empty())
    C13Builders.push_back(nullptr);

  ChecksumInfo = std::move(Checksums);
  C13Builders[0] = llvm::make_unique<DebugSubsectionRecordBuilder>(
      ChecksumInfo->kind(), *ChecksumInfo);
}
