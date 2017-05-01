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

static uint32_t calculateDiSymbolStreamSize(uint32_t SymbolByteSize) {
  uint32_t Size = sizeof(uint32_t); // Signature
  Size += SymbolByteSize;           // Symbol Data
  Size += 0;                        // TODO: Layout.LineBytes
  Size += 0;                        // TODO: Layout.C13Bytes
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
  Layout.C13Bytes = 0;
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
  auto ExpectedSN = MSF.addStream(calculateDiSymbolStreamSize(SymbolByteSize));
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
    // TODO: Write C13 Line data
    // TODO: Figure out what GlobalRefs substream actually is and populate it.
    if (auto EC = SymbolWriter.writeInteger<uint32_t>(0))
      return EC;
    if (SymbolWriter.bytesRemaining() > 0)
      return make_error<RawError>(raw_error_code::stream_too_long);
  }
  return Error::success();
}
