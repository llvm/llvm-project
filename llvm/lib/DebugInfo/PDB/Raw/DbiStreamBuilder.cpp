//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace {
class ModiSubstreamBuilder {};
}

DbiStreamBuilder::DbiStreamBuilder(msf::MSFBuilder &Msf)
    : Msf(Msf), Allocator(Msf.getAllocator()), Age(1), BuildNumber(0),
      PdbDllVersion(0), PdbDllRbld(0), Flags(0), MachineType(PDB_Machine::x86),
      Header(nullptr) {}

void DbiStreamBuilder::setVersionHeader(PdbRaw_DbiVer V) { VerHeader = V; }

void DbiStreamBuilder::setAge(uint32_t A) { Age = A; }

void DbiStreamBuilder::setBuildNumber(uint16_t B) { BuildNumber = B; }

void DbiStreamBuilder::setPdbDllVersion(uint16_t V) { PdbDllVersion = V; }

void DbiStreamBuilder::setPdbDllRbld(uint16_t R) { PdbDllRbld = R; }

void DbiStreamBuilder::setFlags(uint16_t F) { Flags = F; }

void DbiStreamBuilder::setMachineType(PDB_Machine M) { MachineType = M; }

uint32_t DbiStreamBuilder::calculateSerializedLength() const {
  // For now we only support serializing the header.
  return sizeof(DbiStreamHeader) + calculateFileInfoSubstreamSize() +
         calculateModiSubstreamSize();
}

Error DbiStreamBuilder::addModuleInfo(StringRef ObjFile, StringRef Module) {
  auto Entry = llvm::make_unique<ModuleInfo>();
  ModuleInfo *M = Entry.get();
  Entry->Mod = Module;
  Entry->Obj = ObjFile;
  auto Result = ModuleInfos.insert(std::make_pair(Module, std::move(Entry)));
  if (!Result.second)
    return make_error<RawError>(raw_error_code::duplicate_entry,
                                "The specified module already exists");
  ModuleInfoList.push_back(M);
  return Error::success();
}

Error DbiStreamBuilder::addModuleSourceFile(StringRef Module, StringRef File) {
  auto ModIter = ModuleInfos.find(Module);
  if (ModIter == ModuleInfos.end())
    return make_error<RawError>(raw_error_code::no_entry,
                                "The specified module was not found");
  uint32_t Index = SourceFileNames.size();
  SourceFileNames.insert(std::make_pair(File, Index));
  auto &ModEntry = *ModIter;
  ModEntry.second->SourceFiles.push_back(File);
  return Error::success();
}

uint32_t DbiStreamBuilder::calculateModiSubstreamSize() const {
  uint32_t Size = 0;
  for (const auto &M : ModuleInfoList) {
    Size += sizeof(ModuleInfoHeader);
    Size += M->Mod.size() + 1;
    Size += M->Obj.size() + 1;
  }
  return Size;
}

uint32_t DbiStreamBuilder::calculateFileInfoSubstreamSize() const {
  uint32_t Size = 0;
  Size += sizeof(ulittle16_t);                         // NumModules
  Size += sizeof(ulittle16_t);                         // NumSourceFiles
  Size += ModuleInfoList.size() * sizeof(ulittle16_t); // ModIndices
  Size += ModuleInfoList.size() * sizeof(ulittle16_t); // ModFileCounts
  uint32_t NumFileInfos = 0;
  for (const auto &M : ModuleInfoList)
    NumFileInfos += M->SourceFiles.size();
  Size += NumFileInfos * sizeof(ulittle32_t); // FileNameOffsets
  Size += calculateNamesBufferSize();
  return Size;
}

uint32_t DbiStreamBuilder::calculateNamesBufferSize() const {
  uint32_t Size = 0;
  for (const auto &F : SourceFileNames) {
    Size += F.getKeyLength() + 1; // Names[I];
  }
  return Size;
}

Error DbiStreamBuilder::generateModiSubstream() {
  uint32_t Size = calculateModiSubstreamSize();
  auto Data = Allocator.Allocate<uint8_t>(Size);

  ModInfoBuffer = MutableByteStream(MutableArrayRef<uint8_t>(Data, Size));

  StreamWriter ModiWriter(ModInfoBuffer);
  for (const auto &M : ModuleInfoList) {
    ModuleInfoHeader Layout = {};
    Layout.ModDiStream = kInvalidStreamIndex;
    Layout.NumFiles = M->SourceFiles.size();
    if (auto EC = ModiWriter.writeObject(Layout))
      return EC;
    if (auto EC = ModiWriter.writeZeroString(M->Mod))
      return EC;
    if (auto EC = ModiWriter.writeZeroString(M->Obj))
      return EC;
  }
  if (ModiWriter.bytesRemaining() != 0)
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Unexpected bytes in Modi Stream Data");
  return Error::success();
}

Error DbiStreamBuilder::generateFileInfoSubstream() {
  uint32_t Size = calculateFileInfoSubstreamSize();
  uint32_t NameSize = calculateNamesBufferSize();
  auto Data = Allocator.Allocate<uint8_t>(Size);
  uint32_t NamesOffset = Size - NameSize;

  FileInfoBuffer = MutableByteStream(MutableArrayRef<uint8_t>(Data, Size));

  WritableStreamRef MetadataBuffer =
      WritableStreamRef(FileInfoBuffer).keep_front(NamesOffset);
  StreamWriter MetadataWriter(MetadataBuffer);

  uint16_t ModiCount = std::min<uint16_t>(UINT16_MAX, ModuleInfos.size());
  uint16_t FileCount = std::min<uint16_t>(UINT16_MAX, SourceFileNames.size());
  if (auto EC = MetadataWriter.writeInteger(ModiCount)) // NumModules
    return EC;
  if (auto EC = MetadataWriter.writeInteger(FileCount)) // NumSourceFiles
    return EC;
  for (uint16_t I = 0; I < ModiCount; ++I) {
    if (auto EC = MetadataWriter.writeInteger(I)) // Mod Indices
      return EC;
  }
  for (const auto MI : ModuleInfoList) {
    FileCount = static_cast<uint16_t>(MI->SourceFiles.size());
    if (auto EC = MetadataWriter.writeInteger(FileCount)) // Mod File Counts
      return EC;
  }

  // Before writing the FileNameOffsets array, write the NamesBuffer array.
  // A side effect of this is that this will actually compute the various
  // file name offsets, so we can then go back and write the FileNameOffsets
  // array to the other substream.
  NamesBuffer = WritableStreamRef(FileInfoBuffer).drop_front(NamesOffset);
  StreamWriter NameBufferWriter(NamesBuffer);
  for (auto &Name : SourceFileNames) {
    Name.second = NameBufferWriter.getOffset();
    if (auto EC = NameBufferWriter.writeZeroString(Name.getKey()))
      return EC;
  }

  for (const auto MI : ModuleInfoList) {
    for (StringRef Name : MI->SourceFiles) {
      auto Result = SourceFileNames.find(Name);
      if (Result == SourceFileNames.end())
        return make_error<RawError>(raw_error_code::no_entry,
                                    "The source file was not found.");
      if (auto EC = MetadataWriter.writeInteger(Result->second))
        return EC;
    }
  }

  if (NameBufferWriter.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::invalid_format,
                                "The names buffer contained unexpected data.");

  if (MetadataWriter.bytesRemaining() > 0)
    return make_error<RawError>(
        raw_error_code::invalid_format,
        "The metadata buffer contained unexpected data.");

  return Error::success();
}

Error DbiStreamBuilder::finalize() {
  if (Header)
    return Error::success();

  DbiStreamHeader *H = Allocator.Allocate<DbiStreamHeader>();

  if (auto EC = generateModiSubstream())
    return EC;
  if (auto EC = generateFileInfoSubstream())
    return EC;

  H->VersionHeader = *VerHeader;
  H->VersionSignature = -1;
  H->Age = Age;
  H->BuildNumber = BuildNumber;
  H->Flags = Flags;
  H->PdbDllRbld = PdbDllRbld;
  H->PdbDllVersion = PdbDllVersion;
  H->MachineType = static_cast<uint16_t>(MachineType);

  H->ECSubstreamSize = 0;
  H->FileInfoSize = FileInfoBuffer.getLength();
  H->ModiSubstreamSize = ModInfoBuffer.getLength();
  H->OptionalDbgHdrSize = 0;
  H->SecContrSubstreamSize = 0;
  H->SectionMapSize = 0;
  H->TypeServerSize = 0;
  H->SymRecordStreamIndex = kInvalidStreamIndex;
  H->PublicSymbolStreamIndex = kInvalidStreamIndex;
  H->MFCTypeServerIndex = kInvalidStreamIndex;
  H->GlobalSymbolStreamIndex = kInvalidStreamIndex;

  Header = H;
  return Error::success();
}

Error DbiStreamBuilder::finalizeMsfLayout() {
  uint32_t Length = calculateSerializedLength();
  if (auto EC = Msf.setStreamSize(StreamDBI, Length))
    return EC;
  return Error::success();
}

Expected<std::unique_ptr<DbiStream>>
DbiStreamBuilder::build(PDBFile &File, const msf::WritableStream &Buffer) {
  if (!VerHeader.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing DBI Stream Version");
  if (auto EC = finalize())
    return std::move(EC);

  auto StreamData = MappedBlockStream::createIndexedStream(File.getMsfLayout(),
                                                           Buffer, StreamDBI);
  auto Dbi = llvm::make_unique<DbiStream>(File, std::move(StreamData));
  Dbi->Header = Header;
  Dbi->FileInfoSubstream = ReadableStreamRef(FileInfoBuffer);
  Dbi->ModInfoSubstream = ReadableStreamRef(ModInfoBuffer);
  if (auto EC = Dbi->initializeModInfoArray())
    return std::move(EC);
  if (auto EC = Dbi->initializeFileInfo())
    return std::move(EC);
  return std::move(Dbi);
}

Error DbiStreamBuilder::commit(const msf::MSFLayout &Layout,
                               const msf::WritableStream &Buffer) {
  if (auto EC = finalize())
    return EC;

  auto InfoS =
      WritableMappedBlockStream::createIndexedStream(Layout, Buffer, StreamDBI);

  StreamWriter Writer(*InfoS);
  if (auto EC = Writer.writeObject(*Header))
    return EC;

  if (auto EC = Writer.writeStreamRef(ModInfoBuffer))
    return EC;
  if (auto EC = Writer.writeStreamRef(FileInfoBuffer))
    return EC;

  if (Writer.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Unexpected bytes found in DBI Stream");
  return Error::success();
}
