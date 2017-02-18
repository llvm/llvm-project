//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/DbiStreamBuilder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"

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
      Header(nullptr), DbgStreams((int)DbgHeaderType::Max) {}

void DbiStreamBuilder::setVersionHeader(PdbRaw_DbiVer V) { VerHeader = V; }

void DbiStreamBuilder::setAge(uint32_t A) { Age = A; }

void DbiStreamBuilder::setBuildNumber(uint16_t B) { BuildNumber = B; }

void DbiStreamBuilder::setPdbDllVersion(uint16_t V) { PdbDllVersion = V; }

void DbiStreamBuilder::setPdbDllRbld(uint16_t R) { PdbDllRbld = R; }

void DbiStreamBuilder::setFlags(uint16_t F) { Flags = F; }

void DbiStreamBuilder::setMachineType(PDB_Machine M) { MachineType = M; }

void DbiStreamBuilder::setSectionContribs(ArrayRef<SectionContrib> Arr) {
  SectionContribs = Arr;
}

void DbiStreamBuilder::setSectionMap(ArrayRef<SecMapEntry> SecMap) {
  SectionMap = SecMap;
}

Error DbiStreamBuilder::addDbgStream(pdb::DbgHeaderType Type,
                                     ArrayRef<uint8_t> Data) {
  if (DbgStreams[(int)Type].StreamNumber)
    return make_error<RawError>(raw_error_code::duplicate_entry,
                                "The specified stream type already exists");
  auto ExpectedIndex = Msf.addStream(Data.size());
  if (!ExpectedIndex)
    return ExpectedIndex.takeError();
  uint32_t Index = std::move(*ExpectedIndex);
  DbgStreams[(int)Type].Data = Data;
  DbgStreams[(int)Type].StreamNumber = Index;
  return Error::success();
}

uint32_t DbiStreamBuilder::calculateSerializedLength() const {
  // For now we only support serializing the header.
  return sizeof(DbiStreamHeader) + calculateFileInfoSubstreamSize() +
         calculateModiSubstreamSize() + calculateSectionContribsStreamSize() +
         calculateSectionMapStreamSize() + calculateDbgStreamsSize();
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
  return alignTo(Size, sizeof(uint32_t));
}

uint32_t DbiStreamBuilder::calculateSectionContribsStreamSize() const {
  if (SectionContribs.empty())
    return 0;
  return sizeof(enum PdbRaw_DbiSecContribVer) +
         sizeof(SectionContribs[0]) * SectionContribs.size();
}

uint32_t DbiStreamBuilder::calculateSectionMapStreamSize() const {
  if (SectionMap.empty())
    return 0;
  return sizeof(SecMapHeader) + sizeof(SecMapEntry) * SectionMap.size();
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
  return alignTo(Size, sizeof(uint32_t));
}

uint32_t DbiStreamBuilder::calculateNamesBufferSize() const {
  uint32_t Size = 0;
  for (const auto &F : SourceFileNames) {
    Size += F.getKeyLength() + 1; // Names[I];
  }
  return Size;
}

uint32_t DbiStreamBuilder::calculateDbgStreamsSize() const {
  return DbgStreams.size() * sizeof(uint16_t);
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
  if (ModiWriter.bytesRemaining() > sizeof(uint32_t))
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

  uint16_t ModiCount = std::min<uint32_t>(UINT16_MAX, ModuleInfos.size());
  uint16_t FileCount = std::min<uint32_t>(UINT16_MAX, SourceFileNames.size());
  if (auto EC = MetadataWriter.writeInteger(
          ModiCount, llvm::support::little)) // NumModules
    return EC;
  if (auto EC = MetadataWriter.writeInteger(
          FileCount, llvm::support::little)) // NumSourceFiles
    return EC;
  for (uint16_t I = 0; I < ModiCount; ++I) {
    if (auto EC = MetadataWriter.writeInteger(
            I, llvm::support::little)) // Mod Indices
      return EC;
  }
  for (const auto MI : ModuleInfoList) {
    FileCount = static_cast<uint16_t>(MI->SourceFiles.size());
    if (auto EC = MetadataWriter.writeInteger(
            FileCount, llvm::support::little)) // Mod File Counts
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
      if (auto EC = MetadataWriter.writeInteger(Result->second,
                                                llvm::support::little))
        return EC;
    }
  }

  if (NameBufferWriter.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::invalid_format,
                                "The names buffer contained unexpected data.");

  if (MetadataWriter.bytesRemaining() > sizeof(uint32_t))
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
  H->OptionalDbgHdrSize = DbgStreams.size() * sizeof(uint16_t);
  H->SecContrSubstreamSize = calculateSectionContribsStreamSize();
  H->SectionMapSize = calculateSectionMapStreamSize();
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

static uint16_t toSecMapFlags(uint32_t Flags) {
  uint16_t Ret = 0;
  if (Flags & COFF::IMAGE_SCN_MEM_READ)
    Ret |= static_cast<uint16_t>(OMFSegDescFlags::Read);
  if (Flags & COFF::IMAGE_SCN_MEM_WRITE)
    Ret |= static_cast<uint16_t>(OMFSegDescFlags::Write);
  if (Flags & COFF::IMAGE_SCN_MEM_EXECUTE)
    Ret |= static_cast<uint16_t>(OMFSegDescFlags::Execute);
  if (Flags & COFF::IMAGE_SCN_MEM_EXECUTE)
    Ret |= static_cast<uint16_t>(OMFSegDescFlags::Execute);
  if (!(Flags & COFF::IMAGE_SCN_MEM_16BIT))
    Ret |= static_cast<uint16_t>(OMFSegDescFlags::AddressIs32Bit);

  // This seems always 1.
  Ret |= static_cast<uint16_t>(OMFSegDescFlags::IsSelector);

  return Ret;
}

// A utility function to create Section Contributions
// for a given input sections.
std::vector<SectionContrib> DbiStreamBuilder::createSectionContribs(
    ArrayRef<object::coff_section> SecHdrs) {
  std::vector<SectionContrib> Ret;

  // Create a SectionContrib for each input section.
  for (auto &Sec : SecHdrs) {
    Ret.emplace_back();
    auto &Entry = Ret.back();
    memset(&Entry, 0, sizeof(Entry));

    Entry.Off = Sec.PointerToRawData;
    Entry.Size = Sec.SizeOfRawData;
    Entry.Characteristics = Sec.Characteristics;
  }
  return Ret;
}

// A utility function to create a Section Map for a given list of COFF sections.
//
// A Section Map seem to be a copy of a COFF section list in other format.
// I don't know why a PDB file contains both a COFF section header and
// a Section Map, but it seems it must be present in a PDB.
std::vector<SecMapEntry> DbiStreamBuilder::createSectionMap(
    ArrayRef<llvm::object::coff_section> SecHdrs) {
  std::vector<SecMapEntry> Ret;
  int Idx = 0;

  auto Add = [&]() -> SecMapEntry & {
    Ret.emplace_back();
    auto &Entry = Ret.back();
    memset(&Entry, 0, sizeof(Entry));

    Entry.Frame = Idx + 1;

    // We don't know the meaning of these fields yet.
    Entry.SecName = UINT16_MAX;
    Entry.ClassName = UINT16_MAX;

    return Entry;
  };

  for (auto &Hdr : SecHdrs) {
    auto &Entry = Add();
    Entry.Flags = toSecMapFlags(Hdr.Characteristics);
    Entry.SecByteLength = Hdr.VirtualSize;
    ++Idx;
  }

  // The last entry is for absolute symbols.
  auto &Entry = Add();
  Entry.Flags = static_cast<uint16_t>(OMFSegDescFlags::AddressIs32Bit) |
                static_cast<uint16_t>(OMFSegDescFlags::IsAbsoluteAddress);
  Entry.SecByteLength = UINT32_MAX;

  return Ret;
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

  if (!SectionContribs.empty()) {
    if (auto EC = Writer.writeEnum(DbiSecContribVer60, llvm::support::little))
      return EC;
    if (auto EC = Writer.writeArray(SectionContribs))
      return EC;
  }

  if (!SectionMap.empty()) {
    ulittle16_t Size = static_cast<ulittle16_t>(SectionMap.size());
    SecMapHeader SMHeader = {Size, Size};
    if (auto EC = Writer.writeObject(SMHeader))
      return EC;
    if (auto EC = Writer.writeArray(SectionMap))
      return EC;
  }

  if (auto EC = Writer.writeStreamRef(FileInfoBuffer))
    return EC;

  for (auto &Stream : DbgStreams)
    if (auto EC =
            Writer.writeInteger(Stream.StreamNumber, llvm::support::little))
      return EC;

  for (auto &Stream : DbgStreams) {
    if (Stream.StreamNumber == kInvalidStreamIndex)
      continue;
    auto WritableStream = WritableMappedBlockStream::createIndexedStream(
        Layout, Buffer, Stream.StreamNumber);
    StreamWriter DbgStreamWriter(*WritableStream);
    if (auto EC = DbgStreamWriter.writeArray(Stream.Data))
      return EC;
  }

  if (Writer.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Unexpected bytes found in DBI Stream");
  return Error::success();
}
