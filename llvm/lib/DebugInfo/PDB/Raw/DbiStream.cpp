//===- DbiStream.cpp - PDB Dbi Stream (Stream 3) Access -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;
using namespace llvm::support;

namespace {
// Some of the values are stored in bitfields.  Since this needs to be portable
// across compilers and architectures (big / little endian in particular) we
// can't use the actual structures below, but must instead do the shifting
// and masking ourselves.  The struct definitions are provided for reference.

// struct DbiFlags {
//  uint16_t IncrementalLinking : 1;  // True if linked incrementally
//  uint16_t IsStripped : 1;          // True if private symbols were stripped.
//  uint16_t HasCTypes : 1;           // True if linked with /debug:ctypes.
//  uint16_t Reserved : 13;
//};
const uint16_t FlagIncrementalMask = 0x0001;
const uint16_t FlagStrippedMask = 0x0002;
const uint16_t FlagHasCTypesMask = 0x0004;

// struct DbiBuildNo {
//  uint16_t MinorVersion : 8;
//  uint16_t MajorVersion : 7;
//  uint16_t NewVersionFormat : 1;
//};
const uint16_t BuildMinorMask = 0x00FF;
const uint16_t BuildMinorShift = 0;

const uint16_t BuildMajorMask = 0x7F00;
const uint16_t BuildMajorShift = 8;
}

struct DbiStream::HeaderInfo {
  little32_t VersionSignature;
  ulittle32_t VersionHeader;
  ulittle32_t Age;                  // Should match InfoStream.
  ulittle16_t GSSyms;               // Number of global symbols
  ulittle16_t BuildNumber;          // See DbiBuildNo structure.
  ulittle16_t PSSyms;               // Number of public symbols
  ulittle16_t PdbDllVersion;        // version of mspdbNNN.dll
  ulittle16_t SymRecords;           // Number of symbols
  ulittle16_t PdbDllRbld;           // rbld number of mspdbNNN.dll
  little32_t ModiSubstreamSize;     // Size of module info stream
  little32_t SecContrSubstreamSize; // Size of sec. contribution stream
  little32_t SectionMapSize;        // Size of sec. map substream
  little32_t FileInfoSize;          // Size of file info substream
  little32_t TypeServerSize;      // Size of type server map
  ulittle32_t MFCTypeServerIndex; // Index of MFC Type Server
  little32_t OptionalDbgHdrSize;  // Size of DbgHeader info
  little32_t ECSubstreamSize;     // Size of EC stream (what is EC?)
  ulittle16_t Flags;              // See DbiFlags enum.
  ulittle16_t MachineType;        // See PDB_MachineType enum.

  ulittle32_t Reserved; // Pad to 64 bytes
};

DbiStream::DbiStream(PDBFile &File) : Pdb(File), Stream(3, File) {
  static_assert(sizeof(HeaderInfo) == 64, "Invalid HeaderInfo size!");
}

DbiStream::~DbiStream() {}

std::error_code DbiStream::reload() {
  StreamReader Reader(Stream);

  Header.reset(new HeaderInfo());

  if (Stream.getLength() < sizeof(HeaderInfo))
    return std::make_error_code(std::errc::illegal_byte_sequence);
  Reader.readObject(Header.get());

  if (Header->VersionSignature != -1)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  // Require at least version 7, which should be present in all PDBs
  // produced in the last decade and allows us to avoid having to
  // special case all kinds of complicated arcane formats.
  if (Header->VersionHeader < PdbDbiV70)
    return std::make_error_code(std::errc::not_supported);

  if (Header->Age != Pdb.getPDBInfoStream().getAge())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  if (Stream.getLength() !=
      sizeof(HeaderInfo) + Header->ModiSubstreamSize +
          Header->SecContrSubstreamSize + Header->SectionMapSize +
          Header->FileInfoSize + Header->TypeServerSize +
          Header->OptionalDbgHdrSize + Header->ECSubstreamSize)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  // Only certain substreams are guaranteed to be aligned.  Validate
  // them here.
  if (Header->ModiSubstreamSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  if (Header->SecContrSubstreamSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  if (Header->SectionMapSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  if (Header->FileInfoSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  if (Header->TypeServerSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  std::error_code EC;
  ModInfoSubstream.initialize(Reader, Header->ModiSubstreamSize);

  // Since each ModInfo in the stream is a variable length, we have to iterate
  // them to know how many there actually are.
  auto Range =
      llvm::make_range(ModInfoIterator(&ModInfoSubstream.data().front()),
                       ModInfoIterator(&ModInfoSubstream.data().back() + 1));
  for (auto Info : Range)
    ModuleInfos.push_back(ModuleInfoEx(Info));

  if ((EC =
           SecContrSubstream.initialize(Reader, Header->SecContrSubstreamSize)))
    return EC;
  if ((EC = SecMapSubstream.initialize(Reader, Header->SectionMapSize)))
    return EC;
  if ((EC = FileInfoSubstream.initialize(Reader, Header->FileInfoSize)))
    return EC;
  if ((EC = TypeServerMapSubstream.initialize(Reader, Header->TypeServerSize)))
    return EC;
  if ((EC = ECSubstream.initialize(Reader, Header->ECSubstreamSize)))
    return EC;
  if ((EC = DbgHeader.initialize(Reader, Header->OptionalDbgHdrSize)))
    return EC;

  if ((EC = initializeFileInfo()))
    return EC;

  if (Reader.bytesRemaining() > 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  return std::error_code();
}

PdbRaw_DbiVer DbiStream::getDbiVersion() const {
  uint32_t Value = Header->VersionHeader;
  return static_cast<PdbRaw_DbiVer>(Value);
}

uint32_t DbiStream::getAge() const { return Header->Age; }

bool DbiStream::isIncrementallyLinked() const {
  return (Header->Flags & FlagIncrementalMask) != 0;
}

bool DbiStream::hasCTypes() const {
  return (Header->Flags & FlagHasCTypesMask) != 0;
}

bool DbiStream::isStripped() const {
  return (Header->Flags & FlagStrippedMask) != 0;
}

uint16_t DbiStream::getBuildMajorVersion() const {
  return (Header->BuildNumber & BuildMajorMask) >> BuildMajorShift;
}

uint16_t DbiStream::getBuildMinorVersion() const {
  return (Header->BuildNumber & BuildMinorMask) >> BuildMinorShift;
}

uint32_t DbiStream::getPdbDllVersion() const { return Header->PdbDllVersion; }

uint32_t DbiStream::getNumberOfSymbols() const { return Header->SymRecords; }

PDB_Machine DbiStream::getMachineType() const {
  uint16_t Machine = Header->MachineType;
  return static_cast<PDB_Machine>(Machine);
}

ArrayRef<ModuleInfoEx> DbiStream::modules() const { return ModuleInfos; }

std::error_code DbiStream::initializeFileInfo() {
  struct FileInfoSubstreamHeader {
    ulittle16_t NumModules;     // Total # of modules, should match number of
                                // records in the ModuleInfo substream.
    ulittle16_t NumSourceFiles; // Total # of source files.  This value is not
                                // accurate because PDB actually supports more
                                // than 64k source files, so we ignore it and
                                // compute the value from other stream fields.
  };

  // The layout of the FileInfoSubstream is like this:
  // struct {
  //   ulittle16_t NumModules;
  //   ulittle16_t NumSourceFiles;
  //   ulittle16_t ModIndices[NumModules];
  //   ulittle16_t ModFileCounts[NumModules];
  //   ulittle32_t FileNameOffsets[NumSourceFiles];
  //   char Names[][NumSourceFiles];
  // };
  // with the caveat that `NumSourceFiles` cannot be trusted, so
  // it is computed by summing `ModFileCounts`.
  //
  const uint8_t *Buf = &FileInfoSubstream.data().front();
  auto FI = reinterpret_cast<const FileInfoSubstreamHeader *>(Buf);
  Buf += sizeof(FileInfoSubstreamHeader);
  // The number of modules in the stream should be the same as reported by
  // the FileInfoSubstreamHeader.
  if (FI->NumModules != ModuleInfos.size())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  // First is an array of `NumModules` module indices.  This is not used for the
  // same reason that `NumSourceFiles` is not used.  It's an array of uint16's,
  // but it's possible there are more than 64k source files, which would imply
  // more than 64k modules (e.g. object files) as well.  So we ignore this
  // field.
  llvm::ArrayRef<ulittle16_t> ModIndexArray(
      reinterpret_cast<const ulittle16_t *>(Buf), ModuleInfos.size());

  llvm::ArrayRef<ulittle16_t> ModFileCountArray(ModIndexArray.end(),
                                                ModuleInfos.size());

  // Compute the real number of source files.
  uint32_t NumSourceFiles = 0;
  for (auto Count : ModFileCountArray)
    NumSourceFiles += Count;

  // This is the array that in the reference implementation corresponds to
  // `ModInfo::FileLayout::FileNameOffs`, which is commented there as being a
  // pointer. Due to the mentioned problems of pointers causing difficulty
  // when reading from the file on 64-bit systems, we continue to ignore that
  // field in `ModInfo`, and instead build a vector of StringRefs and stores
  // them in `ModuleInfoEx`.  The value written to and read from the file is
  // not used anyway, it is only there as a way to store the offsets for the
  // purposes of later accessing the names at runtime.
  llvm::ArrayRef<little32_t> FileNameOffsets(
      reinterpret_cast<const little32_t *>(ModFileCountArray.end()),
      NumSourceFiles);

  const char *Names = reinterpret_cast<const char *>(FileNameOffsets.end());

  // We go through each ModuleInfo, determine the number N of source files for
  // that module, and then get the next N offsets from the Offsets array, using
  // them to get the corresponding N names from the Names buffer and associating
  // each one with the corresponding module.
  uint32_t NextFileIndex = 0;
  for (size_t I = 0; I < ModuleInfos.size(); ++I) {
    uint32_t NumFiles = ModFileCountArray[I];
    ModuleInfos[I].SourceFiles.resize(NumFiles);
    for (size_t J = 0; J < NumFiles; ++J, ++NextFileIndex) {
      uint32_t FileIndex = FileNameOffsets[NextFileIndex];
      ModuleInfos[I].SourceFiles[J] = StringRef(Names + FileIndex);
    }
  }

  return std::error_code();
}
