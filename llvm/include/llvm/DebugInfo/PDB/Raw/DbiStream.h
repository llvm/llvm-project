//===- DbiStream.h - PDB Dbi Stream (Stream 3) Access -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAM_H

#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace object {
struct FpoData;
struct coff_section;
}

namespace pdb {
class DbiStreamBuilder;
class PDBFile;
class ISectionContribVisitor;

class DbiStream {
  friend class DbiStreamBuilder;

  struct HeaderInfo {
    support::little32_t VersionSignature;
    support::ulittle32_t VersionHeader;
    support::ulittle32_t Age;                     // Should match InfoStream.
    support::ulittle16_t GlobalSymbolStreamIndex; // Global symbol stream #
    support::ulittle16_t BuildNumber;             // See DbiBuildNo structure.
    support::ulittle16_t PublicSymbolStreamIndex; // Public symbols stream #
    support::ulittle16_t PdbDllVersion;           // version of mspdbNNN.dll
    support::ulittle16_t SymRecordStreamIndex;    // Symbol records stream #
    support::ulittle16_t PdbDllRbld;              // rbld number of mspdbNNN.dll
    support::little32_t ModiSubstreamSize;        // Size of module info stream
    support::little32_t SecContrSubstreamSize;    // Size of sec. contrib stream
    support::little32_t SectionMapSize;           // Size of sec. map substream
    support::little32_t FileInfoSize;             // Size of file info substream
    support::little32_t TypeServerSize;           // Size of type server map
    support::ulittle32_t MFCTypeServerIndex;      // Index of MFC Type Server
    support::little32_t OptionalDbgHdrSize;       // Size of DbgHeader info
    support::little32_t ECSubstreamSize; // Size of EC stream (what is EC?)
    support::ulittle16_t Flags;          // See DbiFlags enum.
    support::ulittle16_t MachineType;    // See PDB_MachineType enum.

    support::ulittle32_t Reserved; // Pad to 64 bytes
  };

public:
  DbiStream(PDBFile &File, std::unique_ptr<MappedBlockStream> Stream);
  ~DbiStream();
  Error reload();

  PdbRaw_DbiVer getDbiVersion() const;
  uint32_t getAge() const;
  uint16_t getPublicSymbolStreamIndex() const;
  uint16_t getGlobalSymbolStreamIndex() const;

  uint16_t getFlags() const;
  bool isIncrementallyLinked() const;
  bool hasCTypes() const;
  bool isStripped() const;

  uint16_t getBuildNumber() const;
  uint16_t getBuildMajorVersion() const;
  uint16_t getBuildMinorVersion() const;

  uint16_t getPdbDllRbld() const;
  uint32_t getPdbDllVersion() const;

  uint32_t getSymRecordStreamIndex() const;

  PDB_Machine getMachineType() const;

  enum { InvalidStreamIndex = 0xffff };

  /// If the given stream type is present, returns its stream index. If it is
  /// not present, returns InvalidStreamIndex.
  uint32_t getDebugStreamIndex(DbgHeaderType Type) const;

  ArrayRef<ModuleInfoEx> modules() const;

  Expected<StringRef> getFileNameForIndex(uint32_t Index) const;

  codeview::FixedStreamArray<object::coff_section> getSectionHeaders();

  codeview::FixedStreamArray<object::FpoData> getFpoRecords();

  codeview::FixedStreamArray<SecMapEntry> getSectionMap() const;
  void visitSectionContributions(ISectionContribVisitor &Visitor) const;

  Error commit();

private:
  Error initializeSectionContributionData();
  Error initializeSectionHeadersData();
  Error initializeSectionMapData();
  Error initializeFileInfo();
  Error initializeFpoRecords();

  PDBFile &Pdb;
  std::unique_ptr<MappedBlockStream> Stream;

  std::vector<ModuleInfoEx> ModuleInfos;
  NameHashTable ECNames;

  codeview::StreamRef ModInfoSubstream;
  codeview::StreamRef SecContrSubstream;
  codeview::StreamRef SecMapSubstream;
  codeview::StreamRef FileInfoSubstream;
  codeview::StreamRef TypeServerMapSubstream;
  codeview::StreamRef ECSubstream;

  codeview::StreamRef NamesBuffer;

  codeview::FixedStreamArray<support::ulittle16_t> DbgStreams;

  PdbRaw_DbiSecContribVer SectionContribVersion;
  codeview::FixedStreamArray<SectionContrib> SectionContribs;
  codeview::FixedStreamArray<SectionContrib2> SectionContribs2;
  codeview::FixedStreamArray<SecMapEntry> SectionMap;
  codeview::FixedStreamArray<support::little32_t> FileNameOffsets;

  std::unique_ptr<MappedBlockStream> SectionHeaderStream;
  codeview::FixedStreamArray<object::coff_section> SectionHeaders;

  std::unique_ptr<MappedBlockStream> FpoStream;
  codeview::FixedStreamArray<object::FpoData> FpoRecords;

  const HeaderInfo *Header;
};
}
}

#endif
