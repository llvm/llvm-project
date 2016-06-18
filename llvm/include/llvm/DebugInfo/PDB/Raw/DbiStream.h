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
class PDBFile;
class ISectionContribVisitor;

class DbiStream {
  struct HeaderInfo;

public:
  DbiStream(PDBFile &File, std::unique_ptr<MappedBlockStream> Stream);
  ~DbiStream();
  Error reload();

  PdbRaw_DbiVer getDbiVersion() const;
  uint32_t getAge() const;
  uint16_t getPublicSymbolStreamIndex() const;
  uint16_t getGlobalSymbolStreamIndex() const;

  bool isIncrementallyLinked() const;
  bool hasCTypes() const;
  bool isStripped() const;

  uint16_t getBuildMajorVersion() const;
  uint16_t getBuildMinorVersion() const;

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
