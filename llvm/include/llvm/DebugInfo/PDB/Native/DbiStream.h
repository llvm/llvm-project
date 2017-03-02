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
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/ModInfo.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/Native/StringTable.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamRef.h"
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

public:
  DbiStream(PDBFile &File, std::unique_ptr<msf::MappedBlockStream> Stream);
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

  /// If the given stream type is present, returns its stream index. If it is
  /// not present, returns InvalidStreamIndex.
  uint32_t getDebugStreamIndex(DbgHeaderType Type) const;

  ArrayRef<ModuleInfoEx> modules() const;

  Expected<StringRef> getFileNameForIndex(uint32_t Index) const;

  FixedStreamArray<object::coff_section> getSectionHeaders();

  FixedStreamArray<object::FpoData> getFpoRecords();

  FixedStreamArray<SecMapEntry> getSectionMap() const;
  void visitSectionContributions(ISectionContribVisitor &Visitor) const;

private:
  Error initializeModInfoArray();
  Error initializeSectionContributionData();
  Error initializeSectionHeadersData();
  Error initializeSectionMapData();
  Error initializeFileInfo();
  Error initializeFpoRecords();

  PDBFile &Pdb;
  std::unique_ptr<msf::MappedBlockStream> Stream;

  std::vector<ModuleInfoEx> ModuleInfos;
  StringTable ECNames;

  BinaryStreamRef ModInfoSubstream;
  BinaryStreamRef SecContrSubstream;
  BinaryStreamRef SecMapSubstream;
  BinaryStreamRef FileInfoSubstream;
  BinaryStreamRef TypeServerMapSubstream;
  BinaryStreamRef ECSubstream;

  BinaryStreamRef NamesBuffer;

  FixedStreamArray<support::ulittle16_t> DbgStreams;

  PdbRaw_DbiSecContribVer SectionContribVersion;
  FixedStreamArray<SectionContrib> SectionContribs;
  FixedStreamArray<SectionContrib2> SectionContribs2;
  FixedStreamArray<SecMapEntry> SectionMap;
  FixedStreamArray<support::little32_t> FileNameOffsets;

  std::unique_ptr<msf::MappedBlockStream> SectionHeaderStream;
  FixedStreamArray<object::coff_section> SectionHeaders;

  std::unique_ptr<msf::MappedBlockStream> FpoStream;
  FixedStreamArray<object::FpoData> FpoRecords;

  const DbiStreamHeader *Header;
};
}
}

#endif
