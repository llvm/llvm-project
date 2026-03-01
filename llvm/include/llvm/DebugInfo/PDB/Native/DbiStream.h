//===- DbiStream.h - PDB Dbi Stream (Stream 3) Access -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_DBISTREAM_H
#define LLVM_DEBUGINFO_PDB_NATIVE_DBISTREAM_H

#include "llvm/DebugInfo/CodeView/DebugFrameDataSubsection.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleList.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
class BinaryStream;
namespace object {
struct FpoData;
struct coff_section;
}
namespace msf {
class MappedBlockStream;
}
namespace pdb {
struct DbiStreamHeader;
struct SecMapEntry;
struct SectionContrib2;
struct SectionContrib;
class PDBFile;
class ISectionContribVisitor;

class DbiStream {
  friend class DbiStreamBuilder;

public:
  LLVM_ABI explicit DbiStream(std::unique_ptr<BinaryStream> Stream);
  LLVM_ABI ~DbiStream();
  LLVM_ABI Error reload(PDBFile *Pdb);

  LLVM_ABI PdbRaw_DbiVer getDbiVersion() const;
  LLVM_ABI uint32_t getAge() const;
  LLVM_ABI uint16_t getPublicSymbolStreamIndex() const;
  LLVM_ABI uint16_t getGlobalSymbolStreamIndex() const;

  LLVM_ABI uint16_t getFlags() const;
  LLVM_ABI bool isIncrementallyLinked() const;
  LLVM_ABI bool hasCTypes() const;
  LLVM_ABI bool isStripped() const;

  LLVM_ABI uint16_t getBuildNumber() const;
  LLVM_ABI uint16_t getBuildMajorVersion() const;
  LLVM_ABI uint16_t getBuildMinorVersion() const;

  LLVM_ABI uint16_t getPdbDllRbld() const;
  LLVM_ABI uint32_t getPdbDllVersion() const;

  LLVM_ABI uint32_t getSymRecordStreamIndex() const;

  LLVM_ABI PDB_Machine getMachineType() const;

  const DbiStreamHeader *getHeader() const { return Header; }

  LLVM_ABI BinarySubstreamRef getSectionContributionData() const;
  LLVM_ABI BinarySubstreamRef getSecMapSubstreamData() const;
  LLVM_ABI BinarySubstreamRef getModiSubstreamData() const;
  LLVM_ABI BinarySubstreamRef getFileInfoSubstreamData() const;
  LLVM_ABI BinarySubstreamRef getTypeServerMapSubstreamData() const;
  LLVM_ABI BinarySubstreamRef getECSubstreamData() const;

  /// If the given stream type is present, returns its stream index. If it is
  /// not present, returns InvalidStreamIndex.
  LLVM_ABI uint32_t getDebugStreamIndex(DbgHeaderType Type) const;

  LLVM_ABI const DbiModuleList &modules() const;

  LLVM_ABI FixedStreamArray<object::coff_section> getSectionHeaders() const;

  LLVM_ABI bool hasOldFpoRecords() const;
  LLVM_ABI FixedStreamArray<object::FpoData> getOldFpoRecords() const;
  LLVM_ABI bool hasNewFpoRecords() const;
  LLVM_ABI const codeview::DebugFrameDataSubsectionRef &
  getNewFpoRecords() const;

  LLVM_ABI FixedStreamArray<SecMapEntry> getSectionMap() const;
  LLVM_ABI void
  visitSectionContributions(ISectionContribVisitor &Visitor) const;

  LLVM_ABI Expected<StringRef> getECName(uint32_t NI) const;

private:
  Error initializeSectionContributionData();
  Error initializeSectionHeadersData(PDBFile *Pdb);
  Error initializeSectionMapData();
  Error initializeOldFpoRecords(PDBFile *Pdb);
  Error initializeNewFpoRecords(PDBFile *Pdb);

  Expected<std::unique_ptr<msf::MappedBlockStream>>
  createIndexedStreamForHeaderType(PDBFile *Pdb, DbgHeaderType Type) const;

  std::unique_ptr<BinaryStream> Stream;

  PDBStringTable ECNames;

  BinarySubstreamRef SecContrSubstream;
  BinarySubstreamRef SecMapSubstream;
  BinarySubstreamRef ModiSubstream;
  BinarySubstreamRef FileInfoSubstream;
  BinarySubstreamRef TypeServerMapSubstream;
  BinarySubstreamRef ECSubstream;

  DbiModuleList Modules;

  FixedStreamArray<support::ulittle16_t> DbgStreams;

  PdbRaw_DbiSecContribVer SectionContribVersion =
      PdbRaw_DbiSecContribVer::DbiSecContribVer60;
  FixedStreamArray<SectionContrib> SectionContribs;
  FixedStreamArray<SectionContrib2> SectionContribs2;
  FixedStreamArray<SecMapEntry> SectionMap;

  std::unique_ptr<msf::MappedBlockStream> SectionHeaderStream;
  FixedStreamArray<object::coff_section> SectionHeaders;

  std::unique_ptr<msf::MappedBlockStream> OldFpoStream;
  FixedStreamArray<object::FpoData> OldFpoRecords;
  
  std::unique_ptr<msf::MappedBlockStream> NewFpoStream;
  codeview::DebugFrameDataSubsectionRef NewFpoRecords;

  const DbiStreamHeader *Header;
};
}
}

#endif
