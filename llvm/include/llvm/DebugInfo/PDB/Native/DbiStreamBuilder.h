//===- DbiStreamBuilder.h - PDB Dbi Stream Creation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_DBISTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_NATIVE_DBISTREAMBUILDER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include "llvm/DebugInfo/CodeView/DebugFrameDataSubsection.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamRef.h"

namespace llvm {

class BinaryStreamWriter;
namespace codeview {
struct FrameData;
}
namespace msf {
class MSFBuilder;
struct MSFLayout;
}
namespace pdb {
class DbiModuleDescriptorBuilder;

class DbiStreamBuilder {
public:
  LLVM_ABI DbiStreamBuilder(msf::MSFBuilder &Msf);
  LLVM_ABI ~DbiStreamBuilder();

  DbiStreamBuilder(const DbiStreamBuilder &) = delete;
  DbiStreamBuilder &operator=(const DbiStreamBuilder &) = delete;

  LLVM_ABI void setVersionHeader(PdbRaw_DbiVer V);
  LLVM_ABI void setAge(uint32_t A);
  LLVM_ABI void setBuildNumber(uint16_t B);
  LLVM_ABI void setBuildNumber(uint8_t Major, uint8_t Minor);
  LLVM_ABI void setPdbDllVersion(uint16_t V);
  LLVM_ABI void setPdbDllRbld(uint16_t R);
  LLVM_ABI void setFlags(uint16_t F);
  LLVM_ABI void setMachineType(PDB_Machine M);
  LLVM_ABI void setMachineType(COFF::MachineTypes M);

  // Add given bytes as a new stream.
  LLVM_ABI Error addDbgStream(pdb::DbgHeaderType Type, ArrayRef<uint8_t> Data);

  LLVM_ABI uint32_t addECName(StringRef Name);

  LLVM_ABI uint32_t calculateSerializedLength() const;

  LLVM_ABI void setGlobalsStreamIndex(uint32_t Index);
  LLVM_ABI void setPublicsStreamIndex(uint32_t Index);
  LLVM_ABI void setSymbolRecordStreamIndex(uint32_t Index);
  LLVM_ABI void addNewFpoData(const codeview::FrameData &FD);
  LLVM_ABI void addOldFpoData(const object::FpoData &Fpo);

  LLVM_ABI Expected<DbiModuleDescriptorBuilder &>
  addModuleInfo(StringRef ModuleName);
  LLVM_ABI Error addModuleSourceFile(DbiModuleDescriptorBuilder &Module,
                                     StringRef File);
  LLVM_ABI Expected<uint32_t> getSourceFileNameIndex(StringRef FileName);

  LLVM_ABI Error finalizeMsfLayout();

  LLVM_ABI Error commit(const msf::MSFLayout &Layout,
                        WritableBinaryStreamRef MsfBuffer);

  void addSectionContrib(const SectionContrib &SC) {
    SectionContribs.emplace_back(SC);
  }

  // Populate the Section Map from COFF section headers.
  LLVM_ABI void createSectionMap(ArrayRef<llvm::object::coff_section> SecHdrs);

private:
  struct DebugStream {
    std::function<Error(BinaryStreamWriter &)> WriteFn;
    uint32_t Size = 0;
    uint16_t StreamNumber = kInvalidStreamIndex;
  };

  Error finalize();
  uint32_t calculateModiSubstreamSize() const;
  uint32_t calculateNamesOffset() const;
  uint32_t calculateSectionContribsStreamSize() const;
  uint32_t calculateSectionMapStreamSize() const;
  uint32_t calculateFileInfoSubstreamSize() const;
  uint32_t calculateNamesBufferSize() const;
  uint32_t calculateDbgStreamsSize() const;

  Error generateFileInfoSubstream();

  msf::MSFBuilder &Msf;
  BumpPtrAllocator &Allocator;

  std::optional<PdbRaw_DbiVer> VerHeader;
  uint32_t Age;
  uint16_t BuildNumber;
  uint16_t PdbDllVersion;
  uint16_t PdbDllRbld;
  uint16_t Flags;
  PDB_Machine MachineType;
  uint32_t GlobalsStreamIndex = kInvalidStreamIndex;
  uint32_t PublicsStreamIndex = kInvalidStreamIndex;
  uint32_t SymRecordStreamIndex = kInvalidStreamIndex;

  const DbiStreamHeader *Header;

  std::vector<std::unique_ptr<DbiModuleDescriptorBuilder>> ModiList;

  std::optional<codeview::DebugFrameDataSubsection> NewFpoData;
  std::vector<object::FpoData> OldFpoData;

  StringMap<uint32_t> SourceFileNames;

  PDBStringTableBuilder ECNamesBuilder;
  WritableBinaryStreamRef NamesBuffer;
  MutableBinaryByteStream FileInfoBuffer;
  std::vector<SectionContrib> SectionContribs;
  std::vector<SecMapEntry> SectionMap;
  std::array<std::optional<DebugStream>, (int)DbgHeaderType::Max> DbgStreams;
};
} // namespace pdb
}

#endif
