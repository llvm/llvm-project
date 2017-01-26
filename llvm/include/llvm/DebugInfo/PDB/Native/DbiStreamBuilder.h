//===- DbiStreamBuilder.h - PDB Dbi Stream Creation -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAMBUILDER_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include "llvm/DebugInfo/MSF/ByteStream.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace msf {
class MSFBuilder;
}
namespace object {
struct coff_section;
}
namespace pdb {
class DbiStream;
struct DbiStreamHeader;
class PDBFile;

class DbiStreamBuilder {
public:
  DbiStreamBuilder(msf::MSFBuilder &Msf);

  DbiStreamBuilder(const DbiStreamBuilder &) = delete;
  DbiStreamBuilder &operator=(const DbiStreamBuilder &) = delete;

  void setVersionHeader(PdbRaw_DbiVer V);
  void setAge(uint32_t A);
  void setBuildNumber(uint16_t B);
  void setPdbDllVersion(uint16_t V);
  void setPdbDllRbld(uint16_t R);
  void setFlags(uint16_t F);
  void setMachineType(PDB_Machine M);
  void setSectionContribs(ArrayRef<SectionContrib> SecMap);
  void setSectionMap(ArrayRef<SecMapEntry> SecMap);

  // Add given bytes as a new stream.
  Error addDbgStream(pdb::DbgHeaderType Type, ArrayRef<uint8_t> Data);

  uint32_t calculateSerializedLength() const;

  Error addModuleInfo(StringRef ObjFile, StringRef Module);
  Error addModuleSourceFile(StringRef Module, StringRef File);

  Error finalizeMsfLayout();

  Error commit(const msf::MSFLayout &Layout, const msf::WritableStream &Buffer);

  // A helper function to create Section Contributions from COFF input
  // section headers.
  static std::vector<SectionContrib>
  createSectionContribs(ArrayRef<llvm::object::coff_section> SecHdrs);

  // A helper function to create a Section Map from a COFF section header.
  static std::vector<SecMapEntry>
  createSectionMap(ArrayRef<llvm::object::coff_section> SecHdrs);

private:
  struct DebugStream {
    ArrayRef<uint8_t> Data;
    uint16_t StreamNumber = 0;
  };

  Error finalize();
  uint32_t calculateModiSubstreamSize() const;
  uint32_t calculateSectionContribsStreamSize() const;
  uint32_t calculateSectionMapStreamSize() const;
  uint32_t calculateFileInfoSubstreamSize() const;
  uint32_t calculateNamesBufferSize() const;
  uint32_t calculateDbgStreamsSize() const;

  Error generateModiSubstream();
  Error generateFileInfoSubstream();

  struct ModuleInfo {
    std::vector<StringRef> SourceFiles;
    StringRef Obj;
    StringRef Mod;
  };

  msf::MSFBuilder &Msf;
  BumpPtrAllocator &Allocator;

  Optional<PdbRaw_DbiVer> VerHeader;
  uint32_t Age;
  uint16_t BuildNumber;
  uint16_t PdbDllVersion;
  uint16_t PdbDllRbld;
  uint16_t Flags;
  PDB_Machine MachineType;

  const DbiStreamHeader *Header;

  StringMap<std::unique_ptr<ModuleInfo>> ModuleInfos;
  std::vector<ModuleInfo *> ModuleInfoList;

  StringMap<uint32_t> SourceFileNames;

  msf::WritableStreamRef NamesBuffer;
  msf::MutableByteStream ModInfoBuffer;
  msf::MutableByteStream FileInfoBuffer;
  ArrayRef<SectionContrib> SectionContribs;
  ArrayRef<SecMapEntry> SectionMap;
  llvm::SmallVector<DebugStream, (int)DbgHeaderType::Max> DbgStreams;
};
}
}

#endif
