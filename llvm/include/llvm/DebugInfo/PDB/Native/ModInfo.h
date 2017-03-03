//===- ModInfo.h - PDB module information -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MODINFO_H
#define LLVM_DEBUGINFO_PDB_RAW_MODINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {

namespace pdb {

class ModInfo {
  friend class DbiStreamBuilder;

public:
  ModInfo();
  ModInfo(const ModInfo &Info);
  ~ModInfo();

  static Error initialize(BinaryStreamRef Stream, ModInfo &Info);

  bool hasECInfo() const;
  uint16_t getTypeServerIndex() const;
  uint16_t getModuleStreamIndex() const;
  uint32_t getSymbolDebugInfoByteSize() const;
  uint32_t getLineInfoByteSize() const;
  uint32_t getC13LineInfoByteSize() const;
  uint32_t getNumberOfFiles() const;
  uint32_t getSourceFileNameIndex() const;
  uint32_t getPdbFilePathNameIndex() const;

  StringRef getModuleName() const;
  StringRef getObjFileName() const;

  uint32_t getRecordLength() const;

private:
  StringRef ModuleName;
  StringRef ObjFileName;
  const ModuleInfoHeader *Layout = nullptr;
};

struct ModuleInfoEx {
  ModuleInfoEx(const ModInfo &Info) : Info(Info) {}
  ModuleInfoEx(const ModuleInfoEx &Ex) = default;

  ModInfo Info;
  std::vector<StringRef> SourceFiles;
};

} // end namespace pdb

template <> struct VarStreamArrayExtractor<pdb::ModInfo> {
  Error operator()(BinaryStreamRef Stream, uint32_t &Length,
                   pdb::ModInfo &Info) const {
    if (auto EC = pdb::ModInfo::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_MODINFO_H
