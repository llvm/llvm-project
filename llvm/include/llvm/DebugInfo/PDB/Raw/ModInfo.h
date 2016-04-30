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

#include <stdint.h>
#include <vector>

namespace llvm {
namespace pdb {
class ModInfo {
private:
  struct FileLayout;

public:
  ModInfo(const uint8_t *Bytes);
  ~ModInfo();

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

private:
  const FileLayout *Layout;
};

struct ModuleInfoEx {
  ModuleInfoEx(ModInfo Module) : Info(Module) {}

  ModInfo Info;
  std::vector<StringRef> SourceFiles;
};

class ModInfoIterator {
public:
  ModInfoIterator(const uint8_t *Stream);
  ModInfoIterator(const ModInfoIterator &Other);

  ModInfo operator*();
  ModInfoIterator &operator++();
  ModInfoIterator operator++(int);
  bool operator==(const ModInfoIterator &Other);
  bool operator!=(const ModInfoIterator &Other);
  ModInfoIterator &operator=(const ModInfoIterator &Other);

private:
  const uint8_t *Bytes;
};
}
}

#endif
