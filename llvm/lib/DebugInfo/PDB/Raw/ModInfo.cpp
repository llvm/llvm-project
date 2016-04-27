//===- ModInfo.cpp - PDB module information -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;

namespace {
struct SCBytes {
  ulittle16_t Section;
  char Padding1[2];
  little32_t Offset;
  little32_t Size;
  ulittle32_t Characteristics;
  ulittle16_t ModuleIndex;
  char Padding2[2];
  ulittle32_t DataCrc;
  ulittle32_t RelocCrc;
};

// struct Flags {
//  uint16_t fWritten : 1;   // True if ModInfo is dirty
//  uint16_t fECEnabled : 1; // Is EC symbolic info present?  (What is EC?)
//  uint16_t unused : 6;     // Reserved
//  uint16_t iTSM : 8;       // Type Server Index for this module
//};
const uint16_t HasECFlagMask = 0x2;

const uint16_t TypeServerIndexMask = 0xFF00;
const uint16_t TypeServerIndexShift = 8;
}

struct ModInfo::FileLayout {
  ulittle32_t Mod;           // Currently opened module.  This field is a
                             // pointer in the reference implementation, but
                             // that won't work on 64-bit systems, and anyway
                             // it doesn't make sense to read a pointer from a
                             // file.  For now it is unused, so just ignore it.
  SCBytes SC;                // First section contribution of this module.
  ulittle16_t Flags;         // See Flags definition.
  ulittle16_t ModDiStream;   // Stream Number of module debug info
  ulittle32_t SymBytes;      // Size of local symbol debug info in above stream
  ulittle32_t LineBytes;     // Size of line number debug info in above stream
  ulittle32_t C13Bytes;      // Size of C13 line number info in above stream
  ulittle16_t NumFiles;      // Number of files contributing to this module
  char Padding1[2];          // Padding so the next field is 4-byte aligned.
  ulittle32_t FileNameOffs;  // array of [0..NumFiles) DBI name buffer offsets.
                             // This field is a pointer in the reference
                             // implementation, but as with `Mod`, we ignore it
                             // for now since it is unused.
  ulittle32_t SrcFileNameNI; // Name Index for src file name
  ulittle32_t PdbFilePathNI; // Name Index for path to compiler PDB
  char VarInfo[1];           // Module name followed by Obj File Name

  StringRef getModuleName() const { return StringRef(VarInfo); }

  StringRef getObjectFileName() const {
    return StringRef(getModuleName().end() + 1);
  }
};

ModInfo::ModInfo(const uint8_t *Bytes)
    : Layout(reinterpret_cast<const FileLayout *>(Bytes)) {}

ModInfo::~ModInfo() {}

bool ModInfo::hasECInfo() const { return (Layout->Flags & HasECFlagMask) != 0; }

uint16_t ModInfo::getTypeServerIndex() const {
  return (Layout->Flags & TypeServerIndexMask) >> TypeServerIndexShift;
}

uint16_t ModInfo::getModuleStreamIndex() const { return Layout->ModDiStream; }

uint32_t ModInfo::getSymbolDebugInfoByteSize() const {
  return Layout->SymBytes;
}

uint32_t ModInfo::getLineInfoByteSize() const { return Layout->LineBytes; }

uint32_t ModInfo::getC13LineInfoByteSize() const { return Layout->C13Bytes; }

uint32_t ModInfo::getNumberOfFiles() const { return Layout->NumFiles; }

uint32_t ModInfo::getSourceFileNameIndex() const {
  return Layout->SrcFileNameNI;
}

uint32_t ModInfo::getPdbFilePathNameIndex() const {
  return Layout->PdbFilePathNI;
}

llvm::StringRef ModInfo::getModuleName() const {
  return Layout->getModuleName();
}

llvm::StringRef ModInfo::getObjFileName() const {
  return Layout->getObjectFileName();
}

ModInfoIterator::ModInfoIterator(const uint8_t *Stream) : Bytes(Stream) {}

ModInfoIterator::ModInfoIterator(const ModInfoIterator &Other)
    : Bytes(Other.Bytes) {}

ModInfo ModInfoIterator::operator*() { return ModInfo(Bytes); }

ModInfoIterator &ModInfoIterator::operator++() {
  StringRef Obj = ModInfo(Bytes).getObjFileName();
  Bytes = Obj.bytes_end() + 1;
  Bytes = reinterpret_cast<const uint8_t *>(llvm::alignAddr(Bytes, 4));

  return *this;
}

ModInfoIterator ModInfoIterator::operator++(int) {
  ModInfoIterator Copy(*this);
  ++(*this);
  return Copy;
}

bool ModInfoIterator::operator==(const ModInfoIterator &Other) {
  return Bytes == Other.Bytes;
}

bool ModInfoIterator::operator!=(const ModInfoIterator &Other) {
  return !(*this == Other);
}

ModInfoIterator &ModInfoIterator::operator=(const ModInfoIterator &Other) {
  Bytes = Other.Bytes;
  return *this;
}
