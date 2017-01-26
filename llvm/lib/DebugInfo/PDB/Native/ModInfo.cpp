//===- ModInfo.cpp - PDB module information -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/ModInfo.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::support;

ModInfo::ModInfo() = default;

ModInfo::ModInfo(const ModInfo &Info) = default;

ModInfo::~ModInfo() = default;

Error ModInfo::initialize(ReadableStreamRef Stream, ModInfo &Info) {
  StreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Info.Layout))
    return EC;

  if (auto EC = Reader.readZeroString(Info.ModuleName))
    return EC;

  if (auto EC = Reader.readZeroString(Info.ObjFileName))
    return EC;
  return Error::success();
}

bool ModInfo::hasECInfo() const {
  return (Layout->Flags & ModInfoFlags::HasECFlagMask) != 0;
}

uint16_t ModInfo::getTypeServerIndex() const {
  return (Layout->Flags & ModInfoFlags::TypeServerIndexMask) >>
         ModInfoFlags::TypeServerIndexShift;
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

StringRef ModInfo::getModuleName() const { return ModuleName; }

StringRef ModInfo::getObjFileName() const { return ObjFileName; }

uint32_t ModInfo::getRecordLength() const {
  uint32_t M = ModuleName.str().size() + 1;
  uint32_t O = ObjFileName.str().size() + 1;
  uint32_t Size = sizeof(ModuleInfoHeader) + M + O;
  Size = alignTo(Size, 4);
  return Size;
}
