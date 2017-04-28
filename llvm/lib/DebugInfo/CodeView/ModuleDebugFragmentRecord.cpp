//===- ModuleDebugFragmentRecord.cpp -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"

#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

ModuleDebugFragmentRecord::ModuleDebugFragmentRecord()
    : Kind(ModuleDebugFragmentKind::None) {}

ModuleDebugFragmentRecord::ModuleDebugFragmentRecord(
    ModuleDebugFragmentKind Kind, BinaryStreamRef Data)
    : Kind(Kind), Data(Data) {}

Error ModuleDebugFragmentRecord::initialize(BinaryStreamRef Stream,
                                            ModuleDebugFragmentRecord &Info) {
  const ModuleDebugFragmentHeader *Header;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Header))
    return EC;

  ModuleDebugFragmentKind Kind =
      static_cast<ModuleDebugFragmentKind>(uint32_t(Header->Kind));
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t ModuleDebugFragmentRecord::getRecordLength() const {
  return sizeof(ModuleDebugFragmentHeader) + Data.getLength();
}

ModuleDebugFragmentKind ModuleDebugFragmentRecord::kind() const { return Kind; }

BinaryStreamRef ModuleDebugFragmentRecord::getRecordData() const {
  return Data;
}
