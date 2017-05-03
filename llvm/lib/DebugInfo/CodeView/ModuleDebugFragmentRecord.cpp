//===- ModuleDebugFragmentRecord.cpp -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"

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
  switch (Kind) {
  case ModuleDebugFragmentKind::FileChecksums:
  case ModuleDebugFragmentKind::Lines:
  case ModuleDebugFragmentKind::InlineeLines:
    break;
  default:
    llvm_unreachable("Unexpected debug fragment kind!");
  }
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t ModuleDebugFragmentRecord::getRecordLength() const {
  uint32_t Result = sizeof(ModuleDebugFragmentHeader) + Data.getLength();
  assert(Result % 4 == 0);
  return Result;
}

ModuleDebugFragmentKind ModuleDebugFragmentRecord::kind() const { return Kind; }

BinaryStreamRef ModuleDebugFragmentRecord::getRecordData() const {
  return Data;
}

ModuleDebugFragmentRecordBuilder::ModuleDebugFragmentRecordBuilder(
    ModuleDebugFragmentKind Kind, ModuleDebugFragment &Frag)
    : Kind(Kind), Frag(Frag) {}

uint32_t ModuleDebugFragmentRecordBuilder::calculateSerializedLength() {
  uint32_t Size = sizeof(ModuleDebugFragmentHeader) +
                  alignTo(Frag.calculateSerializedLength(), 4);
  return Size;
}

Error ModuleDebugFragmentRecordBuilder::commit(BinaryStreamWriter &Writer) {
  ModuleDebugFragmentHeader Header;
  Header.Kind = uint32_t(Kind);
  Header.Length =
      calculateSerializedLength() - sizeof(ModuleDebugFragmentHeader);

  if (auto EC = Writer.writeObject(Header))
    return EC;
  if (auto EC = Frag.commit(Writer))
    return EC;
  if (auto EC = Writer.padToAlignment(4))
    return EC;

  return Error::success();
}
