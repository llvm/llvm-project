//===- DebugSubsectionRecord.cpp -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/DebugSubsection.h"

#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

DebugSubsectionRecord::DebugSubsectionRecord()
    : Kind(DebugSubsectionKind::None) {}

DebugSubsectionRecord::DebugSubsectionRecord(DebugSubsectionKind Kind,
                                             BinaryStreamRef Data)
    : Kind(Kind), Data(Data) {}

Error DebugSubsectionRecord::initialize(BinaryStreamRef Stream,
                                        DebugSubsectionRecord &Info) {
  const DebugSubsectionHeader *Header;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Header))
    return EC;

  DebugSubsectionKind Kind =
      static_cast<DebugSubsectionKind>(uint32_t(Header->Kind));
  switch (Kind) {
  case DebugSubsectionKind::FileChecksums:
  case DebugSubsectionKind::Lines:
  case DebugSubsectionKind::InlineeLines:
    break;
  default:
    llvm_unreachable("Unexpected debug fragment kind!");
  }
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t DebugSubsectionRecord::getRecordLength() const {
  uint32_t Result = sizeof(DebugSubsectionHeader) + Data.getLength();
  assert(Result % 4 == 0);
  return Result;
}

DebugSubsectionKind DebugSubsectionRecord::kind() const { return Kind; }

BinaryStreamRef DebugSubsectionRecord::getRecordData() const { return Data; }

DebugSubsectionRecordBuilder::DebugSubsectionRecordBuilder(
    DebugSubsectionKind Kind, DebugSubsection &Frag)
    : Kind(Kind), Frag(Frag) {}

uint32_t DebugSubsectionRecordBuilder::calculateSerializedLength() {
  uint32_t Size = sizeof(DebugSubsectionHeader) +
                  alignTo(Frag.calculateSerializedSize(), 4);
  return Size;
}

Error DebugSubsectionRecordBuilder::commit(BinaryStreamWriter &Writer) {
  DebugSubsectionHeader Header;
  Header.Kind = uint32_t(Kind);
  Header.Length = calculateSerializedLength() - sizeof(DebugSubsectionHeader);

  if (auto EC = Writer.writeObject(Header))
    return EC;
  if (auto EC = Frag.commit(Writer))
    return EC;
  if (auto EC = Writer.padToAlignment(4))
    return EC;

  return Error::success();
}
