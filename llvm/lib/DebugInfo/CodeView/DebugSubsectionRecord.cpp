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
    : Container(CodeViewContainer::ObjectFile),
      Kind(DebugSubsectionKind::None) {}

DebugSubsectionRecord::DebugSubsectionRecord(DebugSubsectionKind Kind,
                                             BinaryStreamRef Data,
                                             CodeViewContainer Container)
    : Container(Container), Kind(Kind), Data(Data) {}

Error DebugSubsectionRecord::initialize(BinaryStreamRef Stream,
                                        DebugSubsectionRecord &Info,
                                        CodeViewContainer Container) {
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
  Info.Container = Container;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t DebugSubsectionRecord::getRecordLength() const {
  uint32_t Result = sizeof(DebugSubsectionHeader) + Data.getLength();
  assert(Result % alignOf(Container) == 0);
  return Result;
}

DebugSubsectionKind DebugSubsectionRecord::kind() const { return Kind; }

BinaryStreamRef DebugSubsectionRecord::getRecordData() const { return Data; }

DebugSubsectionRecordBuilder::DebugSubsectionRecordBuilder(
    std::unique_ptr<DebugSubsection> Subsection, CodeViewContainer Container)
    : Subsection(std::move(Subsection)), Container(Container) {}

uint32_t DebugSubsectionRecordBuilder::calculateSerializedLength() {
  uint32_t Size =
      sizeof(DebugSubsectionHeader) +
      alignTo(Subsection->calculateSerializedSize(), alignOf(Container));
  return Size;
}

Error DebugSubsectionRecordBuilder::commit(BinaryStreamWriter &Writer) {
  assert(Writer.getOffset() % alignOf(Container) == 0 &&
         "Debug Subsection not properly aligned");

  DebugSubsectionHeader Header;
  Header.Kind = uint32_t(Subsection->kind());
  Header.Length = calculateSerializedLength() - sizeof(DebugSubsectionHeader);

  if (auto EC = Writer.writeObject(Header))
    return EC;
  if (auto EC = Subsection->commit(Writer))
    return EC;
  if (auto EC = Writer.padToAlignment(alignOf(Container)))
    return EC;

  return Error::success();
}
