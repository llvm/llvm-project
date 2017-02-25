//===- ModuleSubstream.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"

#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

ModuleSubstream::ModuleSubstream() : Kind(ModuleSubstreamKind::None) {}

ModuleSubstream::ModuleSubstream(ModuleSubstreamKind Kind, BinaryStreamRef Data)
    : Kind(Kind), Data(Data) {}

Error ModuleSubstream::initialize(BinaryStreamRef Stream,
                                  ModuleSubstream &Info) {
  const ModuleSubsectionHeader *Header;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readObject(Header))
    return EC;

  ModuleSubstreamKind Kind =
      static_cast<ModuleSubstreamKind>(uint32_t(Header->Kind));
  if (auto EC = Reader.readStreamRef(Info.Data, Header->Length))
    return EC;
  Info.Kind = Kind;
  return Error::success();
}

uint32_t ModuleSubstream::getRecordLength() const {
  return sizeof(ModuleSubsectionHeader) + Data.getLength();
}

ModuleSubstreamKind ModuleSubstream::getSubstreamKind() const { return Kind; }

BinaryStreamRef ModuleSubstream::getRecordData() const { return Data; }
