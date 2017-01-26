//===- TpiHashing.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"

#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

// Corresponds to `fUDTAnon`.
template <typename T> static bool isAnonymous(T &Rec) {
  StringRef Name = Rec.getName();
  return Name == "<unnamed-tag>" || Name == "__unnamed" ||
         Name.endswith("::<unnamed-tag>") || Name.endswith("::__unnamed");
}

// Computes a hash for a given TPI record.
template <typename T>
static uint32_t getTpiHash(T &Rec, ArrayRef<uint8_t> FullRecord) {
  auto Opts = static_cast<uint16_t>(Rec.getOptions());

  bool ForwardRef =
      Opts & static_cast<uint16_t>(ClassOptions::ForwardReference);
  bool Scoped = Opts & static_cast<uint16_t>(ClassOptions::Scoped);
  bool UniqueName = Opts & static_cast<uint16_t>(ClassOptions::HasUniqueName);
  bool IsAnon = UniqueName && isAnonymous(Rec);

  if (!ForwardRef && !Scoped && !IsAnon)
    return hashStringV1(Rec.getName());
  if (!ForwardRef && UniqueName && !IsAnon)
    return hashStringV1(Rec.getUniqueName());
  return hashBufferV8(FullRecord);
}

template <typename T> static uint32_t getSourceLineHash(T &Rec) {
  char Buf[4];
  support::endian::write32le(Buf, Rec.getUDT().getIndex());
  return hashStringV1(StringRef(Buf, 4));
}

void TpiHashUpdater::visitKnownRecordImpl(CVType &CVR,
                                          UdtSourceLineRecord &Rec) {
  CVR.Hash = getSourceLineHash(Rec);
}

void TpiHashUpdater::visitKnownRecordImpl(CVType &CVR,
                                          UdtModSourceLineRecord &Rec) {
  CVR.Hash = getSourceLineHash(Rec);
}

void TpiHashUpdater::visitKnownRecordImpl(CVType &CVR, ClassRecord &Rec) {
  CVR.Hash = getTpiHash(Rec, CVR.data());
}

void TpiHashUpdater::visitKnownRecordImpl(CVType &CVR, EnumRecord &Rec) {
  CVR.Hash = getTpiHash(Rec, CVR.data());
}

void TpiHashUpdater::visitKnownRecordImpl(CVType &CVR, UnionRecord &Rec) {
  CVR.Hash = getTpiHash(Rec, CVR.data());
}

Error TpiHashVerifier::visitKnownRecord(CVType &CVR, UdtSourceLineRecord &Rec) {
  return verifySourceLine(Rec.getUDT());
}

Error TpiHashVerifier::visitKnownRecord(CVType &CVR,
                                        UdtModSourceLineRecord &Rec) {
  return verifySourceLine(Rec.getUDT());
}

Error TpiHashVerifier::visitKnownRecord(CVType &CVR, ClassRecord &Rec) {
  if (getTpiHash(Rec, CVR.data()) % NumHashBuckets != HashValues[Index])
    return errorInvalidHash();
  return Error::success();
}
Error TpiHashVerifier::visitKnownRecord(CVType &CVR, EnumRecord &Rec) {
  if (getTpiHash(Rec, CVR.data()) % NumHashBuckets != HashValues[Index])
    return errorInvalidHash();
  return Error::success();
}
Error TpiHashVerifier::visitKnownRecord(CVType &CVR, UnionRecord &Rec) {
  if (getTpiHash(Rec, CVR.data()) % NumHashBuckets != HashValues[Index])
    return errorInvalidHash();
  return Error::success();
}

Error TpiHashVerifier::verifySourceLine(codeview::TypeIndex TI) {
  char Buf[4];
  support::endian::write32le(Buf, TI.getIndex());
  uint32_t Hash = hashStringV1(StringRef(Buf, 4));
  if (Hash % NumHashBuckets != HashValues[Index])
    return errorInvalidHash();
  return Error::success();
}

Error TpiHashVerifier::visitTypeBegin(CVType &Rec) {
  ++Index;
  RawRecord = Rec;
  return Error::success();
}
