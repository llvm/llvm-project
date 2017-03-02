//===- TpiHashing.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_TPIHASHING_H
#define LLVM_DEBUGINFO_PDB_TPIHASHING_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace pdb {

class TpiHashUpdater : public codeview::TypeVisitorCallbacks {
public:
  TpiHashUpdater() = default;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  virtual Error visitKnownRecord(codeview::CVType &CVR,                        \
                                 codeview::Name##Record &Record) override {    \
    visitKnownRecordImpl(CVR, Record);                                         \
    return Error::success();                                                   \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  template <typename RecordKind>
  void visitKnownRecordImpl(codeview::CVType &CVR, RecordKind &Record) {
    CVR.Hash = 0;
  }

  void visitKnownRecordImpl(codeview::CVType &CVR,
                            codeview::UdtSourceLineRecord &Rec);
  void visitKnownRecordImpl(codeview::CVType &CVR,
                            codeview::UdtModSourceLineRecord &Rec);
  void visitKnownRecordImpl(codeview::CVType &CVR, codeview::ClassRecord &Rec);
  void visitKnownRecordImpl(codeview::CVType &CVR, codeview::EnumRecord &Rec);
  void visitKnownRecordImpl(codeview::CVType &CVR, codeview::UnionRecord &Rec);
};

class TpiHashVerifier : public codeview::TypeVisitorCallbacks {
public:
  TpiHashVerifier(FixedStreamArray<support::ulittle32_t> &HashValues,
                  uint32_t NumHashBuckets)
      : HashValues(HashValues), NumHashBuckets(NumHashBuckets) {}

  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::UdtSourceLineRecord &Rec) override;
  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::UdtModSourceLineRecord &Rec) override;
  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::ClassRecord &Rec) override;
  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::EnumRecord &Rec) override;
  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::UnionRecord &Rec) override;
  Error visitTypeBegin(codeview::CVType &CVR) override;

private:
  Error verifySourceLine(codeview::TypeIndex TI);

  Error errorInvalidHash() {
    return make_error<RawError>(
        raw_error_code::invalid_tpi_hash,
        "Type index is 0x" +
            utohexstr(codeview::TypeIndex::FirstNonSimpleIndex + Index));
  }

  FixedStreamArray<support::ulittle32_t> HashValues;
  codeview::CVType RawRecord;
  uint32_t NumHashBuckets;
  uint32_t Index = -1;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_TPIHASHING_H
