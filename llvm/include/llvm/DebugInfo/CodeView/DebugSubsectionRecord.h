//===- DebugSubsection.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTRECORD_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTRECORD_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class DebugSubsection;

// Corresponds to the `CV_DebugSSubsectionHeader_t` structure.
struct DebugSubsectionHeader {
  support::ulittle32_t Kind;   // codeview::DebugSubsectionKind enum
  support::ulittle32_t Length; // number of bytes occupied by this record.
};

class DebugSubsectionRecord {
public:
  DebugSubsectionRecord();
  DebugSubsectionRecord(DebugSubsectionKind Kind, BinaryStreamRef Data);

  static Error initialize(BinaryStreamRef Stream, DebugSubsectionRecord &Info);

  uint32_t getRecordLength() const;
  DebugSubsectionKind kind() const;
  BinaryStreamRef getRecordData() const;

private:
  DebugSubsectionKind Kind;
  BinaryStreamRef Data;
};

class DebugSubsectionRecordBuilder {
public:
  DebugSubsectionRecordBuilder(DebugSubsectionKind Kind, DebugSubsection &Frag);
  uint32_t calculateSerializedLength();
  Error commit(BinaryStreamWriter &Writer);

private:
  DebugSubsectionKind Kind;
  DebugSubsection &Frag;
};

} // namespace codeview

template <> struct VarStreamArrayExtractor<codeview::DebugSubsectionRecord> {
  typedef void ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Length,
                       codeview::DebugSubsectionRecord &Info) {
    if (auto EC = codeview::DebugSubsectionRecord::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};

namespace codeview {
typedef VarStreamArray<DebugSubsectionRecord> DebugSubsectionArray;
}
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTRECORD_H
