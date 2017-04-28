//===- ModuleDebugFragment.h ------------------------------------*- C++ -*-===//
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
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

// Corresponds to the `CV_DebugSSubsectionHeader_t` structure.
struct ModuleDebugFragmentHeader {
  support::ulittle32_t Kind;   // codeview::ModuleDebugFragmentKind enum
  support::ulittle32_t Length; // number of bytes occupied by this record.
};

class ModuleDebugFragmentRecord {
public:
  ModuleDebugFragmentRecord();
  ModuleDebugFragmentRecord(ModuleDebugFragmentKind Kind, BinaryStreamRef Data);

  static Error initialize(BinaryStreamRef Stream,
                          ModuleDebugFragmentRecord &Info);
  uint32_t getRecordLength() const;
  ModuleDebugFragmentKind kind() const;
  BinaryStreamRef getRecordData() const;

private:
  ModuleDebugFragmentKind Kind;
  BinaryStreamRef Data;
};

typedef VarStreamArray<ModuleDebugFragmentRecord> ModuleDebugFragmentArray;

} // namespace codeview

template <>
struct VarStreamArrayExtractor<codeview::ModuleDebugFragmentRecord> {
  typedef void ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Length,
                       codeview::ModuleDebugFragmentRecord &Info, void *Ctx) {
    if (auto EC = codeview::ModuleDebugFragmentRecord::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTRECORD_H
