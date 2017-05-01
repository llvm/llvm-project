//===- ModuleDebugFileChecksumFragment.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFILECHECKSUMFRAGMENT_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFILECHECKSUMFRAGMENT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

struct FileChecksumEntry {
  uint32_t FileNameOffset;    // Byte offset of filename in global stringtable.
  FileChecksumKind Kind;      // The type of checksum.
  ArrayRef<uint8_t> Checksum; // The bytes of the checksum.
};
}
}

namespace llvm {
template <> struct VarStreamArrayExtractor<codeview::FileChecksumEntry> {
public:
  typedef void ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Len,
                       codeview::FileChecksumEntry &Item, void *Ctx);
};
}

namespace llvm {
namespace codeview {
class ModuleDebugFileChecksumFragmentRef final : public ModuleDebugFragmentRef {
  typedef VarStreamArray<codeview::FileChecksumEntry> FileChecksumArray;
  typedef FileChecksumArray::Iterator Iterator;

public:
  ModuleDebugFileChecksumFragmentRef()
      : ModuleDebugFragmentRef(ModuleDebugFragmentKind::FileChecksums) {}

  static bool classof(const ModuleDebugFragmentRef *S) {
    return S->kind() == ModuleDebugFragmentKind::FileChecksums;
  }

  Error initialize(BinaryStreamReader Reader);

  Iterator begin() const { return Checksums.begin(); }
  Iterator end() const { return Checksums.end(); }

  const FileChecksumArray &getArray() const { return Checksums; }

private:
  FileChecksumArray Checksums;
};
}
}

#endif
