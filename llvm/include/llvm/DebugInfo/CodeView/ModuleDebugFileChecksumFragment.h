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
#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

class StringTable;

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
                       codeview::FileChecksumEntry &Item);
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

  Iterator begin() { return Checksums.begin(); }
  Iterator end() { return Checksums.end(); }

  const FileChecksumArray &getArray() const { return Checksums; }

private:
  FileChecksumArray Checksums;
};

class ModuleDebugFileChecksumFragment final : public ModuleDebugFragment {
public:
  explicit ModuleDebugFileChecksumFragment(StringTable &Strings);

  static bool classof(const ModuleDebugFragment *S) {
    return S->kind() == ModuleDebugFragmentKind::FileChecksums;
  }

  void addChecksum(StringRef FileName, FileChecksumKind Kind,
                   ArrayRef<uint8_t> Bytes);

  uint32_t calculateSerializedLength() override;
  Error commit(BinaryStreamWriter &Writer) override;
  uint32_t mapChecksumOffset(StringRef FileName) const;

private:
  StringTable &Strings;

  DenseMap<uint32_t, uint32_t> OffsetMap;
  uint32_t SerializedSize = 0;
  llvm::BumpPtrAllocator Storage;
  std::vector<FileChecksumEntry> Checksums;
};
}
}

#endif
