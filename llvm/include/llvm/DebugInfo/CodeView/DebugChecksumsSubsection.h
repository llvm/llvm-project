//===- DebugChecksumsSubsection.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_DEBUGCHECKSUMSSUBSECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_DEBUGCHECKSUMSSUBSECTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

class DebugStringTableSubsection;

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

  Error operator()(BinaryStreamRef Stream, uint32_t &Len,
                   codeview::FileChecksumEntry &Item);
};
}

namespace llvm {
namespace codeview {
class DebugChecksumsSubsectionRef final : public DebugSubsectionRef {
  typedef VarStreamArray<codeview::FileChecksumEntry> FileChecksumArray;
  typedef FileChecksumArray::Iterator Iterator;

public:
  DebugChecksumsSubsectionRef()
      : DebugSubsectionRef(DebugSubsectionKind::FileChecksums) {}

  static bool classof(const DebugSubsectionRef *S) {
    return S->kind() == DebugSubsectionKind::FileChecksums;
  }

  bool valid() const { return Checksums.valid(); }

  Error initialize(BinaryStreamReader Reader);
  Error initialize(BinaryStreamRef Stream);

  Iterator begin() const { return Checksums.begin(); }
  Iterator end() const { return Checksums.end(); }

  const FileChecksumArray &getArray() const { return Checksums; }

private:
  FileChecksumArray Checksums;
};

class DebugChecksumsSubsection final : public DebugSubsection {
public:
  explicit DebugChecksumsSubsection(DebugStringTableSubsection &Strings);

  static bool classof(const DebugSubsection *S) {
    return S->kind() == DebugSubsectionKind::FileChecksums;
  }

  void addChecksum(StringRef FileName, FileChecksumKind Kind,
                   ArrayRef<uint8_t> Bytes);

  uint32_t calculateSerializedSize() const override;
  Error commit(BinaryStreamWriter &Writer) const override;
  uint32_t mapChecksumOffset(StringRef FileName) const;

private:
  DebugStringTableSubsection &Strings;

  DenseMap<uint32_t, uint32_t> OffsetMap;
  uint32_t SerializedSize = 0;
  llvm::BumpPtrAllocator Storage;
  std::vector<FileChecksumEntry> Checksums;
};
}
}

#endif
