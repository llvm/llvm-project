//===----- llvm/MC/DXContainerInfo.h - DXContainer Info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERINFO_H
#define LLVM_MC_DXCONTAINERINFO_H

#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/DXContainer.h"

namespace llvm {

class raw_ostream;

namespace mcdxbc {

struct DebugName {
  dxbc::DebugNameHeader Parameters;
  StringRef Filename;

  DebugName() : Parameters{0, 0} {}
  DebugName(dxbc::DebugNameHeader &Parameters, StringRef Filename)
      : Parameters(Parameters), Filename(Filename) {}

  LLVM_ABI void setFilename(StringRef DebugFilename);
  LLVM_ABI void write(raw_ostream &OS) const;
};

struct CompilerVersion {
  dxbc::CompilerVersionHeader Parameters;
  StringRef CommitSha;
  StringRef CustomVersionString;

  LLVM_ABI CompilerVersion();

  LLVM_ABI void setCommitSha(StringRef CommitSha);
  LLVM_ABI void setVersionString(StringRef VersionString);
  LLVM_ABI void write(raw_ostream &OS) const;

private:
  void updateContentSize();
};

struct SourceInfo {
  struct Section {
    dxbc::SourceInfo::SectionHeader GenericHeader;

    LLVM_ABI void computeGenericHeader(uint32_t ContentSize,
                                       dxbc::SourceInfo::SectionType Type);
  };

  struct SourceContents : public Section {
    struct Entry {
      dxbc::SourceInfo::Contents::Entry Parameters;
      std::string FileContent;

      /// Compute Parameters based on FileContent.
      LLVM_ABI void compute();
    };

    dxbc::SourceInfo::Contents::Header Parameters;
    SmallVector<Entry> Entries;

    /// Compute Parameters based on the content of Args.
    /// Sizes are computed assuming CompressionType == None.
    LLVM_ABI void
    computeUncompressed(dxbc::SourceInfo::Contents::CompressionType Type);
    /// Update Parameters based on the compressed size of section content.
    LLVM_ABI void computeFinalSize(uint32_t CompressedSize);
  };

  struct SourceNames : public Section {
    struct Header {
      uint32_t Flags;
      uint32_t Count;
      uint16_t EntriesSizeInBytes;

      Header() {}
      LLVM_ABI Header(const dxbc::SourceInfo::Names::HeaderOnDisk &H);

      void swapBytes() {
        sys::swapByteOrder(Flags);
        sys::swapByteOrder(Count);
        sys::swapByteOrder(EntriesSizeInBytes);
      }
    };

    struct Entry {
      dxbc::SourceInfo::Names::Entry Parameters;
      StringRef FileName;

      /// Compute Parameters based on FileName and FileContent.
      LLVM_ABI void compute(uint32_t ContentSize);
    };

    Header Parameters;
    SmallVector<Entry> Entries;

    /// Compute headers based on the content of entries.
    LLVM_ABI void compute();
  };

  struct ProgramArgs : public Section {
    using Entry = std::pair<StringRef, StringRef>;

    dxbc::SourceInfo::Args::Header Parameters;
    SmallVector<Entry> Args;

    /// Compute Parameters based on Args.
    LLVM_ABI void compute();
  };

  dxbc::SourceInfo::Header Parameters;
  SourceNames Names;
  SourceContents Contents;
  ProgramArgs Args;

  /// Compute Parameters based on the content of sections.
  LLVM_ABI void compute();
};

/// This data structure is a helper for reading and writing SourceInfo data.
/// This structure is used to represent the extracted data in an inspectable and
/// modifiable format, and can be used to serialize the data back into valid
/// SourceInfo.
struct SourceInfoBuilder {
  bool IsFilled = false;
  bool IsFinalized = false;
  SourceInfo BaseData;
  SmallString<128> CompressedContents;

  void setCompressionType(dxbc::SourceInfo::Contents::CompressionType Type) {
    CompressionType = Type;
  }

  void addFile(StringRef Name, StringRef Content) {
    FileNamesAndContents.emplace_back(Name, Content);
  }
  void addArg(StringRef Name, StringRef Value) {
    Args.emplace_back(Name, Value);
  }

  LLVM_ABI void computeEntries();
  LLVM_ABI void finalize();
  LLVM_ABI void write(raw_ostream &OS) const;

private:
  std::optional<dxbc::SourceInfo::Contents::CompressionType> CompressionType;
  SmallVector<std::pair<StringRef, StringRef>> FileNamesAndContents;
  SmallVector<std::pair<StringRef, StringRef>> Args;

  void recomputeAfterCompression(uint32_t CompressedSize);
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERINFO_H
