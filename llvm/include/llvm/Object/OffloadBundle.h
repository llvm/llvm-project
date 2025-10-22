//===- OffloadBundle.h - Utilities for offload bundles---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
//
// This file contains the binary format used for budingling device metadata with
// an associated device image. The data can then be stored inside a host object
// file to create a fat binary and read by the linker. This is intended to be a
// thin wrapper around the image itself. If this format becomes sufficiently
// complex it should be moved to a standard binary format like msgpack or ELF.
//
//===-------------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_OFFLOADBUNDLE_H
#define LLVM_OBJECT_OFFLOADBUNDLE_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {

namespace object {

// CompressedOffloadBundle represents the format for the compressed offload
// bundles.
//
// The format is as follows:
// - Magic Number (4 bytes) - A constant "CCOB".
// - Version (2 bytes)
// - Compression Method (2 bytes) - Uses the values from
// llvm::compression::Format.
// - Total file size (4 bytes in V2, 8 bytes in V3).
// - Uncompressed Size (4 bytes in V1/V2, 8 bytes in V3).
// - Truncated MD5 Hash (8 bytes).
// - Compressed Data (variable length).
class CompressedOffloadBundle {
private:
  static inline const llvm::StringRef MagicNumber = "CCOB";

public:
  struct CompressedBundleHeader {
    unsigned Version;
    llvm::compression::Format CompressionFormat;
    std::optional<size_t> FileSize;
    size_t UncompressedFileSize;
    uint64_t Hash;

    static llvm::Expected<CompressedBundleHeader> tryParse(llvm::StringRef);
  };

  static inline const uint16_t DefaultVersion = 3;

  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  compress(llvm::compression::Params P, const llvm::MemoryBuffer &Input,
           uint16_t Version, raw_ostream *VerboseStream = nullptr);
  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  decompress(const llvm::MemoryBuffer &Input,
             raw_ostream *VerboseStream = nullptr);
};

/// Bundle entry in binary clang-offload-bundler format.
struct OffloadBundleEntry {
  uint64_t Offset = 0u;
  uint64_t Size = 0u;
  uint64_t IDLength = 0u;
  std::string ID;
  OffloadBundleEntry(uint64_t O, uint64_t S, uint64_t I, StringRef T)
      : Offset(O), Size(S), IDLength(I), ID(T.str()) {}
  void dumpInfo(raw_ostream &OS) {
    OS << "Offset = " << Offset << ", Size = " << Size
       << ", ID Length = " << IDLength << ", ID = " << ID << "\n";
  }
  void dumpURI(raw_ostream &OS, StringRef FilePath) {
    OS << ID.data() << "\tfile://" << FilePath << "#offset=" << Offset
       << "&size=" << Size << "\n";
  }
};

/// Fat binary embedded in object files in clang-offload-bundler format
class OffloadBundleFatBin {

  uint64_t Size = 0u;
  StringRef FileName;
  uint64_t NumberOfEntries;
  bool Decompressed;
  SmallVector<OffloadBundleEntry> Entries;

public:
  std::unique_ptr<MemoryBuffer> DecompressedBuffer;

  SmallVector<OffloadBundleEntry> getEntries() { return Entries; }
  uint64_t getSize() const { return Size; }
  StringRef getFileName() const { return FileName; }
  uint64_t getNumEntries() const { return NumberOfEntries; }
  bool isDecompressed() const { return Decompressed; }

  LLVM_ABI static Expected<std::unique_ptr<OffloadBundleFatBin>>
  create(MemoryBufferRef, uint64_t SectionOffset, StringRef FileName,
         bool Decompress = false);
  LLVM_ABI Error extractBundle(const ObjectFile &Source);

  LLVM_ABI Error dumpEntryToCodeObject();

  LLVM_ABI Error readEntries(StringRef Section, uint64_t SectionOffset);
  void dumpEntries() {
    for (OffloadBundleEntry &Entry : Entries)
      Entry.dumpInfo(outs());
  }

  void printEntriesAsURI() {
    for (OffloadBundleEntry &Entry : Entries)
      Entry.dumpURI(outs(), FileName);
  }

  OffloadBundleFatBin(MemoryBufferRef Source, StringRef File,
                      bool Decompress = false)
      : FileName(File), NumberOfEntries(0), Decompressed(Decompress),
        Entries(SmallVector<OffloadBundleEntry>()) {
    if (Decompress)
      DecompressedBuffer =
          MemoryBuffer::getMemBufferCopy(Source.getBuffer(), File);
  }
};

enum UriTypeT { FILE_URI, MEMORY_URI };

struct OffloadBundleURI {
  int64_t Offset = 0;
  int64_t Size = 0;
  uint64_t ProcessID = 0;
  StringRef FileName;
  UriTypeT URIType;

  // Constructors
  // TODO: add a Copy ctor ?
  OffloadBundleURI(StringRef File, int64_t Off, int64_t Size)
      : Offset(Off), Size(Size), ProcessID(0), FileName(File),
        URIType(FILE_URI) {}

public:
  static Expected<std::unique_ptr<OffloadBundleURI>>
  createOffloadBundleURI(StringRef Str, UriTypeT Type) {
    switch (Type) {
    case FILE_URI:
      return createFileURI(Str);
      break;
    case MEMORY_URI:
      return createMemoryURI(Str);
      break;
    }
    llvm_unreachable("Unknown UriTypeT enum");
  }

  static Expected<std::unique_ptr<OffloadBundleURI>>
  createFileURI(StringRef Str) {
    int64_t O = 0;
    int64_t S = 0;

    if (!Str.consume_front("file://"))
      return createStringError(object_error::parse_failed,
                               "Reading type of URI");

    StringRef FilePathname =
        Str.take_until([](char C) { return (C == '#') || (C == '?'); });
    Str = Str.drop_front(FilePathname.size());

    if (!Str.consume_front("#offset="))
      return createStringError(object_error::parse_failed,
                               "Reading 'offset' in URI");

    StringRef OffsetStr = Str.take_until([](char C) { return C == '&'; });
    OffsetStr.getAsInteger(10, O);
    Str = Str.drop_front(OffsetStr.size());

    if (!Str.consume_front("&size="))
      return createStringError(object_error::parse_failed,
                               "Reading 'size' in URI");

    Str.getAsInteger(10, S);
    std::unique_ptr<OffloadBundleURI> OffloadingURI(
        new OffloadBundleURI(FilePathname, O, S));
    return std::move(OffloadingURI);
  }

  static Expected<std::unique_ptr<OffloadBundleURI>>
  createMemoryURI(StringRef Str) {
    // TODO: add parseMemoryURI type
    return createStringError(object_error::parse_failed,
                             "Memory Type URI is not currently supported.");
  }

  StringRef getFileName() const { return FileName; }
};

/// Extracts fat binary in binary clang-offload-bundler format from object \p
/// Obj and return it in \p Bundles
LLVM_ABI Error extractOffloadBundleFatBinary(
    const ObjectFile &Obj, SmallVectorImpl<OffloadBundleFatBin> &Bundles);

/// Extract code object memory from the given \p Source object file at \p Offset
/// and of \p Size, and copy into \p OutputFileName.
LLVM_ABI Error extractCodeObject(const ObjectFile &Source, int64_t Offset,
                                 int64_t Size, StringRef OutputFileName);

/// Extract code object memory from the given \p Source object file at \p Offset
/// and of \p Size, and copy into \p OutputFileName.
LLVM_ABI Error extractCodeObject(MemoryBufferRef Buffer, int64_t Offset,
                                 int64_t Size, StringRef OutputFileName);
/// Extracts an Offload Bundle Entry given by URI
LLVM_ABI Error extractOffloadBundleByURI(StringRef URIstr);

} // namespace object

} // namespace llvm
#endif
