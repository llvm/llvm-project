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
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {

namespace object {

class CompressedOffloadBundle {
private:
  static inline const size_t MagicSize = 4;
  static inline const size_t VersionFieldSize = sizeof(uint16_t);
  static inline const size_t MethodFieldSize = sizeof(uint16_t);
  static inline const size_t FileSizeFieldSize = sizeof(uint32_t);
  static inline const size_t UncompressedSizeFieldSize = sizeof(uint32_t);
  static inline const size_t HashFieldSize = sizeof(uint64_t);
  static inline const size_t V1HeaderSize =
      MagicSize + VersionFieldSize + MethodFieldSize +
      UncompressedSizeFieldSize + HashFieldSize;
  static inline const size_t V2HeaderSize =
      MagicSize + VersionFieldSize + FileSizeFieldSize + MethodFieldSize +
      UncompressedSizeFieldSize + HashFieldSize;
  static inline const llvm::StringRef MagicNumber = "CCOB";
  static inline const uint16_t Version = 2;

public:
  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  compress(llvm::compression::Params P, const llvm::MemoryBuffer &Input,
           bool Verbose = false);
  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  decompress(llvm::MemoryBufferRef &Input, bool Verbose = false);
};

/// Bundle entry in binary clang-offload-bundler format.
struct OffloadBundleEntry {
  uint64_t Offset = 0u;
  uint64_t Size = 0u;
  uint64_t IDLength = 0u;
  StringRef ID;
  OffloadBundleEntry(uint64_t O, uint64_t S, uint64_t I, StringRef T)
      : Offset(O), Size(S), IDLength(I), ID(T) {}
  void dumpInfo(raw_ostream &OS) {
    OS << "Offset = " << Offset << ", Size = " << Size
       << ", ID Length = " << IDLength << ", ID = " << ID;
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
  SmallVector<OffloadBundleEntry> Entries;

public:
  SmallVector<OffloadBundleEntry> getEntries() { return Entries; }
  uint64_t getSize() const { return Size; }
  StringRef getFileName() const { return FileName; }
  uint64_t getNumEntries() const { return NumberOfEntries; }

  static Expected<std::unique_ptr<OffloadBundleFatBin>>
  create(MemoryBufferRef, uint64_t SectionOffset, StringRef FileName);
  Error extractBundle(const ObjectFile &Source);

  Error dumpEntryToCodeObject();

  Error readEntries(StringRef Section, uint64_t SectionOffset);
  void dumpEntries() {
    for (OffloadBundleEntry &Entry : Entries)
      Entry.dumpInfo(outs());
  }

  void printEntriesAsURI() {
    for (OffloadBundleEntry &Entry : Entries)
      Entry.dumpURI(outs(), FileName);
  }

  OffloadBundleFatBin(MemoryBufferRef Source, StringRef File)
      : FileName(File), NumberOfEntries(0),
        Entries(SmallVector<OffloadBundleEntry>()) {}
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

    if (Str.consume_front("&size="))
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
Error extractOffloadBundleFatBinary(
    const ObjectFile &Obj, SmallVectorImpl<OffloadBundleFatBin> &Bundles);

/// Extract code object memory from the given \p Source object file at \p Offset
/// and of \p Size, and copy into \p OutputFileName.
Error extractCodeObject(const ObjectFile &Source, int64_t Offset, int64_t Size,
                        StringRef OutputFileName);

/// Extracts an Offload Bundle Entry given by URI
Error extractOffloadBundleByURI(StringRef URIstr);

} // namespace object

} // namespace llvm
#endif
