//===--- Offloading.h - Utilities for handling offloading code  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the binary format used for bundling device metadata with
// an associated device image. The data can then be stored inside a host object
// file to create a fat binary and read by the linker. This is intended to be a
// thin wrapper around the image itself. If this format becomes sufficiently
// complex it should be moved to a standard binary format like msgpack or ELF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_OFFLOADBINARY_H
#define LLVM_OBJECT_OFFLOADBINARY_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {

namespace object {

/// The producer of the associated offloading image.
enum OffloadKind : uint16_t {
  OFK_None = 0,
  OFK_OpenMP = (1 << 0),
  OFK_Cuda = (1 << 1),
  OFK_HIP = (1 << 2),
  OFK_SYCL = (1 << 3),
  OFK_LAST = (1 << 4),
};

/// The type of contents the offloading image contains.
enum ImageKind : uint16_t {
  IMG_None = 0,
  IMG_Object,
  IMG_Bitcode,
  IMG_Cubin,
  IMG_Fatbinary,
  IMG_PTX,
  IMG_SPIRV,
  IMG_LAST,
};

/// Flags associated with the Entry.
enum OffloadEntryFlags : uint32_t {
  OIF_None = 0,
  // Entry doesn't contain image. Used to keep metadata only entries.
  OIF_NoImage = (1 << 0),
};

/// A simple binary serialization of an offloading file. We use this format to
/// embed the offloading image into the host executable so it can be extracted
/// and used by the linker.
///
/// Many of these could be stored in the same section by the time the linker
/// sees it so we mark this information with a header. The version is used to
/// detect ABI stability and the size is used to find other offloading entries
/// that may exist in the same section. All offsets are given as absolute byte
/// offsets from the beginning of the file.
class OffloadBinary : public Binary {
public:
  struct Header {
    uint8_t Magic[4] = {0x10, 0xFF, 0x10, 0xAD}; // 0x10FF10AD magic bytes.
    uint32_t Version = OffloadBinary::Version;   // Version identifier.
    uint64_t Size;          // Size in bytes of this entire binary.
    uint64_t EntriesOffset; // Offset in bytes to the start of entries block.
    uint64_t EntriesCount;  // Number of metadata entries in the binary.
  };

  struct Entry {
    ImageKind TheImageKind;     // The kind of the image stored.
    OffloadKind TheOffloadKind; // The producer of this image.
    uint32_t Flags;             // Additional flags associated with the entry.
    uint64_t StringOffset;      // Offset in bytes to the string map.
    uint64_t NumStrings;        // Number of entries in the string map.
    uint64_t ImageOffset;       // Offset in bytes of the actual binary image.
    uint64_t ImageSize;         // Size in bytes of the binary image.
  };

  struct StringEntry {
    uint64_t KeyOffset;
    uint64_t ValueOffset;
    uint64_t ValueSize; // Size of the value in bytes.
  };

  using StringMap = MapVector<StringRef, StringRef>;
  using entry_iterator = SmallVector<std::pair<const Entry *, StringMap>, 1>::const_iterator;
  using entry_iterator_range = iterator_range<entry_iterator>;
  using string_iterator = MapVector<StringRef, StringRef>::const_iterator;
  using string_iterator_range = iterator_range<string_iterator>;

  /// The current version of the binary used for backwards compatibility.
  static const uint32_t Version = 2;

  /// The offloading metadata that will be serialized to a memory buffer.
  struct OffloadingImage {
    ImageKind TheImageKind = ImageKind::IMG_None;
    OffloadKind TheOffloadKind = OffloadKind::OFK_None;
    uint32_t Flags = 0;
    MapVector<StringRef, StringRef> StringData;
    std::unique_ptr<MemoryBuffer> Image;
  };

  /// Attempt to parse the offloading binary stored in \p Data.
  LLVM_ABI static Expected<std::unique_ptr<OffloadBinary>>
      create(MemoryBufferRef);

  /// Serialize the contents of \p OffloadingData to a binary buffer to be read
  /// later.
  LLVM_ABI static SmallString<0>
  write(ArrayRef<OffloadingImage> OffloadingData);

  static uint64_t getAlignment() { return 8; }

  ImageKind getImageKind() const { return Entries[Index].first->TheImageKind; }
  OffloadKind getOffloadKind() const { return Entries[Index].first->TheOffloadKind; }
  uint32_t getVersion() const { return TheHeader->Version; }
  uint64_t getSize() const { return TheHeader->Size; }
  uint64_t getEntriesCount() const { return TheHeader->EntriesCount; }

  StringRef getTriple() const { return getString("triple"); }
  StringRef getArch() const { return getString("arch"); }
  StringRef getImage() const {
    return StringRef(&Buffer[Entries[Index].first->ImageOffset], Entries[Index].first->ImageSize);
  }

  // Iterator access to all entries in the binary
  entry_iterator_range entries() const {
    return make_range(Entries.begin(), Entries.end());
  }
  entry_iterator entries_begin() const { return Entries.begin(); }
  entry_iterator entries_end() const { return Entries.end(); }

  // Access specific entry by index.
  const std::pair<const Entry *, StringMap> &getEntry(size_t Index) const {
    assert(Index < Entries.size() && "Entry index out of bounds");
    return Entries[Index];
  }

  // Iterator over all the key and value pairs in the binary.
  string_iterator_range strings() const { return Entries[Index].second; }

  StringRef getString(StringRef Key) const { return Entries[Index].second.lookup(Key); }

  static bool classof(const Binary *V) { return V->isOffloadFile(); }

private:
  OffloadBinary(MemoryBufferRef Source, const Header *TheHeader,
                const Entry *EntriesBegin)
      : Binary(Binary::ID_Offload, Source), Buffer(Source.getBufferStart()),
        TheHeader(TheHeader) {
    for (uint64_t EI = 0, EE = TheHeader->EntriesCount; EI != EE; ++EI) {
      const Entry *TheEntry = &EntriesBegin[EI];
      const StringEntry *StringMapBegin = reinterpret_cast<const StringEntry *>(
          &Buffer[TheEntry->StringOffset]);
      StringMap Strings;
      for (uint64_t SI = 0, SE = TheEntry->NumStrings; SI != SE; ++SI) {
        StringRef Key = &Buffer[StringMapBegin[SI].KeyOffset];
        StringRef Value = StringRef(
            &Buffer[StringMapBegin[SI].ValueOffset], StringMapBegin[SI].ValueSize);
        Strings.insert({Key, Value});
      }
      Entries.push_back(std::make_pair(TheEntry, std::move(Strings)));
    }
  }

  OffloadBinary(const OffloadBinary &Other) = delete;

  /// Location of the metadata entries within the binary mapped to
  /// the key-value string data.
  SmallVector<std::pair<const Entry *, StringMap>, 1> Entries;
  /// Raw pointer to the MemoryBufferRef for convenience.
  const char *Buffer;
  /// Location of the header within the binary.
  const Header *TheHeader;
  /// Index of Entry represented by the current object.
  const uint64_t Index;
};

/// A class to contain the binary information for a single OffloadBinary that
/// owns its memory.
class OffloadFile : public OwningBinary<OffloadBinary> {
public:
  using TargetID = std::pair<StringRef, StringRef>;

  OffloadFile(std::unique_ptr<OffloadBinary> Binary,
              std::unique_ptr<MemoryBuffer> Buffer)
      : OwningBinary<OffloadBinary>(std::move(Binary), std::move(Buffer)) {}

  /// Make a deep copy of this offloading file.
  OffloadFile copy() const {
    std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBufferCopy(
        getBinary()->getMemoryBufferRef().getBuffer(),
        getBinary()->getMemoryBufferRef().getBufferIdentifier());

    // This parsing should never fail because it has already been parsed.
    auto NewBinaryOrErr = OffloadBinary::create(*Buffer);
    assert(NewBinaryOrErr && "Failed to parse a copy of the binary?");
    if (!NewBinaryOrErr)
      llvm::consumeError(NewBinaryOrErr.takeError());
    return OffloadFile(std::move(*NewBinaryOrErr), std::move(Buffer));
  }

  /// We use the Triple and Architecture pair to group linker inputs together.
  /// This conversion function lets us use these inputs in a hash-map.
  operator TargetID() const {
    return std::make_pair(getBinary()->getTriple(), getBinary()->getArch());
  }
};

/// Extracts embedded device offloading code from a memory \p Buffer to a list
/// of \p Binaries.
LLVM_ABI Error extractOffloadBinaries(MemoryBufferRef Buffer,
                                      SmallVectorImpl<OffloadFile> &Binaries);

/// Convert a string \p Name to an image kind.
LLVM_ABI ImageKind getImageKind(StringRef Name);

/// Convert an image kind to its string representation.
LLVM_ABI StringRef getImageKindName(ImageKind Name);

/// Convert a string \p Name to an offload kind.
LLVM_ABI OffloadKind getOffloadKind(StringRef Name);

/// Convert an offload kind to its string representation.
LLVM_ABI StringRef getOffloadKindName(OffloadKind Name);

/// If the target is AMD we check the target IDs for mutual compatibility. A
/// target id is a string conforming to the folowing BNF syntax:
///
///  target-id ::= '<arch> ( : <feature> ( '+' | '-' ) )*'
///
/// The features 'xnack' and 'sramecc' are currently supported. These can be in
/// the state of on, off, and any when unspecified. A target marked as any can
/// bind with either on or off. This is used to link mutually compatible
/// architectures together. Returns false in the case of an exact match.
LLVM_ABI bool areTargetsCompatible(const OffloadFile::TargetID &LHS,
                                   const OffloadFile::TargetID &RHS);

} // namespace object

} // namespace llvm
#endif
