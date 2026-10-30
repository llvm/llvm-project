//===- MSFZ.h - Compressed Multi-Stream File support ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MSFZ format specification:
// https://github.com/microsoft/pdb-rs/blob/main/docs/pdb/msfz.md

#ifndef LLVM_DEBUGINFO_MSF_MSFZ_H
#define LLVM_DEBUGINFO_MSF_MSFZ_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryStream.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstring>
#include <memory>

namespace llvm {
namespace msf {

struct MSFLayout;

inline constexpr uint8_t MSFZMagic[32] = {
    'M', 'i', 'c',  'r',  'o',    's', 'o', 'f', 't',  ' ', 'M',
    'S', 'F', 'Z',  ' ',  'C',    'o', 'n', 't', 'a',  'i', 'n',
    'e', 'r', '\r', '\n', '\x1a', 'A', 'L', 'D', '\0', '\0'};

enum class MSFZCompression : uint32_t {
  None = 0,
  Zstd = 1,
  Deflate = 2,
};

struct MSFZHeader {
  uint8_t Magic[sizeof(MSFZMagic)];
  support::ulittle64_t Version;
  support::ulittle64_t StreamDirectoryOffset;
  support::ulittle64_t ChunkTableOffset;
  support::ulittle32_t NumStreams;
  support::ulittle32_t StreamDirectoryCompression;
  support::ulittle32_t StreamDirectorySizeCompressed;
  support::ulittle32_t StreamDirectorySizeUncompressed;
  support::ulittle32_t NumChunks;
  support::ulittle32_t ChunkTableSize;
};

struct MSFZChunkEntry {
  support::ulittle64_t FileOffset;
  support::ulittle32_t Compression;
  support::ulittle32_t CompressedSize;
  support::ulittle32_t UncompressedSize;
};

static_assert(sizeof(MSFZHeader) == 80);
static_assert(sizeof(MSFZChunkEntry) == 20);

inline bool isMSFZMagic(ArrayRef<uint8_t> Data) {
  return Data.size() >= sizeof(MSFZMagic) &&
         std::memcmp(Data.data(), MSFZMagic, sizeof(MSFZMagic)) == 0;
}

#if LLVM_ENABLE_ZSTD

/// A read-only MSFZ container. The container owns stream metadata but borrows
/// the underlying file stream and allocator, both of which must outlive it.
class LLVM_ABI MSFZFile {
public:
  /// Parse an MSFZ container from \p File.
  ///
  /// The returned object borrows \p File and \p Allocator; both must outlive
  /// the MSFZFile and every stream created from it. Malformed or unsupported
  /// containers are reported as errors.
  static Expected<std::unique_ptr<MSFZFile>>
  create(BinaryStream &File, BumpPtrAllocator &Allocator);
  ~MSFZFile();

  MSFZFile(const MSFZFile &) = delete;
  MSFZFile &operator=(const MSFZFile &) = delete;

  /// Return the number of entries in the logical stream directory.
  uint32_t getNumStreams() const;

  /// Return the byte size of stream \p StreamIndex.
  ///
  /// \p StreamIndex must be less than getNumStreams(). Nil streams have size
  /// zero; use isNilStream() to distinguish them from empty non-nil streams.
  uint64_t getStreamByteSize(uint32_t StreamIndex) const;

  /// Return whether \p StreamIndex denotes a nil stream.
  ///
  /// \p StreamIndex must be less than getNumStreams().
  bool isNilStream(uint32_t StreamIndex) const;

  /// Create a read-only view of logical stream \p StreamIndex.
  ///
  /// Invalid indices and nil streams are reported as errors. The returned
  /// stream borrows storage owned by this file and its allocator and must not
  /// outlive them.
  Expected<std::unique_ptr<BinaryStream>>
  createStream(uint32_t StreamIndex) const;

private:
  class Impl;
  explicit MSFZFile(std::unique_ptr<Impl> Impl);

  std::unique_ptr<Impl> PImpl;
};

/// A streaming writer that maps a classic MSF layout to an MSFZ container.
/// Although we represent those offsets as if they were in an MSF image, none
/// is ever written to disk. The writer maps addressed MSF blocks to logical
/// streams and requires contiguous, forward writes within each stream.
class LLVM_ABI MSFZWriter : public WritableBinaryStream {
public:
  /// Create a writer using zstd's fastest compression level.
  ///
  /// \p Layout defines the logical streams and their source MSF block
  /// locations, using classic MSF's 32-bit stream sizes and nil markers.
  /// Output is staged in a temporary file; creation errors include invalid
  /// layouts and failures opening that file.
  static Expected<std::unique_ptr<MSFZWriter>> create(StringRef Path,
                                                      const MSFLayout &Layout);

  /// Create a writer using \p CompressionLevel.
  ///
  /// The level must be within the range accepted by LLVM's zstd support.
  /// \p ChunkSize is the maximum uncompressed size of an MSFZ compression
  /// chunk, between 4096 and 1073741824 bytes. Compressed stream data shares
  /// a chunk buffer across MSF block and logical stream boundaries; neither
  /// boundary flushes it. A chunk may contain fragments from multiple streams,
  /// and a stream may span chunks. Full chunks are emitted as they fill;
  /// finalize() emits any remaining partial chunk.
  /// Invalid levels, invalid layouts, and output-file failures are returned as
  /// errors.
  static Expected<std::unique_ptr<MSFZWriter>>
  create(StringRef Path, const MSFLayout &Layout, int CompressionLevel,
         uint32_t ChunkSize = 1024 * 1024);
  ~MSFZWriter() override;

  MSFZWriter(const MSFZWriter &) = delete;
  MSFZWriter &operator=(const MSFZWriter &) = delete;

  llvm::endianness getEndian() const override;

  /// Reading is unsupported; returns an error without interpreting \p Offset.
  Error readBytes(uint64_t Offset, uint64_t Size,
                  ArrayRef<uint8_t> &Buffer) override;

  /// Reading is unsupported; returns an error without interpreting \p Offset.
  Error readLongestContiguousChunk(uint64_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override;

  /// getLength() returns the virtual MSF image's size, not the MSFZ output size.
  uint64_t getLength() override;

  /// Offsets passed to writeBytes() are byte offsets in the (virtual) MSF image,
  /// not offsets in the MSFZ output file or within a logical PDB stream.
  Error writeBytes(uint64_t MSFLayoutOffset, ArrayRef<uint8_t> Data) override;

  /// Satisfy WritableBinaryStream's commit interface without publishing output.
  ///
  /// MSFZ output requires explicit finalization after any PDB Info stream build
  /// ID patch, so callers must invoke finalize() to produce \p Path.
  Error commit() override;

  /// Reserve one stream's entire contents uncompressed immediately after the
  /// MSFZ header, rather than packing it into compression chunks.
  ///
  /// This must be called before writing any stream, at most once per writer.
  /// Only the first \p PatchablePrefixSize bytes can subsequently be changed
  /// with patchStream(). The stream must be non-nil and the prefix must fit
  /// within it.
  Error setInitialUncompressedStream(uint32_t StreamIndex,
                                     uint32_t PatchablePrefixSize);

  /// Replace bytes in the prefix registered by setInitialUncompressedStream().
  ///
  /// \p Offset is a byte offset within the logical stream identified by
  /// \p StreamIndex, not within the virtual MSF image or the MSFZ output file.
  /// The range must be inside both the retained prefix and data already
  /// written. Patching does not change the canonical digest.
  Error patchStream(uint32_t StreamIndex, uint32_t Offset,
                    ArrayRef<uint8_t> Data);

  /// Return a canonical digest of the completed, unpatched logical streams.
  ///
  /// The digest includes stream boundaries, sizes, nil-stream identity, and
  /// contents, but is independent of physical blocks, chunks, compression, and
  /// write scheduling. It is not updated by patchStream(), allowing the digest
  /// to be embedded in the retained PDB Info stream prefix. Incomplete streams
  /// are reported as errors.
  Expected<uint64_t> getCanonicalDigest();

  /// Finish compression and atomically publish the container at its final path.
  ///
  /// All non-nil streams must have been written completely. This operation is
  /// idempotent; compression, I/O, and rename failures are returned as errors.
  /// Destroying a writer before successful finalization discards its temporary
  /// output.
  Error finalize();

private:
  class Impl;
  explicit MSFZWriter(std::unique_ptr<Impl> Impl);

  std::unique_ptr<Impl> PImpl;
};

/// Write the streams described by \p Layout from \p MsfData to an MSFZ file.
///
/// \p MsfData is borrowed for the duration of the call and must contain every
/// block referenced by the layout. The function finalizes \p Path before
/// returning and reports invalid layouts, missing blocks, compression errors,
/// and output failures.
LLVM_ABI Error writeMSFZ(StringRef Path, const MSFLayout &Layout,
                         ArrayRef<uint8_t> MsfData);

#endif

} // namespace msf
} // namespace llvm

#endif
