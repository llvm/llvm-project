//===- MSFZ.cpp - Compressed Multi-Stream File support --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MSFZ.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::msf;

namespace {

constexpr uint32_t ChunkSize = 1024 * 1024;
constexpr uint32_t NilStreamSize = std::numeric_limits<uint32_t>::max();
constexpr uint32_t InvalidStreamIndex = std::numeric_limits<uint32_t>::max();
constexpr uint64_t MaxChunks = uint64_t{1} << 31;
constexpr uint64_t CompressedFragmentBit = uint64_t{1} << 63;
constexpr uint64_t ChunkIndexMask = MaxChunks - 1;
constexpr uint64_t UncompressedFragmentOffsetMask = (uint64_t{1} << 48) - 1;
constexpr uint64_t MinimumFileSize = 0x4000;
constexpr uint32_t MaxDirectorySize = 64 * 1024 * 1024;
constexpr uint32_t MaxCompressedChunkSize = 1U << 30;
constexpr size_t MaxQueuedCompressionBytes = 64 * 1024 * 1024;

void writeU32(raw_ostream &OS, uint32_t Value) {
  support::ulittle32_t LE;
  LE = Value;
  OS.write(reinterpret_cast<const char *>(&LE), sizeof(LE));
}

void writeU64(raw_ostream &OS, uint64_t Value) {
  support::ulittle64_t LE;
  LE = Value;
  OS.write(reinterpret_cast<const char *>(&LE), sizeof(LE));
}

Error streamError(const Twine &Message) {
  return createStringError(inconvertibleErrorCode(), Message);
}

ArrayRef<uint8_t> bytesOf(const void *Data, size_t Size) {
  return {reinterpret_cast<const uint8_t *>(Data), Size};
}

uint64_t encodeCompressedFragmentLocation(uint32_t FirstChunk,
                                          uint32_t ChunkOffset) {
  // Bit 63 marks compression, bits 62..32 hold the first chunk index, and
  // bits 31..0 hold the offset within that chunk.
  return CompressedFragmentBit | (uint64_t(FirstChunk) << 32) | ChunkOffset;
}

} // namespace

class MSFZFile::Impl {
  struct FileRange {
    uint64_t Offset;
    uint64_t Size;
  };

  struct Fragment {
    uint64_t StreamOffset;
    uint32_t Size;
    uint64_t FileOffset;
    uint32_t FirstChunk;
    uint32_t ChunkOffset;
    bool IsCompressed;
  };

  struct Stream {
    uint32_t Size;
    SmallVector<Fragment, 1> Fragments;
  };

  class StreamReader;

public:
  Impl(BinaryStream &File, BumpPtrAllocator &Allocator)
      : File(File), Allocator(Allocator) {}

  Error parse() {
    if (File.getLength() < sizeof(MSFZHeader))
      return streamError("MSFZ file is too small for its header");

    ArrayRef<uint8_t> HeaderBytes;
    if (Error E = File.readBytes(0, sizeof(MSFZHeader), HeaderBytes))
      return E;

    MSFZHeader Header;
    std::memcpy(&Header, HeaderBytes.data(), sizeof(Header));
    if (!isMSFZMagic(Header.Magic))
      return streamError("MSFZ file has an invalid magic");
    if (Header.Version != 0)
      return streamError("MSFZ file has an unsupported version");
    if (Header.NumStreams == 0)
      return streamError("MSFZ file does not contain a stream");
    if (Header.StreamDirectorySizeUncompressed > MaxDirectorySize)
      return streamError("MSFZ stream directory exceeds the supported size");
    if (uint64_t(Header.NumStreams) * sizeof(support::ulittle32_t) >
        Header.StreamDirectorySizeUncompressed)
      return streamError(
          "MSFZ stream count does not fit in the stream directory");

    if (Header.NumChunks > MaxChunks)
      return streamError("MSFZ file has too many chunks");
    uint64_t ExpectedChunkTableSize =
        uint64_t(Header.NumChunks) * sizeof(MSFZChunkEntry);
    if (Header.ChunkTableSize != ExpectedChunkTableSize)
      return streamError("MSFZ file has an invalid chunk table size");

    if (Error E = checkFileRange(Header.StreamDirectoryOffset,
                                 Header.StreamDirectorySizeCompressed,
                                 "stream directory"))
      return E;
    if (Error E = checkFileRange(Header.ChunkTableOffset, Header.ChunkTableSize,
                                 "chunk table"))
      return E;

    OccupiedRanges.push_back({0, sizeof(MSFZHeader)});
    OccupiedRanges.push_back(
        {Header.StreamDirectoryOffset, Header.StreamDirectorySizeCompressed});
    if (Header.ChunkTableSize != 0)
      OccupiedRanges.push_back(
          {Header.ChunkTableOffset, Header.ChunkTableSize});

    if (Error E = parseChunkTable(Header))
      return E;
    if (Error E = parseDirectory(Header))
      return E;
    return validateFileRanges();
  }

  uint32_t getNumStreams() const { return Streams.size(); }

  uint32_t getStreamByteSize(uint32_t StreamIndex) const {
    assert(StreamIndex < Streams.size() && "MSFZ stream index out of range");
    return Streams[StreamIndex].Size;
  }

  bool isNilStream(uint32_t StreamIndex) const {
    assert(StreamIndex < Streams.size() && "MSFZ stream index out of range");
    return Streams[StreamIndex].Size == NilStreamSize;
  }

  Expected<std::unique_ptr<BinaryStream>>
  createStream(uint32_t StreamIndex) const;

private:
  Error checkFileRange(uint64_t Offset, uint64_t Size,
                       StringRef Description) const {
    uint64_t FileSize = File.getLength();
    if (Offset > FileSize || Size > FileSize - Offset)
      return streamError(Twine("MSFZ file has an out-of-range ") + Description);
    return Error::success();
  }

  Error parseChunkTable(const MSFZHeader &Header) {
    if (Header.NumChunks == 0)
      return Error::success();

    ArrayRef<uint8_t> TableBytes;
    if (Error E = File.readBytes(Header.ChunkTableOffset, Header.ChunkTableSize,
                                 TableBytes))
      return E;
    Chunks.reserve(Header.NumChunks);
    for (uint32_t I = 0; I != Header.NumChunks; ++I) {
      MSFZChunkEntry Entry;
      std::memcpy(&Entry, TableBytes.data() + I * sizeof(Entry), sizeof(Entry));

      MSFZCompression Compression =
          static_cast<MSFZCompression>(uint32_t(Entry.Compression));
      if (Entry.CompressedSize == 0 || Entry.UncompressedSize == 0)
        return streamError("MSFZ file has a zero-sized chunk");
      switch (Compression) {
      case MSFZCompression::None:
        if (Entry.CompressedSize != Entry.UncompressedSize)
          return streamError(
              "MSFZ file has an uncompressed chunk with mismatched sizes");
        break;
      case MSFZCompression::Zstd:
        if (Entry.UncompressedSize > MaxCompressedChunkSize)
          return streamError(
              "MSFZ compressed chunk exceeds the supported size");
        break;
      case MSFZCompression::Deflate:
        if (!compression::raw_deflate::isAvailable())
          return streamError(
              "MSFZ raw DEFLATE compression requires LLVM_ENABLE_ZLIB");
        if (Entry.UncompressedSize > MaxCompressedChunkSize)
          return streamError(
              "MSFZ compressed chunk exceeds the supported size");
        break;
      default:
        return streamError("MSFZ file uses an unknown chunk compression");
      }
      if (Error E = checkFileRange(Entry.FileOffset, Entry.CompressedSize,
                                   "chunk payload"))
        return E;
      OccupiedRanges.push_back({Entry.FileOffset, Entry.CompressedSize});
      Chunks.push_back(Entry);
    }
    CachedChunks.resize(Chunks.size());
    return Error::success();
  }

  Error parseDirectory(const MSFZHeader &Header) {
    MSFZCompression Compression = static_cast<MSFZCompression>(
        uint32_t(Header.StreamDirectoryCompression));
    ArrayRef<uint8_t> CompressedDirectory;
    if (Error E = File.readBytes(Header.StreamDirectoryOffset,
                                 Header.StreamDirectorySizeCompressed,
                                 CompressedDirectory))
      return E;

    SmallVector<uint8_t, 0> DirectoryStorage;
    ArrayRef<uint8_t> Directory;
    switch (Compression) {
    case MSFZCompression::None:
      if (Header.StreamDirectorySizeCompressed !=
          Header.StreamDirectorySizeUncompressed)
        return streamError(
            "MSFZ file has an uncompressed directory with mismatched sizes");
      Directory = CompressedDirectory;
      break;
    case MSFZCompression::Zstd:
      if (Error E = compression::zstd::decompress(
              CompressedDirectory, DirectoryStorage,
              Header.StreamDirectorySizeUncompressed))
        return E;
      if (DirectoryStorage.size() != Header.StreamDirectorySizeUncompressed)
        return streamError("MSFZ file has an invalid decompressed directory");
      Directory = DirectoryStorage;
      break;
    case MSFZCompression::Deflate:
      if (!compression::raw_deflate::isAvailable())
        return streamError("MSFZ raw DEFLATE directory compression requires "
                           "LLVM_ENABLE_ZLIB");
      if (Error E = compression::raw_deflate::decompress(
              CompressedDirectory, DirectoryStorage,
              Header.StreamDirectorySizeUncompressed))
        return E;
      Directory = DirectoryStorage;
      break;
    default:
      return streamError("MSFZ file uses an unknown directory compression");
    }

    uint64_t Offset = 0;
    Streams.reserve(Header.NumStreams);
    for (uint32_t StreamIndex = 0; StreamIndex != Header.NumStreams;
         ++StreamIndex) {
      Stream S = {0, {}};
      uint64_t StreamSize = 0;
      while (true) {
        uint32_t FragmentSize;
        if (!readU32(Directory, Offset, FragmentSize))
          return streamError("MSFZ stream directory is truncated");

        if (FragmentSize == NilStreamSize) {
          if (StreamSize != 0 || !S.Fragments.empty())
            return streamError(
                "MSFZ stream directory has an invalid nil stream");
          S.Size = NilStreamSize;
          break;
        }
        if (FragmentSize == 0) {
          if (StreamSize > std::numeric_limits<uint32_t>::max())
            return streamError(
                "MSFZ logical stream exceeds the PDB size limit");
          S.Size = static_cast<uint32_t>(StreamSize);
          break;
        }

        uint64_t FragmentLocation;
        if (!readU64(Directory, Offset, FragmentLocation))
          return streamError("MSFZ stream directory is truncated");
        if (FragmentSize > std::numeric_limits<uint32_t>::max() - StreamSize)
          return streamError("MSFZ logical stream exceeds the PDB size limit");

        Fragment F = {};
        F.StreamOffset = StreamSize;
        F.Size = FragmentSize;
        F.IsCompressed = (FragmentLocation & CompressedFragmentBit) != 0;
        if (F.IsCompressed) {
          F.FirstChunk =
              static_cast<uint32_t>((FragmentLocation >> 32) & ChunkIndexMask);
          F.ChunkOffset = static_cast<uint32_t>(FragmentLocation);
          if (Error E = validateCompressedFragment(F))
            return E;
        } else {
          if (FragmentLocation & ~UncompressedFragmentOffsetMask)
            return streamError(
                "MSFZ uncompressed fragment has non-zero reserved bits");
          F.FileOffset = FragmentLocation & UncompressedFragmentOffsetMask;
          if (Error E = checkFileRange(F.FileOffset, F.Size, "stream fragment"))
            return E;
          OccupiedRanges.push_back({F.FileOffset, F.Size});
        }
        StreamSize += FragmentSize;
        S.Fragments.push_back(F);
      }
      Streams.push_back(std::move(S));
    }

    if (Offset != Directory.size())
      return streamError("MSFZ stream directory has trailing data");
    return Error::success();
  }

  Error validateFileRanges() {
    llvm::sort(OccupiedRanges, [](const FileRange &L, const FileRange &R) {
      return L.Offset < R.Offset;
    });
    for (size_t I = 1; I != OccupiedRanges.size(); ++I) {
      const FileRange &Previous = OccupiedRanges[I - 1];
      const FileRange &Current = OccupiedRanges[I];
      if (Current.Offset < Previous.Offset + Previous.Size)
        return streamError("MSFZ file has overlapping physical regions");
    }
    return Error::success();
  }

  Error validateCompressedFragment(const Fragment &F) const {
    if (F.FirstChunk >= Chunks.size())
      return streamError("MSFZ compressed fragment has an invalid chunk index");

    uint64_t Remaining = F.Size;
    uint32_t ChunkIndex = F.FirstChunk;
    uint32_t ChunkOffset = F.ChunkOffset;
    while (Remaining != 0) {
      if (ChunkIndex >= Chunks.size())
        return streamError(
            "MSFZ compressed fragment extends past the chunk table");
      uint32_t ChunkSize = Chunks[ChunkIndex].UncompressedSize;
      if (ChunkOffset > ChunkSize)
        return streamError("MSFZ compressed fragment has an invalid offset");
      uint64_t Available = ChunkSize - ChunkOffset;
      if (Available == 0)
        return streamError(
            "MSFZ compressed fragment references an empty chunk");
      uint64_t Bytes = std::min(Remaining, Available);
      Remaining -= Bytes;
      ++ChunkIndex;
      ChunkOffset = 0;
    }
    return Error::success();
  }

  Expected<ArrayRef<uint8_t>> getChunk(uint32_t ChunkIndex) const {
    assert(ChunkIndex < Chunks.size() && "MSFZ chunk index out of range");
    if (CachedChunks[ChunkIndex])
      return *CachedChunks[ChunkIndex];

    const MSFZChunkEntry &Entry = Chunks[ChunkIndex];
    ArrayRef<uint8_t> Compressed;
    if (Error E =
            File.readBytes(Entry.FileOffset, Entry.CompressedSize, Compressed))
      return std::move(E);

    MSFZCompression Compression =
        static_cast<MSFZCompression>(uint32_t(Entry.Compression));
    if (Compression == MSFZCompression::None) {
      CachedChunks[ChunkIndex] = Compressed;
      return Compressed;
    }

    SmallVector<uint8_t, 0> Decompressed;
    Error E = Compression == MSFZCompression::Deflate
                  ? compression::raw_deflate::decompress(
                        Compressed, Decompressed, Entry.UncompressedSize)
                  : compression::zstd::decompress(Compressed, Decompressed,
                                                  Entry.UncompressedSize);
    if (E)
      return std::move(E);
    if (Decompressed.size() != Entry.UncompressedSize)
      return streamError("MSFZ file has an invalid decompressed chunk");

    ArrayRef<uint8_t> Stable;
    if (!Decompressed.empty()) {
      uint8_t *Data = Allocator.Allocate<uint8_t>(Decompressed.size());
      std::memcpy(Data, Decompressed.data(), Decompressed.size());
      Stable = ArrayRef<uint8_t>(Data, Decompressed.size());
    }
    CachedChunks[ChunkIndex] = Stable;
    return Stable;
  }

  static bool readU32(ArrayRef<uint8_t> Data, uint64_t &Offset,
                      uint32_t &Value) {
    if (Offset > Data.size() ||
        sizeof(support::ulittle32_t) > Data.size() - Offset)
      return false;
    support::ulittle32_t LE;
    std::memcpy(&LE, Data.data() + Offset, sizeof(LE));
    Value = LE;
    Offset += sizeof(LE);
    return true;
  }

  static bool readU64(ArrayRef<uint8_t> Data, uint64_t &Offset,
                      uint64_t &Value) {
    if (Offset > Data.size() ||
        sizeof(support::ulittle64_t) > Data.size() - Offset)
      return false;
    support::ulittle64_t LE;
    std::memcpy(&LE, Data.data() + Offset, sizeof(LE));
    Value = LE;
    Offset += sizeof(LE);
    return true;
  }

  class StreamReader : public BinaryStream {
  public:
    StreamReader(const Impl &File, const Stream &Stream)
        : File(File), S(Stream) {}

    llvm::endianness getEndian() const override {
      return llvm::endianness::little;
    }

    Error readBytes(uint64_t Offset, uint64_t Size,
                    ArrayRef<uint8_t> &Buffer) override {
      if (Offset > S.Size || Size > uint64_t(S.Size) - Offset)
        return streamError("MSFZ read is outside a logical stream");
      if (Size == 0) {
        Buffer = {};
        return Error::success();
      }

      const Fragment *F = findFragment(Offset);
      assert(F && "validated MSFZ stream has a fragment for every byte");
      uint64_t FragmentOffset = Offset - F->StreamOffset;
      if (Size <= uint64_t(F->Size) - FragmentOffset)
        return readFragment(*F, FragmentOffset, Size, Buffer);

      uint8_t *Data = File.Allocator.Allocate<uint8_t>(Size);
      uint64_t Remaining = Size;
      uint64_t OutputOffset = 0;
      while (Remaining != 0) {
        F = findFragment(Offset);
        assert(F && "validated MSFZ stream has a fragment for every byte");
        FragmentOffset = Offset - F->StreamOffset;
        uint64_t Bytes =
            std::min<uint64_t>(Remaining, F->Size - FragmentOffset);
        ArrayRef<uint8_t> FragmentData;
        if (Error E = readFragment(*F, FragmentOffset, Bytes, FragmentData))
          return E;
        std::memcpy(Data + OutputOffset, FragmentData.data(), Bytes);
        Offset += Bytes;
        OutputOffset += Bytes;
        Remaining -= Bytes;
      }
      Buffer = ArrayRef<uint8_t>(Data, Size);
      return Error::success();
    }

    Error readLongestContiguousChunk(uint64_t Offset,
                                     ArrayRef<uint8_t> &Buffer) override {
      if (Offset >= S.Size)
        return streamError("MSFZ read is outside a logical stream");
      const Fragment *F = findFragment(Offset);
      assert(F && "validated MSFZ stream has a fragment for every byte");
      uint64_t FragmentOffset = Offset - F->StreamOffset;
      if (!F->IsCompressed)
        return File.File.readBytes(F->FileOffset + FragmentOffset,
                                   F->Size - FragmentOffset, Buffer);

      uint32_t ChunkIndex;
      uint32_t ChunkOffset;
      locateChunk(*F, FragmentOffset, ChunkIndex, ChunkOffset);
      Expected<ArrayRef<uint8_t>> Chunk = File.getChunk(ChunkIndex);
      if (!Chunk)
        return Chunk.takeError();
      Buffer = Chunk->drop_front(ChunkOffset)
                   .take_front(std::min<uint64_t>(F->Size - FragmentOffset,
                                                  Chunk->size() - ChunkOffset));
      return Error::success();
    }

    uint64_t getLength() override { return S.Size; }

  private:
    const Fragment *findFragment(uint64_t Offset) const {
      auto It = llvm::partition_point(S.Fragments, [Offset](const Fragment &F) {
        return F.StreamOffset <= Offset;
      });
      if (It == S.Fragments.begin())
        return nullptr;
      --It;
      return Offset - It->StreamOffset < It->Size ? &*It : nullptr;
    }

    void locateChunk(const Fragment &F, uint64_t FragmentOffset,
                     uint32_t &ChunkIndex, uint32_t &ChunkOffset) const {
      ChunkIndex = F.FirstChunk;
      ChunkOffset = F.ChunkOffset;
      while (FragmentOffset >=
             uint64_t(File.Chunks[ChunkIndex].UncompressedSize) - ChunkOffset) {
        FragmentOffset -=
            File.Chunks[ChunkIndex].UncompressedSize - ChunkOffset;
        ++ChunkIndex;
        ChunkOffset = 0;
      }
      ChunkOffset += FragmentOffset;
    }

    Error readFragment(const Fragment &F, uint64_t FragmentOffset,
                       uint64_t Size, ArrayRef<uint8_t> &Buffer) {
      if (!F.IsCompressed)
        return File.File.readBytes(F.FileOffset + FragmentOffset, Size, Buffer);

      uint32_t ChunkIndex;
      uint32_t ChunkOffset;
      locateChunk(F, FragmentOffset, ChunkIndex, ChunkOffset);
      Expected<ArrayRef<uint8_t>> Chunk = File.getChunk(ChunkIndex);
      if (!Chunk)
        return Chunk.takeError();
      if (Size <= Chunk->size() - ChunkOffset) {
        Buffer = Chunk->slice(ChunkOffset, Size);
        return Error::success();
      }

      uint8_t *Data = File.Allocator.Allocate<uint8_t>(Size);
      uint64_t Remaining = Size;
      uint64_t OutputOffset = 0;
      while (Remaining != 0) {
        Chunk = File.getChunk(ChunkIndex);
        if (!Chunk)
          return Chunk.takeError();
        uint64_t Bytes =
            std::min<uint64_t>(Remaining, Chunk->size() - ChunkOffset);
        std::memcpy(Data + OutputOffset, Chunk->data() + ChunkOffset, Bytes);
        Remaining -= Bytes;
        OutputOffset += Bytes;
        ++ChunkIndex;
        ChunkOffset = 0;
      }
      Buffer = ArrayRef<uint8_t>(Data, Size);
      return Error::success();
    }

    const Impl &File;
    const Stream &S;
  };

  BinaryStream &File;
  BumpPtrAllocator &Allocator;
  SmallVector<MSFZChunkEntry, 16> Chunks;
  mutable SmallVector<std::optional<ArrayRef<uint8_t>>, 16> CachedChunks;
  SmallVector<Stream, 32> Streams;
  SmallVector<FileRange, 32> OccupiedRanges;
};

Expected<std::unique_ptr<BinaryStream>>
MSFZFile::Impl::createStream(uint32_t StreamIndex) const {
  if (StreamIndex >= Streams.size() || isNilStream(StreamIndex))
    return streamError("MSFZ stream does not exist");
  return std::unique_ptr<BinaryStream>(
      std::make_unique<StreamReader>(*this, Streams[StreamIndex]));
}

MSFZFile::MSFZFile(std::unique_ptr<Impl> Impl) : PImpl(std::move(Impl)) {}

MSFZFile::~MSFZFile() = default;

Expected<std::unique_ptr<MSFZFile>>
MSFZFile::create(BinaryStream &File, BumpPtrAllocator &Allocator) {
  std::unique_ptr<MSFZFile> Result(
      new MSFZFile(std::make_unique<Impl>(File, Allocator)));
  if (Error E = Result->PImpl->parse())
    return std::move(E);
  return Result;
}

uint32_t MSFZFile::getNumStreams() const { return PImpl->getNumStreams(); }

uint32_t MSFZFile::getStreamByteSize(uint32_t StreamIndex) const {
  return PImpl->getStreamByteSize(StreamIndex);
}

bool MSFZFile::isNilStream(uint32_t StreamIndex) const {
  return PImpl->isNilStream(StreamIndex);
}

Expected<std::unique_ptr<BinaryStream>>
MSFZFile::createStream(uint32_t StreamIndex) const {
  return PImpl->createStream(StreamIndex);
}

class MSFZWriter::Impl {
  struct OutputFragment {
    uint32_t Size;
    uint32_t FirstChunk = 0;
    uint32_t Offset = 0;
    uint64_t FileOffset = 0;
    bool IsCompressed = true;
  };

  struct OutputStream {
    uint32_t Size;
    uint32_t NextOffset = 0;
    MD5 Hash;
    SmallVector<OutputFragment, 1> Fragments;
  };

  struct BlockLocation {
    uint32_t StreamIndex = InvalidStreamIndex;
    uint32_t StreamOffset = 0;
  };

  struct CompressedChunk {
    uint32_t Index;
    SmallVector<uint8_t, 0> Data;
  };

  struct PendingChunk {
    size_t ReservedBytes;
    std::shared_ptr<CompressedChunk> Result;
    std::shared_future<void> Future;
  };

public:
  Impl(sys::fs::TempFile Temp, StringRef Path, int CompressionLevel)
      : Temp(std::move(Temp)), Path(Path), OS(this->Temp.FD,
                                              /*shouldClose=*/false,
                                              /*unbuffered=*/true),
        CompressionLevel(CompressionLevel) {
    unsigned ThreadCount = parallel::strategy.compute_thread_count();
    if (ThreadCount > 1) {
      CompressionPool = std::make_unique<DefaultThreadPool>(parallel::strategy);
      MaxPendingChunks = std::max(
          1U,
          std::min(CompressionPool->getMaxConcurrency(),
                   static_cast<unsigned>(MaxQueuedCompressionBytes /
                                         compressionReservation(ChunkSize))));
    }
  }

  ~Impl() {
    if (CompressionPool)
      CompressionPool->wait();
    OS.flush();
    if (OS.has_error())
      OS.clear_error();
    if (!Finalized)
      consumeError(Temp.discard());
  }

  Error initialize(const MSFLayout &Layout) {
    if (Error E = initializeLayout(Layout))
      return E;

    MSFZHeader Header = {};
    OS.write(reinterpret_cast<const char *>(&Header), sizeof(Header));
    return takeOutputError();
  }

  uint64_t getLength() const { return FileSize; }

  Error writeBytes(uint64_t Offset, ArrayRef<uint8_t> Data) {
    if (Finalized)
      return streamError("MSFZ output: write after finalization");
    if (Data.empty())
      return Error::success();
    if (Offset > FileSize || Data.size() > FileSize - Offset)
      return streamError("MSFZ output: write is outside the MSF layout");

    while (!Data.empty()) {
      uint64_t Block = Offset / BlockSize;
      uint32_t OffsetInBlock = Offset % BlockSize;
      if (Block >= BlockLocations.size())
        return streamError("MSFZ output: write references an invalid block");
      const BlockLocation &Location = BlockLocations[Block];
      if (Location.StreamIndex == InvalidStreamIndex)
        return streamError("MSFZ output: write references a non-stream block");

      OutputStream &Stream = Streams[Location.StreamIndex];
      uint64_t StreamOffset = uint64_t(Location.StreamOffset) + OffsetInBlock;
      if (StreamOffset != Stream.NextOffset)
        return streamError("MSFZ output: stream writes must be forward");

      uint32_t Bytes = static_cast<uint32_t>(
          std::min<uint64_t>(Data.size(), BlockSize - OffsetInBlock));
      if (Bytes > Stream.Size - Stream.NextOffset)
        return streamError("MSFZ output: write exceeds the logical stream");

      if (Error E = append(Location.StreamIndex, Stream.NextOffset,
                           Data.take_front(Bytes)))
        return E;
      Stream.Hash.update(Data.take_front(Bytes));
      Stream.NextOffset += Bytes;
      Offset += Bytes;
      Data = Data.drop_front(Bytes);
    }
    return Error::success();
  }

  Error setInitialUncompressedStream(uint32_t StreamIndex,
                                     uint32_t PatchablePrefixSize) {
    if (Finalized)
      return streamError(
          "MSFZ output: initial uncompressed stream after finalization");
    if (StreamIndex >= Streams.size())
      return streamError("MSFZ output: invalid patchable stream");
    if (PatchablePrefixSize > Streams[StreamIndex].Size ||
        Streams[StreamIndex].Size == NilStreamSize)
      return streamError("MSFZ output: invalid patchable stream prefix");
    if (PatchableStream != InvalidStreamIndex)
      return streamError("MSFZ output: patchable prefix already set");
    for (const OutputStream &Stream : Streams)
      if (Stream.NextOffset != 0)
        return streamError(
            "MSFZ output: initial uncompressed stream set after writing");
    if (!ChunkData.empty() || !Chunks.empty() || !PendingChunks.empty())
      return streamError(
          "MSFZ output: initial uncompressed stream set after writing");

    PatchableStream = StreamIndex;
    PatchablePrefix = PatchablePrefixSize;
    PatchableStreamFileOffset = OS.tell();
    OutputStream &Stream = Streams[StreamIndex];
    if (Stream.Size != 0) {
      Stream.Fragments.push_back(
          {Stream.Size, 0, 0, PatchableStreamFileOffset, false});
      OS.write_zeros(Stream.Size);
    }
    if (Error E = align())
      return E;
    return takeOutputError();
  }

  Error patchStream(uint32_t StreamIndex, uint32_t Offset,
                    ArrayRef<uint8_t> Data) {
    if (Finalized)
      return streamError("MSFZ output: patch after finalization");
    if (StreamIndex != PatchableStream ||
        uint64_t(Offset) + Data.size() > PatchablePrefix)
      return streamError("MSFZ output: patch is outside the retained prefix");
    OutputStream &Stream = Streams[StreamIndex];
    if (uint64_t(Offset) + Data.size() > Stream.NextOffset)
      return streamError("MSFZ output: patch references unwritten data");

    OS.pwrite(reinterpret_cast<const char *>(Data.data()), Data.size(),
              PatchableStreamFileOffset + Offset);
    return takeOutputError();
  }

  Expected<uint64_t> getCanonicalDigest() {
    if (Error E = checkStreamsComplete())
      return std::move(E);

    MD5 Digest;
    static constexpr char Prefix[] = "LLVM MSFZ logical streams v1";
    Digest.update(StringRef(Prefix, sizeof(Prefix) - 1));
    updateDigestU32(Digest, static_cast<uint32_t>(Streams.size()));
    for (const OutputStream &Stream : Streams) {
      updateDigestU32(Digest, Stream.Size);
      MD5 StreamHash = Stream.Hash;
      MD5::MD5Result StreamDigest = StreamHash.result();
      Digest.update({StreamDigest.data(), StreamDigest.size()});
    }
    return Digest.final().low();
  }

  Error finalize() {
    if (Finalized)
      return Error::success();
    if (Error E = checkStreamsComplete())
      return E;
    if (Error E = flushChunk())
      return E;
    if (Error E = drainPendingChunks())
      return E;

    Expected<uint32_t> DirectorySize = getDirectorySize();
    if (!DirectorySize)
      return DirectorySize.takeError();
    if (Chunks.size() > MaxChunks)
      return streamError("MSFZ output: too many chunks");
    uint64_t ChunkTableSize = Chunks.size() * uint64_t(sizeof(MSFZChunkEntry));
    if (ChunkTableSize > std::numeric_limits<uint32_t>::max())
      return streamError("MSFZ output: chunk table is too large");

    MSFZHeader Header = {};
    if (Error E = align())
      return E;
    Header.StreamDirectoryOffset = OS.tell();
    Header.StreamDirectorySizeCompressed = *DirectorySize;
    Header.StreamDirectorySizeUncompressed = *DirectorySize;
    Header.StreamDirectoryCompression =
        static_cast<uint32_t>(MSFZCompression::None);
    if (Error E = writeDirectory())
      return E;

    if (Error E = align())
      return E;
    Header.ChunkTableOffset = OS.tell();
    Header.NumChunks = static_cast<uint32_t>(Chunks.size());
    Header.ChunkTableSize = static_cast<uint32_t>(ChunkTableSize);
    OS.write(reinterpret_cast<const char *>(Chunks.data()), ChunkTableSize);
    if (Error E = takeOutputError())
      return E;

    if (OS.tell() < MinimumFileSize)
      OS.write_zeros(MinimumFileSize - OS.tell());
    if (Error E = takeOutputError())
      return E;

    std::memcpy(Header.Magic, MSFZMagic, sizeof(MSFZMagic));
    Header.Version = 0;
    Header.NumStreams = static_cast<uint32_t>(Streams.size());
    OS.pwrite(reinterpret_cast<const char *>(&Header), sizeof(Header), 0);
    if (Error E = takeOutputError())
      return E;

    if (Error E = Temp.keep(Path))
      return E;
    Finalized = true;
    return Error::success();
  }

private:
  Error initializeLayout(const MSFLayout &Layout) {
    if (!Layout.SB)
      return streamError("MSFZ output: missing MSF super block");
    if (Layout.StreamSizes.size() != Layout.StreamMap.size())
      return streamError("MSFZ output: inconsistent MSF stream directory");
    if (Layout.StreamSizes.empty())
      return streamError("MSFZ output: an MSFZ file must contain a stream");
    if (Layout.StreamSizes.size() > std::numeric_limits<uint32_t>::max())
      return streamError("MSFZ output: too many streams");

    BlockSize = Layout.SB->BlockSize;
    if (BlockSize == 0)
      return streamError("MSFZ output: invalid MSF block size");
    FileSize = uint64_t(BlockSize) * Layout.SB->NumBlocks;
    BlockLocations.resize(Layout.SB->NumBlocks);
    Streams.reserve(Layout.StreamSizes.size());
    for (size_t I = 0; I != Layout.StreamSizes.size(); ++I) {
      uint32_t Size = Layout.StreamSizes[I];
      ArrayRef<support::ulittle32_t> Blocks = Layout.StreamMap[I];
      Streams.push_back({Size});
      if (Size == NilStreamSize) {
        if (!Blocks.empty())
          return streamError("MSFZ output: nil stream has blocks");
        continue;
      }
      uint64_t ExpectedBlocks = bytesToBlocks(Size, BlockSize);
      if (Blocks.size() != ExpectedBlocks)
        return streamError("MSFZ output: stream block count is invalid");
      for (uint32_t BlockIndex = 0; BlockIndex != Blocks.size(); ++BlockIndex) {
        uint32_t Block = Blocks[BlockIndex];
        if (Block >= BlockLocations.size())
          return streamError("MSFZ output: stream block is outside the MSF");
        BlockLocation &Location = BlockLocations[Block];
        if (Location.StreamIndex != InvalidStreamIndex)
          return streamError("MSFZ output: stream block is assigned twice");
        uint64_t StreamOffset = uint64_t(BlockIndex) * BlockSize;
        if (StreamOffset > std::numeric_limits<uint32_t>::max())
          return streamError("MSFZ output: stream block offset is too large");
        Location.StreamIndex = I;
        Location.StreamOffset = static_cast<uint32_t>(StreamOffset);
      }
    }
    return Error::success();
  }

  Error append(uint32_t StreamIndex, uint32_t StreamOffset,
               ArrayRef<uint8_t> Data) {
    OutputStream &Stream = Streams[StreamIndex];
    if (StreamIndex == PatchableStream) {
      OS.pwrite(reinterpret_cast<const char *>(Data.data()), Data.size(),
                PatchableStreamFileOffset + StreamOffset);
      return takeOutputError();
    }

    while (!Data.empty()) {
      if (Chunks.size() >= MaxChunks)
        return streamError("MSFZ output: too many chunks");
      uint32_t Chunk = static_cast<uint32_t>(Chunks.size());
      uint32_t Offset = static_cast<uint32_t>(ChunkData.size());
      uint32_t Bytes = static_cast<uint32_t>(
          std::min<size_t>(Data.size(), ChunkSize - ChunkData.size()));
      if (!Stream.Fragments.empty() &&
          Stream.Fragments.back().FirstChunk == Chunk &&
          Stream.Fragments.back().Offset + Stream.Fragments.back().Size ==
              Offset)
        Stream.Fragments.back().Size += Bytes;
      else
        Stream.Fragments.push_back({Bytes, Chunk, Offset});
      append_range(ChunkData, Data.take_front(Bytes));
      Data = Data.drop_front(Bytes);
      StreamOffset += Bytes;
      if (ChunkData.size() == ChunkSize) {
        if (Error E = flushChunk())
          return E;
      }
    }
    return Error::success();
  }

  Error flushChunk() {
    if (ChunkData.empty())
      return Error::success();
    scope_exit ReleaseChunkData([&] {
      SmallVector<uint8_t, 0> Empty;
      ChunkData.swap(Empty);
    });
    if (Chunks.size() >= MaxChunks)
      return streamError("MSFZ output: too many chunks");

    MSFZChunkEntry Entry = {};
    Entry.Compression = static_cast<uint32_t>(MSFZCompression::Zstd);
    Entry.UncompressedSize = ChunkData.size();
    uint32_t Index = static_cast<uint32_t>(Chunks.size());
    Chunks.push_back(Entry);
    return submitChunk(Index, std::move(ChunkData));
  }

  static size_t compressionReservation(size_t Size) {
    // Account for both the input and zstd's output allocation.
    return 2 * Size + 128 * 1024;
  }

  static void compressChunk(ArrayRef<uint8_t> Data,
                            SmallVectorImpl<uint8_t> &Compressed, int Level) {
    compression::zstd::compress(Data, Compressed, Level);
  }

  Error submitChunk(uint32_t Index, SmallVector<uint8_t, 0> Data) {
    if (!CompressionPool) {
      CompressedChunk Result = {Index, {}};
      llvm::TimeTraceScope TimeScope("Compress MSFZ chunk");
      compressChunk(Data, Result.Data, CompressionLevel);
      return emitChunk(std::move(Result));
    }

    size_t ReservedBytes = compressionReservation(Data.size());
    if (ReservedBytes > MaxQueuedCompressionBytes)
      return streamError("MSFZ output: chunk exceeds compression queue limit");
    while (PendingChunks.size() >= MaxPendingChunks ||
           ReservedBytes > MaxQueuedCompressionBytes - QueuedCompressionBytes)
      if (Error E = emitNextPendingChunk())
        return E;

    auto Result = std::make_shared<CompressedChunk>(CompressedChunk{Index, {}});
    int Level = CompressionLevel;
    std::shared_future<void> Future =
        CompressionPool->async([Result, Data = std::move(Data), Level]() {
          compressChunk(Data, Result->Data, Level);
        });
    PendingChunks.push_back(
        {ReservedBytes, std::move(Result), std::move(Future)});
    QueuedCompressionBytes += ReservedBytes;
    return Error::success();
  }

  Error emitNextPendingChunk() {
    assert(!PendingChunks.empty() &&
           "cannot emit from an empty MSFZ compression queue");
    PendingChunk Pending = std::move(PendingChunks.front());
    PendingChunks.pop_front();
    {
      llvm::TimeTraceScope TimeScope("Wait for MSFZ compression");
      Pending.Future.get();
    }
    QueuedCompressionBytes -= Pending.ReservedBytes;
    return emitChunk(std::move(*Pending.Result));
  }

  Error drainPendingChunks() {
    while (!PendingChunks.empty())
      if (Error E = emitNextPendingChunk())
        return E;
    return Error::success();
  }

  Error emitChunk(CompressedChunk Result) {
    if (Result.Data.size() > std::numeric_limits<uint32_t>::max())
      return streamError("MSFZ output: compressed chunk is too large");
    MSFZChunkEntry &Entry = Chunks[Result.Index];
    Entry.FileOffset = OS.tell();
    Entry.CompressedSize = static_cast<uint32_t>(Result.Data.size());
    OS.write(reinterpret_cast<const char *>(Result.Data.data()),
             Result.Data.size());
    return takeOutputError();
  }

  Expected<uint32_t> getDirectorySize() const {
    uint64_t Size = 0;
    for (const OutputStream &Stream : Streams) {
      uint64_t StreamSize = sizeof(uint32_t);
      if (Stream.Size != NilStreamSize) {
        if (Stream.Fragments.size() >
            (std::numeric_limits<uint64_t>::max() - StreamSize) /
                (sizeof(uint32_t) + sizeof(uint64_t)))
          return streamError("MSFZ output: stream directory is too large");
        StreamSize +=
            Stream.Fragments.size() * (sizeof(uint32_t) + sizeof(uint64_t));
      }
      if (StreamSize > std::numeric_limits<uint64_t>::max() - Size)
        return streamError("MSFZ output: stream directory is too large");
      Size += StreamSize;
    }
    if (Size > std::numeric_limits<uint32_t>::max())
      return streamError("MSFZ output: stream directory is too large");
    return static_cast<uint32_t>(Size);
  }

  Error writeDirectory() {
    for (const OutputStream &Stream : Streams) {
      if (Stream.Size == NilStreamSize) {
        writeU32(OS, NilStreamSize);
        continue;
      }
      for (const OutputFragment &Fragment : Stream.Fragments) {
        writeU32(OS, Fragment.Size);
        uint64_t Location = Fragment.IsCompressed
                                ? encodeCompressedFragmentLocation(
                                      Fragment.FirstChunk, Fragment.Offset)
                                : Fragment.FileOffset;
        writeU64(OS, Location);
      }
      writeU32(OS, 0);
    }
    return takeOutputError();
  }

  Error align() {
    uint64_t Padding = alignTo(OS.tell(), 16) - OS.tell();
    OS.write_zeros(Padding);
    return takeOutputError();
  }

  Error checkStreamsComplete() const {
    for (const OutputStream &Stream : Streams) {
      if (Stream.Size != NilStreamSize && Stream.NextOffset != Stream.Size)
        return streamError("MSFZ output: stream is incomplete");
    }
    return Error::success();
  }

  Error takeOutputError() {
    OS.flush();
    if (!OS.has_error())
      return Error::success();
    std::error_code EC = OS.error();
    OS.clear_error();
    return errorCodeToError(EC);
  }

  static void updateDigestU32(MD5 &Digest, uint32_t Value) {
    support::ulittle32_t LE;
    LE = Value;
    Digest.update(bytesOf(&LE, sizeof(LE)));
  }

  sys::fs::TempFile Temp;
  std::string Path;
  raw_fd_ostream OS;
  int CompressionLevel;
  std::unique_ptr<DefaultThreadPool> CompressionPool;
  std::deque<PendingChunk> PendingChunks;
  size_t QueuedCompressionBytes = 0;
  unsigned MaxPendingChunks = 1;
  uint64_t FileSize = 0;
  uint32_t BlockSize = 0;
  std::vector<BlockLocation> BlockLocations;
  SmallVector<uint8_t, 0> ChunkData;
  SmallVector<MSFZChunkEntry, 16> Chunks;
  SmallVector<OutputStream, 32> Streams;
  uint32_t PatchableStream = InvalidStreamIndex;
  uint32_t PatchablePrefix = 0;
  uint64_t PatchableStreamFileOffset = 0;
  bool Finalized = false;
};

MSFZWriter::MSFZWriter(std::unique_ptr<Impl> Impl) : PImpl(std::move(Impl)) {}

MSFZWriter::~MSFZWriter() = default;

Expected<std::unique_ptr<MSFZWriter>>
MSFZWriter::create(StringRef Path, const MSFLayout &Layout) {
  return create(Path, Layout, compression::zstd::BestSpeedCompression);
}

Expected<std::unique_ptr<MSFZWriter>>
MSFZWriter::create(StringRef Path, const MSFLayout &Layout,
                   int CompressionLevel) {
  int MinLevel = compression::zstd::getMinCompressionLevel();
  int MaxLevel = compression::zstd::getMaxCompressionLevel();
  if (CompressionLevel < MinLevel || CompressionLevel > MaxLevel)
    return streamError(
        Twine("MSFZ output: compression level must be between ") +
        Twine(MinLevel) + " and " + Twine(MaxLevel));

  Expected<sys::fs::TempFile> TempOrErr =
      sys::fs::TempFile::create(Path + ".tmp%%%%%%%");
  if (!TempOrErr)
    return TempOrErr.takeError();
  std::unique_ptr<MSFZWriter> Result(new MSFZWriter(
      std::make_unique<Impl>(std::move(*TempOrErr), Path, CompressionLevel)));
  if (Error E = Result->PImpl->initialize(Layout))
    return std::move(E);
  return Result;
}

llvm::endianness MSFZWriter::getEndian() const {
  return llvm::endianness::little;
}

Error MSFZWriter::readBytes(uint64_t Offset, uint64_t Size,
                            ArrayRef<uint8_t> &Buffer) {
  return streamError("MSFZ writer does not support reads");
}

Error MSFZWriter::readLongestContiguousChunk(uint64_t Offset,
                                             ArrayRef<uint8_t> &Buffer) {
  return streamError("MSFZ writer does not support reads");
}

uint64_t MSFZWriter::getLength() { return PImpl->getLength(); }

Error MSFZWriter::writeBytes(uint64_t Offset, ArrayRef<uint8_t> Data) {
  return PImpl->writeBytes(Offset, Data);
}

Error MSFZWriter::commit() { return Error::success(); }

Error MSFZWriter::setInitialUncompressedStream(uint32_t StreamIndex,
                                               uint32_t PatchablePrefixSize) {
  return PImpl->setInitialUncompressedStream(StreamIndex, PatchablePrefixSize);
}

Error MSFZWriter::patchStream(uint32_t StreamIndex, uint32_t Offset,
                              ArrayRef<uint8_t> Data) {
  return PImpl->patchStream(StreamIndex, Offset, Data);
}

Expected<uint64_t> MSFZWriter::getCanonicalDigest() {
  return PImpl->getCanonicalDigest();
}

Error MSFZWriter::finalize() { return PImpl->finalize(); }

Error llvm::msf::writeMSFZ(StringRef Path, const MSFLayout &Layout,
                           ArrayRef<uint8_t> MsfData) {
  Expected<std::unique_ptr<MSFZWriter>> WriterOrErr =
      MSFZWriter::create(Path, Layout);
  if (!WriterOrErr)
    return WriterOrErr.takeError();
  std::unique_ptr<MSFZWriter> Writer = std::move(*WriterOrErr);

  if (!Layout.SB)
    return streamError("MSFZ output: missing MSF super block");
  uint32_t BlockSize = Layout.SB->BlockSize;
  for (size_t StreamIndex = 0; StreamIndex != Layout.StreamSizes.size();
       ++StreamIndex) {
    uint32_t Size = Layout.StreamSizes[StreamIndex];
    if (Size == 0 || Size == NilStreamSize)
      continue;
    uint32_t Remaining = Size;
    for (uint32_t Block : Layout.StreamMap[StreamIndex]) {
      uint64_t Offset = uint64_t(Block) * BlockSize;
      uint32_t Bytes = std::min(Remaining, BlockSize);
      if (Offset > MsfData.size() || Bytes > MsfData.size() - Offset)
        return streamError("MSFZ output: stream block is outside MSF data");
      if (Error E = Writer->writeBytes(Offset, MsfData.slice(Offset, Bytes)))
        return E;
      Remaining -= Bytes;
      if (Remaining == 0)
        break;
    }
    if (Remaining != 0)
      return streamError("MSFZ output: stream has insufficient MSF blocks");
  }
  return Writer->finalize();
}
