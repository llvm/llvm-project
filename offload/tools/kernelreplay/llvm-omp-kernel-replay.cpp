//===- llvm-omp-kernel-replay.cpp - Replay OpenMP offload kernel ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility to replay the execution of recorded OpenMP
// offload kernels.
//
//===----------------------------------------------------------------------===//

#include "Shared/RecordReplayBlocks.h"
#include "Shared/RecordReplayDiff.h"
#include "Shared/RecordReplayPointers.h"
#include "Shared/Utils.h"
#include "omptarget.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

#define TOOL_NAME "llvm-omp-kernel-replay"
#define TOOL_PREFIX "[" TOOL_NAME "]"

cl::OptionCategory ReplayOptions(TOOL_NAME " Options");

/// The filename to read the JSON kernel description.
static cl::opt<std::string> JsonFilename(cl::Positional,
                                         cl::desc("<input kernel JSON file>"),
                                         cl::Optional);

static cl::opt<std::string> CompactStoreOpt(
    "compact-store",
    cl::desc("Compact the block store of the given recording directory, "
             "removing the blocks that no input snapshot refers to, and exit."),
    cl::value_desc("directory"), cl::cat(ReplayOptions));

static cl::opt<bool> VerifyOpt(
    "verify",
    cl::desc("Verify device memory after replaying against the record output."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool> SaveOutputOpt(
    "save-output",
    cl::desc("Save the device memory output of the replayed kernel execution, "
             "as a diff against the record input."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<uint32_t> NumTeamsOpt("num-teams",
                                     cl::desc("Set the number of teams."),
                                     cl::init(0), cl::cat(ReplayOptions));

static cl::opt<uint32_t> NumThreadsOpt("num-threads",
                                       cl::desc("Set the number of threads."),
                                       cl::init(0), cl::cat(ReplayOptions));

static cl::opt<int32_t> DeviceIdOpt("device-id", cl::desc("Set the device id."),
                                    cl::init(-1), cl::cat(ReplayOptions));

static cl::opt<uint32_t>
    RepetitionsOpt("repetitions",
                   cl::desc("Set the number of replay repetitions."),
                   cl::init(1), cl::cat(ReplayOptions));

static cl::opt<bool>
    IgnoreLimitsOpt("ignore-limits",
                    cl::desc("Ignore thread and team limits (unrecommended)."),
                    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool>
    LoadBitcodeOpt("load-bitcode",
                   cl::desc("Load the recorded IR bitcode image file."),
                   cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool> RelocateOpt(
    "relocate",
    cl::desc("Place the replay memory at any device address instead of the "
             "recorded one, allocating only the memory the kernel uses. The "
             "replay falls back to the recorded address if the recording has "
             "pointers that cannot be relocated."),
    cl::init(false), cl::cat(ReplayOptions));

template <typename... ArgsTy>
Error createErr(const char *ErrFmt, ArgsTy &&...Args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), ErrFmt,
                                 std::forward<ArgsTy>(Args)...);
}

template <typename T>
Error getInteger(const json::Object *Obj, StringRef Key, T &Result) {
  auto OptInt = Obj->getInteger(Key);
  if (!OptInt)
    return createErr("failed to read JSON integer %s", Key.data());
  Result = static_cast<T>(*OptInt);
  return Error::success();
}

Error getPointer(const json::Object *Obj, StringRef Key, void *&Result) {
  auto OptInt = Obj->getInteger(Key);
  if (!OptInt)
    return createErr("failed to read JSON integer %s", Key.data());
  Result = reinterpret_cast<void *>(*OptInt);
  return Error::success();
}

Error getString(const json::Object *Obj, StringRef Key, StringRef &Result) {
  auto OptStr = Obj->getString(Key);
  if (!OptStr)
    return createErr("failed to read JSON string %s", Key.data());
  Result = *OptStr;
  return Error::success();
}

template <typename Func>
Error processIntegerArray(const json::Object *Obj, StringRef Key,
                          Func ProcessFunc) {
  auto Array = Obj->getArray(Key);
  if (!Array)
    return createErr("failed to read JSON array %s", Key.data());

  for (const auto &Val : *Array) {
    if (auto OptInt = Val.getAsInteger())
      ProcessFunc(*OptInt);
    else
      return createErr("failed to read an integer from JSON array %s",
                       Key.data());
  }
  return Error::success();
}

/// Read the header of an output snapshot diff.
static Error readDiffHeader(StringRef Buffer, StringRef Kind,
                            RecordReplayDiffHeaderTy &Header) {
  if (Buffer.size() < sizeof(Header))
    return createErr("the kernel %s output file is truncated", Kind.data());
  std::memcpy(&Header, Buffer.data(), sizeof(Header));
  if (std::memcmp(Header.Magic, RecordReplayDiffMagic, sizeof(Header.Magic)))
    return createErr("the kernel %s output file is not a memory diff",
                     Kind.data());
  return Error::success();
}

/// A lock of a recording directory, held until destroyed.
class DirectoryLockTy {
  int FD = -1;

public:
  DirectoryLockTy() = default;
  DirectoryLockTy(const DirectoryLockTy &) = delete;
  DirectoryLockTy &operator=(const DirectoryLockTy &) = delete;
  ~DirectoryLockTy() {
    if (FD < 0)
      return;
    sys::fs::unlockFile(FD);
    sys::Process::SafelyCloseFileDescriptor(FD);
  }

  /// Lock the directory \p Dir, waiting up to \p Timeout. A shared lock is
  /// skipped if the lock file cannot be opened, e.g., in a read-only directory.
  Error lock(StringRef Dir, bool Exclusive,
             std::chrono::milliseconds Timeout = std::chrono::milliseconds(0)) {
    SmallString<128> Path(Dir);
    sys::path::append(Path, RecordReplayBlockLockName);
    if (sys::fs::openFileForReadWrite(Path, FD, sys::fs::CD_OpenAlways,
                                      sys::fs::OF_None) &&
        sys::fs::openFileForRead(Path, FD)) {
      FD = -1;
      if (Exclusive)
        return createErr("failed to open the lock file %s", Path.c_str());
      return Error::success();
    }
    if (sys::fs::tryLockFile(FD, Timeout,
                             Exclusive ? sys::fs::LockKind::Exclusive
                                       : sys::fs::LockKind::Shared)) {
      sys::Process::SafelyCloseFileDescriptor(FD);
      FD = -1;
      return createErr("the recording directory %s is in use by a recording "
                       "or a compaction",
                       Dir.str().c_str());
    }
    return Error::success();
  }
};

/// Read the header and the block entries of an input snapshot \p File, and
/// return whether it is in block format.
static Expected<bool>
readBlockSnapshot(StringRef File, RecordReplayBlocksHeaderTy &Header,
                  SmallVectorImpl<RecordReplayBlockTy> &Blocks) {
  if (File.size() < sizeof(Header) ||
      std::memcmp(File.data(), RecordReplayBlocksMagic, sizeof(Header.Magic)))
    return false;
  std::memcpy(&Header, File.data(), sizeof(Header));
  if (!Header.BlockSize ||
      Header.NumBlocks != divideCeil(Header.Size, Header.BlockSize) ||
      File.size() - sizeof(Header) !=
          Header.NumBlocks * sizeof(RecordReplayBlockTy))
    return createErr("the kernel record input file has an invalid format");
  Blocks.resize(Header.NumBlocks);
  std::memcpy(Blocks.data(), File.data() + sizeof(Header),
              Blocks.size() * sizeof(RecordReplayBlockTy));
  return true;
}

/// Compact the block store of the recording directory \p Dir: copy the blocks
/// that the input snapshots refer to into a new generation of the store, switch
/// each snapshot to it, and delete the previous generations. The snapshots are
/// replaced atomically, and the previous generations are only deleted once no
/// snapshot refers to them, so an interrupted compaction leaves the recordings
/// valid; compacting again completes it.
static Error compactBlockStore(StringRef Dir) {
  DirectoryLockTy Lock;
  if (auto Err = Lock.lock(Dir, /*Exclusive=*/true))
    return Err;

  // Find the block store generations and the snapshots in block format.
  struct SnapshotTy {
    std::string Path;
    RecordReplayBlocksHeaderTy Header;
    SmallVector<RecordReplayBlockTy> Blocks;
  };
  std::vector<SnapshotTy> Snapshots;
  std::map<uint64_t, std::string> Stores;
  uint64_t NewGeneration = 0;
  std::error_code EC;
  for (sys::fs::directory_iterator It(Dir, EC), End; It != End && !EC;
       It.increment(EC)) {
    StringRef Path = It->path();
    StringRef Name = sys::path::filename(Path);
    uint64_t Generation;
    if (Name.consume_front("record_blocks.") && Name.consume_back(".store") &&
        !Name.getAsInteger(10, Generation)) {
      Stores[Generation] = Path.str();
      NewGeneration = std::max(NewGeneration, Generation + 1);
      continue;
    }
    if (sys::path::extension(Path) != ".record_input")
      continue;
    auto FileOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
    if (!FileOrErr)
      return createErr("failed to read %s", Path.str().c_str());
    SnapshotTy Snapshot{Path.str(), {}, {}};
    auto IsBlocksOrErr = readBlockSnapshot((*FileOrErr)->getBuffer(),
                                           Snapshot.Header, Snapshot.Blocks);
    if (!IsBlocksOrErr)
      return IsBlocksOrErr.takeError();
    if (*IsBlocksOrErr) {
      NewGeneration =
          std::max(NewGeneration, Snapshot.Header.StoreGeneration + 1);
      Snapshots.push_back(std::move(Snapshot));
    }
  }
  if (EC)
    return createErr("failed to read the directory %s", Dir.str().c_str());

  // Copy the blocks that the snapshots refer to into the new generation.
  SmallString<128> NewStorePath(Dir);
  sys::path::append(NewStorePath, getRecordReplayBlockStoreName(NewGeneration));
  std::string TmpStorePath = (NewStorePath + ".tmp").str();
  std::map<uint64_t, std::unique_ptr<MemoryBuffer>> OldStores;
  DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> NewOffsets;
  uint64_t OldSize = 0, NewSize = 0;
  {
    raw_fd_ostream OS(TmpStorePath, EC);
    if (EC)
      return createErr("failed to create %s", TmpStorePath.c_str());
    for (SnapshotTy &Snapshot : Snapshots) {
      std::unique_ptr<MemoryBuffer> &Store =
          OldStores[Snapshot.Header.StoreGeneration];
      if (!Store) {
        auto It = Stores.find(Snapshot.Header.StoreGeneration);
        if (It != Stores.end())
          if (auto StoreOrErr =
                  MemoryBuffer::getFile(It->second, /*IsText=*/false,
                                        /*RequiresNullTerminator=*/false))
            Store = std::move(*StoreOrErr);
        if (!Store) {
          OS.close();
          sys::fs::remove(TmpStorePath);
          return createErr("%s refers to a missing block store",
                           Snapshot.Path.c_str());
        }
      }
      for (RecordReplayBlockTy &Block : Snapshot.Blocks) {
        if (Block.Codec == RecordReplayBlockZero)
          continue;
        auto [It, Inserted] =
            NewOffsets.try_emplace({Block.HashHigh, Block.HashLow}, NewSize);
        if (Inserted) {
          if (Block.Offset > Store->getBufferSize() ||
              Block.StoredSize > Store->getBufferSize() - Block.Offset) {
            OS.close();
            sys::fs::remove(TmpStorePath);
            return createErr("%s refers to data beyond its block store",
                             Snapshot.Path.c_str());
          }
          OS.write(Store->getBufferStart() + Block.Offset, Block.StoredSize);
          NewSize += Block.StoredSize;
        }
        Block.Offset = It->second;
      }
      Snapshot.Header.StoreGeneration = NewGeneration;
    }
    OS.close();
    if (OS.has_error()) {
      OS.clear_error();
      sys::fs::remove(TmpStorePath);
      return createErr("failed to write %s", TmpStorePath.c_str());
    }
  }
  if (sys::fs::rename(TmpStorePath, NewStorePath))
    return createErr("failed to create %s", NewStorePath.c_str());

  // Switch each snapshot to the new generation.
  for (const SnapshotTy &Snapshot : Snapshots) {
    std::string TmpPath = Snapshot.Path + ".tmp";
    raw_fd_ostream OS(TmpPath, EC);
    if (EC)
      return createErr("failed to create %s", TmpPath.c_str());
    OS.write(reinterpret_cast<const char *>(&Snapshot.Header),
             sizeof(Snapshot.Header));
    OS.write(reinterpret_cast<const char *>(Snapshot.Blocks.data()),
             Snapshot.Blocks.size() * sizeof(RecordReplayBlockTy));
    OS.close();
    if (OS.has_error()) {
      OS.clear_error();
      sys::fs::remove(TmpPath);
      return createErr("failed to write %s", TmpPath.c_str());
    }
    if (sys::fs::rename(TmpPath, Snapshot.Path))
      return createErr("failed to replace %s", Snapshot.Path.c_str());
  }

  // No snapshot refers to the previous generations anymore.
  for (const auto &[Generation, Path] : Stores) {
    uint64_t Size = 0;
    if (!sys::fs::file_size(Path, Size))
      OldSize += Size;
    sys::fs::remove(Path);
  }

  outs() << TOOL_PREFIX << " Compacted the block store of " << Snapshots.size()
         << " snapshots from " << OldSize << " to " << NewSize << " bytes ("
         << NewOffsets.size() << " blocks)\n";
  return Error::success();
}

/// Load the record input file. If it is in block format, reassemble the input
/// snapshot from the block store of its directory. \p Writable requests a
/// writable buffer.
static Expected<std::unique_ptr<MemoryBuffer>>
loadRecordInput(const std::filesystem::path &Filepath, bool Writable) {
  // Do not load the snapshot while the directory is being compacted.
  DirectoryLockTy Lock;
  if (auto Err = Lock.lock(Filepath.parent_path().string(),
                           /*Exclusive=*/false, std::chrono::minutes(1)))
    return std::move(Err);

  auto FileOrErr = MemoryBuffer::getFile(Filepath.c_str(), /*IsText=*/false,
                                         /*RequiresNullTerminator=*/false);
  if (!FileOrErr)
    return createErr("failed to read the kernel record input file");
  StringRef File = (*FileOrErr)->getBuffer();

  RecordReplayBlocksHeaderTy Header;
  SmallVector<RecordReplayBlockTy> Blocks;
  auto IsBlocksOrErr = readBlockSnapshot(File, Header, Blocks);
  if (!IsBlocksOrErr)
    return IsBlocksOrErr.takeError();
  if (!*IsBlocksOrErr) {
    // A full snapshot.
    if (!Writable)
      return std::move(*FileOrErr);
    auto Copy = WritableMemoryBuffer::getNewUninitMemBuffer(File.size());
    if (!Copy)
      return createErr("failed to allocate the kernel record input");
    std::memcpy(Copy->getBufferStart(), File.data(), File.size());
    return std::unique_ptr<MemoryBuffer>(std::move(Copy));
  }

  for (const RecordReplayBlockTy &Block : Blocks) {
    if (Block.Codec == RecordReplayBlockZstd &&
        !compression::zstd::isAvailable())
      return createErr("the record input is compressed with zstd, which this "
                       "build does not support");
    if (Block.Codec == RecordReplayBlockZlib &&
        !compression::zlib::isAvailable())
      return createErr("the record input is compressed with zlib, which this "
                       "build does not support");
  }

  std::filesystem::path StorePath =
      Filepath.parent_path() /
      getRecordReplayBlockStoreName(Header.StoreGeneration);
  auto StoreOrErr = MemoryBuffer::getFile(StorePath.c_str(), /*IsText=*/false,
                                          /*RequiresNullTerminator=*/false);
  if (!StoreOrErr)
    return createErr("failed to read the block store file");
  ArrayRef<uint8_t> Store = arrayRefFromStringRef((*StoreOrErr)->getBuffer());

  auto Snapshot = WritableMemoryBuffer::getNewUninitMemBuffer(Header.Size);
  if (!Snapshot)
    return createErr("failed to allocate the kernel record input");

  // Reassemble the blocks in parallel.
  std::atomic<bool> Failed = false;
  parallelFor(0, Blocks.size(), [&](size_t I) {
    const RecordReplayBlockTy &Block = Blocks[I];
    uint64_t Offset = I * Header.BlockSize;
    size_t Size = std::min(Header.BlockSize, Header.Size - Offset);
    uint8_t *Dst =
        reinterpret_cast<uint8_t *>(Snapshot->getBufferStart()) + Offset;
    if (Block.Codec == RecordReplayBlockZero) {
      std::memset(Dst, 0, Size);
      return;
    }
    if (Block.Offset > Store.size() ||
        Block.StoredSize > Store.size() - Block.Offset) {
      Failed = true;
      return;
    }
    ArrayRef<uint8_t> Stored = Store.slice(Block.Offset, Block.StoredSize);
    auto LoadBlock = [&]() -> bool {
      size_t DecompressedSize = Size;
      switch (Block.Codec) {
      case RecordReplayBlockRaw:
        if (Stored.size() != Size)
          return false;
        std::memcpy(Dst, Stored.data(), Size);
        return true;
      case RecordReplayBlockZstd:
        if (Error Err =
                compression::zstd::decompress(Stored, Dst, DecompressedSize)) {
          consumeError(std::move(Err));
          return false;
        }
        return DecompressedSize == Size;
      case RecordReplayBlockZlib:
        if (Error Err =
                compression::zlib::decompress(Stored, Dst, DecompressedSize)) {
          consumeError(std::move(Err));
          return false;
        }
        return DecompressedSize == Size;
      default:
        return false;
      }
    };
    if (!LoadBlock())
      Failed = true;
  });
  if (Failed)
    return createErr("failed to load the kernel record input from the block "
                     "store");
  return std::unique_ptr<MemoryBuffer>(std::move(Snapshot));
}

/// The relocation of the replay memory to another device address.
struct RelocationTy {
  /// The sorted offsets of the pointers into the replay memory.
  SmallVector<uint64_t> PointerOffsets;
  /// The replay memory address and size.
  uint64_t Address = 0;
  uint64_t Size = 0;
  /// The replay memory address minus the recorded one.
  uint64_t Delta = 0;
  /// The relocated input snapshot.
  StringRef Input;

  /// Whether \p Value points into the recorded memory, or just past its end.
  bool pointsToRecordedMemory(uint64_t Value) const {
    return Value >= Address - Delta && Value <= Address - Delta + Size;
  }
};

/// The contents of a pointers file.
struct PointersFileTy {
  RecordReplayPointersHeaderTy Header;
  SmallVector<uint64_t> PointerOffsets;
};

/// Read the pointers file that describes how to relocate the replay memory.
static Expected<PointersFileTy> readPointersFile(StringRef Filename) {
  auto BufferOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  if (!BufferOrErr)
    return createErr("failed to read the pointers file");

  StringRef Buffer = (*BufferOrErr)->getBuffer();
  PointersFileTy File;
  if (Buffer.size() < sizeof(File.Header))
    return createErr("the pointers file is truncated");
  std::memcpy(&File.Header, Buffer.data(), sizeof(File.Header));
  if (std::memcmp(File.Header.Magic, RecordReplayPointersMagic,
                  sizeof(File.Header.Magic)))
    return createErr("the pointers file has an invalid format");
  if (Buffer.size() - sizeof(File.Header) !=
      File.Header.NumPointers * sizeof(uint64_t))
    return createErr("the pointers file is truncated");

  File.PointerOffsets.resize(File.Header.NumPointers);
  std::memcpy(File.PointerOffsets.data(), Buffer.data() + sizeof(File.Header),
              File.Header.NumPointers * sizeof(uint64_t));
  return File;
}

/// Relocate the pointers stored in the regions of the record output diff
/// \p Output to the replay memory, so the replay can be verified against it.
/// The regions must have been checked.
static void relocateRecordOutput(MutableArrayRef<char> Output,
                                 const RelocationTy &Relocation) {
  constexpr uint64_t PtrSize = sizeof(uint64_t);
  uint64_t Pos = sizeof(RecordReplayDiffHeaderTy);
  while (Pos < Output.size()) {
    RecordReplayDiffRegionTy Region;
    std::memcpy(&Region, Output.data() + Pos, sizeof(Region));
    Pos += sizeof(Region);

    // Adjust each pointer that overlaps the region. Its bytes outside the
    // region are unchanged, so they are taken from the input. Every logged
    // pointer of the input was relocated, so its recorded value is the
    // relocated one minus the delta.
    uint64_t RegionEnd = Region.Offset + Region.Size;
    auto It =
        llvm::lower_bound(Relocation.PointerOffsets,
                          Region.Offset - std::min(Region.Offset, PtrSize - 1));
    for (; It != Relocation.PointerOffsets.end() && *It < RegionEnd; ++It) {
      uint64_t PtrOffset = *It;
      if (PtrOffset + PtrSize > Relocation.Input.size())
        continue;

      uint64_t Value;
      std::memcpy(&Value, Relocation.Input.data() + PtrOffset, PtrSize);
      Value -= Relocation.Delta;
      char Bytes[PtrSize];
      std::memcpy(Bytes, &Value, PtrSize);
      for (uint64_t I = 0; I < PtrSize; ++I) {
        uint64_t Offset = PtrOffset + I;
        if (Offset >= Region.Offset && Offset < RegionEnd)
          Bytes[I] = Output[Pos + Offset - Region.Offset];
      }
      std::memcpy(&Value, Bytes, PtrSize);
      if (!Relocation.pointsToRecordedMemory(Value))
        continue;
      Value += Relocation.Delta;
      std::memcpy(Bytes, &Value, PtrSize);

      for (uint64_t I = 0; I < PtrSize; ++I) {
        uint64_t Offset = PtrOffset + I;
        if (Offset >= Region.Offset && Offset < RegionEnd)
          Output[Pos + Offset - Region.Offset] = Bytes[I];
      }
    }
    Pos += Region.Size;
  }
}

/// Load the record output file, decompressing it if needed, and check its
/// regions. If the replay memory is relocated, the pointers in the record
/// output are relocated the same way, so the replay can be verified against
/// it.
static Expected<std::unique_ptr<MemoryBuffer>>
loadRecordOutput(StringRef Filename, const RelocationTy *Relocation) {
  auto FileOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/false,
                                         /*RequiresNullTerminator=*/false);
  if (!FileOrErr)
    return createErr("failed to read the kernel record output file");
  StringRef File = (*FileOrErr)->getBuffer();

  std::unique_ptr<MemoryBuffer> Output;
  RecordReplayDiffCompressedHeaderTy Compressed;
  if (File.size() >= sizeof(Compressed) &&
      !std::memcmp(File.data(), RecordReplayDiffCompressedMagic,
                   sizeof(Compressed.Magic))) {
    std::memcpy(&Compressed, File.data(), sizeof(Compressed));
    std::unique_ptr<WritableMemoryBuffer> Decompressed =
        WritableMemoryBuffer::getNewUninitMemBuffer(Compressed.Size);
    if (!Decompressed)
      return createErr("failed to allocate the kernel record output");
    ArrayRef<uint8_t> Data =
        arrayRefFromStringRef(File.drop_front(sizeof(Compressed)));
    uint8_t *Dest = reinterpret_cast<uint8_t *>(Decompressed->getBufferStart());
    bool IsZstd = Compressed.Codec == RecordReplayDiffZstd;
    if (!(IsZstd ? compression::zstd::isAvailable()
                 : Compressed.Codec == RecordReplayDiffZlib &&
                       compression::zlib::isAvailable()))
      return createErr("the kernel record output file is compressed in a "
                       "format this build cannot read");
    size_t Size = Compressed.Size;
    Error Err = IsZstd ? compression::zstd::decompress(Data, Dest, Size)
                       : compression::zlib::decompress(Data, Dest, Size);
    if (Err || Size != Compressed.Size) {
      consumeError(std::move(Err));
      return createErr("the kernel record output file is corrupted");
    }
    Output = std::move(Decompressed);
  } else if (Relocation) {
    // Relocating needs a writable copy.
    auto CopyOrErr =
        WritableMemoryBuffer::getFile(Filename, /*IsVolatile=*/false);
    if (!CopyOrErr)
      return createErr("failed to read the kernel record output file");
    Output = std::move(*CopyOrErr);
  } else {
    Output = std::move(*FileOrErr);
  }

  StringRef Diff = Output->getBuffer();
  RecordReplayDiffHeaderTy Header;
  if (auto Err = readDiffHeader(Diff, "record", Header))
    return std::move(Err);
  for (uint64_t Pos = sizeof(Header), End = 0; Pos < Diff.size();) {
    RecordReplayDiffRegionTy Region;
    if (Diff.size() - Pos < sizeof(Region))
      return createErr("the kernel record output file is truncated");
    std::memcpy(&Region, Diff.data() + Pos, sizeof(Region));
    Pos += sizeof(Region);
    if (Diff.size() - Pos < Region.Size)
      return createErr("the kernel record output file is truncated");
    if (Region.Offset < End || Region.Size > Header.OutputSize - Region.Offset)
      return createErr("the kernel record output file has an invalid region");
    Pos += Region.Size;
    End = Region.Offset + Region.Size;
  }

  if (Relocation)
    relocateRecordOutput(
        MutableArrayRef<char>(const_cast<char *>(Diff.data()), Diff.size()),
        *Relocation);
  return std::move(Output);
}

/// Replay the kernel and return whether verification occurred.
Error replayKernel() {
  if (RepetitionsOpt == 0)
    return createErr("invalid number of repetitions");

  // Load the kernel descriptor JSON file.
  auto KernelDescrBufferOrErr =
      MemoryBuffer::getFile(JsonFilename, /*isText=*/true,
                            /*RequiresNullTerminator=*/true);
  if (!KernelDescrBufferOrErr)
    return createErr("failed read the kernel info JSON file");

  // Parse the JSON file.
  auto JsonDescrOrErr = json::parse(KernelDescrBufferOrErr.get()->getBuffer());
  if (!JsonDescrOrErr)
    return JsonDescrOrErr.takeError();

  auto JsonObj = JsonDescrOrErr->getAsObject();
  if (!JsonObj)
    return createErr("invalid JSON file");

  // Retrieve the values from the JSON file.
  uint32_t NumTeams, NumThreads, SharedMemorySize, DeviceId, NumArgs;
  if (auto Err = getInteger(JsonObj, "NumTeams", NumTeams))
    return Err;
  if (auto Err = getInteger(JsonObj, "NumThreads", NumThreads))
    return Err;
  if (auto Err = getInteger(JsonObj, "SharedMemorySize", SharedMemorySize))
    return Err;
  if (auto Err = getInteger(JsonObj, "DeviceId", DeviceId))
    return Err;
  if (auto Err = getInteger(JsonObj, "NumArgs", NumArgs))
    return Err;

  uint64_t LoopTripCount, VAllocSize;
  if (auto Err = getInteger(JsonObj, "VAllocSize", VAllocSize))
    return Err;
  if (auto Err = getInteger(JsonObj, "LoopTripCount", LoopTripCount))
    return Err;

  void *VAllocAddr;
  if (auto Err = getPointer(JsonObj, "VAllocAddr", VAllocAddr))
    return Err;

  StringRef KernelName;
  if (auto Err = getString(JsonObj, "Name", KernelName))
    return Err;

  // If needed, adjust number of teams and threads, and the device identifier.
  NumTeams = NumTeamsOpt > 0 ? NumTeamsOpt : NumTeams;
  NumThreads = NumThreadsOpt > 0 ? NumThreadsOpt : NumThreads;
  DeviceId = DeviceIdOpt >= 0 ? DeviceIdOpt : DeviceId;

  // Retrieve the teams and threads limits (min and max).
  SmallVector<uint32_t> TeamsLimits;
  auto Err = processIntegerArray(JsonObj, "TeamsLimits", [&](uint64_t Val) {
    TeamsLimits.push_back(static_cast<uint32_t>(Val));
  });
  if (Err)
    return Err;

  SmallVector<uint32_t> ThreadsLimits;
  Err = processIntegerArray(JsonObj, "ThreadsLimits", [&](uint64_t Val) {
    ThreadsLimits.push_back(static_cast<uint32_t>(Val));
  });
  if (Err)
    return Err;

  // Check that a minimum and maximum have been exported.
  if (TeamsLimits.size() != 2 || ThreadsLimits.size() != 2)
    return createErr("TeamsLimits and ThreadsLimits must have a min and max");

  // Check that the minimum and maximum are specified or both are zero.
  if (bool(TeamsLimits[0]) != bool(TeamsLimits[1]))
    return createErr("TeamsLimits min and max are inconsistent");
  if (bool(ThreadsLimits[0]) != bool(ThreadsLimits[1]))
    return createErr("ThreadsLimits min and max are inconsistent");

  // If the limits were specified, verify the selected values are valid.
  if (!IgnoreLimitsOpt && TeamsLimits[0] > 0 &&
      (NumTeams < TeamsLimits[0] || NumTeams > TeamsLimits[1]))
    return createErr("number of teams (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     NumTeams, TeamsLimits[0], TeamsLimits[1]);
  if (!IgnoreLimitsOpt && ThreadsLimits[0] > 0 &&
      (NumThreads < ThreadsLimits[0] || NumThreads > ThreadsLimits[1]))
    return createErr("number of threads (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     NumThreads, ThreadsLimits[0], ThreadsLimits[1]);

  // Retrieve the arguments of the kernel.
  SmallVector<void *> TgtArgs;
  Err = processIntegerArray(JsonObj, "ArgPtrs", [&](uint64_t Val) {
    TgtArgs.push_back(reinterpret_cast<void *>(Val));
  });
  if (Err)
    return Err;

  SmallVector<ptrdiff_t> TgtArgOffsets;
  Err = processIntegerArray(JsonObj, "ArgOffsets", [&](uint64_t Val) {
    TgtArgOffsets.push_back(static_cast<ptrdiff_t>(Val));
  });
  if (Err)
    return Err;

  // Keep the filepath and directory for future use.
  auto Filepath = std::filesystem::path(JsonFilename.getValue());
  auto Directory = Filepath.parent_path();

  // Load the recorded globals file.
  Filepath.replace_extension("globals");
  auto GlobalsBufferOrErr =
      MemoryBuffer::getFile(Filepath.c_str(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!GlobalsBufferOrErr)
    return createErr("failed to read the globals file");
  auto GlobalsBuffer = std::move(GlobalsBufferOrErr.get());

  const void *BufferPtr = const_cast<char *>(GlobalsBuffer->getBufferStart());
  uint32_t NumGlobals = *((const uint32_t *)(BufferPtr));
  BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

  SmallVector<llvm::offloading::EntryTy> OffloadEntries(
      NumGlobals + 1, {0x0, 0x1, object::OffloadKind::OFK_OpenMP, 0, nullptr,
                       nullptr, 0, 0, nullptr});

  // The first offload entry corresponds to the kernel function.
  OffloadEntries[0].SymbolName = const_cast<char *>(KernelName.data());
  // Use a unique identifier.
  OffloadEntries[0].Address = (void *)0x1;

  // The rest of entries correspond to the recorded global variables.
  for (uint32_t I = 0; I < NumGlobals; ++I) {
    auto &Global = OffloadEntries[I + 1];

    // Use a unique identifier.
    Global.Address = static_cast<char *>(OffloadEntries[0].Address) + I + 1;

    // Setup the offload entry using the information from the file.
    uint32_t NameSize = *((const uint32_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));
    uint64_t Size = *((const uint64_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint64_t));
    Global.Size = Size;
    Global.SymbolName =
        const_cast<char *>(static_cast<const char *>(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, NameSize);
    Global.AuxAddr = const_cast<void *>(BufferPtr);
    BufferPtr = utils::advancePtr(BufferPtr, Size);
  }

  // Load the device image file.
  Filepath.replace_extension(LoadBitcodeOpt ? "bc" : "image");
  auto ImageBufferOrErr =
      MemoryBuffer::getFile(Filepath.c_str(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!ImageBufferOrErr)
    return createErr("failed to read the kernel image file");
  auto ImageBuffer = std::move(ImageBufferOrErr.get());

  // Prepare the device image and binary descriptor.
  __tgt_device_image DeviceImage;
  DeviceImage.ImageStart = const_cast<char *>(ImageBuffer->getBufferStart());
  DeviceImage.ImageEnd = const_cast<char *>(ImageBuffer->getBufferEnd());
  DeviceImage.EntriesBegin = &OffloadEntries[0];
  DeviceImage.EntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;

  __tgt_bin_desc Desc;
  Desc.NumDeviceImages = 1;
  Desc.HostEntriesBegin = &OffloadEntries[0];
  Desc.HostEntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;
  Desc.DeviceImages = &DeviceImage;

  // Register the image and the offload entries.
  __tgt_register_lib(&Desc);

  // Decide whether the replay memory can be placed at another address. That
  // requires knowing every pointer into the recorded memory and, to verify the
  // replay, into the output.
  std::optional<PointersFileTy> Pointers;
  if (RelocateOpt) {
    Filepath.replace_extension("pointers");
    std::string Reason;
    if (!std::filesystem::exists(Filepath)) {
      Reason = "the recording has no pointers file";
    } else {
      auto PointersOrErr = readPointersFile(Filepath.string());
      if (!PointersOrErr)
        return PointersOrErr.takeError();
      const RecordReplayPointersHeaderTy &Header = PointersOrErr->Header;
      if (Header.NumUnknownInputPointers)
        Reason = std::to_string(Header.NumUnknownInputPointers) +
                 " unknown pointers in the recorded memory";
      else if (VerifyOpt && Header.NumUnknownOutputPointers &&
               Header.NumUnknownOutputPointers != RecordReplayUnknownCount)
        Reason = std::to_string(Header.NumUnknownOutputPointers) +
                 " unknown pointers in the recorded output";
      else
        Pointers = std::move(*PointersOrErr);
    }
    if (!Pointers)
      outs() << TOOL_PREFIX << " Replaying at the recorded address: " << Reason
             << "\n";
  }

  // Load the record input file. A relocated input needs a writable copy.
  Filepath.replace_extension("record_input");
  auto RecordInputBufferOrErr =
      loadRecordInput(Filepath, /*Writable=*/Pointers.has_value());
  if (!RecordInputBufferOrErr)
    return RecordInputBufferOrErr.takeError();
  std::unique_ptr<MemoryBuffer> RecordInputBuffer =
      std::move(*RecordInputBufferOrErr);
  uint64_t InputSize = RecordInputBuffer->getBufferSize();

  // A relocated replay memory only needs to hold the input snapshot.
  int Rc = __tgt_activate_record_replay(
      DeviceId, Pointers ? std::max<uint64_t>(InputSize, 1) : VAllocSize,
      Pointers ? nullptr : VAllocAddr, /*IsRecord=*/false, SaveOutputOpt,
      /*EmitReport=*/false, Directory.c_str());
  if (Rc != OMP_TGT_SUCCESS)
    return createErr("failed to activate record replay");

  KernelReplayOutcomeTy Outcome;

  // Relocate the pointers in the input snapshot and the kernel arguments to
  // the replay memory. The first allocation is placed at its start, and it is
  // reused by every replay.
  std::optional<RelocationTy> Relocation;
  if (Pointers && InputSize) {
    void *Address = omp_target_alloc(InputSize, DeviceId);
    if (!Address)
      return createErr("failed to allocate the replay memory");
    Outcome.ReplayDeviceAlloc = Address;

    uint64_t RecordAddress = reinterpret_cast<uintptr_t>(VAllocAddr);
    Relocation.emplace();
    Relocation->PointerOffsets = std::move(Pointers->PointerOffsets);
    Relocation->Address = reinterpret_cast<uintptr_t>(Address);
    Relocation->Size = InputSize;
    Relocation->Delta = Relocation->Address - RecordAddress;
    Relocation->Input = RecordInputBuffer->getBuffer();

    char *Input = const_cast<char *>(RecordInputBuffer->getBufferStart());
    for (uint64_t Offset : Relocation->PointerOffsets) {
      if (Offset + sizeof(uint64_t) > InputSize)
        return createErr("the pointers file has an invalid offset");
      uint64_t Value;
      std::memcpy(&Value, Input + Offset, sizeof(Value));
      Value += Relocation->Delta;
      std::memcpy(Input + Offset, &Value, sizeof(Value));
    }
    for (void *&Arg : TgtArgs) {
      uint64_t Value = reinterpret_cast<uintptr_t>(Arg);
      if (Value >= RecordAddress && Value <= RecordAddress + InputSize)
        Arg = reinterpret_cast<void *>(Value + Relocation->Delta);
    }

    outs() << TOOL_PREFIX << " Relocated the replay memory from " << VAllocAddr
           << " to " << Address << " (" << Relocation->PointerOffsets.size()
           << " pointers)\n";
  }

  // The replay compares the device memory with the record output after the
  // last repetition, without storing the replay output.
  std::unique_ptr<MemoryBuffer> RecordOutput;
  if (VerifyOpt) {
    Filepath.replace_extension("record_output");
    auto RecordOutputOrErr = loadRecordOutput(
        Filepath.string(), Relocation ? &*Relocation : nullptr);
    if (!RecordOutputOrErr)
      return RecordOutputOrErr.takeError();
    RecordOutput = std::move(*RecordOutputOrErr);
  }

  // Perform the kernel replay and verification (if needed) for each repetition.
  for (uint32_t R = 1; R <= RepetitionsOpt; ++R) {
    if (RecordOutput && R == RepetitionsOpt) {
      Outcome.ExpectedOutput = RecordOutput->getBufferStart();
      Outcome.ExpectedOutputSize = RecordOutput->getBufferSize();
    }
    Rc = __tgt_target_kernel_replay(
        /*Loc=*/nullptr, DeviceId, OffloadEntries[0].Address,
        const_cast<char *>(RecordInputBuffer->getBufferStart()),
        R > 0 ? Outcome.ReplayDeviceAlloc : nullptr,
        RecordInputBuffer->getBufferSize(),
        NumGlobals ? &OffloadEntries[1] : nullptr, NumGlobals, TgtArgs.data(),
        TgtArgOffsets.data(), NumArgs, NumTeams, NumThreads, SharedMemorySize,
        LoopTripCount, &Outcome);
    if (Rc != OMP_TGT_SUCCESS)
      return createErr("failed to replay kernel");

    outs() << TOOL_PREFIX << " Replay time (" << R
           << "): " << Outcome.KernelReplayTimeNs << " ns\n";
  }

  // Verify the replay output if requested.
  if (VerifyOpt) {
    using VerificationTy = KernelReplayOutcomeTy::VerificationTy;
    if (Outcome.Verification == VerificationTy::None)
      return createErr("the replay device memory was not verified");
    if (Outcome.Verification == VerificationTy::Mismatch)
      return createErr("replay device memory failed to verify (first "
                       "difference at offset %" PRIu64 ")",
                       Outcome.MismatchOffset);

    // The verification was successful.
    outs() << TOOL_PREFIX << " Replay done, device memory verified\n";
  } else {
    outs() << TOOL_PREFIX << " Replay done, verification skipped\n";
  }
  return Error::success();
}

int main(int Argc, char **Argv) {
  cl::HideUnrelatedOptions(ReplayOptions);
  cl::ParseCommandLineOptions(Argc, Argv, TOOL_NAME "\n");

  auto Run = []() -> Error {
    if (!CompactStoreOpt.empty())
      return compactBlockStore(CompactStoreOpt);
    if (JsonFilename.empty())
      return createErr("no input kernel JSON file");
    return replayKernel();
  };
  if (auto Err = Run()) {
    errs() << TOOL_PREFIX << " Error: " << llvm::toString(std::move(Err))
           << "\n";
    return 1;
  }
  return 0;
}
