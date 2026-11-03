//===-- Shared/RecordReplayBlocks.h - Record replay block store -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The block store format of the input memory snapshots of the kernel record
// replay. The kernels recorded by a program mostly see the same device memory,
// so instead of a full copy per kernel, each snapshot is split into fixed-size
// blocks and every distinct block is stored once, compressed, in a block store
// file shared by all the recordings of the output directory.
//
// The input snapshot file of a kernel is then a RecordReplayBlocksHeaderTy
// followed by NumBlocks RecordReplayBlockTy entries, one per block of the
// snapshot in order. The last block may be shorter than BlockSize. The block
// store file is the concatenation of the stored blocks; entries refer to them
// by offset. All integers use the host byte order.
//
// The block store files are versioned by a generation number, and each
// snapshot names the generation it refers to:
//  - A recording appends to the latest generation, and stores a block only if
//    no snapshot of that generation already refers to the same content, which
//    it finds through the block hashes in the snapshots.
//  - Compacting the directory writes the blocks still referenced by the
//    snapshots into a new generation, switches each snapshot to it with an
//    atomic rename, and only then deletes the previous generations, so an
//    interrupted compaction leaves every snapshot valid.
//  - Recording and compacting hold the exclusive lock of the lock file of the
//    directory, and loading a snapshot holds the shared lock.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_RECORD_REPLAY_BLOCKS_H
#define OMPTARGET_SHARED_RECORD_REPLAY_BLOCKS_H

#include <cstdint>
#include <string>

/// The magic number identifying an input snapshot in block format ("OMPRRBL2").
static constexpr char RecordReplayBlocksMagic[8] = {'O', 'M', 'P', 'R',
                                                    'R', 'B', 'L', '2'};

/// The name of the lock file of the output directory.
static constexpr char RecordReplayBlockLockName[] = "record_blocks.lock";

/// The block size used when recording.
static constexpr uint64_t RecordReplayBlockSize = 64 * 1024;

/// Return the name of the block store file of generation \p Generation.
inline std::string getRecordReplayBlockStoreName(uint64_t Generation) {
  return "record_blocks." + std::to_string(Generation) + ".store";
}

/// The header of an input snapshot in block format.
struct RecordReplayBlocksHeaderTy {
  /// Must be RecordReplayBlocksMagic.
  char Magic[8];
  /// The size of the snapshot.
  uint64_t Size;
  /// The size of the blocks, except the last one.
  uint64_t BlockSize;
  /// The number of block entries that follow the header.
  uint64_t NumBlocks;
  /// The generation of the block store file that the entries refer to.
  uint64_t StoreGeneration;
};

/// How a block is stored.
enum RecordReplayBlockCodecTy : uint32_t {
  /// The block is all zeros, and nothing is stored.
  RecordReplayBlockZero = 0,
  /// The block is stored uncompressed.
  RecordReplayBlockRaw = 1,
  /// The block is stored compressed with zstd.
  RecordReplayBlockZstd = 2,
  /// The block is stored compressed with zlib.
  RecordReplayBlockZlib = 3,
};

/// The entry of a block of an input snapshot.
struct RecordReplayBlockTy {
  /// The offset of the stored block within the block store file.
  uint64_t Offset;
  /// The number of bytes stored for the block.
  uint32_t StoredSize;
  /// A RecordReplayBlockCodecTy value.
  uint32_t Codec;
  /// The 128-bit xxh3 hash of the block content, or zero for a zero block.
  uint64_t HashHigh;
  uint64_t HashLow;
};

#endif // OMPTARGET_SHARED_RECORD_REPLAY_BLOCKS_H
