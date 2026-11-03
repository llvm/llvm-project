//===-- Shared/RecordReplayDiff.h - Record replay diff format ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The format of the output memory snapshots of the kernel record replay. Rather
// than the whole device memory after the kernel execution, an output snapshot
// only stores the memory regions that differ from the input snapshot, i.e., the
// device memory just before the kernel execution.
//
// The file starts with a RecordReplayDiffHeaderTy, followed by a sequence of
// regions, sorted by offset and non-overlapping. Each region is a
// RecordReplayDiffRegionTy followed by the region's Size bytes of output
// memory. All integers use the host byte order.
//
// The encoding is canonical: two diffs relative to the same input snapshot are
// byte-identical if and only if they describe the same output memory. Regions
// separated by fewer than RecordReplayDiffMaxGap unchanged bytes are merged,
// and the output bytes beyond the input snapshot size are always stored.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_RECORD_REPLAY_DIFF_H
#define OMPTARGET_SHARED_RECORD_REPLAY_DIFF_H

#include <cstdint>

/// The magic number identifying an output snapshot diff file ("OMPRRDF1").
static constexpr char RecordReplayDiffMagic[8] = {'O', 'M', 'P', 'R',
                                                  'R', 'D', 'F', '1'};

/// The header of an output snapshot diff file.
struct RecordReplayDiffHeaderTy {
  /// Must be RecordReplayDiffMagic.
  char Magic[8];
  /// The size of the input snapshot the diff is relative to.
  uint64_t InputSize;
  /// The size of the output memory described by the diff.
  uint64_t OutputSize;
};

/// The descriptor of a region of the output memory stored in the diff.
struct RecordReplayDiffRegionTy {
  /// The offset of the region within the output memory.
  uint64_t Offset;
  /// The number of bytes of the region that follow the descriptor.
  uint64_t Size;
};

/// Regions separated by fewer unchanged bytes than this are merged, since
/// storing a short run of unchanged bytes is cheaper than another descriptor.
static constexpr uint64_t RecordReplayDiffMaxGap =
    sizeof(RecordReplayDiffRegionTy);

#endif // OMPTARGET_SHARED_RECORD_REPLAY_DIFF_H
