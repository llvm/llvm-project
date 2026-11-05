//===-- Shared/RecordReplayPointers.h - Record replay pointers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The format of the pointers file of the kernel record replay, which allows
// replaying a kernel with its memory placed at a different device address than
// the recorded one (relocation).
//
// During the recording, the runtime logs the device locations where it stores
// device pointers, e.g., when attaching a pointer to its pointee. The pointers
// file lists the offsets, within the record replay memory, of the logged
// locations that hold a pointer into that memory before the kernel execution.
// Replaying at another address requires adjusting these pointers.
//
// Any other 8-byte aligned value in the input snapshot or the globals that
// points into the record replay memory in use is an unknown pointer, which
// prevents relocating the replay. An unknown pointer in the output snapshot
// prevents verifying a relocated replay. Such values may also be other data
// that happens to look like a pointer, so the check is conservative.
//
// The file is a RecordReplayPointersHeaderTy followed by NumPointers sorted
// uint64_t offsets. All integers use the host byte order.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_RECORD_REPLAY_POINTERS_H
#define OMPTARGET_SHARED_RECORD_REPLAY_POINTERS_H

#include <cstdint>

/// The magic number identifying a pointers file ("OMPRRPT1").
static constexpr char RecordReplayPointersMagic[8] = {'O', 'M', 'P', 'R',
                                                      'R', 'P', 'T', '1'};

/// The value of NumUnknownOutputPointers when the output was not recorded.
static constexpr uint64_t RecordReplayUnknownCount = UINT64_MAX;

/// The header of a pointers file.
struct RecordReplayPointersHeaderTy {
  /// Must be RecordReplayPointersMagic.
  char Magic[8];
  /// The number of pointer offsets that follow the header.
  uint64_t NumPointers;
  /// The number of unknown pointers in the input snapshot and the globals.
  uint64_t NumUnknownInputPointers;
  /// The number of unknown pointers in the output snapshot, or
  /// RecordReplayUnknownCount if the output was not recorded.
  uint64_t NumUnknownOutputPointers;
};

#endif // OMPTARGET_SHARED_RECORD_REPLAY_POINTERS_H
