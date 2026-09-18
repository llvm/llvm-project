//===-- Stream.h - Kernel language stream state ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STREAM_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STREAM_H

#include "OffloadAPI.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <mutex>

namespace llvm {
namespace offload {

enum class QueueKind {
  LegacyDefault,
  PerThreadDefault,
  ExplicitBlocking,
  ExplicitNonBlocking,
};

struct StreamTy {
  StreamTy(ol_queue_handle_t Queue, ol_device_handle_t Device, QueueKind Kind)
      : Queue(Queue), Device(Device), Kind(Kind) {}

  ol_result_t
  waitOnAndTrackDependencyEvents(SmallVectorImpl<ol_event_handle_t> &Events);
  ol_result_t sync();

  ol_queue_handle_t Queue = nullptr;
  ol_device_handle_t Device = nullptr;
  QueueKind Kind = QueueKind::ExplicitBlocking;

private:
  ol_result_t reclaimDependencyEventsLocked();

  static constexpr size_t MaxPendingDependencyEvents = 64;

  std::mutex DependencyEventsLock;
  SmallVector<ol_event_handle_t, 8> DependencyEvents;
};

} // namespace offload
} // namespace llvm

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STREAM_H
