//===--- SyncClassification.h - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm::advisor {

/// Return true if the event name indicates a synchronization operation.
inline bool isSyncEvent(StringRef Name) {
  return Name.contains("Synchronize") || Name.contains("synchronize") ||
         Name.contains("Barrier") || Name.contains("barrier") ||
         Name.contains("hsa_signal_wait") || Name.contains("hsa_queue_flush") ||
         Name.contains("__syncthreads");
}

/// Classify a synchronization event name into a canonical kind.
inline StringRef classifySyncKind(StringRef Name) {
  if (Name.contains("DeviceSynchronize"))
    return "device_sync";
  if (Name.contains("StreamSynchronize"))
    return "stream_sync";
  if (Name.contains("EventSynchronize"))
    return "event_sync";
  if (Name.contains("ThreadSynchronize"))
    return "thread_sync";
  if (Name.contains("hsa_signal_wait"))
    return "signal_wait";
  if (Name.contains("hsa_queue_flush"))
    return "queue_flush";
  if (Name.contains("Barrier"))
    return "barrier";
  if (Name.contains("Synchronize") || Name.contains("synchronize"))
    return "generic_sync";
  return "other";
}

} // namespace llvm::advisor
