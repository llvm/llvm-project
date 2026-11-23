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

namespace llvm {
namespace offload {

enum class QueueKind {
  LegacyDefault,
  PerThreadDefault,
  ExplicitBlocking,
  ExplicitNonBlocking,
};

struct StreamTy {
  ol_queue_handle_t Queue = nullptr;
  ol_device_handle_t Device = nullptr;
  QueueKind Kind = QueueKind::ExplicitBlocking;
};

} // namespace offload
} // namespace llvm

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STREAM_H
