//===-- LanguageUtils.h - Kernel Language utility functions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H

#include "LanguageRuntime.h"
#include "OffloadAPI.h"
#include "State.h"
#include "Stream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using RuntimeState = llvm::offload::StateTy;
using ThreadState = llvm::offload::ThreadStateTy;
using StreamTy = llvm::offload::StreamTy;

/// Convert an ol_result_t to the active language's Error_t.
static inline Error_t convertResult(ol_result_t Result) {
  if (Result == OL_SUCCESS)
    return Success;
  switch (Result->Code) {
  case OL_ERRC_INVALID_VALUE:
  case OL_ERRC_INVALID_ARGUMENT:
  case OL_ERRC_INVALID_NULL_POINTER:
    return ErrorInvalidValue;
  case OL_ERRC_INVALID_SIZE:
    return ErrorInvalidConfiguration;
  case OL_ERRC_INVALID_NULL_HANDLE:
  case OL_ERRC_INVALID_QUEUE:
  case OL_ERRC_INVALID_EVENT:
  case OL_ERRC_INVALID_CONTEXT:
    return ErrorInvalidResourceHandle;
  case OL_ERRC_INVALID_DEVICE:
    return ErrorInvalidDevice;
  default:
    return ErrorUnknown;
  }
}

/// Set the last error for the current thread and return it.
static inline Error_t setLastError(Error_t Error) {
  // TODO: find a more efficient way to set last error
  return static_cast<Error_t>(ThreadState::setLastError(Error));
}

/// Convert an ol_result_t to the active language's Error_t and set it as the
/// last error for the current thread.
static inline Error_t convertAndSetLastError(ol_result_t Result) {
  return setLastError(convertResult(Result));
}

/// Convert between the language-facing opaque stream and the internal stream.
static inline Stream_t makeLanguageStream(StreamTy *Stream) {
  return reinterpret_cast<Stream_t>(Stream);
}

static inline llvm::offload::StreamTy *getInternalStream(Stream_t Stream) {
  return reinterpret_cast<StreamTy *>(Stream);
}

static inline ol_result_t
syncAndDestroyEvents(llvm::SmallVectorImpl<ol_event_handle_t> &Events) {
  ol_result_t FirstError = OL_SUCCESS;
  for (ol_event_handle_t Event : Events) {
    if (!Event)
      continue;

    ol_result_t SyncResult = olSyncEvent(Event);
    if (FirstError == OL_SUCCESS && SyncResult != OL_SUCCESS)
      FirstError = SyncResult;

    ol_result_t DestroyResult = olDestroyEvent(Event);
    if (FirstError == OL_SUCCESS && DestroyResult != OL_SUCCESS)
      FirstError = DestroyResult;
  }
  Events.clear();
  return FirstError;
}

/// Wait for blocking streams before executing if we are legacy default stream.
static inline ol_result_t waitOnBlockingStreams() {
  ol_device_handle_t Device = ThreadState::getDefaultDevice();
  llvm::SmallPtrSet<StreamTy *, 8> BlockingStreams =
      RuntimeState::getBlockingStreams(Device);
  if (!RuntimeState::hasLegacyDefaultStream(Device) || BlockingStreams.empty())
    return OL_SUCCESS;

  StreamTy *DefaultStream = ThreadState::getDefaultStream();
  llvm::SmallVector<ol_event_handle_t, 8> Events;
  for (StreamTy *BlockingStream : BlockingStreams) {
    ol_event_handle_t Event = nullptr;
    ol_result_t Result =
        olCreateEvent(BlockingStream->Queue, OL_EVENT_FLAGS_NONE, &Event);
    if (Result != OL_SUCCESS) {
      if (Event)
        Events.push_back(Event);
      syncAndDestroyEvents(Events);
      return Result;
    }
    Events.push_back(Event);
  }

  return DefaultStream->waitOnAndTrackDependencyEvents(Events);
}

/// Wait for the legacy default stream to complete before launching a kernel on
/// a blocking stream.
static inline ol_result_t waitOnLegacyDefaultStream(StreamTy *SourceStream,
                                                    ol_device_handle_t Device) {
  if (!RuntimeState::hasLegacyDefaultStream(Device))
    return OL_SUCCESS;

  StreamTy *DefaultStream = ThreadState::getDefaultStream();
  assert(DefaultStream->Kind == llvm::offload::QueueKind::LegacyDefault &&
         "Default stream is not a legacy default stream");

  ol_event_handle_t Event = nullptr;
  ol_result_t Result =
      olCreateEvent(DefaultStream->Queue, OL_EVENT_FLAGS_NONE, &Event);
  if (Result != OL_SUCCESS) {
    if (Event) {
      llvm::SmallVector<ol_event_handle_t, 1> Events = {Event};
      syncAndDestroyEvents(Events);
    }
    return Result;
  }
  return SourceStream->waitOnAndTrackDependencyEvents(
      llvm::ArrayRef<ol_event_handle_t>(&Event, 1));
}

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H
