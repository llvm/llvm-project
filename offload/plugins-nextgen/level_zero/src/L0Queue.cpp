//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Level Zero Queue abstraction.
//
//===----------------------------------------------------------------------===//

#include "L0Queue.h"
#include "L0Device.h"
#include "L0Kernel.h"
#include "L0Plugin.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <vector>

namespace llvm::omp::target::plugin {

/// common methods

Error L0QueueTy::init(ze_context_handle_t UserZeCtx) {
  auto CmdListOrErr = Device.getCmdListManager(UserZeCtx, CreateQueueInOrder);
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  CmdList = *CmdListOrErr;
  return initImpl();
}

Error L0QueueTy::deinit() {
  if (auto Err = deinitImpl())
    return Err;
  reset();

  if (CmdList)
    if (auto Err = Device.releaseCmdListManager(CmdList))
      return Err;

  CmdList = nullptr;
  return Plugin::success();
}

Error L0QueueTy::dispatchLaunchKernel(ze_kernel_handle_t Kernel,
                                      L0LaunchEnvTy &KEnv,
                                      ze_event_handle_t SignalEvent,
                                      uint32_t NumWaitEvents,
                                      ze_event_handle_t *WaitEvents) {
  // Unlock KEnv lock after launching the kernel.
  llvm::scope_exit UnlockGuard([&KEnv]() { KEnv.Lock.unlock(); });
  return CmdList->appendLaunchKernelWithArgs(
      Kernel, &KEnv.GroupCounts, &KEnv.GroupSizes, KEnv.ArgPtrs, SignalEvent,
      NumWaitEvents, WaitEvents, KEnv.IsCooperative);
}

Error L0QueueTy::memoryFill(void *Ptr, const void *Pattern, size_t PatternSize,
                            size_t Size) {
  assert(PatternSize <= Size && "PatternSize > Size is unsupported");

  if (Size == 0 || PatternSize == 0)
    return Plugin::success();

  if (llvm::isPowerOf2_64(PatternSize) && (Size % PatternSize == 0) &&
      PatternSize <= Device.getMaxMemFillPatternSize()) {
    // Native L0 memory fill is possible directly.
    return memoryFillImpl(Ptr, Pattern, PatternSize, Size);
  }

  auto *PatternBytes = static_cast<const unsigned char *>(Pattern);
  // Check if all bytes are equal.
  if (std::memcmp(PatternBytes, PatternBytes + 1, PatternSize - 1) == 0) {
    // Substitution of 1 as PatternSize is equivalent,
    // so native L0 memory fill is still possible.
    return memoryFillImpl(Ptr, Pattern, 1, Size);
  }

  // TODO: if we insist on plugins supporting arbitrary pattern sizes, extra
  // detection of repeating power-of-two patterns could be added here to allow
  // native L0 memory fill for those cases as well.

  return memoryFillReplicateImpl(Ptr, Pattern, PatternSize, Size);
}

/// Construct a seed by repeating \p Pattern. When \p PatternSize is at most
/// \p MinSize, the seed size is a multiple of \p PatternSize in the range
/// [MinSize, 2 * MinSize). Otherwise, return a copy of \p Pattern.
static std::vector<unsigned char>
extendPattern(const void *Pattern, size_t PatternSize, size_t MinSize) {
  assert(PatternSize > 0 && MinSize > 0 && "Invalid pattern extension size");
  const auto *PatternBytes = static_cast<const unsigned char *>(Pattern);
  if (PatternSize > MinSize)
    return std::vector<unsigned char>(PatternBytes, PatternBytes + PatternSize);

  const size_t NumPatterns = (MinSize + PatternSize - 1) / PatternSize;
  std::vector<unsigned char> Seed(NumPatterns * PatternSize);
  std::copy_n(PatternBytes, PatternSize, Seed.begin());
  for (size_t Offset = PatternSize; Offset < Seed.size(); ++Offset)
    Seed[Offset] = Seed[Offset - PatternSize];
  return Seed;
}

Error L0QueueTy::memoryFillReplicateImpl(void *Ptr, const void *Pattern,
                                         size_t PatternSize, size_t Size) {
  auto *Dst = static_cast<unsigned char *>(Ptr);

  // Extend small patterns to avoid several inefficient device copies.
  const auto Seed = extendPattern(Pattern, PatternSize, /*MinSize=*/1024);
  size_t BytesFilled = std::min(Seed.size(), Size);

  const auto TgtType = Device.getMemAllocType(Ptr);
  // dataSubmit() writes host/shared destinations directly, so complete earlier
  // queue work before modifying the destination from the host.
  if (TgtType == ZE_MEMORY_TYPE_HOST || TgtType == ZE_MEMORY_TYPE_SHARED) {
    if (auto Err = synchronize())
      return Err;
  }

  if (auto Err = dataSubmit(Dst, Seed.data(), BytesFilled))
    return Err;

  // Complete the seed submission before its host storage goes out of scope.
  if (auto Err = synchronize())
    return Err;

  // Clone the seed, doubling each time, until it fills the entire destination.
  while (BytesFilled < Size) {
    if (auto Err = dataFence())
      return Err;
    const size_t CopyChunkSize = std::min(BytesFilled, Size - BytesFilled);
    if (auto Err = memoryCopy(Dst + BytesFilled, Dst, CopyChunkSize))
      return Err;
    BytesFilled += CopyChunkSize;
  }
  return Plugin::success();
}

// L0AsyncQueueTy implementation.

Error L0AsyncQueueTy::deinitImpl() {
  Error AllErrors = Plugin::success();
  for (auto &Event : WaitEvents) {
    if (auto Err = Device.releaseEvent(Event))
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
  }
  WaitEvents.clear();
  return AllErrors;
}

void L0AsyncQueueTy::resetImpl() {
  WaitEvents.clear();
  KernelEvent = nullptr;
  H2MList.clear();
  USM2MList.clear();
}

void L0AsyncQueueTy::processCopyQueues() {
  auto ProcessQueue = [](auto &Queue) {
    for (auto &[Src, Dst, Size] : Queue)
      std::copy_n(static_cast<const char *>(Src), Size,
                  static_cast<char *>(Dst));
    Queue.clear();
  };

  ProcessQueue(USM2MList);
  ProcessQueue(H2MList);
}

Error L0AsyncQueueTy::synchronizeImpl() {
  Error SyncErrors = Plugin::success();

  // Wait for all events. We should wait and reset events in reverse order
  // to avoid premature event reset. If we have a kernel event in the
  // queue, it is the last event to wait for since all wait events of the
  // kernel are signaled before the kernel is invoked. We always invoke
  // synchronization on kernel event to support printf().
  bool WaitDone = false;
  for (auto Itr = WaitEvents.rbegin(); Itr != WaitEvents.rend(); Itr++) {
    if (!WaitDone) {
      SyncErrors = joinErrors(std::move(SyncErrors),
                              CmdList->eventHostSynchronize(*Itr));
      if (*Itr == KernelEvent)
        WaitDone = true;
    }
    if (auto Err = Device.releaseEvent(*Itr))
      SyncErrors = joinErrors(std::move(SyncErrors), std::move(Err));
  }
  WaitEvents.clear();
  KernelEvent = nullptr;

  processCopyQueues();

  return SyncErrors;
}

Expected<bool> L0AsyncQueueTy::hasPendingWorkImpl() {
  if (!WaitEvents.empty())
    return true;

  processCopyQueues();
  return false;
}

std::tuple<size_t, ze_event_handle_t *> L0AsyncQueueTy::getMemCopyEvents() {
  return KernelEvent ? std::make_tuple(1, &KernelEvent)
                     : std::make_tuple(0, nullptr);
}

std::tuple<size_t, ze_event_handle_t *>
L0AsyncQueueTy::getLaunchKernelEvents() {
  return WaitEvents.empty()
             ? std::make_tuple(0, nullptr)
             : std::make_tuple(WaitEvents.size(), WaitEvents.data());
}

Error L0AsyncQueueTy::memoryCopyImpl(void *Dst, const void *Src, size_t Size) {
  auto EventOrErr = Device.getEvent();
  if (!EventOrErr)
    return EventOrErr.takeError();
  ze_event_handle_t SignalEvent = *EventOrErr;
  auto [NumWaitEvents, WaitEventsPtr] = getMemCopyEvents();

  Error AllErrors = CmdList->appendMemoryCopy(Dst, Src, Size, SignalEvent,
                                              NumWaitEvents, WaitEventsPtr);
  if (!AllErrors) {
    WaitEvents.push_back(SignalEvent);
  } else {
    if (auto Err = Device.releaseEvent(SignalEvent))
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
  }
  return AllErrors;
}

Error L0AsyncQueueTy::dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                       int64_t Size) {
  auto TgtPtrType = Device.getMemAllocType(TgtPtr);
  if (TgtPtrType == ZE_MEMORY_TYPE_HOST ||
      TgtPtrType == ZE_MEMORY_TYPE_SHARED) {
    bool CopyNow = true;
    if (KernelEvent) {
      // Delay Host/Shared USM to host memory copy since it must wait for
      // kernel completion.
      USM2MList.emplace_back(
          PendingCopyDescTy{TgtPtr, HstPtr, static_cast<size_t>(Size)});
      CopyNow = false;
    }
    if (CopyNow) {
      std::copy_n(static_cast<const char *>(TgtPtr), Size,
                  static_cast<char *>(HstPtr));
    }
    return Plugin::success();
  }

  void *DstPtr = HstPtr;
  if (Device.isDiscreteDevice() &&
      static_cast<size_t>(Size) <=
          Device.getPlugin().getOptions().StagingBufferSize &&
      Device.getMemAllocType(HstPtr) != ZE_MEMORY_TYPE_HOST) {
    auto PtrOrErr = Device.getStagingBuffer().get(/*IsAsync*/ true);
    if (!PtrOrErr)
      return PtrOrErr.takeError();
    DstPtr = *PtrOrErr;
  }

  if (auto Err = memoryCopy(DstPtr, TgtPtr, Size))
    return Err;

  if (DstPtr != HstPtr)
    H2MList.emplace_back(
        PendingCopyDescTy{DstPtr, HstPtr, static_cast<size_t>(Size)});
  return Plugin::success();
}

Error L0AsyncQueueTy::dataSubmitImpl(void *TgtPtr, const void *HstPtr,
                                     int64_t Size) {
  const auto TgtPtrType = Device.getMemAllocType(TgtPtr);
  if (TgtPtrType == ZE_MEMORY_TYPE_SHARED ||
      TgtPtrType == ZE_MEMORY_TYPE_HOST) {
    std::copy_n(static_cast<const char *>(HstPtr), Size,
                static_cast<char *>(TgtPtr));
    return Plugin::success();
  }

  const void *SrcPtr = HstPtr;

  if (Device.isDiscreteDevice() &&
      static_cast<size_t>(Size) <=
          Device.getPlugin().getOptions().StagingBufferSize &&
      Device.getMemAllocType(HstPtr) != ZE_MEMORY_TYPE_HOST) {
    auto PtrOrErr = Device.getStagingBuffer().get(/*IsAsync*/ true);
    if (!PtrOrErr)
      return PtrOrErr.takeError();
    SrcPtr = *PtrOrErr;
    std::copy_n(static_cast<const char *>(HstPtr), Size,
                static_cast<char *>(const_cast<void *>(SrcPtr)));
  }

  return memoryCopy(TgtPtr, SrcPtr, Size);
}

Error L0AsyncQueueTy::dataFenceImpl() {
  return CmdList->appendBarrier(/*SignalEvent*/ nullptr, /*NumWaitEvents*/ 0,
                                /*WaitEvents*/ nullptr);
}

Error L0AsyncQueueTy::launchKernelImpl(ze_kernel_handle_t Kernel,
                                       L0LaunchEnvTy &KEnv) {
  auto EventOrError = Device.getEvent();
  if (!EventOrError)
    return EventOrError.takeError();
  ze_event_handle_t SignalEvent = *EventOrError;
  auto [NumWaitEvents, WaitEventsPtr] = getLaunchKernelEvents();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device.getDeviceId(),
       "Kernel depends on %zu data copying events.\n", NumWaitEvents);
  Error AllErrors = dispatchLaunchKernel(Kernel, KEnv, SignalEvent,
                                         NumWaitEvents, WaitEventsPtr);
  if (AllErrors) {
    if (auto Err = Device.releaseEvent(SignalEvent))
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
    return AllErrors;
  }
  WaitEvents.push_back(SignalEvent);
  KernelEvent = SignalEvent;
  return Plugin::success();
}

Error L0AsyncQueueTy::hostCallImpl(void (*Callback)(void *), void *UserData) {
  return Plugin::error(ErrorCode::UNIMPLEMENTED,
                       "Host function callbacks are not yet implemented for "
                       "out-of-order async queues");
}

Error L0AsyncQueueTy::memoryFillImpl(void *Ptr, const void *Pattern,
                                     size_t PatternSize, size_t Size) {
  auto EventOrErr = Device.getEvent();
  if (!EventOrErr)
    return EventOrErr.takeError();
  auto [NumWaitEvents, WaitEventsPtr] = getMemCopyEvents();
  ze_event_handle_t SignalEvent = *EventOrErr;
  if (auto Err = CmdList->appendMemoryFill(Ptr, Pattern, PatternSize, Size,
                                           SignalEvent, NumWaitEvents,
                                           WaitEventsPtr)) {
    if (auto ReleaseErr = Device.releaseEvent(SignalEvent))
      return joinErrors(std::move(Err), std::move(ReleaseErr));
    return Err;
  }
  WaitEvents.push_back(SignalEvent);
  return Plugin::success();
}

// L0AsyncOrderedQueue implementation.
Error L0AsyncOrderedQueueTy::synchronizeImpl() {
  Error SyncErrors = Plugin::success();

  ze_event_handle_t LastEvent =
      WaitEvents.empty() ? nullptr : WaitEvents.back();
  // Only need to wait for the last event.
  if (LastEvent) {
    SyncErrors = joinErrors(std::move(SyncErrors),
                            CmdList->eventHostSynchronize(LastEvent));
  }
  // Synchronize on kernel event to support printf().
  ze_event_handle_t KE = KernelEvent;
  if (KE && KE != LastEvent && !SyncErrors) {
    SyncErrors =
        joinErrors(std::move(SyncErrors), CmdList->eventHostSynchronize(KE));
  }
  for (auto &Event : WaitEvents) {
    if (auto Err = Device.releaseEvent(Event))
      SyncErrors = joinErrors(std::move(SyncErrors), std::move(Err));
  }

  processCopyQueues();
  WaitEvents.clear();
  KernelEvent = nullptr;

  return SyncErrors;
}

std::tuple<size_t, ze_event_handle_t *>
L0AsyncOrderedQueueTy::getMemCopyEvents() {
  return WaitEvents.empty() ? std::make_tuple(0, nullptr)
                            : std::make_tuple(1, &WaitEvents.back());
}

std::tuple<size_t, ze_event_handle_t *>
L0AsyncOrderedQueueTy::getLaunchKernelEvents() {
  return WaitEvents.empty() ? std::make_tuple(0, nullptr)
                            : std::make_tuple(1, &WaitEvents.back());
}

Error L0AsyncOrderedQueueTy::hostCallImpl(void (*Callback)(void *),
                                          void *UserData) {
  return Plugin::error(ErrorCode::UNIMPLEMENTED,
                       "Host function callbacks are not yet implemented for "
                       "ordered async queues");
}

// L0InorderQueueTy implementation.
Error L0InorderQueueTy::synchronizeImpl() { return CmdList->hostSynchronize(); }

Expected<bool> L0InorderQueueTy::hasPendingWorkImpl() {
  return CmdList->queryPendingWork();
}

Error L0InorderQueueTy::memoryCopyImpl(void *Dst, const void *Src,
                                       size_t Size) {
  return CmdList->appendMemoryCopy(Dst, Src, Size);
}

Error L0InorderQueueTy::launchKernelImpl(ze_kernel_handle_t Kernel,
                                         L0LaunchEnvTy &KEnv) {
  return dispatchLaunchKernel(Kernel, KEnv);
}

Error L0InorderQueueTy::hostCallImpl(void (*Callback)(void *), void *UserData) {
  return CmdList->appendHostFunction(Callback, UserData);
}

// L0SyncQueueTy implementation.
Error L0SyncQueueTy::memoryCopyImpl(void *Dst, const void *Src, size_t Size) {
  if (auto Err = L0InorderQueueTy::memoryCopyImpl(Dst, Src, Size))
    return Err;
  return CmdList->hostSynchronize();
}

Error L0SyncQueueTy::launchKernelImpl(ze_kernel_handle_t Kernel,
                                      L0LaunchEnvTy &KEnv) {
  if (auto Err = L0InorderQueueTy::launchKernelImpl(Kernel, KEnv))
    return Err;
  return CmdList->hostSynchronize();
}

Error L0SyncQueueTy::hostCallImpl(void (*Callback)(void *), void *UserData) {
  if (auto Err = L0InorderQueueTy::hostCallImpl(Callback, UserData))
    return Err;
  return CmdList->hostSynchronize();
}

Error L0SyncQueueTy::memoryFillImpl(void *Ptr, const void *Pattern,
                                    size_t PatternSize, size_t Size) {
  if (auto Err =
          L0InorderQueueTy::memoryFillImpl(Ptr, Pattern, PatternSize, Size))
    return Err;
  return CmdList->hostSynchronize();
}

// L0QueueCache implementation.
Expected<L0QueueTy *> L0QueueCacheTy::getQueue(L0DeviceTy &Device) {
  {
    std::lock_guard<std::mutex> Lock(Mtx);
    auto Itr = Queues.find(&Device);
    if (Itr != Queues.end() && !Itr->second.empty()) {
      L0QueueTy *Queue = Itr->second.back();
      Itr->second.pop_back();
      return Queue;
    }
  }
  L0QueueTy *Queue = nullptr;
  switch (Device.getPlugin().getOptions().CommandMode) {
  case CommandModeTy::Async:
    Queue = new L0AsyncQueueTy(Device);
    break;
  case CommandModeTy::AsyncOrdered:
    Queue = new L0AsyncOrderedQueueTy(Device);
    break;
  case CommandModeTy::Sync:
    Queue = new L0SyncQueueTy(Device);
    break;
  case CommandModeTy::InOrder:
    Queue = new L0InorderQueueTy(Device);
    break;
  }
  Queue->setUserCtx(&UserCtx);
  if (auto Err = Queue->init(UserCtx.getZeContext())) {
    delete Queue;
    return std::move(Err);
  }
  return Queue;
}

void L0QueueCacheTy::releaseQueue(L0QueueTy *Queue) {
  if (!Queue)
    return;
  L0DeviceTy &Device = Queue->getDevice();
  Queue->reset();
  std::lock_guard<std::mutex> Lock(Mtx);
  Queues[&Device].push_back(Queue);
}

Error L0QueueCacheTy::deinit() {
  Error AllErrors = Error::success();
  std::lock_guard<std::mutex> Lock(Mtx);
  for (auto &Bucket : Queues) {
    for (auto *Queue : Bucket.second) {
      if (auto Err = Queue->deinit())
        AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
      delete Queue;
    }
  }
  Queues.clear();
  return AllErrors;
}

} // namespace llvm::omp::target::plugin
