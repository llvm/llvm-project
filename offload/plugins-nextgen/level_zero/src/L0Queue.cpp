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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <vector>

namespace llvm::omp::target::plugin {

/// common methods

Error L0QueueTy::init() {
  auto CmdListOrErr = Device.getCmdListManager(CreateQueueInOrder);
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
  if (KEnv.IsPtrArg)
    return CmdList->appendLaunchKernelWithArgs(
        Kernel, &KEnv.GroupCounts, &KEnv.GroupSizes, KEnv.ArgPtrs, SignalEvent,
        NumWaitEvents, WaitEvents, KEnv.IsCooperative);

  return CmdList->appendLaunchKernel(Kernel, &KEnv.GroupCounts, SignalEvent,
                                     NumWaitEvents, WaitEvents,
                                     KEnv.IsCooperative);
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

  auto PatternBytes = static_cast<const unsigned char *>(Pattern);
  // Check if all bytes are equal.
  if (std::memcmp(PatternBytes, PatternBytes + 1, PatternSize - 1) == 0) {
    // Substitution of 1 as PatternSize is equivalent,
    // so native L0 memory fill is still possible.
    return memoryFillImpl(Ptr, Pattern, 1, Size);
  }

  // TODO: if we insist on plugins supporting arbitrary pattern sizes, extra
  // detection of repeating power-of-two patterns could be added here to allow
  // native L0 memory fill for those cases as well.

  // Native L0 fill cannot handle this pattern size, but target memory is
  // host-accessible, so fall back to a software fill.
  const auto TgtType = Device.getMemAllocType(Ptr);
  if (TgtType == ZE_MEMORY_TYPE_HOST || TgtType == ZE_MEMORY_TYPE_SHARED)
    return memoryFillHostImpl(Ptr, Pattern, PatternSize, Size);

  // We know at this point that TgtType == ZE_MEMORY_TYPE_DEVICE.
  // Native fill and software fill are both impossible.
  // Seed the pattern once and grow the filled region with device copies,
  // doubling the amount copied each time.
  return memoryFillReplicateImpl(Ptr, Pattern, PatternSize, Size);
}

Error L0QueueTy::memoryFillHostImpl(void *Ptr, const void *Pattern,
                                    size_t PatternSize, size_t Size) {
  auto *Dst = static_cast<unsigned char *>(Ptr);
  const auto *Pat = static_cast<const unsigned char *>(Pattern);
  // Seed the pattern once.
  std::copy_n(Pat, PatternSize, Dst);
  // Replicate the pattern until it fills the entire destination.
  for (size_t Offset = PatternSize; Offset < Size; ++Offset) {
    Dst[Offset] = Dst[Offset - PatternSize];
  }
  return Plugin::success();
}

/// Replicate the pattern in \p Buf (of \p Size bytes) on the host until it is
/// at least \p MinExtendedSize bytes long. The result is
/// never larger than max(Size, 2 * MinExtendedSize).
static std::vector<unsigned char> extendPattern(unsigned char *Buf, size_t Size,
                                                size_t MinExtendedSize) {
  assert(Size > 0 && MinExtendedSize > 0 &&
         "Invalid pattern size or extension size");
  const size_t NumPatterns =
      std::max(static_cast<size_t>(1), (MinExtendedSize + Size - 1) / Size);
  std::vector<unsigned char> Extended(NumPatterns * Size);
  // Seed the pattern.
  std::copy_n(Buf, Size, Extended.begin());
  // Replicate the pattern until we reach the desired size.
  for (size_t Offset = Size; Offset < Extended.size(); ++Offset) {
    Extended[Offset] = Extended[Offset - Size];
  }
  return Extended;
}

Error L0QueueTy::memoryFillReplicateImpl(void *Ptr, const void *Pattern,
                                         size_t PatternSize, size_t Size) {
  auto *Dst = static_cast<unsigned char *>(Ptr);

  // Grow the pattern on the host first - avoids several inefficient small
  // device copies.
  constexpr size_t MinExtendedSeedSize = 1024;
  const auto ExtendedPattern =
      extendPattern(static_cast<unsigned char *>(const_cast<void *>(Pattern)),
                    PatternSize, std::min(Size, MinExtendedSeedSize));

  // Seed the (extended) pattern once using dataSubmit.
  size_t BytesFilled = std::min(ExtendedPattern.size(), Size);
  if (auto Err = dataSubmit(Dst, ExtendedPattern.data(), BytesFilled))
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
  auto processQueue = [](auto &Queue) {
    for (auto &[Src, Dst, Size] : Queue)
      std::copy_n(static_cast<const char *>(Src), Size,
                  static_cast<char *>(Dst));
    Queue.clear();
  };

  processQueue(USM2MList);
  processQueue(H2MList);
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

// L0QueueCache implementation.
Expected<L0QueueTy *> L0QueueCacheTy::getQueue() {
  {
    std::lock_guard<std::mutex> Lock(Mtx);
    if (!Queues.empty()) {
      L0QueueTy *Queue = Queues.back();
      Queues.pop_back();
      return Queue;
    }
  }
  L0QueueTy *Queue = nullptr;
  switch (CachedCmdMode) {
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
  if (auto Err = Queue->init()) {
    delete Queue;
    return std::move(Err);
  }
  return Queue;
}

void L0QueueCacheTy::releaseQueue(L0QueueTy *Queue) {
  if (!Queue)
    return;
  Queue->reset();
  std::lock_guard<std::mutex> Lock(Mtx);
  Queues.push_back(Queue);
}

Error L0QueueCacheTy::deinit() {
  Error AllErrors = Error::success();
  std::lock_guard<std::mutex> Lock(Mtx);
  for (auto *Queue : Queues) {
    if (auto Err = Queue->deinit())
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
    delete Queue;
  }
  Queues.clear();
  return AllErrors;
}

} // namespace llvm::omp::target::plugin
