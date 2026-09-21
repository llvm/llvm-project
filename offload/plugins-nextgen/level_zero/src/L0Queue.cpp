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
#include "L0Trace.h"
#include "PluginInterface.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
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

  bool AppendLaunchKernelWithArgsAvailable =
      Device.getL0Context().LaunchKernelWithArguments.available();

  if (AppendLaunchKernelWithArgsAvailable &&
      Device.getL0Context().AppendLaunchKernelWithArgsSupported.load(
          std::memory_order_acquire)) {
    auto Err = CmdList->appendLaunchKernelWithArgs(
        Kernel, &KEnv.GroupCounts, &KEnv.GroupSizes, KEnv.ArgPtrs, SignalEvent,
        NumWaitEvents, WaitEvents, KEnv.IsCooperative);

    if (!Err)
      return Plugin::success();

    // Check if Err is ErrorCode::UNSUPPORTED, if so consume it
    Err = llvm::handleErrors(
        std::move(Err),
        [&](std::unique_ptr<error::OffloadError> E) -> llvm::Error {
          if (E->convertToErrorCode() ==
              error::make_error_code(error::ErrorCode::UNSUPPORTED))
            return llvm::Error::success(); // Swallow error
          return llvm::Error(std::move(E));
        });

    if (Err)
      // Err is still here, so it was not ErrorCode::UNSUPPORTED
      return Err;

    // No Err - it was ErrorCode::UNSUPPORTED, continue into fallback
  }

  auto &KernelProperties = KEnv.KernelPR;
  if (KernelProperties.NumKernelArgs > 0 &&
      (KEnv.ArgPtrs == nullptr || KEnv.ArgSizes == nullptr))
    return error::createOffloadError(
        ErrorCode::INVALID_ARGUMENT,
        "zeCommandListAppendLaunchKernelWithArguments is not supported on this "
        "platform and fallback is not possible, becaue ArgPtrs(%p) or "
        "ArgSizes(%p) were not provided!",
        KEnv.ArgPtrs, KEnv.ArgSizes);

  // Submit kernel using older set of APIs - zeKernelSetArgumentValue
  auto &GroupSizes = KEnv.GroupSizes;
  CALL_ZE_RET_ERROR(zeKernelSetGroupSize, Kernel, GroupSizes.groupSizeX,
                    GroupSizes.groupSizeY, GroupSizes.groupSizeZ);

  for (uint32_t KernelArg = 0; KernelArg < KernelProperties.NumKernelArgs;
       KernelArg++) {
    uint32_t ArgSize = KEnv.ArgSizes[KernelArg];

    CALL_ZE_RET_ERROR(zeKernelSetArgumentValue, Kernel, KernelArg, ArgSize,
                      KEnv.ArgPtrs[KernelArg]);
  }

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
