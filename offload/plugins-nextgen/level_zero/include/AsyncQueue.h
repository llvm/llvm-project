//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Async Queue wrapper for Level Zero.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H

#include "L0Defs.h"
#include "L0Trace.h"
#include "PluginInterface.h"

#include <mutex>
#include <tuple>

#include "L0CmdListManager.h"
#include "L0Options.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;
struct L0LaunchEnvTy;

/// Abstract queue that supports asynchronous command submission.
struct AsyncQueueTy {
  /// Device owning this queue.
  L0DeviceTy &Device;
  /// Underlying immediate command list.
  L0CmdListManagerTy *CmdList = nullptr;
  /// Whether the queue is in-order or out-of-order.
  bool IsInorder;

  AsyncQueueTy(L0DeviceTy &Device, bool IsInorder = true)
      : Device(Device), IsInorder(IsInorder) {}
  virtual ~AsyncQueueTy() {}

  /// Clear data.
  void reset() { resetImpl(); }

  ze_command_list_handle_t getCmdList() const { return CmdList->getCmdList(); }

  Error init();
  Error deinit();
  Error synchronize() { return synchronizeImpl(); }
  Expected<bool> hasPendingWork() { return hasPendingWorkImpl(); }

  Error memoryCopy(void *Dst, const void *Src, size_t Size) {
    if (Size == 0)
      return Plugin::success();
    if (Dst == Src)
      return Plugin::success();

    return memoryCopyImpl(Dst, Src, Size);
  }

  Error dataRetrieve(void *HstPtr, const void *TgtPtr, int64_t Size) {
    return dataRetrieveImpl(HstPtr, TgtPtr, Size);
  }

  Error dataSubmit(void *TgtPtr, const void *HstPtr, int64_t Size) {
    return dataSubmitImpl(TgtPtr, HstPtr, Size);
  }

  Error memoryFill(void *Ptr, const void *Pattern, size_t PatternSize,
                   size_t Size) {
    return memoryFillImpl(Ptr, Pattern, PatternSize, Size);
  }

  Error dispatchLaunchKernel(ze_kernel_handle_t Kernel, L0LaunchEnvTy &KEnv,
                             ze_event_handle_t SignalEvent = nullptr,
                             uint32_t NumWaitEvents = 0,
                             ze_event_handle_t *WaitEvents = nullptr);
  Error launchKernel(ze_kernel_handle_t Kernel, L0LaunchEnvTy &KEnv) {
    return launchKernelImpl(Kernel, KEnv);
  }

  virtual Error initImpl() { return Plugin::success(); }
  virtual Error deinitImpl() { return Plugin::success(); }
  virtual void resetImpl() {}

  virtual Error synchronizeImpl() = 0;
  virtual Expected<bool> hasPendingWorkImpl() = 0;
  virtual Error memoryCopyImpl(void *Dst, const void *Src, size_t Size) = 0;
  virtual Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                 int64_t Size) {
    return memoryCopy(HstPtr, TgtPtr, Size);
  }
  virtual Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size) {
    return memoryCopy(TgtPtr, HstPtr, Size);
  }
  virtual Error launchKernelImpl(ze_kernel_handle_t Kernel,
                                 L0LaunchEnvTy &KEnv) = 0;

  virtual Error memoryFillImpl(void *Ptr, const void *Pattern,
                               size_t PatternSize, size_t Size) {
    return CmdList->appendMemoryFill(Ptr, Pattern, PatternSize, Size);
  }
};

class L0AsyncQueueTy : public AsyncQueueTy {
protected:
  /// List of events attached to submitted commands.
  llvm::SmallVector<ze_event_handle_t> WaitEvents;
  /// Kernel event not signaled.
  ze_event_handle_t KernelEvent = nullptr;
  /// Pending staging buffer to host copies.
  llvm::SmallVector<std::tuple<void *, void *, size_t>> H2MList;
  /// Pending USM memory copy commands that must wait for kernel completion.
  llvm::SmallVector<std::tuple<const void *, void *, size_t>> USM2MList;

  virtual std::tuple<size_t, ze_event_handle_t *> getMemCopyEvents();
  virtual std::tuple<size_t, ze_event_handle_t *> getLaunchKernelEvents();
  void processCopyQueues();

public:
  L0AsyncQueueTy(L0DeviceTy &Device)
      : AsyncQueueTy(Device, /*IsInorder*/ false) {}
  virtual ~L0AsyncQueueTy() {}

  L0AsyncQueueTy(const L0AsyncQueueTy &) = delete;
  L0AsyncQueueTy(const L0AsyncQueueTy &&) = delete;
  L0AsyncQueueTy &operator=(const L0AsyncQueueTy &) = delete;
  L0AsyncQueueTy &operator=(const L0AsyncQueueTy &&) = delete;

  Error deinitImpl() override;
  void resetImpl() override;
  Error synchronizeImpl() override;
  Expected<bool> hasPendingWorkImpl() override;
  Error memoryCopyImpl(void *Dst, const void *Src, size_t Size) override;
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                         int64_t Size) override;
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size) override;
  Error launchKernelImpl(ze_kernel_handle_t Kernel,
                         L0LaunchEnvTy &KEnv) override;
  Error memoryFillImpl(void *Ptr, const void *Pattern, size_t PatternSize,
                       size_t Size) override;
};

class L0AsyncOrderedQueueTy : public L0AsyncQueueTy {
public:
  L0AsyncOrderedQueueTy(L0DeviceTy &Device) : L0AsyncQueueTy(Device) {}
  virtual ~L0AsyncOrderedQueueTy() {}

  L0AsyncOrderedQueueTy(const L0AsyncOrderedQueueTy &) = delete;
  L0AsyncOrderedQueueTy(const L0AsyncOrderedQueueTy &&) = delete;
  L0AsyncOrderedQueueTy &operator=(const L0AsyncOrderedQueueTy &) = delete;
  L0AsyncOrderedQueueTy &operator=(const L0AsyncOrderedQueueTy &&) = delete;

  Error synchronizeImpl() override;
  std::tuple<size_t, ze_event_handle_t *> getMemCopyEvents() override;
  std::tuple<size_t, ze_event_handle_t *> getLaunchKernelEvents() override;
};

class L0InorderQueueTy : public AsyncQueueTy {
public:
  L0InorderQueueTy(L0DeviceTy &Device) : AsyncQueueTy(Device) {}
  virtual ~L0InorderQueueTy() {}

  L0InorderQueueTy(const L0InorderQueueTy &) = delete;
  L0InorderQueueTy(const L0InorderQueueTy &&) = delete;
  L0InorderQueueTy &operator=(const L0InorderQueueTy &) = delete;
  L0InorderQueueTy &operator=(const L0InorderQueueTy &&) = delete;

  Error synchronizeImpl() override;
  Expected<bool> hasPendingWorkImpl() override;
  Error memoryCopyImpl(void *Dst, const void *Src, size_t Size) override;
  Error launchKernelImpl(ze_kernel_handle_t Kernel,
                         L0LaunchEnvTy &KEnv) override;
};

class L0SyncQueueTy : public L0InorderQueueTy {
public:
  L0SyncQueueTy(L0DeviceTy &Device) : L0InorderQueueTy(Device) {}
  virtual ~L0SyncQueueTy() {}

  L0SyncQueueTy(const L0SyncQueueTy &) = delete;
  L0SyncQueueTy(const L0SyncQueueTy &&) = delete;
  L0SyncQueueTy &operator=(const L0SyncQueueTy &) = delete;
  L0SyncQueueTy &operator=(const L0SyncQueueTy &&) = delete;

  Error synchronizeImpl() override { return Plugin::success(); }
  Expected<bool> hasPendingWorkImpl() override { return false; }
  Error memoryCopyImpl(void *Dst, const void *Src, size_t Size) override;
  Error launchKernelImpl(ze_kernel_handle_t Kernel,
                         L0LaunchEnvTy &KEnv) override;
};

/// Simple cache for queue objects.
class L0QueueCacheTy {
  L0DeviceTy &Device;
  llvm::SmallVector<AsyncQueueTy *> Queues;
  std::mutex Mtx;
  CommandModeTy CachedCmdMode = CommandModeTy::InOrder;

public:
  L0QueueCacheTy(L0DeviceTy &Device) : Device(Device) {}
  Expected<AsyncQueueTy *> getQueue();
  void releaseQueue(AsyncQueueTy *Queue);
  Error deinit();
  void setCommandMode(CommandModeTy CmdMode) { CachedCmdMode = CmdMode; }
};

} // namespace llvm::omp::target::plugin
#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
