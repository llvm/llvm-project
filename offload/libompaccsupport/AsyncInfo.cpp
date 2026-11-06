//===-- AsyncInfo.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"

#include "Shared/Debug.h"
#include "device.h"

#include <cassert>
#include <cstdint>

using namespace llvm::omp::target::debug;
using llvm::SmallVector;

AsyncInfoTy::AsyncInfoTy(DeviceTy &Device, SyncTy SyncType)
    : Device(Device), SyncType(SyncType) {
  if (auto Res = olCreateQueue(Device.Context, Device.DeviceHandle, &Queue)) {
    REPORT() << "Failed to create queue for device " << Device.DeviceID << ": "
             << Res->Details;
    Queue = nullptr;
  }
}

AsyncInfoTy::~AsyncInfoTy() {
  finalize();
  if (Queue) {
    if (auto Res = olDestroyQueue(Queue))
      REPORT() << "Failed to destroy queue " << Queue << ": " << Res->Details;
  }
}

int AsyncInfoTy::finalize() {
  int Result = OFFLOAD_SUCCESS;
  if (!isQueueEmpty()) {
    switch (SyncType) {
    case SyncTy::BLOCKING:
      // If we have a queue we need to synchronize it now.
      Result = Device.synchronize(*this);
      break;
    case SyncTy::NON_BLOCKING:
      Result = Device.queryAsync(*this);
      break;
    case SyncTy::STATIC_NON_BLOCKING:
      Result = Device.queryAsyncStatic(*this) == OFFLOAD_FAIL ? OFFLOAD_FAIL
                                                              : OFFLOAD_SUCCESS;
      break;
    }
  }

  // Run any pending post-processing function registered on this async object.
  if (Result == OFFLOAD_SUCCESS && isQueueEmpty()) {
    ODBG(ODT_DataTransfer)
        << "Synchronization complete, running post-processing";
    Result = runPostProcessing();
  }

  return Result;
}

static int32_t processPPFs(SmallVector<AsyncInfoTy::PostProcFuncTy> &PPFs) {
  for (size_t I = 0; I < PPFs.size(); ++I)
    if (int Res = PPFs[I](); Res != OFFLOAD_SUCCESS)
      return Res;
  return OFFLOAD_SUCCESS;
}

int AsyncInfoTy::synchronize() {
  assert(SyncType == SyncTy::STATIC_NON_BLOCKING);

  // We still have not created a queue, or this specific device does not
  // generate queues.
  if (!Queue)
    return OFFLOAD_SUCCESS;

  // STATIC_NON_BLOCKING retains its queue thus more careful handling of the
  // post processing functions is required.

  // Collect the enqueued PPFs until this point.
  SmallVector<PostProcFuncTy> LocalPPFs;
  {
    std::lock_guard<std::mutex> PPFGuard{PostProcessingFunctionsMutex};
    std::swap(LocalPPFs, PostProcessingFunctions);
  }
  int Result = Device.synchronizeStatic(*this);
  // Run any pending post-processing function collected _before_ we
  // synchronize. This is important as between before the synchronization and
  // after we could have enqueued more post processing operations, which we
  // must not run yet.
  if (Result == OFFLOAD_SUCCESS)
    Result = processPPFs(LocalPPFs);

  return Result;
}

int AsyncInfoTy::query() {
  // If we don't have a queue, there are no pending actions.
  if (!Queue)
    return 0;
  return Device.queryAsyncStatic(*this);
}

void *&AsyncInfoTy::getVoidPtrLocation() {
  BufferLocations.push_back(nullptr);
  return BufferLocations.back();
}

bool AsyncInfoTy::isDone() const { return isQueueEmpty(); }

int32_t AsyncInfoTy::runPostProcessing() {
  // Post-processing procedures might add new procedures themselves, so
  // repeatedly process them until we are done.
  while (true) {
    SmallVector<PostProcFuncTy> LocalPPFs;
    {
      std::lock_guard<std::mutex> PPFGuard{PostProcessingFunctionsMutex};
      std::swap(LocalPPFs, PostProcessingFunctions);
    }
    if (LocalPPFs.size() == 0)
      return OFFLOAD_SUCCESS;
    int32_t Result = processPPFs(LocalPPFs);
    if (Result != OFFLOAD_SUCCESS)
      return Result;
  }
}

bool AsyncInfoTy::isQueueEmpty() const {
  if (!Queue)
    return true;
  bool IsComplete;
  if (auto Res = olQueryQueue(Queue, &IsComplete)) {
    REPORT() << "Failed to query queue " << Queue << ": " << Res->Details;
    return false;
  }
  return IsComplete;
}
