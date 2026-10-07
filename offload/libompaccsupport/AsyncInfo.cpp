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

// Temporary helper from liboffload until all usage of AsyncInfo is migrated
// to liboffload queues.
namespace llvm::offload::tmp {
__tgt_async_info *__ol_tgt_GetAsyncInfoFromQueue(ol_queue_handle_t Queue);
} // namespace llvm::offload::tmp

AsyncInfoTy::AsyncInfoTy(DeviceTy &Device, SyncTy SyncType)
    : Device(Device), SyncType(SyncType) {
  if (auto Res = olCreateQueue(Device.Context, Device.DeviceHandle, &Queue)) {
    REPORT() << "Failed to create queue for device " << Device.DeviceID << ": "
             << Res->Details;
    Queue = nullptr;
  }
}

AsyncInfoTy::~AsyncInfoTy() {
  synchronize();
  if (Queue)
    if (auto Res = olDestroyQueue(Queue))
      REPORT() << "Failed to destroy queue " << Queue << ": " << Res->Details;
}

AsyncInfoTy::operator __tgt_async_info *() {
  return llvm::offload::tmp::__ol_tgt_GetAsyncInfoFromQueue(Queue);
}

int AsyncInfoTy::synchronize() {
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

void *&AsyncInfoTy::getVoidPtrLocation() {
  BufferLocations.push_back(nullptr);
  return BufferLocations.back();
}

bool AsyncInfoTy::isDone() const { return isQueueEmpty(); }

int32_t AsyncInfoTy::runPostProcessing() {
  size_t Size = PostProcessingFunctions.size();
  for (size_t I = 0; I < Size; ++I) {
    const int Result = PostProcessingFunctions[I]();
    if (Result != OFFLOAD_SUCCESS)
      return Result;
  }

  // Clear the vector up until the last known function, since post-processing
  // procedures might add new procedures themselves.
  const auto *PrevBegin = PostProcessingFunctions.begin();
  PostProcessingFunctions.erase(PrevBegin, PrevBegin + Size);

  return OFFLOAD_SUCCESS;
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
