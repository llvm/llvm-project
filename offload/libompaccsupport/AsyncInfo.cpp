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

int AsyncInfoTy::synchronize() {
  int Result = OFFLOAD_SUCCESS;
  if (!isQueueEmpty()) {
    switch (SyncType) {
    case SyncTy::BLOCKING:
      // If we have a queue we need to synchronize it now.
      Result = Device.synchronize(*this);
      assert(AsyncInfo.Queue == nullptr &&
             "The device plugin should have nulled the queue to indicate there "
             "are no outstanding actions!");
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

bool AsyncInfoTy::isQueueEmpty() const { return AsyncInfo.Queue == nullptr; }
