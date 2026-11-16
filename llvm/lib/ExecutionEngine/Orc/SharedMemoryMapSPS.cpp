//===- SharedMemoryMapSPS.cpp - SPS shared-memory map bindings ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SharedMemoryMapSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RecordProxy.h"

namespace llvm::orc::sps {

Expected<SharedMemoryMapBindings> createSharedMemoryMapBindings(JITDylib &JD) {
  SharedMemoryMapBindings B;
  // Instance is the executor-side mapper object -- a data symbol passed as the
  // first argument to each call, not a wrapper to proxy.
  if (auto Err = lookupAndApply(
          JD,
          {recordAddr(rt::sps_ci::SharedMemoryMapperInstanceName, &B.Instance),
           recordProxy<SharedMemoryMapReserveProxySpec>(&B.Reserve),
           recordProxy<SharedMemoryMapInitializeProxySpec>(&B.Initialize),
           recordProxy<SharedMemoryMapDeinitializeProxySpec>(&B.Deinitialize),
           recordProxy<SharedMemoryMapReleaseProxySpec>(&B.Release)}))
    return std::move(Err);
  return std::move(B);
}

Expected<SharedMemoryMapBindings>
createSharedMemoryMapBindings(ExecutionSession &ES) {
  return createSharedMemoryMapBindings(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
