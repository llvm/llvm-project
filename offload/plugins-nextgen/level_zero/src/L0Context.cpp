//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Level Zero Context abstraction
//
//===----------------------------------------------------------------------===//

#include "L0Context.h"
#include "L0Plugin.h"

namespace llvm::omp::target::plugin {

L0ContextTy::L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
                         int32_t /*DriverId*/)
    : Plugin(Plugin), zeDriver(zeDriver) {
  CALL_ZE_RET_VOID(zeDriverGetApiVersion, zeDriver, &APIVersion);
  DP("Driver API version is %" PRIx32 "\n", APIVersion);

  ze_context_desc_t Desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CALL_ZE_RET_VOID(zeContextCreate, zeDriver, &Desc, &zeContext);

  EventPool.init(zeContext, 0);
  HostMemAllocator.initHostPool(*this, Plugin.getOptions());
}

StagingBufferTy &L0ContextTy::getStagingBuffer() {
  auto &TLS = Plugin.getContextTLS(getZeContext());
  auto &Buffer = TLS.getStagingBuffer();
  const auto &Options = Plugin.getOptions();
  if (!Buffer.initialized())
    Buffer.init(getZeContext(), Options.StagingBufferSize,
                Options.StagingBufferCount);
  return Buffer;
}

} // namespace llvm::omp::target::plugin
