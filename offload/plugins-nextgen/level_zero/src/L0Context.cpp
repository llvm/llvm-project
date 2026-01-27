//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Level Zero Context abstraction.
//
//===----------------------------------------------------------------------===//

#include "L0Context.h"
#include "L0Plugin.h"

namespace llvm::omp::target::plugin {

Error L0ContextTy::init() {
  CALL_ZE_RET_ERROR(zeDriverGetApiVersion, zeDriver, &APIVersion);
  ODBG(OLDT_Init) << "Driver API version is "
                  << llvm::format(PRIx32, APIVersion);

  ze_context_desc_t Desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CALL_ZE_RET_ERROR(zeContextCreate, zeDriver, &Desc, &zeContext);
  if (auto Err = EventPool.init(zeContext, 0))
    return Err;
  if (auto Err = HostMemAllocator.initHostPool(*this, Plugin.getOptions()))
    return Err;
  return Plugin::success();
}

Error L0ContextTy::deinit() {
  if (auto Err = EventPool.deinit())
    return Err;
  if (auto Err = HostMemAllocator.deinit())
    return Err;
  if (zeContext)
    CALL_ZE_RET_ERROR(zeContextDestroy, zeContext);
  return Plugin::success();
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
