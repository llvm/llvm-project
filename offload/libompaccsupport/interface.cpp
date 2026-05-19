//===-------- interface.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PluginManager.h"
#include "omptarget.h"

EXTERN void __tgt_rtl_init() { initRuntime(/*OffloadEnabled=*/true); }
EXTERN void __tgt_rtl_deinit() { deinitRuntime(); }

////////////////////////////////////////////////////////////////////////////////
/// Initialize all available devices without registering any image
EXTERN void __tgt_init_all_rtls() {
  assert(PM && "Runtime not initialized");
  PM->initializeAllDevices();
}

EXTERN void __tgt_register_rpc_callback(unsigned (*Callback)(void *,
                                                             unsigned)) {
  for (auto &Plugin : PM->plugins())
    if (Plugin.is_initialized())
      Plugin.getRPCServer().registerCallback(Callback);
}
