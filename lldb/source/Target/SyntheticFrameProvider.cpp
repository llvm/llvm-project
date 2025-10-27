//===-- SyntheticFrameProvider.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SyntheticFrameProvider.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

SyntheticFrameProvider::SyntheticFrameProvider(ThreadSP thread_sp)
    : m_thread_sp(std::move(thread_sp)) {}

SyntheticFrameProvider::~SyntheticFrameProvider() = default;

llvm::Expected<SyntheticFrameProviderSP>
SyntheticFrameProvider::CreateInstance(ThreadSP thread_sp) {
  if (!thread_sp)
    return llvm::createStringError(
        "cannot create synthetic frame provider: Invalid thread");

  // Iterate through all registered SyntheticFrameProvider plugins
  SyntheticFrameProviderCreateInstance create_callback = nullptr;
  for (uint32_t idx = 0;
       (create_callback =
            PluginManager::GetSyntheticFrameProviderCreateCallbackAtIndex(
                idx)) != nullptr;
       ++idx) {
    auto provider_or_err = create_callback(thread_sp);
    if (!provider_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Target), provider_or_err.takeError(),
                     "Failed to create synthetic frame provider: {0}");
      continue;
    }

    if (auto frame_provider_up = std::move(*provider_or_err))
      return std::move(frame_provider_up);
  }

  return llvm::createStringError(
      "cannot create synthetic frame provider: No suitable plugin found");
}
