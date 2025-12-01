//===----------------------------------------------------------------------===//
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

SyntheticFrameProvider::SyntheticFrameProvider(StackFrameListSP input_frames)
    : m_input_frames(std::move(input_frames)) {}

SyntheticFrameProvider::~SyntheticFrameProvider() = default;

void SyntheticFrameProviderDescriptor::Dump(Stream *s) const {
  if (!s)
    return;

  s->Printf("  Name: %s\n", GetName().str().c_str());

  // Show thread filter information.
  if (thread_specs.empty()) {
    s->PutCString("  Thread Filter: (applies to all threads)\n");
  } else {
    s->Printf("  Thread Filter: %zu specification(s)\n", thread_specs.size());
    for (size_t i = 0; i < thread_specs.size(); ++i) {
      const ThreadSpec &spec = thread_specs[i];
      s->Printf("    [%zu] ", i);
      spec.GetDescription(s, lldb::eDescriptionLevelVerbose);
      s->PutChar('\n');
    }
  }
}

llvm::Expected<SyntheticFrameProviderSP> SyntheticFrameProvider::CreateInstance(
    StackFrameListSP input_frames,
    const SyntheticFrameProviderDescriptor &descriptor) {
  if (!input_frames)
    return llvm::createStringError(
        "cannot create synthetic frame provider: invalid input frames");

  // Iterate through all registered ScriptedFrameProvider plugins.
  ScriptedFrameProviderCreateInstance create_callback = nullptr;
  for (uint32_t idx = 0;
       (create_callback =
            PluginManager::GetScriptedFrameProviderCreateCallbackAtIndex(
                idx)) != nullptr;
       ++idx) {
    auto provider_or_err = create_callback(input_frames, descriptor);
    if (!provider_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Target), provider_or_err.takeError(),
                     "Failed to create synthetic frame provider: {0}");
      continue;
    }

    if (auto frame_provider_up = std::move(*provider_or_err))
      return std::move(frame_provider_up);
  }

  return llvm::createStringError(
      "cannot create synthetic frame provider: no suitable plugin found");
}

llvm::Expected<SyntheticFrameProviderSP> SyntheticFrameProvider::CreateInstance(
    StackFrameListSP input_frames, llvm::StringRef plugin_name,
    const std::vector<ThreadSpec> &thread_specs) {
  if (!input_frames)
    return llvm::createStringError(
        "cannot create synthetic frame provider: invalid input frames");

  // Look up the specific C++ plugin by name.
  SyntheticFrameProviderCreateInstance create_callback =
      PluginManager::GetSyntheticFrameProviderCreateCallbackForPluginName(
          plugin_name);

  if (!create_callback)
    return llvm::createStringError(
        "cannot create synthetic frame provider: C++ plugin '%s' not found",
        plugin_name.str().c_str());

  auto provider_or_err = create_callback(input_frames, thread_specs);
  if (!provider_or_err)
    return provider_or_err.takeError();

  if (auto frame_provider_sp = std::move(*provider_or_err))
    return std::move(frame_provider_sp);

  return llvm::createStringError(
      "cannot create synthetic frame provider: C++ plugin '%s' returned null",
      plugin_name.str().c_str());
}
