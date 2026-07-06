//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SyntheticFrameProvider.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Interfaces/ScriptedFrameProviderInterface.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

SyntheticFrameProvider::SyntheticFrameProvider(StackFrameListSP input_frames)
    : m_input_frames(std::move(input_frames)) {}

SyntheticFrameProvider::~SyntheticFrameProvider() = default;

void ScriptedFrameProviderDescriptor::Dump(Stream *s) const {
  if (!s)
    return;

  s->Format("  ID: {0}\n", GetID());
  s->Format("  Name: {0}\n", GetName());

  std::string description = GetDescription();
  if (!description.empty())
    s->Format("  Description: {0}\n", description);

  // Show priority information.
  std::optional<uint32_t> priority = GetPriority();
  if (priority.has_value())
    s->Format("  Priority: {0}\n", *priority);
  else
    s->PutCString("  Priority: Default (no priority specified)\n");

  // Show thread filter information.
  if (thread_specs.empty()) {
    s->PutCString("  Thread Filter: (applies to all threads)\n");
  } else {
    s->Format("  Thread Filter: {0} specification(s)\n", thread_specs.size());
    for (size_t i = 0; i < thread_specs.size(); ++i) {
      const ThreadSpec &spec = thread_specs[i];
      s->Format("    [{0}] ", i);
      spec.GetDescription(s, lldb::eDescriptionLevelVerbose);
      s->EOL();
    }
  }
}

uint32_t ScriptedFrameProviderDescriptor::GetHash() const {
  if (!scripted_metadata_sp)
    return 0;

  return scripted_metadata_sp->GetHash();
}

std::string ScriptedFrameProviderDescriptor::GetDescription() const {
  // If we have an interface, call get_description() to fetch it.
  if (interface_sp && scripted_metadata_sp)
    return interface_sp->GetDescription(scripted_metadata_sp->GetClassName());
  return {};
}

std::optional<uint32_t> ScriptedFrameProviderDescriptor::GetPriority() const {
  // If we have an interface, call get_priority() to fetch it.
  if (interface_sp && scripted_metadata_sp)
    return interface_sp->GetPriority(scripted_metadata_sp->GetClassName());
  return std::nullopt;
}

llvm::Expected<SyntheticFrameProviderSP> SyntheticFrameProvider::CreateInstance(
    StackFrameListSP input_frames,
    const ScriptedFrameProviderDescriptor &descriptor) {
  if (!input_frames)
    return llvm::createStringError(
        "cannot create synthetic frame provider: invalid input frames");

  // Iterate through all registered ScriptedFrameProvider plugins.
  for (auto create_callback :
       PluginManager::GetScriptedFrameProviderCreateCallbacks()) {
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
