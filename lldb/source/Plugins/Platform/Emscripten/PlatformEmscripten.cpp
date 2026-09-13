//===-- PlatformEmscripten.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformEmscripten.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_emscripten;

LLDB_PLUGIN_DEFINE(PlatformEmscripten)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformEmscripten::CreateInstance(bool force,
                                              const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create =
      force || (arch && arch->IsValid() && arch->GetTriple().isOSEmscripten());
  LLDB_LOG(log, "create = {0}", create);
  return create ? PlatformSP(new PlatformEmscripten(false)) : PlatformSP();
}

llvm::StringRef PlatformEmscripten::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local Emscripten user platform plug-in.";
  return "Remote Emscripten user platform plug-in.";
}

void PlatformEmscripten::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(__EMSCRIPTEN__)
    PlatformSP platform_sp(new PlatformEmscripten(true));
    platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(platform_sp);
#endif
    PluginManager::RegisterPlugin(GetPluginNameStatic(false),
                                  GetPluginDescriptionStatic(false),
                                  PlatformEmscripten::CreateInstance, nullptr);
  }
}

void PlatformEmscripten::Terminate() {
  if (g_initialize_count > 0 && --g_initialize_count == 0)
    PluginManager::UnregisterPlugin(PlatformEmscripten::CreateInstance);

  PlatformPOSIX::Terminate();
}

PlatformEmscripten::PlatformEmscripten(bool is_host) : PlatformPOSIX(is_host) {
  if (is_host)
    m_supported_architectures.push_back(HostInfo::GetArchitecture());
  else
    m_supported_architectures = CreateArchList(
        {llvm::Triple::wasm32, llvm::Triple::wasm64}, llvm::Triple::Emscripten);
}

std::vector<ArchSpec> PlatformEmscripten::GetSupportedArchitectures(
    const ArchSpec &process_host_arch) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetSupportedArchitectures(process_host_arch);
  return m_supported_architectures;
}

bool PlatformEmscripten::CanDebugProcess() {
  return !IsHost() && IsConnected();
}
