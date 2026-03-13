//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginRegistry.h"
#include "Plugins/Accelerator/Mock/LLDBServerPluginMockAccelerator.h"
#include "llvm/Support/Threading.h"

using namespace lldb_private::lldb_server;

LLDBServerPluginRegistry &LLDBServerPluginRegistry::Instance() {
  static LLDBServerPluginRegistry registry;
  static llvm::once_flag once;

  llvm::call_once(once, []() { registry.Initialize(); });

  return registry;
}

void LLDBServerPluginRegistry::Initialize() {
  // Register all known accelerator plugins
  Register(PluginKind::Accelerator,
           [](LLDBServerPluginAccelerator::GDBServer &gdb_server,
              lldb_private::MainLoop &main_loop)
               -> std::unique_ptr<LLDBServerPluginAccelerator> {
             return std::make_unique<LLDBServerPluginMockAccelerator>(gdb_server,
                                                                      main_loop);
           });
}

void LLDBServerPluginRegistry::Register(PluginKind kind,
                                        AcceleratorFactory factory) {
  m_accelerator_factories.push_back(std::move(factory));
}

std::vector<std::unique_ptr<LLDBServerPluginAccelerator>>
LLDBServerPluginRegistry::TryInstantiateAllAcceleratorPlugins(
    LLDBServerPluginAccelerator::GDBServer &gdb_server, MainLoop &main_loop) {
  std::vector<std::unique_ptr<LLDBServerPluginAccelerator>> activated;
  for (const AcceleratorFactory &factory : m_accelerator_factories) {
    std::unique_ptr<LLDBServerPluginAccelerator> plugin = factory(gdb_server, main_loop);
    if (plugin && plugin->TryActivate())
      activated.push_back(std::move(plugin));
  }
  return activated;
}
