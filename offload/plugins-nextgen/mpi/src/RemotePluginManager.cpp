//===-- RemotePluginManager.cpp - Plugin loading and communication API ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling plugins.
//
//===----------------------------------------------------------------------===//

#include "RemotePluginManager.h"
#include "Shared/Debug.h"
#include "Shared/Profile.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>

using namespace llvm;
using namespace llvm::sys;

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/RemoteTargets.def"

void RemotePluginManager::init() {
  TIMESCOPE();
  DP("Loading RTLs...\n");

  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Plugins.emplace_back(                                                      \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()));              \
  } while (false);
#include "Shared/RemoteTargets.def"

  DP("RTLs loaded!\n");
}

void RemotePluginManager::deinit() {
  TIMESCOPE();
  DP("Unloading RTLs...\n");

  for (auto &Plugin : Plugins) {
    if (auto Err = Plugin->deinit()) {
      [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
      DP("Failed to deinit plugin: %s\n", InfoMsg.c_str());
    }
    Plugin.release();
  }

  DP("RTLs unloaded!\n");
}

void RemotePluginManager::initDevices(GenericPluginTy &RTL) {
  int32_t NumDevices = RTL.getNumDevices();
  int32_t Ret;
  for (int32_t DeviceID = 0; DeviceID < NumDevices; DeviceID++) {
    Ret = RTL.init_device(DeviceID);
    if (Ret != OFFLOAD_SUCCESS)
      DP("Failed to initialize device %d\n", DeviceID);
  }
}

void RemotePluginManager::initAllPlugins() {
  for (auto &R : Plugins)
    initDevices(*R);
}

/// Return the number of usable devices.
int RemotePluginManager::getNumDevices() {
  int32_t NumDevices = 0;
  for (auto &Plugin : Plugins) {
    if (!Plugin->is_initialized()) {
      if (auto Err = Plugin->init()) {
        [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
        DP("Failed to init plugin: %s\n", InfoMsg.c_str());
        continue;
      }
      DP("Registered plugin %s with %d visible device(s)\n", Plugin->getName(),
         Plugin->number_of_devices());
    }
    NumDevices += Plugin->number_of_devices();
  }
  return NumDevices;
}

int RemotePluginManager::getNumDevices(int32_t PluginId) {
  int32_t NumPlugins = getNumUsedPlugins();
  assert(PluginId < NumPlugins && "Invalid PluginId");
  if (!Plugins[PluginId]->is_initialized()) {
    if (auto Err = Plugins[PluginId]->init()) {
      [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
      DP("Failed to init plugin: %s\n", InfoMsg.c_str());
    }
  }
  return Plugins[PluginId]->number_of_devices();
}
