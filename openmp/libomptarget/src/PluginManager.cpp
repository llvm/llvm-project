//===-- PluginManager.cpp - Plugin loading and communication API ---------===//
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

#include "PluginManager.h"

using namespace llvm;
using namespace llvm::sys;

PluginManager *PM;

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* AMDGPU target        */ "libomptarget.rtl.amdgpu",
    /* CUDA target          */ "libomptarget.rtl.cuda",
    /* x86_64 target        */ "libomptarget.rtl.x86_64",
    /* AArch64 target       */ "libomptarget.rtl.aarch64",
    /* PowerPC target       */ "libomptarget.rtl.ppc64",
};

PluginAdaptorTy::PluginAdaptorTy(const std::string &Name) : Name(Name) {
  DP("Attempting to load library '%s'...\n", Name.c_str());

  std::string ErrMsg;
  LibraryHandler = std::make_unique<DynamicLibrary>(
      DynamicLibrary::getPermanentLibrary(Name.c_str(), &ErrMsg));

  if (!LibraryHandler->isValid()) {
    // Library does not exist or cannot be found.
    DP("Unable to load library '%s': %s!\n", Name.c_str(), ErrMsg.c_str());
    return;
  }

  DP("Successfully loaded library '%s'!\n", Name.c_str());

#define PLUGIN_API_HANDLE(NAME, MANDATORY)                                     \
  NAME = reinterpret_cast<decltype(NAME)>(                                     \
      LibraryHandler->getAddressOfSymbol(GETNAME(__tgt_rtl_##NAME)));          \
  if (MANDATORY && !NAME) {                                                    \
    DP("Invalid plugin as necessary interface is not found.\n");               \
    return;                                                                    \
  }

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  // Remove plugin on failure to call optional init_plugin
  int32_t Rc = init_plugin();
  if (Rc != OFFLOAD_SUCCESS) {
    DP("Unable to initialize library '%s': %u!\n", Name.c_str(), Rc);
    return;
  }

  // No devices are supported by this RTL?
  NumberOfDevices = number_of_devices();
  if (!NumberOfDevices) {
    DP("No devices supported in this RTL\n");
    return;
  }

  DP("Registered '%s' with %d devices!\n", Name.c_str(), NumberOfDevices);
}

void PluginManager::init() {
  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (const char *Name : RTLNames) {
    PluginAdaptors.emplace_back(std::string(Name) + ".so");
    if (PluginAdaptors.back().getNumDevices() <= 0)
      PluginAdaptors.pop_back();
  }

  DP("RTLs loaded!\n");
}

void PluginManager::initPlugin(PluginAdaptorTy &Plugin) {
  // If this RTL is not already in use, initialize it.
  if (Plugin.isUsed() || !Plugin.NumberOfDevices)
    return;

  // Initialize the device information for the RTL we are about to use.
  const size_t Start = Devices.size();
  Devices.reserve(Start + Plugin.NumberOfDevices);
  for (int32_t DeviceId = 0; DeviceId < Plugin.NumberOfDevices; DeviceId++) {
    Devices.push_back(std::make_unique<DeviceTy>(&Plugin));
    // global device ID
    Devices[Start + DeviceId]->DeviceID = Start + DeviceId;
    // RTL local device ID
    Devices[Start + DeviceId]->RTLDeviceID = DeviceId;
  }

  // Initialize the index of this RTL and save it in the used RTLs.
  Plugin.DeviceOffset = Start;

  // If possible, set the device identifier offset in the plugin.
  if (Plugin.set_device_offset)
    Plugin.set_device_offset(Start);

  DP("RTL " DPxMOD " has index %d!\n", DPxPTR(Plugin.LibraryHandler.get()),
     Plugin.DeviceOffset);
}

void PluginManager::initAllPlugins() {
  for (auto &R : PluginAdaptors)
    initPlugin(R);
}
