//===- lib/Passes/PassPlugin.cpp - Load Plugins for PR Passes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

using namespace mlir;

llvm::Expected<PassPlugin> PassPlugin::load(const std::string &filename) {
  std::string Error;
  auto library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &Error);
  if (!library.isValid())
    return llvm::make_error<llvm::StringError>(
        Twine("Could not load library '") + filename + "': " + Error,
        llvm::inconvertibleErrorCode());

  PassPlugin plugin{filename, library};

  // mlirGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  intptr_t getDetailsFn =
      (intptr_t)library.getAddressOfSymbol("mlirGetPassPluginInfo");

  if (!getDetailsFn)
    return llvm::make_error<llvm::StringError>(
        Twine("Plugin entry point not found in '") + filename,
        llvm::inconvertibleErrorCode());

  plugin.info =
      reinterpret_cast<decltype(mlirGetPassPluginInfo) *>(getDetailsFn)();

  if (plugin.info.apiVersion != MLIR_PLUGIN_API_VERSION)
    return llvm::make_error<llvm::StringError>(
        Twine("Wrong API version on plugin '") + filename + "'. Got version " +
            Twine(plugin.info.apiVersion) + ", supported version is " +
            Twine(MLIR_PLUGIN_API_VERSION) + ".",
        llvm::inconvertibleErrorCode());

  if (!plugin.info.registerPassRegistryCallbacks)
    return llvm::make_error<llvm::StringError>(
        Twine("Empty entry callback in plugin '") + filename + "'.'",
        llvm::inconvertibleErrorCode());

  return plugin;
}
