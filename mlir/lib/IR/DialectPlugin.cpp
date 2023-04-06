//===- lib/IR/DialectPlugin.cpp - Load Dialect Plugins --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

using namespace mlir;

llvm::Expected<DialectPlugin> DialectPlugin::load(const std::string &filename) {
  std::string error;
  auto library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &error);
  if (!library.isValid())
    return llvm::make_error<llvm::StringError>(
        Twine("Could not load library '") + filename + "': " + error,
        llvm::inconvertibleErrorCode());

  DialectPlugin plugin{filename, library};

  // mlirGetDialectPluginInfo should be resolved to the definition from the
  // plugin we are currently loading.
  intptr_t getDetailsFn =
      (intptr_t)library.getAddressOfSymbol("mlirGetDialectPluginInfo");

  if (!getDetailsFn)
    return llvm::make_error<llvm::StringError>(
        Twine("Plugin entry point not found in '") + filename,
        llvm::inconvertibleErrorCode());

  plugin.info =
      reinterpret_cast<decltype(mlirGetDialectPluginInfo) *>(getDetailsFn)();

  if (plugin.info.apiVersion != MLIR_PLUGIN_API_VERSION)
    return llvm::make_error<llvm::StringError>(
        Twine("Wrong API version on plugin '") + filename + "'. Got version " +
            Twine(plugin.info.apiVersion) + ", supported version is " +
            Twine(MLIR_PLUGIN_API_VERSION) + ".",
        llvm::inconvertibleErrorCode());

  if (!plugin.info.registerDialectRegistryCallbacks)
    return llvm::make_error<llvm::StringError>(
        Twine("Empty entry callback in plugin '") + filename + "'.'",
        llvm::inconvertibleErrorCode());

  return plugin;
}
