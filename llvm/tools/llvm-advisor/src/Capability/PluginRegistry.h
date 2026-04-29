//===------------------- PluginRegistry.h - LLVM Advisor
//-------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Loads external advisor plugins (.so / .dll / .dylib) and registers their
// capabilities into a CapabilityRegistry.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Capability/llvm-advisor-plugin.h"
#include "llvm/Support/DynamicLibrary.h"

namespace llvm::advisor {

class CapabilityRegistry;

class PluginRegistry {
public:
  Error load(StringRef Path);
  Error loadVerified(StringRef Path, StringRef BLAKE3);
  bool isLoaded(StringRef Path) const { return Loaded.contains(Path); }

  /// Resolve symbols from all loaded plugins and register their specs +
  /// runners into the given CapabilityRegistry.
  Error registerPlugins(CapabilityRegistry &Registry);

private:
  struct Plugin {
    std::string Path;
    sys::DynamicLibrary Handle;
    void *RegisterFn = nullptr;
    void *RunFn = nullptr;
    void *FreeFn = nullptr;
  };

  SmallVector<Plugin, 4> Plugins;
  StringSet<> Loaded;
};

} // namespace llvm::advisor
