//===- mlir/IR/DialectPlugin.h - Public Plugin API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the public entry point for dialect plugins.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTPLUGIN_H
#define MLIR_IR_DIALECTPLUGIN_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassPlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace mlir {
extern "C" {
/// Information about the plugin required to load its dialects & passes
///
/// This struct defines the core interface for dialect plugins and is supposed
/// to be filled out by plugin implementors. MLIR-side users of a plugin are
/// expected to use the \c DialectPlugin class below to interface with it.
struct DialectPluginLibraryInfo {
  /// The API version understood by this plugin, usually
  /// \c MLIR_PLUGIN_API_VERSION
  uint32_t apiVersion;
  /// A meaningful name of the plugin.
  const char *pluginName;
  /// The version of the plugin.
  const char *pluginVersion;

  /// The callback for registering dialect plugin with a \c DialectRegistry
  /// instance
  void (*registerDialectRegistryCallbacks)(DialectRegistry *);
};
}

/// A loaded dialect plugin.
///
/// An instance of this class wraps a loaded dialect plugin and gives access to
/// its interface defined by the \c DialectPluginLibraryInfo it exposes.
class DialectPlugin {
public:
  /// Attempts to load a dialect plugin from a given file.
  ///
  /// \returns Returns an error if either the library cannot be found or loaded,
  /// there is no public entry point, or the plugin implements the wrong API
  /// version.
  static llvm::Expected<DialectPlugin> load(const std::string &filename);

  /// Get the filename of the loaded plugin.
  StringRef getFilename() const { return filename; }

  /// Get the plugin name
  StringRef getPluginName() const { return info.pluginName; }

  /// Get the plugin version
  StringRef getPluginVersion() const { return info.pluginVersion; }

  /// Get the plugin API version
  uint32_t getAPIVersion() const { return info.apiVersion; }

  /// Invoke the DialectRegistry callback registration
  void
  registerDialectRegistryCallbacks(DialectRegistry &dialectRegistry) const {
    info.registerDialectRegistryCallbacks(&dialectRegistry);
  }

private:
  DialectPlugin(const std::string &filename,
                const llvm::sys::DynamicLibrary &library)
      : filename(filename), library(library), info() {}

  std::string filename;
  llvm::sys::DynamicLibrary library;
  DialectPluginLibraryInfo info;
};
} // namespace mlir

/// The public entry point for a dialect plugin.
///
/// When a plugin is loaded by the driver, it will call this entry point to
/// obtain information about this plugin and about how to register its dialects.
/// This function needs to be implemented by the plugin, see the example below:
///
/// ```
/// extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
/// mlirGetDialectPluginInfo() {
///   return {
///     MLIR_PLUGIN_API_VERSION, "MyPlugin", "v0.1", [](DialectRegistry) { ... }
///   };
/// }
/// ```
extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo();

#endif /* MLIR_IR_DIALECTPLUGIN_H */
