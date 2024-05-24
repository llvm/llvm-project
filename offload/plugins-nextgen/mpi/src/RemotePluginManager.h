//===-- ProxyRemotePluginManager.h - Remote Plugin Manager ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing remote devices that are handled by MPI Plugin.
//
//===----------------------------------------------------------------------===//

#ifndef REMOTE_PLUGIN_MANAGER_H
#define REMOTE_PLUGIN_MANAGER_H

#include "PluginInterface.h"
#include "Shared/APITypes.h"
#include "Shared/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>

using llvm::sys::DynamicLibrary;

using GenericPluginTy = llvm::omp::target::plugin::GenericPluginTy;
using GenericDeviceTy = llvm::omp::target::plugin::GenericDeviceTy;

/// Device Image Storage. This class is used to store Device Image data
/// in the remote device process.
struct DeviceImage : __tgt_device_image {
  llvm::SmallVector<unsigned char, 1> ImageBuffer;
  llvm::SmallVector<llvm::offloading::EntryTy, 16> Entries;
  llvm::SmallVector<char> FlattenedEntryNames;

  DeviceImage() {
    ImageStart = nullptr;
    ImageEnd = nullptr;
    EntriesBegin = nullptr;
    EntriesEnd = nullptr;
  }

  DeviceImage(size_t ImageSize, size_t EntryCount)
      : ImageBuffer(ImageSize + alignof(void *)), Entries(EntryCount) {
    // Align the image buffer to alignof(void *).
    ImageStart = ImageBuffer.begin();
    std::align(alignof(void *), ImageSize, ImageStart, ImageSize);
    ImageEnd = (void *)((size_t)ImageStart + ImageSize);
  }

  void setImageEntries(llvm::SmallVector<size_t> EntryNameSizes) {
    // Adjust the entry names to use the flattened name buffer.
    size_t EntryCount = Entries.size();
    size_t TotalNameSize = 0;
    for (size_t I = 0; I < EntryCount; I++) {
      TotalNameSize += EntryNameSizes[I];
    }
    FlattenedEntryNames.resize(TotalNameSize);

    for (size_t I = EntryCount; I > 0; I--) {
      TotalNameSize -= EntryNameSizes[I - 1];
      Entries[I - 1].SymbolName = &FlattenedEntryNames[TotalNameSize];
    }

    // Set the entries pointers.
    EntriesBegin = Entries.begin();
    EntriesEnd = Entries.end();
  }

  /// Get the image size.
  size_t getSize() const { return utils::getPtrDiff(ImageEnd, ImageStart); }

  /// Getter and setter for the dynamic library.
  DynamicLibrary &getDynamicLibrary() { return DynLib; }
  void setDynamicLibrary(const DynamicLibrary &Lib) { DynLib = Lib; }

private:
  DynamicLibrary DynLib;
};

/// Struct for the data required to handle plugins
struct RemotePluginManager {

  RemotePluginManager() {}

  void init();

  void deinit();

  /// Initialize as many devices as possible for this plugin. Devices that fail
  /// to initialize are ignored.
  void initDevices(GenericPluginTy &RTL);

  /// Return the number of usable devices.
  int getNumDevices();

  int getNumDevices(int32_t PluginId);

  int getNumUsedPlugins() const { return Plugins.size(); }

  // Initialize all plugins.
  void initAllPlugins();

  /// Iterator range for all plugins (in use or not, but always valid).
  auto plugins() { return llvm::make_pointee_range(Plugins); }

  auto getPlugin(int32_t PluginId) { return &Plugins[PluginId]; }

  // List of all plugins, in use or not.
  llvm::SmallVector<std::unique_ptr<GenericPluginTy>> Plugins;
};

#endif // REMOTE_PLUGIN_MANAGER_H
