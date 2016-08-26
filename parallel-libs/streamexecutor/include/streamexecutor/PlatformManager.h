//===-- PlatformManager.h - The PlatformManager class -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PlatformManager is the entry point into the StreamExecutor API. A user
/// begins be calling PlatformManager::getPlatformByName("cuda") where "cuda"
/// can be replaced by any supported platform name. This gives the user a
/// Platform object that can be used to create Device objects for that platform,
/// etc.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMMANAGER_H
#define STREAMEXECUTOR_PLATFORMMANAGER_H

#include <map>

#include "streamexecutor/Platform.h"
#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

/// A singleton that holds a reference to a Platform object for each
/// supported StreamExecutor platform.
class PlatformManager {
public:
  /// Gets a reference to the Platform with the given name.
  ///
  /// The name parameter is not case-sensitive, so the following arguments are
  /// all equivalent: "cuda", "CUDA", "Cuda", "cUdA".
  ///
  /// Returns an error if no platform is present for the name.
  ///
  /// Ownership of the platform is NOT transferred to the caller.
  static Expected<Platform *> getPlatformByName(llvm::StringRef Name);

private:
  PlatformManager();
  PlatformManager(const PlatformManager &) = delete;
  PlatformManager operator=(const PlatformManager &) = delete;

  std::map<std::string, std::unique_ptr<Platform>> PlatformsByName;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMMANAGER_H
