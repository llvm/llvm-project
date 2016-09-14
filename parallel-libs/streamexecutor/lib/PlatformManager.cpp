//===-- PlatformManager.cpp - PlatformManager implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of PlatformManager class internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/PlatformManager.h"
#include "streamexecutor/platforms/host/HostPlatform.h"

namespace streamexecutor {

PlatformManager::PlatformManager() {
  // TODO(jhen): Register known platforms by name.
  // We have a couple of options here:
  //  * Use build-system flags to set preprocessor macros that select the
  //    appropriate code to include here.
  //  * Use static initialization tricks to have platform libraries register
  //    themselves when they are loaded.

  PlatformsByName.emplace("host", llvm::make_unique<host::HostPlatform>());
}

Expected<Platform *> PlatformManager::getPlatformByName(llvm::StringRef Name) {
  static PlatformManager Instance;
  auto Iterator = Instance.PlatformsByName.find(Name.lower());
  if (Iterator != Instance.PlatformsByName.end())
    return Iterator->second.get();
  return make_error("no available platform with name " + Name);
}

} // namespace streamexecutor
