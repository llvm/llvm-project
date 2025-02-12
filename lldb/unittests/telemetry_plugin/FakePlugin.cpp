//===-- FakePlugin.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Telemetry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"

#include "FakePlugin.h"

#include <memory>

LLDB_PLUGIN_DEFINE(FakePlugin)

llvm::Error FakePlugin::preDispatch(TelemetryInfo *entry) {
  dynamic_cast<FakeTelemetryInfo>(entry)->msg = "In FakePlugin";
  return Error::success();
}

void FakePlugin::Initialize() {
  lldb_private::telemetry::TelemetryManager::setInstance(
      std::make_unique<FakePlugin>());
  // TODO: do we need all the PluginManagerL::RegisterPlugin()  stuff???
}

void FakePlugin::Terminate() {
  // nothing to do?
}
