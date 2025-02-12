//===-- FakePlugin.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FakePlugin.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Telemetry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"

#include <memory>

LLDB_PLUGIN_DEFINE(FakePlugin)

namespace lldb_private {

FakePlugin::FakePlugin()
    : telemetry::TelemetryManager(
          std::make_unique<llvm::telemetry::Config>(true)) {}

llvm::Error FakePlugin::preDispatch(llvm::telemetry::TelemetryInfo *entry) {
  if (auto *fake_entry = llvm::dyn_cast<FakeTelemetryInfo>(entry)) {
    fake_entry->msg = "In FakePlugin";
  }

  return llvm::Error::success();
}

void FakePlugin::Initialize() {
  telemetry::TelemetryManager::setInstance(std::make_unique<FakePlugin>());
  // TODO: do we need all the PluginManagerL::RegisterPlugin()  stuff???
}

void FakePlugin::Terminate() {
  // nothing to do?
}

} // namespace lldb_private
