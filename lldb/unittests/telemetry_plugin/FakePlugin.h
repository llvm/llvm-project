//===-- FakePlugin.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TELEMETRY_FAKE_PLUGIN_H
#define LLDB_SOURCE_PLUGINS_TELEMETRY_FAKE_PLUGIN_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/Telemetry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"

struct FakeTelemetryInfo : public llvm::telemetry::TelemetryInfo {
  std::string msg;
};

class FakePlugin : public lldb_private::TelemetryManager {
public:
  FakePlugin() = default;

  // TelemetryManager interface
  llvm::Error preDistpatch(TelemetryInfo *entry) override;

  // Plugin interface
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "FakeTelemetryPlugin"; }
};

#endif
