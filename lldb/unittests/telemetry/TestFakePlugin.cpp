//===-- TestFakePlugin.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Telemetry.h"
#include "plugin/FakePlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"
#include "gtest/gtest.h"

#include <memory>

TEST(TelemetryTest, PluginTest) {
  // This would have been called by the plugin reg in a "real" plugin
  // For tests, we just call it directly.
  lldb_private::FakePlugin::Initialize();

  auto ins = lldb_private::telemetry::TelemetryManager::getInstance();

  ASSERT_NE(ins, nullptr);
  lldb_private::FakeTelemetryInfo entry;
  entry.msg = "";

  auto stat = ins->preDispatch(&entry);
  ASSERT_FALSE(stat);
  ASSERT_EQ("In FakePlugin", entry.msg);
}
