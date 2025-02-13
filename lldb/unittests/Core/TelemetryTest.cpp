//===-- TelemetryTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/llvm-config.h"

#ifdef LLVM_BUILD_TELEMETRY

#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Telemetry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"
#include "gtest/gtest.h"

#include <memory>

namespace lldb_private {

struct FakeTelemetryInfo : public llvm::telemetry::TelemetryInfo {
  std::string msg;
};

class FakePlugin : public telemetry::TelemetryManager {
public:
  FakePlugin()
      : telemetry::TelemetryManager(
            std::make_unique<llvm::telemetry::Config>(true)) {}

  // TelemetryManager interface
  llvm::Error dispatch(llvm::telemetry::TelemetryInfo *entry) override {
    if (auto *fake_entry = llvm::dyn_cast<FakeTelemetryInfo>(entry)) {
      fake_entry->msg = "In FakePlugin";
    }

    return llvm::Error::success();
  }

  // Plugin interface
  llvm::StringRef GetPluginName() override { return "FakeTelemetryPlugin"; }

  static void Initialize() {
    telemetry::TelemetryManager::setInstance(std::make_unique<FakePlugin>());
  }

  static void Terminate() { telemetry::TelemetryManager::setInstance(nullptr); }
};

} // namespace lldb_private

TEST(TelemetryTest, PluginTest) {
  // This would have been called by the plugin reg in a "real" plugin
  // For tests, we just call it directly.
  lldb_private::FakePlugin::Initialize();

  auto ins = lldb_private::telemetry::TelemetryManager::getInstance();

  ASSERT_NE(ins, nullptr);
  lldb_private::FakeTelemetryInfo entry;
  entry.msg = "";

  auto stat = ins->dispatch(&entry);
  ASSERT_FALSE(stat);
  ASSERT_EQ("In FakePlugin", entry.msg);

  ASSERT_EQ("FakeTelemetryPlugin", ins->GetPluginName());
}

#endif // LLVM_BUILD_TELEMETRY
