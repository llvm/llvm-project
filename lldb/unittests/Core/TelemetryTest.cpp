//===-- TelemetryTest.cpp ------------------------------------------------===//
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
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace lldb_private {

struct FakeTelemetryInfo : public llvm::telemetry::TelemetryInfo {
  std::string msg;
};

class TestDestination : public llvm::telemetry::Destination {
public:
  TestDestination(std::vector<const llvm::telemetry::TelemetryInfo *> *entries)
      : received_entries(entries) {}

  llvm::Error
  receiveEntry(const llvm::telemetry::TelemetryInfo *entry) override {
    received_entries->push_back(entry);
    return llvm::Error::success();
  }

  llvm::StringLiteral name() const override { return "TestDestination"; }

private:
  std::vector<const llvm::telemetry::TelemetryInfo *> *received_entries;
};

class FakePlugin : public telemetry::TelemetryManager {
public:
  FakePlugin()
      : telemetry::TelemetryManager(
            std::make_unique<llvm::telemetry::Config>(true)) {}

  // TelemetryManager interface
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override {
    if (auto *fake_entry = llvm::dyn_cast<FakeTelemetryInfo>(entry))
      fake_entry->msg = "In FakePlugin";

    return llvm::Error::success();
  }

  llvm::StringRef GetInstanceName() const override {
    return "FakeTelemetryPlugin";
  }

  static void Initialize() {
    telemetry::TelemetryManager::SetInstance(std::make_unique<FakePlugin>());
  }

  static void Terminate() { telemetry::TelemetryManager::SetInstance(nullptr); }
};

} // namespace lldb_private

#if LLVM_ENABLE_TELEMETRY
#define TELEMETRY_TEST(suite, test) TEST(suite, test)
#else
#define TELEMETRY_TEST(suite, test) TEST(DISABLED_##suite, test)
#endif


TELEMETRY_TEST(TelemetryTest, PluginTest) {
  // This would have been called by the plugin reg in a "real" plugin
  // For tests, we just call it directly.
  lldb_private::FakePlugin::Initialize();

  auto *ins = lldb_private::telemetry::TelemetryManager::GetInstance();
  ASSERT_NE(ins, nullptr);

  std::vector<const ::llvm::telemetry::TelemetryInfo *> expected_entries;
  ins->addDestination(
      std::make_unique<lldb_private::TestDestination>(&expected_entries));

  lldb_private::FakeTelemetryInfo entry;
  entry.msg = "";

  ASSERT_THAT_ERROR(ins->dispatch(&entry), ::llvm::Succeeded());
  ASSERT_EQ(1U, expected_entries.size());
  EXPECT_EQ("In FakePlugin",
            llvm::dyn_cast<lldb_private::FakeTelemetryInfo>(expected_entries[0])
                ->msg);

  ASSERT_EQ("FakeTelemetryPlugin", ins->GetInstanceName());
}
