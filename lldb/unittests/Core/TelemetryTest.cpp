//===-- TelemetryTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/PluginManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Telemetry/Telemetry.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace lldb_private {

struct FakeTelemetryInfo : public telemetry::LLDBBaseTelemetryInfo {
  std::string msg;
  int num;

  ::llvm::telemetry::KindType getKind() const override { return 0b11111111; }
};

class TestDestination : public llvm::telemetry::Destination {
public:
  TestDestination(
      std::vector<std::unique_ptr<llvm::telemetry::TelemetryInfo>> *entries)
      : received_entries(entries) {}

  llvm::Error
  receiveEntry(const llvm::telemetry::TelemetryInfo *entry) override {
    // Save a copy of the entry for later verification (because the original
    // entry might have gone out of scope by the time verification is done.
    if (auto *fake_entry = llvm::dyn_cast<FakeTelemetryInfo>(entry))
      received_entries->push_back(
          std::make_unique<FakeTelemetryInfo>(*fake_entry));
    return llvm::Error::success();
  }

  llvm::StringLiteral name() const override { return "TestDestination"; }

private:
  std::vector<std::unique_ptr<llvm::telemetry::TelemetryInfo>>
      *received_entries;
};

class FakePlugin : public telemetry::TelemetryManager {
public:
  FakePlugin()
      : telemetry::TelemetryManager(std::make_unique<telemetry::LLDBConfig>(
            /*enable_telemetry=*/true, /*detailed_command_telemetry=*/true)) {}

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

using namespace lldb_private::telemetry;

class TelemetryTest : public testing::Test {
public:
  lldb_private::SubsystemRAII<lldb_private::FakePlugin> subsystems;
  std::vector<std::unique_ptr<::llvm::telemetry::TelemetryInfo>>
      received_entries;

  void SetUp() override {
    auto *ins = lldb_private::telemetry::TelemetryManager::GetInstance();
    ASSERT_NE(ins, nullptr);

    ins->addDestination(
        std::make_unique<lldb_private::TestDestination>(&received_entries));
  }
};

#if LLVM_ENABLE_TELEMETRY
#define TELEMETRY_TEST(suite, test) TEST_F(suite, test)
#else
#define TELEMETRY_TEST(suite, test) TEST(DISABLED_##suite, test)
#endif

TELEMETRY_TEST(TelemetryTest, PluginTest) {
  lldb_private::telemetry::TelemetryManager *ins =
      lldb_private::telemetry::TelemetryManager::GetInstance();

  lldb_private::FakeTelemetryInfo entry;
  entry.msg = "";

  ASSERT_THAT_ERROR(ins->dispatch(&entry), ::llvm::Succeeded());
  ASSERT_EQ(1U, received_entries.size());
  EXPECT_EQ("In FakePlugin",
            llvm::dyn_cast<lldb_private::FakeTelemetryInfo>(received_entries[0])
                ->msg);

  ASSERT_EQ("FakeTelemetryPlugin", ins->GetInstanceName());
}

TELEMETRY_TEST(TelemetryTest, ScopedDispatcherTest) {
  {
    ScopedDispatcher<lldb_private::FakeTelemetryInfo> helper(
        [](lldb_private::FakeTelemetryInfo *info) { info->num = 0; });
  }

  {
    ScopedDispatcher<lldb_private::FakeTelemetryInfo> helper(
        [](lldb_private::FakeTelemetryInfo *info) { info->num = 1; });
  }

  {
    ScopedDispatcher<lldb_private::FakeTelemetryInfo> helper(
        [](lldb_private::FakeTelemetryInfo *info) { info->num = 2; });
  }

  EXPECT_EQ(3U, received_entries.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(
        i, llvm::dyn_cast<lldb_private::FakeTelemetryInfo>(received_entries[i])
               ->num);
  }
}
