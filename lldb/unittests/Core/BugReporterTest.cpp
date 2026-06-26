//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/BugReporter.h"
#include "lldb/Core/PluginManager.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class AppliesReporter : public BugReporter {
public:
  llvm::StringRef GetPluginName() override { return "applies"; }
  llvm::Error File(const Diagnostics::Report &) override {
    return llvm::Error::success();
  }
  static std::unique_ptr<BugReporter> Create() {
    return std::make_unique<AppliesReporter>();
  }
};

class DeclinesReporter : public BugReporter {
public:
  llvm::StringRef GetPluginName() override { return "declines"; }
  llvm::Error File(const Diagnostics::Report &) override {
    return llvm::Error::success();
  }
  static std::unique_ptr<BugReporter> Create() { return nullptr; }
};
} // namespace

TEST(BugReporterPluginTest, PicksFirstThatApplies) {
  // Registration order is priority: declines registers first but returns null,
  // so applies wins.
  PluginManager::RegisterPlugin("declines", "", &DeclinesReporter::Create);
  PluginManager::RegisterPlugin("applies", "", &AppliesReporter::Create);

  std::unique_ptr<BugReporter> reporter =
      PluginManager::CreateBugReporterInstance();
  ASSERT_NE(reporter, nullptr);
  EXPECT_EQ(reporter->GetPluginName(), "applies");

  PluginManager::UnregisterPlugin(&AppliesReporter::Create);
  PluginManager::UnregisterPlugin(&DeclinesReporter::Create);
}

TEST(BugReporterPluginTest, NullWhenAllDecline) {
  PluginManager::RegisterPlugin("declines", "", &DeclinesReporter::Create);

  EXPECT_EQ(PluginManager::CreateBugReporterInstance(), nullptr);

  PluginManager::UnregisterPlugin(&DeclinesReporter::Create);
}
