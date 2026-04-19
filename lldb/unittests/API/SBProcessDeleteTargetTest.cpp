//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "TestingSupport/SubsystemRAII.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {

class SBProcessDeleteTargetTest : public testing::Test {
protected:
  void SetUp() override {
    debugger = SBDebugger::Create(/*source_init_files=*/false);
  }

  void TearDown() override { SBDebugger::Destroy(debugger); }

  static bool DebuggerSupportsLLVMTarget(llvm::StringRef target) {
    SBStructuredData data = SBDebugger::GetBuildConfiguration()
                                .GetValueForKey("targets")
                                .GetValueForKey("value");
    for (size_t i = 0; i < data.GetSize(); ++i) {
      char buf[100] = {0};
      size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
      if (llvm::StringRef(buf, size) == target)
        return true;
    }
    return false;
  }

  static std::string GetCoreFixturePath(llvm::StringRef name) {
    llvm::SmallString<256> path(__FILE__);
    llvm::sys::path::remove_filename(path);
    llvm::sys::path::append(path, "..", "..", "test", "API");
    llvm::sys::path::append(path, "tools", "lldb-dap", "coreFile", name);
    return std::string(path.str());
  }

  void LoadCore(SBTarget &target, SBProcess &process) {
    std::string binary_path = GetCoreFixturePath("linux-x86_64.out");
    std::string core_path = GetCoreFixturePath("linux-x86_64.core");
    ASSERT_TRUE(llvm::sys::fs::exists(binary_path));
    ASSERT_TRUE(llvm::sys::fs::exists(core_path));

    SBError error;
    target = debugger.CreateTarget(binary_path.c_str(), /*target_triple=*/"",
                                   /*platform_name=*/"",
                                   /*add_dependent_modules=*/false, error);
    EXPECT_TRUE(target);
    EXPECT_TRUE(error.Success()) << error.GetCString();
    debugger.SetSelectedTarget(target);

    process = target.LoadCore(core_path.c_str());
    EXPECT_TRUE(process);
  }

  SubsystemRAII<lldb::SBDebugger> subsystems;
  SBDebugger debugger;
};

} // namespace

TEST_F(SBProcessDeleteTargetTest, GetTargetAndSaveCoreFailSafelyAfterDelete) {
  if (!DebuggerSupportsLLVMTarget("X86"))
    GTEST_SKIP() << "X86 target is not enabled";

  SBTarget target;
  SBProcess process;
  LoadCore(target, process);
  ASSERT_TRUE(target.IsValid());
  ASSERT_TRUE(process.IsValid());
  ASSERT_EQ(process.GetState(), eStateStopped);

  SBSaveCoreOptions options;
  options.SetProcess(process);

  ASSERT_TRUE(debugger.DeleteTarget(target));
  ASSERT_FALSE(target.IsValid());
  EXPECT_FALSE(process.IsValid());

  EXPECT_FALSE(process.GetTarget().IsValid());

  SBError error = process.SaveCore(options);
  EXPECT_TRUE(error.Fail());
  EXPECT_STREQ("SBProcess is invalid because its target has been deleted",
               error.GetCString());
}
