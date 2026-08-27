//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBTestUtilities.h"

#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

bool lldb_private::DebuggerSupportsLLVMTarget(llvm::StringRef target) {
  lldb::SBStructuredData data = lldb::SBDebugger::GetBuildConfiguration()
                                    .GetValueForKey("targets")
                                    .GetValueForKey("value");
  for (size_t i = 0; i < data.GetSize(); i++) {
    char buf[100] = {0};
    size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
    if (llvm::StringRef(buf, size) == target)
      return true;
  }

  return false;
}

std::pair<lldb::SBTarget, lldb::SBProcess>
lldb_private::LoadCore(lldb::SBDebugger &debugger, llvm::StringRef binary_path,
                       llvm::StringRef core_path) {
  EXPECT_TRUE(debugger);

  llvm::Expected<lldb_private::TestFile> binary_yaml =
      lldb_private::TestFile::fromYamlFile(binary_path);
  EXPECT_THAT_EXPECTED(binary_yaml, llvm::Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> binary_file =
      binary_yaml->writeToTemporaryFile();
  EXPECT_THAT_EXPECTED(binary_file, llvm::Succeeded());
  lldb::SBError error;
  lldb::SBTarget target = debugger.CreateTarget(
      /*filename=*/binary_file->TmpName.data(), /*target_triple=*/"",
      /*platform_name=*/"", /*add_dependent_modules=*/false, /*error=*/error);
  EXPECT_TRUE(target);
  EXPECT_TRUE(error.Success()) << error.GetCString();
  debugger.SetSelectedTarget(target);

  llvm::Expected<lldb_private::TestFile> core_yaml =
      lldb_private::TestFile::fromYamlFile(core_path);
  EXPECT_THAT_EXPECTED(core_yaml, llvm::Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> core_file =
      core_yaml->writeToTemporaryFile();
  EXPECT_THAT_EXPECTED(core_file, llvm::Succeeded());
  lldb::SBProcess process = target.LoadCore(core_file->TmpName.data());
  EXPECT_TRUE(process);

  EXPECT_THAT_ERROR(binary_file->discard(), llvm::Succeeded());
  EXPECT_THAT_ERROR(core_file->discard(), llvm::Succeeded());

  return std::make_pair(target, process);
}
