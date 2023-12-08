//===- llvm/unittest/Support/CommandLineInit/CommandLineInitTest.cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Check if preset options in libSupport -- e.g., "help", "version", etc. --
/// are correctly initialized and registered before getRegisteredOptions is
/// invoked.
///
/// Most code here comes from llvm/utils/unittest/UnitTestMain/TestMain.cpp,
/// except that llvm::cl::ParseCommandLineOptions() call is removed.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

#if defined(_WIN32)
#include <windows.h>
#if defined(_MSC_VER)
#include <crtdbg.h>
#endif
#endif

using namespace llvm;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

#if defined(_WIN32)
  // Disable all of the possible ways Windows conspires to make automated
  // testing impossible.
  ::SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#if defined(_MSC_VER)
  ::_set_error_mode(_OUT_TO_STDERR);
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
#endif

  return RUN_ALL_TESTS();
}

TEST(CommandLineInitTest, GetPresetOptions) {
  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(cl::SubCommand::getTopLevel());

  for (auto *Str :
       {"help", "help-hidden", "help-list", "help-list-hidden", "version"})
    EXPECT_EQ(Map.count(Str), (size_t)1)
        << "Could not get preset option `" << Str << '`';
}
