//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include <cstdlib>

using namespace llgs_tests;
using namespace llvm;

std::string TestBase::getLogFileName() {
  const auto *test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  assert(test_info);

  const char *Dir = getenv("LOG_FILE_DIRECTORY");
  if (!Dir)
    return "";

  if (!llvm::sys::fs::is_directory(Dir)) {
    GTEST_LOG_(WARNING) << "Cannot access log directory: " << Dir;
    return "";
  }

  SmallString<64> DirStr(Dir);
  sys::path::append(DirStr, std::string("server-") +
                                test_info->test_case_name() + "-" +
                                test_info->name() + ".log");
  return std::string(DirStr.str());
}

