//===-- TestBase.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  return DirStr.str();
}

