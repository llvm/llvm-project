//===-- PythonTestSuite.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

using namespace lldb_private;

class PythonTestSuite : public testing::Test {
public:
  void SetUp() override;

  void TearDown() override;

private:
  PyGILState_STATE m_gil_state;
};
