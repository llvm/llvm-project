//===-- PythonTestSuite.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
