//===-- PythonTestSuite.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"

#include "PythonTestSuite.h"

using namespace lldb_private;

void PythonTestSuite::SetUp() {
  FileSystem::Initialize();
  HostInfoBase::Initialize();
  // ScriptInterpreterPython::Initialize() depends on HostInfo being
  // initializedso it can compute the python directory etc.
  ScriptInterpreterPython::Initialize();
  ScriptInterpreterPython::InitializePrivate();

  // Although we don't care about concurrency for the purposes of running
  // this test suite, Python requires the GIL to be locked even for
  // deallocating memory, which can happen when you call Py_DECREF or
  // Py_INCREF.  So acquire the GIL for the entire duration of this
  // test suite.
  m_gil_state = PyGILState_Ensure();
}

void PythonTestSuite::TearDown() {
  PyGILState_Release(m_gil_state);

  ScriptInterpreterPython::Terminate();
  HostInfoBase::Terminate();
  FileSystem::Terminate();
}
