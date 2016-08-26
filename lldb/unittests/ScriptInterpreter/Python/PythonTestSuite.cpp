//===-- PythonTestSuite.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"

#include "lldb/Host/HostInfo.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"

#include "PythonTestSuite.h"

using namespace lldb_private;

void
PythonTestSuite::SetUp()
{
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

void
PythonTestSuite::TearDown()
{
    PyGILState_Release(m_gil_state);

    ScriptInterpreterPython::Terminate();
}
