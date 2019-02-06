//===-- PythonExceptionStateTest.cpp ------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ScriptInterpreter/Python/PythonDataObjects.h"
#include "Plugins/ScriptInterpreter/Python/PythonExceptionState.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"

#include "PythonTestSuite.h"

using namespace lldb_private;

class PythonExceptionStateTest : public PythonTestSuite {
public:
protected:
  void RaiseException() {
    PyErr_SetString(PyExc_RuntimeError, "PythonExceptionStateTest test error");
  }
};

TEST_F(PythonExceptionStateTest, TestExceptionStateChecking) {
  PyErr_Clear();
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  RaiseException();
  EXPECT_TRUE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}

TEST_F(PythonExceptionStateTest, TestAcquisitionSemantics) {
  PyErr_Clear();
  PythonExceptionState no_error(false);
  EXPECT_FALSE(no_error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
  RaiseException();
  PythonExceptionState error(false);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
  error.Discard();

  PyErr_Clear();
  RaiseException();
  error.Acquire(false);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}

TEST_F(PythonExceptionStateTest, TestDiscardSemantics) {
  PyErr_Clear();

  // Test that discarding an exception does not restore the exception
  // state even when auto-restore==true is set
  RaiseException();
  PythonExceptionState error(true);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  error.Discard();
  EXPECT_FALSE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
}

TEST_F(PythonExceptionStateTest, TestResetSemantics) {
  PyErr_Clear();

  // Resetting when auto-restore is true should restore.
  RaiseException();
  PythonExceptionState error(true);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
  error.Reset();
  EXPECT_FALSE(error.IsError());
  EXPECT_TRUE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();

  // Resetting when auto-restore is false should discard.
  RaiseException();
  PythonExceptionState error2(false);
  EXPECT_TRUE(error2.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
  error2.Reset();
  EXPECT_FALSE(error2.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}

TEST_F(PythonExceptionStateTest, TestManualRestoreSemantics) {
  PyErr_Clear();
  RaiseException();
  PythonExceptionState error(false);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  error.Restore();
  EXPECT_FALSE(error.IsError());
  EXPECT_TRUE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}

TEST_F(PythonExceptionStateTest, TestAutoRestoreSemantics) {
  PyErr_Clear();
  // Test that using the auto-restore flag correctly restores the exception
  // state on destruction, and not using the auto-restore flag correctly
  // does NOT restore the state on destruction.
  {
    RaiseException();
    PythonExceptionState error(false);
    EXPECT_TRUE(error.IsError());
    EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
  }
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
  {
    RaiseException();
    PythonExceptionState error(true);
    EXPECT_TRUE(error.IsError());
    EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());
  }
  EXPECT_TRUE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}

TEST_F(PythonExceptionStateTest, TestAutoRestoreChanged) {
  // Test that if we re-acquire with different auto-restore semantics,
  // that the new semantics are respected.
  PyErr_Clear();

  RaiseException();
  PythonExceptionState error(false);
  EXPECT_TRUE(error.IsError());

  error.Reset();
  EXPECT_FALSE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  RaiseException();
  error.Acquire(true);
  EXPECT_TRUE(error.IsError());
  EXPECT_FALSE(PythonExceptionState::HasErrorOccurred());

  error.Reset();
  EXPECT_FALSE(error.IsError());
  EXPECT_TRUE(PythonExceptionState::HasErrorOccurred());

  PyErr_Clear();
}
