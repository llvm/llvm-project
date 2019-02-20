//===-- PythonExceptionState.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONEXCEPTIONSTATE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONEXCEPTIONSTATE_H

#ifndef LLDB_DISABLE_PYTHON

#include "PythonDataObjects.h"

namespace lldb_private {

class PythonExceptionState {
public:
  explicit PythonExceptionState(bool restore_on_exit);
  ~PythonExceptionState();

  void Acquire(bool restore_on_exit);

  void Restore();

  void Discard();

  void Reset();

  static bool HasErrorOccurred();

  bool IsError() const;

  PythonObject GetType() const;

  PythonObject GetValue() const;

  PythonObject GetTraceback() const;

  std::string Format() const;

private:
  std::string ReadBacktrace() const;

  bool m_restore_on_exit;

  PythonObject m_type;
  PythonObject m_value;
  PythonObject m_traceback;
};
}

#endif

#endif
