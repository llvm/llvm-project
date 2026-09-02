//===-- PythonReadline.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONREADLINE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONREADLINE_H

#include "lldb/Host/Config.h"

// No need to hack into Python's readline module if libedit isn't used.
#if LLDB_ENABLE_LIBEDIT && defined(__linux__)
// NOTE: Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included.
#include <Python.h>

// The symbol conflict bug was fixed in python 3.9 here
// https://github.com/python/cpython/issues/82815 commit
// https://github.com/python/cpython/commit/7105319ada2e663659020cbe9fdf7ff38f421ab2
// and backported to 3.8 point release (don't know the exact version).
// TODO: remove LLDB_USE_LIBEDIT_READLINE_COMPACT_MODULE when
// LLDB_MINIMUM_PYTHON_VERSION is greater than 3.8.
#if PY_VERSION_HEX < 0x03090000
#define LLDB_USE_LIBEDIT_READLINE_COMPAT_MODULE 1

PyMODINIT_FUNC initlldb_readline(void);
#endif // PY_VERSION_HEX < 0x03090000

#endif

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONREADLINE_H
