//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H

#include <string>

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

// GetPythonValueFormatString provides a system independent type safe way to
// convert a variable's type into a python value format. Python value formats
// are defined in terms of builtin C types and could change from system to as
// the underlying typedef for uint* types, size_t, off_t and other values
// change.

template <typename T> const char *GetPythonValueFormatString(T t);
template <> const char *GetPythonValueFormatString(char *);
template <> const char *GetPythonValueFormatString(char);
template <> const char *GetPythonValueFormatString(unsigned char);
template <> const char *GetPythonValueFormatString(short);
template <> const char *GetPythonValueFormatString(unsigned short);
template <> const char *GetPythonValueFormatString(int);
template <> const char *GetPythonValueFormatString(unsigned int);
template <> const char *GetPythonValueFormatString(long);
template <> const char *GetPythonValueFormatString(unsigned long);
template <> const char *GetPythonValueFormatString(long long);
template <> const char *GetPythonValueFormatString(unsigned long long);
template <> const char *GetPythonValueFormatString(float t);
template <> const char *GetPythonValueFormatString(double t);

extern "C" void *LLDBSwigPythonCreateScriptedProcess(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::TargetSP &target_sp, StructuredDataImpl *args_impl,
    std::string &error_string);

extern "C" void *LLDBSWIGPython_CastPyObjectToSBData(void *data);
extern "C" void *LLDBSWIGPython_CastPyObjectToSBError(void *data);
extern "C" void *LLDBSWIGPython_CastPyObjectToSBValue(void *data);

}; // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H
