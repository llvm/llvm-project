//===-- ScriptedPythonInterface.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"
#include "ScriptedPythonInterface.h"

using namespace lldb;
using namespace lldb_private;

ScriptedPythonInterface::ScriptedPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedInterface(), m_interpreter(interpreter) {}

Status
ScriptedPythonInterface::GetStatusFromMethod(llvm::StringRef method_name) {
  Status error;
  Dispatch<Status>(method_name, error);

  return error;
}

template <>
Status ScriptedPythonInterface::ExtractValueFromPythonObject<Status>(
    python::PythonObject &p, Status &error) {
  if (lldb::SBError *sb_error = reinterpret_cast<lldb::SBError *>(
          LLDBSWIGPython_CastPyObjectToSBError(p.get())))
    error = m_interpreter.GetStatusFromSBError(*sb_error);
  else
    error.SetErrorString("Couldn't cast lldb::SBError to lldb::Status.");

  return error;
}

template <>
lldb::DataExtractorSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DataExtractorSP>(
    python::PythonObject &p, Status &error) {
  lldb::SBData *sb_data = reinterpret_cast<lldb::SBData *>(
      LLDBSWIGPython_CastPyObjectToSBData(p.get()));

  if (!sb_data) {
    error.SetErrorString("Couldn't cast lldb::SBError to lldb::Status.");
    return nullptr;
  }

  return m_interpreter.GetDataExtractorFromSBData(*sb_data);
}

#endif
