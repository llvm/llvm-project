//===-- ScriptedPythonInterface.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
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
StructuredData::ArraySP
ScriptedPythonInterface::ExtractValueFromPythonObject<StructuredData::ArraySP>(
    python::PythonObject &p, Status &error) {
  python::PythonList result_list(python::PyRefType::Borrowed, p.get());
  return result_list.CreateStructuredArray();
}

template <>
StructuredData::DictionarySP
ScriptedPythonInterface::ExtractValueFromPythonObject<
    StructuredData::DictionarySP>(python::PythonObject &p, Status &error) {
  python::PythonDictionary result_dict(python::PyRefType::Borrowed, p.get());
  return result_dict.CreateStructuredDictionary();
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
    error.SetErrorString(
        "Couldn't cast lldb::SBData to lldb::DataExtractorSP.");
    return nullptr;
  }

  return m_interpreter.GetDataExtractorFromSBData(*sb_data);
}

template <>
llvm::Optional<MemoryRegionInfo>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    llvm::Optional<MemoryRegionInfo>>(python::PythonObject &p, Status &error) {

  lldb::SBMemoryRegionInfo *sb_mem_reg_info =
      reinterpret_cast<lldb::SBMemoryRegionInfo *>(
          LLDBSWIGPython_CastPyObjectToSBMemoryRegionInfo(p.get()));

  if (!sb_mem_reg_info) {
    error.SetErrorString(
        "Couldn't cast lldb::SBMemoryRegionInfo to lldb::MemoryRegionInfoSP.");
    return {};
  }

  return m_interpreter.GetOpaqueTypeFromSBMemoryRegionInfo(*sb_mem_reg_info);
}

#endif
