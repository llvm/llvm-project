//===-- ScriptedPythonInterface.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "API/ScriptInterpreterBridge.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedPythonInterface.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/ValueObject/ValueObjectList.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;

ScriptedPythonInterface::ScriptedPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedInterface(), m_interpreter(interpreter) {}

template <>
llvm::Expected<StructuredData::ArraySP>
ScriptedPythonInterface::ExtractValueFromPythonObject<StructuredData::ArraySP>(
    python::PythonObject &p) {
  python::PythonList result_list(python::PyRefType::Borrowed, p.get());
  return result_list.CreateStructuredArray();
}

template <>
llvm::Expected<StructuredData::DictionarySP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    StructuredData::DictionarySP>(python::PythonObject &p) {
  python::PythonDictionary result_dict(python::PyRefType::Borrowed, p.get());
  return result_dict.CreateStructuredDictionary();
}

template <>
llvm::Expected<Status>
ScriptedPythonInterface::ExtractValueFromPythonObject<Status>(
    python::PythonObject &p) {
  if (lldb::SBError *sb_error = reinterpret_cast<lldb::SBError *>(
          python::LLDBSWIGPython_CastPyObjectToSBError(p.get())))
    return ScriptInterpreterBridge::GetStatus(*sb_error);
  return llvm::createStringError("couldn't cast lldb::SBError to lldb::Status");
}

template <>
llvm::Expected<Event *>
ScriptedPythonInterface::ExtractValueFromPythonObject<Event *>(
    python::PythonObject &p) {
  if (lldb::SBEvent *sb_event = reinterpret_cast<lldb::SBEvent *>(
          python::LLDBSWIGPython_CastPyObjectToSBEvent(p.get())))
    return ScriptInterpreterBridge::GetEvent(*sb_event);
  return llvm::createStringError(
      "couldn't cast lldb::SBEvent to lldb_private::Event");
}

template <>
llvm::Expected<CommandReturnObject *>
ScriptedPythonInterface::ExtractValueFromPythonObject<CommandReturnObject *>(
    python::PythonObject &p) {
  if (lldb::SBCommandReturnObject *sb_cmd_retobj =
          reinterpret_cast<lldb::SBCommandReturnObject *>(
              python::LLDBSWIGPython_CastPyObjectToSBCommandReturnObject(
                  p.get())))
    return ScriptInterpreterBridge::GetCommandReturnObject(*sb_cmd_retobj);
  return llvm::createStringError("couldn't cast lldb::SBCommandReturnObject to "
                                 "lldb_private::CommandReturnObject");
}

template <>
llvm::Expected<lldb::StreamSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StreamSP>(
    python::PythonObject &p) {
  if (lldb::SBStream *sb_stream = reinterpret_cast<lldb::SBStream *>(
          python::LLDBSWIGPython_CastPyObjectToSBStream(p.get())))
    return ScriptInterpreterBridge::GetStream(*sb_stream);
  return llvm::createStringError(
      "couldn't cast lldb::SBStream to lldb_private::Stream");
}

template <>
llvm::Expected<lldb::StackFrameSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StackFrameSP>(
    python::PythonObject &p) {
  if (lldb::SBFrame *sb_frame = reinterpret_cast<lldb::SBFrame *>(
          python::LLDBSWIGPython_CastPyObjectToSBFrame(p.get())))
    return ScriptInterpreterBridge::GetStackFrame(*sb_frame);
  return llvm::createStringError(
      "couldn't cast lldb::SBFrame to lldb_private::StackFrame");
}

template <>
llvm::Expected<lldb::ThreadSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ThreadSP>(
    python::PythonObject &p) {
  if (lldb::SBThread *sb_thread = reinterpret_cast<lldb::SBThread *>(
          python::LLDBSWIGPython_CastPyObjectToSBThread(p.get())))
    return ScriptInterpreterBridge::GetThread(*sb_thread);
  return llvm::createStringError(
      "couldn't cast lldb::SBThread to lldb_private::Thread");
}

template <>
llvm::Expected<SymbolContext>
ScriptedPythonInterface::ExtractValueFromPythonObject<SymbolContext>(
    python::PythonObject &p) {
  if (lldb::SBSymbolContext *sb_symbol_context =
          reinterpret_cast<lldb::SBSymbolContext *>(
              python::LLDBSWIGPython_CastPyObjectToSBSymbolContext(p.get())))
    return ScriptInterpreterBridge::GetSymbolContext(*sb_symbol_context);
  return llvm::createStringError(
      "couldn't cast lldb::SBSymbolContext to lldb_private::SymbolContext");
}

template <>
llvm::Expected<lldb::DataExtractorSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DataExtractorSP>(
    python::PythonObject &p) {
  lldb::SBData *sb_data = reinterpret_cast<lldb::SBData *>(
      python::LLDBSWIGPython_CastPyObjectToSBData(p.get()));

  if (!sb_data) {
    return llvm::createStringError(
        "couldn't cast lldb::SBData to lldb::DataExtractorSP");
  }

  return ScriptInterpreterBridge::GetDataExtractor(*sb_data);
}

template <>
llvm::Expected<lldb::BreakpointSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::BreakpointSP>(
    python::PythonObject &p) {
  lldb::SBBreakpoint *sb_breakpoint = reinterpret_cast<lldb::SBBreakpoint *>(
      python::LLDBSWIGPython_CastPyObjectToSBBreakpoint(p.get()));

  if (!sb_breakpoint) {
    return llvm::createStringError(
        "couldn't cast lldb::SBBreakpoint to lldb::BreakpointSP");
  }

  return ScriptInterpreterBridge::GetBreakpoint(*sb_breakpoint);
}

template <>
llvm::Expected<lldb::BreakpointLocationSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::BreakpointLocationSP>(python::PythonObject &p) {
  lldb::SBBreakpointLocation *sb_break_loc =
      reinterpret_cast<lldb::SBBreakpointLocation *>(
          python::LLDBSWIGPython_CastPyObjectToSBBreakpointLocation(p.get()));

  if (!sb_break_loc) {
    return llvm::createStringError(
        "couldn't cast lldb::SBBreakpointLocation to "
        "lldb::BreakpointLocationSP");
  }

  return ScriptInterpreterBridge::GetBreakpointLocation(*sb_break_loc);
}

template <>
llvm::Expected<lldb::ProcessAttachInfoSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessAttachInfoSP>(python::PythonObject &p) {
  lldb::SBAttachInfo *sb_attach_info = reinterpret_cast<lldb::SBAttachInfo *>(
      python::LLDBSWIGPython_CastPyObjectToSBAttachInfo(p.get()));

  if (!sb_attach_info) {
    return llvm::createStringError(
        "couldn't cast lldb::SBAttachInfo to lldb::ProcessAttachInfoSP");
  }

  return ScriptInterpreterBridge::GetProcessAttachInfo(*sb_attach_info);
}

template <>
llvm::Expected<lldb::ProcessLaunchInfoSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessLaunchInfoSP>(python::PythonObject &p) {
  lldb::SBLaunchInfo *sb_launch_info = reinterpret_cast<lldb::SBLaunchInfo *>(
      python::LLDBSWIGPython_CastPyObjectToSBLaunchInfo(p.get()));

  if (!sb_launch_info) {
    return llvm::createStringError(
        "couldn't cast lldb::SBLaunchInfo to lldb::ProcessLaunchInfoSP");
  }

  return ScriptInterpreterBridge::GetProcessLaunchInfo(*sb_launch_info);
}

template <>
llvm::Expected<lldb::ThreadPlanSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ThreadPlanSP>(
    python::PythonObject &p) {
  lldb::SBThreadPlan *sb_thread_plan = reinterpret_cast<lldb::SBThreadPlan *>(
      python::LLDBSWIGPython_CastPyObjectToSBThreadPlan(p.get()));

  if (!sb_thread_plan) {
    return llvm::createStringError(
        "couldn't cast lldb::SBThreadPlan to lldb::ThreadPlanSP");
  }

  return ScriptInterpreterBridge::GetThreadPlan(*sb_thread_plan);
}

template <>
llvm::Expected<std::optional<MemoryRegionInfo>>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    std::optional<MemoryRegionInfo>>(python::PythonObject &p) {

  lldb::SBMemoryRegionInfo *sb_mem_reg_info =
      reinterpret_cast<lldb::SBMemoryRegionInfo *>(
          python::LLDBSWIGPython_CastPyObjectToSBMemoryRegionInfo(p.get()));

  if (!sb_mem_reg_info) {
    return llvm::createStringError("couldn't cast lldb::SBMemoryRegionInfo to "
                                   "lldb_private::MemoryRegionInfo");
  }

  return ScriptInterpreterBridge::GetMemoryRegionInfo(*sb_mem_reg_info);
}

template <>
llvm::Expected<lldb::ExecutionContextRefSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ExecutionContextRefSP>(python::PythonObject &p) {

  lldb::SBExecutionContext *sb_exe_ctx =
      reinterpret_cast<lldb::SBExecutionContext *>(
          python::LLDBSWIGPython_CastPyObjectToSBExecutionContext(p.get()));

  if (!sb_exe_ctx) {
    return llvm::createStringError("couldn't cast lldb::SBExecutionContext to "
                                   "lldb::ExecutionContextRefSP");
  }

  return ScriptInterpreterBridge::GetExecutionContextRef(*sb_exe_ctx);
}

template <>
llvm::Expected<lldb::DescriptionLevel>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DescriptionLevel>(
    python::PythonObject &p) {
  llvm::Expected<unsigned long long> unsigned_or_err = p.AsUnsignedLongLong();
  if (!unsigned_or_err)
    return unsigned_or_err.takeError();
  unsigned long long unsigned_val = *unsigned_or_err;
  if (unsigned_val >= lldb::DescriptionLevel::kNumDescriptionLevels)
    return llvm::createStringError(
        "value too large for lldb::DescriptionLevel");
  return static_cast<lldb::DescriptionLevel>(unsigned_val);
}

template <>
llvm::Expected<lldb::StepType>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StepType>(
    python::PythonObject &p) {
  llvm::Expected<unsigned long long> unsigned_or_err = p.AsUnsignedLongLong();
  if (!unsigned_or_err)
    return unsigned_or_err.takeError();
  unsigned long long unsigned_val = *unsigned_or_err;
  if (unsigned_val >= lldb::eStepTypeScripted)
    return llvm::createStringError("value too large for lldb::StepType");
  return static_cast<lldb::StepType>(unsigned_val);
}

template <>
llvm::Expected<lldb::StackFrameListSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StackFrameListSP>(
    python::PythonObject &p) {

  lldb::SBFrameList *sb_frame_list = reinterpret_cast<lldb::SBFrameList *>(
      python::LLDBSWIGPython_CastPyObjectToSBFrameList(p.get()));

  if (!sb_frame_list) {
    return llvm::createStringError(
        "couldn't cast lldb::SBFrameList to lldb::StackFrameListSP");
  }

  return ScriptInterpreterBridge::GetStackFrameList(*sb_frame_list);
}

template <>
llvm::Expected<lldb::ValueObjectSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ValueObjectSP>(
    python::PythonObject &p) {
  lldb::SBValue *sb_value = reinterpret_cast<lldb::SBValue *>(
      python::LLDBSWIGPython_CastPyObjectToSBValue(p.get()));
  if (!sb_value) {
    return llvm::createStringError(
        "couldn't cast lldb::SBValue to lldb::ValueObjectSP");
  }

  return ScriptInterpreterBridge::GetValueObject(*sb_value);
}

template <>
llvm::Expected<lldb::TargetSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::TargetSP>(
    python::PythonObject &p) {
  lldb::SBTarget *sb_target = reinterpret_cast<lldb::SBTarget *>(
      python::LLDBSWIGPython_CastPyObjectToSBTarget(p.get()));
  if (!sb_target) {
    return llvm::createStringError(
        "couldn't cast lldb::SBTarget to lldb::TargetSP");
  }

  return ScriptInterpreterBridge::GetTarget(*sb_target);
}

template <>
llvm::Expected<lldb::ValueObjectListSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ValueObjectListSP>(
    python::PythonObject &p) {
  // Two Python return shapes are accepted here so callers can go through
  // Dispatch<ValueObjectListSP>() uniformly: an `SBValueList` wrapper
  // (what most extension methods return) and a plain Python `list` of
  // `SBValue` (what `get_recognized_arguments` is documented to return).
  lldb::ValueObjectListSP out = std::make_shared<ValueObjectList>();
  if (auto *sb_value_list = reinterpret_cast<lldb::SBValueList *>(
          python::LLDBSWIGPython_CastPyObjectToSBValueList(p.get()))) {
    for (uint32_t i = 0, e = sb_value_list->GetSize(); i < e; ++i) {
      SBValue value = sb_value_list->GetValueAtIndex(i);
      out->Append(ScriptInterpreterBridge::GetValueObject(value));
    }
    return out;
  }

  // Fallback: a plain Python `list` of `SBValue`. Round-trip through
  // `CreateStructuredObject` so we don't touch the `PyList_*` C API
  // directly; unknown-shape items surface as `StructuredData::Generic`
  // holding their opaque `PyObject*`, which we hand back to SWIG to
  // recover the SBValue wrapper.
  StructuredData::ObjectSP structured = p.CreateStructuredObject();
  StructuredData::Array *arr = structured ? structured->GetAsArray() : nullptr;
  if (arr) {
    size_t index = 0;
    llvm::Error extract_error = llvm::Error::success();
    arr->ForEach([&](StructuredData::Object *item) {
      const size_t item_index = index++;
      StructuredData::Generic *generic = item ? item->GetAsGeneric() : nullptr;
      if (!generic) {
        // Keep walking the list so a malformed one names every bad item
        // rather than only the first.
        extract_error =
            llvm::joinErrors(std::move(extract_error),
                             llvm::createStringError(llvm::formatv(
                                 "ValueObjectList item at index {0} is not a "
                                 "StructuredData::Generic",
                                 item_index)));
        return true;
      }
      auto *sb_value = reinterpret_cast<lldb::SBValue *>(
          python::LLDBSWIGPython_CastPyObjectToSBValue(
              static_cast<PyObject *>(generic->GetValue())));
      if (sb_value)
        if (auto valobj_sp = ScriptInterpreterBridge::GetValueObject(*sb_value))
          out->Append(valobj_sp);
      return true;
    });
    if (extract_error)
      return std::move(extract_error);
    return out;
  }

  return llvm::createStringError(
      "couldn't extract ValueObjectList from Python return value");
}

template <>
llvm::Expected<std::optional<lldb::ValueType>>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    std::optional<lldb::ValueType>>(python::PythonObject &p) {
  if (p.IsNone())
    return std::nullopt;

  llvm::Expected<unsigned long long> val = p.AsUnsignedLongLong();
  if (!val)
    return val.takeError();
  unsigned long long unmasked = *val & ~kValueTypeFlagsMask;
  unsigned long long flags = *val & kValueTypeFlagsMask;
  if (unmasked == eValueTypeInvalid || unmasked > kLastValueType)
    return llvm::createStringError(llvm::formatv(
        "value type invalid or too large (got {0} | {1:x})", unmasked, flags));

  return static_cast<ValueType>(unmasked | flags);
}

template <>
llvm::Expected<lldb::DebuggerSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DebuggerSP>(
    python::PythonObject &p) {
  if (lldb::SBDebugger *sb_dbg = reinterpret_cast<lldb::SBDebugger *>(
          python::LLDBSWIGPython_CastPyObjectToSBDebugger(p.get())))
    return ScriptInterpreterBridge::GetDebugger(*sb_dbg);
  return llvm::createStringError(
      "couldn't cast lldb::SBDebugger to lldb::DebuggerSP");
}

template <>
llvm::Expected<std::vector<std::string>>
ScriptedPythonInterface::ExtractValueFromPythonObject<std::vector<std::string>>(
    python::PythonObject &p) {
  std::vector<std::string> result;
  python::PythonList list(python::PyRefType::Borrowed, p.get());
  if (!list.IsValid()) {
    return llvm::createStringError(
        "couldn't extract std::vector<std::string>: not a Python list");
  }

  const uint32_t size = list.GetSize();
  result.reserve(size);
  for (uint32_t i = 0; i < size; ++i) {
    python::PythonString item(python::PyRefType::Borrowed,
                              list.GetItemAtIndex(i).get());
    if (!item.IsValid())
      continue;
    result.push_back(item.GetString().str());
  }
  return result;
}
