//===-- ScriptedFrameProviderPythonInterface.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "../lldb-python.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedFrameProviderPythonInterface.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedFrameProviderPythonInterface::ScriptedFrameProviderPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedFrameProviderInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedFrameProviderPythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, lldb::ThreadSP thread_sp,
    StructuredData::DictionarySP args_sp) {
  if (!thread_sp)
    return llvm::createStringError("Invalid thread");

  StructuredDataImpl sd_impl(args_sp);
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     thread_sp, sd_impl);
}

StructuredData::ObjectSP ScriptedFrameProviderPythonInterface::GetFrameAtIndex(
    lldb::StackFrameListSP real_frames, uint32_t index) {
  Status error;
  StructuredData::ObjectSP obj =
      Dispatch("get_frame_at_index", error, real_frames, index);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj;
}

#endif
