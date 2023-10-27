//===-- LLDBUtils.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBUtils.h"
#include "VSCode.h"

namespace lldb_vscode {

void RunLLDBCommands(llvm::StringRef prefix,
                     const llvm::ArrayRef<std::string> &commands,
                     llvm::raw_ostream &strm) {
  if (commands.empty())
    return;
  lldb::SBCommandInterpreter interp = g_vsc.debugger.GetCommandInterpreter();
  if (!prefix.empty())
    strm << prefix << "\n";
  for (const auto &command : commands) {
    lldb::SBCommandReturnObject result;
    strm << "(lldb) " << command << "\n";
    interp.HandleCommand(command.c_str(), result);
    auto output_len = result.GetOutputSize();
    if (output_len) {
      const char *output = result.GetOutput();
      strm << output;
    }
    auto error_len = result.GetErrorSize();
    if (error_len) {
      const char *error = result.GetError();
      strm << error;
    }
  }
}

std::string RunLLDBCommands(llvm::StringRef prefix,
                            const llvm::ArrayRef<std::string> &commands) {
  std::string s;
  llvm::raw_string_ostream strm(s);
  RunLLDBCommands(prefix, commands, strm);
  strm.flush();
  return s;
}

bool ThreadHasStopReason(lldb::SBThread &thread) {
  switch (thread.GetStopReason()) {
  case lldb::eStopReasonTrace:
  case lldb::eStopReasonPlanComplete:
  case lldb::eStopReasonBreakpoint:
  case lldb::eStopReasonWatchpoint:
  case lldb::eStopReasonInstrumentation:
  case lldb::eStopReasonSignal:
  case lldb::eStopReasonException:
  case lldb::eStopReasonExec:
  case lldb::eStopReasonProcessorTrace:
  case lldb::eStopReasonFork:
  case lldb::eStopReasonVFork:
  case lldb::eStopReasonVForkDone:
    return true;
  case lldb::eStopReasonThreadExiting:
  case lldb::eStopReasonInvalid:
  case lldb::eStopReasonNone:
    break;
  }
  return false;
}

static uint32_t constexpr THREAD_INDEX_SHIFT = 19;

uint32_t GetLLDBThreadIndexID(uint64_t dap_frame_id) {
  return dap_frame_id >> THREAD_INDEX_SHIFT;
}

uint32_t GetLLDBFrameID(uint64_t dap_frame_id) {
  return dap_frame_id & ((1u << THREAD_INDEX_SHIFT) - 1);
}

int64_t MakeVSCodeFrameID(lldb::SBFrame &frame) {
  return ((int64_t)frame.GetThread().GetIndexID() << THREAD_INDEX_SHIFT) |
         frame.GetFrameID();
}

} // namespace lldb_vscode
