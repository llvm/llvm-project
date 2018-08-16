//===-- LLDBUtils.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    return true;
  case lldb::eStopReasonThreadExiting:
  case lldb::eStopReasonInvalid:
  case lldb::eStopReasonNone:
    break;
  }
  return false;
}

int64_t MakeVSCodeFrameID(lldb::SBFrame &frame) {
  return (int64_t)frame.GetThread().GetIndexID() << 32 |
         (int64_t)frame.GetFrameID();
}

} // namespace lldb_vscode
