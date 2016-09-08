//===-- Xcode.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Xcode.h"
#include <string>

using namespace std;
using namespace lldb_perf;

void Xcode::FetchVariable(SBValue value, uint32_t expand, bool verbose) {
  auto name = value.GetName();
  auto num_value = value.GetValueAsUnsigned(0);
  auto summary = value.GetSummary();
  auto in_scope = value.IsInScope();
  auto has_children = value.MightHaveChildren();
  auto type_1 = value.GetType();
  auto type_2 = value.GetType();
  auto type_name_1 = value.GetTypeName();
  auto type_3 = value.GetType();
  auto type_name_2 = value.GetTypeName();
  if (verbose)
    printf("%s %s = 0x%llx (%llu) %s\n", value.GetTypeName(), value.GetName(),
           num_value, num_value, summary);
  if (expand > 0) {
    auto count = value.GetNumChildren();
    for (int i = 0; i < count; i++) {
      SBValue child(value.GetChildAtIndex(i, lldb::eDynamicCanRunTarget, true));
      FetchVariable(child, expand - 1, verbose);
    }
  }
}

void Xcode::FetchModules(SBTarget target, bool verbose) {
  auto count = target.GetNumModules();
  for (int i = 0; i < count; i++) {
    SBModule module(target.GetModuleAtIndex(i));
    auto fspec = module.GetFileSpec();
    std::string path(1024, 0);
    fspec.GetPath(&path[0], 1024);
    auto uuid = module.GetUUIDBytes();
    if (verbose) {
      printf("%s %s\n", path.c_str(), module.GetUUIDString());
    }
  }
}

void Xcode::FetchVariables(SBFrame frame, uint32_t expand, bool verbose) {
  auto values =
      frame.GetVariables(true, true, true, false, eDynamicCanRunTarget);
  auto count = values.GetSize();
  for (int i = 0; i < count; i++) {
    SBValue value(values.GetValueAtIndex(i));
    FetchVariable(value, expand, verbose);
  }
}

void Xcode::FetchFrames(SBProcess process, bool variables, bool verbose) {
  auto pCount = process.GetNumThreads();
  for (int p = 0; p < pCount; p++) {
    SBThread thread(process.GetThreadAtIndex(p));
    auto tCount = thread.GetNumFrames();
    if (verbose)
      printf("%s %d %d {%d}\n", thread.GetQueueName(), tCount,
             thread.GetStopReason(), eStopReasonBreakpoint);
    for (int t = 0; t < tCount; t++) {
      SBFrame frame(thread.GetFrameAtIndex(t));
      auto fp = frame.GetFP();
      SBThread thread_dup = frame.GetThread();
      SBFileSpec filespec(process.GetTarget().GetExecutable());
      std::string path(1024, 0);
      filespec.GetPath(&path[0], 1024);
      auto state = process.GetState();
      auto pCount_dup = process.GetNumThreads();
      auto byte_size = process.GetAddressByteSize();
      auto pc = frame.GetPC();
      SBSymbolContext context(frame.GetSymbolContext(0x0000006e));
      SBModule module(context.GetModule());
      SBLineEntry entry(context.GetLineEntry());
      SBFileSpec entry_filespec(process.GetTarget().GetExecutable());
      std::string entry_path(1024, 0);
      entry_filespec.GetPath(&entry_path[0], 1024);
      auto line_1 = entry.GetLine();
      auto line_2 = entry.GetLine();
      auto fname = frame.GetFunctionName();
      if (verbose)
        printf("%llu %s %d %d %llu %s %d %s\n", fp, path.c_str(), state,
               byte_size, pc, entry_path.c_str(), line_1, fname);
      if (variables)
        FetchVariables(frame, 0, verbose);
    }
  }
}

void Xcode::RunExpression(SBFrame frame, const char *expression, bool po,
                          bool verbose) {
  SBValue value(frame.EvaluateExpression(expression, eDynamicCanRunTarget));
  FetchVariable(value, 0, verbose);
  if (po) {
    auto descr = value.GetObjectDescription();
    if (descr)
      printf("po = %s\n", descr);
  }
}

void Xcode::Next(SBThread thread) { thread.StepOver(); }

void Xcode::Continue(SBProcess process) { process.Continue(); }

void Xcode::RunCommand(SBDebugger debugger, const char *cmd, bool verbose) {
  SBCommandReturnObject sb_ret;
  auto interpreter = debugger.GetCommandInterpreter();
  interpreter.HandleCommand(cmd, sb_ret);
  if (verbose)
    printf("%s\n%s\n", sb_ret.GetOutput(false), sb_ret.GetError(false));
}

SBThread Xcode::GetThreadWithStopReason(SBProcess process, StopReason reason) {
  auto threads_count = process.GetNumThreads();
  for (auto thread_num = 0; thread_num < threads_count; thread_num++) {
    SBThread thread(process.GetThreadAtIndex(thread_num));
    if (thread.GetStopReason() == reason) {
      return thread;
    }
  }
  return SBThread();
}

SBBreakpoint Xcode::CreateFileLineBreakpoint(SBTarget target, const char *file,
                                             uint32_t line) {
  return target.BreakpointCreateByLocation(file, line);
}
