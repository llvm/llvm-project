//===-- lldb-expression-fuzzer.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// \file
// This file is a fuzzer for LLDB's expression evaluator. It uses protobufs
// and the libprotobuf-mutator to create valid C-like inputs for the
// expression evaluator.
//
//===---------------------------------------------------------------------===//

#include <string>

#include "cxx_proto.pb.h"
#include "handle_cxx.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBLaunchInfo.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "proto_to_cxx.h"
#include "src/libfuzzer/libfuzzer_macro.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace lldb;
using namespace llvm;
using namespace clang_fuzzer;

char **originalargv;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  SBDebugger::Initialize();

  // The path for a simple compiled program is needed to create a
  // target for the debugger and that path is passed in through argv
  originalargv = *argv;
  return 0;
}

DEFINE_BINARY_PROTO_FUZZER(const clang_fuzzer::Function &input) {
  auto input_string = clang_fuzzer::FunctionToString(input);

  // Get the second argument from argv and strip the '--' from it.
  // This will be used as the path for the object file to create a target from
  std::string raw_path = originalargv[2];
  StringRef obj_path = raw_path.erase(0, 2);

  // Create a debugger and a target
  SBDebugger debugger = SBDebugger::Create(false);
  SBTarget target = debugger.CreateTarget(obj_path.str().c_str());

  // Create a breakpoint on the only line in the program
  SBBreakpoint breakpoint = target.BreakpointCreateByLocation(obj_path.str().c_str(), 1);

  // Create launch info and error for launching the process
  SBLaunchInfo launch_info = target.GetLaunchInfo();
  SBError error;

  // Launch the process and evaluate the fuzzer's input data
  // as an expression
  SBProcess process = target.Launch(launch_info, error);
  target.EvaluateExpression(input_string.c_str());

  debugger.DeleteTarget(target);
  SBDebugger::Destroy(debugger);
  SBModule::GarbageCollectAllocatedModules();
}
