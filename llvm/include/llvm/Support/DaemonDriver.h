//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the function `runWithDaemonSupport` to run a tool
/// which implements `LLVMTool` as a daemon, as described in
/// docs/DaemonDriver.rst.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DAEMONDRIVER_H
#define LLVM_SUPPORT_DAEMONDRIVER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ToolInterface.h"

namespace llvm {
/// This function checks for the `--daemon` command line option and, if it is
/// present, runs the given tool in daemon mode, as described in
/// `docs/DaemonMode.rst`. Otherwise, the tool is run
/// with standard input as the input source - a normal invocation. One-time
/// initialization logic, for example constructing `InitLLVM` and initializing
/// passes, should be performed before this function is called. Per-invocation
/// initialization logic, for example parsing command line options or setting up
/// the pass pipeline, should be performed inside of the tool's `run` function.
/// The `resetState` function is responsible for resetting all global state that
/// may affect the tool's output on the next invocation.
LLVM_ABI int runWithDaemonSupport(LLVMTool &Tool, int Argc, char **Argv);
} // namespace llvm

#endif // LLVM_SUPPORT_DAEMONDRIVER_H
