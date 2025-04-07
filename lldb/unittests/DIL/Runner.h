//===-- Runner.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DIL_RUNNER_H_
#define LLDB_DIL_RUNNER_H_

#include <string>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"

lldb::SBProcess LaunchTestProgram(lldb::SBDebugger debugger,
                                  const std::string &source_path,
                                  const std::string &binary_path,
                                  const std::string &break_line);

#endif // LLDB_DIL_RUNNER_H_