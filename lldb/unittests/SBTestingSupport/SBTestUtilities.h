//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_SBTESTINGSUPPORT_SBTESTUTILITIES_H
#define LLDB_UNITTESTS_SBTESTINGSUPPORT_SBTESTUTILITIES_H

#include "lldb/API/SBDebugger.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace lldb_private {

/// Check if the debugger supports the given platform.
bool DebuggerSupportsLLVMTarget(llvm::StringRef target);

std::pair<lldb::SBTarget, lldb::SBProcess> LoadCore(lldb::SBDebugger &debugger,
                                                    llvm::StringRef binary_path,
                                                    llvm::StringRef core_path);

} // namespace lldb_private

#endif
