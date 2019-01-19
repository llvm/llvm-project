//===-- cli-wrapper-pt.h----------------------------------*- C++ -*-==========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CLI Wrapper of PTDecoder Tool to enable it to be used through LLDB's CLI.
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

bool PTPluginInitialize(lldb::SBDebugger &debugger);
