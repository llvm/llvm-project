//===-- cli-wrapper-pt.h----------------------------------*- C++ -*-==========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// CLI Wrapper of PTDecoder Tool to enable it to be used through LLDB's CLI.
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

bool PTPluginInitialize(lldb::SBDebugger &debugger);
