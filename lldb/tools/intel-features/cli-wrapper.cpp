//===-- cli-wrapper.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// CLI Wrapper for hardware features of Intel(R) architecture based processors
// to enable them to be used through LLDB's CLI. For details, please refer to
// cli wrappers of each individual feature, residing in their respective
// folders.
//
// Compile this into a shared lib and load by placing at appropriate locations
// on disk or by using "plugin load" command at the LLDB command line.
//
//===----------------------------------------------------------------------===//

#ifdef BUILD_INTEL_MPX
#include "intel-mpx/cli-wrapper-mpxtable.h"
#endif

#ifdef BUILD_INTEL_PT
#include "intel-pt/cli-wrapper-pt.h"
#endif

#include "lldb/API/SBDebugger.h"

namespace lldb {
bool PluginInitialize(lldb::SBDebugger debugger);
}

bool lldb::PluginInitialize(lldb::SBDebugger debugger) {

#ifdef BUILD_INTEL_PT
  PTPluginInitialize(debugger);
#endif

#ifdef BUILD_INTEL_MPX
  MPXPluginInitialize(debugger);
#endif

  return true;
}
