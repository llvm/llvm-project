//===-- WindowsMiniDump.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_WindowsMiniDump_h_
#define liblldb_WindowsMiniDump_h_

#include "lldb/Target/Process.h"

namespace lldb_private {

bool
SaveMiniDump(const lldb::ProcessSP &process_sp,
             const lldb_private::FileSpec &outfile,
             lldb_private::Error &error);

}  // namespace lldb_private

#endif
