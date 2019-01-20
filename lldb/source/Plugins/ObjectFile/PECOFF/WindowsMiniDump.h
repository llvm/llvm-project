//===-- WindowsMiniDump.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_WindowsMiniDump_h_
#define liblldb_WindowsMiniDump_h_

#include "lldb/Target/Process.h"

namespace lldb_private {

bool SaveMiniDump(const lldb::ProcessSP &process_sp,
                  const lldb_private::FileSpec &outfile,
                  lldb_private::Status &error);

} // namespace lldb_private

#endif
