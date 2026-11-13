//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_I386_H
#define LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_I386_H

#include "lldb/Register/Common/RegisterInfo_x86.h"

namespace lldb_private {
RegisterLayout CreateOpenBSDI386RegisterLayout(const ArchSpec &arch);
} // namespace lldb_private
#endif
