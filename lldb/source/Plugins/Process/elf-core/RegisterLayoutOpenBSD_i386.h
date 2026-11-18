//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_I386_H
#define LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_I386_H

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutOpenBSD_i386 final : public RegisterLayout_x86 {
public:
  explicit RegisterLayoutOpenBSD_i386(
      const ArchSpec &arch, RegisterLayout_x86::TagWordView tag_word =
                                RegisterLayout_x86::TagWordView::Full);

  RegisterLayoutOpenBSD_i386() = delete;
  RegisterLayoutOpenBSD_i386(const RegisterLayoutOpenBSD_i386 &) = delete;
  RegisterLayoutOpenBSD_i386(RegisterLayoutOpenBSD_i386 &&) = delete;
};

} // namespace lldb_private
#endif
