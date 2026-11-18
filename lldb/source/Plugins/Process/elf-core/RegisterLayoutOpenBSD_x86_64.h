//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_X86_64_H
#define LLDB_PLUGINS_PROCESS_ELF_CORE_REGISTERLAYOUTOPENBSD_X86_64_H

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutOpenBSD_x86_64 final : public RegisterLayout_x86 {
public:
  explicit RegisterLayoutOpenBSD_x86_64(
      const ArchSpec &arch, RegisterLayout_x86::TagWordView tag_word =
                                RegisterLayout_x86::TagWordView::Full);

  RegisterLayoutOpenBSD_x86_64() = delete;
  RegisterLayoutOpenBSD_x86_64(const RegisterLayoutOpenBSD_x86_64 &) = delete;
  RegisterLayoutOpenBSD_x86_64(RegisterLayoutOpenBSD_x86_64 &&) = delete;
};

} // namespace lldb_private
#endif
