//===-- RegisterContextOpenBSD_i386.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextOpenBSD_i386_H_
#define liblldb_RegisterContextOpenBSD_i386_H_

#include "RegisterInfoInterface.h"

class RegisterContextOpenBSD_i386 : public lldb_private::RegisterInfoInterface {
public:
  RegisterContextOpenBSD_i386(const lldb_private::ArchSpec &target_arch);

  size_t GetGPRSize() const override;

  const lldb_private::RegisterInfo *GetRegisterInfo() const override;

  uint32_t GetRegisterCount() const override;
};

#endif
