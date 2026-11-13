//===-- RegisterContextWindows_x86.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_x86_H_
#define liblldb_RegisterContextWindows_x86_H_

#if defined(__i386__) || defined(_M_IX86)

#include "RegisterContextWindows.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows_x86 : public RegisterContextWindows {
public:
  // Constructors and Destructors
  RegisterContextWindows_x86(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows_x86();

  // Subclasses must override these functions

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

private:
  bool ReadRegisterHelper(DWORD flags_required, const char *reg_name,
                          DWORD value, RegisterValue &reg_value) const;
};
}

#endif // defined(__i386__) || defined(_M_IX86)

#endif // #ifndef liblldb_RegisterContextWindows_x86_H_
