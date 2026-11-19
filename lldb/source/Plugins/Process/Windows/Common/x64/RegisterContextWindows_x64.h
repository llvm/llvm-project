//===-- RegisterContextWindows_x64.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_x64_H_
#define liblldb_RegisterContextWindows_x64_H_

#if defined(__x86_64__) || defined(_M_X64)

#include "RegisterContextWindows.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows_x64 : public RegisterContextWindows {
public:
  // Constructors and Destructors
  RegisterContextWindows_x64(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows_x64();

  // Subclasses must override these functions

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;
};
} // namespace lldb_private

#endif // defined(__x86_64__) || defined(_M_X64)

#endif // #ifndef liblldb_RegisterContextWindows_x64_H_
