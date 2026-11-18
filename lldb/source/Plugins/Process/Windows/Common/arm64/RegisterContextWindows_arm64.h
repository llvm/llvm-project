//===-- RegisterContextWindows_arm64.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_arm64_H_
#define liblldb_RegisterContextWindows_arm64_H_

#if defined(__aarch64__) || defined(_M_ARM64)

#include "RegisterContextWindows.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows_arm64 : public RegisterContextWindows {
public:
  // Constructors and Destructors
  RegisterContextWindows_arm64(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows_arm64();

  // Subclasses must override these functions

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;
};
} // namespace lldb_private

#endif // defined(__aarch64__) || defined(_M_ARM64)

#endif // #ifndef liblldb_RegisterContextWindows_arm64_H_
