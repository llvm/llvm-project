//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_REGISTER_CORE_REGISTERCONTEXTCORE_ARM_H
#define LLDB_SOURCE_PLUGINS_REGISTER_CORE_REGISTERCONTEXTCORE_ARM_H

#include "Register/RegisterInfoInterface.h"
#include "Register/Common/RegisterInfo_arm.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Log.h"

class RegisterContextCore_arm : public lldb_private::RegisterContext {
public:
  RegisterContextCore_arm(
      lldb_private::Thread &thread,
      std::unique_ptr<RegisterInfo_arm> register_info);

  ~RegisterContextCore_arm() override;

  void Invalidate();

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  virtual size_t GetGPRSize();

  virtual unsigned GetRegisterSize(unsigned reg);

  virtual unsigned GetRegisterOffset(unsigned reg);

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const lldb_private::RegisterSet *GetRegisterSet(size_t set) override;

  const char *GetRegisterName(unsigned reg);

protected:
  std::unique_ptr<RegisterInfo_arm> m_register_info_up;

  virtual const lldb_private::RegisterInfo *GetRegisterInfo();

  bool IsGPR(unsigned reg);

  bool IsFPR(unsigned reg);

  size_t GetFPUSize() { return sizeof(RegisterInfo_arm::FPU); }

  virtual bool ReadGPR() = 0;
  virtual bool ReadFPR() = 0;
  virtual bool WriteGPR() = 0;
  virtual bool WriteFPR() = 0;
};

#endif // LLDB_SOURCE_PLUGINS_REGISTER_CORE_REGISTERCONTEXTCORE_ARM_H
