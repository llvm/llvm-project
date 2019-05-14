//===-- RegisterContextWindows.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_H_
#define liblldb_RegisterContextWindows_H_

#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class Thread;

class RegisterContextWindows : public lldb_private::RegisterContext {
public:
  // Constructors and Destructors
  RegisterContextWindows(Thread &thread, uint32_t concrete_frame_idx);

  virtual ~RegisterContextWindows();

  // Subclasses must override these functions
  void InvalidateAllRegisters() override;

  bool ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

  // Subclasses can override these functions if desired
  uint32_t NumSupportedHardwareBreakpoints() override;

  uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  bool ClearHardwareBreakpoint(uint32_t hw_idx) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read,
                                 bool write) override;

  bool ClearHardwareWatchpoint(uint32_t hw_index) override;

  bool HardwareSingleStep(bool enable) override;

protected:
  virtual bool CacheAllRegisterValues();

  CONTEXT m_context;
  bool m_context_stale;
};
}

#endif // #ifndef liblldb_RegisterContextWindows_H_
