//===-- NativeRegisterContextDBReg_arm.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextDBReg_arm_h
#define lldb_NativeRegisterContextDBReg_arm_h

#include "Plugins/Process/Utility/NativeRegisterContextDBReg.h"

namespace lldb_private {

class NativeRegisterContextDBReg_arm : public NativeRegisterContextDBReg {
public:
  NativeRegisterContextDBReg_arm()
      : NativeRegisterContextDBReg(/*enable_bit=*/0x1U) {}

private:
  uint32_t GetWatchpointSize(uint32_t wp_index) override;

  std::optional<WatchpointDetails>
  AdjustWatchpoint(const WatchpointDetails &details) override;

  BreakpointDetails AdjustBreakpoint(const BreakpointDetails &details) override;

  uint32_t MakeBreakControlValue(size_t size) override;

  uint32_t MakeWatchControlValue(size_t size, uint32_t watch_flags) override;

  bool ValidateBreakpoint(size_t size,
                          [[maybe_unused]] lldb::addr_t addr) override {
    // Break on 4 or 2 byte instructions.
    return size == 4 || size == 2;
  }
};

} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextDBReg_arm_h
