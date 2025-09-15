//===-- NativeRegisterContextDBReg_arm64.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextDBReg_arm64_h
#define lldb_NativeRegisterContextDBReg_arm64_h

#include "Plugins/Process/Utility/NativeRegisterContextDBReg.h"

namespace lldb_private {

class NativeRegisterContextDBReg_arm64 : public NativeRegisterContextDBReg {
public:
  NativeRegisterContextDBReg_arm64()
      : NativeRegisterContextDBReg(/*enable_bit=*/0x1U) {}

private:
  uint32_t GetWatchpointSize(uint32_t wp_index) override;

  std::optional<WatchpointDetails>
  AdjustWatchpoint(const WatchpointDetails &details) override;

  uint32_t MakeBreakControlValue(size_t size) override;

  uint32_t MakeWatchControlValue(size_t size, uint32_t watch_flags) override;
};

} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextDBReg_arm64_h
