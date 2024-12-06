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
  uint32_t GetWatchpointSize(uint32_t wp_index) override;

  bool ValidateWatchpoint(size_t size, lldb::addr_t &addr) override;

  uint32_t MakeControlValue(size_t size, uint32_t *watch_flags = NULL) override;
};

} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextDBReg_arm64_h
