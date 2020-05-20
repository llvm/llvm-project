//===-- AArch64.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ABI_AARCH64_ABIAARCH64_H
#define LLDB_SOURCE_PLUGINS_ABI_AARCH64_ABIAARCH64_H

#include "lldb/Target/ABI.h"

class ABIAArch64: public lldb_private::MCBasedABI {
public:
  static void Initialize();
  static void Terminate();

protected:
  std::pair<uint32_t, uint32_t>
  GetEHAndDWARFNums(llvm::StringRef name) override;

  std::string GetMCName(std::string reg) override {
    MapRegisterName(reg, "v", "q");
    return reg;
  }

  uint32_t GetGenericNum(llvm::StringRef name) override;

  using lldb_private::MCBasedABI::MCBasedABI;
};
#endif
