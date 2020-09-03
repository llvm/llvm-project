//===-- ABIX86_64.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ABI_X86_ABIX86_64_H
#define LLDB_SOURCE_PLUGINS_ABI_X86_ABIX86_64_H

#include "lldb/Target/ABI.h"
#include "lldb/lldb-private.h"

class ABIX86_64 : public lldb_private::MCBasedABI {
protected:
  std::string GetMCName(std::string name) override {
    MapRegisterName(name, "stmm", "st");
    return name;
  }

private:
  using lldb_private::MCBasedABI::MCBasedABI;
};

#endif // LLDB_SOURCE_PLUGINS_ABI_X86_ABIX86_64_H
