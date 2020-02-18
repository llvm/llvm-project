//===-- SourceReference.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_VSCODE_SOURCEREFERENCE_H
#define LLDB_TOOLS_LLDB_VSCODE_SOURCEREFERENCE_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace lldb_vscode {

struct SourceReference {
  std::string content;
  llvm::DenseMap<lldb::addr_t, int64_t> addr_to_line;

  int64_t GetLineForPC(lldb::addr_t pc) const {
    auto addr_line = addr_to_line.find(pc);
    if (addr_line != addr_to_line.end())
      return addr_line->second;
    return 0;
  }
};

} // namespace lldb_vscode

#endif
