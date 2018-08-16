//===-- SourceReference.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_SOURCEREFERENCE_H_
#define LLDBVSCODE_SOURCEREFERENCE_H_

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
