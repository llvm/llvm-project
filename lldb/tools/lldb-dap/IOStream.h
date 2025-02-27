//===-- IOStream.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_IOSTREAM_H
#define LLDB_TOOLS_LLDB_DAP_IOSTREAM_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace lldb_dap {

struct InputStream {
  lldb::IOObjectSP descriptor;

  explicit InputStream(lldb::IOObjectSP descriptor)
      : descriptor(std::move(descriptor)) {}

  bool read_full(size_t length, std::string &text);

  bool read_line(std::string &line);

  bool read_expected(llvm::StringRef expected);
};

struct OutputStream {
  lldb::IOObjectSP descriptor;

  explicit OutputStream(lldb::IOObjectSP descriptor)
      : descriptor(std::move(descriptor)) {}

  bool write_full(llvm::StringRef str);
};
} // namespace lldb_dap

#endif
