//===-- DAPLog.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"
#include "llvm/ADT/StringRef.h"
#include <fstream>
#include <mutex>

using namespace llvm;

namespace lldb_dap {

Log::Log(StringRef filename) : m_stream(std::ofstream(filename.str())) {}

void Log::WriteMessage(StringRef message) {
  std::scoped_lock<std::mutex> lock(m_mutex);
  m_stream << message.str();
  m_stream.flush();
}

} // namespace lldb_dap
