//===-- DAPLog.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <mutex>
#include <system_error>

using namespace llvm;

namespace lldb_dap {

Log::Log(StringRef filename, std::error_code &EC) : m_stream(filename, EC) {}

void Log::WriteMessage(StringRef message) {
  std::lock_guard<std::mutex> lock(m_mutex);
  std::chrono::duration<double> now{
      std::chrono::system_clock::now().time_since_epoch()};
  m_stream << formatv("{0:f9} ", now.count()).str() << message << "\n";
  m_stream.flush();
}

} // namespace lldb_dap
