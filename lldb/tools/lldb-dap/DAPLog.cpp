//===-- DAPLog.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <mutex>

using namespace llvm;

namespace lldb_dap {

void Log::Emit(StringRef message) {
  std::lock_guard<Log::Mutex> lock(m_mutex);
  std::chrono::duration<double> now{
      std::chrono::system_clock::now().time_since_epoch()};
  m_stream << formatv("{0:f9} ", now.count()).str() << m_prefix << message
           << "\n";
  m_stream.flush();
}

void Log::Emit(StringRef file, size_t line, StringRef message) {
  std::lock_guard<Log::Mutex> lock(m_mutex);
  std::chrono::duration<double> now{
      std::chrono::system_clock::now().time_since_epoch()};
  m_stream << formatv("{0:f9} {1}:{2} ", now.count(), sys::path::filename(file),
                      line)
                  .str()
           << m_prefix << message << "\n";
  m_stream.flush();
}

} // namespace lldb_dap
