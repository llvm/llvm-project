//===-- DAPLog.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <mutex>

using namespace llvm;

namespace lldb_dap {

void Log::Emit(StringRef message) { Emit(message, "", 0); }

void Log::Emit(StringRef message, StringRef file, size_t line) {
  std::lock_guard<Log::Mutex> lock(m_mutex);
  const llvm::sys::TimePoint<> time = std::chrono::system_clock::now();
  m_stream << formatv("[{0:%H:%M:%S.%L}]", time) << " ";
  if (!file.empty())
    m_stream << sys::path::filename(file) << ":" << line << " ";
  if (!m_prefix.empty())
    m_stream << m_prefix;
  m_stream << message << "\n";
  m_stream.flush();
}

} // namespace lldb_dap
