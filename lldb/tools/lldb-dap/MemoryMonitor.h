//===-- MemoryMonitor.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_WATCHPOINT_H
#define LLDB_TOOLS_LLDB_DAP_WATCHPOINT_H

#include <functional>
#include <memory>

namespace lldb_dap {

class MemoryMonitor {
public:
  using Callback = std::function<void()>;

  MemoryMonitor(Callback callback) : m_callback(callback) {}
  virtual ~MemoryMonitor() = default;

  /// MemoryMonitor is not copyable.
  /// @{
  MemoryMonitor(const MemoryMonitor &) = delete;
  MemoryMonitor &operator=(const MemoryMonitor &) = delete;
  /// @}

  static std::unique_ptr<MemoryMonitor> Create(Callback callback);

  virtual void Start() = 0;
  virtual void Stop() = 0;

protected:
  Callback m_callback;
};

} // namespace lldb_dap

#endif
