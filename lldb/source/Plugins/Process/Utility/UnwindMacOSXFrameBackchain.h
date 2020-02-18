//===-- UnwindMacOSXFrameBackchain.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_UNWINDMACOSXFRAMEBACKCHAIN_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_UNWINDMACOSXFRAMEBACKCHAIN_H

#include <vector>

#include "lldb/Target/Unwind.h"
#include "lldb/lldb-private.h"

class UnwindMacOSXFrameBackchain : public lldb_private::Unwind {
public:
  UnwindMacOSXFrameBackchain(lldb_private::Thread &thread);

  ~UnwindMacOSXFrameBackchain() override = default;

protected:
  void DoClear() override { m_cursors.clear(); }

  uint32_t DoGetFrameCount() override;

  bool DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                             lldb::addr_t &pc,
                             bool &behaves_like_zeroth_frame) override;

  lldb::RegisterContextSP
  DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  friend class RegisterContextMacOSXFrameBackchain;

  struct Cursor {
    lldb::addr_t pc; // Program counter
    lldb::addr_t fp; // Frame pointer for us with backchain
  };

private:
  std::vector<Cursor> m_cursors;

  size_t GetStackFrameData_i386(const lldb_private::ExecutionContext &exe_ctx);

  size_t
  GetStackFrameData_x86_64(const lldb_private::ExecutionContext &exe_ctx);

  // For UnwindMacOSXFrameBackchain only
  DISALLOW_COPY_AND_ASSIGN(UnwindMacOSXFrameBackchain);
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_UNWINDMACOSXFRAMEBACKCHAIN_H
