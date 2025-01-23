//===-- UnwindWasm.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_UnwindWasm_h_
#define lldb_UnwindWasm_h_

#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Unwind.h"
#include <vector>

namespace lldb_private {
namespace wasm {

/// UnwindWasm manages stack unwinding for a WebAssembly process.
class UnwindWasm : public lldb_private::Unwind {
public:
  UnwindWasm(lldb_private::Thread &thread)
      : Unwind(thread), m_frames(), m_unwind_complete(false) {}
  ~UnwindWasm() override = default;

protected:
  /// Unwind protocol.
  /// \{
  void DoClear() override {
    m_frames.clear();
    m_unwind_complete = false;
  }

  uint32_t DoGetFrameCount() override;

  bool DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                             lldb::addr_t &pc,
                             bool &behaves_like_zeroth_frame) override;

  lldb::RegisterContextSP
  DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;
  /// \}

private:
  std::vector<lldb::addr_t> m_frames;
  bool m_unwind_complete;

  UnwindWasm(const UnwindWasm &);
  const UnwindWasm &operator=(const UnwindWasm &) = delete;
};

} // namespace wasm
} // namespace lldb_private

#endif // lldb_UnwindWasm_h_
