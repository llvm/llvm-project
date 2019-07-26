//===-- UnwindDPU.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef dpu_UnwindDPU_h_
#define dpu_UnwindDPU_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

class UnwindDPU : public lldb_private::Unwind {
public:
  UnwindDPU(lldb_private::Thread &thread);

  ~UnwindDPU() override = default;

protected:
  void DoClear() override;

  uint32_t DoGetFrameCount() override;

  bool DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                             lldb::addr_t &pc) override;

  lldb::RegisterContextSP
  DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

private:
  struct Cursor {
    lldb::addr_t pc; // The start address of the function/symbol for this
                           // frame - current pc if unknown
    lldb::addr_t cfa;      // The canonical frame address for this stack frame
    lldb::RegisterContextSP reg_ctx_sp;

    Cursor()
        : pc(LLDB_INVALID_ADDRESS), cfa(LLDB_INVALID_ADDRESS),
          reg_ctx_sp() {}
  };

  typedef std::shared_ptr<Cursor> CursorSP;
  std::vector<CursorSP> m_frames;

  void SetFrame(CursorSP *prev_frame, lldb::addr_t cfa, lldb::addr_t pc,
                lldb::RegisterContextSP reg_ctx_sp);

  void GetFunction(Function **fct, lldb::addr_t pc);

  lldb::addr_t ReadMemory(lldb::addr_t src_addr);

  bool PCIsInstructionReturn(lldb::addr_t pc);
};
} // namespace lldb_private
#endif /* dpu_UnwindDPU_h_ */
