//===-- RegisterContextDPU.h --------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextDPU_h_
#define lldb_RegisterContextDPU_h_

#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/RegisterNumber.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RegisterContextDPU;
typedef std::shared_ptr<RegisterContextDPU> RegisterContextDPUSP;

class RegisterContextDPU : public lldb_private::RegisterContext {
public:
  RegisterContextDPU(Thread &thread, RegisterContextDPUSP prev_frame_reg_ctx_sp,
                     lldb::addr_t cfa, lldb::addr_t pc, uint32_t frame_number);

  ~RegisterContextDPU() override = default;

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &value) override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

  bool ReadRegisterFromSavedLocation(const RegisterInfo *reg_info,
                                     RegisterValue &value);

  bool WriteRegisterToSavedLocation(const RegisterInfo *reg_info,
                                    const RegisterValue &value);

  void GetFunction(Function **fct, lldb::addr_t pc);

  bool PCIsInstructionReturn(lldb::addr_t pc);

  bool GetUnwindPlanSP(Function *fct, lldb::UnwindPlanSP &unwind_plan_sp);

private:
  bool LookForRegisterLocation(const RegisterInfo *reg_info, lldb::addr_t &addr);

  bool PCInPrologue(lldb::addr_t start_addr, uint32_t nb_callee_saved_regs);

  Thread &m_thread;
  lldb::RegisterContextSP reg_ctx_sp;
  uint32_t m_frame_number;
  lldb::addr_t m_cfa, m_pc;
  RegisterContextDPUSP m_prev_frame;
};
}

#endif /* lldb_RegisterContextDPU_h_ */
