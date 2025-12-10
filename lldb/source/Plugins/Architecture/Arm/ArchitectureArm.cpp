//===-- ArchitectureArm.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/Arm/ArchitectureArm.h"
#include "Plugins/Process/Utility/ARMDefines.h"
#include "Plugins/Process/Utility/InstructionUtils.h"
#include "Utility/ARM_DWARF_Registers.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/RegisterNumber.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/UnwindLLDB.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ArchitectureArm)

void ArchitectureArm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Arm-specific algorithms",
                                &ArchitectureArm::Create);
}

void ArchitectureArm::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureArm::Create);
}

std::unique_ptr<Architecture> ArchitectureArm::Create(const ArchSpec &arch) {
  if (arch.GetMachine() != llvm::Triple::arm)
    return nullptr;
  return std::unique_ptr<Architecture>(new ArchitectureArm());
}

void ArchitectureArm::OverrideStopInfo(Thread &thread) const {
  // We need to check if we are stopped in Thumb mode in a IT instruction and
  // detect if the condition doesn't pass. If this is the case it means we
  // won't actually execute this instruction. If this happens we need to clear
  // the stop reason to no thread plans think we are stopped for a reason and
  // the plans should keep going.
  //
  // We do this because when single stepping many ARM processes, debuggers
  // often use the BVR/BCR registers that says "stop when the PC is not equal
  // to its current value". This method of stepping means we can end up
  // stopping on instructions inside an if/then block that wouldn't get
  // executed. By fixing this we can stop the debugger from seeming like you
  // stepped through both the "if" _and_ the "else" clause when source level
  // stepping because the debugger stops regardless due to the BVR/BCR
  // triggering a stop.
  //
  // It also means we can set breakpoints on instructions inside an if/then
  // block and correctly skip them if we use the BKPT instruction. The ARM and
  // Thumb BKPT instructions are unconditional even when executed in a Thumb IT
  // block.
  //
  // If your debugger inserts software traps in ARM/Thumb code, it will need to
  // use 16 and 32 bit instruction for 16 and 32 bit thumb instructions
  // respectively. If your debugger inserts a 16 bit thumb trap on top of a 32
  // bit thumb instruction for an opcode that is inside an if/then, it will
  // change the it/then to conditionally execute your
  // 16 bit trap and then cause your program to crash if it executes the
  // trailing 16 bits (the second half of the 32 bit thumb instruction you
  // partially overwrote).

  RegisterContextSP reg_ctx_sp(thread.GetRegisterContext());
  if (!reg_ctx_sp)
    return;

  const uint32_t cpsr = reg_ctx_sp->GetFlags(0);
  if (cpsr == 0)
    return;

  // Read the J and T bits to get the ISETSTATE
  const uint32_t J = Bit32(cpsr, 24);
  const uint32_t T = Bit32(cpsr, 5);
  const uint32_t ISETSTATE = J << 1 | T;
  if (ISETSTATE == 0) {
// NOTE: I am pretty sure we want to enable the code below
// that detects when we stop on an instruction in ARM mode that is conditional
// and the condition doesn't pass. This can happen if you set a breakpoint on
// an instruction that is conditional. We currently will _always_ stop on the
// instruction which is bad. You can also run into this while single stepping
// and you could appear to run code in the "if" and in the "else" clause
// because it would stop at all of the conditional instructions in both. In
// such cases, we really don't want to stop at this location.
// I will check with the lldb-dev list first before I enable this.
#if 0
    // ARM mode: check for condition on instruction
    const addr_t pc = reg_ctx_sp->GetPC();
    Status error;
    // If we fail to read the opcode we will get UINT64_MAX as the result in
    // "opcode" which we can use to detect if we read a valid opcode.
    const uint64_t opcode = thread.GetProcess()->ReadUnsignedIntegerFromMemory(pc, 4, UINT64_MAX, error);
    if (opcode <= UINT32_MAX)
    {
        const uint32_t condition = Bits32((uint32_t)opcode, 31, 28);
        if (!ARMConditionPassed(condition, cpsr))
        {
            // We ARE stopped on an ARM instruction whose condition doesn't
            // pass so this instruction won't get executed. Regardless of why
            // it stopped, we need to clear the stop info
            thread.SetStopInfo (StopInfoSP());
        }
    }
#endif
  } else if (ISETSTATE == 1) {
    // Thumb mode
    const uint32_t ITSTATE = Bits32(cpsr, 15, 10) << 2 | Bits32(cpsr, 26, 25);
    if (ITSTATE != 0) {
      const uint32_t condition = Bits32(ITSTATE, 7, 4);
      if (!ARMConditionPassed(condition, cpsr)) {
        // We ARE stopped in a Thumb IT instruction on an instruction whose
        // condition doesn't pass so this instruction won't get executed.
        // Regardless of why it stopped, we need to clear the stop info
        thread.SetStopInfo(StopInfoSP());
      }
    }
  }
}

addr_t ArchitectureArm::GetCallableLoadAddress(addr_t code_addr,
                                               AddressClass addr_class) const {
  bool is_alternate_isa = false;

  switch (addr_class) {
  case AddressClass::eData:
  case AddressClass::eDebug:
    return LLDB_INVALID_ADDRESS;
  case AddressClass::eCodeAlternateISA:
    is_alternate_isa = true;
    break;
  default: break;
  }

  if ((code_addr & 2u) || is_alternate_isa)
    return code_addr | 1u;
  return code_addr;
}

addr_t ArchitectureArm::GetOpcodeLoadAddress(addr_t opcode_addr,
                                             AddressClass addr_class) const {
  switch (addr_class) {
  case AddressClass::eData:
  case AddressClass::eDebug:
    return LLDB_INVALID_ADDRESS;
  default: break;
  }
  return opcode_addr & ~(1ull);
}

// The ARM M-Profile Armv7-M Architecture Reference Manual,
// subsection "B1.5 Armv7-M exception model", see the parts
// describing "Exception entry behavior" and "Exception
// return behavior".
// When an exception happens on this processor, certain registers are
// saved below the stack pointer, the stack pointer is decremented,
// a special value is put in the link register to indicate the
// exception has been taken, and an exception handler function
// is invoked.
//
// Detect that special value in $lr, and if present, add
// unwind rules for the registers that were saved above this
// stack frame's CFA.  Overwrite any register locations that
// the current_unwindplan has for these registers; they are
// not correct when we're invoked this way.
UnwindPlanSP ArchitectureArm::GetArchitectureUnwindPlan(
    Thread &thread, RegisterContextUnwind *regctx,
    std::shared_ptr<const UnwindPlan> current_unwindplan) {

  ProcessSP process_sp = thread.GetProcess();
  if (!process_sp)
    return {};

  const ArchSpec arch = process_sp->GetTarget().GetArchitecture();
  if (!arch.GetTriple().isArmMClass() || arch.GetAddressByteSize() != 4)
    return {};

  // Get the caller's LR value from regctx (the LR value
  // at function entry to this function).
  RegisterNumber ra_regnum(thread, eRegisterKindGeneric,
                           LLDB_REGNUM_GENERIC_RA);
  uint32_t ra_regnum_lldb = ra_regnum.GetAsKind(eRegisterKindLLDB);

  if (ra_regnum_lldb == LLDB_INVALID_REGNUM)
    return {};

  UnwindLLDB::ConcreteRegisterLocation regloc = {};
  bool got_concrete_location = false;
  if (regctx->SavedLocationForRegister(ra_regnum_lldb, regloc) ==
      UnwindLLDB::RegisterSearchResult::eRegisterFound) {
    got_concrete_location = true;
  } else {
    RegisterNumber pc_regnum(thread, eRegisterKindGeneric,
                             LLDB_REGNUM_GENERIC_PC);
    uint32_t pc_regnum_lldb = pc_regnum.GetAsKind(eRegisterKindLLDB);
    if (regctx->SavedLocationForRegister(pc_regnum_lldb, regloc) ==
        UnwindLLDB::RegisterSearchResult::eRegisterFound)
      got_concrete_location = true;
  }

  if (!got_concrete_location)
    return {};

  addr_t callers_return_address = LLDB_INVALID_ADDRESS;
  const RegisterInfo *reg_info = regctx->GetRegisterInfoAtIndex(ra_regnum_lldb);
  if (reg_info) {
    RegisterValue reg_value;
    if (regctx->ReadRegisterValueFromRegisterLocation(regloc, reg_info,
                                                      reg_value)) {
      callers_return_address = reg_value.GetAsUInt32();
    }
  }

  if (callers_return_address == LLDB_INVALID_ADDRESS)
    return {};

  // ARMv7-M ARM says that the LR will be set to
  // one of these values when an exception has taken
  // place:
  //    if HaveFPExt() then
  //      if CurrentMode==Mode_Handler then
  //        LR = Ones(27):NOT(CONTROL.FPCA):'0001';
  //      else
  //        LR = Ones(27):NOT(CONTROL.FPCA):'1':CONTROL.SPSEL:'01';
  //    else
  //      if CurrentMode==Mode_Handler then
  //        LR = Ones(28):'0001';
  //      else
  //        LR = Ones(29):CONTROL.SPSEL:'01';

  // Top 27 bits are set for an exception return.
  const uint32_t exception_return = -1U & ~0b11111U;
  // Bit4 is 1 if only GPRs were saved.
  const uint32_t gprs_only = 0b10000;
  // Bit<1:0> are '01'.
  const uint32_t lowbits = 0b01;

  if ((callers_return_address & exception_return) != exception_return)
    return {};
  if ((callers_return_address & lowbits) != lowbits)
    return {};

  const bool fp_regs_saved = !(callers_return_address & gprs_only);

  const RegisterKind plan_regkind = current_unwindplan->GetRegisterKind();
  UnwindPlanSP new_plan = std::make_shared<UnwindPlan>(plan_regkind);
  new_plan->SetSourceName("Arm Cortex-M exception return UnwindPlan");
  new_plan->SetSourcedFromCompiler(eLazyBoolNo);
  new_plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  new_plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);

  int stored_regs_size = fp_regs_saved ? 0x68 : 0x20;

  uint32_t gpr_regs[] = {dwarf_r0,  dwarf_r1, dwarf_r2, dwarf_r3,
                         dwarf_r12, dwarf_lr, dwarf_pc, dwarf_cpsr};
  const int gpr_reg_count = std::size(gpr_regs);
  uint32_t fpr_regs[] = {dwarf_s0,  dwarf_s1,  dwarf_s2,  dwarf_s3,
                         dwarf_s4,  dwarf_s5,  dwarf_s6,  dwarf_s7,
                         dwarf_s8,  dwarf_s9,  dwarf_s10, dwarf_s11,
                         dwarf_s12, dwarf_s13, dwarf_s14, dwarf_s15};
  const int fpr_reg_count = std::size(fpr_regs);

  RegisterContextSP reg_ctx_sp = thread.GetRegisterContext();
  std::vector<uint32_t> saved_regs;
  for (int i = 0; i < gpr_reg_count; i++) {
    uint32_t regno = gpr_regs[i];
    reg_ctx_sp->ConvertBetweenRegisterKinds(eRegisterKindDWARF, gpr_regs[i],
                                            plan_regkind, regno);
    saved_regs.push_back(regno);
  }
  if (fp_regs_saved) {
    for (int i = 0; i < fpr_reg_count; i++) {
      uint32_t regno = fpr_regs[i];
      reg_ctx_sp->ConvertBetweenRegisterKinds(eRegisterKindDWARF, fpr_regs[i],
                                              plan_regkind, regno);
      saved_regs.push_back(regno);
    }
  }

  addr_t cfa;
  if (!regctx->GetCFA(cfa))
    return {};

  // The CPSR value saved to stack is actually (from Armv7-M ARM)
  //   "XPSR<31:10>:frameptralign:XPSR<8:0>"
  // Bit 9 indicates that the stack pointer was aligned (to
  // an 8-byte alignment) when the exception happened, and we must
  // account for that when restoring the original stack pointer value.
  Status error;
  uint32_t callers_xPSR =
      process_sp->ReadUnsignedIntegerFromMemory(cfa + 0x1c, 4, 0, error);
  const bool align_stack = callers_xPSR & (1U << 9);
  uint32_t callers_sp = cfa + stored_regs_size;
  if (align_stack)
    callers_sp |= 4;

  Log *log = GetLog(LLDBLog::Unwind);
  LLDB_LOGF(log,
            "ArchitectureArm::GetArchitectureUnwindPlan found caller return "
            "addr of 0x%" PRIx64 ", for frame with CFA 0x%" PRIx64
            ", fp_regs_saved %d, stored_regs_size 0x%x, align stack %d",
            callers_return_address, cfa, fp_regs_saved, stored_regs_size,
            align_stack);

  uint32_t sp_regnum = dwarf_sp;
  reg_ctx_sp->ConvertBetweenRegisterKinds(eRegisterKindDWARF, dwarf_sp,
                                          plan_regkind, sp_regnum);

  const int row_count = current_unwindplan->GetRowCount();
  for (int i = 0; i < row_count; i++) {
    UnwindPlan::Row row = *current_unwindplan->GetRowAtIndex(i);
    uint32_t offset = 0;
    const size_t saved_reg_count = saved_regs.size();
    for (size_t j = 0; j < saved_reg_count; j++) {
      // The locations could be set with
      // SetRegisterLocationToIsConstant(regno, cfa+offset)
      // expressing it in terms of CFA addr+offset - this UnwindPlan
      // is only used once, with this specific CFA.  I'm not sure
      // which will be clearer for someone reading the unwind log.
      row.SetRegisterLocationToAtCFAPlusOffset(saved_regs[j], offset, true);
      offset += 4;
    }
    row.SetRegisterLocationToIsCFAPlusOffset(sp_regnum, callers_sp - cfa, true);
    new_plan->AppendRow(row);
  }
  return new_plan;
}
