//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/UnwindInfoChecker/UnwindInfoHistory.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <optional>

using namespace llvm;

std::optional<dwarf::UnwindTable::const_iterator>
UnwindInfoHistory::getCurrentUnwindRow() const {
  if (!Table.size())
    return std::nullopt;

  return --Table.end();
}

void UnwindInfoHistory::update(const MCCFIInstruction &CFIDirective) {
  auto DwarfOperations = convert(CFIDirective);
  if (!DwarfOperations) {
    Context->reportError(
        CFIDirective.getLoc(),
        "couldn't apply this directive to the unwinding information state");
  }

  auto LastRow = getCurrentUnwindRow();
  dwarf::UnwindRow Row =
      LastRow.has_value() ? *(LastRow.value()) : dwarf::UnwindRow();
  if (Error Err = Table.parseRows(DwarfOperations.value(), Row, nullptr)) {
    Context->reportError(
        CFIDirective.getLoc(),
        formatv("could not parse this CFI directive due to: {0}",
                toString(std::move(Err))));

    // Proceed the analysis by ignoring this CFI directive.
    return;
  }
  Table.insertRow(Row);
}

std::optional<dwarf::CFIProgram>
UnwindInfoHistory::convert(MCCFIInstruction CFIDirective) {
  //! FIXME, this way of instantiating CFI program does not look right, either
  //! refactor CFIProgram to not depend on the Code/Data Alignment or add a new
  //! type that is independent from this and is also feedable to UnwindTable.
  auto DwarfOperations = dwarf::CFIProgram(
      1 /* TODO */, 1 /* TODO */, Context->getTargetTriple().getArch());

  switch (CFIDirective.getOperation()) {
  case MCCFIInstruction::OpSameValue:
    DwarfOperations.addInstruction(dwarf::DW_CFA_same_value,
                                   CFIDirective.getRegister());
    break;
  case MCCFIInstruction::OpRememberState:
    DwarfOperations.addInstruction(dwarf::DW_CFA_remember_state);
    break;
  case MCCFIInstruction::OpRestoreState:
    DwarfOperations.addInstruction(dwarf::DW_CFA_restore_state);
    break;
  case MCCFIInstruction::OpOffset:
    DwarfOperations.addInstruction(dwarf::DW_CFA_offset,
                                   CFIDirective.getRegister(),
                                   CFIDirective.getOffset());
    break;
  case MCCFIInstruction::OpLLVMDefAspaceCfa:
    DwarfOperations.addInstruction(dwarf::DW_CFA_LLVM_def_aspace_cfa,
                                   CFIDirective.getRegister());
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa_register,
                                   CFIDirective.getRegister());
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa_offset,
                                   CFIDirective.getOffset());
    break;
  case MCCFIInstruction::OpDefCfa:
    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa,
                                   CFIDirective.getRegister(),
                                   CFIDirective.getOffset());
    break;
  case MCCFIInstruction::OpRelOffset:
    if (!getCurrentUnwindRow()) // TODO maybe replace it with assert
      return std::nullopt;

    DwarfOperations.addInstruction(
        dwarf::DW_CFA_offset, CFIDirective.getRegister(),
        CFIDirective.getOffset() -
            getCurrentUnwindRow().value()->getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    if (!getCurrentUnwindRow()) // TODO maybe replace it with assert
      return std::nullopt;

    DwarfOperations.addInstruction(
        dwarf::DW_CFA_def_cfa_offset,
        CFIDirective.getOffset() +
            getCurrentUnwindRow().value()->getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpEscape:
    // TODO It's now feasible but for now, I ignore it
    break;
  case MCCFIInstruction::OpRestore:
    DwarfOperations.addInstruction(dwarf::DW_CFA_restore);
    break;
  case MCCFIInstruction::OpUndefined:
    DwarfOperations.addInstruction(dwarf::DW_CFA_undefined,
                                   CFIDirective.getRegister());
    break;
  case MCCFIInstruction::OpRegister:
    DwarfOperations.addInstruction(dwarf::DW_CFA_register,
                                   CFIDirective.getRegister(),
                                   CFIDirective.getRegister2());
    break;
  case MCCFIInstruction::OpWindowSave:
    // TODO make sure these are the same.
    DwarfOperations.addInstruction(dwarf::DW_CFA_GNU_window_save);
    break;
  case MCCFIInstruction::OpNegateRAState:
    // TODO make sure these are the same.
    DwarfOperations.addInstruction(dwarf::DW_CFA_AARCH64_negate_ra_state);
    break;
  case MCCFIInstruction::OpNegateRAStateWithPC:
    // TODO make sure these are the same.
    DwarfOperations.addInstruction(
        dwarf::DW_CFA_AARCH64_negate_ra_state_with_pc);
    break;
  case MCCFIInstruction::OpGnuArgsSize:
    DwarfOperations.addInstruction(dwarf::DW_CFA_GNU_args_size);
    break;
  case MCCFIInstruction::OpLabel:
    // TODO, I don't know what it is, I have to implement it.
    break;
  case MCCFIInstruction::OpValOffset:
    DwarfOperations.addInstruction(dwarf::DW_CFA_val_offset,
                                   CFIDirective.getRegister(),
                                   CFIDirective.getOffset());
    break;
  }

  return DwarfOperations;
}
