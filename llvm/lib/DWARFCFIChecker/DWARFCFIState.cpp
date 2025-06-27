//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIState.h"
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
DWARFCFIState::getCurrentUnwindRow() const {
  if (!Table.size())
    return std::nullopt;

  return --Table.end();
}

void DWARFCFIState::update(const MCCFIInstruction &Directive) {
  auto CFIP = convert(Directive);

  auto MaybeCurrentRow = getCurrentUnwindRow();

  // This is a copy of the last row of the table (or a new empty row), its value
  // will be updated by `parseRows`.
  dwarf::UnwindRow NewRow = MaybeCurrentRow.has_value()
                                ? *(MaybeCurrentRow.value())
                                : dwarf::UnwindRow();

  // `parseRows` updates the current row by applying the `CFIProgram` to it.
  // During this process, it may create multiple rows that should be placed in
  // the unwinding table, preceding the newly updated row and following the
  // previous rows. These middle rows are stored in `PrecedingRows`.
  dwarf::UnwindTable::RowContainer PrecedingRows;
  if (Error Err = parseRows(CFIP, NewRow, nullptr).moveInto(PrecedingRows)) {
    Context->reportError(
        Directive.getLoc(),
        formatv("could not parse this CFI directive due to: {0}",
                toString(std::move(Err))));

    // Proceed the analysis by ignoring this CFI directive.
    return;
  }

  Table.insert(Table.end(), PrecedingRows.begin(), PrecedingRows.end());
  Table.push_back(NewRow);
}

dwarf::CFIProgram DWARFCFIState::convert(MCCFIInstruction Directive) {
  auto CFIP = dwarf::CFIProgram(
      /* CodeAlignmentFactor */ 1, /* DataAlignmentFactor */ 1,
      Context->getTargetTriple().getArch());

  auto MaybeCurrentRow = getCurrentUnwindRow();
  switch (Directive.getOperation()) {
  case MCCFIInstruction::OpSameValue:
    CFIP.addInstruction(dwarf::DW_CFA_same_value, Directive.getRegister());
    break;
  case MCCFIInstruction::OpRememberState:
    CFIP.addInstruction(dwarf::DW_CFA_remember_state);
    break;
  case MCCFIInstruction::OpRestoreState:
    CFIP.addInstruction(dwarf::DW_CFA_restore_state);
    break;
  case MCCFIInstruction::OpOffset:
    CFIP.addInstruction(dwarf::DW_CFA_offset, Directive.getRegister(),
                        Directive.getOffset());
    break;
  case MCCFIInstruction::OpLLVMDefAspaceCfa:
    CFIP.addInstruction(dwarf::DW_CFA_LLVM_def_aspace_cfa,
                        Directive.getRegister());
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    CFIP.addInstruction(dwarf::DW_CFA_def_cfa_register,
                        Directive.getRegister());
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    CFIP.addInstruction(dwarf::DW_CFA_def_cfa_offset, Directive.getOffset());
    break;
  case MCCFIInstruction::OpDefCfa:
    CFIP.addInstruction(dwarf::DW_CFA_def_cfa, Directive.getRegister(),
                        Directive.getOffset());
    break;
  case MCCFIInstruction::OpRelOffset:
    assert(
        MaybeCurrentRow &&
        "Cannot define relative offset to a non-existing CFA unwinding rule");

    CFIP.addInstruction(dwarf::DW_CFA_offset, Directive.getRegister(),
                        Directive.getOffset() -
                            (*MaybeCurrentRow)->getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    assert(MaybeCurrentRow &&
           "Cannot adjust CFA offset of a non-existing CFA unwinding rule");

    CFIP.addInstruction(dwarf::DW_CFA_def_cfa_offset,
                        Directive.getOffset() +
                            (*MaybeCurrentRow)->getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpEscape:
    // TODO: DWARFExpressions are not supported yet, ignoring expression here.
    break;
  case MCCFIInstruction::OpRestore:
    CFIP.addInstruction(dwarf::DW_CFA_restore);
    break;
  case MCCFIInstruction::OpUndefined:
    CFIP.addInstruction(dwarf::DW_CFA_undefined, Directive.getRegister());
    break;
  case MCCFIInstruction::OpRegister:
    CFIP.addInstruction(dwarf::DW_CFA_register, Directive.getRegister(),
                        Directive.getRegister2());
    break;
  case MCCFIInstruction::OpWindowSave:
    CFIP.addInstruction(dwarf::DW_CFA_GNU_window_save);
    break;
  case MCCFIInstruction::OpNegateRAState:
    CFIP.addInstruction(dwarf::DW_CFA_AARCH64_negate_ra_state);
    break;
  case MCCFIInstruction::OpNegateRAStateWithPC:
    CFIP.addInstruction(dwarf::DW_CFA_AARCH64_negate_ra_state_with_pc);
    break;
  case MCCFIInstruction::OpGnuArgsSize:
    CFIP.addInstruction(dwarf::DW_CFA_GNU_args_size);
    break;
  case MCCFIInstruction::OpLabel:
    // TODO: I don't know what it is, I have to implement it.
    llvm_unreachable("CFI label instruction is not implemented yet, and it is "
                     "not decided to be ignored.");
    break;
  case MCCFIInstruction::OpValOffset:
    CFIP.addInstruction(dwarf::DW_CFA_val_offset, Directive.getRegister(),
                        Directive.getOffset());
    break;
  }

  return CFIP;
}
