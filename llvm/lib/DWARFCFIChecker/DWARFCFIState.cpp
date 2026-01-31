//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIState.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <optional>

using namespace llvm;

std::optional<dwarf::UnwindRow> DWARFCFIState::getCurrentUnwindRow() const {
  if (!IsInitiated)
    return std::nullopt;
  return Row;
}

void DWARFCFIState::update(const MCCFIInstruction &Directive) {
  auto CFIP = convert(Directive);

  // This is a copy of the current row, its value will be updated by
  // `parseRows`.
  dwarf::UnwindRow NewRow = Row;

  // `parseRows` updates the current row by applying the `CFIProgram` to it.
  // During this process, it may create multiple rows preceding the newly
  // updated row and following the previous rows. These middle rows are stored
  // in `PrecedingRows`. For now, there is no need to store these rows in the
  // state, so they are ignored in the end.
  dwarf::UnwindTable::RowContainer PrecedingRows;

  // TODO: `.cfi_remember_state` and `.cfi_restore_state` directives are not
  // supported yet. The reason is that `parseRows` expects the stack of states
  // to be produced and used in a single `CFIProgram`. However, in this use
  // case, each instruction creates its own `CFIProgram`, which means the stack
  // of states is forgotten between instructions. To fix it, `parseRows` should
  // be refactored to read the current stack of states from the argument and
  // update it based on the `CFIProgram.`
  if (Error Err = parseRows(CFIP, NewRow, nullptr).takeError()) {
    Context->reportError(
        Directive.getLoc(),
        formatv("could not parse this CFI directive due to: {0}",
                toString(std::move(Err))));

    // Proceed the analysis by ignoring this CFI directive.
    return;
  }

  Row = NewRow;
  IsInitiated = true;
}

dwarf::CFIProgram DWARFCFIState::convert(MCCFIInstruction Directive) {
  auto CFIP = dwarf::CFIProgram(
      /* CodeAlignmentFactor */ 1, /* DataAlignmentFactor */ 1,
      Context->getTargetTriple().getArch());

  switch (Directive.getOperation()) {
  case MCCFIInstruction::OpSameValue:
    CFIP.addInstruction(dwarf::DW_CFA_same_value, Directive.getRegister());
    break;
  case MCCFIInstruction::OpRememberState:
    // TODO: remember state is not supported yet, the following line does not
    // work:
    // CFIP.addInstruction(dwarf::DW_CFA_remember_state);
    // The reason is explained in the `DWARFCFIState::update` method where
    // `dwarf::parseRows` is used.
    Context->reportWarning(Directive.getLoc(),
                           "this directive is not supported, ignoring it");
    break;
  case MCCFIInstruction::OpRestoreState:
    // TODO: restore state is not supported yet, the following line does not
    // work:
    // CFIP.addInstruction(dwarf::DW_CFA_restore_state);
    // The reason is explained in the `DWARFCFIState::update` method where
    // `dwarf::parseRows` is used.
    Context->reportWarning(Directive.getLoc(),
                           "this directive is not supported, ignoring it");
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
        IsInitiated &&
        "cannot define relative offset to a non-existing CFA unwinding rule");

    CFIP.addInstruction(dwarf::DW_CFA_offset, Directive.getRegister(),
                        Directive.getOffset() - Row.getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    assert(IsInitiated &&
           "cannot adjust CFA offset of a non-existing CFA unwinding rule");

    CFIP.addInstruction(dwarf::DW_CFA_def_cfa_offset,
                        Directive.getOffset() + Row.getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpEscape:
    // TODO: DWARFExpressions are not supported yet, ignoring expression here.
    Context->reportWarning(Directive.getLoc(),
                           "this directive is not supported, ignoring it");
    break;
  case MCCFIInstruction::OpRestore:
    // The `.cfi_restore register` directive restores the register's unwinding
    // information to its CIE value. However, assemblers decide where CIE ends
    // and the FDE starts, so the functionality of this directive depends on the
    // assembler's decision and cannot be validated.
    Context->reportWarning(
        Directive.getLoc(),
        "this directive behavior depends on the assembler, ignoring it");
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
    // `.cfi_label` does not have any functional effect on unwinding process.
    break;
  case MCCFIInstruction::OpValOffset:
    CFIP.addInstruction(dwarf::DW_CFA_val_offset, Directive.getRegister(),
                        Directive.getOffset());
    break;
  }

  return CFIP;
}
