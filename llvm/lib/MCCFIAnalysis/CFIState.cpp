// TODO check what includes to keep and what to remove
#include "llvm/MCCFIAnalysis/CFIState.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <optional>

using namespace llvm;

CFIState::CFIState(const CFIState &Other) : Context(Other.Context) {
  // TODO maybe remove it
  State = Other.State;
}

CFIState &CFIState::operator=(const CFIState &Other) {
  // TODO maybe remove it
  if (this != &Other) {
    State = Other.State;
    Context = Other.Context;
  }

  return *this;
}

std::optional<DWARFRegType>
CFIState::getReferenceRegisterForCallerValueOfRegister(DWARFRegType Reg) const {
  // TODO maybe move it the Location class
  auto LastRow = getLastRow();
  assert(LastRow && "The state is empty.");

  auto UnwinLoc = LastRow->getRegisterLocations().getRegisterLocation(Reg);
  assert(UnwinLoc &&
         "The register should be tracked inside the register states");

  switch (UnwinLoc->getLocation()) {
  case dwarf::UnwindLocation::Location::Undefined:
  case dwarf::UnwindLocation::Location::Constant:
  case dwarf::UnwindLocation::Location::Unspecified:
    // TODO here should look into expr and find the registers, but for now it's
    // TODO like this:
  case dwarf::UnwindLocation::Location::DWARFExpr:
    return std::nullopt;
  case dwarf::UnwindLocation::Location::Same:
    return Reg;
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    return UnwinLoc->getRegister();
  case dwarf::UnwindLocation::Location::CFAPlusOffset:
    // TODO check if it's ok to assume CFA is always depending on other
    // TODO register, if yes assert it here!
    return LastRow->getCFAValue().getRegister();
  }
}

void CFIState::apply(const MCCFIInstruction &CFIDirective) {
  auto DwarfOperations = convertMC2DWARF(CFIDirective);
  if (!DwarfOperations) {
    Context->reportError(
        CFIDirective.getLoc(),
        "couldn't apply this directive to the unwinding information state");
  }

  auto &&LastRow = getLastRow();
  dwarf::UnwindRow Row = LastRow ? LastRow.value() : dwarf::UnwindRow();
  if (Error Err = State.parseRows(DwarfOperations.value(), Row, nullptr)) {
    // ! FIXME what should I do with this error?
    Context->reportError(
        CFIDirective.getLoc(),
        formatv("could not parse this CFI directive due to: {0}",
                toString(std::move(Err))));
    assert(false);
  }
  State.insertRow(Row);
}

std::optional<dwarf::CFIProgram>
CFIState::convertMC2DWARF(MCCFIInstruction CFIDirective) {
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
    if (!getLastRow()) // TODO maybe replace it with assert
      return std::nullopt;

    DwarfOperations.addInstruction(
        dwarf::DW_CFA_offset, CFIDirective.getRegister(),
        CFIDirective.getOffset() - getLastRow()->getCFAValue().getOffset());
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    if (!getLastRow()) // TODO maybe replace it with assert
      return std::nullopt;

    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa_offset,
                                   CFIDirective.getOffset() +
                                       getLastRow()->getCFAValue().getOffset());
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

std::optional<dwarf::UnwindRow> CFIState::getLastRow() const {
  if (!State.size())
    return std::nullopt;

  //! FIXME too dirty
  auto &&it = State.end();
  return *--it;
}