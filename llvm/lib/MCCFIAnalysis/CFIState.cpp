// TODO check what includes to keep and what to remove
#include "llvm/MCCFIAnalysis/CFIState.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

using namespace llvm;

dwarf::CFIProgram llvm::convertMC2DWARF(MCContext &Context,
                                        MCCFIInstruction CFIDirective,
                                        int64_t CFAOffset) {
  //! FIXME, this way of instantiating CFI program does not look right, either
  //! refactor CFIProgram to not depend on the Code/Data Alignment or add a new
  //! type that is independent from this and is also feedable to UnwindTable.
  auto DwarfOperations =
      dwarf::CFIProgram(0, 0, Context.getTargetTriple().getArch());

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
    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa,
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
    DwarfOperations.addInstruction(dwarf::DW_CFA_offset,
                                   CFIDirective.getRegister(),
                                   CFIDirective.getOffset() - CFAOffset);
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    DwarfOperations.addInstruction(dwarf::DW_CFA_def_cfa_offset,
                                   CFIDirective.getOffset() + CFAOffset);
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

using DWARFRegType = int64_t;

std::string RegisterCFIState::dump() {
  switch (RetrieveApproach) {
  case Undefined:
    return "undefined";
  case SameValue:
    return "same value";
  case AnotherRegister:
    return formatv("stored in another register, which is reg#{0}",
                   Info.Register);
  case OffsetFromCFAAddr:
    return formatv("offset {0} from CFA", Info.OffsetFromCFA);
  case OffsetFromCFAVal:
    return formatv("CFA value + {0}", Info.OffsetFromCFA);
  case Other:
    return "other";
  }
}

bool RegisterCFIState::operator==(const RegisterCFIState &OtherState) const {
  if (RetrieveApproach != OtherState.RetrieveApproach)
    return false;

  switch (RetrieveApproach) {
  case Undefined:
  case SameValue:
  case Other:
    return true;
  case AnotherRegister:
    return Info.Register == OtherState.Info.Register;
  case OffsetFromCFAAddr:
  case OffsetFromCFAVal:
    return Info.OffsetFromCFA == OtherState.Info.OffsetFromCFA;
  }
}

bool RegisterCFIState::operator!=(const RegisterCFIState &OtherState) const {
  return !(*this == OtherState);
}

RegisterCFIState RegisterCFIState::createUndefined() {
  RegisterCFIState State;
  State.RetrieveApproach = Undefined;

  return State;
}

RegisterCFIState RegisterCFIState::createSameValue() {
  RegisterCFIState State;
  State.RetrieveApproach = SameValue;

  return State;
}

RegisterCFIState
RegisterCFIState::createAnotherRegister(DWARFRegType Register) {
  RegisterCFIState State;
  State.RetrieveApproach = AnotherRegister;
  State.Info.Register = Register;

  return State;
}

RegisterCFIState RegisterCFIState::createOffsetFromCFAAddr(int OffsetFromCFA) {
  RegisterCFIState State;
  State.RetrieveApproach = OffsetFromCFAAddr;
  State.Info.OffsetFromCFA = OffsetFromCFA;

  return State;
}

RegisterCFIState RegisterCFIState::createOffsetFromCFAVal(int OffsetFromCFA) {
  RegisterCFIState State;
  State.RetrieveApproach = OffsetFromCFAVal;
  State.Info.OffsetFromCFA = OffsetFromCFA;

  return State;
}

RegisterCFIState RegisterCFIState::createOther() {
  RegisterCFIState State;
  State.RetrieveApproach = Other;

  return State;
}

CFIState::CFIState() : CFARegister(-1), CFAOffset(-1) {}

CFIState::CFIState(const CFIState &Other) {
  CFARegister = Other.CFARegister;
  CFAOffset = Other.CFAOffset;
  RegisterCFIStates = Other.RegisterCFIStates;
}

CFIState &CFIState::operator=(const CFIState &Other) {
  if (this != &Other) {
    CFARegister = Other.CFARegister;
    CFAOffset = Other.CFAOffset;
    RegisterCFIStates = Other.RegisterCFIStates;
  }

  return *this;
}

CFIState::CFIState(DWARFRegType CFARegister, int CFIOffset)
    : CFARegister(CFARegister), CFAOffset(CFIOffset) {}

std::optional<DWARFRegType>
CFIState::getReferenceRegisterForCallerValueOfRegister(DWARFRegType Reg) const {
  assert(RegisterCFIStates.count(Reg) &&
         "The register should be tracked inside the register states");
  auto &&RegState = RegisterCFIStates.at(Reg);
  switch (RegState.RetrieveApproach) {
  case RegisterCFIState::Undefined:
  case RegisterCFIState::Other:
    return std::nullopt;
  case RegisterCFIState::SameValue:
    return Reg;
  case RegisterCFIState::AnotherRegister:
    return RegState.Info.Register;
  case RegisterCFIState::OffsetFromCFAAddr:
  case RegisterCFIState::OffsetFromCFAVal:
    return CFARegister;
  }
}

bool CFIState::apply(const MCCFIInstruction &CFIDirective) {
  switch (CFIDirective.getOperation()) {
  case MCCFIInstruction::OpDefCfaRegister:
    CFARegister = CFIDirective.getRegister();
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    CFAOffset = CFIDirective.getOffset();
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    CFAOffset += CFIDirective.getOffset();
    break;
  case MCCFIInstruction::OpDefCfa:
    CFARegister = CFIDirective.getRegister();
    CFAOffset = CFIDirective.getOffset();
    break;
  case MCCFIInstruction::OpOffset:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createOffsetFromCFAAddr(CFIDirective.getOffset());
    break;
  case MCCFIInstruction::OpRegister:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createAnotherRegister(CFIDirective.getRegister2());
    break;
  case MCCFIInstruction::OpRelOffset:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createOffsetFromCFAAddr(CFIDirective.getOffset() -
                                                  CFAOffset);
    break;
  case MCCFIInstruction::OpUndefined:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createUndefined();
    break;
  case MCCFIInstruction::OpSameValue:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createSameValue();
    break;
  case MCCFIInstruction::OpValOffset:
    RegisterCFIStates[CFIDirective.getRegister()] =
        RegisterCFIState::createOffsetFromCFAVal(CFIDirective.getOffset());
    break;
  case MCCFIInstruction::OpRestoreState:
  case MCCFIInstruction::OpRememberState:
  case MCCFIInstruction::OpLLVMDefAspaceCfa:
  case MCCFIInstruction::OpRestore:
  case MCCFIInstruction::OpEscape:
  case MCCFIInstruction::OpWindowSave:
  case MCCFIInstruction::OpNegateRAState:
  case MCCFIInstruction::OpNegateRAStateWithPC:
  case MCCFIInstruction::OpGnuArgsSize:
  case MCCFIInstruction::OpLabel:
    // These instructions are not supported.
    return false;
    break;
  }

  return true;
}
