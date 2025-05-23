#include "llvm/MC/MCCFIAnalysis/CFIState.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

using namespace llvm;

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
