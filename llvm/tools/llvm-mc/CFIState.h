#ifndef LLVM_TOOLS_LLVM_MC_CFI_STATE_H
#define LLVM_TOOLS_LLVM_MC_CFI_STATE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <optional>
namespace llvm {

using DWARFRegType = int64_t;

struct RegisterCFIState {
  enum Approach {
    Undefined,
    SameValue,
    AnotherRegister,
    OffsetFromCFAAddr,
    OffsetFromCFAVal,
    Other,
  } RetrieveApproach;

  // TODO use a correct type for this.
  union {
    int OffsetFromCFA;
    DWARFRegType Register;
  } Info;

  bool operator==(const RegisterCFIState &OtherState) const {
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

  static RegisterCFIState createUndefined() {
    RegisterCFIState State;
    State.RetrieveApproach = Undefined;

    return State;
  }

  static RegisterCFIState createSameValue() {
    RegisterCFIState State;
    State.RetrieveApproach = SameValue;

    return State;
  }

  static RegisterCFIState createAnotherRegister(DWARFRegType Register) {
    RegisterCFIState State;
    State.RetrieveApproach = AnotherRegister;
    State.Info.Register = Register;

    return State;
  }

  static RegisterCFIState createOffsetFromCFAAddr(int OffsetFromCFA) {
    RegisterCFIState State;
    State.RetrieveApproach = OffsetFromCFAAddr;
    State.Info.OffsetFromCFA = OffsetFromCFA;

    return State;
  }

  static RegisterCFIState createOffsetFromCFAVal(int OffsetFromCFA) {
    RegisterCFIState State;
    State.RetrieveApproach = OffsetFromCFAVal;
    State.Info.OffsetFromCFA = OffsetFromCFA;

    return State;
  }

  static RegisterCFIState createOther() {
    RegisterCFIState State;
    State.RetrieveApproach = Other;

    return State;
  }
};
struct CFIState {
  DenseMap<DWARFRegType, RegisterCFIState> RegisterCFIStates;
  DWARFRegType CFARegister;
  int CFAOffset;

  CFIState() : CFARegister(-1), CFAOffset(-1) {}

  CFIState(const CFIState &Other) {
    CFARegister = Other.CFARegister;
    CFAOffset = Other.CFAOffset;
    RegisterCFIStates = Other.RegisterCFIStates;
  }

  CFIState &operator=(const CFIState &Other) {
    if (this != &Other) {
      CFARegister = Other.CFARegister;
      CFAOffset = Other.CFAOffset;
      RegisterCFIStates = Other.RegisterCFIStates;
    }

    return *this;
  }

  CFIState(DWARFRegType CFARegister, int CFIOffset)
      : CFARegister(CFARegister), CFAOffset(CFIOffset) {}

  std::optional<DWARFRegType>
  getReferenceRegisterForCallerValueOfRegister(DWARFRegType Reg) const {
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

  bool apply(const MCCFIInstruction &CFIDirective) {
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
};
} // namespace llvm

#endif