#ifndef LLVM_TOOLS_LLVM_MC_CFI_STATE_H
#define LLVM_TOOLS_LLVM_MC_CFI_STATE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCDwarf.h"
#include <optional>
#include <string>
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

  union {
    int OffsetFromCFA;
    DWARFRegType Register;
  } Info;

  // TODO change it to be like other dump methods
  std::string dump();

  bool operator==(const RegisterCFIState &OtherState) const;
  bool operator!=(const RegisterCFIState &OtherState) const;

  static RegisterCFIState createUndefined();
  static RegisterCFIState createSameValue();
  static RegisterCFIState createAnotherRegister(DWARFRegType Register);
  static RegisterCFIState createOffsetFromCFAAddr(int OffsetFromCFA);
  static RegisterCFIState createOffsetFromCFAVal(int OffsetFromCFA);
  static RegisterCFIState createOther();
};

struct CFIState {
  DenseMap<DWARFRegType, RegisterCFIState> RegisterCFIStates;
  DWARFRegType CFARegister;
  int CFAOffset;

  CFIState();
  CFIState(const CFIState &Other);

  CFIState &operator=(const CFIState &Other);

  CFIState(DWARFRegType CFARegister, int CFIOffset);

  std::optional<DWARFRegType>
  getReferenceRegisterForCallerValueOfRegister(DWARFRegType Reg) const;

  bool apply(const MCCFIInstruction &CFIDirective);
};
} // namespace llvm

#endif