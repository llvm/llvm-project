#ifndef LLVM_TOOLS_LLVM_MC_CFI_STATE_H
#define LLVM_TOOLS_LLVM_MC_CFI_STATE_H

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include <optional>
namespace llvm {

using DWARFRegType = int64_t;

class CFIState {
private:
  MCContext *Context;
  // TODO the choice is questionable, should the state be the Table? or just a
  // TODO row? In the current code we are using only the last row.
  dwarf::UnwindTable State;

public:
  std::optional<dwarf::UnwindRow>
  getLastRow() const; // TODO maybe move it UnwindRow

  CFIState(MCContext *Context) : Context(Context) {};
  CFIState(const CFIState &Other);
  CFIState &operator=(const CFIState &Other);

  std::optional<DWARFRegType>
  getReferenceRegisterForCallerValueOfRegister(DWARFRegType Reg) const;

  void apply(const MCCFIInstruction &CFIDirective);

private:
  std::optional<dwarf::CFIProgram> convertMC2DWARF(
      MCCFIInstruction CFIDirective); // TODO maybe move it somewhere else
};
} // namespace llvm

#endif