#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "CFIState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include <set>

namespace llvm {

class CFIAnalysis {
  MCContext *Context;
  MCInstrInfo const &MCII;
  MCRegisterInfo const *MCRI;
  CFIState State;
  bool IsEH;

private:
  // The CFI analysis only keeps track and cares about super registers, not the
  // subregisters. All reads to/writes from subregisters and considered the same
  // operation to super registers. Other operations like loading and stores are
  // considered only if they are exactly doing the operation to or from a super
  // register.
  // As en example, if you spill a sub register to stack, the CFI analysis does
  // not consider that a register spilling.
  bool isSuperReg(MCPhysReg Reg);

  SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>> getAllSuperRegs();

  MCPhysReg getSuperReg(MCPhysReg Reg);

public:
  CFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
              MCInstrAnalysis *MCIA, bool IsEH,
              ArrayRef<MCCFIInstruction> PrologueCFIDirectives);

  void update(const MCInst &Inst, ArrayRef<MCCFIInstruction> CFIDirectives);

private:
  void checkRegDiff(
      const MCInst &Inst, DWARFRegType Reg, const CFIState &PrevState,
      const CFIState &NextState,
      const dwarf::UnwindLocation
          &PrevRegState, // TODO maybe should get them from prev next state
      const dwarf::UnwindLocation
          &NextRegState, // TODO themselves instead of by arguments.
      const std::set<DWARFRegType> &Reads,
      const std::set<DWARFRegType> &Writes);

  void checkCFADiff(const MCInst &Inst, const CFIState &PrevState,
                    const CFIState &NextState,
                    const std::set<DWARFRegType> &Reads,
                    const std::set<DWARFRegType> &Writes);
};

} // namespace llvm
#endif