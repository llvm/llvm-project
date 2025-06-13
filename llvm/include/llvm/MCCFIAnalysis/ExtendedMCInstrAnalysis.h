#ifndef LLVM_TOOLS_LLVM_MC_EXTENDED_MC_INSTR_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_EXTENDED_MC_INSTR_ANALYSIS_H

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

namespace llvm {

// TODO remove this class entirely
class ExtendedMCInstrAnalysis {
private:
public:
  ExtendedMCInstrAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                          MCInstrAnalysis *MCIA) {}

  /// Extra semantic information needed from MC layer:

  MCPhysReg getFlagsReg() const { return 57; }
};

} // namespace llvm

#endif