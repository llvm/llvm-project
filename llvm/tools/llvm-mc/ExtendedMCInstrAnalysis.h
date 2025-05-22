#ifndef LLVM_TOOLS_LLVM_MC_EXTENDED_MC_INSTR_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_EXTENDED_MC_INSTR_ANALYSIS_H

#include "bolt/Core/MCPlusBuilder.h"
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
#include <memory>

namespace llvm {

class ExtendedMCInstrAnalysis {
private:
  std::unique_ptr<bolt::MCPlusBuilder> MCPB;

  static bolt::MCPlusBuilder *
  createMCPlusBuilder(const Triple::ArchType Arch,
                      const MCInstrAnalysis *Analysis, const MCInstrInfo *Info,
                      const MCRegisterInfo *RegInfo,
                      const MCSubtargetInfo *STI) {
    if (Arch == Triple::x86_64)
      return bolt::createX86MCPlusBuilder(Analysis, Info, RegInfo, STI);

    llvm_unreachable("architecture unsupported by ExtendedMCInstrAnalysis");
  }

public:
  ExtendedMCInstrAnalysis(MCContext &Context, MCInstrInfo const &MCII,
                          MCInstrAnalysis *MCIA) {
    MCPB.reset(createMCPlusBuilder(Context.getTargetTriple().getArch(), MCIA,
                                   &MCII, Context.getRegisterInfo(),
                                   Context.getSubtargetInfo()));
  }

  /// Extra semantic information needed from MC layer:

  MCPhysReg getStackPointer() const { return MCPB->getStackPointer(); }
  MCPhysReg getFlagsReg() const { return MCPB->getFlagsReg(); }
};

} // namespace llvm

#endif