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
#include <cstdint>
#include <memory>
#include <optional>

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

  bool doesConstantChange(const MCInst &Inst, MCPhysReg Reg, int64_t &HowMuch) {
    if (isPush(Inst) && Reg == getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = -getPushSize(Inst);
      return true;
    }

    if (isPop(Inst) && Reg == getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = getPopSize(Inst);
      return true;
    }

    return false;
  }

  // Tries to guess Reg1's value in a form of Reg2 (before Inst's execution) +
  // Diff.
  bool isInConstantDistanceOfEachOther(const MCInst &Inst, MCPhysReg &Reg1,
                                       MCPhysReg Reg2, int &Diff) {
    {
      MCPhysReg From;
      MCPhysReg To;
      if (isRegToRegMove(Inst, From, To) && From == Reg2) {
        Reg1 = To;
        Diff = 0;
        return true;
      }
    }

    return false;
  }

  bool doStoreFromReg(const MCInst &Inst, MCPhysReg StoringReg,
                      MCPhysReg FromReg, int64_t &Offset) {
    if (isPush(Inst) && FromReg == getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      Offset = -getPushSize(Inst);
      return true;
    }

    {
      bool IsLoad;
      bool IsStore;
      bool IsStoreFromReg;
      MCPhysReg SrcReg;
      int32_t SrcImm;
      uint16_t StackPtrReg;
      int64_t StackOffset;
      uint8_t Size;
      bool IsSimple;
      bool IsIndexed;
      if (isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, SrcReg, SrcImm,
                        StackPtrReg, StackOffset, Size, IsSimple, IsIndexed)) {
        // TODO make sure that simple means that it's store and does nothing
        // more.
        if (IsStore && IsSimple && StackPtrReg == FromReg && IsStoreFromReg &&
            SrcReg == StoringReg) {
          Offset = StackOffset;
          return true;
        }
      }
    }

    return false;
  }

  bool doLoadFromReg(const MCInst &Inst, MCPhysReg FromReg, int64_t &Offset,
                     MCPhysReg &LoadingReg) {
    if (isPop(Inst) && FromReg == getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      Offset = 0;
      LoadingReg = Inst.getOperand(0).getReg();
      return true;
    }

    {
      bool IsLoad;
      bool IsStore;
      bool IsStoreFromReg;
      MCPhysReg SrcReg;
      int32_t SrcImm;
      uint16_t StackPtrReg;
      int64_t StackOffset;
      uint8_t Size;
      bool IsSimple;
      bool IsIndexed;
      if (isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, SrcReg, SrcImm,
                        StackPtrReg, StackOffset, Size, IsSimple, IsIndexed)) {
        // TODO make sure that simple means that it's store and does nothing
        // more.
        if (IsLoad && IsSimple && StackPtrReg == FromReg) {
          Offset = StackOffset;
          LoadingReg = SrcReg;
          return true;
        }
      }
    }

    {
      if (isMoveMem2Reg(Inst)) {
        auto X86MemAccess = evaluateX86MemoryOperand(Inst).value();
        if (X86MemAccess.BaseRegNum == FromReg &&
            (X86MemAccess.ScaleImm == 0 || X86MemAccess.IndexRegNum == 0) &&
            !X86MemAccess.DispExpr) {
          LoadingReg = Inst.getOperand(0).getReg();
          Offset = X86MemAccess.DispImm;
          return true;
        }
      }
    }

    return false;
  }

private:
  bool isPush(const MCInst &Inst) const { return MCPB->isPush(Inst); }
  int getPushSize(const MCInst &Inst) const { return MCPB->getPushSize(Inst); }

  bool isPop(const MCInst &Inst) const { return MCPB->isPop(Inst); }
  int getPopSize(const MCInst &Inst) const { return MCPB->getPopSize(Inst); }

  bool isRegToRegMove(const MCInst &Inst, MCPhysReg &From,
                      MCPhysReg &To) const {
    return MCPB->isRegToRegMove(Inst, From, To);
  }
  bool isConditionalMove(const MCInst &Inst) const {
    return MCPB->isConditionalMove(Inst);
  }
  bool isMoveMem2Reg(const MCInst &Inst) const {
    return MCPB->isMoveMem2Reg(Inst);
  }
  bool isSUB(const MCInst &Inst) const { return MCPB->isSUB(Inst); }

  int getMemoryOperandNo(const MCInst &Inst) const {
    return MCPB->getMemoryOperandNo(Inst);
  }
  std::optional<bolt::MCPlusBuilder::X86MemOperand>
  evaluateX86MemoryOperand(const MCInst &Inst) const {
    return MCPB->evaluateX86MemoryOperand(Inst);
  }
  bool isStackAccess(const MCInst &Inst, bool &IsLoad, bool &IsStore,
                     bool &IsStoreFromReg, MCPhysReg &Reg, int32_t &SrcImm,
                     uint16_t &StackPtrReg, int64_t &StackOffset, uint8_t &Size,
                     bool &IsSimple, bool &IsIndexed) const {
    return MCPB->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg,
                               SrcImm, StackPtrReg, StackOffset, Size, IsSimple,
                               IsIndexed);
  }
};

} // namespace llvm

#endif