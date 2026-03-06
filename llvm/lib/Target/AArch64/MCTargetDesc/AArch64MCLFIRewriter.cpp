//===- AArch64MCLFIRewriter.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64MCLFIRewriter class, the AArch64 specific
// subclass of MCLFIRewriter.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCLFIRewriter.h"
#include "AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool>
    NoLFIGuardElim("aarch64-lfi-no-guard-elim", cl::Hidden,
                   cl::desc("Disable LFI guard elimination optimization"),
                   cl::init(false));

namespace {
// LFI reserved registers.
constexpr MCRegister LFIBaseReg = AArch64::X27;
constexpr MCRegister LFIAddrReg = AArch64::X28;
constexpr MCRegister LFIScratchReg = AArch64::X26;
constexpr MCRegister LFICtxReg = AArch64::X25;

// Offset into the context register block (pointed to by LFICtxReg) where the
// thread pointer is stored. This is a scaled offset (multiplied by 8 for
// 64-bit loads), so a value of 4 means an actual byte offset of 32.
constexpr unsigned LFITPOffset = 4;

unsigned convertUiToRoW(unsigned Op);
unsigned convertPreToRoW(unsigned Op);
unsigned convertPostToRoW(unsigned Op);
unsigned convertRoXToRoW(unsigned Op, unsigned &Shift);
bool getRoWShift(unsigned Op, unsigned &Shift);
unsigned getPrePostScale(unsigned Op);
unsigned convertPrePostToBase(unsigned Op, bool &IsPre, bool &IsNoOffset);
int getSIMDNaturalOffset(unsigned Op);

bool isSyscall(const MCInst &Inst) { return Inst.getOpcode() == AArch64::SVC; }

// Instructions that have mayLoad/mayStore set in TableGen but don't actually
// perform memory accesses (barriers, hints, waits).
bool isNotMemAccess(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::DMB:
  case AArch64::DSB:
  case AArch64::ISB:
  case AArch64::HINT:
    return true;
  default:
    return false;
  }
}

bool isTLSRead(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::MRS &&
         Inst.getOperand(1).getImm() == AArch64SysReg::TPIDR_EL0;
}

bool isTLSWrite(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::MSR &&
         Inst.getOperand(0).getImm() == AArch64SysReg::TPIDR_EL0;
}

bool mayPrefetch(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::PRFMl:
  case AArch64::PRFMroW:
  case AArch64::PRFMroX:
  case AArch64::PRFMui:
  case AArch64::PRFUMi:
    return true;
  default:
    return false;
  }
}

bool isPACIASP(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::PACIASP ||
         (Inst.getOpcode() == AArch64::HINT &&
          Inst.getOperand(0).getImm() == 25);
}

bool isDCZVA(const MCInst &Inst) {
  // DC ZVA is encoded as SYSxt with op1=3, Cn=7, Cm=4, op2=1
  if (Inst.getOpcode() != AArch64::SYSxt)
    return false;
  return Inst.getOperand(0).getImm() == 3 && // op1
         Inst.getOperand(1).getImm() == 7 && // Cn
         Inst.getOperand(2).getImm() == 4 && // Cm
         Inst.getOperand(3).getImm() == 1;   // op2
}

bool isAuthenticatedBranch(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::BRAA:
  case AArch64::BRAAZ:
  case AArch64::BRAB:
  case AArch64::BRABZ:
    return true;
  default:
    return false;
  }
}

bool isAuthenticatedCall(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::BLRAA:
  case AArch64::BLRAAZ:
  case AArch64::BLRAB:
  case AArch64::BLRABZ:
    return true;
  default:
    return false;
  }
}

bool isAuthenticatedReturn(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::RETAA:
  case AArch64::RETAB:
    return true;
  default:
    return false;
  }
}

bool isExceptionReturn(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::ERET:
  case AArch64::ERETAA:
  case AArch64::ERETAB:
    return true;
  default:
    return false;
  }
}

} // anonymous namespace

bool AArch64MCLFIRewriter::mayModifyStack(const MCInst &Inst) const {
  return mayModifyRegister(Inst, AArch64::SP);
}

bool AArch64MCLFIRewriter::mayModifyReserved(const MCInst &Inst) const {
  return mayModifyRegister(Inst, LFIAddrReg) ||
         mayModifyRegister(Inst, LFIBaseReg) ||
         mayModifyRegister(Inst, LFICtxReg);
}

bool AArch64MCLFIRewriter::mayModifyLR(const MCInst &Inst) const {
  // PACIASP signs LR but doesn't affect control flow safety.
  if (isPACIASP(Inst))
    return false;
  return mayModifyRegister(Inst, AArch64::LR);
}

void AArch64MCLFIRewriter::onLabel(const MCSymbol *Symbol, MCStreamer &Out) {
  // Flush deferred LR guard before a label, since labels are potential branch
  // targets and the code after the label may use LR for control flow.
  if (DeferredLRGuard && LastSTI) {
    emitAddMask(AArch64::LR, AArch64::LR, Out, *LastSTI);
    DeferredLRGuard = false;
  }

  // Reset the state for guard elimination.
  ActiveGuard = false;
}

void AArch64MCLFIRewriter::finish(MCStreamer &Out) {
  // Flush deferred LR guard at end of stream.
  if (DeferredLRGuard && LastSTI) {
    emitAddMask(AArch64::LR, AArch64::LR, Out, *LastSTI);
    DeferredLRGuard = false;
  }
}

void AArch64MCLFIRewriter::emitInst(const MCInst &Inst, MCStreamer &Out,
                                    const MCSubtargetInfo &STI) {
  // Guard elimination: invalidate guard if instruction modifies guarded
  // register, x28 (which holds the guarded value), or affects control flow.
  if (ActiveGuard) {
    const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());
    if (Desc.mayAffectControlFlow(Inst, *RegInfo) ||
        mayModifyRegister(Inst, ActiveGuardReg) ||
        mayModifyRegister(Inst, getWRegFromXReg(ActiveGuardReg)) ||
        mayModifyRegister(Inst, LFIAddrReg))
      ActiveGuard = false;
  }

  Out.emitInstruction(Inst, STI);
}

void AArch64MCLFIRewriter::emitAddMask(MCRegister Dest, MCRegister Src,
                                       MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  // Guard elimination: skip if same guard already active.
  if (!NoLFIGuardElim && Dest == LFIAddrReg && ActiveGuard &&
      ActiveGuardReg == Src)
    return;

  // add Dest, LFIBaseReg, W(Src), uxtw
  MCInst Inst;
  Inst.setOpcode(AArch64::ADDXrx);
  Inst.addOperand(MCOperand::createReg(Dest));
  Inst.addOperand(MCOperand::createReg(LFIBaseReg));
  Inst.addOperand(MCOperand::createReg(getWRegFromXReg(Src)));
  Inst.addOperand(
      MCOperand::createImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTW, 0)));
  emitInst(Inst, Out, STI);

  // Register Src as an actively guarded value.
  if (Dest == LFIAddrReg) {
    ActiveGuard = true;
    ActiveGuardReg = Src;
  }
}

void AArch64MCLFIRewriter::emitBranch(unsigned Opcode, MCRegister Target,
                                      MCStreamer &Out,
                                      const MCSubtargetInfo &STI) {
  MCInst Branch;
  Branch.setOpcode(Opcode);
  Branch.addOperand(MCOperand::createReg(Target));
  emitInst(Branch, Out, STI);
}

void AArch64MCLFIRewriter::emitMov(MCRegister Dest, MCRegister Src,
                                   MCStreamer &Out,
                                   const MCSubtargetInfo &STI) {
  // orr Dest, xzr, Src
  MCInst Inst;
  Inst.setOpcode(AArch64::ORRXrs);
  Inst.addOperand(MCOperand::createReg(Dest));
  Inst.addOperand(MCOperand::createReg(AArch64::XZR));
  Inst.addOperand(MCOperand::createReg(Src));
  Inst.addOperand(MCOperand::createImm(0));
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::emitAddImm(MCRegister Dest, MCRegister Src,
                                      int64_t Imm, MCStreamer &Out,
                                      const MCSubtargetInfo &STI) {
  assert(std::abs(Imm) <= 4095);
  MCInst Inst;
  if (Imm >= 0) {
    Inst.setOpcode(AArch64::ADDXri);
    Inst.addOperand(MCOperand::createReg(Dest));
    Inst.addOperand(MCOperand::createReg(Src));
    Inst.addOperand(MCOperand::createImm(Imm));
    Inst.addOperand(MCOperand::createImm(0)); // shift
  } else {
    Inst.setOpcode(AArch64::SUBXri);
    Inst.addOperand(MCOperand::createReg(Dest));
    Inst.addOperand(MCOperand::createReg(Src));
    Inst.addOperand(MCOperand::createImm(-Imm));
    Inst.addOperand(MCOperand::createImm(0)); // shift
  }
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::emitAddReg(MCRegister Dest, MCRegister Src1,
                                      MCRegister Src2, unsigned Shift,
                                      MCStreamer &Out,
                                      const MCSubtargetInfo &STI) {
  // add Dest, Src1, Src2, lsl #Shift
  MCInst Inst;
  Inst.setOpcode(AArch64::ADDXrs);
  Inst.addOperand(MCOperand::createReg(Dest));
  Inst.addOperand(MCOperand::createReg(Src1));
  Inst.addOperand(MCOperand::createReg(Src2));
  Inst.addOperand(
      MCOperand::createImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, Shift)));
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::emitAddRegExtend(MCRegister Dest, MCRegister Src1,
                                            MCRegister Src2,
                                            AArch64_AM::ShiftExtendType ExtType,
                                            unsigned Shift, MCStreamer &Out,
                                            const MCSubtargetInfo &STI) {
  // add Dest, Src1, Src2, ExtType #Shift
  MCInst Inst;
  if (ExtType == AArch64_AM::SXTX || ExtType == AArch64_AM::UXTX)
    Inst.setOpcode(AArch64::ADDXrx64);
  else
    Inst.setOpcode(AArch64::ADDXrx);
  Inst.addOperand(MCOperand::createReg(Dest));
  Inst.addOperand(MCOperand::createReg(Src1));
  Inst.addOperand(MCOperand::createReg(Src2));
  Inst.addOperand(
      MCOperand::createImm(AArch64_AM::getArithExtendImm(ExtType, Shift)));
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::emitMemRoW(unsigned Opcode, const MCOperand &DataOp,
                                      MCRegister BaseReg, MCStreamer &Out,
                                      const MCSubtargetInfo &STI) {
  // Emits: Op DataOp, [LFIBaseReg, W(BaseReg), uxtw].
  MCInst Inst;
  Inst.setOpcode(Opcode);
  Inst.addOperand(DataOp);
  Inst.addOperand(MCOperand::createReg(LFIBaseReg));
  Inst.addOperand(MCOperand::createReg(getWRegFromXReg(BaseReg)));
  Inst.addOperand(MCOperand::createImm(0)); // S bit = 0 (UXTW).
  Inst.addOperand(MCOperand::createImm(0)); // Shift amount = 0 (unscaled).
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::rewriteIndirectBranch(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  assert(Inst.getOperand(0).isReg());
  MCRegister BranchReg = Inst.getOperand(0).getReg();

  // Guard the branch target through X28.
  emitAddMask(LFIAddrReg, BranchReg, Out, STI);
  emitBranch(Inst.getOpcode(), LFIAddrReg, Out, STI);
}

void AArch64MCLFIRewriter::rewriteCall(const MCInst &Inst, MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  if (Inst.getOperand(0).isReg())
    rewriteIndirectBranch(Inst, Out, STI);
  else
    emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::rewriteReturn(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  if (isExceptionReturn(Inst.getOpcode())) {
    error(Inst, "exception returns (ERET/ERETAA/ERETAB) are not "
                "supported by LFI");
    return;
  }

  // Regular RET has an operand, handle it normally.
  assert(Inst.getNumOperands() > 0 && Inst.getOperand(0).isReg());
  // RET through LR is safe since LR is always within sandbox.
  if (Inst.getOperand(0).getReg() != AArch64::LR)
    rewriteIndirectBranch(Inst, Out, STI);
  else
    emitInst(Inst, Out, STI);
}

bool AArch64MCLFIRewriter::rewriteLoadStoreRoW(const MCInst &Inst,
                                               MCStreamer &Out,
                                               const MCSubtargetInfo &STI) {
  unsigned Op = Inst.getOpcode();
  unsigned MemOp;

  // Case 1: Indexed load/store with immediate offset.
  // ldr xN, [xM, #0] -> ldr xN, [x27, wM, uxtw]
  // ldr xN, [xM, #imm] -> fall back to basic (non-zero offset)
  if ((MemOp = convertUiToRoW(Op)) != AArch64::INSTRUCTION_LIST_END) {
    MCRegister BaseReg = Inst.getOperand(1).getReg();
    if (BaseReg == AArch64::SP)
      return false;
    const MCOperand &OffsetOp = Inst.getOperand(2);
    if (OffsetOp.isImm() && OffsetOp.getImm() == 0) {
      emitMemRoW(MemOp, Inst.getOperand(0), BaseReg, Out, STI);
      return true;
    }
    return false;
  }

  // Case 2: Pre-index load/store.
  // ldr xN, [xM, #imm]! -> add xM, xM, #imm; ldr xN, [x27, wM, uxtw]
  // Pre-index: update base before the access.
  if ((MemOp = convertPreToRoW(Op)) != AArch64::INSTRUCTION_LIST_END) {
    MCRegister BaseReg = Inst.getOperand(2).getReg();
    if (BaseReg == AArch64::SP)
      return false;
    int64_t Imm = Inst.getOperand(3).getImm();
    emitAddImm(BaseReg, BaseReg, Imm, Out, STI);
    emitMemRoW(MemOp, Inst.getOperand(1), BaseReg, Out, STI);
    return true;
  }

  // Case 3: Post-index load/store.
  // ldr xN, [xM], #imm -> ldr xN, [x27, wM, uxtw]; add xM, xM, #imm
  // Post-index: update base after the access.
  if ((MemOp = convertPostToRoW(Op)) != AArch64::INSTRUCTION_LIST_END) {
    MCRegister BaseReg = Inst.getOperand(2).getReg();
    if (BaseReg == AArch64::SP)
      return false;
    int64_t Imm = Inst.getOperand(3).getImm();
    emitMemRoW(MemOp, Inst.getOperand(1), BaseReg, Out, STI);
    emitAddImm(BaseReg, BaseReg, Imm, Out, STI);
    return true;
  }

  // Case 4: Register-offset-X load/store.
  // ldr xN, [xM1, xM2] -> add x26, xM1, xM2; ldr xN, [x27, w26, uxtw]
  // ldr xN, [xM1, xM2, sxtx #shift] -> add x26, xM1, xM2, sxtx #shift; ldr xN,
  // [x27, w26, uxtw]
  unsigned Shift;
  if ((MemOp = convertRoXToRoW(Op, Shift)) != AArch64::INSTRUCTION_LIST_END) {
    MCRegister Reg1 = Inst.getOperand(1).getReg();
    MCRegister Reg2 = Inst.getOperand(2).getReg();
    int64_t Extend = Inst.getOperand(3).getImm();
    int64_t IsShift = Inst.getOperand(4).getImm();

    if (!IsShift)
      Shift = 0;

    if (Extend) {
      // Sign-extend: add Scratch, Reg1, Reg2, sxtx #Shift
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::SXTX, Shift, Out,
                       STI);
    } else {
      // No extend: add Scratch, Reg1, Reg2, lsl #Shift
      emitAddReg(LFIScratchReg, Reg1, Reg2, Shift, Out, STI);
    }
    emitMemRoW(MemOp, Inst.getOperand(0), LFIScratchReg, Out, STI);
    return true;
  }

  // Case 5: Register-offset-W load/store.
  // ldr xN, [xM1, wM2, uxtw] -> add x26, xM1, wM2, uxtw;
  //                             ldr xN, [x27, w26, uxtw]
  // ldr xN, [xM1, wM2, sxtw #shift] -> add x26, xM1, wM2, sxtw #shift;
  //                                    ldr xN, [x27, w26, uxtw]
  if (getRoWShift(Op, Shift)) {
    MemOp = Op;
    MCRegister Reg1 = Inst.getOperand(1).getReg();
    MCRegister Reg2 = Inst.getOperand(2).getReg();
    int64_t S = Inst.getOperand(3).getImm();
    int64_t IsShift = Inst.getOperand(4).getImm();

    if (!IsShift)
      Shift = 0;

    if (S) {
      // Sign-extend: add Scratch, Reg1, Reg2, sxtw #Shift
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::SXTW, Shift, Out,
                       STI);
    } else {
      // Unsigned extend: add Scratch, Reg1, Reg2, uxtw #Shift
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::UXTW, Shift, Out,
                       STI);
    }
    emitMemRoW(MemOp, Inst.getOperand(0), LFIScratchReg, Out, STI);
    return true;
  }

  return false;
}

void AArch64MCLFIRewriter::rewriteLoadStoreBasic(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;
  unsigned Opcode = Inst.getOpcode();

  uint64_t AddrMode = TSFlags & AArch64::MemOpAddrModeMask;
  if (AddrMode == AArch64::MemOpAddrModeLiteral) {
    error(Inst, "PC-relative literal loads are not supported in LFI");
    return;
  }

  int BaseIdx = AArch64::getMemOpBaseRegIdx(TSFlags);
  if (BaseIdx < 0) {
    warning(Inst, "memory instruction not sandboxed: unknown addressing mode");
    emitInst(Inst, Out, STI);
    return;
  }

  MCRegister BaseReg = Inst.getOperand(BaseIdx).getReg();

  // Stack accesses without register offset don't need rewriting.
  if (BaseReg == AArch64::SP) {
    int OffsetIdx = AArch64::getMemOpOffsetIdx(TSFlags);
    if (OffsetIdx < 0 || !Inst.getOperand(OffsetIdx).isReg()) {
      emitInst(Inst, Out, STI);
      return;
    }
  }

  // Guard the base register.
  emitAddMask(LFIAddrReg, BaseReg, Out, STI);

  // Check if this is a pre/post-index instruction that needs special handling.
  bool IsPrePostIdx = AArch64::isMemOpPrePostIdx(TSFlags);
  bool IsPre = false;
  bool IsNoOffset = false;
  unsigned BaseOpcode = convertPrePostToBase(Opcode, IsPre, IsNoOffset);

  if (IsPrePostIdx && BaseOpcode != AArch64::INSTRUCTION_LIST_END) {
    // This is a pair instruction (LDP/STP) with pre/post-index.
    // We need to demote it to the base indexed form.
    //
    // For pre-index:  ldp x0, x1, [x2, #16]! -> ldp x0, x1, [x28, #16];
    //                                           add x2, x2, #16
    // For post-index: ldp x0, x1, [x2], #16  -> ldp x0, x1, [x28];
    //                                           add x2, x2, #16
    MCInst NewInst;
    NewInst.setOpcode(BaseOpcode);
    NewInst.setLoc(Inst.getLoc());

    // Copy operands up to (but not including) the base register.
    // For LDPXpre: operands are [wback, Rt, Rt2, Rn, #imm]
    // We skip wback (operand 0) and copy Rt, Rt2, then add LFIAddrReg.
    for (int I = 1; I < BaseIdx; ++I)
      NewInst.addOperand(Inst.getOperand(I));

    // Add the guarded base register.
    NewInst.addOperand(MCOperand::createReg(LFIAddrReg));

    // For pre-index, include the offset; for post-index, use zero.
    int OffsetIdx = AArch64::getMemOpOffsetIdx(TSFlags);
    if (IsPre && OffsetIdx >= 0) {
      NewInst.addOperand(Inst.getOperand(OffsetIdx));
    } else if (!IsNoOffset) {
      NewInst.addOperand(MCOperand::createImm(0));
    }
    emitInst(NewInst, Out, STI);

    // Update the base register with the scaled offset.
    if (OffsetIdx >= 0) {
      const MCOperand &OffsetOp = Inst.getOperand(OffsetIdx);
      if (OffsetOp.isImm()) {
        int64_t Scale = getPrePostScale(Opcode);
        int64_t Offset = OffsetOp.getImm() * Scale;
        emitAddImm(BaseReg, BaseReg, Offset, Out, STI);
      } else if (OffsetOp.isReg()) {
        // SIMD post-index uses a register offset (XZR for natural offset).
        MCRegister OffReg = OffsetOp.getReg();
        if (OffReg == AArch64::XZR) {
          int NaturalOffset = getSIMDNaturalOffset(Opcode);
          if (NaturalOffset > 0) {
            emitAddImm(BaseReg, BaseReg, NaturalOffset, Out, STI);
          }
        } else if (OffReg != AArch64::WZR) {
          // Regular register offset.
          emitAddReg(BaseReg, BaseReg, OffReg, 0, Out, STI);
        }
      }
    }
  } else if (IsPrePostIdx) {
    // All scalar pre/post-index instructions are handled by
    // rewriteLoadStoreRoW, and all pair/SIMD pre/post-index instructions are
    // handled above. This path should not be reachable.
    error(Inst, "unhandled pre/post-index instruction without uxtw form in LFI "
                "rewriter");
  } else {
    // Non-pre/post instruction: just replace the base register.
    MCInst NewInst;
    NewInst.setOpcode(Opcode);
    NewInst.setLoc(Inst.getLoc());
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      if ((int)I == BaseIdx) {
        NewInst.addOperand(MCOperand::createReg(LFIAddrReg));
      } else {
        NewInst.addOperand(Inst.getOperand(I));
      }
    }
    emitInst(NewInst, Out, STI);
  }
}

void AArch64MCLFIRewriter::rewriteLoadStore(const MCInst &Inst, MCStreamer &Out,
                                            const MCSubtargetInfo &STI) {
  bool IsStore = mayStore(Inst);
  bool IsLoad = mayLoad(Inst) || mayPrefetch(Inst);

  // Check if this memory access needs sandboxing based on LFI mode.
  // - Default: sandbox both loads and stores
  // - +no-lfi-loads: stores-only mode, skip loads
  // - +no-lfi-loads+no-lfi-stores: jumps-only mode, skip all memory
  bool SkipLoads = STI.hasFeature(AArch64::FeatureNoLFILoads);
  bool SkipStores = STI.hasFeature(AArch64::FeatureNoLFIStores);

  if ((!IsLoad || SkipLoads) && (!IsStore || SkipStores)) {
    emitInst(Inst, Out, STI);
    return;
  }

  // Try RoW optimization first, then fall back to basic rewriting.
  if (rewriteLoadStoreRoW(Inst, Out, STI))
    return;

  rewriteLoadStoreBasic(Inst, Out, STI);
}

void AArch64MCLFIRewriter::rewriteStackModification(
    const MCInst &Inst, MCStreamer &Out, const MCSubtargetInfo &STI) {
  // If this is a load/store that also modifies SP (like push/pop patterns),
  // handle the memory access first.
  if (mayLoad(Inst) || mayStore(Inst)) {
    if (mayModifyLR(Inst))
      return rewriteLRModification(Inst, Out, STI);
    emitInst(Inst, Out, STI);
    return;
  }

  // In jumps-only mode (+no-lfi-loads+no-lfi-stores), no stack sandboxing
  // needed.
  bool SkipLoads = STI.hasFeature(AArch64::FeatureNoLFILoads);
  bool SkipStores = STI.hasFeature(AArch64::FeatureNoLFIStores);
  if (SkipLoads && SkipStores) {
    emitInst(Inst, Out, STI);
    return;
  }

  // Redirect SP modification to scratch, then sandbox.
  MCInst ModInst;
  ModInst.setOpcode(Inst.getOpcode());
  ModInst.setLoc(Inst.getLoc());

  assert(Inst.getOperand(0).isReg() &&
         Inst.getOperand(0).getReg() == AArch64::SP);

  ModInst.addOperand(MCOperand::createReg(LFIScratchReg));
  for (unsigned I = 1, E = Inst.getNumOperands(); I != E; ++I)
    ModInst.addOperand(Inst.getOperand(I));

  emitInst(ModInst, Out, STI);
  emitAddMask(AArch64::SP, LFIScratchReg, Out, STI);
}

void AArch64MCLFIRewriter::rewriteLRModification(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  // Emit the instruction with memory sandboxing if needed.
  if (mayLoad(Inst) || mayStore(Inst))
    rewriteLoadStore(Inst, Out, STI);
  else
    emitInst(Inst, Out, STI);

  // Defer the LR guard until the next control flow instruction.
  //
  // This allows for compatibility with PAC authentication by allowing for the
  // authentication instruction to run before the mask (which destroys the PAC
  // bits).
  DeferredLRGuard = true;
}

void AArch64MCLFIRewriter::rewriteAuthenticatedReturn(
    const MCInst &Inst, MCStreamer &Out, const MCSubtargetInfo &STI) {
  // Expand RETAA/RETAB to: AUTIASP/AUTIBSP, guard LR, RET
  unsigned Opcode = Inst.getOpcode();

  // Emit the appropriate AUTxSP instruction.
  MCInst Auth;
  if (Opcode == AArch64::RETAA)
    Auth.setOpcode(AArch64::AUTIASP);
  else
    Auth.setOpcode(AArch64::AUTIBSP);
  emitInst(Auth, Out, STI);

  // Guard LR and emit RET.
  emitAddMask(AArch64::LR, AArch64::LR, Out, STI);

  MCInst Ret;
  Ret.setOpcode(AArch64::RET);
  Ret.addOperand(MCOperand::createReg(AArch64::LR));
  emitInst(Ret, Out, STI);
}

void AArch64MCLFIRewriter::rewriteAuthenticatedBranchOrCall(
    const MCInst &Inst, unsigned BranchOpcode, MCStreamer &Out,
    const MCSubtargetInfo &STI) {
  unsigned Opcode = Inst.getOpcode();
  MCRegister TargetReg = Inst.getOperand(0).getReg();

  MCInst Auth;
  switch (Opcode) {
  case AArch64::BRAA:
  case AArch64::BLRAA:
    Auth.setOpcode(AArch64::AUTIA);
    Auth.addOperand(MCOperand::createReg(TargetReg)); // dst
    Auth.addOperand(MCOperand::createReg(TargetReg)); // src (tied)
    Auth.addOperand(Inst.getOperand(1));              // modifier
    break;
  case AArch64::BRAAZ:
  case AArch64::BLRAAZ:
    Auth.setOpcode(AArch64::AUTIZA);
    Auth.addOperand(MCOperand::createReg(TargetReg)); // dst
    Auth.addOperand(MCOperand::createReg(TargetReg)); // src (tied)
    break;
  case AArch64::BRAB:
  case AArch64::BLRAB:
    Auth.setOpcode(AArch64::AUTIB);
    Auth.addOperand(MCOperand::createReg(TargetReg)); // dst
    Auth.addOperand(MCOperand::createReg(TargetReg)); // src (tied)
    Auth.addOperand(Inst.getOperand(1));              // modifier
    break;
  case AArch64::BRABZ:
  case AArch64::BLRABZ:
    Auth.setOpcode(AArch64::AUTIZB);
    Auth.addOperand(MCOperand::createReg(TargetReg)); // dst
    Auth.addOperand(MCOperand::createReg(TargetReg)); // src (tied)
    break;
  default:
    llvm_unreachable("unexpected authenticated branch/call opcode");
  }
  emitInst(Auth, Out, STI);

  // Guard the target and branch/call.
  emitAddMask(LFIAddrReg, TargetReg, Out, STI);
  emitBranch(BranchOpcode, LFIAddrReg, Out, STI);
}

void AArch64MCLFIRewriter::emitSyscall(MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  // Save LR to scratch.
  emitMov(LFIScratchReg, AArch64::LR, Out, STI);

  // Load syscall handler address from negative offset from sandbox base.
  MCInst Load;
  Load.setOpcode(AArch64::LDURXi);
  Load.addOperand(MCOperand::createReg(AArch64::LR));
  Load.addOperand(MCOperand::createReg(LFIBaseReg));
  Load.addOperand(MCOperand::createImm(-8));
  emitInst(Load, Out, STI);

  // Call the runtime.
  emitBranch(AArch64::BLR, AArch64::LR, Out, STI);

  // Restore LR with guard.
  emitAddMask(AArch64::LR, LFIScratchReg, Out, STI);
}

void AArch64MCLFIRewriter::rewriteSyscall(const MCInst &, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  emitSyscall(Out, STI);
}

void AArch64MCLFIRewriter::rewriteTLSRead(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  // mrs xN, tpidr_el0 -> ldr xN, [x25, #TP]
  MCRegister DestReg = Inst.getOperand(0).getReg();

  MCInst Load;
  Load.setOpcode(AArch64::LDRXui);
  Load.addOperand(MCOperand::createReg(DestReg));
  Load.addOperand(MCOperand::createReg(LFICtxReg));
  Load.addOperand(MCOperand::createImm(LFITPOffset));
  emitInst(Load, Out, STI);
}

void AArch64MCLFIRewriter::rewriteTLSWrite(const MCInst &Inst, MCStreamer &Out,
                                           const MCSubtargetInfo &STI) {
  // msr tpidr_el0, xN -> str xN, [x25, #TP]
  MCRegister SrcReg = Inst.getOperand(1).getReg();

  MCInst Store;
  Store.setOpcode(AArch64::STRXui);
  Store.addOperand(MCOperand::createReg(SrcReg));
  Store.addOperand(MCOperand::createReg(LFICtxReg));
  Store.addOperand(MCOperand::createImm(LFITPOffset));
  emitInst(Store, Out, STI);
}

void AArch64MCLFIRewriter::rewriteDCZVA(const MCInst &Inst, MCStreamer &Out,
                                        const MCSubtargetInfo &STI) {
  // dc zva, xN -> add x28, x27, wN, uxtw; dc zva, x28
  MCRegister AddrReg = Inst.getOperand(4).getReg();

  emitAddMask(LFIAddrReg, AddrReg, Out, STI);

  MCInst NewInst;
  NewInst.setOpcode(AArch64::SYSxt);
  NewInst.addOperand(Inst.getOperand(0)); // op1
  NewInst.addOperand(Inst.getOperand(1)); // Cn
  NewInst.addOperand(Inst.getOperand(2)); // Cm
  NewInst.addOperand(Inst.getOperand(3)); // op2
  NewInst.addOperand(MCOperand::createReg(LFIAddrReg));
  emitInst(NewInst, Out, STI);
}

void AArch64MCLFIRewriter::doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  // Reserved register modification is an error.
  if (mayModifyReserved(Inst)) {
    error(Inst, "illegal modification of reserved LFI register");
    return;
  }

  // System instructions.
  if (isSyscall(Inst))
    return rewriteSyscall(Inst, Out, STI);

  if (isTLSRead(Inst))
    return rewriteTLSRead(Inst, Out, STI);

  if (isTLSWrite(Inst))
    return rewriteTLSWrite(Inst, Out, STI);

  if (isDCZVA(Inst))
    return rewriteDCZVA(Inst, Out, STI);

  // Authenticated PAC instructions are expanded to their component operations.
  if (isAuthenticatedReturn(Inst.getOpcode()))
    return rewriteAuthenticatedReturn(Inst, Out, STI);

  if (isAuthenticatedBranch(Inst.getOpcode()))
    return rewriteAuthenticatedBranchOrCall(Inst, AArch64::BR, Out, STI);

  if (isAuthenticatedCall(Inst.getOpcode()))
    return rewriteAuthenticatedBranchOrCall(Inst, AArch64::BLR, Out, STI);

  // Emit deferred LR guard before control flow instructions.
  if (DeferredLRGuard) {
    if (isReturn(Inst) || isIndirectBranch(Inst) || isCall(Inst) ||
        isBranch(Inst)) {
      emitAddMask(AArch64::LR, AArch64::LR, Out, STI);
      DeferredLRGuard = false;
    }
  }

  // Control flow.
  if (isReturn(Inst))
    return rewriteReturn(Inst, Out, STI);

  if (isIndirectBranch(Inst))
    return rewriteIndirectBranch(Inst, Out, STI);

  if (isCall(Inst))
    return rewriteCall(Inst, Out, STI);

  if (isBranch(Inst))
    return emitInst(Inst, Out, STI);

  // Register modifications that require sandboxing.
  if (mayModifyStack(Inst))
    return rewriteStackModification(Inst, Out, STI);

  if (mayModifyLR(Inst))
    return rewriteLRModification(Inst, Out, STI);

  if (!isNotMemAccess(Inst) &&
      (mayLoad(Inst) || mayStore(Inst) || mayPrefetch(Inst)))
    return rewriteLoadStore(Inst, Out, STI);

  emitInst(Inst, Out, STI);
}

bool AArch64MCLFIRewriter::rewriteInst(const MCInst &Inst, MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  if (!Enabled || Guard)
    return false;
  Guard = true;
  LastSTI = &STI;

  doRewriteInst(Inst, Out, STI);

  Guard = false;
  return true;
}

namespace {

// RoW (Register-offset-W) Opcode Conversion Tables
//
// These tables convert various load/store addressing modes to the
// register-offset-W form ([X27, Wn, uxtw]) which provides sandboxing in a
// single instruction by zero-extending the 32-bit offset register.

// Convert indexed (ui) load/store to RoW form.
// Example: LDRXui -> LDRXroW
unsigned convertUiToRoW(unsigned Op) {
  switch (Op) {
  case AArch64::LDRBBui:
    return AArch64::LDRBBroW;
  case AArch64::LDRBui:
    return AArch64::LDRBroW;
  case AArch64::LDRDui:
    return AArch64::LDRDroW;
  case AArch64::LDRHHui:
    return AArch64::LDRHHroW;
  case AArch64::LDRHui:
    return AArch64::LDRHroW;
  case AArch64::LDRQui:
    return AArch64::LDRQroW;
  case AArch64::LDRSBWui:
    return AArch64::LDRSBWroW;
  case AArch64::LDRSBXui:
    return AArch64::LDRSBXroW;
  case AArch64::LDRSHWui:
    return AArch64::LDRSHWroW;
  case AArch64::LDRSHXui:
    return AArch64::LDRSHXroW;
  case AArch64::LDRSWui:
    return AArch64::LDRSWroW;
  case AArch64::LDRSui:
    return AArch64::LDRSroW;
  case AArch64::LDRWui:
    return AArch64::LDRWroW;
  case AArch64::LDRXui:
    return AArch64::LDRXroW;
  case AArch64::PRFMui:
    return AArch64::PRFMroW;
  case AArch64::STRBBui:
    return AArch64::STRBBroW;
  case AArch64::STRBui:
    return AArch64::STRBroW;
  case AArch64::STRDui:
    return AArch64::STRDroW;
  case AArch64::STRHHui:
    return AArch64::STRHHroW;
  case AArch64::STRHui:
    return AArch64::STRHroW;
  case AArch64::STRQui:
    return AArch64::STRQroW;
  case AArch64::STRSui:
    return AArch64::STRSroW;
  case AArch64::STRWui:
    return AArch64::STRWroW;
  case AArch64::STRXui:
    return AArch64::STRXroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

// Convert pre-index load/store to RoW form.
unsigned convertPreToRoW(unsigned Op) {
  switch (Op) {
  case AArch64::LDRBBpre:
    return AArch64::LDRBBroW;
  case AArch64::LDRBpre:
    return AArch64::LDRBroW;
  case AArch64::LDRDpre:
    return AArch64::LDRDroW;
  case AArch64::LDRHHpre:
    return AArch64::LDRHHroW;
  case AArch64::LDRHpre:
    return AArch64::LDRHroW;
  case AArch64::LDRQpre:
    return AArch64::LDRQroW;
  case AArch64::LDRSBWpre:
    return AArch64::LDRSBWroW;
  case AArch64::LDRSBXpre:
    return AArch64::LDRSBXroW;
  case AArch64::LDRSHWpre:
    return AArch64::LDRSHWroW;
  case AArch64::LDRSHXpre:
    return AArch64::LDRSHXroW;
  case AArch64::LDRSWpre:
    return AArch64::LDRSWroW;
  case AArch64::LDRSpre:
    return AArch64::LDRSroW;
  case AArch64::LDRWpre:
    return AArch64::LDRWroW;
  case AArch64::LDRXpre:
    return AArch64::LDRXroW;
  case AArch64::STRBBpre:
    return AArch64::STRBBroW;
  case AArch64::STRBpre:
    return AArch64::STRBroW;
  case AArch64::STRDpre:
    return AArch64::STRDroW;
  case AArch64::STRHHpre:
    return AArch64::STRHHroW;
  case AArch64::STRHpre:
    return AArch64::STRHroW;
  case AArch64::STRQpre:
    return AArch64::STRQroW;
  case AArch64::STRSpre:
    return AArch64::STRSroW;
  case AArch64::STRWpre:
    return AArch64::STRWroW;
  case AArch64::STRXpre:
    return AArch64::STRXroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

// Convert post-index load/store to RoW form.
unsigned convertPostToRoW(unsigned Op) {
  switch (Op) {
  case AArch64::LDRBBpost:
    return AArch64::LDRBBroW;
  case AArch64::LDRBpost:
    return AArch64::LDRBroW;
  case AArch64::LDRDpost:
    return AArch64::LDRDroW;
  case AArch64::LDRHHpost:
    return AArch64::LDRHHroW;
  case AArch64::LDRHpost:
    return AArch64::LDRHroW;
  case AArch64::LDRQpost:
    return AArch64::LDRQroW;
  case AArch64::LDRSBWpost:
    return AArch64::LDRSBWroW;
  case AArch64::LDRSBXpost:
    return AArch64::LDRSBXroW;
  case AArch64::LDRSHWpost:
    return AArch64::LDRSHWroW;
  case AArch64::LDRSHXpost:
    return AArch64::LDRSHXroW;
  case AArch64::LDRSWpost:
    return AArch64::LDRSWroW;
  case AArch64::LDRSpost:
    return AArch64::LDRSroW;
  case AArch64::LDRWpost:
    return AArch64::LDRWroW;
  case AArch64::LDRXpost:
    return AArch64::LDRXroW;
  case AArch64::STRBBpost:
    return AArch64::STRBBroW;
  case AArch64::STRBpost:
    return AArch64::STRBroW;
  case AArch64::STRDpost:
    return AArch64::STRDroW;
  case AArch64::STRHHpost:
    return AArch64::STRHHroW;
  case AArch64::STRHpost:
    return AArch64::STRHroW;
  case AArch64::STRQpost:
    return AArch64::STRQroW;
  case AArch64::STRSpost:
    return AArch64::STRSroW;
  case AArch64::STRWpost:
    return AArch64::STRWroW;
  case AArch64::STRXpost:
    return AArch64::STRXroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

// Convert register-offset-X to RoW form, also returns the shift amount.
unsigned convertRoXToRoW(unsigned Op, unsigned &Shift) {
  Shift = 0;
  switch (Op) {
  case AArch64::LDRBBroX:
    return AArch64::LDRBBroW;
  case AArch64::LDRBroX:
    return AArch64::LDRBroW;
  case AArch64::LDRDroX:
    Shift = 3;
    return AArch64::LDRDroW;
  case AArch64::LDRHHroX:
    Shift = 1;
    return AArch64::LDRHHroW;
  case AArch64::LDRHroX:
    Shift = 1;
    return AArch64::LDRHroW;
  case AArch64::LDRQroX:
    Shift = 4;
    return AArch64::LDRQroW;
  case AArch64::LDRSBWroX:
    return AArch64::LDRSBWroW;
  case AArch64::LDRSBXroX:
    return AArch64::LDRSBXroW;
  case AArch64::LDRSHWroX:
    Shift = 1;
    return AArch64::LDRSHWroW;
  case AArch64::LDRSHXroX:
    Shift = 1;
    return AArch64::LDRSHXroW;
  case AArch64::LDRSWroX:
    Shift = 2;
    return AArch64::LDRSWroW;
  case AArch64::LDRSroX:
    Shift = 2;
    return AArch64::LDRSroW;
  case AArch64::LDRWroX:
    Shift = 2;
    return AArch64::LDRWroW;
  case AArch64::LDRXroX:
    Shift = 3;
    return AArch64::LDRXroW;
  case AArch64::PRFMroX:
    Shift = 3;
    return AArch64::PRFMroW;
  case AArch64::STRBBroX:
    return AArch64::STRBBroW;
  case AArch64::STRBroX:
    return AArch64::STRBroW;
  case AArch64::STRDroX:
    Shift = 3;
    return AArch64::STRDroW;
  case AArch64::STRHHroX:
    Shift = 1;
    return AArch64::STRHHroW;
  case AArch64::STRHroX:
    Shift = 1;
    return AArch64::STRHroW;
  case AArch64::STRQroX:
    Shift = 4;
    return AArch64::STRQroW;
  case AArch64::STRSroX:
    Shift = 2;
    return AArch64::STRSroW;
  case AArch64::STRWroX:
    Shift = 2;
    return AArch64::STRWroW;
  case AArch64::STRXroX:
    Shift = 3;
    return AArch64::STRXroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

// Check if Op is a register-offset-W instruction and return its shift amount.
// Returns true if recognized, false otherwise.
bool getRoWShift(unsigned Op, unsigned &Shift) {
  Shift = 0;
  switch (Op) {
  case AArch64::LDRBBroW:
  case AArch64::LDRBroW:
  case AArch64::LDRSBWroW:
  case AArch64::LDRSBXroW:
  case AArch64::STRBBroW:
  case AArch64::STRBroW:
    return true;
  case AArch64::LDRHHroW:
  case AArch64::LDRHroW:
  case AArch64::LDRSHWroW:
  case AArch64::LDRSHXroW:
  case AArch64::STRHHroW:
  case AArch64::STRHroW:
    Shift = 1;
    return true;
  case AArch64::LDRSWroW:
  case AArch64::LDRSroW:
  case AArch64::LDRWroW:
  case AArch64::STRSroW:
  case AArch64::STRWroW:
    Shift = 2;
    return true;
  case AArch64::LDRDroW:
  case AArch64::LDRXroW:
  case AArch64::PRFMroW:
  case AArch64::STRDroW:
  case AArch64::STRXroW:
    Shift = 3;
    return true;
  case AArch64::LDRQroW:
  case AArch64::STRQroW:
    Shift = 4;
    return true;
  default:
    return false;
  }
}

// Pre/Post-Index Conversion Tables
//
// These functions convert pre/post-index instructions to their base indexed
// form and provide the scaling factor for the immediate offset.

// Get the scaling factor for pair instruction pre/post-index immediates.
// LDP/STP encode scaled offsets, so we need to multiply by this factor.
unsigned getPrePostScale(unsigned Op) {
  switch (Op) {
  case AArch64::LDPDpost:
  case AArch64::LDPDpre:
  case AArch64::STPDpost:
  case AArch64::STPDpre:
  case AArch64::LDPXpost:
  case AArch64::LDPXpre:
  case AArch64::STPXpost:
  case AArch64::STPXpre:
    return 8;
  case AArch64::LDPQpost:
  case AArch64::LDPQpre:
  case AArch64::STPQpost:
  case AArch64::STPQpre:
    return 16;
  case AArch64::LDPSWpost:
  case AArch64::LDPSWpre:
  case AArch64::LDPSpost:
  case AArch64::LDPSpre:
  case AArch64::STPSpost:
  case AArch64::STPSpre:
  case AArch64::LDPWpost:
  case AArch64::LDPWpre:
  case AArch64::STPWpost:
  case AArch64::STPWpre:
    return 4;
  default:
    return 1;
  }
}

// Convert pre/post-index opcode to its base indexed form.
// Also sets IsPre to true if it's a pre-index instruction.
// Sets IsNoOffset to true if the base form has no offset operand.
unsigned convertPrePostToBase(unsigned Op, bool &IsPre, bool &IsNoOffset) {
  IsPre = false;
  IsNoOffset = false;
  switch (Op) {
  // LDP/STP pairs.
  case AArch64::LDPDpost:
    return AArch64::LDPDi;
  case AArch64::LDPDpre:
    IsPre = true;
    return AArch64::LDPDi;
  case AArch64::LDPQpost:
    return AArch64::LDPQi;
  case AArch64::LDPQpre:
    IsPre = true;
    return AArch64::LDPQi;
  case AArch64::LDPSWpost:
    return AArch64::LDPSWi;
  case AArch64::LDPSWpre:
    IsPre = true;
    return AArch64::LDPSWi;
  case AArch64::LDPSpost:
    return AArch64::LDPSi;
  case AArch64::LDPSpre:
    IsPre = true;
    return AArch64::LDPSi;
  case AArch64::LDPWpost:
    return AArch64::LDPWi;
  case AArch64::LDPWpre:
    IsPre = true;
    return AArch64::LDPWi;
  case AArch64::LDPXpost:
    return AArch64::LDPXi;
  case AArch64::LDPXpre:
    IsPre = true;
    return AArch64::LDPXi;
  case AArch64::STPDpost:
    return AArch64::STPDi;
  case AArch64::STPDpre:
    IsPre = true;
    return AArch64::STPDi;
  case AArch64::STPQpost:
    return AArch64::STPQi;
  case AArch64::STPQpre:
    IsPre = true;
    return AArch64::STPQi;
  case AArch64::STPSpost:
    return AArch64::STPSi;
  case AArch64::STPSpre:
    IsPre = true;
    return AArch64::STPSi;
  case AArch64::STPWpost:
    return AArch64::STPWi;
  case AArch64::STPWpre:
    IsPre = true;
    return AArch64::STPWi;
  case AArch64::STPXpost:
    return AArch64::STPXi;
  case AArch64::STPXpre:
    IsPre = true;
    return AArch64::STPXi;
  // SIMD single structure post-index.
  case AArch64::LD1i8_POST:
    IsNoOffset = true;
    return AArch64::LD1i8;
  case AArch64::LD1i16_POST:
    IsNoOffset = true;
    return AArch64::LD1i16;
  case AArch64::LD1i32_POST:
    IsNoOffset = true;
    return AArch64::LD1i32;
  case AArch64::LD1i64_POST:
    IsNoOffset = true;
    return AArch64::LD1i64;
  case AArch64::ST1i8_POST:
    IsNoOffset = true;
    return AArch64::ST1i8;
  case AArch64::ST1i16_POST:
    IsNoOffset = true;
    return AArch64::ST1i16;
  case AArch64::ST1i32_POST:
    IsNoOffset = true;
    return AArch64::ST1i32;
  case AArch64::ST1i64_POST:
    IsNoOffset = true;
    return AArch64::ST1i64;
  case AArch64::LD2i8_POST:
    IsNoOffset = true;
    return AArch64::LD2i8;
  case AArch64::LD2i16_POST:
    IsNoOffset = true;
    return AArch64::LD2i16;
  case AArch64::LD2i32_POST:
    IsNoOffset = true;
    return AArch64::LD2i32;
  case AArch64::LD2i64_POST:
    IsNoOffset = true;
    return AArch64::LD2i64;
  case AArch64::ST2i8_POST:
    IsNoOffset = true;
    return AArch64::ST2i8;
  case AArch64::ST2i16_POST:
    IsNoOffset = true;
    return AArch64::ST2i16;
  case AArch64::ST2i32_POST:
    IsNoOffset = true;
    return AArch64::ST2i32;
  case AArch64::ST2i64_POST:
    IsNoOffset = true;
    return AArch64::ST2i64;
  case AArch64::LD3i8_POST:
    IsNoOffset = true;
    return AArch64::LD3i8;
  case AArch64::LD3i16_POST:
    IsNoOffset = true;
    return AArch64::LD3i16;
  case AArch64::LD3i32_POST:
    IsNoOffset = true;
    return AArch64::LD3i32;
  case AArch64::LD3i64_POST:
    IsNoOffset = true;
    return AArch64::LD3i64;
  case AArch64::ST3i8_POST:
    IsNoOffset = true;
    return AArch64::ST3i8;
  case AArch64::ST3i16_POST:
    IsNoOffset = true;
    return AArch64::ST3i16;
  case AArch64::ST3i32_POST:
    IsNoOffset = true;
    return AArch64::ST3i32;
  case AArch64::ST3i64_POST:
    IsNoOffset = true;
    return AArch64::ST3i64;
  case AArch64::LD4i8_POST:
    IsNoOffset = true;
    return AArch64::LD4i8;
  case AArch64::LD4i16_POST:
    IsNoOffset = true;
    return AArch64::LD4i16;
  case AArch64::LD4i32_POST:
    IsNoOffset = true;
    return AArch64::LD4i32;
  case AArch64::LD4i64_POST:
    IsNoOffset = true;
    return AArch64::LD4i64;
  case AArch64::ST4i8_POST:
    IsNoOffset = true;
    return AArch64::ST4i8;
  case AArch64::ST4i16_POST:
    IsNoOffset = true;
    return AArch64::ST4i16;
  case AArch64::ST4i32_POST:
    IsNoOffset = true;
    return AArch64::ST4i32;
  case AArch64::ST4i64_POST:
    IsNoOffset = true;
    return AArch64::ST4i64;
  // SIMD replicate post-index.
  case AArch64::LD1Rv8b_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv8b;
  case AArch64::LD1Rv16b_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv16b;
  case AArch64::LD1Rv4h_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv4h;
  case AArch64::LD1Rv8h_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv8h;
  case AArch64::LD1Rv2s_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv2s;
  case AArch64::LD1Rv4s_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv4s;
  case AArch64::LD1Rv1d_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv1d;
  case AArch64::LD1Rv2d_POST:
    IsNoOffset = true;
    return AArch64::LD1Rv2d;
  case AArch64::LD2Rv8b_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv8b;
  case AArch64::LD2Rv16b_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv16b;
  case AArch64::LD2Rv4h_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv4h;
  case AArch64::LD2Rv8h_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv8h;
  case AArch64::LD2Rv2s_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv2s;
  case AArch64::LD2Rv4s_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv4s;
  case AArch64::LD2Rv1d_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv1d;
  case AArch64::LD2Rv2d_POST:
    IsNoOffset = true;
    return AArch64::LD2Rv2d;
  case AArch64::LD3Rv8b_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv8b;
  case AArch64::LD3Rv16b_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv16b;
  case AArch64::LD3Rv4h_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv4h;
  case AArch64::LD3Rv8h_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv8h;
  case AArch64::LD3Rv2s_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv2s;
  case AArch64::LD3Rv4s_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv4s;
  case AArch64::LD3Rv1d_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv1d;
  case AArch64::LD3Rv2d_POST:
    IsNoOffset = true;
    return AArch64::LD3Rv2d;
  case AArch64::LD4Rv8b_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv8b;
  case AArch64::LD4Rv16b_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv16b;
  case AArch64::LD4Rv4h_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv4h;
  case AArch64::LD4Rv8h_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv8h;
  case AArch64::LD4Rv2s_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv2s;
  case AArch64::LD4Rv4s_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv4s;
  case AArch64::LD4Rv1d_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv1d;
  case AArch64::LD4Rv2d_POST:
    IsNoOffset = true;
    return AArch64::LD4Rv2d;
  // SIMD multiple structures post-index (One).
  case AArch64::LD1Onev8b_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev8b;
  case AArch64::LD1Onev16b_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev16b;
  case AArch64::LD1Onev4h_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev4h;
  case AArch64::LD1Onev8h_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev8h;
  case AArch64::LD1Onev2s_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev2s;
  case AArch64::LD1Onev4s_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev4s;
  case AArch64::LD1Onev1d_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev1d;
  case AArch64::LD1Onev2d_POST:
    IsNoOffset = true;
    return AArch64::LD1Onev2d;
  case AArch64::ST1Onev8b_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev8b;
  case AArch64::ST1Onev16b_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev16b;
  case AArch64::ST1Onev4h_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev4h;
  case AArch64::ST1Onev8h_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev8h;
  case AArch64::ST1Onev2s_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev2s;
  case AArch64::ST1Onev4s_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev4s;
  case AArch64::ST1Onev1d_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev1d;
  case AArch64::ST1Onev2d_POST:
    IsNoOffset = true;
    return AArch64::ST1Onev2d;
  // SIMD multiple structures post-index (Two).
  case AArch64::LD1Twov8b_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov8b;
  case AArch64::LD1Twov16b_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov16b;
  case AArch64::LD1Twov4h_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov4h;
  case AArch64::LD1Twov8h_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov8h;
  case AArch64::LD1Twov2s_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov2s;
  case AArch64::LD1Twov4s_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov4s;
  case AArch64::LD1Twov1d_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov1d;
  case AArch64::LD1Twov2d_POST:
    IsNoOffset = true;
    return AArch64::LD1Twov2d;
  case AArch64::ST1Twov8b_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov8b;
  case AArch64::ST1Twov16b_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov16b;
  case AArch64::ST1Twov4h_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov4h;
  case AArch64::ST1Twov8h_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov8h;
  case AArch64::ST1Twov2s_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov2s;
  case AArch64::ST1Twov4s_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov4s;
  case AArch64::ST1Twov1d_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov1d;
  case AArch64::ST1Twov2d_POST:
    IsNoOffset = true;
    return AArch64::ST1Twov2d;
  // SIMD multiple structures post-index (Three).
  case AArch64::LD1Threev8b_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev8b;
  case AArch64::LD1Threev16b_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev16b;
  case AArch64::LD1Threev4h_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev4h;
  case AArch64::LD1Threev8h_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev8h;
  case AArch64::LD1Threev2s_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev2s;
  case AArch64::LD1Threev4s_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev4s;
  case AArch64::LD1Threev1d_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev1d;
  case AArch64::LD1Threev2d_POST:
    IsNoOffset = true;
    return AArch64::LD1Threev2d;
  case AArch64::ST1Threev8b_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev8b;
  case AArch64::ST1Threev16b_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev16b;
  case AArch64::ST1Threev4h_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev4h;
  case AArch64::ST1Threev8h_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev8h;
  case AArch64::ST1Threev2s_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev2s;
  case AArch64::ST1Threev4s_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev4s;
  case AArch64::ST1Threev1d_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev1d;
  case AArch64::ST1Threev2d_POST:
    IsNoOffset = true;
    return AArch64::ST1Threev2d;
  // SIMD multiple structures post-index (Four).
  case AArch64::LD1Fourv8b_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv8b;
  case AArch64::LD1Fourv16b_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv16b;
  case AArch64::LD1Fourv4h_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv4h;
  case AArch64::LD1Fourv8h_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv8h;
  case AArch64::LD1Fourv2s_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv2s;
  case AArch64::LD1Fourv4s_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv4s;
  case AArch64::LD1Fourv1d_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv1d;
  case AArch64::LD1Fourv2d_POST:
    IsNoOffset = true;
    return AArch64::LD1Fourv2d;
  case AArch64::ST1Fourv8b_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv8b;
  case AArch64::ST1Fourv16b_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv16b;
  case AArch64::ST1Fourv4h_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv4h;
  case AArch64::ST1Fourv8h_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv8h;
  case AArch64::ST1Fourv2s_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv2s;
  case AArch64::ST1Fourv4s_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv4s;
  case AArch64::ST1Fourv1d_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv1d;
  case AArch64::ST1Fourv2d_POST:
    IsNoOffset = true;
    return AArch64::ST1Fourv2d;
  // LD2/ST2 multiple structures.
  case AArch64::LD2Twov8b_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov8b;
  case AArch64::LD2Twov16b_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov16b;
  case AArch64::LD2Twov4h_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov4h;
  case AArch64::LD2Twov8h_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov8h;
  case AArch64::LD2Twov2s_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov2s;
  case AArch64::LD2Twov4s_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov4s;
  case AArch64::LD2Twov2d_POST:
    IsNoOffset = true;
    return AArch64::LD2Twov2d;
  case AArch64::ST2Twov8b_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov8b;
  case AArch64::ST2Twov16b_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov16b;
  case AArch64::ST2Twov4h_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov4h;
  case AArch64::ST2Twov8h_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov8h;
  case AArch64::ST2Twov2s_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov2s;
  case AArch64::ST2Twov4s_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov4s;
  case AArch64::ST2Twov2d_POST:
    IsNoOffset = true;
    return AArch64::ST2Twov2d;
  // LD3/ST3 multiple structures.
  case AArch64::LD3Threev8b_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev8b;
  case AArch64::LD3Threev16b_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev16b;
  case AArch64::LD3Threev4h_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev4h;
  case AArch64::LD3Threev8h_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev8h;
  case AArch64::LD3Threev2s_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev2s;
  case AArch64::LD3Threev4s_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev4s;
  case AArch64::LD3Threev2d_POST:
    IsNoOffset = true;
    return AArch64::LD3Threev2d;
  case AArch64::ST3Threev8b_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev8b;
  case AArch64::ST3Threev16b_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev16b;
  case AArch64::ST3Threev4h_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev4h;
  case AArch64::ST3Threev8h_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev8h;
  case AArch64::ST3Threev2s_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev2s;
  case AArch64::ST3Threev4s_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev4s;
  case AArch64::ST3Threev2d_POST:
    IsNoOffset = true;
    return AArch64::ST3Threev2d;
  // LD4/ST4 multiple structures.
  case AArch64::LD4Fourv8b_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv8b;
  case AArch64::LD4Fourv16b_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv16b;
  case AArch64::LD4Fourv4h_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv4h;
  case AArch64::LD4Fourv8h_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv8h;
  case AArch64::LD4Fourv2s_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv2s;
  case AArch64::LD4Fourv4s_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv4s;
  case AArch64::LD4Fourv2d_POST:
    IsNoOffset = true;
    return AArch64::LD4Fourv2d;
  case AArch64::ST4Fourv8b_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv8b;
  case AArch64::ST4Fourv16b_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv16b;
  case AArch64::ST4Fourv4h_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv4h;
  case AArch64::ST4Fourv8h_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv8h;
  case AArch64::ST4Fourv2s_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv2s;
  case AArch64::ST4Fourv4s_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv4s;
  case AArch64::ST4Fourv2d_POST:
    IsNoOffset = true;
    return AArch64::ST4Fourv2d;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

// Get the natural offset for SIMD post-index instructions.
// These instructions have XZR as the register operand when using the
// natural (implicit) offset.
int getSIMDNaturalOffset(unsigned Op) {
  switch (Op) {
  // LD1/ST1 single structure.
  case AArch64::LD1i8_POST:
  case AArch64::ST1i8_POST:
    return 1;
  case AArch64::LD1i16_POST:
  case AArch64::ST1i16_POST:
    return 2;
  case AArch64::LD1i32_POST:
  case AArch64::ST1i32_POST:
    return 4;
  case AArch64::LD1i64_POST:
  case AArch64::ST1i64_POST:
    return 8;
  // LD2/ST2 single structure.
  case AArch64::LD2i8_POST:
  case AArch64::ST2i8_POST:
    return 2;
  case AArch64::LD2i16_POST:
  case AArch64::ST2i16_POST:
    return 4;
  case AArch64::LD2i32_POST:
  case AArch64::ST2i32_POST:
    return 8;
  case AArch64::LD2i64_POST:
  case AArch64::ST2i64_POST:
    return 16;
  // LD3/ST3 single structure.
  case AArch64::LD3i8_POST:
  case AArch64::ST3i8_POST:
    return 3;
  case AArch64::LD3i16_POST:
  case AArch64::ST3i16_POST:
    return 6;
  case AArch64::LD3i32_POST:
  case AArch64::ST3i32_POST:
    return 12;
  case AArch64::LD3i64_POST:
  case AArch64::ST3i64_POST:
    return 24;
  // LD4/ST4 single structure.
  case AArch64::LD4i8_POST:
  case AArch64::ST4i8_POST:
    return 4;
  case AArch64::LD4i16_POST:
  case AArch64::ST4i16_POST:
    return 8;
  case AArch64::LD4i32_POST:
  case AArch64::ST4i32_POST:
    return 16;
  case AArch64::LD4i64_POST:
  case AArch64::ST4i64_POST:
    return 32;
  // LD1R.
  case AArch64::LD1Rv8b_POST:
  case AArch64::LD1Rv16b_POST:
    return 1;
  case AArch64::LD1Rv4h_POST:
  case AArch64::LD1Rv8h_POST:
    return 2;
  case AArch64::LD1Rv2s_POST:
  case AArch64::LD1Rv4s_POST:
    return 4;
  case AArch64::LD1Rv1d_POST:
  case AArch64::LD1Rv2d_POST:
    return 8;
  // LD2R.
  case AArch64::LD2Rv8b_POST:
  case AArch64::LD2Rv16b_POST:
    return 2;
  case AArch64::LD2Rv4h_POST:
  case AArch64::LD2Rv8h_POST:
    return 4;
  case AArch64::LD2Rv2s_POST:
  case AArch64::LD2Rv4s_POST:
    return 8;
  case AArch64::LD2Rv1d_POST:
  case AArch64::LD2Rv2d_POST:
    return 16;
  // LD3R.
  case AArch64::LD3Rv8b_POST:
  case AArch64::LD3Rv16b_POST:
    return 3;
  case AArch64::LD3Rv4h_POST:
  case AArch64::LD3Rv8h_POST:
    return 6;
  case AArch64::LD3Rv2s_POST:
  case AArch64::LD3Rv4s_POST:
    return 12;
  case AArch64::LD3Rv1d_POST:
  case AArch64::LD3Rv2d_POST:
    return 24;
  // LD4R.
  case AArch64::LD4Rv8b_POST:
  case AArch64::LD4Rv16b_POST:
    return 4;
  case AArch64::LD4Rv4h_POST:
  case AArch64::LD4Rv8h_POST:
    return 8;
  case AArch64::LD4Rv2s_POST:
  case AArch64::LD4Rv4s_POST:
    return 16;
  case AArch64::LD4Rv1d_POST:
  case AArch64::LD4Rv2d_POST:
    return 32;
  // LD1/ST1 multiple structures (8b).
  case AArch64::LD1Onev8b_POST:
  case AArch64::ST1Onev8b_POST:
    return 8;
  case AArch64::LD1Twov8b_POST:
  case AArch64::ST1Twov8b_POST:
    return 16;
  case AArch64::LD1Threev8b_POST:
  case AArch64::ST1Threev8b_POST:
    return 24;
  case AArch64::LD1Fourv8b_POST:
  case AArch64::ST1Fourv8b_POST:
    return 32;
  // LD1/ST1 multiple structures (16b).
  case AArch64::LD1Onev16b_POST:
  case AArch64::ST1Onev16b_POST:
    return 16;
  case AArch64::LD1Twov16b_POST:
  case AArch64::ST1Twov16b_POST:
    return 32;
  case AArch64::LD1Threev16b_POST:
  case AArch64::ST1Threev16b_POST:
    return 48;
  case AArch64::LD1Fourv16b_POST:
  case AArch64::ST1Fourv16b_POST:
    return 64;
  // LD1/ST1 multiple structures (4h).
  case AArch64::LD1Onev4h_POST:
  case AArch64::ST1Onev4h_POST:
    return 8;
  case AArch64::LD1Twov4h_POST:
  case AArch64::ST1Twov4h_POST:
    return 16;
  case AArch64::LD1Threev4h_POST:
  case AArch64::ST1Threev4h_POST:
    return 24;
  case AArch64::LD1Fourv4h_POST:
  case AArch64::ST1Fourv4h_POST:
    return 32;
  // LD1/ST1 multiple structures (8h).
  case AArch64::LD1Onev8h_POST:
  case AArch64::ST1Onev8h_POST:
    return 16;
  case AArch64::LD1Twov8h_POST:
  case AArch64::ST1Twov8h_POST:
    return 32;
  case AArch64::LD1Threev8h_POST:
  case AArch64::ST1Threev8h_POST:
    return 48;
  case AArch64::LD1Fourv8h_POST:
  case AArch64::ST1Fourv8h_POST:
    return 64;
  // LD1/ST1 multiple structures (2s).
  case AArch64::LD1Onev2s_POST:
  case AArch64::ST1Onev2s_POST:
    return 8;
  case AArch64::LD1Twov2s_POST:
  case AArch64::ST1Twov2s_POST:
    return 16;
  case AArch64::LD1Threev2s_POST:
  case AArch64::ST1Threev2s_POST:
    return 24;
  case AArch64::LD1Fourv2s_POST:
  case AArch64::ST1Fourv2s_POST:
    return 32;
  // LD1/ST1 multiple structures (4s).
  case AArch64::LD1Onev4s_POST:
  case AArch64::ST1Onev4s_POST:
    return 16;
  case AArch64::LD1Twov4s_POST:
  case AArch64::ST1Twov4s_POST:
    return 32;
  case AArch64::LD1Threev4s_POST:
  case AArch64::ST1Threev4s_POST:
    return 48;
  case AArch64::LD1Fourv4s_POST:
  case AArch64::ST1Fourv4s_POST:
    return 64;
  // LD1/ST1 multiple structures (1d).
  case AArch64::LD1Onev1d_POST:
  case AArch64::ST1Onev1d_POST:
    return 8;
  case AArch64::LD1Twov1d_POST:
  case AArch64::ST1Twov1d_POST:
    return 16;
  case AArch64::LD1Threev1d_POST:
  case AArch64::ST1Threev1d_POST:
    return 24;
  case AArch64::LD1Fourv1d_POST:
  case AArch64::ST1Fourv1d_POST:
    return 32;
  // LD1/ST1 multiple structures (2d).
  case AArch64::LD1Onev2d_POST:
  case AArch64::ST1Onev2d_POST:
    return 16;
  case AArch64::LD1Twov2d_POST:
  case AArch64::ST1Twov2d_POST:
    return 32;
  case AArch64::LD1Threev2d_POST:
  case AArch64::ST1Threev2d_POST:
    return 48;
  case AArch64::LD1Fourv2d_POST:
  case AArch64::ST1Fourv2d_POST:
    return 64;
  // LD2/ST2 multiple structures.
  case AArch64::LD2Twov8b_POST:
  case AArch64::ST2Twov8b_POST:
    return 16;
  case AArch64::LD2Twov16b_POST:
  case AArch64::ST2Twov16b_POST:
    return 32;
  case AArch64::LD2Twov4h_POST:
  case AArch64::ST2Twov4h_POST:
    return 16;
  case AArch64::LD2Twov8h_POST:
  case AArch64::ST2Twov8h_POST:
    return 32;
  case AArch64::LD2Twov2s_POST:
  case AArch64::ST2Twov2s_POST:
    return 16;
  case AArch64::LD2Twov4s_POST:
  case AArch64::ST2Twov4s_POST:
    return 32;
  case AArch64::LD2Twov2d_POST:
  case AArch64::ST2Twov2d_POST:
    return 32;
  // LD3/ST3 multiple structures.
  case AArch64::LD3Threev8b_POST:
  case AArch64::ST3Threev8b_POST:
    return 24;
  case AArch64::LD3Threev16b_POST:
  case AArch64::ST3Threev16b_POST:
    return 48;
  case AArch64::LD3Threev4h_POST:
  case AArch64::ST3Threev4h_POST:
    return 24;
  case AArch64::LD3Threev8h_POST:
  case AArch64::ST3Threev8h_POST:
    return 48;
  case AArch64::LD3Threev2s_POST:
  case AArch64::ST3Threev2s_POST:
    return 24;
  case AArch64::LD3Threev4s_POST:
  case AArch64::ST3Threev4s_POST:
    return 48;
  case AArch64::LD3Threev2d_POST:
  case AArch64::ST3Threev2d_POST:
    return 48;
  // LD4/ST4 multiple structures.
  case AArch64::LD4Fourv8b_POST:
  case AArch64::ST4Fourv8b_POST:
    return 32;
  case AArch64::LD4Fourv16b_POST:
  case AArch64::ST4Fourv16b_POST:
    return 64;
  case AArch64::LD4Fourv4h_POST:
  case AArch64::ST4Fourv4h_POST:
    return 32;
  case AArch64::LD4Fourv8h_POST:
  case AArch64::ST4Fourv8h_POST:
    return 64;
  case AArch64::LD4Fourv2s_POST:
  case AArch64::ST4Fourv2s_POST:
    return 32;
  case AArch64::LD4Fourv4s_POST:
  case AArch64::ST4Fourv4s_POST:
    return 64;
  case AArch64::LD4Fourv2d_POST:
  case AArch64::ST4Fourv2d_POST:
    return 64;
  default:
    return -1;
  }
}

} // anonymous namespace
