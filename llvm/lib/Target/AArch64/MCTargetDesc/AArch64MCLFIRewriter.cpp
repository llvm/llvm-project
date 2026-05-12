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

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// LFI reserved registers.
static constexpr MCRegister LFIBaseReg = AArch64::X27;
static constexpr MCRegister LFIAddrReg = AArch64::X28;
static constexpr MCRegister LFIScratchReg = AArch64::X26;
static constexpr MCRegister LFICtxReg = AArch64::X25;

// Offset into the context register block (pointed to by LFICtxReg) where the
// thread pointer is stored. This is a scaled offset (multiplied by 8 for
// 64-bit loads), so a value of 2 means an actual byte offset of 16.
static constexpr unsigned LFITPOffset = 2;

// Byte offset from the sandbox base register where the syscall handler address
// is stored (negative because it is below the sandbox base).
static constexpr int LFISyscallOffset = -8;

static bool isSyscall(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::SVC;
}

static bool isPrivilegedTP(int64_t Reg) {
  return Reg == AArch64SysReg::TPIDR_EL1 || Reg == AArch64SysReg::TPIDR_EL2 ||
         Reg == AArch64SysReg::TPIDR_EL3;
}

static bool isTPRead(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::MRS &&
         Inst.getOperand(1).getImm() == AArch64SysReg::TPIDR_EL0;
}

static bool isTPWrite(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::MSR &&
         Inst.getOperand(0).getImm() == AArch64SysReg::TPIDR_EL0;
}

static bool isPrivilegedTPAccess(const MCInst &Inst) {
  if (Inst.getOpcode() == AArch64::MRS)
    return isPrivilegedTP(Inst.getOperand(1).getImm());
  if (Inst.getOpcode() == AArch64::MSR)
    return isPrivilegedTP(Inst.getOperand(0).getImm());
  return false;
}

// Classification functions are limited to Armv8.1-A. Instructions outside of
// this subset are not guaranteed to be rewritten and as a result may fail LFI
// verification after compilation.

// Instructions that have mayLoad/mayStore set in TableGen but don't actually
// perform memory accesses.
static bool isFakeMemAccess(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::DMB:
  case AArch64::DSB:
  case AArch64::ISB:
  case AArch64::HINT:
    // The range of sub-architectures supported by LFI do not include any load
    // or store instructions in the HINT space.
    return true;
  default:
    return false;
  }
}

static bool mayPrefetch(const MCInst &Inst) {
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

// User-mode DC/IC instructions that take a virtual address operand. Encoded as
// SYSxt with op1=3, Cn=7, op2=1 where the Cm field selects the operation.
static bool isVASysOp(const MCInst &Inst) {
  if (Inst.getOpcode() != AArch64::SYSxt)
    return false;
  if (Inst.getOperand(0).getImm() != 3 || Inst.getOperand(1).getImm() != 7 ||
      Inst.getOperand(3).getImm() != 1)
    return false;
  switch (Inst.getOperand(2).getImm()) {
  case 4:  // DC ZVA
  case 5:  // IC IVAU
  case 10: // DC CVAC
  case 11: // DC CVAU
  case 12: // DC CVAP
  case 13: // DC CVADP
  case 14: // DC CIVAC
    return true;
  default:
    return false;
  }
}

static MCInst replaceRegAt(const MCInst &Inst, unsigned Idx,
                           MCRegister NewReg) {
  MCInst New;
  New.setOpcode(Inst.getOpcode());
  New.setLoc(Inst.getLoc());
  for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
    if (I == Idx) {
      assert(Inst.getOperand(I).isReg());
      New.addOperand(MCOperand::createReg(NewReg));
    } else {
      New.addOperand(Inst.getOperand(I));
    }
  }
  return New;
}

// Memory instruction information for the base load/store rewriting path.
// Provides operand indices for the base register and offset, and whether
// the instruction uses pre/post-indexed addressing.
struct MemInstInfo {
  int BaseRegIdx;
  std::optional<unsigned> OffsetIdx;
  bool IsPrePost;
  bool IsLiteral;
};

// Returns memory instruction info for a given opcode, or std::nullopt if
// the opcode is not a recognized memory instruction.
static std::optional<MemInstInfo> getMemInstInfo(unsigned Op);

// AArch64 load/store opcode suffixes used throughout this file:
//   Ui:  Unsigned immediate offset, scaled by access size: [Xn, #imm].
//   RoW: Register offset with 32-bit W register: [Xn, Wm, uxtw #shift].
//   RoX: Register offset with 64-bit X register: [Xn, Xm, lsl #shift].

static unsigned convertUiToRoW(unsigned Op);
static unsigned convertPreToRoW(unsigned Op);
static unsigned convertPostToRoW(unsigned Op);
static unsigned convertRoXToRoW(unsigned Op, unsigned &Shift);
static bool getRoWShift(unsigned Op, unsigned &Shift);
static unsigned getPrePostScale(unsigned Op);
static unsigned convertPrePostToBase(unsigned Op, bool &IsPre,
                                     bool &IsNoOffset);
static int getSIMDNaturalOffset(unsigned Op);

bool AArch64MCLFIRewriter::mayModifySP(const MCInst &Inst) const {
  return mayModifyRegister(Inst, AArch64::SP);
}

MCRegister AArch64MCLFIRewriter::mayModifyReserved(const MCInst &Inst) const {
  for (MCRegister Reg : {LFIAddrReg, LFIBaseReg, LFICtxReg}) {
    if (mayModifyRegister(Inst, Reg))
      return Reg;
  }
  return {};
}

void AArch64MCLFIRewriter::emitInst(const MCInst &Inst, MCStreamer &Out,
                                    const MCSubtargetInfo &STI) {
  Out.emitInstruction(Inst, STI);
}

void AArch64MCLFIRewriter::emitAddMask(MCRegister Dest, MCRegister Src,
                                       MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  // add Dest, LFIBaseReg, W(Src), uxtw
  MCInst Inst;
  Inst.setOpcode(AArch64::ADDXrx);
  Inst.addOperand(MCOperand::createReg(Dest));
  Inst.addOperand(MCOperand::createReg(LFIBaseReg));
  Inst.addOperand(MCOperand::createReg(getWRegFromXReg(Src)));
  Inst.addOperand(
      MCOperand::createImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTW, 0)));
  emitInst(Inst, Out, STI);
}

void AArch64MCLFIRewriter::emitBranch(unsigned Opcode, MCRegister Target,
                                      MCStreamer &Out,
                                      const MCSubtargetInfo &STI) {
  MCInst Branch;
  Branch.setOpcode(Opcode);
  Branch.addOperand(MCOperand::createReg(Target));
  emitInst(Branch, Out, STI);
}

void AArch64MCLFIRewriter::emitPendingTLSDescCall(MCStreamer &Out,
                                                  const MCSubtargetInfo &STI) {
  if (!PendingTLSDescCall)
    return;
  MCInst Marker;
  Marker.setOpcode(AArch64::TLSDESCCALL);
  Marker.addOperand(MCOperand::createExpr(PendingTLSDescCall));
  PendingTLSDescCall = nullptr;
  emitInst(Marker, Out, STI);
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
    // add Dest, Src, Imm
    Inst.setOpcode(AArch64::ADDXri);
    Inst.addOperand(MCOperand::createReg(Dest));
    Inst.addOperand(MCOperand::createReg(Src));
    Inst.addOperand(MCOperand::createImm(Imm));
    Inst.addOperand(MCOperand::createImm(0)); // shift
  } else {
    // sub Dest, Src, -Imm
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
  // Op DataOp, [LFIBaseReg, W(BaseReg), uxtw]
  MCInst Inst;
  Inst.setOpcode(Opcode);
  Inst.addOperand(DataOp);
  Inst.addOperand(MCOperand::createReg(LFIBaseReg));
  Inst.addOperand(MCOperand::createReg(getWRegFromXReg(BaseReg)));
  Inst.addOperand(MCOperand::createImm(0)); // S bit = 0 (UXTW).
  Inst.addOperand(MCOperand::createImm(0)); // Shift amount = 0 (unscaled).
  emitInst(Inst, Out, STI);
}

// {br,blr} xN
// ->
// add x28, x27, wN, uxtw
// {br,blr} x28
void AArch64MCLFIRewriter::rewriteIndirectBranch(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  assert(Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
         "expected register operand");
  MCRegister BranchReg = Inst.getOperand(0).getReg();

  // Guard the branch target through X28.
  emitAddMask(LFIAddrReg, BranchReg, Out, STI);

  emitPendingTLSDescCall(Out, STI);

  emitBranch(Inst.getOpcode(), LFIAddrReg, Out, STI);
}

// ret xN (where xN != x30)
// ->
// add x28, x27, wN, uxtw
// ret x28
//
// ret (x30) is safe since x30 is always within the sandbox.
void AArch64MCLFIRewriter::rewriteReturn(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  assert(Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
         "expected register operand");
  // RET through LR is safe since LR is always within sandbox.
  if (Inst.getOperand(0).getReg() != AArch64::LR)
    rewriteIndirectBranch(Inst, Out, STI);
  else
    emitInst(Inst, Out, STI);
}

// modify x30
// ->
// modify x30
// add x30, x27, w30, uxtw
void AArch64MCLFIRewriter::rewriteLRModification(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  if (!isFakeMemAccess(Inst) &&
      (mayLoad(Inst) || mayStore(Inst) || mayPrefetch(Inst)))
    rewriteLoadStore(Inst, Out, STI);
  else
    emitInst(Inst, Out, STI);
  emitAddMask(AArch64::LR, AArch64::LR, Out, STI);
}

// svc #0
// ->
// mov x26, x30
// ldur x30, [x27, #-8]
// blr x30
// add x30, x27, w26, uxtw
void AArch64MCLFIRewriter::rewriteSyscall(const MCInst &, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  // Save LR to scratch.
  emitMov(LFIScratchReg, AArch64::LR, Out, STI);

  // Load syscall handler address from negative offset from sandbox base.
  MCInst Load;
  Load.setOpcode(AArch64::LDURXi);
  Load.addOperand(MCOperand::createReg(AArch64::LR));
  Load.addOperand(MCOperand::createReg(LFIBaseReg));
  Load.addOperand(MCOperand::createImm(LFISyscallOffset));
  emitInst(Load, Out, STI);

  // Call the runtime.
  emitBranch(AArch64::BLR, AArch64::LR, Out, STI);

  // Restore LR with guard.
  emitAddMask(AArch64::LR, LFIScratchReg, Out, STI);
}

// mrs xN, tpidr_el0
// ->
// ldr xN, [x25, #16]
void AArch64MCLFIRewriter::rewriteTPRead(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  MCRegister DestReg = Inst.getOperand(0).getReg();

  MCInst Load;
  Load.setOpcode(AArch64::LDRXui);
  Load.addOperand(MCOperand::createReg(DestReg));
  Load.addOperand(MCOperand::createReg(LFICtxReg));
  Load.addOperand(MCOperand::createImm(LFITPOffset));
  emitInst(Load, Out, STI);
}

// msr tpidr_el0, xN
// ->
// str xN, [x25, #16]
void AArch64MCLFIRewriter::rewriteTPWrite(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  MCRegister SrcReg = Inst.getOperand(1).getReg();

  MCInst Store;
  Store.setOpcode(AArch64::STRXui);
  Store.addOperand(MCOperand::createReg(SrcReg));
  Store.addOperand(MCOperand::createReg(LFICtxReg));
  Store.addOperand(MCOperand::createImm(LFITPOffset));
  emitInst(Store, Out, STI);
}

bool AArch64MCLFIRewriter::rewriteLoadStoreRoW(const MCInst &Inst,
                                               MCStreamer &Out,
                                               const MCSubtargetInfo &STI) {
  unsigned Op = Inst.getOpcode();
  unsigned MemOp;

  // Case 1: Indexed load/store with zero immediate offset.
  // ldr xN, [xM, #0] -> ldr xN, [x27, wM, uxtw]
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

  // Case 2: Pre-index load/store with writeback.
  // ldr xN, [xM, #imm]! -> add xM, xM, #imm; ldr xN, [x27, wM, uxtw]
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
  //
  // In this case, even if xM1 is SP we must do a full rewrite, since an
  // arbitrary register value is being added as the offset.
  unsigned Shift;
  if ((MemOp = convertRoXToRoW(Op, Shift)) != AArch64::INSTRUCTION_LIST_END) {
    MCRegister Reg1 = Inst.getOperand(1).getReg();
    MCRegister Reg2 = Inst.getOperand(2).getReg();
    int64_t Extend = Inst.getOperand(3).getImm();
    int64_t IsShift = Inst.getOperand(4).getImm();

    if (!IsShift)
      Shift = 0;

    if (Extend)
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::SXTX, Shift, Out,
                       STI);
    else
      emitAddReg(LFIScratchReg, Reg1, Reg2, Shift, Out, STI);
    emitMemRoW(MemOp, Inst.getOperand(0), LFIScratchReg, Out, STI);
    return true;
  }

  // Case 5: Register-offset-W load/store.
  // ldr xN, [xM1, wM2, uxtw] -> add x26, xM1, wM2, uxtw;
  //                             ldr xN, [x27, w26, uxtw]
  if (getRoWShift(Op, Shift)) {
    MCRegister Reg1 = Inst.getOperand(1).getReg();
    MCRegister Reg2 = Inst.getOperand(2).getReg();
    int64_t S = Inst.getOperand(3).getImm();
    int64_t IsShift = Inst.getOperand(4).getImm();

    if (!IsShift)
      Shift = 0;

    if (S)
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::SXTW, Shift, Out,
                       STI);
    else
      emitAddRegExtend(LFIScratchReg, Reg1, Reg2, AArch64_AM::UXTW, Shift, Out,
                       STI);
    emitMemRoW(Op, Inst.getOperand(0), LFIScratchReg, Out, STI);
    return true;
  }

  return false;
}

void AArch64MCLFIRewriter::rewriteLoadStoreBase(const MCInst &Inst,
                                                MCStreamer &Out,
                                                const MCSubtargetInfo &STI) {
  unsigned Opcode = Inst.getOpcode();
  auto Info = getMemInstInfo(Opcode);

  if (!Info)
    return error(Inst, "unknown addressing mode for memory instruction in LFI");

  if (Info->IsLiteral)
    return error(Inst, "PC-relative literal loads are not supported in LFI");

  MCRegister BaseReg = Inst.getOperand(Info->BaseRegIdx).getReg();

  // Stack accesses don't need address sandboxing, except when sp is modified
  // with a non-zero register post-index operand.
  bool BaseIsSP = BaseReg == AArch64::SP;
  if (BaseIsSP) {
    if (!Info->OffsetIdx || !Inst.getOperand(*Info->OffsetIdx).isReg())
      return emitInst(Inst, Out, STI);
    MCRegister OffReg = Inst.getOperand(*Info->OffsetIdx).getReg();
    if (OffReg == AArch64::XZR || OffReg == AArch64::WZR)
      return emitInst(Inst, Out, STI);
  }

  // Guard the base register, unless it is SP.
  if (!BaseIsSP)
    emitAddMask(LFIAddrReg, BaseReg, Out, STI);

  if (!Info->IsPrePost) {
    // Non-pre/post instruction: replace the base register operand.
    MCInst NewInst = replaceRegAt(Inst, Info->BaseRegIdx, LFIAddrReg);
    emitInst(NewInst, Out, STI);
    return;
  }

  bool IsPre = false;
  bool IsNoOffset = false;
  unsigned BaseOpcode = convertPrePostToBase(Opcode, IsPre, IsNoOffset);

  if (BaseOpcode == AArch64::INSTRUCTION_LIST_END)
    return error(Inst, "unhandled pre/post-index instruction in LFI rewriter");

  // Demote pre/post-index to base indexed form.
  MCInst NewInst;
  NewInst.setOpcode(BaseOpcode);
  NewInst.setLoc(Inst.getLoc());

  // Skip writeback operand (operand 0) and copy data operands up to base.
  for (int I = 1; I < Info->BaseRegIdx; ++I)
    NewInst.addOperand(Inst.getOperand(I));

  // Add the access base register (LFIAddrReg or SP).
  MCRegister AccessBase = BaseIsSP ? AArch64::SP : LFIAddrReg;
  NewInst.addOperand(MCOperand::createReg(AccessBase));

  // For pre-index, include the offset; for post-index, use zero.
  if (IsPre && Info->OffsetIdx)
    NewInst.addOperand(Inst.getOperand(*Info->OffsetIdx));
  else if (!IsNoOffset)
    NewInst.addOperand(MCOperand::createImm(0));

  emitInst(NewInst, Out, STI);

  if (!Info->OffsetIdx)
    return;

  // Update the base register with the offset. If the base is SP, a register
  // offset must be sandboxed (the result is otherwise unbounded), and ADDXrs
  // cannot take SP, so the extended-register form via the scratch register is
  // used.
  const MCOperand &OffsetOp = Inst.getOperand(*Info->OffsetIdx);
  if (OffsetOp.isImm()) {
    int64_t Scale = getPrePostScale(Opcode);
    int64_t Offset = OffsetOp.getImm() * Scale;
    emitAddImm(BaseReg, BaseReg, Offset, Out, STI);
  } else if (OffsetOp.isReg()) {
    // SIMD post-index uses a register offset (XZR for natural offset).
    MCRegister OffReg = OffsetOp.getReg();
    if (OffReg == AArch64::XZR) {
      int NaturalOffset = getSIMDNaturalOffset(Opcode);
      if (NaturalOffset > 0)
        emitAddImm(BaseReg, BaseReg, NaturalOffset, Out, STI);
    } else if (OffReg != AArch64::WZR) {
      if (BaseIsSP) {
        emitAddRegExtend(LFIScratchReg, AArch64::SP, OffReg, AArch64_AM::UXTX,
                         0, Out, STI);
        emitAddMask(AArch64::SP, LFIScratchReg, Out, STI);
      } else {
        emitAddReg(BaseReg, BaseReg, OffReg, 0, Out, STI);
      }
    }
  }
}

void AArch64MCLFIRewriter::rewriteLoadStore(const MCInst &Inst, MCStreamer &Out,
                                            const MCSubtargetInfo &STI) {
  bool IsStore = mayStore(Inst);
  bool IsLoad = mayLoad(Inst) || mayPrefetch(Inst);

  bool SkipLoads = STI.hasFeature(AArch64::FeatureNoLFILoads);
  bool SkipStores = STI.hasFeature(AArch64::FeatureNoLFIStores);

  if ((!IsLoad || SkipLoads) && (!IsStore || SkipStores))
    return emitInst(Inst, Out, STI);

  if (rewriteLoadStoreRoW(Inst, Out, STI))
    return;

  rewriteLoadStoreBase(Inst, Out, STI);
}

// modify sp
// ->
// modify x26
// add sp, x27, w26, uxtw
void AArch64MCLFIRewriter::rewriteSPModification(const MCInst &Inst,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  // Route through rewriteLRModification or rewriteLoadStore for memory
  // accesses. Those helpers automatically handle dangerous stack modifications
  // that can happen via register post-index.
  if (mayLoad(Inst) || mayStore(Inst)) {
    if (mayModifyRegister(Inst, AArch64::LR))
      return rewriteLRModification(Inst, Out, STI);
    return rewriteLoadStore(Inst, Out, STI);
  }

  // No stack sandboxing if sandboxing is disabled for both loads and stores.
  bool SkipLoads = STI.hasFeature(AArch64::FeatureNoLFILoads);
  bool SkipStores = STI.hasFeature(AArch64::FeatureNoLFIStores);
  if (SkipLoads && SkipStores)
    return emitInst(Inst, Out, STI);

  // Redirect SP modification destination to scratch, then sandbox.
  MCInst ModInst = replaceRegAt(Inst, 0, LFIScratchReg);
  emitInst(ModInst, Out, STI);
  emitAddMask(AArch64::SP, LFIScratchReg, Out, STI);
}

// {dc,ic} <op>, xN
// ->
// add x28, x27, wN, uxtw
// {dc,ic} <op>, x28
void AArch64MCLFIRewriter::rewriteVASysOp(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  MCRegister AddrReg = Inst.getOperand(4).getReg();

  emitAddMask(LFIAddrReg, AddrReg, Out, STI);

  MCInst NewInst;
  NewInst.setOpcode(AArch64::SYSxt);
  NewInst.addOperand(Inst.getOperand(0));
  NewInst.addOperand(Inst.getOperand(1));
  NewInst.addOperand(Inst.getOperand(2));
  NewInst.addOperand(Inst.getOperand(3));
  NewInst.addOperand(MCOperand::createReg(LFIAddrReg));
  emitInst(NewInst, Out, STI);
}

// NOTE: when adding new rewrites, the size estimates in
// AArch64InstrInfo::getLFIInstSizeInBytes must be updated to match.
void AArch64MCLFIRewriter::doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  if (Inst.getOpcode() == AArch64::TLSDESCCALL) {
    PendingTLSDescCall = Inst.getOperand(0).getExpr();
    return;
  }

  // Reserved register modification is an error.
  if (MCRegister Reg = mayModifyReserved(Inst)) {
    error(Inst, Twine("illegal modification of reserved LFI register ") +
                    RegInfo->getName(Reg));
    return;
  }

  // System instructions.
  if (isSyscall(Inst))
    return rewriteSyscall(Inst, Out, STI);

  if (isTPRead(Inst))
    return rewriteTPRead(Inst, Out, STI);

  if (isTPWrite(Inst))
    return rewriteTPWrite(Inst, Out, STI);

  if (isPrivilegedTPAccess(Inst)) {
    error(Inst, "illegal access to privileged thread pointer register");
    return;
  }

  if (isVASysOp(Inst))
    return rewriteVASysOp(Inst, Out, STI);

  // Control flow.
  switch (Inst.getOpcode()) {
  case AArch64::RET:
    return rewriteReturn(Inst, Out, STI);
  case AArch64::BR:
  case AArch64::BLR:
    return rewriteIndirectBranch(Inst, Out, STI);
  }

  // Register modifications that require sandboxing.
  if (mayModifySP(Inst))
    return rewriteSPModification(Inst, Out, STI);

  // Link register modification.
  if (explicitlyModifiesRegister(Inst, AArch64::LR))
    return rewriteLRModification(Inst, Out, STI);

  // Memory access.
  if (!isFakeMemAccess(Inst) &&
      (mayLoad(Inst) || mayStore(Inst) || mayPrefetch(Inst)))
    return rewriteLoadStore(Inst, Out, STI);

  emitInst(Inst, Out, STI);
}

// This function is made available to the size estimator so that it can
// classify Pre/Post-index instructions.
bool llvm::isLFIPrePostMemAccess(unsigned Opcode) {
  if (convertPreToRoW(Opcode) != AArch64::INSTRUCTION_LIST_END)
    return true;
  if (convertPostToRoW(Opcode) != AArch64::INSTRUCTION_LIST_END)
    return true;
  bool IsPre, IsNoOffset;
  if (convertPrePostToBase(Opcode, IsPre, IsNoOffset) !=
      AArch64::INSTRUCTION_LIST_END)
    return true;
  return false;
}

bool AArch64MCLFIRewriter::rewriteInst(const MCInst &Inst, MCStreamer &Out,
                                       const MCSubtargetInfo &STI) {
  // The guard prevents rewrite-recursion when we emit instructions from inside
  // the rewriter (such instructions should not be rewritten).
  if (!Enabled || Guard)
    return false;
  Guard = true;

  doRewriteInst(Inst, Out, STI);

  Guard = false;
  return true;
}

// Opcode X-macro Tables
//
// These macros define groups of related opcodes and are used to generate
// multiple switch tables without repetition. Each macro takes a callback X and
// invokes X(NAME, VALUE) for each entry.

// Scalar memory ops that have ui, pre, post, roX, and roW variants.
// PRFM is excluded because it has no pre/post forms.
// The second column is the log2 element size (shift for scaled addressing).
#define SCALAR_MEM_OPS(X)                                                      \
  X(LDRBB, 0)                                                                  \
  X(LDRB, 0)                                                                   \
  X(LDRSBW, 0)                                                                 \
  X(LDRSBX, 0)                                                                 \
  X(STRBB, 0)                                                                  \
  X(STRB, 0)                                                                   \
  X(LDRHH, 1)                                                                  \
  X(LDRH, 1)                                                                   \
  X(LDRSHW, 1)                                                                 \
  X(LDRSHX, 1)                                                                 \
  X(STRHH, 1)                                                                  \
  X(STRH, 1)                                                                   \
  X(LDRSW, 2)                                                                  \
  X(LDRS, 2)                                                                   \
  X(LDRW, 2)                                                                   \
  X(STRS, 2)                                                                   \
  X(STRW, 2)                                                                   \
  X(LDRD, 3)                                                                   \
  X(LDRX, 3)                                                                   \
  X(STRD, 3)                                                                   \
  X(STRX, 3)                                                                   \
  X(LDRQ, 4)                                                                   \
  X(STRQ, 4)

// LDP/STP pair ops. The second column is the scale factor for pre/post
// immediates.
#define PAIR_OPS(X)                                                            \
  X(LDPD, 8)                                                                   \
  X(LDPQ, 16)                                                                  \
  X(LDPSW, 4)                                                                  \
  X(LDPS, 4)                                                                   \
  X(LDPW, 4)                                                                   \
  X(LDPX, 8)                                                                   \
  X(STPD, 8)                                                                   \
  X(STPQ, 16)                                                                  \
  X(STPS, 4)                                                                   \
  X(STPW, 4)                                                                   \
  X(STPX, 8)

// SIMD post-index ops, split into three groups by operand layout.
// The second column is the natural byte offset for post-index.

// Lane stores: Vt, idx, [Rn] (base=2) / wback, Vt, idx, [Rn], Xm (base=3).

// clang-format off
#define SIMD_LANE_STORE_OPS(X)                                                 \
  X(ST1i8, 1) X(ST1i16, 2) X(ST1i32,  4) X(ST1i64,  8)                         \
  X(ST2i8, 2) X(ST2i16, 4) X(ST2i32,  8) X(ST2i64, 16)                         \
  X(ST3i8, 3) X(ST3i16, 6) X(ST3i32, 12) X(ST3i64, 24)                         \
  X(ST4i8, 4) X(ST4i16, 8) X(ST4i32, 16) X(ST4i64, 32)

// Lane loads: Vt(out), Vt(tied), idx, [Rn] (base=3) /
//             wback, Vt(out), Vt(tied), idx, [Rn], Xm (base=4).
#define SIMD_LANE_LOAD_OPS(X)                                                  \
  X(LD1i8, 1) X(LD1i16, 2) X(LD1i32,  4) X(LD1i64,  8)                         \
  X(LD2i8, 2) X(LD2i16, 4) X(LD2i32,  8) X(LD2i64, 16)                         \
  X(LD3i8, 3) X(LD3i16, 6) X(LD3i32, 12) X(LD3i64, 24)                         \
  X(LD4i8, 4) X(LD4i16, 8) X(LD4i32, 16) X(LD4i64, 32)

// Multiple structure and replicate: Vt, [Rn] (base=1) /
//                                   wback, Vt, [Rn], Xm (base=2).
// Each line pairs the LD and ST variants (replicates are LD-only).
#define SIMD_MULTI_OPS(X)                                                      \
  /* Replicate loads (LD only). */                                             \
  X(LD1Rv8b,  1) X(LD1Rv16b,  1) X(LD1Rv4h,  2) X(LD1Rv8h,  2)                 \
  X(LD1Rv2s,  4) X(LD1Rv4s,   4) X(LD1Rv1d,  8) X(LD1Rv2d,  8)                 \
  X(LD2Rv8b,  2) X(LD2Rv16b,  2) X(LD2Rv4h,  4) X(LD2Rv8h,  4)                 \
  X(LD2Rv2s,  8) X(LD2Rv4s,   8) X(LD2Rv1d, 16) X(LD2Rv2d, 16)                 \
  X(LD3Rv8b,  3) X(LD3Rv16b,  3) X(LD3Rv4h,  6) X(LD3Rv8h,  6)                 \
  X(LD3Rv2s, 12) X(LD3Rv4s,  12) X(LD3Rv1d, 24) X(LD3Rv2d, 24)                 \
  X(LD4Rv8b,  4) X(LD4Rv16b,  4) X(LD4Rv4h,  8) X(LD4Rv8h,  8)                 \
  X(LD4Rv2s, 16) X(LD4Rv4s,  16) X(LD4Rv1d, 32) X(LD4Rv2d, 32)                 \
  /* LD1/ST1 One register. */                                                  \
  X(LD1Onev8b, 8) X(LD1Onev16b, 16)                                            \
  X(LD1Onev4h, 8) X(LD1Onev8h,  16)                                            \
  X(LD1Onev2s, 8) X(LD1Onev4s,  16)                                            \
  X(LD1Onev1d, 8) X(LD1Onev2d,  16)                                            \
  X(ST1Onev8b, 8) X(ST1Onev16b, 16)                                            \
  X(ST1Onev4h, 8) X(ST1Onev8h,  16)                                            \
  X(ST1Onev2s, 8) X(ST1Onev4s,  16)                                            \
  X(ST1Onev1d, 8) X(ST1Onev2d,  16)                                            \
  /* LD1/ST1 Two registers. */                                                 \
  X(LD1Twov8b, 16) X(LD1Twov16b, 32)                                           \
  X(LD1Twov4h, 16) X(LD1Twov8h,  32)                                           \
  X(LD1Twov2s, 16) X(LD1Twov4s,  32)                                           \
  X(LD1Twov1d, 16) X(LD1Twov2d,  32)                                           \
  X(ST1Twov8b, 16) X(ST1Twov16b, 32)                                           \
  X(ST1Twov4h, 16) X(ST1Twov8h,  32)                                           \
  X(ST1Twov2s, 16) X(ST1Twov4s,  32)                                           \
  X(ST1Twov1d, 16) X(ST1Twov2d,  32)                                           \
  /* LD1/ST1 Three registers. */                                               \
  X(LD1Threev8b, 24) X(LD1Threev16b, 48)                                       \
  X(LD1Threev4h, 24) X(LD1Threev8h,  48)                                       \
  X(LD1Threev2s, 24) X(LD1Threev4s,  48)                                       \
  X(LD1Threev1d, 24) X(LD1Threev2d,  48)                                       \
  X(ST1Threev8b, 24) X(ST1Threev16b, 48)                                       \
  X(ST1Threev4h, 24) X(ST1Threev8h,  48)                                       \
  X(ST1Threev2s, 24) X(ST1Threev4s,  48)                                       \
  X(ST1Threev1d, 24) X(ST1Threev2d,  48)                                       \
  /* LD1/ST1 Four registers. */                                                \
  X(LD1Fourv8b, 32) X(LD1Fourv16b, 64)                                         \
  X(LD1Fourv4h, 32) X(LD1Fourv8h,  64)                                         \
  X(LD1Fourv2s, 32) X(LD1Fourv4s,  64)                                         \
  X(LD1Fourv1d, 32) X(LD1Fourv2d,  64)                                         \
  X(ST1Fourv8b, 32) X(ST1Fourv16b, 64)                                         \
  X(ST1Fourv4h, 32) X(ST1Fourv8h,  64)                                         \
  X(ST1Fourv2s, 32) X(ST1Fourv4s,  64)                                         \
  X(ST1Fourv1d, 32) X(ST1Fourv2d,  64)                                         \
  /* LD2/ST2 Two registers. */                                                 \
  X(LD2Twov8b, 16) X(LD2Twov16b, 32)                                           \
  X(LD2Twov4h, 16) X(LD2Twov8h,  32)                                           \
  X(LD2Twov2s, 16) X(LD2Twov4s,  32) X(LD2Twov2d, 32)                          \
  X(ST2Twov8b, 16) X(ST2Twov16b, 32)                                           \
  X(ST2Twov4h, 16) X(ST2Twov8h,  32)                                           \
  X(ST2Twov2s, 16) X(ST2Twov4s,  32) X(ST2Twov2d, 32)                          \
  /* LD3/ST3 Three registers. */                                               \
  X(LD3Threev8b, 24) X(LD3Threev16b, 48)                                       \
  X(LD3Threev4h, 24) X(LD3Threev8h,  48)                                       \
  X(LD3Threev2s, 24) X(LD3Threev4s,  48) X(LD3Threev2d, 48)                    \
  X(ST3Threev8b, 24) X(ST3Threev16b, 48)                                       \
  X(ST3Threev4h, 24) X(ST3Threev8h,  48)                                       \
  X(ST3Threev2s, 24) X(ST3Threev4s,  48) X(ST3Threev2d, 48)                    \
  /* LD4/ST4 Four registers. */                                                \
  X(LD4Fourv8b, 32) X(LD4Fourv16b, 64)                                         \
  X(LD4Fourv4h, 32) X(LD4Fourv8h,  64)                                         \
  X(LD4Fourv2s, 32) X(LD4Fourv4s,  64) X(LD4Fourv2d,  64)                      \
  X(ST4Fourv8b, 32) X(ST4Fourv16b, 64)                                         \
  X(ST4Fourv4h, 32) X(ST4Fourv8h,  64)                                         \
  X(ST4Fourv2s, 32) X(ST4Fourv4s,  64) X(ST4Fourv2d,  64)
// clang-format on

// Union of all SIMD post-index ops.
#define SIMD_POST_OPS(X)                                                       \
  SIMD_LANE_STORE_OPS(X)                                                       \
  SIMD_LANE_LOAD_OPS(X)                                                        \
  SIMD_MULTI_OPS(X)

// Memory Instruction Info Table
//
// Returns information about how to find the base register and offset operands
// in a memory instruction, and whether it uses pre/post-indexed addressing.
// Used by rewriteLoadStoreBase as a fallback when RoW conversion isn't
// possible.
static std::optional<MemInstInfo> getMemInstInfo(unsigned Op) {
  switch (Op) {
  default:
    return std::nullopt;

  // PC-relative literal loads: Rt, label
  case AArch64::LDRWl:
  case AArch64::LDRXl:
  case AArch64::LDRSWl:
  case AArch64::LDRSl:
  case AArch64::LDRDl:
  case AArch64::LDRQl:
  case AArch64::PRFMl:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{0, std::nullopt, false, true};

    // Scalar indexed/unscaled/register-offset: Rt, [Rn, ...]
#define X(NAME, S)                                                             \
  case AArch64::NAME##ui:                                                      \
  case AArch64::NAME##roW:                                                     \
  case AArch64::NAME##roX:
    SCALAR_MEM_OPS(X)
#undef X
  case AArch64::PRFMui:
  case AArch64::PRFMroW:
  case AArch64::PRFMroX:
  // Unscaled variants.
  case AArch64::LDURBBi:
  case AArch64::LDURBi:
  case AArch64::LDURDi:
  case AArch64::LDURHHi:
  case AArch64::LDURHi:
  case AArch64::LDURQi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSBXi:
  case AArch64::LDURSHWi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSWi:
  case AArch64::LDURSi:
  case AArch64::LDURWi:
  case AArch64::LDURXi:
  case AArch64::STURBBi:
  case AArch64::STURBi:
  case AArch64::STURDi:
  case AArch64::STURHHi:
  case AArch64::STURHi:
  case AArch64::STURQi:
  case AArch64::STURSi:
  case AArch64::STURWi:
  case AArch64::STURXi:
  case AArch64::PRFUMi:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{1, 2, false, false};

    // Scalar pre/post-index: wback, Rt, [Rn], #imm
#define X(NAME, S)                                                             \
  case AArch64::NAME##pre:                                                     \
  case AArch64::NAME##post:
    SCALAR_MEM_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, 3, true, false};

  // Exclusive/acquire loads: Rt, [Rn]
  case AArch64::LDXRB:
  case AArch64::LDXRH:
  case AArch64::LDXRW:
  case AArch64::LDXRX:
  case AArch64::LDAXRB:
  case AArch64::LDAXRH:
  case AArch64::LDAXRW:
  case AArch64::LDAXRX:
  case AArch64::LDARB:
  case AArch64::LDARH:
  case AArch64::LDARW:
  case AArch64::LDARX:
  case AArch64::LDLARB:
  case AArch64::LDLARH:
  case AArch64::LDLARW:
  case AArch64::LDLARX:
  // RCPC loads: Rt, [Rn]
  case AArch64::LDAPRB:
  case AArch64::LDAPRH:
  case AArch64::LDAPRW:
  case AArch64::LDAPRX:
  // Store-release: Rt, [Rn]
  case AArch64::STLRB:
  case AArch64::STLRH:
  case AArch64::STLRW:
  case AArch64::STLRX:
  case AArch64::STLLRB:
  case AArch64::STLLRH:
  case AArch64::STLLRW:
  case AArch64::STLLRX:
    // SIMD multiple/replicate (base form): Vt, [Rn]
#define X(NAME, OFF) case AArch64::NAME:
    SIMD_MULTI_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{1, std::nullopt, false, false};

  // Exclusive stores: Ws, Rt, [Rn]
  case AArch64::STXRB:
  case AArch64::STXRH:
  case AArch64::STXRW:
  case AArch64::STXRX:
  case AArch64::STLXRB:
  case AArch64::STLXRH:
  case AArch64::STLXRW:
  case AArch64::STLXRX:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, std::nullopt, false, false};

  // Exclusive pair loads: Rt, Rt2, [Rn]
  case AArch64::LDXPW:
  case AArch64::LDXPX:
  case AArch64::LDAXPW:
  case AArch64::LDAXPX:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, std::nullopt, false, false};

    // Pair indexed loads/stores: Rt, Rt2, [Rn, #imm]
#define X(NAME, SCALE) case AArch64::NAME##i:
    PAIR_OPS(X)
#undef X
  case AArch64::LDNPDi:
  case AArch64::LDNPQi:
  case AArch64::LDNPSi:
  case AArch64::LDNPWi:
  case AArch64::LDNPXi:
  case AArch64::STNPDi:
  case AArch64::STNPQi:
  case AArch64::STNPSi:
  case AArch64::STNPWi:
  case AArch64::STNPXi:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, 3, false, false};

    // CAS: Rs(out), Rs(in, tied), Rt, [Rn]
#define CAS_VARIANTS(NAME)                                                     \
  case AArch64::NAME##B:                                                       \
  case AArch64::NAME##H:                                                       \
  case AArch64::NAME##W:                                                       \
  case AArch64::NAME##X:
    CAS_VARIANTS(CAS)
    CAS_VARIANTS(CASA)
    CAS_VARIANTS(CASL)
    CAS_VARIANTS(CASAL)
#undef CAS_VARIANTS
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, std::nullopt, false, false};

    // LSE atomics: Rs, Rt, [Rn] - all 16 size/ordering variants per operation.
#define LSE_VARIANTS(NAME)                                                     \
  case AArch64::NAME##B:                                                       \
  case AArch64::NAME##H:                                                       \
  case AArch64::NAME##W:                                                       \
  case AArch64::NAME##X:                                                       \
  case AArch64::NAME##AB:                                                      \
  case AArch64::NAME##AH:                                                      \
  case AArch64::NAME##AW:                                                      \
  case AArch64::NAME##AX:                                                      \
  case AArch64::NAME##LB:                                                      \
  case AArch64::NAME##LH:                                                      \
  case AArch64::NAME##LW:                                                      \
  case AArch64::NAME##LX:                                                      \
  case AArch64::NAME##ALB:                                                     \
  case AArch64::NAME##ALH:                                                     \
  case AArch64::NAME##ALW:                                                     \
  case AArch64::NAME##ALX:
    LSE_VARIANTS(LDADD)
    LSE_VARIANTS(LDCLR)
    LSE_VARIANTS(LDEOR)
    LSE_VARIANTS(LDSET)
    LSE_VARIANTS(LDSMAX)
    LSE_VARIANTS(LDSMIN)
    LSE_VARIANTS(LDUMAX)
    LSE_VARIANTS(LDUMIN)
    LSE_VARIANTS(SWP)
#undef LSE_VARIANTS
    // SIMD lane stores (base form): Vt, idx, [Rn]
#define X(NAME, OFF) case AArch64::NAME:
    SIMD_LANE_STORE_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, std::nullopt, false, false};

    // SIMD lane loads (base form): Vt(out), Vt(tied), idx, [Rn]
#define X(NAME, OFF) case AArch64::NAME:
    SIMD_LANE_LOAD_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, std::nullopt, false, false};

  // Exclusive store pairs: Ws, Rt, Rt2, [Rn]
  case AArch64::STXPW:
  case AArch64::STXPX:
  case AArch64::STLXPW:
  case AArch64::STLXPX:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, std::nullopt, false, false};

    // Pair pre/post-index: wback, Rt, Rt2, [Rn, #imm]! / [Rn], #imm
#define X(NAME, SCALE)                                                         \
  case AArch64::NAME##pre:                                                     \
  case AArch64::NAME##post:
    PAIR_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, 4, true, false};

  // CASP: Ws_Ws2(out), Ws_Ws2(in, tied), Wt_Wt2, [Rn]
  case AArch64::CASPW:
  case AArch64::CASPX:
  case AArch64::CASPAW:
  case AArch64::CASPAX:
  case AArch64::CASPLW:
  case AArch64::CASPLX:
  case AArch64::CASPALW:
  case AArch64::CASPALX:
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, std::nullopt, false, false};

    // SIMD multiple/replicate post-index: wback, Vt, [Rn], Xm
#define X(NAME, OFF) case AArch64::NAME##_POST:
    SIMD_MULTI_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{2, 3, true, false};

    // SIMD lane store post-index: wback, Vt, idx, [Rn], Xm
#define X(NAME, OFF) case AArch64::NAME##_POST:
    SIMD_LANE_STORE_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{3, 4, true, false};

    // SIMD lane load post-index: wback, Vt(out), Vt(tied), idx, [Rn], Xm
#define X(NAME, OFF) case AArch64::NAME##_POST:
    SIMD_LANE_LOAD_OPS(X)
#undef X
    // BaseIdx, OffsetIdx, IsPrePost, IsLiteral
    return MemInstInfo{4, 5, true, false};
  }
}

// RoW (Register-offset-W) Opcode Conversion Tables
//
// These tables convert various load/store addressing modes to the
// register-offset-W form ([X27, Wn, uxtw]) which provides sandboxing in a
// single instruction by zero-extending the 32-bit offset register.
//
// All scalar load/store instructions that support RoW addressing are listed
// here. Each entry is (NAME, SHIFT) where NAME is the opcode base (e.g. LDRX)
// and SHIFT is the log2 element size used for scaled addressing.
static unsigned convertUiToRoW(unsigned Op) {
  switch (Op) {
#define X(NAME, S)                                                             \
  case AArch64::NAME##ui:                                                      \
    return AArch64::NAME##roW;
    SCALAR_MEM_OPS(X)
#undef X
  case AArch64::PRFMui:
    return AArch64::PRFMroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

static unsigned convertPreToRoW(unsigned Op) {
  switch (Op) {
#define X(NAME, S)                                                             \
  case AArch64::NAME##pre:                                                     \
    return AArch64::NAME##roW;
    SCALAR_MEM_OPS(X)
#undef X
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

static unsigned convertPostToRoW(unsigned Op) {
  switch (Op) {
#define X(NAME, S)                                                             \
  case AArch64::NAME##post:                                                    \
    return AArch64::NAME##roW;
    SCALAR_MEM_OPS(X)
#undef X
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

static unsigned convertRoXToRoW(unsigned Op, unsigned &Shift) {
  Shift = 0;
  switch (Op) {
#define X(NAME, S)                                                             \
  case AArch64::NAME##roX:                                                     \
    Shift = S;                                                                 \
    return AArch64::NAME##roW;
    SCALAR_MEM_OPS(X)
#undef X
  case AArch64::PRFMroX:
    Shift = 3;
    return AArch64::PRFMroW;
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

static bool getRoWShift(unsigned Op, unsigned &Shift) {
  Shift = 0;
  switch (Op) {
#define X(NAME, S)                                                             \
  case AArch64::NAME##roW:                                                     \
    Shift = S;                                                                 \
    return true;
    SCALAR_MEM_OPS(X)
#undef X
  case AArch64::PRFMroW:
    Shift = 3;
    return true;
  default:
    return false;
  }
}

// Pre/Post-Index to Base Conversion and Natural Offset Tables
//
// SIMD post-index instructions are listed once in SIMD_POST_OPS and used to
// generate both convertPrePostToBase and getSIMDNaturalOffset.
// Each entry is (NAME, OFFSET) where the post opcode is NAME##_POST, the base
// opcode is NAME, and OFFSET is the natural byte increment.
static unsigned convertPrePostToBase(unsigned Op, bool &IsPre,
                                     bool &IsNoOffset) {
  IsPre = false;
  IsNoOffset = false;
  switch (Op) {
    // LDP/STP pairs.
#define X(NAME, SCALE)                                                         \
  case AArch64::NAME##post:                                                    \
    return AArch64::NAME##i;                                                   \
  case AArch64::NAME##pre:                                                     \
    IsPre = true;                                                              \
    return AArch64::NAME##i;
    PAIR_OPS(X)
#undef X
    // SIMD post-index.
#define X(NAME, OFF)                                                           \
  case AArch64::NAME##_POST:                                                   \
    IsNoOffset = true;                                                         \
    return AArch64::NAME;
    SIMD_POST_OPS(X)
#undef X
  default:
    return AArch64::INSTRUCTION_LIST_END;
  }
}

static unsigned getPrePostScale(unsigned Op) {
  switch (Op) {
#define X(NAME, SCALE)                                                         \
  case AArch64::NAME##post:                                                    \
  case AArch64::NAME##pre:                                                     \
    return SCALE;
    PAIR_OPS(X)
#undef X
  default:
    return 1;
  }
}

static int getSIMDNaturalOffset(unsigned Op) {
  switch (Op) {
#define X(NAME, OFF)                                                           \
  case AArch64::NAME##_POST:                                                   \
    return OFF;
    SIMD_POST_OPS(X)
#undef X
  default:
    return -1;
  }
}
