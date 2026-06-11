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

namespace llvm::AArch64 {
struct LFIVariantEntry {
  unsigned Inst;
  uint8_t AddrMode;
  uint8_t Log2Size;
  unsigned RoWInst;
};
struct PairVariantEntry {
  unsigned Inst;
  bool IsPre;
  uint8_t Scale;
  unsigned BaseInst;
};
struct SIMDPostEntry {
  unsigned Inst;
  uint8_t NaturalOffset;
  unsigned BaseInst;
};
struct MemInfoEntry {
  unsigned Inst;
  uint8_t BaseIdx;
  uint8_t OffsetIdx;
  bool HasOffset;
  bool IsPrePost;
  bool IsLiteral;
};

// LFI addressing-mode codes (must match AArch64LFI.td's LFI_AM_* defs).
enum LFIAddrMode : uint8_t {
  LFI_AM_Ui = 0,
  LFI_AM_RoW = 1,
  LFI_AM_RoX = 2,
  LFI_AM_Pre = 3,
  LFI_AM_Post = 4,
};

#define GET_LFIVariantTable_DECL
#define GET_PairVariantTable_DECL
#define GET_SIMDPostTable_DECL
#define GET_MemInfoTable_DECL
#define GET_LFIVariantTable_IMPL
#define GET_PairVariantTable_IMPL
#define GET_SIMDPostTable_IMPL
#define GET_MemInfoTable_IMPL
// The LFI tables defined in AArch64LFI.td are emitted into this file alongside
// the system operand tables (single -gen-searchable-tables output).
#include "AArch64GenSystemOperands.inc"
} // namespace llvm::AArch64

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
  case AArch64::CLREX:
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
  MCInst New = Inst;
  assert(New.getOperand(Idx).isReg());
  New.getOperand(Idx).setReg(NewReg);
  return New;
}

// AArch64 load/store opcode suffixes used throughout this file:
//   Ui:  Unsigned immediate offset, scaled by access size: [Xn, #imm].
//   RoW: Register offset with 32-bit W register: [Xn, Wm, uxtw #shift].
//   RoX: Register offset with 64-bit X register: [Xn, Xm, lsl #shift].

// Scalar load/store variant lookup. If Op is a scalar mem instruction with
// addressing mode ExpectedMode, returns the RoW variant of the same family.
// Returns INSTRUCTION_LIST_END otherwise.
static unsigned convertVariantToRoW(unsigned Op, unsigned ExpectedMode) {
  const AArch64::LFIVariantEntry *E = AArch64::lookupLFIVariantByOpcode(Op);
  if (!E || E->AddrMode != ExpectedMode)
    return AArch64::INSTRUCTION_LIST_END;
  return E->RoWInst;
}

static unsigned convertRoXToRoW(unsigned Op, unsigned &Shift) {
  Shift = 0;
  const AArch64::LFIVariantEntry *E = AArch64::lookupLFIVariantByOpcode(Op);
  if (!E || E->AddrMode != AArch64::LFI_AM_RoX)
    return AArch64::INSTRUCTION_LIST_END;
  Shift = E->Log2Size;
  return E->RoWInst;
}

static bool getRoWShift(unsigned Op, unsigned &Shift) {
  Shift = 0;
  const AArch64::LFIVariantEntry *E = AArch64::lookupLFIVariantByOpcode(Op);
  if (!E || E->AddrMode != AArch64::LFI_AM_RoW)
    return false;
  Shift = E->Log2Size;
  return true;
}

// Pre/post-index conversion to base form. Both LDP/STP pair pre/post forms and
// SIMD post-index forms come from generated lookup tables. The pair table sets
// IsPre to distinguish pre-index from post-index. The SIMD table is
// post-index-only so IsNoOffset is set to indicate the demoted base form takes
// no immediate offset.
static unsigned convertPrePostToBase(unsigned Op, bool &IsPre,
                                     bool &IsNoOffset) {
  IsPre = false;
  IsNoOffset = false;
  if (const auto *E = AArch64::lookupPairVariantByOpcode(Op)) {
    IsPre = E->IsPre;
    return E->BaseInst;
  }
  if (const auto *E = AArch64::lookupSIMDPostByOpcode(Op)) {
    IsNoOffset = true;
    return E->BaseInst;
  }
  return AArch64::INSTRUCTION_LIST_END;
}

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
  if ((MemOp = convertVariantToRoW(Op, AArch64::LFI_AM_Ui)) !=
      AArch64::INSTRUCTION_LIST_END) {
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
  if ((MemOp = convertVariantToRoW(Op, AArch64::LFI_AM_Pre)) !=
      AArch64::INSTRUCTION_LIST_END) {
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
  if ((MemOp = convertVariantToRoW(Op, AArch64::LFI_AM_Post)) !=
      AArch64::INSTRUCTION_LIST_END) {
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
  const AArch64::MemInfoEntry *Info = AArch64::lookupMemInfoByOpcode(Opcode);

  if (!Info) {
    warning(Inst, "unknown addressing mode for memory instruction in LFI");
    return emitInst(Inst, Out, STI);
  }

  if (Info->IsLiteral)
    return error(Inst, "PC-relative literal loads are not supported in LFI");

  MCRegister BaseReg = Inst.getOperand(Info->BaseIdx).getReg();

  // Stack accesses don't need address sandboxing, except when sp is modified
  // with a non-zero register post-index operand.
  bool BaseIsSP = BaseReg == AArch64::SP;
  if (BaseIsSP) {
    if (!Info->HasOffset || !Inst.getOperand(Info->OffsetIdx).isReg())
      return emitInst(Inst, Out, STI);
    MCRegister OffReg = Inst.getOperand(Info->OffsetIdx).getReg();
    if (OffReg == AArch64::XZR || OffReg == AArch64::WZR)
      return emitInst(Inst, Out, STI);
  }

  // Guard the base register, unless it is SP.
  if (!BaseIsSP)
    emitAddMask(LFIAddrReg, BaseReg, Out, STI);

  if (!Info->IsPrePost) {
    // Non-pre/post instruction: replace the base register operand.
    MCInst NewInst = replaceRegAt(Inst, Info->BaseIdx, LFIAddrReg);
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
  for (int I = 1; I < Info->BaseIdx; ++I)
    NewInst.addOperand(Inst.getOperand(I));

  // Add the access base register (LFIAddrReg or SP).
  MCRegister AccessBase = BaseIsSP ? AArch64::SP : LFIAddrReg;
  NewInst.addOperand(MCOperand::createReg(AccessBase));

  // For pre-index, include the offset; for post-index, use zero.
  if (IsPre && Info->HasOffset)
    NewInst.addOperand(Inst.getOperand(Info->OffsetIdx));
  else if (!IsNoOffset)
    NewInst.addOperand(MCOperand::createImm(0));

  emitInst(NewInst, Out, STI);

  if (!Info->HasOffset)
    return;

  // Update the base register with the offset. If the base is SP, a register
  // offset must be sandboxed (the result is otherwise unbounded), and ADDXrs
  // cannot take SP, so the extended-register form via the scratch register is
  // used.
  const MCOperand &OffsetOp = Inst.getOperand(Info->OffsetIdx);
  if (OffsetOp.isImm()) {
    // Pair pre/post immediates are scaled by element size; other pre/post
    // forms (scalar, SIMD) use the raw immediate (scale = 1).
    int64_t Scale = 1;
    if (const auto *E = AArch64::lookupPairVariantByOpcode(Opcode))
      Scale = E->Scale;
    int64_t Offset = OffsetOp.getImm() * Scale;
    emitAddImm(BaseReg, BaseReg, Offset, Out, STI);
  } else if (OffsetOp.isReg()) {
    // SIMD post-index uses a register offset (XZR for natural offset).
    MCRegister OffReg = OffsetOp.getReg();
    if (OffReg == AArch64::XZR) {
      if (const auto *E = AArch64::lookupSIMDPostByOpcode(Opcode))
        emitAddImm(BaseReg, BaseReg, E->NaturalOffset, Out, STI);
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

  // Special case: mov sp, xN -> add sp, x27, wN, uxtw
  if (Inst.getOpcode() == AArch64::ADDXri && Inst.getOperand(2).getImm() == 0 &&
      Inst.getOperand(3).getImm() == 0)
    return emitAddMask(AArch64::SP, Inst.getOperand(1).getReg(), Out, STI);

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
  if (convertVariantToRoW(Opcode, AArch64::LFI_AM_Pre) !=
      AArch64::INSTRUCTION_LIST_END)
    return true;
  if (convertVariantToRoW(Opcode, AArch64::LFI_AM_Post) !=
      AArch64::INSTRUCTION_LIST_END)
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
