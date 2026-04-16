//===- X86MCLFIRewriter.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86MCLFIRewriter class, which rewrites X86-64
// instructions for LFI (Lightweight Fault Isolation) sandboxing.
//
//===----------------------------------------------------------------------===//

#include "X86MCLFIRewriter.h"
#include "X86BaseInfo.h"
#include "X86MCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// LFI reserved registers.
static constexpr MCRegister LFIBaseReg = X86::R14;
static constexpr MCRegister LFIScratchReg = X86::R11;
static constexpr MCRegister LFITPReg = X86::R15;

// Byte offset into the context register file (pointed to by R15) where the
// thread pointer is stored.
static constexpr int TPOffset = 16;

static bool isSyscall(const MCInst &Inst) {
  return Inst.getOpcode() == X86::SYSCALL;
}

// Find the index of the first memory operand with %fs segment override.
// Returns -1 if not found.
static int findFSMemOperand(const MCInst &Inst, const MCInstrInfo &InstInfo) {
  const MCInstrDesc &Desc = InstInfo.get(Inst.getOpcode());
  for (unsigned I = 0, E = Desc.getNumOperands(); I < E; ++I) {
    if (Desc.operands()[I].OperandType == MCOI::OPERAND_MEMORY) {
      if (I + 4 < Inst.getNumOperands() && Inst.getOperand(I + 4).isReg() &&
          Inst.getOperand(I + 4).getReg() == X86::FS)
        return I;
      I += 4;
    }
  }
  return -1;
}

// syscall
// ->
// leaq .Ltmp(%rip), %r11
// jmpq *(%r14)
// .Ltmp:
static void emitLFICall(MCStreamer &Out, const MCSubtargetInfo &STI) {
  MCSymbol *Symbol = Out.getContext().createTempSymbol();

  // leaq .Ltmp(%rip), %r11
  MCInst Lea;
  Lea.setOpcode(X86::LEA64r);
  Lea.addOperand(MCOperand::createReg(LFIScratchReg));
  Lea.addOperand(MCOperand::createReg(X86::RIP));
  Lea.addOperand(MCOperand::createImm(1));
  Lea.addOperand(MCOperand::createReg(X86::NoRegister));
  Lea.addOperand(
      MCOperand::createExpr(MCSymbolRefExpr::create(Symbol, Out.getContext())));
  Lea.addOperand(MCOperand::createReg(X86::NoRegister));
  Out.emitInstruction(Lea, STI);

  // jmpq *(%r14)
  MCInst Jmp;
  Jmp.setOpcode(X86::JMP64m);
  Jmp.addOperand(MCOperand::createReg(LFIBaseReg));
  Jmp.addOperand(MCOperand::createImm(1));
  Jmp.addOperand(MCOperand::createReg(X86::NoRegister));
  Jmp.addOperand(MCOperand::createImm(0));
  Jmp.addOperand(MCOperand::createReg(X86::NoRegister));
  Out.emitInstruction(Jmp, STI);

  Out.emitLabel(Symbol);
}

void X86::X86MCLFIRewriter::rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                                           const MCSubtargetInfo &STI) {
  emitLFICall(Out, STI);
}

// Emit: movq TPOffset(%r15), %Reg
static void emitTPLoad(MCRegister Reg, MCStreamer &Out,
                       const MCSubtargetInfo &STI) {
  MCInst Mov;
  Mov.setOpcode(X86::MOV64rm);
  Mov.addOperand(MCOperand::createReg(Reg));
  Mov.addOperand(MCOperand::createReg(LFITPReg));
  Mov.addOperand(MCOperand::createImm(1));
  Mov.addOperand(MCOperand::createReg(X86::NoRegister));
  Mov.addOperand(MCOperand::createImm(TPOffset));
  Mov.addOperand(MCOperand::createReg(X86::NoRegister));
  Out.emitInstruction(Mov, STI);
}

bool X86::X86MCLFIRewriter::isFSAccess(const MCInst &Inst) {
  return (mayLoad(Inst) || mayStore(Inst)) &&
         findFSMemOperand(Inst, *InstInfo) >= 0;
}

// Rewrite %fs-segment memory accesses to use the virtual thread pointer stored
// at TPOffset(%r15). The actual memory access is currently unsandboxed because
// load/store sandboxing is not yet supported. Example rewrites:
//
// movq %fs:0, %rax
// ->
// movq 16(%r15), %rax
//
// movq %fs:(%rdi), %rax
// ->
// movq 16(%r15), %rax
// movq (%rax, %rdi), %rax
//
// movq 8(%rdi, %rsi, 2), %rax
// ->
// movq 16(%r15), %rax
// leaq (%rax, %rdi), %rax
// movq 8(%rax, %rsi, 2), %rax
void X86::X86MCLFIRewriter::rewriteFSAccess(const MCInst &Inst, MCStreamer &Out,
                                            const MCSubtargetInfo &STI) {
  int MemIdx = findFSMemOperand(Inst, *InstInfo);
  assert(MemIdx >= 0);

  MCRegister BaseReg = Inst.getOperand(MemIdx).getReg();
  MCRegister IndexReg = Inst.getOperand(MemIdx + 2).getReg();
  bool HasBase = BaseReg != X86::NoRegister;
  bool HasIndex = IndexReg != X86::NoRegister;
  bool HasDisp = !Inst.getOperand(MemIdx + 3).isImm() ||
                 Inst.getOperand(MemIdx + 3).getImm() != 0;

  // %fs:0 -> TPOffset(%r15)
  if (!HasBase && !HasIndex && !HasDisp) {
    MCInst Modified(Inst);
    Modified.getOperand(MemIdx).setReg(LFITPReg);
    Modified.getOperand(MemIdx + 3).setImm(TPOffset);
    Modified.getOperand(MemIdx + 4).setReg(X86::NoRegister);
    return Out.emitInstruction(Modified, STI);
  }

  // Use the dest register as TP temporary when it is available and not used in
  // the addressing mode, otherwise use %r11.
  MCRegister TPDest = LFIScratchReg;
  if (MemIdx > 0 && Inst.getOperand(0).isReg()) {
    const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());
    MCRegister DestReg = Inst.getOperand(0).getReg();
    if (Desc.getOperandConstraint(0, MCOI::TIED_TO) == -1 &&
        X86MCRegisterClasses[X86::GR64RegClassID].contains(DestReg) &&
        (!HasBase || DestReg != BaseReg) && (!HasIndex || DestReg != IndexReg))
      TPDest = DestReg;
  }

  emitTPLoad(TPDest, Out, STI);

  // Both slots occupied: fold base into TPDest via lea.
  if (HasBase && HasIndex) {
    MCInst Lea;
    Lea.setOpcode(X86::LEA64r);
    Lea.addOperand(MCOperand::createReg(TPDest));
    Lea.addOperand(MCOperand::createReg(TPDest));
    Lea.addOperand(MCOperand::createImm(1));
    Lea.addOperand(MCOperand::createReg(BaseReg));
    Lea.addOperand(MCOperand::createImm(0));
    Lea.addOperand(MCOperand::createReg(X86::NoRegister));
    Out.emitInstruction(Lea, STI);
  }

  MCInst Modified(Inst);
  Modified.getOperand(MemIdx).setReg(TPDest);
  if (HasBase && !HasIndex)
    Modified.getOperand(MemIdx + 2).setReg(BaseReg);
  Modified.getOperand(MemIdx + 4).setReg(X86::NoRegister);
  Out.emitInstruction(Modified, STI);
}

void X86::X86MCLFIRewriter::doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  if (mayModifyRegister(Inst, LFIBaseReg) || mayModifyRegister(Inst, LFITPReg))
    return error(Inst, "illegal modification of reserved LFI register");

  if (isSyscall(Inst))
    return rewriteSyscall(Inst, Out, STI);

  if (isFSAccess(Inst))
    return rewriteFSAccess(Inst, Out, STI);

  // Pass through all other instructions unchanged.
  Out.emitInstruction(Inst, STI);
}

bool X86::X86MCLFIRewriter::rewriteInst(const MCInst &Inst, MCStreamer &Out,
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
