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

// Find the index of the memory operand if it has an %fs segment override.
// Returns -1 if there is no memory operand or no %fs override.
static int findFSMemOperand(const MCInst &Inst, const MCInstrInfo &InstInfo) {
  const MCInstrDesc &Desc = InstInfo.get(Inst.getOpcode());
  int MemRefIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefIdx < 0)
    return -1;
  int MemIdx = MemRefIdx + X86II::getOperandBias(Desc);
  const MCOperand &Seg = Inst.getOperand(MemIdx + X86::AddrSegmentReg);
  if (Seg.isReg() && Seg.getReg() == X86::FS)
    return MemIdx;
  return -1;
}

// syscall
// ->
// leaq .Ltmp(%rip), %r11
// jmpq *(%r14)
// .Ltmp:
void X86::X86MCLFIRewriter::rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                                           const MCSubtargetInfo &STI) {
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
  Jmp.addOperand(MCOperand::createImm(-8));
  Jmp.addOperand(MCOperand::createReg(X86::NoRegister));
  Out.emitInstruction(Jmp, STI);

  Out.emitLabel(Symbol);
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
// movq %fs:8(%rdi, %rsi, 2), %rax
// ->
// movq 16(%r15), %rax
// leaq (%rax, %rdi), %rax
// movq 8(%rax, %rsi, 2), %rax
void X86::X86MCLFIRewriter::rewriteFSAccess(const MCInst &Inst, MCStreamer &Out,
                                            const MCSubtargetInfo &STI) {
  int MemIdx = findFSMemOperand(Inst, *InstInfo);
  assert(MemIdx >= 0);

  MCRegister BaseReg = Inst.getOperand(MemIdx + X86::AddrBaseReg).getReg();
  MCRegister IndexReg = Inst.getOperand(MemIdx + X86::AddrIndexReg).getReg();
  bool HasBase = BaseReg != X86::NoRegister;
  bool HasIndex = IndexReg != X86::NoRegister;
  bool HasDisp = !Inst.getOperand(MemIdx + X86::AddrDisp).isImm() ||
                 Inst.getOperand(MemIdx + X86::AddrDisp).getImm() != 0;

  // %fs:0 -> TPOffset(%r15)
  if (!HasBase && !HasIndex && !HasDisp) {
    MCInst Modified(Inst);
    Modified.getOperand(MemIdx + X86::AddrBaseReg).setReg(LFITPReg);
    Modified.getOperand(MemIdx + X86::AddrDisp).setImm(TPOffset);
    Modified.getOperand(MemIdx + X86::AddrSegmentReg).setReg(X86::NoRegister);
    return Out.emitInstruction(Modified, STI);
  }

  // Use the dest register as TP temporary when it is available and not used in
  // the addressing mode, otherwise use %r11 (e.g., movq %fs:(%rax), %rax).
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

  // Both slots occupied: fold base into TPDest via lea. For example:
  //
  // movq %fs:8(%rdi,%rsi,2), %rax
  // ->
  // movq 16(%r15), %rax
  // leaq (%rax,%rdi), %rax
  // movq 8(%rax,%rsi,2), %rax
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

  // Emit the access with TPDest as the new base, and the original base
  // (offset from %fs) as the new index. For example:
  //
  // movq %fs:(%rdi), %rax
  // ->
  // movq 16(%r15), %rax
  // movq (%rax,%rdi), %rax
  MCInst Modified(Inst);
  Modified.getOperand(MemIdx + X86::AddrBaseReg).setReg(TPDest);
  if (HasBase && !HasIndex)
    Modified.getOperand(MemIdx + X86::AddrIndexReg).setReg(BaseReg);
  Modified.getOperand(MemIdx + X86::AddrSegmentReg).setReg(X86::NoRegister);
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
