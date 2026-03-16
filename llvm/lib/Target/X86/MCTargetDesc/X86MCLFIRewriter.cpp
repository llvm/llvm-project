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

static bool isTPRead(const MCInst &Inst) {
  // Match movq %fs:0, %rX
  return Inst.getOpcode() == X86::MOV64rm &&
         Inst.getOperand(1).getReg() == X86::NoRegister &&
         Inst.getOperand(2).isImm() && Inst.getOperand(2).getImm() == 1 &&
         Inst.getOperand(3).getReg() == X86::NoRegister &&
         Inst.getOperand(4).isImm() && Inst.getOperand(4).getImm() == 0 &&
         Inst.getOperand(5).getReg() == X86::FS;
}

// syscall
// ->
// leaq .Ltmp(%rip), %r11
// jmpq *(%r14)
// .Ltmp:
void X86::X86MCLFIRewriter::emitLFICall(MCStreamer &Out,
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
  Jmp.addOperand(MCOperand::createImm(0));
  Jmp.addOperand(MCOperand::createReg(X86::NoRegister));
  Out.emitInstruction(Jmp, STI);

  Out.emitLabel(Symbol);
}

void X86::X86MCLFIRewriter::expandSyscall(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  emitLFICall(Out, STI);
}

// movq %fs:0, %rX
// ->
// movq TPOffset(%r15), %rX
void X86::X86MCLFIRewriter::expandTPRead(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  MCRegister DestReg = Inst.getOperand(0).getReg();

  MCInst Mov;
  Mov.setOpcode(X86::MOV64rm);
  Mov.addOperand(MCOperand::createReg(DestReg));
  Mov.addOperand(MCOperand::createReg(LFITPReg));        // Base
  Mov.addOperand(MCOperand::createImm(1));               // Scale
  Mov.addOperand(MCOperand::createReg(X86::NoRegister)); // Index
  Mov.addOperand(MCOperand::createImm(TPOffset));        // Displacement
  Mov.addOperand(MCOperand::createReg(X86::NoRegister)); // Segment
  Out.emitInstruction(Mov, STI);
}

void X86::X86MCLFIRewriter::doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  if (mayModifyRegister(Inst, LFIBaseReg) || mayModifyRegister(Inst, LFITPReg))
    return error(Inst, "illegal modification of reserved LFI register");

  if (isSyscall(Inst))
    return expandSyscall(Inst, Out, STI);

  if (isTPRead(Inst))
    return expandTPRead(Inst, Out, STI);

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
