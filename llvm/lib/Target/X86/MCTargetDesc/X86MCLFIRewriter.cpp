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
#include "llvm/MC/MCInstBuilder.h"
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

static bool isDirectCall(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case X86::CALLpcrel32:
  case X86::CALL64pcrel32:
    return true;
  default:
    return false;
  }
}

static bool isSupportedIndirectBranch(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case X86::JMP64r:
  case X86::JMP64r_NT:
  case X86::JMP64m:
  case X86::JMP64m_NT:
  case X86::CALL64r:
  case X86::CALL64r_NT:
  case X86::CALL64m:
  case X86::CALL64m_NT:
    return true;
  default:
    return false;
  }
}

static bool hasNoTrackPrefix(const MCInst &Inst, const MCInstrInfo &InstInfo) {
  return (InstInfo.get(Inst.getOpcode()).TSFlags & X86II::NOTRACK) ||
         (Inst.getFlags() & X86::IP_HAS_NOTRACK);
}

// Find the index of the memory operand if it has an %fs segment override.
// Returns -1 if there is no memory operand or no %fs override.
static int findFSMemOperand(const MCInst &Inst, const MCInstrInfo &InstInfo) {
  int MemIdx = X86II::getMemoryOperandIdx(InstInfo.get(Inst.getOpcode()));
  if (MemIdx < 0)
    return -1;
  const MCOperand &Seg = Inst.getOperand(MemIdx + X86::AddrSegmentReg);
  if (Seg.isReg() && Seg.getReg() == X86::FS)
    return MemIdx;
  return -1;
}

// Return true if the instruction reads from Reg.
static bool readsRegister(const MCInst &Inst, const MCInstrDesc &Desc,
                          MCRegister Reg, const MCRegisterInfo &RI) {
  for (unsigned I = Desc.getNumDefs(), E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && RI.regsOverlap(Op.getReg(), Reg))
      return true;
  }
  for (MCPhysReg Use : Desc.implicit_uses())
    if (RI.regsOverlap(Use, Reg))
      return true;
  return false;
}

// Return true if Reg is absent or a 64-bit general-purpose register.
static bool isGR64OrNone(MCRegister Reg) {
  return Reg == X86::NoRegister ||
         getX86MCRegisterClass(X86::GR64RegClassID).contains(Reg);
}

// syscall
// ->
// .bundle_lock
// leaq .Ltmp(%rip), %r11
// jmpq *(%r14)
// .Ltmp:
// .bundle_unlock
void X86::X86MCLFIRewriter::rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                                           const MCSubtargetInfo &STI) {
  Out.emitBundleLock(/*AlignToEnd=*/false, STI);

  MCSymbol *Symbol = Out.getContext().createTempSymbol();

  // leaq .Ltmp(%rip), %r11
  Out.emitInstruction(
      MCInstBuilder(X86::LEA64r)
          .addReg(LFIScratchReg)
          .addReg(X86::RIP)
          .addImm(1)
          .addReg(X86::NoRegister)
          .addExpr(MCSymbolRefExpr::create(Symbol, Out.getContext()))
          .addReg(X86::NoRegister),
      STI);

  // jmpq *-8(%r14)
  Out.emitInstruction(MCInstBuilder(X86::JMP64m)
                          .addReg(LFIBaseReg)
                          .addImm(1)
                          .addReg(X86::NoRegister)
                          .addImm(-8)
                          .addReg(X86::NoRegister),
                      STI);

  Out.emitLabel(Symbol);
  Out.emitBundleUnlock(STI);
}

// andl $-LFIBundleSize, %eX
// addq %r14, %rX
void X86::X86MCLFIRewriter::emitSandboxBranchReg(MCRegister Reg,
                                                 MCStreamer &Out,
                                                 const MCSubtargetInfo &STI) {
  MCRegister Reg32 = RegInfo->getSubReg(Reg, X86::sub_32bit);

  Out.emitInstruction(MCInstBuilder(X86::AND32ri8)
                          .addReg(Reg32)
                          .addReg(Reg32)
                          .addImm(-static_cast<int64_t>(LFIBundleSize)),
                      STI);

  Out.emitInstruction(
      MCInstBuilder(X86::ADD64rr).addReg(Reg).addReg(Reg).addReg(LFIBaseReg),
      STI);
}

// Rewrite an indirect jump or call so that it can only target a bundle
// boundary inside the sandbox.
//
// jmpq *%rX
// ->
// .bundle_lock
// andl $-32, %eX
// addq %r14, %rX
// jmpq *%rX
// .bundle_unlock
//
// A branch through memory loads its target into the scratch register first,
// and then dispatches through it.
//
// jmpq *(%rdi)
// ->
// movq (%rdi), %r11
// .bundle_lock
// andl $-32, %r11d
// addq %r14, %r11
// jmpq *%r11
// .bundle_unlock
void X86::X86MCLFIRewriter::rewriteIndirectBranch(const MCInst &Inst,
                                                  MCStreamer &Out,
                                                  const MCSubtargetInfo &STI) {
  MCRegister Target;
  int MemIdx = X86II::getMemoryOperandIdx(InstInfo->get(Inst.getOpcode()));
  if (MemIdx >= 0) {
    Target = LFIScratchReg;

    // Construct the load and then apply the rewriter to it.
    MCInstBuilder Mov(X86::MOV64rm);
    Mov.addReg(Target);
    for (unsigned I = 0; I < X86::AddrNumOperands; ++I)
      Mov.addOperand(Inst.getOperand(MemIdx + I));
    doRewriteInst(Mov, Out, STI);
  } else {
    Target = Inst.getOperand(0).getReg();

    if (Target == LFIBaseReg || Target == LFITPReg || Target == X86::RSP)
      return error(Inst, "indirect branch through reserved register");
  }

  Out.emitBundleLock(/*AlignToEnd=*/isCall(Inst), STI);

  emitSandboxBranchReg(Target, Out, STI);

  MCInst Branch =
      MCInstBuilder(isCall(Inst) ? X86::CALL64r : X86::JMP64r).addReg(Target);
  if (hasNoTrackPrefix(Inst, *InstInfo))
    Branch.setFlags(Branch.getFlags() | X86::IP_HAS_NOTRACK);
  Out.emitInstruction(Branch, STI);

  Out.emitBundleUnlock(STI);
}

// Direct calls are not rewritten, but must be placed at the end of a bundle
// so that the return address they push is bundle-aligned.
void X86::X86MCLFIRewriter::rewriteDirectCall(const MCInst &Inst,
                                              MCStreamer &Out,
                                              const MCSubtargetInfo &STI) {
  Out.emitBundleLock(/*AlignToEnd=*/true, STI);
  Out.emitInstruction(Inst, STI);
  Out.emitBundleUnlock(STI);
}

// ret
// ->
// popq %r11
// .bundle_lock
// andl $-32, %r11d
// addq %r14, %r11
// jmpq *%r11
// .bundle_unlock
void X86::X86MCLFIRewriter::rewriteReturn(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  if (Inst.getOpcode() != X86::RET64 && Inst.getOpcode() != X86::RETI64)
    return error(Inst, "unsupported return instruction");

  Out.emitInstruction(MCInstBuilder(X86::POP64r).addReg(LFIScratchReg), STI);

  if (Inst.getOpcode() == X86::RETI64) {
    // Return with an immediate is rewritten recursively so that the stack
    // pointer modification goes through the rewriter.
    doRewriteInst(MCInstBuilder(X86::ADD64ri32)
                      .addReg(X86::RSP)
                      .addReg(X86::RSP)
                      .addOperand(Inst.getOperand(0)),
                  Out, STI);
  }

  Out.emitBundleLock(/*AlignToEnd=*/false, STI);

  emitSandboxBranchReg(LFIScratchReg, Out, STI);

  Out.emitInstruction(MCInstBuilder(X86::JMP64r).addReg(LFIScratchReg), STI);

  Out.emitBundleUnlock(STI);
}

// Emit: movq TPOffset(%r15), %Reg
static void emitTPLoad(MCRegister Reg, MCStreamer &Out,
                       const MCSubtargetInfo &STI) {
  Out.emitInstruction(MCInstBuilder(X86::MOV64rm)
                          .addReg(Reg)
                          .addReg(LFITPReg)
                          .addImm(1)
                          .addReg(X86::NoRegister)
                          .addImm(TPOffset)
                          .addReg(X86::NoRegister),
                      STI);
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

  if (!isGR64OrNone(BaseReg) || !isGR64OrNone(IndexReg) ||
      BaseReg == X86::RSP || BaseReg == X86::RIP)
    return error(Inst, "unsupported addressing mode for %fs access");

  const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());

  // Reuse operand 0 as the TP temporary when the instruction writes it without
  // also reading it, otherwise use %r11.
  MCRegister TPDest = LFIScratchReg;
  if (MemIdx > 0 && Inst.getOperand(0).isReg()) {
    MCRegister DestReg = Inst.getOperand(0).getReg();
    if (Desc.getNumDefs() > 0 &&
        getX86MCRegisterClass(X86::GR64RegClassID).contains(DestReg) &&
        !readsRegister(Inst, Desc, DestReg, *RegInfo))
      TPDest = DestReg;
  }

  if (TPDest == LFIScratchReg &&
      readsRegister(Inst, Desc, LFIScratchReg, *RegInfo))
    return error(Inst, "%fs access reads reserved register %r11");

  emitTPLoad(TPDest, Out, STI);

  // Both slots occupied: the compute base via lea. For example:
  //
  // movq %fs:8(%rdi,%rsi,2), %rax
  // ->
  // movq 16(%r15), %rax
  // leaq (%rax,%rdi), %rax
  // movq 8(%rax,%rsi,2), %rax
  if (HasBase && HasIndex) {
    Out.emitInstruction(MCInstBuilder(X86::LEA64r)
                            .addReg(TPDest)
                            .addReg(TPDest)
                            .addImm(1)
                            .addReg(BaseReg)
                            .addImm(0)
                            .addReg(X86::NoRegister),
                        STI);
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

  if (isReturn(Inst))
    return rewriteReturn(Inst, Out, STI);

  if (isDirectCall(Inst))
    return rewriteDirectCall(Inst, Out, STI);

  if (isIndirectBranch(Inst) || isCall(Inst)) {
    if (!isSupportedIndirectBranch(Inst))
      return error(Inst, "unsupported indirect branch");
    return rewriteIndirectBranch(Inst, Out, STI);
  }

  if (isFSAccess(Inst))
    return rewriteFSAccess(Inst, Out, STI);

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
