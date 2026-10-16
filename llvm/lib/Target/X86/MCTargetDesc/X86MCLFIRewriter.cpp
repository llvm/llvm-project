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
#include <algorithm>
#include <optional>

using namespace llvm;

// LFI reserved registers.
static constexpr MCRegister LFIBaseReg = X86::R14;
static constexpr MCRegister LFIScratchReg = X86::R11;
static constexpr MCRegister LFITPReg = X86::R15;

// Segment register whose base always holds the sandbox base address.
static constexpr MCRegister LFIBaseSeg = X86::GS;

// Byte offset into the context register file (pointed to by R15) where the
// thread pointer is stored.
static constexpr int TPOffset = 16;

static bool isPrefix(const MCInst &Inst, const MCInstrInfo &InstInfo) {
  return (InstInfo.get(Inst.getOpcode()).TSFlags & X86II::FormMask) ==
         X86II::PrefixByte;
}

static bool isSyscall(const MCInst &Inst) {
  return Inst.getOpcode() == X86::SYSCALL;
}

static bool isDirectCall(const MCInst &Inst) {
  return Inst.getOpcode() == X86::CALL64pcrel32;
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

static bool writesSegmentBase(unsigned Opcode) {
  switch (Opcode) {
  case X86::WRFSBASE:
  case X86::WRFSBASE64:
  case X86::WRGSBASE:
  case X86::WRGSBASE64:
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

MCRegister X86::X86MCLFIRewriter::getReg32(MCRegister Reg) const {
  if (getX86MCRegisterClass(X86::GR32RegClassID).contains(Reg))
    return Reg;
  return RegInfo->getSubReg(Reg, X86::sub_32bit);
}

void X86::X86MCLFIRewriter::emitWithPrefixes(const MCInst &Inst,
                                             MCStreamer &Out,
                                             const MCSubtargetInfo &STI) {
  for (const MCInst &Prefix : Prefixes)
    Out.emitInstruction(Prefix, STI);
  Prefixes.clear();
  Out.emitInstruction(Inst, STI);
}

void X86::X86MCLFIRewriter::discardPrefixes() {
  for (const MCInst &Prefix : Prefixes)
    error(Prefix, "unsupported instruction prefix");
  Prefixes.clear();
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
// movq %gs:(%edi), %r11
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
    Mov.setLoc(Inst.getLoc());
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
  emitWithPrefixes(Inst, Out, STI);
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
//
// A return with an immediate additionally pops the immediate off the stack
// after loading the return address.
//
// retq $16
// ->
// popq %r11
// addq $16, %rsp
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
                      .addOperand(Inst.getOperand(0))
                      .setLoc(Inst.getLoc()),
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
// at TPOffset(%r15). The rewritten access is fed back through the rewriter so
// that it is sandboxed like other memory accesses. Example rewrites:
//
// movq %fs:0, %rax
// ->
// movq 16(%r15), %rax
//
// movq %fs:(%rdi), %rax
// ->
// movq 16(%r15), %rax
// movq %gs:(%eax,%edi), %rax
//
// movq %fs:8(%rdi, %rsi, 2), %rax
// ->
// movq 16(%r15), %rax
// leaq (%rax, %rdi), %rax
// movq %gs:8(%eax, %esi, 2), %rax
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

  // %fs:0 -> TPOffset(%r15). The context register always points into the
  // runtime's thread-local block, so this access is safe.
  if (!HasBase && !HasIndex && !HasDisp) {
    MCInst Modified(Inst);
    Modified.getOperand(MemIdx + X86::AddrBaseReg).setReg(LFITPReg);
    Modified.getOperand(MemIdx + X86::AddrDisp).setImm(TPOffset);
    Modified.getOperand(MemIdx + X86::AddrSegmentReg).setReg(X86::NoRegister);
    return emitWithPrefixes(Modified, Out, STI);
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
    if (Desc.getNumDefs() > 0 && DestReg != X86::RSP &&
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
  // movq %gs:8(%eax,%esi,2), %rax
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
  // movq %gs:(%eax,%edi), %rax
  MCInst Modified(Inst);
  Modified.getOperand(MemIdx + X86::AddrBaseReg).setReg(TPDest);
  if (HasBase && !HasIndex)
    Modified.getOperand(MemIdx + X86::AddrIndexReg).setReg(BaseReg);
  Modified.getOperand(MemIdx + X86::AddrSegmentReg).setReg(X86::NoRegister);
  doRewriteInst(Modified, Out, STI);
}

// Return true if Reg always contains an address within the sandbox.
static bool isSandboxBaseReg(MCRegister Reg) {
  return Reg == LFIBaseReg || Reg == X86::RSP || Reg == X86::RIP;
}

// Return true if Inst reads or writes memory through an implicit pointer
// register instead of an addressing mode, and so cannot be sandboxed by
// rewriting its operands.
static bool hasImplicitMemPointer(unsigned Opcode) {
  switch (Opcode) {
  // xlat loads from (%rbx,%al).
  case X86::XLAT:
  // maskmov stores to (%rdi).
  case X86::MASKMOVDQU:
  case X86::MASKMOVDQU64:
  case X86::VMASKMOVDQU:
  case X86::VMASKMOVDQU64:
  case X86::MMX_MASKMOVQ:
  case X86::MMX_MASKMOVQ64:
    return true;
  default:
    return false;
  }
}

// Return true if Inst has a memory operand that is not dereferenced.
static bool ignoresMemOperand(unsigned Opcode) {
  switch (Opcode) {
  case X86::LEA16r:
  case X86::LEA32r:
  case X86::LEA64r:
  case X86::LEA64_32r:
  case X86::NOOPW:
  case X86::NOOPL:
  case X86::NOOPQ:
    return true;
  default:
    return false;
  }
}

// Return true if the index register of the memory operand at MemIdx is a
// general-purpose register, and so can be truncated along with the base.
static bool hasGPRIndex(const MCInst &Inst, int MemIdx) {
  MCRegister Index = Inst.getOperand(MemIdx + X86::AddrIndexReg).getReg();
  return Index && (getX86MCRegisterClass(X86::GR64RegClassID).contains(Index) ||
                   getX86MCRegisterClass(X86::GR32RegClassID).contains(Index));
}

// Rewrite the addressing mode of the memory operand at MemIdx so that the
// access it performs stays inside the sandbox.
//
// movq 8(%rax,%rcx,4), %rdi
// ->
// movq %gs:8(%eax,%ecx,4), %rdi
void X86::X86MCLFIRewriter::sandboxMemOperand(MCInst &Inst, int MemIdx) const {
  MCOperand &Base = Inst.getOperand(MemIdx + X86::AddrBaseReg);
  MCOperand &Index = Inst.getOperand(MemIdx + X86::AddrIndexReg);
  MCOperand &Segment = Inst.getOperand(MemIdx + X86::AddrSegmentReg);

  // A base register that always points into the sandbox needs no rewriting, as
  // long as there is no index register.
  if (isSandboxBaseReg(Base.getReg()) && !Index.getReg())
    return;

  // An absolute address has no register to truncate, so make it relative to
  // the sandbox base instead.
  if (!Base.getReg() && !Index.getReg()) {
    Base.setReg(LFIBaseReg);
    return;
  }

  // If there is no Base and the Index is not a GPR, then we use r14d (all
  // zeroes) as the base for the rewritten addressing mode.
  bool GPRIndex = hasGPRIndex(Inst, MemIdx);

  Segment.setReg(LFIBaseSeg);
  if (Base.getReg())
    Base.setReg(getReg32(Base.getReg()));
  else if (!GPRIndex)
    Base.setReg(getReg32(LFIBaseReg));
  if (GPRIndex)
    Index.setReg(getReg32(Index.getReg()));
}

void X86::X86MCLFIRewriter::sandboxMemAccess(MCInst &Inst) const {
  if (ignoresMemOperand(Inst.getOpcode()))
    return;

  int MemIdx = X86II::getMemoryOperandIdx(InstInfo->get(Inst.getOpcode()));
  if (MemIdx >= 0)
    sandboxMemOperand(Inst, MemIdx);
}

void X86::X86MCLFIRewriter::rewriteMemAccess(const MCInst &Inst,
                                             MCStreamer &Out,
                                             const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());

  // The moffs form of mov encodes an absolute address in the instruction
  // instead of an addressing mode, so it is unsupported.
  if ((Desc.TSFlags & X86II::FormMask) == X86II::RawFrmMemOffs ||
      hasImplicitMemPointer(Inst.getOpcode()))
    return error(Inst, "unsupported memory access");

  MCInst Sandboxed(Inst);
  sandboxMemAccess(Sandboxed);
  emitWithPrefixes(Sandboxed, Out, STI);
}

namespace {
struct StringOpPointers {
  bool UsesDest;
  bool UsesSrc;
};
} // namespace

static std::optional<StringOpPointers> getStringOpPointers(unsigned Opcode) {
  switch (Opcode) {
  case X86::STOSB:
  case X86::STOSW:
  case X86::STOSL:
  case X86::STOSQ:
  case X86::REP_STOSB_64:
  case X86::REP_STOSW_64:
  case X86::REP_STOSD_64:
  case X86::REP_STOSQ_64:
  case X86::SCASB:
  case X86::SCASW:
  case X86::SCASL:
  case X86::SCASQ:
    return StringOpPointers{true, false};
  case X86::MOVSB:
  case X86::MOVSW:
  case X86::MOVSL:
  case X86::MOVSQ:
  case X86::REP_MOVSB_64:
  case X86::REP_MOVSW_64:
  case X86::REP_MOVSD_64:
  case X86::REP_MOVSQ_64:
  case X86::CMPSB:
  case X86::CMPSW:
  case X86::CMPSL:
  case X86::CMPSQ:
    return StringOpPointers{true, true};
  case X86::LODSB:
  case X86::LODSW:
  case X86::LODSL:
  case X86::LODSQ:
    return StringOpPointers{false, true};
  default:
    return std::nullopt;
  }
}

static void emitStringPtrFixup(MCRegister Reg, MCRegister Reg32,
                               MCStreamer &Out, const MCSubtargetInfo &STI) {
  Out.emitInstruction(MCInstBuilder(X86::MOV32rr).addReg(Reg32).addReg(Reg32),
                      STI);

  Out.emitInstruction(MCInstBuilder(X86::LEA64r)
                          .addReg(Reg)
                          .addReg(LFIBaseReg)
                          .addImm(1)
                          .addReg(Reg)
                          .addImm(0)
                          .addReg(X86::NoRegister),
                      STI);
}

// rep movsq
// ->
// .bundle_lock
// movl %edi, %edi
// leaq (%r14,%rdi), %rdi
// movl %esi, %esi
// leaq (%r14,%rsi), %rsi
// rep movsq
// .bundle_unlock
void X86::X86MCLFIRewriter::rewriteStringOperation(const MCInst &Inst,
                                                   MCStreamer &Out,
                                                   const MCSubtargetInfo &STI) {
  std::optional<StringOpPointers> Ptrs = getStringOpPointers(Inst.getOpcode());
  assert(Ptrs);

  Out.emitBundleLock(/*AlignToEnd=*/false, STI);

  if (Ptrs->UsesDest)
    emitStringPtrFixup(X86::RDI, X86::EDI, Out, STI);
  if (Ptrs->UsesSrc)
    emitStringPtrFixup(X86::RSI, X86::ESI, Out, STI);

  emitWithPrefixes(Inst, Out, STI);

  Out.emitBundleUnlock(STI);
}

// The register form of an instruction that modifies the stack and reads its
// source from memory, or 0 if there is none.
static unsigned memToRegOpcode(unsigned Opcode) {
  switch (Opcode) {
  case X86::MOV64rm:
    return X86::MOV64rr;
  case X86::ADD64rm:
    return X86::ADD64rr;
  case X86::SUB64rm:
    return X86::SUB64rr;
  case X86::AND64rm:
    return X86::AND64rr;
  case X86::OR64rm:
    return X86::OR64rr;
  case X86::XOR64rm:
    return X86::XOR64rr;
  default:
    return 0;
  }
}

// The 32-bit form of a 64-bit instruction, or 0 if the instruction has no
// 32-bit form that can be used to modify the stack pointer.
static unsigned demote64BitOpcode(unsigned Opcode) {
  switch (Opcode) {
  case X86::MOV64rr:
    return X86::MOV32rr;
  case X86::MOV64ri32:
    return X86::MOV32ri;
  case X86::LEA64r:
    return X86::LEA64_32r;
  case X86::ADD64rr:
    return X86::ADD32rr;
  case X86::ADD64ri8:
    return X86::ADD32ri8;
  case X86::ADD64ri32:
    return X86::ADD32ri;
  case X86::SUB64rr:
    return X86::SUB32rr;
  case X86::SUB64ri8:
    return X86::SUB32ri8;
  case X86::SUB64ri32:
    return X86::SUB32ri;
  case X86::AND64rr:
    return X86::AND32rr;
  case X86::AND64ri8:
    return X86::AND32ri8;
  case X86::AND64ri32:
    return X86::AND32ri;
  case X86::OR64rr:
    return X86::OR32rr;
  case X86::OR64ri8:
    return X86::OR32ri8;
  case X86::OR64ri32:
    return X86::OR32ri;
  case X86::XOR64rr:
    return X86::XOR32rr;
  case X86::XOR64ri8:
    return X86::XOR32ri8;
  case X86::XOR64ri32:
    return X86::XOR32ri;
  default:
    return 0;
  }
}

MCRegister X86::X86MCLFIRewriter::getWrittenStackReg(const MCInst &Inst) const {
  const MCInstrDesc &Desc = InstInfo->get(Inst.getOpcode());
  for (unsigned I = 0, E = Desc.getNumDefs(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && RegInfo->regsOverlap(Op.getReg(), X86::RSP))
      return Op.getReg();
  }
  return X86::NoRegister;
}

// Emit leaq (%rsp,%r14), %rsp
static void emitStackFixup(MCStreamer &Out, const MCSubtargetInfo &STI) {
  Out.emitInstruction(MCInstBuilder(X86::LEA64r)
                          .addReg(X86::RSP)
                          .addReg(X86::RSP)
                          .addImm(1)
                          .addReg(LFIBaseReg)
                          .addImm(0)
                          .addReg(X86::NoRegister),
                      STI);
}

// addq $8, %rsp
// ->
// .bundle_lock
// addl $8, %esp
// leaq (%rsp,%r14), %rsp
// .bundle_unlock
void X86::X86MCLFIRewriter::rewriteStackModification(
    MCRegister StackReg, const MCInst &Inst, MCStreamer &Out,
    const MCSubtargetInfo &STI) {
  if (StackReg != X86::RSP && StackReg != X86::ESP)
    return error(Inst, "unsupported modification of the stack pointer");

  // popq %rsp loads through the stack pointer that it overwrites, so it cannot
  // simply be demoted. Pop into the scratch register instead, and move the
  // result into %esp.
  if (Inst.getOpcode() == X86::POP64r) {
    Out.emitInstruction(MCInstBuilder(X86::POP64r).addReg(LFIScratchReg), STI);

    Out.emitBundleLock(/*AlignToEnd=*/false, STI);
    Out.emitInstruction(
        MCInstBuilder(X86::MOV32rr).addReg(X86::ESP).addReg(X86::R11D), STI);
    emitStackFixup(Out, STI);
    Out.emitBundleUnlock(STI);
    return;
  }

  // movq 8(%rdi), %rsp
  // ->
  // movq %gs:8(%edi), %r11
  // .bundle_lock
  // movl %r11d, %esp
  // leaq (%rsp,%r14), %rsp
  // .bundle_unlock
  int MemIdx = X86II::getMemoryOperandIdx(InstInfo->get(Inst.getOpcode()));
  if (MemIdx >= 0 && !ignoresMemOperand(Inst.getOpcode())) {
    unsigned RegOpcode = memToRegOpcode(Inst.getOpcode());
    if (!RegOpcode)
      return error(Inst, "unsupported modification of the stack pointer");

    MCInstBuilder Load(X86::MOV64rm);
    Load.addReg(LFIScratchReg);
    for (unsigned I = 0; I < X86::AddrNumOperands; ++I)
      Load.addOperand(Inst.getOperand(MemIdx + I));
    Load.setLoc(Inst.getLoc());
    doRewriteInst(Load, Out, STI);

    MCInstBuilder RegForm(RegOpcode);
    for (int I = 0; I < MemIdx; ++I)
      RegForm.addOperand(Inst.getOperand(I));
    RegForm.addReg(LFIScratchReg);
    RegForm.setLoc(Inst.getLoc());
    return doRewriteInst(RegForm, Out, STI);
  }

  MCInst Modified(Inst);
  if (StackReg == X86::RSP) {
    unsigned Opcode = demote64BitOpcode(Inst.getOpcode());
    if (!Opcode)
      return error(Inst, "unsupported modification of the stack pointer");
    Modified.setOpcode(Opcode);

    ArrayRef<MCOperandInfo> OpInfo = InstInfo->get(Opcode).operands();
    for (unsigned I = 0, E = std::min<unsigned>(OpInfo.size(),
                                                Modified.getNumOperands());
         I < E; ++I) {
      MCOperand &Op = Modified.getOperand(I);
      if (OpInfo[I].OperandType != MCOI::OPERAND_REGISTER || !Op.isReg())
        continue;
      if (getX86MCRegisterClass(X86::GR64RegClassID).contains(Op.getReg()))
        Op.setReg(RegInfo->getSubReg(Op.getReg(), X86::sub_32bit));
    }
  }

  sandboxMemAccess(Modified);

  Out.emitBundleLock(/*AlignToEnd=*/false, STI);
  emitWithPrefixes(Modified, Out, STI);
  emitStackFixup(Out, STI);
  Out.emitBundleUnlock(STI);
}

// leave
// ->
// .bundle_lock
// movl %ebp, %esp
// leaq (%rsp,%r14), %rsp
// .bundle_unlock
// popq %rbp
void X86::X86MCLFIRewriter::rewriteLeave(const MCInst &Inst, MCStreamer &Out,
                                         const MCSubtargetInfo &STI) {
  doRewriteInst(MCInstBuilder(X86::MOV64rr)
                    .addReg(X86::RSP)
                    .addReg(X86::RBP)
                    .setLoc(Inst.getLoc()),
                Out, STI);
  Out.emitInstruction(MCInstBuilder(X86::POP64r).addReg(X86::RBP), STI);
}

void X86::X86MCLFIRewriter::doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                                          const MCSubtargetInfo &STI) {
  if (!STI.hasFeature(X86::Is64Bit))
    return error(Inst, "LFI only supports 64-bit mode");

  // Hold a stand-alone prefix back, so it can be applied to the rewritten
  // instruction.
  if (isPrefix(Inst, *InstInfo)) {
    Prefixes.push_back(Inst);
    return;
  }

  if (mayModifyRegister(Inst, LFIBaseReg) ||
      mayModifyRegister(Inst, LFITPReg) || mayModifyRegister(Inst, LFIBaseSeg))
    return error(Inst, "illegal modification of reserved LFI register");

  if (writesSegmentBase(Inst.getOpcode()))
    return error(Inst, "illegal modification of reserved segment base");

  if (isSyscall(Inst))
    return rewriteSyscall(Inst, Out, STI);

  if (isReturn(Inst))
    return rewriteReturn(Inst, Out, STI);

  if (isDirectCall(Inst))
    return rewriteDirectCall(Inst, Out, STI);

  // jmpabs is disallowed since it jumps to an absolute address.
  if (Inst.getOpcode() == X86::JMPABS64i)
    return error(Inst, "unsupported branch instruction");

  if (isIndirectBranch(Inst) || isCall(Inst)) {
    if (!isSupportedIndirectBranch(Inst))
      return error(Inst, "unsupported indirect branch");
    return rewriteIndirectBranch(Inst, Out, STI);
  }

  for (const MCOperand &Op : Inst)
    if (Op.isReg() && Op.getReg() == LFIBaseSeg)
      return error(Inst, "invalid use of reserved segment register %gs");

  if (isFSAccess(Inst))
    return rewriteFSAccess(Inst, Out, STI);

  if (getStringOpPointers(Inst.getOpcode()))
    return rewriteStringOperation(Inst, Out, STI);

  if (Inst.getOpcode() == X86::LEAVE64)
    return rewriteLeave(Inst, Out, STI);

  if (MCRegister StackReg = getWrittenStackReg(Inst))
    return rewriteStackModification(StackReg, Inst, Out, STI);

  rewriteMemAccess(Inst, Out, STI);
}

bool X86::X86MCLFIRewriter::rewriteInst(const MCInst &Inst, MCStreamer &Out,
                                        const MCSubtargetInfo &STI) {
  // The guard prevents rewrite-recursion when we emit instructions from inside
  // the rewriter (such instructions should not be rewritten).
  if (!Enabled || Guard)
    return false;
  Guard = true;

  size_t PrefixesBefore = Prefixes.size();

  doRewriteInst(Inst, Out, STI);

  if (Prefixes.size() == PrefixesBefore)
    discardPrefixes();

  Guard = false;
  return true;
}

void X86::X86MCLFIRewriter::onLabel(const MCSymbol *Symbol, MCStreamer &Out) {
  discardPrefixes();
}

void X86::X86MCLFIRewriter::finish(MCStreamer &Out) { discardPrefixes(); }
