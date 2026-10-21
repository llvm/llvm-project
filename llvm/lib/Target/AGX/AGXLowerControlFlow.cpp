//===-- AGXLowerControlFlow.cpp - Lower structured control flow -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expands the structured control flow pseudos into execution mask
// instructions on the nesting counter r0l and jumps (like SILowerControlFlow):
//
//   AGX_IF c, %flow     if_icmp r0l, ne, c, 0, 1;   jmp_exec_none %flow
//   AGX_ELSE %end       else_fcmp r0l, eq, 0.0, 0.0, 1; jmp_exec_none %end
//   AGX_LOOP brk, %hdr  while_icmp r0l, eq, brk, 0, 1;  jmp_exec_any %hdr
//   AGX_LOOP_BEGIN      if_fcmp r0l, eq, 0.0, 0.0, 1 (push_exec 1)
//   AGX_END_CF          pop_exec r0l, 1, at the start of the block
//   AGX_BR %bb          jmp_exec_any %bb; jmp_exec_none %bb
//
// The jumps only skip code no thread executes; the execution mask
// instructions alone implement the semantics. The nesting counter is zeroed
// at kernel entry. A jump_exec_none over a short run of ALU instructions is
// left out: on G14C an ALU instruction issued with no active thread costs
// about as little as a taken jump over it, while memory instructions still
// cost a large part of their latency (AGX Performance Simulator
// tasks/ptx-re-blockers.md, B31).
//
// Shuffles read the index of every thread of a quad, including inactive
// ones, whose register may hold anything. Register indices are therefore
// passed in r0h (like Mesa's agx_lower_divergent_shuffle), which is zero at
// entry and set back to zero after each shuffle, so the threads that are
// inactive at a shuffle read zero. A reset followed in its block by another
// shuffle index, with no change of the execution mask between them, is left
// out: the same threads are active at both shuffles.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-control-flow"

namespace {

class AGXLowerControlFlow : public MachineFunctionPass {
public:
  static char ID;
  AGXLowerControlFlow() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Control Flow";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char AGXLowerControlFlow::ID = 0;

INITIALIZE_PASS(AGXLowerControlFlow, DEBUG_TYPE, "AGX Lower Control Flow",
                false, false)

FunctionPass *llvm::createAGXLowerControlFlowPass() {
  return new AGXLowerControlFlow();
}

static cl::opt<unsigned> MaxUnskippedInstrs(
    "agx-max-unskipped-instrs",
    cl::desc("Most ALU instructions an if or else body may have and still be "
             "run with an empty execution mask instead of jumped over"),
    cl::init(4), cl::Hidden);

// Integer conditions.
static constexpr unsigned ICondUEQ = 0;
static constexpr unsigned ICondNE = 8;
// Float condition eq, and the 8-bit float immediate 0.0.
static constexpr unsigned FCondEQ = 0;
static constexpr unsigned FPImmZero = 0;

/// Build an execution mask instruction: (nest, cc, a, b, n).
static MachineInstrBuilder buildExecOp(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       const DebugLoc &DL,
                                       const AGXInstrInfo &TII, unsigned Opc,
                                       unsigned CC) {
  return BuildMI(MBB, I, DL, TII.get(Opc)).addImm(0).addImm(CC);
}

static bool isShuffle(unsigned Opc) {
  switch (Opc) {
  case AGX::SIMD_SHUFFLE_32:
  case AGX::SIMD_SHUFFLE_16:
  case AGX::SIMD_SHUFFLE_XOR_32:
  case AGX::SIMD_SHUFFLE_XOR_16:
  case AGX::SIMD_SHUFFLE_UP_32:
  case AGX::SIMD_SHUFFLE_UP_16:
  case AGX::SIMD_SHUFFLE_DOWN_32:
  case AGX::SIMD_SHUFFLE_DOWN_16:
  case AGX::QUAD_SHUFFLE_32:
  case AGX::QUAD_SHUFFLE_16:
  case AGX::QUAD_SHUFFLE_XOR_32:
  case AGX::QUAD_SHUFFLE_XOR_16:
  case AGX::QUAD_SHUFFLE_UP_32:
  case AGX::QUAD_SHUFFLE_UP_16:
  case AGX::QUAD_SHUFFLE_DOWN_32:
  case AGX::QUAD_SHUFFLE_DOWN_16:
    return true;
  default:
    return false;
  }
}

static bool isControlFlowPseudo(unsigned Opc) {
  switch (Opc) {
  case AGX::AGX_IF:
  case AGX::AGX_ELSE:
  case AGX::AGX_LOOP:
  case AGX::AGX_IF_ICMP:
  case AGX::AGX_IF_FCMP:
  case AGX::AGX_LOOP_ICMP:
  case AGX::AGX_LOOP_FCMP:
  case AGX::AGX_LOOP_BEGIN:
  case AGX::AGX_END_CF:
  case AGX::AGX_BR:
    return true;
  default:
    return false;
  }
}

/// Whether the blocks a jump_exec_none at the end of MBB would skip, up to
/// Target, are a straight run of at most MaxUnskippedInstrs ALU instructions,
/// so that running them with an empty execution mask is cheaper than the jump.
static bool isShortALURun(MachineBasicBlock &MBB, MachineBasicBlock *Target) {
  unsigned Count = 0;
  MachineFunction &MF = *MBB.getParent();
  for (MachineBasicBlock *S : MBB.successors())
    if (S != Target && !MBB.isLayoutSuccessor(S))
      return false;
  for (auto B = std::next(MBB.getIterator()); &*B != Target; ++B) {
    if (B == MF.end() || B->succ_size() != 1)
      return false;
    MachineBasicBlock *Next = *B->succ_begin();
    if (!B->isLayoutSuccessor(Next))
      return false;
    for (const MachineInstr &MI : *B) {
      if (MI.isMetaInstruction())
        continue;
      // A branch to the next block is left out.
      if (MI.getOpcode() == AGX::AGX_BR && MI.getOperand(0).getMBB() == Next)
        continue;
      if (isControlFlowPseudo(MI.getOpcode()) || MI.mayLoadOrStore() ||
          MI.hasUnmodeledSideEffects() || MI.isCall() || MI.isTerminator() ||
          AGXInstrInfo::isExecMaskOp(MI))
        return false;
      // A shuffle also moves its index to r0h and resets it.
      Count += isShuffle(MI.getOpcode()) ? 3 : 1;
      if (Count > MaxUnskippedInstrs)
        return false;
    }
  }
  return true;
}

/// Jump over the code between MBB and Target when no thread runs it, unless
/// that code is cheap enough to run with an empty mask. Without the jump, the
/// edge to Target is removed: MBB falls through.
static void buildSkip(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      const DebugLoc &DL, const AGXInstrInfo &TII,
                      MachineBasicBlock *Target) {
  if (!isShortALURun(MBB, Target)) {
    BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(Target);
    return;
  }
  if (!MBB.isLayoutSuccessor(Target) && MBB.isSuccessor(Target))
    MBB.removeSuccessor(Target);
}

/// Pass the register index of a shuffle in r0h. Returns true if MI was
/// changed.
static bool lowerShuffleIndex(MachineInstr &MI, const AGXInstrInfo &TII) {
  // (dst, src0_mods, src0, src1_mods, src1)
  MachineOperand &Index = MI.getOperand(4);
  if (!Index.isReg() || !AGX::GPR16RegClass.contains(Index.getReg()))
    return false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  TII.copyPhysReg(MBB, MI, DL, AGX::R0H, Index.getReg(), Index.isKill());
  Index.setReg(AGX::R0H);
  Index.setIsKill();
  BuildMI(MBB, std::next(MI.getIterator()), DL, TII.get(AGX::MOV_IMM16),
          AGX::R0H)
      .addImm(0);
  return true;
}

/// Remove a threadgroup barrier that follows another one in its block with
/// nothing between them that touches memory, has side effects or changes the
/// execution mask: the second one orders nothing more.
static bool removeRepeatedBarriers(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool AfterBarrier = false;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isMetaInstruction())
        continue;
      if (MI.getOpcode() == AGX::THREADGROUP_BARRIER) {
        if (AfterBarrier) {
          MI.eraseFromParent();
          Changed = true;
        }
        AfterBarrier = true;
        continue;
      }
      if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall() ||
          MI.isTerminator() || AGXInstrInfo::isExecMaskOp(MI))
        AfterBarrier = false;
    }
  }
  return Changed;
}

/// Remove the r0h resets that the next shuffle index of their block
/// overwrites before the execution mask can change.
static void removeRedundantIndexResets(MachineFunction &MF) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (MachineBasicBlock &MBB : MF) {
    MachineInstr *Reset = nullptr;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      bool IsReset = MI.getOpcode() == AGX::MOV_IMM16 &&
                     MI.getOperand(0).getReg() == AGX::R0H &&
                     MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0;
      if (IsReset) {
        Reset = &MI;
        continue;
      }
      if (!Reset)
        continue;
      if (AGXInstrInfo::isExecMaskOp(MI) || MI.isTerminator() || MI.isCall() ||
          MI.modifiesRegister(AGX::R0L, TRI)) {
        Reset = nullptr;
        continue;
      }
      if (MI.readsRegister(AGX::R0H, TRI)) {
        Reset = nullptr;
        continue;
      }
      if (MI.modifiesRegister(AGX::R0H, TRI)) {
        Reset->eraseFromParent();
        Reset = nullptr;
      }
    }
  }
}

bool AGXLowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const AGXInstrInfo &TII = *MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  bool HasControlFlow = false;
  bool HasShuffleIndex = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      MachineBasicBlock::iterator I = MI.getIterator();
      const DebugLoc &DL = MI.getDebugLoc();
      switch (MI.getOpcode()) {
      case AGX::AGX_IF: {
        buildExecOp(MBB, I, DL, TII, AGX::IF_ICMP, ICondNE)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        buildSkip(MBB, I, DL, TII, MI.getOperand(1).getMBB());
        break;
      }
      case AGX::AGX_ELSE:
        buildExecOp(MBB, I, DL, TII, AGX::ELSE_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        buildSkip(MBB, I, DL, TII, MI.getOperand(0).getMBB());
        break;
      case AGX::AGX_LOOP:
        // Threads stay in the loop while brk is zero.
        buildExecOp(MBB, I, DL, TII, AGX::WHILE_ICMP, ICondUEQ)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY))
            .addMBB(MI.getOperand(1).getMBB());
        break;
      case AGX::AGX_IF_ICMP:
      case AGX::AGX_IF_FCMP:
      case AGX::AGX_LOOP_ICMP:
      case AGX::AGX_LOOP_FCMP: {
        // (cc, src0_mods, src0, src1_mods, src1, target). An if keeps the
        // threads where the comparison holds; a loop keeps those where it
        // does not (the comparison is the break condition).
        unsigned Opc = MI.getOpcode();
        bool Loop = Opc == AGX::AGX_LOOP_ICMP || Opc == AGX::AGX_LOOP_FCMP;
        bool Float = Opc == AGX::AGX_IF_FCMP || Opc == AGX::AGX_LOOP_FCMP;
        unsigned CC = MI.getOperand(0).getImm() ^ (Loop ? 8 : 0);
        unsigned Exec = Loop ? (Float ? AGX::WHILE_FCMP : AGX::WHILE_ICMP)
                             : (Float ? AGX::IF_FCMP : AGX::IF_ICMP);
        buildExecOp(MBB, I, DL, TII, Exec, CC)
            .add(MI.getOperand(1))
            .add(MI.getOperand(2))
            .add(MI.getOperand(3))
            .add(MI.getOperand(4))
            .addImm(1);
        if (Loop)
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY))
              .addMBB(MI.getOperand(5).getMBB());
        else
          buildSkip(MBB, I, DL, TII, MI.getOperand(5).getMBB());
        break;
      }
      case AGX::AGX_LOOP_BEGIN:
        buildExecOp(MBB, I, DL, TII, AGX::IF_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        break;
      case AGX::AGX_END_CF: {
        // The construct reconverges at the start of the block. Scheduling
        // may have moved instructions of the block above the pseudo, since
        // they have no dependency on it: the pop goes first (like
        // SILowerControlFlow::emitEndCf), after any earlier pops.
        MachineBasicBlock::iterator Pos = MBB.begin();
        while (Pos != I && Pos->getOpcode() == AGX::POP_EXEC)
          ++Pos;
        BuildMI(MBB, Pos, DL, TII.get(AGX::POP_EXEC)).addImm(0).addImm(1);
        break;
      }
      case AGX::AGX_BR: {
        MachineBasicBlock *Target = MI.getOperand(0).getMBB();
        if (!MBB.isLayoutSuccessor(Target)) {
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY)).addMBB(Target);
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(Target);
        }
        MI.eraseFromParent();
        continue;
      }
      default:
        if (isShuffle(MI.getOpcode()))
          HasShuffleIndex |= lowerShuffleIndex(MI, TII);
        continue;
      }
      HasControlFlow = true;
      MI.eraseFromParent();
    }
  }

  if (HasShuffleIndex)
    removeRedundantIndexResets(MF);
  removeRepeatedBarriers(MF);

  MachineBasicBlock &Entry = MF.front();
  if (HasShuffleIndex)
    BuildMI(Entry, Entry.begin(), DebugLoc(), TII.get(AGX::MOV_IMM16),
            AGX::R0H)
        .addImm(0);
  if (HasControlFlow) {
    BuildMI(Entry, Entry.begin(), DebugLoc(), TII.get(AGX::SET_NEST))
        .addImm(0);
  }
  return true;
}
