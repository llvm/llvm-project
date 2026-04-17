//===----- X86DynAllocaExpander.cpp - Expand DynAlloca pseudo instruction -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that expands DynAlloca pseudo-instructions.
//
// It performs a conservative analysis to determine whether each allocation
// falls within a region of the stack that is safe to use, or whether stack
// probes must be emitted.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace {

class X86DynAllocaExpander {
public:
  bool run(MachineFunction &MF);

private:
  /// Strategies for lowering a DynAlloca.
  enum Lowering { TouchAndSub, Sub, Probe };

  /// Deterministic-order map from DynAlloca instruction to desired lowering.
  typedef MapVector<MachineInstr*, Lowering> LoweringMap;

  /// Compute which lowering to use for each DynAlloca instruction.
  void computeLowerings(MachineFunction &MF, LoweringMap& Lowerings);

  /// Get the appropriate lowering based on current offset and amount.
  Lowering getLowering(int64_t CurrentOffset, int64_t AllocaAmount);

  Register materializeWinAllocaAmount(MachineInstr *MI,
                                      MachineBasicBlock::iterator I,
                                      const DebugLoc &DL);

  /// Lower a DynAlloca instruction.
  void lower(MachineInstr* MI, Lowering L);

  MachineRegisterInfo *MRI = nullptr;
  const X86Subtarget *STI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const X86RegisterInfo *TRI = nullptr;
  Register StackPtr;
  unsigned SlotSize = 0;
  int64_t StackProbeSize = 0;
  bool NoStackArgProbe = false;
};

class X86DynAllocaExpanderLegacy : public MachineFunctionPass {
public:
  X86DynAllocaExpanderLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  StringRef getPassName() const override { return "X86 DynAlloca Expander"; }

public:
  static char ID;
};

char X86DynAllocaExpanderLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(X86DynAllocaExpanderLegacy, "x86-dyn-alloca-expander",
                "X86 DynAlloca Expander", false, false)

FunctionPass *llvm::createX86DynAllocaExpanderLegacyPass() {
  return new X86DynAllocaExpanderLegacy();
}

/// Return the allocation amount for a DynAlloca instruction, or -1 if unknown.
static bool isDynAllocaOpcode(unsigned Opc) {
  return Opc == X86::DYN_ALLOCA_32 || Opc == X86::DYN_ALLOCA_64 ||
         Opc == X86::WIN_ALLOCA_64;
}

static unsigned getDynAllocaAmountOperandIndex(const MachineInstr *MI) {
  return MI->getOpcode() == X86::WIN_ALLOCA_64 ? 1 : 0;
}

static int64_t getConstantRegValue(Register Reg, MachineRegisterInfo *MRI) {
  MachineInstr *Def = MRI->getUniqueVRegDef(Reg);

  if (!Def ||
      (Def->getOpcode() != X86::MOV32ri && Def->getOpcode() != X86::MOV64ri) ||
      !Def->getOperand(1).isImm())
    return -1;

  return Def->getOperand(1).getImm();
}

static int64_t getWinAllocaAmount(MachineInstr *MI, MachineRegisterInfo *MRI,
                                  unsigned StackAlign) {
  assert(MI->getOpcode() == X86::WIN_ALLOCA_64);
  assert(MI->getOperand(1).isReg());

  int64_t RawAmount = getConstantRegValue(MI->getOperand(1).getReg(), MRI);
  if (RawAmount < 0)
    return -1;

  uint64_t Amount = static_cast<uint64_t>(RawAmount);
  unsigned Alignment = MI->getOperand(2).getImm();
  if (Alignment > StackAlign)
    Amount += Alignment - 1;
  Amount = alignTo(Amount, StackAlign);
  if (Amount > static_cast<uint64_t>(INT64_MAX))
    return -1;
  return static_cast<int64_t>(Amount);
}

static int64_t getDynAllocaAmount(MachineInstr *MI, MachineRegisterInfo *MRI,
                                  unsigned StackAlign = 0) {
  assert(isDynAllocaOpcode(MI->getOpcode()));
  if (MI->getOpcode() == X86::WIN_ALLOCA_64)
    return getWinAllocaAmount(MI, MRI, StackAlign);

  unsigned AmountOperandIndex = getDynAllocaAmountOperandIndex(MI);
  assert(MI->getOperand(AmountOperandIndex).isReg());
  return getConstantRegValue(MI->getOperand(AmountOperandIndex).getReg(), MRI);
}

X86DynAllocaExpander::Lowering
X86DynAllocaExpander::getLowering(int64_t CurrentOffset,
                                  int64_t AllocaAmount) {
  // For a non-constant amount or a large amount, we have to probe.
  if (AllocaAmount < 0 || AllocaAmount > StackProbeSize)
    return Probe;

  // If it fits within the safe region of the stack, just subtract.
  if (CurrentOffset + AllocaAmount <= StackProbeSize)
    return Sub;

  // Otherwise, touch the current tip of the stack, then subtract.
  return TouchAndSub;
}

static bool isPushPop(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32r:
  case X86::PUSH32rmm:
  case X86::PUSH32rmr:
  case X86::PUSH32i:
  case X86::PUSH64r:
  case X86::PUSH64rmm:
  case X86::PUSH64rmr:
  case X86::PUSH64i32:
  case X86::POP32r:
  case X86::POP64r:
    return true;
  default:
    return false;
  }
}

void X86DynAllocaExpander::computeLowerings(MachineFunction &MF,
                                            LoweringMap &Lowerings) {
  // Do a one-pass reverse post-order walk of the CFG to conservatively estimate
  // the offset between the stack pointer and the lowest touched part of the
  // stack, and use that to decide how to lower each DynAlloca instruction.

  // Initialize OutOffset[B], the stack offset at exit from B, to something big.
  DenseMap<MachineBasicBlock *, int64_t> OutOffset;
  for (MachineBasicBlock &MBB : MF)
    OutOffset[&MBB] = INT32_MAX;

  // Note: we don't know the offset at the start of the entry block since the
  // prologue hasn't been inserted yet, and how much that will adjust the stack
  // pointer depends on register spills, which have not been computed yet.

  // Compute the reverse post-order.
  ReversePostOrderTraversal<MachineFunction*> RPO(&MF);
  bool HasStableWin64MSVCCallFrame =
      MF.getInfo<X86MachineFunctionInfo>()->hasWin64MSVCDynAllocaCallFrame();

  for (MachineBasicBlock *MBB : RPO) {
    int64_t Offset = -1;
    for (MachineBasicBlock *Pred : MBB->predecessors())
      Offset = std::max(Offset, OutOffset[Pred]);
    if (Offset == -1) Offset = INT32_MAX;

    for (MachineInstr &MI : *MBB) {
      if (isDynAllocaOpcode(MI.getOpcode())) {
        // A DynAlloca moves StackPtr, and potentially touches it.
        int64_t Amount = getDynAllocaAmount(
            &MI, MRI, STI->getFrameLowering()->getStackAlign().value());
        Lowering L = getLowering(Offset, Amount);
        Lowerings[&MI] = L;
        switch (L) {
        case Sub:
          Offset += Amount;
          break;
        case TouchAndSub:
          Offset = Amount;
          break;
        case Probe:
          Offset = 0;
          break;
        }
      } else if (MI.isCall() || isPushPop(MI)) {
        // Calls, pushes and pops touch the tip of the stack.
        Offset = 0;
      } else if (MI.getOpcode() == X86::ADJCALLSTACKUP32 ||
                 MI.getOpcode() == X86::ADJCALLSTACKUP64) {
        if (!HasStableWin64MSVCCallFrame ||
            MI.getOpcode() != X86::ADJCALLSTACKUP64)
          Offset -= MI.getOperand(0).getImm();
      } else if (MI.getOpcode() == X86::ADJCALLSTACKDOWN32 ||
                 MI.getOpcode() == X86::ADJCALLSTACKDOWN64) {
        if (!HasStableWin64MSVCCallFrame ||
            MI.getOpcode() != X86::ADJCALLSTACKDOWN64)
          Offset += MI.getOperand(0).getImm();
      } else if (MI.modifiesRegister(StackPtr, TRI)) {
        // Any other modification of SP means we've lost track of it.
        Offset = INT32_MAX;
      }
    }

    OutOffset[MBB] = Offset;
  }
}

static unsigned getSubOpcode(bool Is64Bit) {
  if (Is64Bit)
    return X86::SUB64ri32;
  return X86::SUB32ri;
}

Register X86DynAllocaExpander::materializeWinAllocaAmount(
    MachineInstr *MI, MachineBasicBlock::iterator I, const DebugLoc &DL) {
  assert(MI->getOpcode() == X86::WIN_ALLOCA_64);

  MachineBasicBlock *MBB = MI->getParent();
  unsigned StackAlign = STI->getFrameLowering()->getStackAlign().value();
  unsigned Alignment = MI->getOperand(2).getImm();
  Register RawAmountReg = MI->getOperand(1).getReg();

  uint64_t AlignmentSlack = Alignment > StackAlign ? Alignment - 1 : 0;
  uint64_t RoundUpBias = StackAlign - 1;
  uint64_t Addend = AlignmentSlack + RoundUpBias;

  Register AddendReg = MRI->createVirtualRegister(&X86::GR64RegClass);
  BuildMI(*MBB, I, DL, TII->get(X86::MOV64ri), AddendReg).addImm(Addend);

  Register UnalignedAmountReg = MRI->createVirtualRegister(&X86::GR64RegClass);
  BuildMI(*MBB, I, DL, TII->get(X86::ADD64rr), UnalignedAmountReg)
      .addReg(RawAmountReg)
      .addReg(AddendReg);

  Register AmountReg = MRI->createVirtualRegister(&X86::GR64RegClass);
  BuildMI(*MBB, I, DL, TII->get(X86::AND64ri32), AmountReg)
      .addReg(UnalignedAmountReg)
      .addImm(-int64_t(StackAlign));
  return AmountReg;
}

void X86DynAllocaExpander::lower(MachineInstr *MI, Lowering L) {
  const DebugLoc &DL = MI->getDebugLoc();
  MachineBasicBlock *MBB = MI->getParent();
  unsigned Win64MSVCDynAllocaCallFrameSize =
      MBB->getParent()
          ->getInfo<X86MachineFunctionInfo>()
          ->getWin64MSVCDynAllocaCallFrameSize();
  MachineBasicBlock::iterator I = *MI;

  bool IsWinAlloca = MI->getOpcode() == X86::WIN_ALLOCA_64;
  int64_t Amount = getDynAllocaAmount(
      MI, MRI, STI->getFrameLowering()->getStackAlign().value());
  if (Amount == 0 && !IsWinAlloca) {
    MI->eraseFromParent();
    return;
  }

  // These two variables differ on x32, which is a 64-bit target with a
  // 32-bit alloca.
  bool Is64Bit = STI->is64Bit();
  bool Is64BitAlloca = MI->getOpcode() == X86::DYN_ALLOCA_64 ||
                       MI->getOpcode() == X86::WIN_ALLOCA_64;
  assert(SlotSize == 4 || SlotSize == 8);

  std::optional<MachineFunction::DebugInstrOperandPair> InstrNum;
  if (unsigned Num = MI->peekDebugInstrNum(); !IsWinAlloca && Num) {
    // Operand 2 of DYN_ALLOCAs contains the stack def.
    InstrNum = {Num, 2};
  }

  unsigned AmountOperandIndex = getDynAllocaAmountOperandIndex(MI);
  Register AmountReg = MI->getOperand(AmountOperandIndex).getReg();

  if (Amount != 0) {
    switch (L) {
    case TouchAndSub: {
      assert(Amount >= SlotSize);

      // Use a push to touch the top of the stack.
      unsigned RegA = Is64Bit ? X86::RAX : X86::EAX;
      BuildMI(*MBB, I, DL, TII->get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
          .addReg(RegA, RegState::Undef);
      Amount -= SlotSize;
      if (!Amount)
        break;

      // Fall through to make any remaining adjustment.
      [[fallthrough]];
    }
    case Sub:
      assert(Amount > 0);
      if (Amount == SlotSize) {
        // Use push to save size.
        unsigned RegA = Is64Bit ? X86::RAX : X86::EAX;
        BuildMI(*MBB, I, DL, TII->get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
            .addReg(RegA, RegState::Undef);
      } else {
        // Sub.
        BuildMI(*MBB, I, DL, TII->get(getSubOpcode(Is64BitAlloca)), StackPtr)
            .addReg(StackPtr)
            .addImm(Amount);
      }
      break;
    case Probe:
      Register ProbeAmountReg = AmountReg;
      if (IsWinAlloca) {
        if (Amount >= 0) {
          ProbeAmountReg = MRI->createVirtualRegister(&X86::GR64RegClass);
          BuildMI(*MBB, I, DL, TII->get(X86::MOV64ri), ProbeAmountReg)
              .addImm(Amount);
        } else {
          ProbeAmountReg = materializeWinAllocaAmount(MI, I, DL);
        }
      }
      if (!NoStackArgProbe) {
        // The probe lowering expects the amount in RAX/EAX.
        unsigned RegA = Is64BitAlloca ? X86::RAX : X86::EAX;
        BuildMI(*MBB, MI, DL, TII->get(TargetOpcode::COPY), RegA)
            .addReg(ProbeAmountReg);

        // Do the probe.
        STI->getFrameLowering()->emitStackProbe(*MBB->getParent(), *MBB, MI, DL,
                                                /*InProlog=*/false, InstrNum);
      } else {
        // Sub
        BuildMI(*MBB, I, DL,
                TII->get(Is64BitAlloca ? X86::SUB64rr : X86::SUB32rr), StackPtr)
            .addReg(StackPtr)
            .addReg(ProbeAmountReg);
      }
      break;
    }
  }

  if (IsWinAlloca) {
    Register DstReg = MI->getOperand(0).getReg();
    unsigned Alignment = MI->getOperand(2).getImm();
    unsigned StackAlign = STI->getFrameLowering()->getStackAlign().value();

    if (Alignment > StackAlign) {
      Register TmpReg = MRI->createVirtualRegister(&X86::GR64RegClass);
      addRegOffset(BuildMI(*MBB, I, DL, TII->get(X86::LEA64r), TmpReg),
                   StackPtr, false,
                   Win64MSVCDynAllocaCallFrameSize + Alignment - 1);
      BuildMI(*MBB, I, DL, TII->get(X86::AND64ri32), DstReg)
          .addReg(TmpReg)
          .addImm(-int64_t(Alignment));
    } else if (Win64MSVCDynAllocaCallFrameSize != 0) {
      addRegOffset(BuildMI(*MBB, I, DL, TII->get(X86::LEA64r), DstReg),
                   StackPtr, false, Win64MSVCDynAllocaCallFrameSize);
    } else {
      BuildMI(*MBB, I, DL, TII->get(TargetOpcode::COPY), DstReg)
          .addReg(StackPtr);
    }
  }

  MI->eraseFromParent();

  // Delete the definition of AmountReg.
  if (MRI->use_empty(AmountReg))
    if (MachineInstr *AmountDef = MRI->getUniqueVRegDef(AmountReg))
      AmountDef->eraseFromParent();
}

bool X86DynAllocaExpander::run(MachineFunction &MF) {
  if (!MF.getInfo<X86MachineFunctionInfo>()->hasDynAlloca())
    return false;

  MRI = &MF.getRegInfo();
  STI = &MF.getSubtarget<X86Subtarget>();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  StackPtr = TRI->getStackRegister();
  SlotSize = TRI->getSlotSize();
  StackProbeSize = STI->getTargetLowering()->getStackProbeSize(MF);
  NoStackArgProbe = MF.getFunction().hasFnAttribute("no-stack-arg-probe");
  auto *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  auto &MFI = MF.getFrameInfo();

  unsigned Win64MSVCDynAllocaCallFrameSize = 0;
  if (STI->isTargetWin64() && STI->isTargetWindowsMSVC()) {
    MFI.computeMaxCallFrameSize(MF);
    Align StackAlign = STI->getFrameLowering()->getStackAlign();
    Win64MSVCDynAllocaCallFrameSize =
        static_cast<unsigned>(alignTo(MFI.getMaxCallFrameSize(), StackAlign));
    assert(Win64MSVCDynAllocaCallFrameSize == 0 ||
           !X86FI->getHasPushSequences());
    assert(Win64MSVCDynAllocaCallFrameSize == 0 ||
           !X86FI->hasPreallocatedCall());
  }
  X86FI->setWin64MSVCDynAllocaCallFrameSize(Win64MSVCDynAllocaCallFrameSize);

  if (NoStackArgProbe)
    StackProbeSize = INT64_MAX;

  LoweringMap Lowerings;
  computeLowerings(MF, Lowerings);
  for (auto &P : Lowerings)
    lower(P.first, P.second);

  return true;
}

bool X86DynAllocaExpanderLegacy::runOnMachineFunction(MachineFunction &MF) {
  return X86DynAllocaExpander().run(MF);
}

PreservedAnalyses
X86DynAllocaExpanderPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  bool Changed = X86DynAllocaExpander().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
