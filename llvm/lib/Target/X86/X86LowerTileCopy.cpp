//===-- X86LowerTileCopy.cpp - Expand Tile Copy Instructions---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which lower AMX tile copy instructions. Since
// there is no tile copy instruction, we need store tile register to stack
// and load from stack to another tile register. We need extra GR to hold
// the stride, and we need stack slot to hold the tile data register.
// We would run this pass after copy propagation, so that we don't miss copy
// optimization. And we would run this pass before prolog/epilog insertion,
// so that we can allocate stack slot.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugLoc.h"

using namespace llvm;

#define DEBUG_TYPE "x86-lower-tile-copy"

// ACE has no TILELOADD/TILESTORED, so a tile moves to and from memory one row
// at a time through a scratch ZMM. Each tile is 16 rows of 64 bytes (1KB).
static const unsigned ACENumTileRows = 16;
static const unsigned ACETileRowSize = 64;

/// Emit the ACE tile spill sequence, storing \p SrcReg to \p TileSS.
static void emitACETileSpill(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             const X86InstrInfo *TII, Register SrcReg,
                             bool SrcKill, int TileSS, Register ScratchZMM) {
  const DebugLoc &DL = MI->getDebugLoc();
  for (unsigned Row = 0; Row < ACENumTileRows; ++Row) {
    // tilemovrow $row, %tmm, %zmm
    // Only kill src on the last row read.
    bool KillSrcNow = (Row == ACENumTileRows - 1) && SrcKill;
    BuildMI(MBB, MI, DL, TII->get(X86::TILEMOVROWrti), ScratchZMM)
        .addReg(SrcReg, getKillRegState(KillSrcNow))
        .addImm(Row);

    // vmovups %zmm, row*64(%sp)
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZmr));
    addFrameReference(MIB, TileSS, Row * ACETileRowSize);
    MIB.addReg(ScratchZMM, RegState::Kill);
  }
}

/// Emit the ACE tile reload sequence, loading \p DstReg from \p TileSS.
static void emitACETileReload(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const X86InstrInfo *TII, Register DstReg,
                              int TileSS, Register ScratchZMM) {
  const DebugLoc &DL = MI->getDebugLoc();
  for (unsigned Row = 0; Row < ACENumTileRows; ++Row) {
    // vmovups row*64(%sp), %zmm
    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZrm), ScratchZMM);
    addFrameReference(MIB, TileSS, Row * ACETileRowSize);

    // tilemovrow $row, %zmm, %tmm
    BuildMI(MBB, MI, DL, TII->get(X86::TILEMOVROWri), DstReg)
        .addReg(ScratchZMM, RegState::Kill)
        .addImm(Row);
  }
}

/// Pick a ZMM that is dead here, or NoRegister if none is available.
static Register findScratchZMM(MachineFunction &MF,
                               const TargetRegisterInfo *TRI,
                               const LiveRegUnits &UsedRegs) {
  BitVector VR512Regs =
      TRI->getAllocatableSet(MF, TRI->getRegClass(X86::VR512RegClassID));
  for (auto RegT : VR512Regs.set_bits())
    if (UsedRegs.available(RegT))
      return RegT;
  return X86::NoRegister;
}

namespace {

class X86LowerTileCopyLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86LowerTileCopyLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "X86 Lower Tile Copy"; }
};

} // namespace

char X86LowerTileCopyLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(X86LowerTileCopyLegacy, DEBUG_TYPE, "Tile Copy Lowering",
                      false, false)
INITIALIZE_PASS_END(X86LowerTileCopyLegacy, DEBUG_TYPE, "Tile Copy Lowering",
                    false, false)

void X86LowerTileCopyLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createX86LowerTileCopyLegacyPass() {
  return new X86LowerTileCopyLegacy();
}

static bool lowerTileCopy(MachineFunction &MF) {
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  if (FuncInfo->getAMXProgModel() != AMXProgModelEnum::ManagedRA &&
      FuncInfo->getACEProgModel() != ACEProgModelEnum::ACE_ManagedRA)
    return false;

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  assert((ST.hasAMXTILE() || ST.hasACEV1()) &&
         "Only supported on AMX-TILE or ACE v1 targets");

  const X86InstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  BitVector GR64Regs =
      TRI->getAllocatableSet(MF, TRI->getRegClass(X86::GR64RegClassID));
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    LiveRegUnits UsedRegs(*TRI);
    UsedRegs.addLiveOuts(MBB);
    for (MachineInstr &MI : llvm::make_early_inc_range(reverse(MBB))) {
      if (MI.isDebugInstr())
        continue;
      UsedRegs.stepBackward(MI);

      // Expand the tile spill/reload pseudos inserted by the register
      // allocator. getLoadStoreRegOpcode is the only place these are created,
      // and it only does so for ACE targets.
      unsigned Opcode = MI.getOpcode();
      if (Opcode == X86::ACE_TILESPILL || Opcode == X86::ACE_TILERELOAD) {
        assert(ST.hasACEV1() && "ACE tile spill pseudo on non-ACE target");
        bool IsSpill = Opcode == X86::ACE_TILESPILL;
        // ACE_TILESPILL is (mem, tile), ACE_TILERELOAD is (tile, mem).
        MachineOperand &TileMO =
            MI.getOperand(IsSpill ? X86::AddrNumOperands : 0);
        int TileSS =
            MI.getOperand((IsSpill ? 0 : 1) + X86::AddrBaseReg).getIndex();
        const DebugLoc &DL = MI.getDebugLoc();

        Register ScratchZMM = findScratchZMM(MF, TRI, UsedRegs);
        int ZmmSS = -1;
        if (!ScratchZMM) {
          // No available register? Save ZMM0 and reload it after use.
          ScratchZMM = X86::ZMM0;
          ZmmSS = MF.getFrameInfo().CreateSpillStackObject(64, Align(64));
          addFrameReference(BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZmr)),
                            ZmmSS)
              .addReg(X86::ZMM0);
        }

        if (IsSpill)
          emitACETileSpill(MBB, MI, TII, TileMO.getReg(), TileMO.isKill(),
                           TileSS, ScratchZMM);
        else
          emitACETileReload(MBB, MI, TII, TileMO.getReg(), TileSS, ScratchZMM);

        if (ZmmSS != -1)
          addFrameReference(
              BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZrm), X86::ZMM0),
              ZmmSS);

        MI.eraseFromParent();
        Changed = true;
        continue;
      }

      if (!MI.isCopy())
        continue;
      MachineOperand &DstMO = MI.getOperand(0);
      MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();
      Register DstReg = DstMO.getReg();
      if (!X86::TILERegClass.contains(DstReg, SrcReg))
        continue;

      // Allocate stack slot for tile register (1KB for ACE, same for AMX)
      unsigned Size = TRI->getSpillSize(X86::TILERegClass);
      Align Alignment = TRI->getSpillAlign(X86::TILERegClass);
      int TileSS = MF.getFrameInfo().CreateSpillStackObject(Size, Alignment);

      const DebugLoc &DL = MI.getDebugLoc();

      // ACE configures palette 2, which has no TILELOADD/TILESTORED, so copy
      // the tile row-by-row through a scratch ZMM.
      if (ST.hasACEV1()) {
        Register ScratchZMM = findScratchZMM(MF, TRI, UsedRegs);
        int ZmmSS = -1;
        if (!ScratchZMM) {
          // No available register? Save ZMM0 and reload it after use.
          ScratchZMM = X86::ZMM0;
          ZmmSS = MF.getFrameInfo().CreateSpillStackObject(64, Align(64));
          addFrameReference(BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZmr)),
                            ZmmSS)
              .addReg(X86::ZMM0);
        }

        emitACETileSpill(MBB, MI, TII, SrcReg, SrcMO.isKill(), TileSS,
                         ScratchZMM);
        emitACETileReload(MBB, MI, TII, DstReg, TileSS, ScratchZMM);

        if (ZmmSS != -1)
          addFrameReference(
              BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZrm), X86::ZMM0),
              ZmmSS);
      } else {
        // AMX: Use TILESTORED/TILELOADD (original code)
        int StrideSS = 0;

        // Pick a killed register to avoid a save/reload.
        Register GR64Cand = X86::NoRegister;
        for (auto RegT : GR64Regs.set_bits()) {
          if (UsedRegs.available(RegT)) {
            GR64Cand = RegT;
            break;
          }
        }

        if (GR64Cand) {
          // mov 64 %reg
          BuildMI(MBB, MI, DL, TII->get(X86::MOV64ri), GR64Cand).addImm(64);
        } else {
          // No available register? Save RAX and reload it after use.

          // Allocate stack slot for stride register
          Size = TRI->getSpillSize(X86::GR64RegClass);
          Alignment = TRI->getSpillAlign(X86::GR64RegClass);
          StrideSS = MF.getFrameInfo().CreateSpillStackObject(Size, Alignment);

          // mov %reg (%sp)
          addFrameReference(BuildMI(MBB, MI, DL, TII->get(X86::MOV64mr)),
                            StrideSS)
              .addReg(X86::RAX);
          // mov 64 %reg
          BuildMI(MBB, MI, DL, TII->get(X86::MOV64ri), X86::RAX).addImm(64);
        }
        // tilestored %tmm, (%sp, %idx)
#define GET_EGPR_IF_ENABLED(OPC) (ST.hasEGPR() ? OPC##_EVEX : OPC)
        unsigned Opc = GET_EGPR_IF_ENABLED(X86::TILESTORED);
        MachineInstr *NewMI =
            addFrameReference(BuildMI(MBB, MI, DL, TII->get(Opc)), TileSS)
                .addReg(SrcReg, getKillRegState(SrcMO.isKill()));
        MachineOperand *MO = &NewMI->getOperand(X86::AddrIndexReg);
        MO->setReg(GR64Cand ? GR64Cand : X86::RAX);
        // tileloadd (%sp, %idx), %tmm
        Opc = GET_EGPR_IF_ENABLED(X86::TILELOADD);
#undef GET_EGPR_IF_ENABLED
        NewMI = addFrameReference(BuildMI(MBB, MI, DL, TII->get(Opc), DstReg),
                                  TileSS);
        MO = &NewMI->getOperand(1 + X86::AddrIndexReg);
        MO->setReg(GR64Cand ? GR64Cand : X86::RAX);
        MO->setIsKill(true);
        if (!GR64Cand) {
          // restore %rax
          // mov (%sp) %rax
          addFrameReference(
              BuildMI(MBB, MI, DL, TII->get(X86::MOV64rm), X86::RAX), StrideSS);
        }
      }

      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

bool X86LowerTileCopyLegacy::runOnMachineFunction(MachineFunction &MF) {
  return lowerTileCopy(MF);
}

PreservedAnalyses
X86LowerTileCopyPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  return lowerTileCopy(MF) ? getMachineFunctionPassPreservedAnalyses()
                                 .preserveSet<CFGAnalyses>()
                           : PreservedAnalyses::all();
}
