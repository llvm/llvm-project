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

/// Generate ACE-specific tile copy using TILEMOVROW row-by-row.
/// ACE v1 doesn't have TILELOADD/TILESTORED, so we need to copy
/// tiles through ZMM registers, one row at a time.
///
/// Spill sequence (tile -> stack):
///   for row = 0 to 15:
///     TILEMOVROW zmm_temp, tmm_src, row   ; read row from tile to ZMM
///     VMOVUPS [stack + row*64], zmm_temp  ; store ZMM to stack
///
/// Reload sequence (stack -> tile):
///   for row = 0 to 15:
///     VMOVUPS zmm_temp, [stack + row*64]  ; load ZMM from stack
///     TILEMOVROW tmm_dst, zmm_temp, row   ; write row from ZMM to tile
///
static void emitACETileCopy(MachineBasicBlock &MBB, MachineInstr &MI,
                            const X86Subtarget &ST, const X86InstrInfo *TII,
                            const TargetRegisterInfo *TRI, Register SrcReg,
                            Register DstReg, bool SrcKill, int TileSS,
                            Register ScratchZMM) {
  const DebugLoc &DL = MI.getDebugLoc();

  // Each tile has 16 rows of 64 bytes each (1KB total)
  const unsigned NumRows = 16;
  const unsigned RowSize = 64;

  // Spill: Read each row from source tile to ZMM, then store to stack
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    // TILEMOVROW zmm, tmm, imm8 (read direction: tile -> ZMM)
    // Uses TILEMOVROWrti instruction
    // Only kill src on the last row read
    bool KillSrcNow = (Row == NumRows - 1) && SrcKill;
    BuildMI(MBB, MI, DL, TII->get(X86::TILEMOVROWrti), ScratchZMM)
        .addReg(SrcReg, getKillRegState(KillSrcNow))
        .addImm(Row);

    // VMOVUPS [stack + row*64], zmm
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZmr));
    addFrameReference(MIB, TileSS, Row * RowSize);
    MIB.addReg(ScratchZMM, RegState::Kill);
  }

  // Reload: Load each row from stack to ZMM, then write to dest tile
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    // VMOVUPS zmm, [stack + row*64]
    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZrm), ScratchZMM);
    addFrameReference(MIB, TileSS, Row * RowSize);

    // TILEMOVROW tmm, zmm, imm8 (write direction: ZMM -> tile)
    // Uses TILEMOVROWri instruction (ACE-specific)
    BuildMI(MBB, MI, DL, TII->get(X86::TILEMOVROWri), DstReg)
        .addReg(ScratchZMM, RegState::Kill)
        .addImm(Row);
  }
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

      // Check if this is ACE-only (no AMX TILELOADD/TILESTORED available)
      if (ST.hasACEV1() && !ST.hasAMXTILE()) {
        // ACE v1: Use TILEMOVROW row-by-row copy
        // Need a scratch ZMM register for the transfer

        // Find an available ZMM register
        BitVector VR512Regs =
            TRI->getAllocatableSet(MF, TRI->getRegClass(X86::VR512RegClassID));
        Register ScratchZMM = X86::NoRegister;
        for (auto RegT : VR512Regs.set_bits()) {
          if (UsedRegs.available(RegT)) {
            ScratchZMM = RegT;
            break;
          }
        }

        if (!ScratchZMM) {
          // If no ZMM available, use ZMM0 and save/restore it
          ScratchZMM = X86::ZMM0;

          // Allocate stack slot for scratch ZMM
          int ZmmSS = MF.getFrameInfo().CreateSpillStackObject(64, Align(64));

          // Save ZMM0
          MachineInstrBuilder MIB =
              BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZmr));
          addFrameReference(MIB, ZmmSS);
          MIB.addReg(X86::ZMM0);

          // Emit ACE tile copy
          emitACETileCopy(MBB, MI, ST, TII, TRI, SrcReg, DstReg, SrcMO.isKill(),
                          TileSS, ScratchZMM);

          // Restore ZMM0
          MIB = BuildMI(MBB, MI, DL, TII->get(X86::VMOVUPSZrm), X86::ZMM0);
          addFrameReference(MIB, ZmmSS);
        } else {
          // Use available scratch ZMM
          emitACETileCopy(MBB, MI, ST, TII, TRI, SrcReg, DstReg, SrcMO.isKill(),
                          TileSS, ScratchZMM);
        }
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
