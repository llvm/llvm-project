//===- SILoadStoreOptimizer.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass tries to fuse DS instructions with close by immediate offsets.
// This will fuse operations such as
//  ds_read_b32 v0, v2 offset:16
//  ds_read_b32 v1, v2 offset:32
// ==>
//   ds_read2_b32 v[0:1], v2, offset0:4 offset1:8
//
// The same is done for certain SMEM opcodes, e.g.:
//  s_buffer_load_dword s4, s[0:3], 4
//  s_buffer_load_dword s5, s[0:3], 8
// ==>
//  s_buffer_load_dwordx2 s[4:5], s[0:3], 4
//
//
// Future improvements:
//
// - This currently relies on the scheduler to place loads and stores next to
//   each other, and then only merges adjacent pairs of instructions. It would
//   be good to be more flexible with interleaved instructions, and possibly run
//   before scheduling. It currently missing stores of constants because loading
//   the constant into the data register is placed between the stores, although
//   this is arguably a scheduling problem.
//
// - Live interval recomputing seems inefficient. This currently only matches
//   one pair, and recomputes live intervals and moves on to the next pair. It
//   would be better to compute a list of all merges that need to occur.
//
// - With a list of instructions to process, we can also merge more. If a
//   cluster of loads have offsets that are too large to fit in the 8-bit
//   offsets, but are close enough to fit in the 8 bits, we can add to the base
//   pointer and use the new reduced offsets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "si-load-store-opt"

namespace {

class SILoadStoreOptimizer : public MachineFunctionPass {
  enum InstClassEnum {
    DS_READ_WRITE,
    S_BUFFER_LOAD_IMM,
    BUFFER_LOAD_OFFEN,
    BUFFER_LOAD_OFFSET,
    BUFFER_STORE_OFFEN,
    BUFFER_STORE_OFFSET,
  };

  struct CombineInfo {
    MachineBasicBlock::iterator I;
    MachineBasicBlock::iterator Paired;
    unsigned EltSize;
    unsigned Offset0;
    unsigned Offset1;
    unsigned BaseOff;
    InstClassEnum InstClass;
    bool GLC0;
    bool GLC1;
    bool SLC0;
    bool SLC1;
    bool UseST64;
    bool IsX2;
    SmallVector<MachineInstr*, 8> InstsToMove;
   };

private:
  const SISubtarget *STM = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AliasAnalysis *AA = nullptr;
  unsigned CreatedX2;

  static bool offsetsCanBeCombined(CombineInfo &CI);

  bool findMatchingInst(CombineInfo &CI);
  MachineBasicBlock::iterator mergeRead2Pair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeWrite2Pair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeSBufferLoadImmPair(CombineInfo &CI);
  MachineBasicBlock::iterator mergeBufferLoadPair(CombineInfo &CI);
  unsigned promoteBufferStoreOpcode(const MachineInstr &I, bool &IsX2,
                                    bool &IsOffen) const;
  MachineBasicBlock::iterator mergeBufferStorePair(CombineInfo &CI);

public:
  static char ID;

  SILoadStoreOptimizer() : MachineFunctionPass(ID) {
    initializeSILoadStoreOptimizerPass(*PassRegistry::getPassRegistry());
  }

  bool optimizeBlock(MachineBasicBlock &MBB);

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Load / Store Optimizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace.

INITIALIZE_PASS_BEGIN(SILoadStoreOptimizer, DEBUG_TYPE,
                      "SI Load / Store Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SILoadStoreOptimizer, DEBUG_TYPE,
                    "SI Load / Store Optimizer", false, false)

char SILoadStoreOptimizer::ID = 0;

char &llvm::SILoadStoreOptimizerID = SILoadStoreOptimizer::ID;

FunctionPass *llvm::createSILoadStoreOptimizerPass() {
  return new SILoadStoreOptimizer();
}

static void moveInstsAfter(MachineBasicBlock::iterator I,
                           ArrayRef<MachineInstr*> InstsToMove) {
  MachineBasicBlock *MBB = I->getParent();
  ++I;
  for (MachineInstr *MI : InstsToMove) {
    MI->removeFromParent();
    MBB->insert(I, MI);
  }
}

static void addDefsToList(const MachineInstr &MI, DenseSet<unsigned> &Defs) {
  // XXX: Should this be looking for implicit defs?
  for (const MachineOperand &Def : MI.defs())
    Defs.insert(Def.getReg());
}

static bool memAccessesCanBeReordered(MachineBasicBlock::iterator A,
                                      MachineBasicBlock::iterator B,
                                      const SIInstrInfo *TII,
                                      AliasAnalysis * AA) {
  // RAW or WAR - cannot reorder
  // WAW - cannot reorder
  // RAR - safe to reorder
  return !(A->mayStore() || B->mayStore()) ||
    TII->areMemAccessesTriviallyDisjoint(*A, *B, AA);
}

// Add MI and its defs to the lists if MI reads one of the defs that are
// already in the list. Returns true in that case.
static bool
addToListsIfDependent(MachineInstr &MI,
                      DenseSet<unsigned> &Defs,
                      SmallVectorImpl<MachineInstr*> &Insts) {
  for (MachineOperand &Use : MI.operands()) {
    // If one of the defs is read, then there is a use of Def between I and the
    // instruction that I will potentially be merged with. We will need to move
    // this instruction after the merged instructions.

    if (Use.isReg() && Use.readsReg() && Defs.count(Use.getReg())) {
      Insts.push_back(&MI);
      addDefsToList(MI, Defs);
      return true;
    }
  }

  return false;
}

static bool
canMoveInstsAcrossMemOp(MachineInstr &MemOp,
                        ArrayRef<MachineInstr*> InstsToMove,
                        const SIInstrInfo *TII,
                        AliasAnalysis *AA) {
  assert(MemOp.mayLoadOrStore());

  for (MachineInstr *InstToMove : InstsToMove) {
    if (!InstToMove->mayLoadOrStore())
      continue;
    if (!memAccessesCanBeReordered(MemOp, *InstToMove, TII, AA))
        return false;
  }
  return true;
}

bool SILoadStoreOptimizer::offsetsCanBeCombined(CombineInfo &CI) {
  // XXX - Would the same offset be OK? Is there any reason this would happen or
  // be useful?
  if (CI.Offset0 == CI.Offset1)
    return false;

  // This won't be valid if the offset isn't aligned.
  if ((CI.Offset0 % CI.EltSize != 0) || (CI.Offset1 % CI.EltSize != 0))
    return false;

  unsigned EltOffset0 = CI.Offset0 / CI.EltSize;
  unsigned EltOffset1 = CI.Offset1 / CI.EltSize;
  CI.UseST64 = false;
  CI.BaseOff = 0;

  // Handle SMEM and VMEM instructions.
  if (CI.InstClass != DS_READ_WRITE) {
    unsigned Diff = CI.IsX2 ? 2 : 1;
    return (EltOffset0 + Diff == EltOffset1 ||
            EltOffset1 + Diff == EltOffset0) &&
           CI.GLC0 == CI.GLC1 &&
           (CI.InstClass == S_BUFFER_LOAD_IMM || CI.SLC0 == CI.SLC1);
  }

  // If the offset in elements doesn't fit in 8-bits, we might be able to use
  // the stride 64 versions.
  if ((EltOffset0 % 64 == 0) && (EltOffset1 % 64) == 0 &&
      isUInt<8>(EltOffset0 / 64) && isUInt<8>(EltOffset1 / 64)) {
    CI.Offset0 = EltOffset0 / 64;
    CI.Offset1 = EltOffset1 / 64;
    CI.UseST64 = true;
    return true;
  }

  // Check if the new offsets fit in the reduced 8-bit range.
  if (isUInt<8>(EltOffset0) && isUInt<8>(EltOffset1)) {
    CI.Offset0 = EltOffset0;
    CI.Offset1 = EltOffset1;
    return true;
  }

  // Try to shift base address to decrease offsets.
  unsigned OffsetDiff = std::abs((int)EltOffset1 - (int)EltOffset0);
  CI.BaseOff = std::min(CI.Offset0, CI.Offset1);

  if ((OffsetDiff % 64 == 0) && isUInt<8>(OffsetDiff / 64)) {
    CI.Offset0 = (EltOffset0 - CI.BaseOff / CI.EltSize) / 64;
    CI.Offset1 = (EltOffset1 - CI.BaseOff / CI.EltSize) / 64;
    CI.UseST64 = true;
    return true;
  }

  if (isUInt<8>(OffsetDiff)) {
    CI.Offset0 = EltOffset0 - CI.BaseOff / CI.EltSize;
    CI.Offset1 = EltOffset1 - CI.BaseOff / CI.EltSize;
    return true;
  }

  return false;
}

bool SILoadStoreOptimizer::findMatchingInst(CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  MachineBasicBlock::iterator E = MBB->end();
  MachineBasicBlock::iterator MBBI = CI.I;

  unsigned AddrOpName[3] = {0};
  int AddrIdx[3];
  const MachineOperand *AddrReg[3];
  unsigned NumAddresses = 0;

  switch (CI.InstClass) {
  case DS_READ_WRITE:
    AddrOpName[NumAddresses++] = AMDGPU::OpName::addr;
    break;
  case S_BUFFER_LOAD_IMM:
    AddrOpName[NumAddresses++] = AMDGPU::OpName::sbase;
    break;
  case BUFFER_LOAD_OFFEN:
  case BUFFER_STORE_OFFEN:
    AddrOpName[NumAddresses++] = AMDGPU::OpName::srsrc;
    AddrOpName[NumAddresses++] = AMDGPU::OpName::vaddr;
    AddrOpName[NumAddresses++] = AMDGPU::OpName::soffset;
    break;
  case BUFFER_LOAD_OFFSET:
  case BUFFER_STORE_OFFSET:
    AddrOpName[NumAddresses++] = AMDGPU::OpName::srsrc;
    AddrOpName[NumAddresses++] = AMDGPU::OpName::soffset;
    break;
  }

  for (unsigned i = 0; i < NumAddresses; i++) {
    AddrIdx[i] = AMDGPU::getNamedOperandIdx(CI.I->getOpcode(), AddrOpName[i]);
    AddrReg[i] = &CI.I->getOperand(AddrIdx[i]);

    // We only ever merge operations with the same base address register, so don't
    // bother scanning forward if there are no other uses.
    if (AddrReg[i]->isReg() &&
        (TargetRegisterInfo::isPhysicalRegister(AddrReg[i]->getReg()) ||
         MRI->hasOneNonDBGUse(AddrReg[i]->getReg())))
      return false;
  }

  ++MBBI;

  DenseSet<unsigned> DefsToMove;
  addDefsToList(*CI.I, DefsToMove);

  for ( ; MBBI != E; ++MBBI) {
    if (MBBI->getOpcode() != CI.I->getOpcode()) {
      // This is not a matching DS instruction, but we can keep looking as
      // long as one of these conditions are met:
      // 1. It is safe to move I down past MBBI.
      // 2. It is safe to move MBBI down past the instruction that I will
      //    be merged into.

      if (MBBI->hasUnmodeledSideEffects()) {
        // We can't re-order this instruction with respect to other memory
        // operations, so we fail both conditions mentioned above.
        return false;
      }

      if (MBBI->mayLoadOrStore() &&
        !memAccessesCanBeReordered(*CI.I, *MBBI, TII, AA)) {
        // We fail condition #1, but we may still be able to satisfy condition
        // #2.  Add this instruction to the move list and then we will check
        // if condition #2 holds once we have selected the matching instruction.
        CI.InstsToMove.push_back(&*MBBI);
        addDefsToList(*MBBI, DefsToMove);
        continue;
      }

      // When we match I with another DS instruction we will be moving I down
      // to the location of the matched instruction any uses of I will need to
      // be moved down as well.
      addToListsIfDependent(*MBBI, DefsToMove, CI.InstsToMove);
      continue;
    }

    // Don't merge volatiles.
    if (MBBI->hasOrderedMemoryRef())
      return false;

    // Handle a case like
    //   DS_WRITE_B32 addr, v, idx0
    //   w = DS_READ_B32 addr, idx0
    //   DS_WRITE_B32 addr, f(w), idx1
    // where the DS_READ_B32 ends up in InstsToMove and therefore prevents
    // merging of the two writes.
    if (addToListsIfDependent(*MBBI, DefsToMove, CI.InstsToMove))
      continue;

    bool Match = true;
    for (unsigned i = 0; i < NumAddresses; i++) {
      const MachineOperand &AddrRegNext = MBBI->getOperand(AddrIdx[i]);

      if (AddrReg[i]->isImm() || AddrRegNext.isImm()) {
        if (AddrReg[i]->isImm() != AddrRegNext.isImm() ||
            AddrReg[i]->getImm() != AddrRegNext.getImm()) {
          Match = false;
          break;
        }
        continue;
      }

      // Check same base pointer. Be careful of subregisters, which can occur with
      // vectors of pointers.
      if (AddrReg[i]->getReg() != AddrRegNext.getReg() ||
          AddrReg[i]->getSubReg() != AddrRegNext.getSubReg()) {
        Match = false;
        break;
      }
    }

    if (Match) {
      int OffsetIdx = AMDGPU::getNamedOperandIdx(CI.I->getOpcode(),
                                                 AMDGPU::OpName::offset);
      CI.Offset0 = CI.I->getOperand(OffsetIdx).getImm();
      CI.Offset1 = MBBI->getOperand(OffsetIdx).getImm();
      CI.Paired = MBBI;

      if (CI.InstClass == DS_READ_WRITE) {
        CI.Offset0 &= 0xffff;
        CI.Offset1 &= 0xffff;
      } else {
        CI.GLC0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::glc)->getImm();
        CI.GLC1 = TII->getNamedOperand(*MBBI, AMDGPU::OpName::glc)->getImm();
        if (CI.InstClass != S_BUFFER_LOAD_IMM) {
          CI.SLC0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::slc)->getImm();
          CI.SLC1 = TII->getNamedOperand(*MBBI, AMDGPU::OpName::slc)->getImm();
        }
      }

      // Check both offsets fit in the reduced range.
      // We also need to go through the list of instructions that we plan to
      // move and make sure they are all safe to move down past the merged
      // instruction.
      if (offsetsCanBeCombined(CI))
        if (canMoveInstsAcrossMemOp(*MBBI, CI.InstsToMove, TII, AA))
          return true;
    }

    // We've found a load/store that we couldn't merge for some reason.
    // We could potentially keep looking, but we'd need to make sure that
    // it was safe to move I and also all the instruction in InstsToMove
    // down past this instruction.
    // check if we can move I across MBBI and if we can move all I's users
    if (!memAccessesCanBeReordered(*CI.I, *MBBI, TII, AA) ||
      !canMoveInstsAcrossMemOp(*MBBI, CI.InstsToMove, TII, AA))
      break;
  }
  return false;
}

MachineBasicBlock::iterator  SILoadStoreOptimizer::mergeRead2Pair(
  CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be careful, since the addresses could be subregisters themselves in weird
  // cases, like vectors of pointers.
  const auto *AddrReg = TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);

  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdst);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdst);

  unsigned NewOffset0 = CI.Offset0;
  unsigned NewOffset1 = CI.Offset1;
  unsigned Opc = (CI.EltSize == 4) ? AMDGPU::DS_READ2_B32
                                   : AMDGPU::DS_READ2_B64;

  if (CI.UseST64)
    Opc = (CI.EltSize == 4) ? AMDGPU::DS_READ2ST64_B32
                            : AMDGPU::DS_READ2ST64_B64;

  unsigned SubRegIdx0 = (CI.EltSize == 4) ? AMDGPU::sub0 : AMDGPU::sub0_sub1;
  unsigned SubRegIdx1 = (CI.EltSize == 4) ? AMDGPU::sub1 : AMDGPU::sub2_sub3;

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(SubRegIdx0, SubRegIdx1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) &&
         "Computed offset doesn't fit");

  const MCInstrDesc &Read2Desc = TII->get(Opc);

  const TargetRegisterClass *SuperRC
    = (CI.EltSize == 4) ? &AMDGPU::VReg_64RegClass : &AMDGPU::VReg_128RegClass;
  unsigned DestReg = MRI->createVirtualRegister(SuperRC);

  DebugLoc DL = CI.I->getDebugLoc();

  unsigned BaseReg = AddrReg->getReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;
    BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::V_ADD_I32_e32), BaseReg)
           .addImm(CI.BaseOff)
           .addReg(AddrReg->getReg());
  }

  MachineInstrBuilder Read2 =
    BuildMI(*MBB, CI.Paired, DL, Read2Desc, DestReg)
      .addReg(BaseReg, BaseRegFlags) // addr
      .addImm(NewOffset0)            // offset0
      .addImm(NewOffset1)            // offset1
      .addImm(0)                     // gds
      .setMemRefs(CI.I->mergeMemRefsWith(*CI.Paired));

  (void)Read2;

  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);

  // Copy to the old destination registers.
  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  MachineBasicBlock::iterator Next = std::next(CI.I);
  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();

  DEBUG(dbgs() << "Inserted read2: " << *Read2 << '\n');
  return Next;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeWrite2Pair(
  CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();

  // Be sure to use .addOperand(), and not .addReg() with these. We want to be
  // sure we preserve the subregister index and any register flags set on them.
  const MachineOperand *Addr = TII->getNamedOperand(*CI.I, AMDGPU::OpName::addr);
  const MachineOperand *Data0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::data0);
  const MachineOperand *Data1
    = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::data0);

  unsigned NewOffset0 = CI.Offset0;
  unsigned NewOffset1 = CI.Offset1;
  unsigned Opc = (CI.EltSize == 4) ? AMDGPU::DS_WRITE2_B32
                                   : AMDGPU::DS_WRITE2_B64;

  if (CI.UseST64)
    Opc = (CI.EltSize == 4) ? AMDGPU::DS_WRITE2ST64_B32
                            : AMDGPU::DS_WRITE2ST64_B64;

  if (NewOffset0 > NewOffset1) {
    // Canonicalize the merged instruction so the smaller offset comes first.
    std::swap(NewOffset0, NewOffset1);
    std::swap(Data0, Data1);
  }

  assert((isUInt<8>(NewOffset0) && isUInt<8>(NewOffset1)) &&
         (NewOffset0 != NewOffset1) &&
         "Computed offset doesn't fit");

  const MCInstrDesc &Write2Desc = TII->get(Opc);
  DebugLoc DL = CI.I->getDebugLoc();

  unsigned BaseReg = Addr->getReg();
  unsigned BaseRegFlags = 0;
  if (CI.BaseOff) {
    BaseReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BaseRegFlags = RegState::Kill;
    BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::V_ADD_I32_e32), BaseReg)
           .addImm(CI.BaseOff)
           .addReg(Addr->getReg());
  }

  MachineInstrBuilder Write2 =
    BuildMI(*MBB, CI.Paired, DL, Write2Desc)
      .addReg(BaseReg, BaseRegFlags) // addr
      .add(*Data0)                   // data0
      .add(*Data1)                   // data1
      .addImm(NewOffset0)            // offset0
      .addImm(NewOffset1)            // offset1
      .addImm(0)                     // gds
      .setMemRefs(CI.I->mergeMemRefsWith(*CI.Paired));

  moveInstsAfter(Write2, CI.InstsToMove);

  MachineBasicBlock::iterator Next = std::next(CI.I);
  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();

  DEBUG(dbgs() << "Inserted write2 inst: " << *Write2 << '\n');
  return Next;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeSBufferLoadImmPair(
  CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  unsigned Opcode = CI.IsX2 ? AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM :
                              AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM;

  const TargetRegisterClass *SuperRC =
    CI.IsX2 ? &AMDGPU::SReg_128RegClass : &AMDGPU::SReg_64_XEXECRegClass;
  unsigned DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset0, CI.Offset1);

  BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg)
      .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::sbase))
      .addImm(MergedOffset) // offset
      .addImm(CI.GLC0)      // glc
      .setMemRefs(CI.I->mergeMemRefsWith(*CI.Paired));

  unsigned SubRegIdx0 = CI.IsX2 ? AMDGPU::sub0_sub1 : AMDGPU::sub0;
  unsigned SubRegIdx1 = CI.IsX2 ? AMDGPU::sub2_sub3 : AMDGPU::sub1;

  // Handle descending offsets
  if (CI.Offset0 > CI.Offset1)
    std::swap(SubRegIdx0, SubRegIdx1);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::sdst);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::sdst);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  MachineBasicBlock::iterator Next = std::next(CI.I);
  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return Next;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeBufferLoadPair(
  CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  unsigned Opcode;

  if (CI.InstClass == BUFFER_LOAD_OFFEN) {
    Opcode = CI.IsX2 ? AMDGPU::BUFFER_LOAD_DWORDX4_OFFEN :
                       AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN;
  } else {
    Opcode = CI.IsX2 ? AMDGPU::BUFFER_LOAD_DWORDX4_OFFSET :
                       AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET;
  }

  const TargetRegisterClass *SuperRC =
    CI.IsX2 ? &AMDGPU::VReg_128RegClass : &AMDGPU::VReg_64RegClass;
  unsigned DestReg = MRI->createVirtualRegister(SuperRC);
  unsigned MergedOffset = std::min(CI.Offset0, CI.Offset1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode), DestReg);

  if (CI.InstClass == BUFFER_LOAD_OFFEN)
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
      .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
      .addImm(MergedOffset) // offset
      .addImm(CI.GLC0)      // glc
      .addImm(CI.SLC0)      // slc
      .addImm(0)            // tfe
      .setMemRefs(CI.I->mergeMemRefsWith(*CI.Paired));

  unsigned SubRegIdx0 = CI.IsX2 ? AMDGPU::sub0_sub1 : AMDGPU::sub0;
  unsigned SubRegIdx1 = CI.IsX2 ? AMDGPU::sub2_sub3 : AMDGPU::sub1;

  // Handle descending offsets
  if (CI.Offset0 > CI.Offset1)
    std::swap(SubRegIdx0, SubRegIdx1);

  // Copy to the old destination registers.
  const MCInstrDesc &CopyDesc = TII->get(TargetOpcode::COPY);
  const auto *Dest0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Dest1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, CopyDesc)
      .add(*Dest0) // Copy to same destination including flags and sub reg.
      .addReg(DestReg, 0, SubRegIdx0);
  MachineInstr *Copy1 = BuildMI(*MBB, CI.Paired, DL, CopyDesc)
                            .add(*Dest1)
                            .addReg(DestReg, RegState::Kill, SubRegIdx1);

  moveInstsAfter(Copy1, CI.InstsToMove);

  MachineBasicBlock::iterator Next = std::next(CI.I);
  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return Next;
}

unsigned SILoadStoreOptimizer::promoteBufferStoreOpcode(
  const MachineInstr &I, bool &IsX2, bool &IsOffen) const {
  IsX2 = false;
  IsOffen = false;

  switch (I.getOpcode()) {
  case AMDGPU::BUFFER_STORE_DWORD_OFFEN:
    IsOffen = true;
    return AMDGPU::BUFFER_STORE_DWORDX2_OFFEN;
  case AMDGPU::BUFFER_STORE_DWORD_OFFEN_exact:
    IsOffen = true;
    return AMDGPU::BUFFER_STORE_DWORDX2_OFFEN_exact;
  case AMDGPU::BUFFER_STORE_DWORDX2_OFFEN:
    IsX2 = true;
    IsOffen = true;
    return AMDGPU::BUFFER_STORE_DWORDX4_OFFEN;
  case AMDGPU::BUFFER_STORE_DWORDX2_OFFEN_exact:
    IsX2 = true;
    IsOffen = true;
    return AMDGPU::BUFFER_STORE_DWORDX4_OFFEN_exact;
  case AMDGPU::BUFFER_STORE_DWORD_OFFSET:
    return AMDGPU::BUFFER_STORE_DWORDX2_OFFSET;
  case AMDGPU::BUFFER_STORE_DWORD_OFFSET_exact:
    return AMDGPU::BUFFER_STORE_DWORDX2_OFFSET_exact;
  case AMDGPU::BUFFER_STORE_DWORDX2_OFFSET:
    IsX2 = true;
    return AMDGPU::BUFFER_STORE_DWORDX4_OFFSET;
  case AMDGPU::BUFFER_STORE_DWORDX2_OFFSET_exact:
    IsX2 = true;
    return AMDGPU::BUFFER_STORE_DWORDX4_OFFSET_exact;
  }
  return 0;
}

MachineBasicBlock::iterator SILoadStoreOptimizer::mergeBufferStorePair(
  CombineInfo &CI) {
  MachineBasicBlock *MBB = CI.I->getParent();
  DebugLoc DL = CI.I->getDebugLoc();
  bool Unused1, Unused2;
  unsigned Opcode = promoteBufferStoreOpcode(*CI.I, Unused1, Unused2);

  unsigned SubRegIdx0 = CI.IsX2 ? AMDGPU::sub0_sub1 : AMDGPU::sub0;
  unsigned SubRegIdx1 = CI.IsX2 ? AMDGPU::sub2_sub3 : AMDGPU::sub1;

  // Handle descending offsets
  if (CI.Offset0 > CI.Offset1)
    std::swap(SubRegIdx0, SubRegIdx1);

  // Copy to the new source register.
  const TargetRegisterClass *SuperRC =
    CI.IsX2 ? &AMDGPU::VReg_128RegClass : &AMDGPU::VReg_64RegClass;
  unsigned SrcReg = MRI->createVirtualRegister(SuperRC);

  const auto *Src0 = TII->getNamedOperand(*CI.I, AMDGPU::OpName::vdata);
  const auto *Src1 = TII->getNamedOperand(*CI.Paired, AMDGPU::OpName::vdata);

  BuildMI(*MBB, CI.Paired, DL, TII->get(AMDGPU::REG_SEQUENCE), SrcReg)
      .add(*Src0)
      .addImm(SubRegIdx0)
      .add(*Src1)
      .addImm(SubRegIdx1);

  auto MIB = BuildMI(*MBB, CI.Paired, DL, TII->get(Opcode))
      .addReg(SrcReg, RegState::Kill);

  if (CI.InstClass == BUFFER_STORE_OFFEN)
      MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::vaddr));

  MIB.add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::srsrc))
      .add(*TII->getNamedOperand(*CI.I, AMDGPU::OpName::soffset))
      .addImm(std::min(CI.Offset0, CI.Offset1)) // offset
      .addImm(CI.GLC0)      // glc
      .addImm(CI.SLC0)      // slc
      .addImm(0)            // tfe
      .setMemRefs(CI.I->mergeMemRefsWith(*CI.Paired));

  moveInstsAfter(MIB, CI.InstsToMove);

  MachineBasicBlock::iterator Next = std::next(CI.I);
  CI.I->eraseFromParent();
  CI.Paired->eraseFromParent();
  return Next;
}

// Scan through looking for adjacent LDS operations with constant offsets from
// the same base register. We rely on the scheduler to do the hard work of
// clustering nearby loads, and assume these are all adjacent.
bool SILoadStoreOptimizer::optimizeBlock(MachineBasicBlock &MBB) {
  bool Modified = false;

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I;

    // Don't combine if volatile.
    if (MI.hasOrderedMemoryRef()) {
      ++I;
      continue;
    }

    CombineInfo CI;
    CI.I = I;
    unsigned Opc = MI.getOpcode();
    if (Opc == AMDGPU::DS_READ_B32 || Opc == AMDGPU::DS_READ_B64) {
      CI.InstClass = DS_READ_WRITE;
      CI.EltSize = (Opc == AMDGPU::DS_READ_B64) ? 8 : 4;
      if (findMatchingInst(CI)) {
        Modified = true;
        I = mergeRead2Pair(CI);
      } else {
        ++I;
      }

      continue;
    }
    if (Opc == AMDGPU::DS_WRITE_B32 || Opc == AMDGPU::DS_WRITE_B64) {
      CI.InstClass = DS_READ_WRITE;
      CI.EltSize = (Opc == AMDGPU::DS_WRITE_B64) ? 8 : 4;
      if (findMatchingInst(CI)) {
        Modified = true;
        I = mergeWrite2Pair(CI);
      } else {
        ++I;
      }

      continue;
    }
    if (STM->hasSBufferLoadStoreAtomicDwordxN() &&
        (Opc == AMDGPU::S_BUFFER_LOAD_DWORD_IMM ||
         Opc == AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM)) {
      // EltSize is in units of the offset encoding.
      CI.InstClass = S_BUFFER_LOAD_IMM;
      CI.EltSize = AMDGPU::getSMRDEncodedOffset(*STM, 4);
      CI.IsX2 = Opc == AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM;
      if (findMatchingInst(CI)) {
        Modified = true;
        I = mergeSBufferLoadImmPair(CI);
        if (!CI.IsX2)
          CreatedX2++;
      } else {
        ++I;
      }
      continue;
    }
    if (Opc == AMDGPU::BUFFER_LOAD_DWORD_OFFEN ||
        Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN ||
        Opc == AMDGPU::BUFFER_LOAD_DWORD_OFFSET ||
        Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET) {
      if (Opc == AMDGPU::BUFFER_LOAD_DWORD_OFFEN ||
          Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN)
        CI.InstClass = BUFFER_LOAD_OFFEN;
      else
        CI.InstClass = BUFFER_LOAD_OFFSET;

      CI.EltSize = 4;
      CI.IsX2 = Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN ||
                Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET;
      if (findMatchingInst(CI)) {
        Modified = true;
        I = mergeBufferLoadPair(CI);
        if (!CI.IsX2)
          CreatedX2++;
      } else {
        ++I;
      }
      continue;
    }

    bool StoreIsX2, IsOffen;
    if (promoteBufferStoreOpcode(*I, StoreIsX2, IsOffen)) {
      CI.InstClass = IsOffen ? BUFFER_STORE_OFFEN : BUFFER_STORE_OFFSET;
      CI.EltSize = 4;
      CI.IsX2 = StoreIsX2;
      if (findMatchingInst(CI)) {
        Modified = true;
        I = mergeBufferStorePair(CI);
        if (!CI.IsX2)
          CreatedX2++;
      } else {
        ++I;
      }
      continue;
    }

    ++I;
  }

  return Modified;
}

bool SILoadStoreOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  STM = &MF.getSubtarget<SISubtarget>();
  if (!STM->loadStoreOptEnabled())
    return false;

  TII = STM->getInstrInfo();
  TRI = &TII->getRegisterInfo();

  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  assert(MRI->isSSA() && "Must be run on SSA");

  DEBUG(dbgs() << "Running SILoadStoreOptimizer\n");

  bool Modified = false;
  CreatedX2 = 0;

  for (MachineBasicBlock &MBB : MF)
    Modified |= optimizeBlock(MBB);

  // Run again to convert x2 to x4.
  if (CreatedX2 >= 1) {
    for (MachineBasicBlock &MBB : MF)
      Modified |= optimizeBlock(MBB);
  }

  return Modified;
}
