//===-- GCNMFMAAccumTileReuse.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Reuse MFMA 512-bit accumulator virtual registers when live intervals do not
/// overlap. The SSA machine scheduler runs before register coalescing and often
/// leaves one distinct tile vreg per MFMA. This pass assigns non-overlapping
/// tiles to a minimal set of canonical vregs via allocation hints before register
/// allocation. Hints express that non-overlapping tiles may share a physical
/// register without merging SSA names (replaceRegWith would break dominance).
//
//===----------------------------------------------------------------------===//

#include "GCNMFMAAccumTileReuse.h"
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-mfma-accum-tile-reuse"

static cl::opt<bool>
    EnableMFMAAccumTileReuse("amdgpu-mfma-accum-tile-reuse", cl::init(true),
                             cl::Hidden,
                             cl::desc("Reuse non-overlapping MFMA 512-bit "
                                      "accumulator vregs before RA"));

STATISTIC(NumMFMAAccumTileCandidates,
          "Number of MFMA 512-bit dest vregs considered");
STATISTIC(NumMFMAAccumTilesBefore, "MFMA tile colors before reuse");
STATISTIC(NumMFMAAccumTilesAfter, "MFMA tile colors after reuse");
STATISTIC(NumMFMAAccumVRegsMerged,
          "Number of MFMA tile vregs hinted to share canonical tiles");

static bool isMFMA512DestClass(const TargetRegisterClass *RC,
                               const SIRegisterInfo &TRI) {
  if (!RC)
    return false;
  return TRI.getRegSizeInBits(*RC) == 512;
}

class GCNMFMAAccumTileReuseImpl {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;

public:
  GCNMFMAAccumTileReuseImpl(const GCNSubtarget &ST, MachineRegisterInfo &MRI,
                            LiveIntervals &LIS)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()), MRI(MRI),
        LIS(LIS) {}

  bool run(MachineFunction &MF);
};

class GCNMFMAAccumTileReuseLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNMFMAAccumTileReuseLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "GCN MFMA Accumulator Tile Reuse";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

bool GCNMFMAAccumTileReuseImpl::run(MachineFunction &MF) {
  SmallSetVector<Register, 32> TileVRegs;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!TII.isMFMA(MI))
        continue;

      Register Dest = MI.getOperand(0).getReg();
      if (!Dest.isVirtual())
        continue;

      if (!isMFMA512DestClass(MRI.getRegClass(Dest), TRI))
        continue;

      if (!LIS.hasInterval(Dest))
        continue;

      TileVRegs.insert(Dest);
    }
  }

  NumMFMAAccumTileCandidates += TileVRegs.size();
  if (TileVRegs.size() <= 1)
    return false;

  NumMFMAAccumTilesBefore += TileVRegs.size();

  SmallVector<Register, 32> Sorted(TileVRegs.begin(), TileVRegs.end());
  llvm::sort(Sorted, [&](Register A, Register B) {
    SlotIndex StartA = LIS.getInterval(A).beginIndex();
    SlotIndex StartB = LIS.getInterval(B).beginIndex();
    if (StartA != StartB)
      return StartA < StartB;
    return A.id() < B.id();
  });

  // Greedy interval coloring: each color is a set of mutually non-overlapping
  // tile vregs. The first vreg in each color is the canonical representative.
  SmallVector<SmallVector<Register, 4>, 8> Colors;
  for (Register Reg : Sorted) {
    const LiveInterval &LI = LIS.getInterval(Reg);
    unsigned ColorIdx = 0;
    for (; ColorIdx < Colors.size(); ++ColorIdx) {
      bool Interferes = false;
      for (Register Other : Colors[ColorIdx]) {
        if (LI.overlaps(LIS.getInterval(Other))) {
          Interferes = true;
          break;
        }
      }
      if (!Interferes)
        break;
    }

    if (ColorIdx == Colors.size())
      Colors.emplace_back();
    Colors[ColorIdx].push_back(Reg);
  }

  NumMFMAAccumTilesAfter += Colors.size();

  bool Changed = false;
  for (const SmallVector<Register, 4> &Color : Colors) {
    if (Color.size() <= 1)
      continue;

    Register Canonical = Color.front();
    for (unsigned I = 1, E = Color.size(); I != E; ++I) {
      Register Reg = Color[I];
      MRI.setSimpleHint(Reg, Canonical);
      ++NumMFMAAccumVRegsMerged;
      Changed = true;
    }
  }

  return Changed;
}

bool GCNMFMAAccumTileReuseLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || !EnableMFMAAccumTileReuse ||
      !amdgpuUseSSAMachineScheduler())
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMAIInsts())
    return false;

  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNMFMAAccumTileReuseImpl(ST, MF.getRegInfo(), LIS).run(MF);
}

INITIALIZE_PASS_BEGIN(GCNMFMAAccumTileReuseLegacy, DEBUG_TYPE,
                      "GCN MFMA Accumulator Tile Reuse", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNMFMAAccumTileReuseLegacy, DEBUG_TYPE,
                    "GCN MFMA Accumulator Tile Reuse", false, false)

char GCNMFMAAccumTileReuseLegacy::ID = 0;

char &llvm::GCNMFMAAccumTileReuseID = GCNMFMAAccumTileReuseLegacy::ID;

FunctionPass *llvm::createGCNMFMAAccumTileReuseLegacyPass() {
  return new GCNMFMAAccumTileReuseLegacy();
}

PreservedAnalyses
GCNMFMAAccumTileReusePass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  if (!EnableMFMAAccumTileReuse || !amdgpuUseSSAMachineScheduler())
    return PreservedAnalyses::all();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMAIInsts())
    return PreservedAnalyses::all();

  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  if (!GCNMFMAAccumTileReuseImpl(ST, MF.getRegInfo(), LIS).run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
