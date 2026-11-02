//===- SIMaterializeWMMAScale.cpp - Materialize WMMA scales to VGPRs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copy an SGPR WMMA scale into a low VGPR before the scheduler.
///
/// v_wmma_scale scale operands accept an SGPR or a VGPR. depctr_va_ssrc is one
/// deep, so each VALU instruction that reads an SGPR scale takes the only
/// scalar-source slot and a run of such WMMA ops serializes. One cross-bank
/// copy pays for that slot once; later WMMA ops then read a VGPR.
///
/// A single WMMA is left on the SGPR: the copy would still occupy va_ssrc and
/// would add a VALU op. A scale defined outside a loop is copied at its def
/// even for one in-loop reader, so the loop body does not hit the counter. A
/// copy inside a loop that modifies exec stays in that loop. The VGPR latches
/// each lane as it drops out, so a use after the loop keeps the SGPR and reads
/// the reconverged scalar. A subregister use is copied at that width, so two
/// lanes of one SGPR tuple stay independent.
//
//===----------------------------------------------------------------------===//

#include "SIMaterializeWMMAScale.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-materialize-wmma-scale"

STATISTIC(NumWMMAScalesMaterialized,
          "Number of SGPR WMMA scales materialized into VGPRs");

/// VALU SGPR-source issue slots. depctr_va_ssrc is a 3-bit counter.
static constexpr unsigned VaSsrcDepth = 7;

namespace {

class SIMaterializeWMMAScale {
  struct ScaleUse {
    MachineInstr *MI;
    unsigned OpNo;
  };

  MachineRegisterInfo *MRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  DenseMap<const MachineLoop *, bool> ExecLoopCache;

  bool loopModifiesExec(const MachineLoop &L);
  bool crossesUnsafeDivergentLoop(const MachineBasicBlock *DefBB,
                                  const MachineBasicBlock *UseBB);
  bool materialize(Register Reg, unsigned SubReg, ArrayRef<ScaleUse> Uses);

public:
  SIMaterializeWMMAScale() = default;
  bool run(MachineFunction &MF, const MachineLoopInfo &Loops);
};

class SIMaterializeWMMAScaleLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIMaterializeWMMAScaleLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Materialize WMMA Scale"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIMaterializeWMMAScaleLegacy, DEBUG_TYPE,
                      "SI Materialize WMMA Scale", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIMaterializeWMMAScaleLegacy, DEBUG_TYPE,
                    "SI Materialize WMMA Scale", false, false)

char SIMaterializeWMMAScaleLegacy::ID = 0;

char &llvm::SIMaterializeWMMAScaleLegacyID = SIMaterializeWMMAScaleLegacy::ID;

FunctionPass *llvm::createSIMaterializeWMMAScaleLegacyPass() {
  return new SIMaterializeWMMAScaleLegacy();
}

bool SIMaterializeWMMAScale::loopModifiesExec(const MachineLoop &L) {
  auto [It, Inserted] = ExecLoopCache.try_emplace(&L, false);
  if (!Inserted)
    return It->second;

  for (const MachineBasicBlock *MBB : L.getBlocks()) {
    for (const MachineInstr &MI : *MBB) {
      if (MI.modifiesRegister(TRI->getExec(), TRI)) {
        It->second = true;
        return true;
      }
    }
  }
  return false;
}

bool SIMaterializeWMMAScale::crossesUnsafeDivergentLoop(
    const MachineBasicBlock *DefBB, const MachineBasicBlock *UseBB) {
  // A loop that contains the copy and not the use, and that modifies exec,
  // reconverges to one SGPR value. The VGPR would still hold each lane's
  // latched value.
  for (const MachineLoop *L = MLI->getLoopFor(DefBB); L && !L->contains(UseBB);
       L = L->getParentLoop()) {
    if (loopModifiesExec(*L))
      return true;
  }
  return false;
}

bool SIMaterializeWMMAScale::materialize(Register Reg, unsigned SubReg,
                                         ArrayRef<ScaleUse> Uses) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;

  const MachineBasicBlock *DefBB = Def->getParent();
  SmallVector<ScaleUse, 4> Rewritable;
  for (const ScaleUse &Use : Uses) {
    if (!crossesUnsafeDivergentLoop(DefBB, Use.MI->getParent()))
      Rewritable.push_back(Use);
  }
  if (Rewritable.empty())
    return false;

  SmallPtrSet<MachineInstr *, 4> Readers;
  bool DefOutsideReaderLoop = false;
  for (const ScaleUse &Use : Rewritable) {
    Readers.insert(Use.MI);
    if (const MachineLoop *UseLoop = MLI->getLoopFor(Use.MI->getParent()))
      DefOutsideReaderLoop |= !UseLoop->contains(DefBB);
  }

  // One SGPR read occupies the only va_ssrc slot. A copy does too, so it
  // wins only when more than one WMMA would otherwise take that slot, or
  // when a loop would take it on every iteration.
  if (Readers.size() <= VaSsrcDepth && !DefOutsideReaderLoop)
    return false;

  // Size the VGPR from the subregister, not the surrounding tuple. Copying
  // the whole SGPR would move lanes this use does not read.
  unsigned BitWidth =
      SubReg ? TRI->getSubRegIdxSize(SubReg)
             : TRI->getRegSizeInBits(*MRI->getRegClass(Reg)).getFixedValue();
  const TargetRegisterClass *VRC =
      TRI->getAlignedLo256VGPRClassForBitWidth(BitWidth);
  if (!VRC)
    return false;

  Register VReg = MRI->createVirtualRegister(VRC);
  MachineBasicBlock &MBB = *Def->getParent();
  MachineBasicBlock::iterator InsertPt =
      Def->isPHI() ? MBB.getFirstNonPHI() : std::next(Def->getIterator());
  BuildMI(MBB, InsertPt, Def->getDebugLoc(), TII->get(TargetOpcode::COPY), VReg)
      .addUse(Reg, {}, SubReg);

  for (const ScaleUse &Use : Rewritable) {
    MachineOperand &MO = Use.MI->getOperand(Use.OpNo);
    MO.setReg(VReg);
    MO.setSubReg(0);
    MO.setIsKill(false);
  }

  LLVM_DEBUG(dbgs() << "Materialized WMMA scale " << printReg(Reg, TRI, SubReg)
                    << " -> " << printReg(VReg, TRI) << '\n');
  ++NumWMMAScalesMaterialized;
  return true;
}

bool SIMaterializeWMMAScale::run(MachineFunction &MF,
                                 const MachineLoopInfo &Loops) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasWMMAN16Insts() && !ST.hasWMMAF4Insts())
    return false;

  MRI = &MF.getRegInfo();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MLI = &Loops;
  ExecLoopCache.clear();

  // Insertion order keeps copy placement deterministic. Subregister lanes of
  // one tuple are separate scales.
  MapVector<std::pair<Register, unsigned>, SmallVector<ScaleUse, 4>> ScaleUses;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!SIInstrInfo::isWMMA(MI) ||
          !AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::scale_src0))
        continue;

      for (AMDGPU::OpName Name :
           {AMDGPU::OpName::scale_src0, AMDGPU::OpName::scale_src1}) {
        int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
        if (Idx < 0)
          continue;
        MachineOperand &MO = MI.getOperand(Idx);
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        if (!TRI->isSGPRReg(*MRI, MO.getReg()))
          continue;
        ScaleUses[{MO.getReg(), MO.getSubReg()}].push_back(
            {&MI, static_cast<unsigned>(Idx)});
      }
    }
  }

  bool Changed = false;
  for (const auto &[Key, Uses] : ScaleUses)
    Changed |= materialize(Key.first, Key.second, Uses);
  return Changed;
}

bool SIMaterializeWMMAScaleLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &Loops = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return SIMaterializeWMMAScale().run(MF, Loops);
}

PreservedAnalyses
SIMaterializeWMMAScalePass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (MF.getFunction().hasOptNone())
    return PreservedAnalyses::all();

  auto &Loops = MFAM.getResult<MachineLoopAnalysis>(MF);
  if (!SIMaterializeWMMAScale().run(MF, Loops))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineLoopAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
