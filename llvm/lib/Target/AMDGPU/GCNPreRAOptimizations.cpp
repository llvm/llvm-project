//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  bool processReg(Register Reg);
  void hintTrue16Copy(const MachineInstr &MI);
  bool optimizeBVHStack(MachineInstr &MI);
  bool optimizeSGPRToVGPRCopy(MachineInstr &MI);
  bool narrowTupleCopy(MachineInstr &MI);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

FunctionPass *llvm::createGCNPreRAOptimizationsLegacyPass() {
  return new GCNPreRAOptimizationsLegacy();
}

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

void GCNPreRAOptimizationsImpl::hintTrue16Copy(const MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const TargetRegisterClass *DstRC = TRI->getRegClassForReg(*MRI, Dst);
  bool IsDst16Bit = AMDGPU::VGPR_16RegClass.hasSubClassEq(DstRC);
  if (Dst.isVirtual() && IsDst16Bit && Src.isPhysical() &&
      TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
  if (Src.isVirtual() && MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
      Dst.isPhysical() && DstRC == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
  if (!Dst.isVirtual() || !Src.isVirtual())
    return;
  if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
      MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
    MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
  }
  if (IsDst16Bit && MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
}

bool GCNPreRAOptimizationsImpl::optimizeBVHStack(MachineInstr &MI) {
  SmallVector<Register, 2> UseRegs;

  // Find BVH sources for this DS_BVH_STACK instruction.
  auto CheckUse = [&](MachineOperand &Use) {
    Register Reg = Use.getReg();
    for (const MachineInstr &Src : MRI->def_instructions(Reg)) {
      if (!SIInstrInfo::isImage(Src))
        continue;
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Src.getOpcode());
      const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
          AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
      if (!BaseInfo->BVH)
        continue;
      UseRegs.push_back(Reg);
      break;
    }
  };
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data0));
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data1));

  if (UseRegs.empty())
    return false;

  // Add implicit uses for entire BVH source registers.
  // This avoids partial reallocation of register which could
  // introduce a premature s_wait_bvhcnt.
  for (Register Reg : UseRegs) {
    MI.addOperand(MachineOperand::CreateReg(Reg, false, true));
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }
  LLVM_DEBUG(dbgs() << "Added implicit uses to: " << MI);

  return true;
}

// A scalar move of an immediate, which has a VALU equivalent of the same width.
static bool isMovImm(const MachineInstr &Def) {
  switch (Def.getOpcode()) {
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::S_MOV_B64_IMM_PSEUDO:
    return Def.getNumOperands() == 2 && Def.getOperand(1).isImm();
  default:
    return false;
  }
}

// A def of a tuple subregister that can be redone directly in a VGPR: an
// immediate move or a copy of a 32 bit virtual register.
static bool isRewritableTupleDef(const MachineInstr &Def, Register Tuple,
                                 const SIRegisterInfo *TRI,
                                 const MachineRegisterInfo *MRI) {
  if (isMovImm(Def))
    return true;
  switch (Def.getOpcode()) {
  case AMDGPU::COPY: {
    const MachineOperand &CopySrc = Def.getOperand(1);
    if (!CopySrc.isReg())
      return false;
    Register CopySrcReg = CopySrc.getReg();
    if (CopySrcReg == Tuple)
      return true;
    if (!CopySrcReg.isVirtual())
      return false;
    return (CopySrc.getSubReg()
                ? TRI->getSubRegIdxSize(CopySrc.getSubReg())
                : TRI->getRegSizeInBits(CopySrcReg, *MRI)) == 32;
  }
  default:
    return false;
  }
}

// Build a register tuple directly in VGPRs instead of building it in SGPRs and
// copying the whole tuple:
//
//   undef %0.sub0:sgpr_256 = S_MOV_B32 1
//   %0.sub1:sgpr_256 = COPY %0.sub0
//   ...
//   %1:vreg_256 = COPY %0:sgpr_256
// =>
//   undef %1.sub0:vreg_256 = V_MOV_B32_e32 1, implicit $exec
//   %1.sub1:vreg_256 = COPY %1.sub0
//   ...
//
// Runs after SIFoldOperands and the coalescer, so any use that could have taken
// the SGPR tuple has already folded or coalesced it.
bool GCNPreRAOptimizationsImpl::optimizeSGPRToVGPRCopy(MachineInstr &MI) {
  const MachineOperand &DstMO = MI.getOperand(0);
  const MachineOperand &SrcMO = MI.getOperand(1);
  if (DstMO.getSubReg() || SrcMO.getSubReg())
    return false;

  Register Dst = DstMO.getReg();
  Register Src = SrcMO.getReg();
  if (!Dst.isVirtual() || !Src.isVirtual())
    return false;

  const TargetRegisterClass *DstRC = MRI->getRegClass(Dst);
  const TargetRegisterClass *SrcRC = MRI->getRegClass(Src);
  if (!TRI->isVGPRClass(DstRC) || !TRI->isSGPRClass(SrcRC) ||
      DstRC->getSizeInBits() != SrcRC->getSizeInBits() ||
      DstRC->getSizeInBits() <= 32)
    return false;

  // The tuple must be dead after this copy, otherwise the SGPR value is still
  // needed and rewriting the defs only adds copies. Uses inside the tuple
  // initialization itself are rewritten together with the defs.
  for (MachineInstr &Use : MRI->use_nodbg_instructions(Src)) {
    if (&Use == &MI)
      continue;
    if (Use.isCopy() && Use.getOperand(0).getReg() == Src)
      continue;
    return false;
  }

  // Nothing of Dst may be live before the copy, the defs are rewritten in
  // place and would clobber it.
  if (!LIS->hasInterval(Dst) || !LIS->hasInterval(Src))
    return false;

  SlotIndex CopyIdx = LIS->getInstructionIndex(MI);
  if (LIS->getInterval(Dst).beginIndex() != CopyIdx.getRegSlot())
    return false;

  // All defs must be 32 bit subregister defs in this block, before the copy.
  // Immediates and copies are redone in the VGPR tuple, anything else stays
  // where it is and its result is copied into the tuple.
  // A wide copy between properly aligned register classes is expanded into 64
  // bit moves, splitting it into 32 bit pieces would double the move count.
  // Without that the copy already costs one move per element, so building the
  // tuple in VGPRs is never worse and gets rid of the SGPR tuple.
  bool CopyIsPaired = DstRC->getSizeInBits() % 64 == 0 &&
                      TRI->isProperlyAlignedRC(*DstRC) &&
                      (ST->hasVMovB64Inst() || ST->hasPkMovB32());

  SmallVector<MachineInstr *, 8> Defs;
  bool AnyRewritten = !CopyIsPaired;
  for (MachineOperand &DefMO : MRI->def_operands(Src)) {
    MachineInstr &Def = *DefMO.getParent();
    if (Def.getParent() != MI.getParent() ||
        !SlotIndex::isEarlierInstr(LIS->getInstructionIndex(Def), CopyIdx))
      return false;

    unsigned SubIdx = DefMO.getSubReg();
    if (!SubIdx || DefMO.isTied() || is_contained(Defs, &Def))
      return false;

    // A 64 bit immediate move is redone as a single V_MOV_B64, and a 64 bit
    // copy inside the tuple stays a single move, so both are worth splitting
    // even though that half of the wide copy costs one move either way.
    unsigned SubSize = TRI->getSubRegIdxSize(SubIdx);
    bool IsMovImm = isMovImm(Def) &&
                    (SubSize == 32 ||
                     TII->isImmOperandLegal(TII->get(AMDGPU::V_MOV_B64_PSEUDO),
                                            1, Def.getOperand(1)));
    bool IsTupleCopy = Def.isCopy() && Def.getOperand(1).getReg() == Src;
    if (SubSize != 32 &&
        (SubSize != 64 || !(ST->hasVMovB64Inst() || ST->hasPkMovB32()) ||
         !(IsMovImm || IsTupleCopy)))
      return false;

    // A def that is redone in the VGPR tuple goes away, so the wide copy pays
    // for itself. Everything else at best trades it for a narrow one.
    bool Rewritable = isRewritableTupleDef(Def, Src, TRI, MRI);
    if (Rewritable)
      AnyRewritten = true;
    if (!Rewritable && !TRI->getSubRegisterClass(SrcRC, SubIdx))
      return false;

    Defs.push_back(&Def);
  }

  // Splitting the copy without getting rid of at least one of the defs just
  // trades one wide copy for a series of narrow ones.
  if (!AnyRewritten)
    return false;

  // The tuple is rebuilt at the copy, keep the defs in their original order.
  sort(Defs, [this](const MachineInstr *A, const MachineInstr *B) {
    return SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*A),
                                     LIS->getInstructionIndex(*B));
  });

  LLVM_DEBUG(dbgs() << "Building tuple in VGPRs for: " << MI);

  SmallVector<Register, 8> NewVRegs;
  SmallSetVector<Register, 8> Recompute;
  for (MachineInstr *Def : Defs) {
    MachineOperand &DefMO = *Def->findRegisterDefOperand(Src, TRI);
    unsigned SubIdx = DefMO.getSubReg();
    RegState UndefFlag = getUndefRegState(DefMO.isUndef());
    bool KeepDef = !isRewritableTupleDef(*Def, Src, TRI, MRI);
    unsigned Opc = AMDGPU::COPY;
    if (!KeepDef) {
      if (Def->getOpcode() == AMDGPU::S_MOV_B32)
        Opc = AMDGPU::V_MOV_B32_e32;
      else if (isMovImm(*Def))
        Opc = AMDGPU::V_MOV_B64_PSEUDO;
    }

    // Anything that is not redone in the VGPR tuple stays at its original slot
    // index: moving it could extend the live range of a physical register, and
    // it may not be safe to move at all. Redefine it into a plain 32 bit
    // virtual register in place and copy that into the tuple instead.
    MachineOperand DefSrcMO = Def->getOperand(1);
    DefSrcMO.clearParent();
    if (KeepDef) {
      Register Tmp =
          MRI->createVirtualRegister(TRI->getSubRegisterClass(SrcRC, SubIdx));
      DefMO.setReg(Tmp);
      DefMO.setSubReg(0);
      DefMO.setIsUndef(false);
      NewVRegs.push_back(Tmp);
      DefSrcMO = MachineOperand::CreateReg(Tmp, false);
    } else if (DefSrcMO.isReg()) {
      if (DefSrcMO.getReg() == Src)
        DefSrcMO.setReg(Dst);
      else if (DefSrcMO.getReg().isVirtual())
        // Moving the def to the copy extends the live range of what it reads.
        Recompute.insert(DefSrcMO.getReg());
    }

    auto *NewDef =
        BuildMI(*MI.getParent(), MI, Def->getDebugLoc(), TII->get(Opc))
            .addDef(Dst, UndefFlag, SubIdx)
            .add(DefSrcMO)
            .getInstr();
    if (!KeepDef) {
      LIS->RemoveMachineInstrFromMaps(*Def);
      Def->eraseFromParent();
    }
    LIS->InsertMachineInstrInMaps(*NewDef);
    LLVM_DEBUG(dbgs() << "  " << *NewDef);
  }

  LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();

  LIS->removeInterval(Src);
  LIS->removeInterval(Dst);
  LIS->createAndComputeVirtRegInterval(Dst);
  for (Register Tmp : NewVRegs)
    LIS->createAndComputeVirtRegInterval(Tmp);
  for (Register Reg : Recompute) {
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }

  return true;
}

// Copying a whole tuple when the uses only read some of its subregisters
// writes the rest for nothing. Copy only the subregisters that are read:
//
//   %1:vreg_64 = COPY %0:sreg_64
//   ... = %1.sub1
// =>
//   %2:vgpr_32 = COPY %0.sub1:sreg_64
//   ... = %2
bool GCNPreRAOptimizationsImpl::narrowTupleCopy(MachineInstr &MI) {
  const MachineOperand &DstMO = MI.getOperand(0);
  const MachineOperand &SrcMO = MI.getOperand(1);
  if (DstMO.getSubReg() || SrcMO.getSubReg())
    return false;

  Register Dst = DstMO.getReg();
  Register Src = SrcMO.getReg();
  if (!Dst.isVirtual() || !Src.isVirtual() || !MRI->hasOneDef(Dst))
    return false;

  const TargetRegisterClass *DstRC = MRI->getRegClass(Dst);
  const TargetRegisterClass *SrcRC = MRI->getRegClass(Src);
  unsigned Size = DstRC->getSizeInBits();
  if (Size <= 32 || Size != SrcRC->getSizeInBits())
    return false;

  SmallSetVector<unsigned, 8> SubRegs;
  for (const MachineOperand &UseMO : MRI->use_nodbg_operands(Dst)) {
    unsigned SubIdx = UseMO.getSubReg();
    if (!SubIdx || TRI->getSubRegIdxSize(SubIdx) != 32 ||
        !TRI->getSubRegisterClass(DstRC, SubIdx) ||
        !TRI->getSubRegisterClass(SrcRC, SubIdx))
      return false;
    SubRegs.insert(SubIdx);
  }

  if (SubRegs.empty() || SubRegs.size() * 32 >= Size)
    return false;

  LLVM_DEBUG(dbgs() << "Narrowing tuple copy: " << MI);

  SmallVector<Register, 8> NewVRegs;
  for (unsigned SubIdx : SubRegs) {
    Register NewReg =
        MRI->createVirtualRegister(TRI->getSubRegisterClass(DstRC, SubIdx));
    auto *NewCopy = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                            TII->get(AMDGPU::COPY), NewReg)
                        .addReg(Src, RegState::NoFlags, SubIdx)
                        .getInstr();
    LIS->InsertMachineInstrInMaps(*NewCopy);
    NewVRegs.push_back(NewReg);
    LLVM_DEBUG(dbgs() << "  " << *NewCopy);

    for (MachineOperand &UseMO :
         make_early_inc_range(MRI->use_nodbg_operands(Dst))) {
      if (UseMO.getSubReg() != SubIdx)
        continue;
      UseMO.setReg(NewReg);
      UseMO.setSubReg(AMDGPU::NoSubRegister);
    }
  }

  MRI->markUsesInDebugValueAsUndef(Dst);
  LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();

  LIS->removeInterval(Src);
  LIS->removeInterval(Dst);
  LIS->createAndComputeVirtRegInterval(Src);
  for (Register NewReg : NewVRegs)
    LIS->createAndComputeVirtRegInterval(NewReg);

  return true;
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  this->ST = &ST;
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isCopy())
        Changed |= optimizeSGPRToVGPRCopy(MI) || narrowTupleCopy(MI);
    }
  }

  const bool HasBVHStack = ST.hasBVHDualAndBVH8Insts();
  const bool HasRealTrue16 = ST.useRealTrue16Insts();

  if (!HasRealTrue16 && !HasBVHStack)
    return Changed;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Add RA hints to improve True16 COPY elimination.
      if (HasRealTrue16 && MI.getOpcode() == AMDGPU::COPY) {
        hintTrue16Copy(MI);
        continue;
      }
      // Add implicit uses to avoid early wait on intersect ray instructions.
      if (HasBVHStack &&
          (MI.getOpcode() == AMDGPU::DS_BVH_STACK_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP1_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP2_RTN_B64)) {
        Changed |= optimizeBVHStack(MI);
        continue;
      }
    }
  }

  return Changed;
}
