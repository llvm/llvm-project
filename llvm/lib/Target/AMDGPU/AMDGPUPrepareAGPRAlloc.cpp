//===-- AMDGPUPrepareAGPRAlloc.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make simple transformations to relax register constraints for cases which can
// allocate to AGPRs or VGPRs. Replace materialize of inline immediates into
// AGPR or VGPR with a pseudo with an AV_* class register constraint. This
// allows later passes to inflate the register class if necessary. The register
// allocator does not know to replace instructions to relax constraints.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPrepareAGPRAlloc.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "amdgpu-prepare-agpr-alloc"

namespace {

class AMDGPUPrepareAGPRAllocImpl {
private:
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;

  bool isAV64Imm(const MachineOperand &MO) const;
  bool hoistCopiesOverCall(MachineBasicBlock &MBB);

public:
  AMDGPUPrepareAGPRAllocImpl(const GCNSubtarget &ST, MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()), MRI(MRI) {}
  bool run(MachineFunction &MF);
};

class AMDGPUPrepareAGPRAllocLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrepareAGPRAllocLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Prepare AGPR Alloc"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrepareAGPRAllocLegacy, DEBUG_TYPE,
                "AMDGPU Prepare AGPR Alloc", false, false)

char AMDGPUPrepareAGPRAllocLegacy::ID = 0;

char &llvm::AMDGPUPrepareAGPRAllocLegacyID = AMDGPUPrepareAGPRAllocLegacy::ID;

bool AMDGPUPrepareAGPRAllocLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  return AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo()).run(MF);
}

PreservedAnalyses
AMDGPUPrepareAGPRAllocPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo()).run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUPrepareAGPRAllocImpl::isAV64Imm(const MachineOperand &MO) const {
  return MO.isImm() && TII.isLegalAV64PseudoImm(MO.getImm());
}

/// \returns true if \p Mask leaves at least one register of \p RC alive.
static bool preservesAnyOf(const uint32_t *Mask,
                           const TargetRegisterClass &RC) {
  return any_of(RC, [Mask](MCPhysReg Reg) {
    return !MachineOperand::clobbersPhysReg(Mask, Reg);
  });
}

/// Move reads of AGPR values above the calls that would otherwise clobber them.
///
/// A value that stays in AGPRs across a call the AGPRs do not survive has to be
/// read out one register at a time by the allocator, and by then the reads are
/// just unrelated 32-bit values: nothing records that some of them have to land
/// in adjacent registers to serve a later wide read. The allocator picks
/// whatever is free, and the wide read turns into a second round of moves to
/// gather them. Reading the value before the call instead leaves a live range
/// in a class the call does preserve, and the tuple it has to form is known
/// while the allocator is still choosing registers for it.
///
/// A read is moved above the first such call that follows the definition of the
/// value it reads, so the value crosses no call at all.
bool AMDGPUPrepareAGPRAllocImpl::hoistCopiesOverCall(MachineBasicBlock &MBB) {
  // Every call that discards the whole of the AGPR file while keeping part of
  // the VGPR file, each with the reads to move above it.
  SmallVector<std::pair<MachineInstr *, SmallVector<MachineInstr *, 8>>, 4>
      Calls;
  // How many of those calls had been passed when a value was defined. The read
  // of that value belongs above the call at that index, the first one it would
  // otherwise cross.
  DenseMap<Register, unsigned> FirstCallAfterDef;
  // A read is lane masked, so it cannot move above a point where the active
  // lanes change. Calls before the most recent such point are not targets.
  unsigned Reachable = 0;

  for (MachineInstr &MI : MBB) {
    // A call's mask names EXEC among what it does not preserve, but a callee
    // restores the lane mask it was entered with, so a call is not a point
    // where the active lanes change.
    if (!MI.isCall() && MI.modifiesRegister(AMDGPU::EXEC, &TRI))
      Reachable = Calls.size();

    if (MI.isCall()) {
      // A call without a register mask says nothing about what it keeps.
      const uint32_t *Mask = nullptr;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isRegMask()) {
          Mask = MO.getRegMask();
          break;
        }
      }

      // Only worth moving a read out of a file the call discards into one it
      // keeps. AGPRs are callee saved from gfx90a on, where there is nothing
      // to gain.
      if (Mask && !preservesAnyOf(Mask, AMDGPU::AGPR_32RegClass) &&
          preservesAnyOf(Mask, AMDGPU::VGPR_32RegClass))
        Calls.emplace_back(&MI, SmallVector<MachineInstr *, 8>());
    }

    // Only AGPR values are ever worth reading early, so only those need to
    // record where they were defined. Recorded after the call above is
    // counted, so that a value a call returns cannot be read before the call
    // that produces it.
    //
    // Not MI.defs(): an inline asm declares no defs in its description and
    // writes its results through variadic operands.
    for (const MachineOperand &MO : MI.all_defs()) {
      Register Reg = MO.getReg();
      if (!Reg.isVirtual())
        continue;
      const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
      if (RC && TRI.isAGPRClass(RC))
        FirstCallAfterDef.try_emplace(Reg, Calls.size());
    }

    if (!MI.isCopy())
      continue;

    const MachineOperand &Dst = MI.getOperand(0);
    const MachineOperand &Src = MI.getOperand(1);
    if (!Dst.getReg().isVirtual() || !Src.getReg().isVirtual() ||
        Dst.getSubReg())
      continue;

    // Reading the value early has to read the same thing, and the value it
    // lands in must not be written anywhere else.
    if (!MRI.hasOneDef(Src.getReg()) || !MRI.hasOneDef(Dst.getReg()))
      continue;

    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src.getReg());
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst.getReg());
    if (!TRI.isAGPRClass(SrcRC) || !TRI.hasVGPRs(DstRC) || TRI.hasAGPRs(DstRC))
      continue;

    auto Def = FirstCallAfterDef.find(Src.getReg());
    if (Def == FirstCallAfterDef.end())
      continue;

    unsigned Target = std::max(Def->second, Reachable);
    if (Target >= Calls.size())
      continue;

    LLVM_DEBUG(dbgs() << "  moving " << MI << "  above "
                      << *Calls[Target].first);
    Calls[Target].second.push_back(&MI);
  }

  bool Changed = false;
  for (auto &[Call, Reads] : Calls) {
    // Reading more of a value than it holds means the reads overlap, and
    // moving them all up would raise the pressure across the call rather than
    // shift it from one file to the other. Leave such a value where it is.
    DenseMap<Register, unsigned> ReadBits;
    for (MachineInstr *MI : Reads) {
      ReadBits[MI->getOperand(1).getReg()] +=
          TRI.getRegSizeInBits(*MRI.getRegClass(MI->getOperand(0).getReg()));
    }

    for (MachineInstr *MI : Reads) {
      Register Src = MI->getOperand(1).getReg();
      if (ReadBits.lookup(Src) > TRI.getRegSizeInBits(*MRI.getRegClass(Src)))
        continue;
      MBB.splice(Call->getIterator(), &MBB, MI->getIterator());
      Changed = true;
    }
  }

  return Changed;
}

bool AMDGPUPrepareAGPRAllocImpl::run(MachineFunction &MF) {
  if (MRI.isReserved(AMDGPU::AGPR0))
    return false;

  const MCInstrDesc &AVImmPseudo32 = TII.get(AMDGPU::AV_MOV_B32_IMM_PSEUDO);
  const MCInstrDesc &AVImmPseudo64 = TII.get(AMDGPU::AV_MOV_B64_IMM_PSEUDO);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    Changed |= hoistCopiesOverCall(MBB);

    for (MachineInstr &MI : MBB) {
      if ((MI.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
           TII.isInlineConstant(MI, 1)) ||
          (MI.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64 &&
           MI.getOperand(1).isImm())) {
        MI.setDesc(AVImmPseudo32);
        Changed = true;
        continue;
      }

      // TODO: If only half of the value is rewritable, is it worth splitting it
      // up?
      if ((MI.getOpcode() == AMDGPU::V_MOV_B64_e64 ||
           MI.getOpcode() == AMDGPU::V_MOV_B64_PSEUDO) &&
          isAV64Imm(MI.getOperand(1))) {
        MI.setDesc(AVImmPseudo64);
        Changed = true;
        continue;
      }
    }
  }

  return Changed;
}
