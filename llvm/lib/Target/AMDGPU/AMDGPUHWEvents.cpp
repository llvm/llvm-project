//===- AMDGPUHWEvents.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHWEvents.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace AMDGPU {
void HWEventSet::print(raw_ostream &OS) const {
  ListSeparator LS(", ");
  for (HWEvent Event : hw_events()) {
    if (contains(Event))
      OS << LS << toString(Event);
  }
}

void HWEventSet::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

static std::optional<HWEvent>
getExpertSchedulingEventType(const MachineInstr &Inst, const SIInstrInfo &TII) {
  if (TII.isVALU(Inst)) {
    // Core/Side-, DP-, XDL- and TRANS-MACC VALU instructions complete
    // out-of-order with respect to each other, so each of these classes
    // has its own event.

    if (TII.isXDL(Inst))
      return HWEvent::VGPR_XDL_WRITE;

    if (TII.isTRANS(Inst))
      return HWEvent::VGPR_TRANS_WRITE;

    if (AMDGPU::isDPMACCInstruction(Inst.getOpcode()))
      return HWEvent::VGPR_DPMACC_WRITE;

    return HWEvent::VGPR_CSMACC_WRITE;
  }

  // FLAT and LDS instructions may read their VGPR sources out-of-order
  // with respect to each other and all other VMEM instructions, so
  // each of these also has a separate event.

  if (TII.isFLAT(Inst))
    return HWEvent::VGPR_FLAT_READ;

  if (TII.isDS(Inst))
    return HWEvent::VGPR_LDS_READ;

  if (TII.isVMEM(Inst) || TII.isVIMAGE(Inst) || TII.isVSAMPLE(Inst))
    return HWEvent::VGPR_VMEM_READ;

  // Otherwise, no hazard.

  return {};
}

static HWEvent getVmemHWEvent(const MachineInstr &Inst, const GCNSubtarget &ST,
                              const SIInstrInfo &TII) {
  switch (Inst.getOpcode()) {
  // FIXME: GLOBAL_INV needs to be tracked with xcnt too.
  case AMDGPU::GLOBAL_INV:
    return HWEvent::GLOBAL_INV_ACCESS; // tracked using loadcnt, but doesn't
                                       // write VGPRs
  case AMDGPU::GLOBAL_WB:
  case AMDGPU::GLOBAL_WBINV:
    return HWEvent::VMEM_WRITE_ACCESS; // tracked using storecnt
  default:
    break;
  }

  assert(SIInstrInfo::isVMEM(Inst));
  // LDS DMA loads are also stores, but on the LDS side. On the VMEM side
  // these should use VM_CNT.
  if (!ST.hasVscnt() || SIInstrInfo::mayWriteLDSThroughDMA(Inst))
    return HWEvent::VMEM_ACCESS;
  if (Inst.mayStore() &&
      (!Inst.mayLoad() || SIInstrInfo::isAtomicNoRet(Inst))) {
    if (TII.mayAccessScratch(Inst))
      return HWEvent::SCRATCH_WRITE_ACCESS;
    return HWEvent::VMEM_WRITE_ACCESS;
  }
  if (!ST.hasExtendedWaitCounts() || SIInstrInfo::isFLAT(Inst))
    return HWEvent::VMEM_ACCESS;

  if (SIInstrInfo::isImage(Inst)) {
    const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
    const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
        AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);

    if (BaseInfo->BVH)
      return HWEvent::VMEM_BVH_READ_ACCESS;

    // We have to make an additional check for isVSAMPLE here since some
    // instructions don't have a sampler, but are still classified as sampler
    // instructions for the purposes of e.g. waitcnt.
    if (BaseInfo->Sampler || BaseInfo->MSAA || SIInstrInfo::isVSAMPLE(Inst))
      return HWEvent::VMEM_SAMPLER_READ_ACCESS;
  }

  return HWEvent::VMEM_ACCESS;
}

HWEventSet getEventsFor(const MachineInstr &Inst, const GCNSubtarget &ST,
                        bool IsExpertMode) {
  const SIInstrInfo &TII = *ST.getInstrInfo();

  HWEventSet Events;
  if (IsExpertMode) {
    if (const auto ET = getExpertSchedulingEventType(Inst, TII))
      Events.insert(*ET);
  }

  if (TII.isDS(Inst) && TII.usesLGKM_CNT(Inst)) {
    if (TII.isAlwaysGDS(Inst.getOpcode()) ||
        TII.hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      Events.insert(HWEvent::GDS_ACCESS);
      Events.insert(HWEvent::GDS_GPR_LOCK);
    } else {
      Events.insert(HWEvent::LDS_ACCESS);
    }
  } else if (TII.isFLAT(Inst)) {
    if (SIInstrInfo::isGFX12CacheInvOrWBInst(Inst.getOpcode())) {
      Events.insert(getVmemHWEvent(Inst, ST, TII));
    } else {
      assert(Inst.mayLoadOrStore());
      if (TII.mayAccessVMEMThroughFlat(Inst)) {
        if (ST.hasWaitXcnt())
          Events.insert(HWEvent::VMEM_GROUP);
        Events.insert(getVmemHWEvent(Inst, ST, TII));
      }
      if (TII.mayAccessLDSThroughFlat(Inst))
        Events.insert(HWEvent::LDS_ACCESS);
    }
  } else if (SIInstrInfo::isVMEM(Inst) &&
             (!AMDGPU::getMUBUFIsBufferInv(Inst.getOpcode()) ||
              Inst.getOpcode() == AMDGPU::BUFFER_WBL2)) {
    // BUFFER_WBL2 is included here because unlike invalidates, has to be
    // followed "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback has
    // completed.
    if (ST.hasWaitXcnt())
      Events.insert(HWEvent::VMEM_GROUP);
    Events.insert(getVmemHWEvent(Inst, ST, TII));
    if (ST.vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst))) {
      Events.insert(HWEvent::VMW_GPR_LOCK);
    }
  } else if (TII.isSMRD(Inst)) {
    if (ST.hasWaitXcnt())
      Events.insert(HWEvent::SMEM_GROUP);
    Events.insert(HWEvent::SMEM_ACCESS);
  } else if (SIInstrInfo::isLDSDIR(Inst)) {
    Events.insert(HWEvent::EXP_LDS_ACCESS);
  } else if (SIInstrInfo::isEXP(Inst)) {
    unsigned Imm = TII.getNamedOperand(Inst, AMDGPU::OpName::tgt)->getImm();
    if (Imm >= AMDGPU::Exp::ET_PARAM0 && Imm <= AMDGPU::Exp::ET_PARAM31)
      Events.insert(HWEvent::EXP_PARAM_ACCESS);
    else if (Imm >= AMDGPU::Exp::ET_POS0 && Imm <= AMDGPU::Exp::ET_POS_LAST)
      Events.insert(HWEvent::EXP_POS_ACCESS);
    else
      Events.insert(HWEvent::EXP_GPR_LOCK);
  } else if (SIInstrInfo::isSBarrierSCCWrite(Inst.getOpcode())) {
    Events.insert(HWEvent::SCC_WRITE);
  } else {
    switch (Inst.getOpcode()) {
    case AMDGPU::S_SENDMSG:
    case AMDGPU::S_SENDMSG_RTN_B32:
    case AMDGPU::S_SENDMSG_RTN_B64:
    case AMDGPU::S_SENDMSGHALT:
      Events.insert(HWEvent::SQ_MESSAGE);
      break;
    case AMDGPU::S_MEMTIME:
    case AMDGPU::S_MEMREALTIME:
    case AMDGPU::S_GET_BARRIER_STATE_M0:
    case AMDGPU::S_GET_BARRIER_STATE_IMM:
      Events.insert(HWEvent::SMEM_ACCESS);
      break;
    }
  }
  return Events;
}
} // namespace AMDGPU
} // namespace llvm
