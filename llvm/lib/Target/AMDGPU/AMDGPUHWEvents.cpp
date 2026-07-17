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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void HWEvents::dump() const { dbgs() << *this << "\n"; }
#endif

static HWEvents getExpertSchedulingEventType(const MachineInstr &Inst,
                                             const SIInstrInfo &TII) {
  if (TII.isVALU(Inst, /*AllowLDSDMA=*/false)) {
    // Core/Side-, DP-, XDL- and TRANS-MACC VALU instructions complete
    // out-of-order with respect to each other, so each of these classes
    // has its own event.

    if (TII.isXDL(Inst))
      return HWEvents::VGPR_XDL_READ | HWEvents::VGPR_XDL_WRITE;

    if (TII.isTRANS(Inst))
      return HWEvents::VGPR_TRANS_READ | HWEvents::VGPR_TRANS_WRITE;

    if (AMDGPU::isDPMACCInstruction(Inst.getOpcode()))
      return HWEvents::VGPR_DPMACC_READ | HWEvents::VGPR_DPMACC_WRITE;

    return HWEvents::VGPR_CSMACC_READ | HWEvents::VGPR_CSMACC_WRITE;
  }

  // FLAT and LDS instructions may read their VGPR sources out-of-order
  // with respect to each other and all other VMEM instructions, so
  // each of these also has a separate event.

  if (TII.isFLAT(Inst))
    return HWEvents::VGPR_FLAT_READ;

  if (TII.isDS(Inst))
    return HWEvents::VGPR_LDS_READ;

  if (TII.isVMEM(Inst) || TII.isVIMAGE(Inst) || TII.isVSAMPLE(Inst))
    return HWEvents::VGPR_VMEM_READ;

  // Otherwise, no hazard.
  return HWEvents::NONE;
}

HWEvents getSimplifiedVMEMEventsFor(const MachineInstr &Inst,
                                    const SIInstrInfo &TII) {
  switch (Inst.getOpcode()) {
  // FIXME: GLOBAL_INV needs to be tracked with xcnt too.
  case AMDGPU::GLOBAL_INV:
    return HWEvents::GLOBAL_INV_ACCESS; // tracked using loadcnt, but doesn't
                                        // write VGPRs
  case AMDGPU::GLOBAL_WB:
  case AMDGPU::GLOBAL_WBINV:
    return HWEvents::VMEM_WRITE_ACCESS; // tracked using storecnt
  default:
    break;
  }

  assert(SIInstrInfo::isVMEM(Inst));
  // LDS DMA loads are also stores, but on the LDS side. On the VMEM side
  // these should use VM_CNT.
  if (SIInstrInfo::mayWriteLDSThroughDMA(Inst))
    return HWEvents::VMEM_READ_ACCESS;

  if (Inst.mayStore() &&
      (!Inst.mayLoad() || SIInstrInfo::isAtomicNoRet(Inst))) {
    if (TII.mayAccessScratch(Inst))
      return HWEvents::SCRATCH_WRITE_ACCESS;
    return HWEvents::VMEM_WRITE_ACCESS;
  }

  if (SIInstrInfo::isFLAT(Inst))
    return HWEvents::VMEM_READ_ACCESS;

  if (SIInstrInfo::isImage(Inst)) {
    const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
    const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
        AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);

    if (BaseInfo->BVH)
      return HWEvents::VMEM_BVH_READ_ACCESS;

    // We have to make an additional check for isVSAMPLE here since some
    // instructions don't have a sampler, but are still classified as sampler
    // instructions for the purposes of e.g. waitcnt.
    if (BaseInfo->Sampler || BaseInfo->MSAA || SIInstrInfo::isVSAMPLE(Inst))
      return HWEvents::VMEM_SAMPLER_READ_ACCESS;
  }

  return HWEvents::VMEM_READ_ACCESS;
}

static HWEvents getEventsForImpl(const MachineInstr &Inst,
                                 const GCNSubtarget &ST, const SIInstrInfo &TII,
                                 bool TgSplit) {
  if (TII.isDS(Inst) && TII.usesLGKM_CNT(Inst)) {
    if (TII.isAlwaysGDS(Inst.getOpcode()) ||
        TII.hasModifiersSet(Inst, AMDGPU::OpName::gds))
      return HWEvents::GDS_ACCESS | HWEvents::GDS_GPR_LOCK;

    return HWEvents::LDS_ACCESS;
  }

  if (TII.isFLAT(Inst)) {
    if (SIInstrInfo::isGFX12CacheInvOrWBInst(Inst.getOpcode()))
      return getSimplifiedVMEMEventsFor(Inst, TII);

    assert(Inst.mayLoadOrStore());
    HWEvents E = HWEvents::NONE;
    if (TII.mayAccessVMEMThroughFlat(Inst)) {
      if (ST.hasWaitXcnt())
        E |= HWEvents::VMEM_GROUP;
      E |= getSimplifiedVMEMEventsFor(Inst, TII);
    }

    if (TII.mayAccessLDSThroughFlat(Inst, TgSplit))
      E |= HWEvents::LDS_ACCESS;

    if (SIInstrInfo::usesASYNC_CNT(Inst))
      E |= HWEvents::ASYNC_ACCESS;

    return E;
  }

  if (SIInstrInfo::usesTENSOR_CNT(Inst))
    return HWEvents::TENSOR_ACCESS;

  if (SIInstrInfo::isVMEM(Inst) &&
      (!AMDGPU::getMUBUFIsBufferInv(Inst.getOpcode()) ||
       Inst.getOpcode() == AMDGPU::BUFFER_WBL2)) {
    // BUFFER_WBL2 is included here because unlike invalidates, has to be
    // followed "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback has
    // completed.
    HWEvents E = getSimplifiedVMEMEventsFor(Inst, TII);
    if (ST.hasWaitXcnt())
      E |= HWEvents::VMEM_GROUP;
    if (ST.vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst)))
      E |= HWEvents::VMW_GPR_LOCK;

    return E;
  }

  if (TII.isSMRD(Inst)) {
    if (ST.hasWaitXcnt())
      return HWEvents::SMEM_GROUP | HWEvents::SMEM_ACCESS;
    return HWEvents::SMEM_ACCESS;
  }

  if (SIInstrInfo::isLDSDIR(Inst)) {
    return HWEvents::EXP_LDS_ACCESS;
  }

  if (SIInstrInfo::isEXP(Inst)) {
    unsigned Imm = TII.getNamedOperand(Inst, AMDGPU::OpName::tgt)->getImm();
    if (Imm >= AMDGPU::Exp::ET_PARAM0 && Imm <= AMDGPU::Exp::ET_PARAM31)
      return HWEvents::EXP_PARAM_ACCESS;
    if (Imm >= AMDGPU::Exp::ET_POS0 && Imm <= AMDGPU::Exp::ET_POS_LAST)
      return HWEvents::EXP_POS_ACCESS;
    return HWEvents::EXP_GPR_LOCK;
  }

  if (SIInstrInfo::isSBarrierSCCWrite(Inst.getOpcode())) {
    return HWEvents::SCC_WRITE;
  }

  switch (Inst.getOpcode()) {
  case AMDGPU::S_SENDMSG:
  case AMDGPU::S_SENDMSG_RTN_B32:
  case AMDGPU::S_SENDMSG_RTN_B64:
  case AMDGPU::S_SENDMSGHALT:
    return HWEvents::SQ_MESSAGE;
  case AMDGPU::S_MEMTIME:
  case AMDGPU::S_MEMREALTIME:
  case AMDGPU::S_GET_BARRIER_STATE_M0:
  case AMDGPU::S_GET_BARRIER_STATE_IMM:
    return HWEvents::SMEM_ACCESS;
  }

  return HWEvents::NONE;
}

HWEvents getEventsFor(const MachineInstr &Inst, const GCNSubtarget &ST,
                      bool IsExpertMode, bool TgSplit) {
  const SIInstrInfo &TII = *ST.getInstrInfo();

  if (IsExpertMode)
    return getEventsForImpl(Inst, ST, TII, TgSplit) |
           getExpertSchedulingEventType(Inst, TII);
  return getEventsForImpl(Inst, ST, TII, TgSplit);
}
} // namespace AMDGPU

raw_ostream &operator<<(raw_ostream &OS, AMDGPU::HWEvents Events) {
  ListSeparator LS(" | ");
#define AMDGPU_HW_EVENT(E, V)                                                  \
  if (Events & AMDGPU::HWEvents::E)                                            \
    OS << LS << #E << " ";
#include "AMDGPUHWEvents.def"
  return OS;
}

} // namespace llvm
