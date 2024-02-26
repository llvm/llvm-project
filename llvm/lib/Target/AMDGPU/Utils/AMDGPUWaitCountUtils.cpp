//===-- AMDGPUWaitCountUtils.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Common interface to insert various wait counts for memory operations.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitCountUtils.h"
#include "AMDGPU.h"
#include "AMDGPUBaseInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "amdgpu-waitcount-utils"

using namespace llvm;
using namespace llvm::AMDGPU;
namespace llvm {

namespace AMDGPU {

static bool updateOperandIfDifferent(MachineInstr &MI, uint16_t OpName,
                                     unsigned NewEnc) {
  int OpIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpName);
  assert(OpIdx >= 0);

  MachineOperand &MO = MI.getOperand(OpIdx);

  if (NewEnc == MO.getImm())
    return false;

  MO.setImm(NewEnc);
  return true;
}

/// Determine if \p MI is a gfx12+ single-counter S_WAIT_*CNT instruction,
/// and if so, which counter it is waiting on.
static std::optional<InstCounterType> counterTypeForInstr(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_WAIT_LOADCNT:
    return LOAD_CNT;
  case AMDGPU::S_WAIT_EXPCNT:
    return EXP_CNT;
  case AMDGPU::S_WAIT_STORECNT:
    return STORE_CNT;
  case AMDGPU::S_WAIT_SAMPLECNT:
    return SAMPLE_CNT;
  case AMDGPU::S_WAIT_BVHCNT:
    return BVH_CNT;
  case AMDGPU::S_WAIT_DSCNT:
    return DS_CNT;
  case AMDGPU::S_WAIT_KMCNT:
    return KM_CNT;
  default:
    return {};
  }
}

bool updateVMCntOnly(const MachineInstr &Inst) {
  return SIInstrInfo::isVMEM(Inst) || SIInstrInfo::isFLATGlobal(Inst) ||
         SIInstrInfo::isFLATScratch(Inst);
}

bool isWaitInstr(MachineInstr &Inst) {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(Inst.getOpcode());
  return Opcode == AMDGPU::S_WAITCNT ||
         (Opcode == AMDGPU::S_WAITCNT_VSCNT && Inst.getOperand(0).isReg() &&
          Inst.getOperand(0).getReg() == AMDGPU::SGPR_NULL) ||
         Opcode == AMDGPU::S_WAIT_LOADCNT_DSCNT ||
         Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT ||
         counterTypeForInstr(Opcode).has_value();
}

VmemType getVmemType(const MachineInstr &Inst) {
  assert(updateVMCntOnly(Inst));
  if (!SIInstrInfo::isMIMG(Inst) && !SIInstrInfo::isVIMAGE(Inst) &&
      !SIInstrInfo::isVSAMPLE(Inst))
    return VMEM_NOSAMPLER;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return BaseInfo->BVH       ? VMEM_BVH
         : BaseInfo->Sampler ? VMEM_SAMPLER
                             : VMEM_NOSAMPLER;
}

/// \returns true if the callee inserts an s_waitcnt 0 on function entry.
bool callWaitsOnFunctionEntry(const MachineInstr &MI) {
  // Currently all conventions wait, but this may not always be the case.
  //
  // TODO: If IPRA is enabled, and the callee is isSafeForNoCSROpt, it may make
  // senses to omit the wait and do it in the caller.
  return true;
}

/// \returns true if the callee is expected to wait for any outstanding waits
/// before returning.
bool callWaitsOnFunctionReturn(const MachineInstr &MI) { return true; }

// Mapping from event to counter according to the table masks.
InstCounterType eventCounter(const unsigned *masks, WaitEventType E) {
  for (auto T : inst_counter_types()) {
    if (masks[T] & (1 << E))
      return T;
  }
  llvm_unreachable("event type has no associated counter");
}

bool readsVCCZ(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return (Opc == AMDGPU::S_CBRANCH_VCCNZ || Opc == AMDGPU::S_CBRANCH_VCCZ) &&
         !MI.getOperand(1).isUndef();
}

bool isCacheInvOrWBInst(MachineInstr &Inst) {
  auto Opc = Inst.getOpcode();
  return Opc == AMDGPU::GLOBAL_INV || Opc == AMDGPU::GLOBAL_WB ||
         Opc == AMDGPU::GLOBAL_WBINV;
}

#ifndef NDEBUG
static bool isNormalMode(InstCounterType MaxCounter) {
  return MaxCounter == NUM_NORMAL_INST_CNTS;
}
#endif // NDEBUG

unsigned &getCounterRef(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  switch (T) {
  case LOAD_CNT:
    return Wait.LoadCnt;
  case EXP_CNT:
    return Wait.ExpCnt;
  case DS_CNT:
    return Wait.DsCnt;
  case STORE_CNT:
    return Wait.StoreCnt;
  case SAMPLE_CNT:
    return Wait.SampleCnt;
  case BVH_CNT:
    return Wait.BvhCnt;
  case KM_CNT:
    return Wait.KmCnt;
  default:
    llvm_unreachable("bad InstCounterType");
  }
}

void addWait(AMDGPU::Waitcnt &Wait, InstCounterType T, unsigned Count) {
  unsigned &WC = getCounterRef(Wait, T);
  WC = std::min(WC, Count);
}

void setNoWait(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  getCounterRef(Wait, T) = ~0u;
}

unsigned getWait(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  return getCounterRef(Wait, T);
}

WaitCntGenerator *getWaitCntGenerator(MachineFunction &MF,
                                      WaitCntGeneratorPreGFX12 &WCGPreGFX12,
                                      WaitCntGeneratorGFX12Plus &WCGGFX12Plus,
                                      InstCounterType &MaxCounter) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  WaitCntGenerator *WCG = nullptr;

  if (ST->hasExtendedWaitCounts()) {
    MaxCounter = NUM_EXTENDED_INST_CNTS;
    WCGGFX12Plus = WaitCntGeneratorGFX12Plus(ST, MaxCounter);
    WCG = &WCGGFX12Plus;
  } else {
    MaxCounter = NUM_NORMAL_INST_CNTS;
    WCGPreGFX12 = WaitCntGeneratorPreGFX12(ST);
    WCG = &WCGPreGFX12;
  }

  return WCG;
}

//===----------------------------------------------------------------------===//
// WaitcntBrackets member functions.
//===----------------------------------------------------------------------===//

RegInterval WaitcntBrackets::getRegInterval(const MachineInstr *MI,
                                            const MachineRegisterInfo *MRI,
                                            const SIRegisterInfo *TRI,
                                            unsigned OpNo) const {
  const MachineOperand &Op = MI->getOperand(OpNo);
  if (!TRI->isInAllocatableClass(Op.getReg()))
    return {-1, -1};

  // A use via a PW operand does not need a waitcnt.
  // A partial write is not a WAW.
  assert(!Op.getSubReg() || !Op.isUndef());

  RegInterval Result;

  unsigned Reg = TRI->getEncodingValue(AMDGPU::getMCReg(Op.getReg(), *ST)) &
                 AMDGPU::HWEncoding::REG_IDX_MASK;

  if (TRI->isVectorRegister(*MRI, Op.getReg())) {
    assert(Reg >= Encoding.VGPR0 && Reg <= Encoding.VGPRL);
    Result.first = Reg - Encoding.VGPR0;
    if (TRI->isAGPR(*MRI, Op.getReg()))
      Result.first += AGPR_OFFSET;
    assert(Result.first >= 0 && Result.first < SQ_MAX_PGM_VGPRS);
  } else if (TRI->isSGPRReg(*MRI, Op.getReg())) {
    assert(Reg >= Encoding.SGPR0 && Reg < SQ_MAX_PGM_SGPRS);
    Result.first = Reg - Encoding.SGPR0 + NUM_ALL_VGPRS;
    assert(Result.first >= NUM_ALL_VGPRS &&
           Result.first < SQ_MAX_PGM_SGPRS + NUM_ALL_VGPRS);
  }
  // TODO: Handle TTMP
  // else if (TRI->isTTMP(*MRI, Reg.getReg())) ...
  else
    return {-1, -1};

  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Op.getReg());
  unsigned Size = TRI->getRegSizeInBits(*RC);
  Result.second = Result.first + ((Size + 16) / 32);

  return Result;
}

void WaitcntBrackets::setExpScore(const MachineInstr *MI,
                                  const SIInstrInfo *TII,
                                  const SIRegisterInfo *TRI,
                                  const MachineRegisterInfo *MRI, unsigned OpNo,
                                  unsigned Val) {
  auto [RegLow, RegHigh] = getRegInterval(MI, MRI, TRI, OpNo);
  assert(TRI->isVectorRegister(*MRI, MI->getOperand(OpNo).getReg()));
  for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo) {
    setRegScore(RegNo, EXP_CNT, Val);
  }
}

void WaitcntBrackets::updateByEvent(const SIInstrInfo *TII,
                                    const SIRegisterInfo *TRI,
                                    const MachineRegisterInfo *MRI,
                                    WaitEventType E, MachineInstr &Inst) {
  InstCounterType T = eventCounter(WaitEventMaskForInst, E);

  unsigned UB = getScoreUB(T);
  unsigned CurrScore = UB + 1;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  PendingEvents |= 1 << E;
  setScoreUB(T, CurrScore);

  if (T == EXP_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII->isDS(Inst) && (Inst.mayStore() || Inst.mayLoad())) {
      int AddrOpIdx =
          AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::addr);
      // All GDS operations must protect their address register (same as
      // export.)
      if (AddrOpIdx != -1) {
        setExpScore(&Inst, TII, TRI, MRI, AddrOpIdx, CurrScore);
      }

      if (Inst.mayStore()) {
        if (AMDGPU::hasNamedOperand(Inst.getOpcode(), AMDGPU::OpName::data0)) {
          setExpScore(&Inst, TII, TRI, MRI,
                      AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                                 AMDGPU::OpName::data0),
                      CurrScore);
        }
        if (AMDGPU::hasNamedOperand(Inst.getOpcode(), AMDGPU::OpName::data1)) {
          setExpScore(&Inst, TII, TRI, MRI,
                      AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                                 AMDGPU::OpName::data1),
                      CurrScore);
        }
      } else if (SIInstrInfo::isAtomicRet(Inst) && !SIInstrInfo::isGWS(Inst) &&
                 Inst.getOpcode() != AMDGPU::DS_APPEND &&
                 Inst.getOpcode() != AMDGPU::DS_CONSUME &&
                 Inst.getOpcode() != AMDGPU::DS_ORDERED_COUNT) {
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          const MachineOperand &Op = Inst.getOperand(I);
          if (Op.isReg() && !Op.isDef() &&
              TRI->isVectorRegister(*MRI, Op.getReg())) {
            setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
          }
        }
      }
    } else if (TII->isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMTBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      }
    } else if (TII->isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isLDSDIR(Inst)) {
      // LDSDIR instructions attach the score to the destination.
      setExpScore(
          &Inst, TII, TRI, MRI,
          AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::vdst),
          CurrScore);
    } else {
      if (TII->isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          MachineOperand &DefMO = Inst.getOperand(I);
          if (DefMO.isReg() && DefMO.isDef() &&
              TRI->isVGPR(*MRI, DefMO.getReg())) {
            setRegScore(
                TRI->getEncodingValue(AMDGPU::getMCReg(DefMO.getReg(), *ST)),
                EXP_CNT, CurrScore);
          }
        }
      }
      for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
        MachineOperand &MO = Inst.getOperand(I);
        if (MO.isReg() && !MO.isDef() &&
            TRI->isVectorRegister(*MRI, MO.getReg())) {
          setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
        }
      }
    }
#if 0 // TODO: check if this is handled by MUBUF code above.
  } else if (Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORD ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX2 ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX4) {
    MachineOperand *MO = TII->getNamedOperand(Inst, AMDGPU::OpName::data);
    unsigned OpNo;//TODO: find the OpNo for this operand;
    auto [RegLow, RegHigh] = getRegInterval(&Inst, MRI, TRI, OpNo);
    for (int RegNo = RegLow; RegNo < RegHigh;
    ++RegNo) {
      setRegScore(RegNo + NUM_ALL_VGPRS, t, CurrScore);
    }
#endif
  } else /* LGKM_CNT || EXP_CNT || VS_CNT || NUM_INST_CNTS */ {
    // Match the score to the destination registers.
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      auto &Op = Inst.getOperand(I);
      if (!Op.isReg() || !Op.isDef())
        continue;
      auto [RegLow, RegHigh] = getRegInterval(&Inst, MRI, TRI, I);
      if (T == LOAD_CNT || T == SAMPLE_CNT || T == BVH_CNT) {
        if (RegLow >= NUM_ALL_VGPRS)
          continue;
        if (updateVMCntOnly(Inst)) {
          // updateVMCntOnly should only leave us with VGPRs
          // MUBUF, MTBUF, MIMG, FlatGlobal, and FlatScratch only have VGPR/AGPR
          // defs. That's required for a sane index into `VgprMemTypes` below
          assert(TRI->isVectorRegister(*MRI, Op.getReg()));
          VmemType V = getVmemType(Inst);
          for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo)
            VgprVmemTypes[RegNo] |= 1 << V;
        }
      }
      for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo) {
        setRegScore(RegNo, T, CurrScore);
      }
    }
    if (Inst.mayStore() &&
        (TII->isDS(Inst) || TII->mayWriteLDSThroughDMA(Inst))) {
      // MUBUF and FLAT LDS DMA operations need a wait on vmcnt before LDS
      // written can be accessed. A load from LDS to VMEM does not need a wait.
      unsigned Slot = 0;
      for (const auto *MemOp : Inst.memoperands()) {
        if (!MemOp->isStore() ||
            MemOp->getAddrSpace() != AMDGPUAS::LOCAL_ADDRESS)
          continue;
        // Comparing just AA info does not guarantee memoperands are equal
        // in general, but this is so for LDS DMA in practice.
        auto AAI = MemOp->getAAInfo();
        // Alias scope information gives a way to definitely identify an
        // original memory object and practically produced in the module LDS
        // lowering pass. If there is no scope available we will not be able
        // to disambiguate LDS aliasing as after the module lowering all LDS
        // is squashed into a single big object. Do not attempt to use one of
        // the limited LDSDMAStores for something we will not be able to use
        // anyway.
        if (!AAI || !AAI.Scope)
          break;
        for (unsigned I = 0, E = LDSDMAStores.size(); I != E && !Slot; ++I) {
          for (const auto *MemOp : LDSDMAStores[I]->memoperands()) {
            if (MemOp->isStore() && AAI == MemOp->getAAInfo()) {
              Slot = I + 1;
              break;
            }
          }
        }
        if (Slot || LDSDMAStores.size() == NUM_EXTRA_VGPRS - 1)
          break;
        LDSDMAStores.push_back(&Inst);
        Slot = LDSDMAStores.size();
        break;
      }
      setRegScore(SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS + Slot, T, CurrScore);
      if (Slot)
        setRegScore(SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS, T, CurrScore);
    }
  }
}

void WaitcntBrackets::print(raw_ostream &OS) {
  OS << '\n';
  for (auto T : inst_counter_types(MaxCounter)) {
    unsigned SR = getScoreRange(T);

    switch (T) {
    case LOAD_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "LOAD" : "VM") << "_CNT("
         << SR << "): ";
      break;
    case DS_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "DS" : "LGKM") << "_CNT("
         << SR << "): ";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << SR << "): ";
      break;
    case STORE_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "STORE" : "VS") << "_CNT("
         << SR << "): ";
      break;
    case SAMPLE_CNT:
      OS << "    SAMPLE_CNT(" << SR << "): ";
      break;
    case BVH_CNT:
      OS << "    BVH_CNT(" << SR << "): ";
      break;
    case KM_CNT:
      OS << "    KM_CNT(" << SR << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << SR << "): ";
      break;
    }

    if (SR != 0) {
      // Print vgpr scores.
      unsigned LB = getScoreLB(T);

      for (int J = 0; J <= VgprUB; J++) {
        unsigned RegScore = getRegScore(J, T);
        if (RegScore <= LB)
          continue;
        unsigned RelScore = RegScore - LB - 1;
        if (J < SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS) {
          OS << RelScore << ":v" << J << " ";
        } else {
          OS << RelScore << ":ds ";
        }
      }
      // Also need to print sgpr scores for lgkm_cnt.
      if (T == SmemAccessCounter) {
        for (int J = 0; J <= SgprUB; J++) {
          unsigned RegScore = getRegScore(J + NUM_ALL_VGPRS, T);
          if (RegScore <= LB)
            continue;
          unsigned RelScore = RegScore - LB - 1;
          OS << RelScore << ":s" << J << " ";
        }
      }
    }
    OS << '\n';
  }
  OS << '\n';
}

/// Simplify the waitcnt, in the sense of removing redundant counts, and return
/// whether a waitcnt instruction is needed at all.
void WaitcntBrackets::simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const {
  simplifyWaitcnt(LOAD_CNT, Wait.LoadCnt);
  simplifyWaitcnt(EXP_CNT, Wait.ExpCnt);
  simplifyWaitcnt(DS_CNT, Wait.DsCnt);
  simplifyWaitcnt(STORE_CNT, Wait.StoreCnt);
  simplifyWaitcnt(SAMPLE_CNT, Wait.SampleCnt);
  simplifyWaitcnt(BVH_CNT, Wait.BvhCnt);
  simplifyWaitcnt(KM_CNT, Wait.KmCnt);
}

void WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  // The number of outstanding events for this type, T, can be calculated
  // as (UB - LB). If the current Count is greater than or equal to the number
  // of outstanding events, then the wait for this counter is redundant.
  if (Count >= getScoreRange(T))
    Count = ~0u;
}

void WaitcntBrackets::determineWait(InstCounterType T, int RegNo,
                                    AMDGPU::Waitcnt &Wait) const {
  unsigned ScoreToWait = getRegScore(RegNo, T);

  // If the score of src_operand falls within the bracket, we need an
  // s_waitcnt instruction.
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);
  if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
    if ((T == LOAD_CNT || T == DS_CNT) && hasPendingFlat() &&
        !ST->hasFlatLgkmVMemCountInOrder()) {
      // If there is a pending FLAT operation, and this is a VMem or LGKM
      // waitcnt and the target can report early completion, then we need
      // to force a waitcnt 0.
      addWait(Wait, T, 0);
    } else if (counterOutOfOrder(T)) {
      // Counter can get decremented out-of-order when there
      // are multiple types event in the bracket. Also emit an s_wait counter
      // with a conservative value of 0 for the counter.
      addWait(Wait, T, 0);
    } else {
      // If a counter has been maxed out avoid overflow by waiting for
      // MAX(CounterType) - 1 instead.
      unsigned NeededWait = std::min(UB - ScoreToWait, getWaitCountMax(T) - 1);
      addWait(Wait, T, NeededWait);
    }
  }
}

void WaitcntBrackets::applyWaitcnt(const AMDGPU::Waitcnt &Wait) {
  applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
  applyWaitcnt(EXP_CNT, Wait.ExpCnt);
  applyWaitcnt(DS_CNT, Wait.DsCnt);
  applyWaitcnt(STORE_CNT, Wait.StoreCnt);
  applyWaitcnt(SAMPLE_CNT, Wait.SampleCnt);
  applyWaitcnt(BVH_CNT, Wait.BvhCnt);
  applyWaitcnt(KM_CNT, Wait.KmCnt);
}

void WaitcntBrackets::applyWaitcnt(InstCounterType T, unsigned Count) {
  const unsigned UB = getScoreUB(T);
  if (Count >= UB)
    return;
  if (Count != 0) {
    if (counterOutOfOrder(T))
      return;
    setScoreLB(T, std::max(getScoreLB(T), UB - Count));
  } else {
    setScoreLB(T, UB);
    PendingEvents &= ~WaitEventMaskForInst[T];
  }
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if (T == SmemAccessCounter && hasPendingEvent(SMEM_ACCESS))
    return true;
  return hasMixedPendingEvents(T);
}

WaitCntBitMaskFn WaitcntBrackets::getWaitCntBitMaskFn(InstCounterType T) {
  switch (T) {
  case LOAD_CNT:
    if (ST->hasExtendedWaitCounts())
      return getLoadcntBitMask;

    return getVmcntBitMask;
  case DS_CNT:
    if (ST->hasExtendedWaitCounts())
      return getDscntBitMask;

    return getLgkmcntBitMask;
  case EXP_CNT:
    return getExpcntBitMask;
  case STORE_CNT:
    return getStorecntBitMask;
  case SAMPLE_CNT:
    return getSamplecntBitMask;
  case BVH_CNT:
    return getBvhcntBitMask;
  case KM_CNT:
    return getKmcntBitMask;
  default:
    llvm_unreachable("bad InstCounterType in getWaitCntBitMaskFn");
  }
}

bool WaitcntBrackets::mergeScore(const MergeInfo &M, unsigned &Score,
                                 unsigned OtherScore) {
  unsigned MyShifted = Score <= M.OldLB ? 0 : Score + M.MyShift;
  unsigned OtherShifted =
      OtherScore <= M.OtherLB ? 0 : OtherScore + M.OtherShift;
  Score = std::max(MyShifted, OtherShifted);
  return OtherShifted > MyShifted;
}

/// Merge the pending events and associater score brackets of \p Other into
/// this brackets status.
///
/// Returns whether the merge resulted in a change that requires tighter waits
/// (i.e. the merged brackets strictly dominate the original brackets).
bool WaitcntBrackets::merge(const WaitcntBrackets &Other) {
  bool StrictDom = false;

  VgprUB = std::max(VgprUB, Other.VgprUB);
  SgprUB = std::max(SgprUB, Other.SgprUB);

  for (auto T : inst_counter_types(MaxCounter)) {
    // Merge event flags for this counter
    const unsigned OldEvents = PendingEvents & WaitEventMaskForInst[T];
    const unsigned OtherEvents = Other.PendingEvents & WaitEventMaskForInst[T];
    if (OtherEvents & ~OldEvents)
      StrictDom = true;
    PendingEvents |= OtherEvents;

    // Merge scores for this counter
    const unsigned MyPending = ScoreUBs[T] - ScoreLBs[T];
    const unsigned OtherPending = Other.ScoreUBs[T] - Other.ScoreLBs[T];
    const unsigned NewUB = ScoreLBs[T] + std::max(MyPending, OtherPending);
    if (NewUB < ScoreLBs[T])
      report_fatal_error("waitcnt score overflow");

    MergeInfo M;
    M.OldLB = ScoreLBs[T];
    M.OtherLB = Other.ScoreLBs[T];
    M.MyShift = NewUB - ScoreUBs[T];
    M.OtherShift = NewUB - Other.ScoreUBs[T];

    ScoreUBs[T] = NewUB;

    StrictDom |= mergeScore(M, LastFlat[T], Other.LastFlat[T]);

    for (int J = 0; J <= VgprUB; J++)
      StrictDom |= mergeScore(M, VgprScores[T][J], Other.VgprScores[T][J]);

    if (isSmemCounter(T)) {
      for (int J = 0; J <= SgprUB; J++)
        StrictDom |= mergeScore(M, SgprScores[J], Other.SgprScores[J]);
    }
  }

  for (int J = 0; J <= VgprUB; J++) {
    unsigned char NewVmemTypes = VgprVmemTypes[J] | Other.VgprVmemTypes[J];
    StrictDom |= NewVmemTypes != VgprVmemTypes[J];
    VgprVmemTypes[J] = NewVmemTypes;
  }

  return StrictDom;
}

//===----------------------------------------------------------------------===//
// WaitCntGeneratorPreGFX12 member functions.
//===----------------------------------------------------------------------===//

AMDGPU::Waitcnt
WaitCntGeneratorPreGFX12::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt && ST->hasVscnt() ? 0 : ~0u);
}

/// Combine consecutive S_WAITCNT and S_WAITCNT_VSCNT instructions that
/// precede \p It and follow \p OldWaitcntInstr and apply any extra waits
/// from \p Wait that were added by previous passes. Currently this pass
/// conservatively assumes that these preexisting waits are required for
/// correctness.
bool WaitCntGeneratorPreGFX12::applyPreexistingWaitcnt(
    WaitcntBrackets &ScoreBrackets, MachineInstr &OldWaitcntInstr,
    AMDGPU::Waitcnt &Wait, MachineBasicBlock::instr_iterator It) const {
  assert(ST);
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  MachineInstr *WaitcntInstr = nullptr;
  MachineInstr *WaitcntVsCntInstr = nullptr;

  for (auto &II :
       make_early_inc_range(make_range(OldWaitcntInstr.getIterator(), It))) {
    if (II.isMetaInstruction())
      continue;

    unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(II.getOpcode());
    bool IsSoft = Opcode != II.getOpcode();

    // Update required wait count. If this is a soft waitcnt (= it was added
    // by an earlier pass), it may be entirely removed.
    if (Opcode == AMDGPU::S_WAITCNT) {
      unsigned IEnc = II.getOperand(0).getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeWaitcnt(IV, IEnc);
      if (IsSoft)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);

      // Merge consecutive waitcnt of the same type by erasing multiples.
      if (WaitcntInstr || (!Wait.hasWaitExceptStoreCnt() && IsSoft)) {
        II.eraseFromParent();
        Modified = true;
      } else
        WaitcntInstr = &II;
    } else {
      assert(Opcode == AMDGPU::S_WAITCNT_VSCNT);
      assert(II.getOperand(0).getReg() == AMDGPU::SGPR_NULL);

      unsigned OldVSCnt =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (IsSoft)
        ScoreBrackets.simplifyWaitcnt(InstCounterType::STORE_CNT, OldVSCnt);
      Wait.StoreCnt = std::min(Wait.StoreCnt, OldVSCnt);

      if (WaitcntVsCntInstr || (!Wait.hasWaitStoreCnt() && IsSoft)) {
        II.eraseFromParent();
        Modified = true;
      } else
        WaitcntVsCntInstr = &II;
    }
  }

  if (WaitcntInstr) {
    Modified |= updateOperandIfDifferent(*WaitcntInstr, AMDGPU::OpName::simm16,
                                         AMDGPU::encodeWaitcnt(IV, Wait));
    Modified |= promoteSoftWaitCnt(WaitcntInstr);

    ScoreBrackets.applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
    ScoreBrackets.applyWaitcnt(EXP_CNT, Wait.ExpCnt);
    ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
    Wait.LoadCnt = ~0u;
    Wait.ExpCnt = ~0u;
    Wait.DsCnt = ~0u;

    LLVM_DEBUG(It == WaitcntInstr->getParent()->end()
                   ? dbgs()
                         << "applyPreexistingWaitcnt\n"
                         << "New Instr at block end: " << *WaitcntInstr << '\n'
                   : dbgs() << "applyPreexistingWaitcnt\n"
                            << "Old Instr: " << *It
                            << "New Instr: " << *WaitcntInstr << '\n');
  }

  if (WaitcntVsCntInstr) {
    Modified |= updateOperandIfDifferent(*WaitcntVsCntInstr,
                                         AMDGPU::OpName::simm16, Wait.StoreCnt);
    Modified |= promoteSoftWaitCnt(WaitcntVsCntInstr);

    ScoreBrackets.applyWaitcnt(STORE_CNT, Wait.StoreCnt);
    Wait.StoreCnt = ~0u;

    LLVM_DEBUG(It == WaitcntVsCntInstr->getParent()->end()
                   ? dbgs() << "applyPreexistingWaitcnt\n"
                            << "New Instr at block end: " << *WaitcntVsCntInstr
                            << '\n'
                   : dbgs() << "applyPreexistingWaitcnt\n"
                            << "Old Instr: " << *It
                            << "New Instr: " << *WaitcntVsCntInstr << '\n');
  }

  return Modified;
}

/// Generate S_WAITCNT and/or S_WAITCNT_VSCNT instructions for any
/// required counters in \p Wait
bool WaitCntGeneratorPreGFX12::createNewWaitcnt(
    MachineBasicBlock &Block, MachineBasicBlock::instr_iterator It,
    AMDGPU::Waitcnt Wait) {
  assert(ST);
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Waits for VMcnt, LKGMcnt and/or EXPcnt are encoded together into a
  // single instruction while VScnt has its own instruction.
  if (Wait.hasWaitExceptStoreCnt()) {
    unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT)).addImm(Enc);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  if (Wait.hasWaitStoreCnt()) {
    assert(ST->hasVscnt());

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT_VSCNT))
            .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
            .addImm(Wait.StoreCnt);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
}

//===----------------------------------------------------------------------===//
// WaitCntGeneratorGFX12Plus member functions.
//===----------------------------------------------------------------------===//

AMDGPU::Waitcnt
WaitCntGeneratorGFX12Plus::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt ? 0 : ~0u, 0, 0, 0);
}

/// Combine consecutive S_WAIT_*CNT instructions that precede \p It and
/// follow \p OldWaitcntInstr and apply any extra waits from \p Wait that
/// were added by previous passes. Currently this pass conservatively
/// assumes that these preexisting waits are required for correctness.
bool WaitCntGeneratorGFX12Plus::applyPreexistingWaitcnt(
    WaitcntBrackets &ScoreBrackets, MachineInstr &OldWaitcntInstr,
    AMDGPU::Waitcnt &Wait, MachineBasicBlock::instr_iterator It) const {
  assert(ST);
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  MachineInstr *CombinedLoadDsCntInstr = nullptr;
  MachineInstr *CombinedStoreDsCntInstr = nullptr;
  MachineInstr *WaitInstrs[NUM_EXTENDED_INST_CNTS] = {};

  for (auto &II :
       make_early_inc_range(make_range(OldWaitcntInstr.getIterator(), It))) {
    if (II.isMetaInstruction())
      continue;

    MachineInstr **UpdatableInstr;

    // Update required wait count. If this is a soft waitcnt (= it was added
    // by an earlier pass), it may be entirely removed.

    unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(II.getOpcode());
    bool IsSoft = Opcode != II.getOpcode();

    if (Opcode == AMDGPU::S_WAIT_LOADCNT_DSCNT) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeLoadcntDscnt(IV, OldEnc);
      if (IsSoft)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      UpdatableInstr = &CombinedLoadDsCntInstr;
    } else if (Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeStorecntDscnt(IV, OldEnc);
      if (IsSoft)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      UpdatableInstr = &CombinedStoreDsCntInstr;
    } else {
      std::optional<InstCounterType> CT = counterTypeForInstr(Opcode);
      assert(CT.has_value());
      unsigned OldCnt =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (IsSoft)
        ScoreBrackets.simplifyWaitcnt(CT.value(), OldCnt);
      addWait(Wait, CT.value(), OldCnt);
      UpdatableInstr = &WaitInstrs[CT.value()];
    }

    // Merge consecutive waitcnt of the same type by erasing multiples.
    if (!*UpdatableInstr) {
      *UpdatableInstr = &II;
    } else {
      II.eraseFromParent();
      Modified = true;
    }
  }

  if (CombinedLoadDsCntInstr) {
    // Only keep an S_WAIT_LOADCNT_DSCNT if both counters actually need
    // to be waited for. Otherwise, let the instruction be deleted so
    // the appropriate single counter wait instruction can be inserted
    // instead, when new S_WAIT_*CNT instructions are inserted by
    // createNewWaitcnt(). As a side effect, resetting the wait counts will
    // cause any redundant S_WAIT_LOADCNT or S_WAIT_DSCNT to be removed by
    // the loop below that deals with single counter instructions.
    if (Wait.LoadCnt != ~0u && Wait.DsCnt != ~0u) {
      unsigned NewEnc = AMDGPU::encodeLoadcntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedLoadDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedLoadDsCntInstr);
      ScoreBrackets.applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
      ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
      Wait.LoadCnt = ~0u;
      Wait.DsCnt = ~0u;

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applyPreexistingWaitcnt\n"
                              << "New Instr at block end: "
                              << *CombinedLoadDsCntInstr << '\n'
                     : dbgs() << "applyPreexistingWaitcnt\n"
                              << "Old Instr: " << *It << "New Instr: "
                              << *CombinedLoadDsCntInstr << '\n');
    } else {
      CombinedLoadDsCntInstr->eraseFromParent();
      Modified = true;
    }
  }

  if (CombinedStoreDsCntInstr) {
    // Similarly for S_WAIT_STORECNT_DSCNT.
    if (Wait.StoreCnt != ~0u && Wait.DsCnt != ~0u) {
      unsigned NewEnc = AMDGPU::encodeStorecntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedStoreDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedStoreDsCntInstr);
      ScoreBrackets.applyWaitcnt(STORE_CNT, Wait.StoreCnt);
      ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
      Wait.StoreCnt = ~0u;
      Wait.DsCnt = ~0u;

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applyPreexistingWaitcnt\n"
                              << "New Instr at block end: "
                              << *CombinedStoreDsCntInstr << '\n'
                     : dbgs() << "applyPreexistingWaitcnt\n"
                              << "Old Instr: " << *It << "New Instr: "
                              << *CombinedStoreDsCntInstr << '\n');
    } else {
      CombinedStoreDsCntInstr->eraseFromParent();
      Modified = true;
    }
  }

  // Look for an opportunity to convert existing S_WAIT_LOADCNT,
  // S_WAIT_STORECNT and S_WAIT_DSCNT into new S_WAIT_LOADCNT_DSCNT
  // or S_WAIT_STORECNT_DSCNT. This is achieved by selectively removing
  // instructions so that createNewWaitcnt() will create new combined
  // instructions to replace them.

  if (Wait.DsCnt != ~0u) {
    // This is a vector of addresses in WaitInstrs pointing to instructions
    // that should be removed if they are present.
    SmallVector<MachineInstr **, 2> WaitsToErase;

    // If it's known that both DScnt and either LOADcnt or STOREcnt (but not
    // both) need to be waited for, ensure that there are no existing
    // individual wait count instructions for these.

    if (Wait.LoadCnt != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[LOAD_CNT]);
      WaitsToErase.push_back(&WaitInstrs[DS_CNT]);
    } else if (Wait.StoreCnt != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[STORE_CNT]);
      WaitsToErase.push_back(&WaitInstrs[DS_CNT]);
    }

    for (MachineInstr **WI : WaitsToErase) {
      if (!*WI)
        continue;

      (*WI)->eraseFromParent();
      *WI = nullptr;
      Modified = true;
    }
  }

  for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
    if (!WaitInstrs[CT])
      continue;

    unsigned NewCnt = getWait(Wait, CT);
    if (NewCnt != ~0u) {
      Modified |= updateOperandIfDifferent(*WaitInstrs[CT],
                                           AMDGPU::OpName::simm16, NewCnt);
      Modified |= promoteSoftWaitCnt(WaitInstrs[CT]);

      ScoreBrackets.applyWaitcnt(CT, NewCnt);
      setNoWait(Wait, CT);

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applyPreexistingWaitcnt\n"
                              << "New Instr at block end: " << *WaitInstrs[CT]
                              << '\n'
                     : dbgs() << "applyPreexistingWaitcnt\n"
                              << "Old Instr: " << *It
                              << "New Instr: " << *WaitInstrs[CT] << '\n');
    } else {
      WaitInstrs[CT]->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

/// Generate S_WAIT_*CNT instructions for any required counters in \p Wait
bool WaitCntGeneratorGFX12Plus::createNewWaitcnt(
    MachineBasicBlock &Block, MachineBasicBlock::instr_iterator It,
    AMDGPU::Waitcnt Wait) {
  assert(ST);
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Check for opportunities to use combined wait instructions.
  if (Wait.DsCnt != ~0u) {
    MachineInstr *SWaitInst = nullptr;

    if (Wait.LoadCnt != ~0u) {
      unsigned Enc = AMDGPU::encodeLoadcntDscnt(IV, Wait);

      SWaitInst = BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
                      .addImm(Enc);

      Wait.LoadCnt = ~0u;
      Wait.DsCnt = ~0u;
    } else if (Wait.StoreCnt != ~0u) {
      unsigned Enc = AMDGPU::encodeStorecntDscnt(IV, Wait);

      SWaitInst =
          BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAIT_STORECNT_DSCNT))
              .addImm(Enc);

      Wait.StoreCnt = ~0u;
      Wait.DsCnt = ~0u;
    }

    if (SWaitInst) {
      Modified = true;

      LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
                 if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
                 dbgs() << "New Instr: " << *SWaitInst << '\n');
    }
  }

  // Generate an instruction for any remaining counter that needs
  // waiting for.

  for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
    unsigned Count = getWait(Wait, CT);
    if (Count == ~0u)
      continue;

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(Count);

    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
}

//===----------------------------------------------------------------------===//
// AMDGPUWaitCntInserter member functions.
//===----------------------------------------------------------------------===//

// This is a flat memory operation. Check to see if it has memory tokens other
// than LDS. Other address spaces supported by flat memory operations involve
// global memory.
bool AMDGPUWaitCntInserter::mayAccessVMEMThroughFlat(
    const MachineInstr &MI) const {
  assert(TII->isFLAT(MI));

  // All flat instructions use the VMEM counter.
  assert(TII->usesVM_CNT(MI));

  // If there are no memory operands then conservatively assume the flat
  // operation may access VMEM.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves VMEM.
  // Flat operations only supported FLAT, LOCAL (LDS), or address spaces
  // involving VMEM such as GLOBAL, CONSTANT, PRIVATE (SCRATCH), etc. The REGION
  // (GDS) address space is not supported by flat operations. Therefore, simply
  // return true unless only the LDS address space is found.
  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    assert(AS != AMDGPUAS::REGION_ADDRESS);
    if (AS != AMDGPUAS::LOCAL_ADDRESS)
      return true;
  }

  return false;
}

// This is a flat memory operation. Check to see if it has memory tokens for
// either scratch or FLAT.
bool AMDGPUWaitCntInserter::mayAccessScratchThroughFlat(
    const MachineInstr &MI) const {
  assert(TII->isFLAT(MI));

  // SCRATCH instructions always access scratch.
  if (TII->isFLATScratch(MI))
    return true;

  // GLOBAL instructions never access scratch.
  if (TII->isFLATGlobal(MI))
    return false;

  // If there are no memory operands then conservatively assume the flat
  // operation may access scratch.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves scratch.
  return any_of(MI.memoperands(), [](const MachineMemOperand *Memop) {
    unsigned AS = Memop->getAddrSpace();
    return AS == AMDGPUAS::PRIVATE_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS;
  });
}

bool AMDGPUWaitCntInserter::isVMEMOrFlatVMEM(const MachineInstr &MI) const {
  return SIInstrInfo::isVMEM(MI) ||
         (SIInstrInfo::isFLAT(MI) && mayAccessVMEMThroughFlat(MI));
}

bool AMDGPUWaitCntInserter::generateWaitcnt(
    AMDGPU::Waitcnt Wait, MachineBasicBlock::instr_iterator It,
    MachineBasicBlock &Block, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr) {
  bool Modified = false;

  if (OldWaitcntInstr)
    // Try to merge the required wait with preexisting waitcnt instructions.
    // Also erase redundant waitcnt.
    Modified =
        WCG->applyPreexistingWaitcnt(ScoreBrackets, *OldWaitcntInstr, Wait, It);

  // Any counts that could have been applied to any existing waitcnt
  // instructions will have been done so, now deal with any remaining.
  ScoreBrackets.applyWaitcnt(Wait);

  // ExpCnt can be merged into VINTERP.
  if (Wait.ExpCnt != ~0u && It != Block.instr_end() &&
      SIInstrInfo::isVINTERP(*It)) {
    MachineOperand *WaitExp =
        TII->getNamedOperand(*It, AMDGPU::OpName::waitexp);
    if (Wait.ExpCnt < WaitExp->getImm()) {
      WaitExp->setImm(Wait.ExpCnt);
      Modified = true;
    }
    Wait.ExpCnt = ~0u;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n"
                      << "Update Instr: " << *It);
  }

  if (WCG->createNewWaitcnt(Block, It, Wait))
    Modified = true;

  return Modified;
}

// Add a waitcnt to flush the LOADcnt, SAMPLEcnt and BVHcnt counters at the
// end of the given block if needed.
bool AMDGPUWaitCntInserter::generateWaitcntBlockEnd(
    MachineBasicBlock &Block, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr) {
  AMDGPU::Waitcnt Wait;

  unsigned LoadCntPending = ScoreBrackets.hasPendingEvent(LOAD_CNT);
  unsigned SampleCntPending = ScoreBrackets.hasPendingEvent(SAMPLE_CNT);
  unsigned BvhCntPending = ScoreBrackets.hasPendingEvent(BVH_CNT);

  if (LoadCntPending == 0 && SampleCntPending == 0 && BvhCntPending == 0)
    return false;

  if (LoadCntPending != 0)
    Wait.LoadCnt = 0;
  if (SampleCntPending != 0)
    Wait.SampleCnt = 0;
  if (BvhCntPending != 0)
    Wait.BvhCnt = 0;

  return generateWaitcnt(Wait, Block.instr_end(), Block, ScoreBrackets,
                         OldWaitcntInstr);
}

bool AMDGPUWaitCntInserter::insertWaitCntsInFunction(MachineFunction &MF,
                                                     VGPRInstsSet *VGPRInsts) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const unsigned *WaitEventMaskForInst = WCG->getWaitEventMask();
  InstCounterType SmemAccessCounter =
      eventCounter(WaitEventMaskForInst, SMEM_ACCESS);

  unsigned NumVGPRsMax = ST->getAddressableNumVGPRs();
  unsigned NumSGPRsMax = ST->getAddressableNumSGPRs();
  assert(NumVGPRsMax <= SQ_MAX_PGM_VGPRS);
  assert(NumSGPRsMax <= SQ_MAX_PGM_SGPRS);

  RegisterEncoding Encoding = {};
  Encoding.VGPR0 =
      TRI->getEncodingValue(AMDGPU::VGPR0) & AMDGPU::HWEncoding::REG_IDX_MASK;
  Encoding.VGPRL = Encoding.VGPR0 + NumVGPRsMax - 1;
  Encoding.SGPR0 =
      TRI->getEncodingValue(AMDGPU::SGPR0) & AMDGPU::HWEncoding::REG_IDX_MASK;
  Encoding.SGPRL = Encoding.SGPR0 + NumSGPRsMax - 1;

  MapVector<MachineBasicBlock *, BlockInfo> BlockInfos;
  BlockInfos.clear();
  bool Modified = false;

  MachineBasicBlock &EntryBB = MF.front();
  MachineBasicBlock::iterator I = EntryBB.begin();

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to do the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    for (MachineBasicBlock::iterator E = EntryBB.end();
         I != E && (I->isPHI() || I->isMetaInstruction()); ++I)
      ;

    if (ST->hasExtendedWaitCounts()) {
      BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
          .addImm(0);
      for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
        if (CT == LOAD_CNT || CT == DS_CNT || CT == STORE_CNT)
          continue;

        BuildMI(EntryBB, I, DebugLoc(),
                TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(0);
      }
    } else {
      BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT)).addImm(0);
    }

    auto NonKernelInitialState = std::make_unique<WaitcntBrackets>(
        ST, MaxCounter, Encoding, WaitEventMaskForInst, SmemAccessCounter);
    NonKernelInitialState->setStateOnFunctionEntryOrReturn();
    BlockInfos[&EntryBB].Incoming = std::move(NonKernelInitialState);

    Modified = true;
  }

  // Keep iterating over the blocks in reverse post order, inserting and
  // updating s_waitcnt where needed, until a fix point is reached.
  for (auto *MBB : ReversePostOrderTraversal<MachineFunction *>(&MF))
    BlockInfos.insert({MBB, BlockInfo()});

  std::unique_ptr<WaitcntBrackets> Brackets;
  bool Repeat;
  do {
    Repeat = false;

    for (auto BII = BlockInfos.begin(), BIE = BlockInfos.end(); BII != BIE;
         ++BII) {
      MachineBasicBlock *MBB = BII->first;
      BlockInfo &BI = BII->second;
      if (!BI.Dirty)
        continue;

      if (BI.Incoming) {
        if (!Brackets)
          Brackets = std::make_unique<WaitcntBrackets>(*BI.Incoming);
        else
          *Brackets = *BI.Incoming;
      } else {
        if (!Brackets)
          Brackets = std::make_unique<WaitcntBrackets>(ST, MaxCounter, Encoding,
                                                       WaitEventMaskForInst,
                                                       SmemAccessCounter);
        else
          *Brackets = WaitcntBrackets(ST, MaxCounter, Encoding,
                                      WaitEventMaskForInst, SmemAccessCounter);
      }

      Modified |= insertWaitcntInBlock(MF, *MBB, *Brackets, VGPRInsts);
      BI.Dirty = false;

      if (Brackets->hasPendingEvent()) {
        BlockInfo *MoveBracketsToSucc = nullptr;
        for (MachineBasicBlock *Succ : MBB->successors()) {
          auto SuccBII = BlockInfos.find(Succ);
          BlockInfo &SuccBI = SuccBII->second;
          if (!SuccBI.Incoming) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII)
              Repeat = true;
            if (!MoveBracketsToSucc) {
              MoveBracketsToSucc = &SuccBI;
            } else {
              SuccBI.Incoming = std::make_unique<WaitcntBrackets>(*Brackets);
            }
          } else if (SuccBI.Incoming->merge(*Brackets)) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII)
              Repeat = true;
          }
        }
        if (MoveBracketsToSucc)
          MoveBracketsToSucc->Incoming = std::move(Brackets);
      }
    }
  } while (Repeat);

  if (ST->hasScalarStores()) {
    SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;
    bool HaveScalarStores = false;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!HaveScalarStores && TII->isScalarStore(MI))
          HaveScalarStores = true;

        if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
            MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG)
          EndPgmBlocks.push_back(&MBB);
      }
    }

    if (HaveScalarStores) {
      // If scalar writes are used, the cache must be flushed or else the next
      // wave to reuse the same scratch memory can be clobbered.
      //
      // Insert s_dcache_wb at wave termination points if there were any scalar
      // stores, and only if the cache hasn't already been flushed. This could
      // be improved by looking across blocks for flushes in postdominating
      // blocks from the stores but an explicitly requested flush is probably
      // very rare.
      for (MachineBasicBlock *MBB : EndPgmBlocks) {
        bool SeenDCacheWB = false;

        for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
             I != E; ++I) {
          if (I->getOpcode() == AMDGPU::S_DCACHE_WB)
            SeenDCacheWB = true;
          else if (TII->isScalarStore(*I))
            SeenDCacheWB = false;

          // FIXME: It would be better to insert this before a waitcnt if any.
          if ((I->getOpcode() == AMDGPU::S_ENDPGM ||
               I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) &&
              !SeenDCacheWB) {
            Modified = true;
            BuildMI(*MBB, I, I->getDebugLoc(), TII->get(AMDGPU::S_DCACHE_WB));
          }
        }
      }
    }
  }

  // Insert DEALLOC_VGPR messages before previously identified S_ENDPGM
  // instructions.
  for (MachineInstr *MI : *VGPRInsts) {
    if (ST->requiresNopBeforeDeallocVGPRs()) {
      BuildMI(*MI->getParent(), MI, DebugLoc(), TII->get(AMDGPU::S_NOP))
          .addImm(0);
    }
    BuildMI(*MI->getParent(), MI, DebugLoc(), TII->get(AMDGPU::S_SENDMSG))
        .addImm(AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus);
    Modified = true;
  }
  VGPRInsts->clear();

  return Modified;
}

} // namespace AMDGPU

} // namespace llvm
