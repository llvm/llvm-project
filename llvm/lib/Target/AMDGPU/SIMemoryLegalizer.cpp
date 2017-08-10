//===- SIMemoryLegalizer.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Memory legalizer - implements memory model. More information can be
/// found here:
///   http://llvm.org/docs/AMDGPUUsage.html#memory-model
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineModuleInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include <cassert>
#include <list>

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "si-memory-legalizer"
#define PASS_NAME "SI Memory Legalizer"

namespace {

class SIMemoryLegalizer final : public MachineFunctionPass {
private:
  struct AtomicInfo final {
    SyncScope::ID SSID = SyncScope::System;
    AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent;
    AtomicOrdering FailureOrdering = AtomicOrdering::SequentiallyConsistent;

    AtomicInfo() = default;

    AtomicInfo(SyncScope::ID SSID,
               AtomicOrdering Ordering,
               AtomicOrdering FailureOrdering)
        : SSID(SSID),
          Ordering(Ordering),
          FailureOrdering(FailureOrdering) {}

    AtomicInfo(const MachineMemOperand *MMO)
        : SSID(MMO->getSyncScopeID()),
          Ordering(MMO->getOrdering()),
          FailureOrdering(MMO->getFailureOrdering()) {}
  };

  /// \brief LLVM context.
  LLVMContext *CTX = nullptr;

  /// \brief Machine module info.
  const AMDGPUMachineModuleInfo *MMI = nullptr;

  /// \brief Instruction info.
  const SIInstrInfo *TII = nullptr;

  /// \brief Immediate for "vmcnt(0)".
  unsigned Vmcnt0Immediate = 0;

  /// \brief Opcode for cache invalidation instruction (L1).
  unsigned Wbinvl1Opcode = 0;

  /// \brief List of atomic pseudo instructions.
  std::list<MachineBasicBlock::iterator> AtomicPseudoMIs;

  /// \brief Inserts "buffer_wbinvl1_vol" instruction \p Before or after \p MI.
  /// Always returns true.
  bool insertBufferWbinvl1Vol(MachineBasicBlock::iterator &MI,
                              bool Before = true) const;
  /// \brief Inserts "s_waitcnt vmcnt(0)" instruction \p Before or after \p MI.
  /// Always returns true.
  bool insertWaitcntVmcnt0(MachineBasicBlock::iterator &MI,
                           bool Before = true) const;

  /// \brief Sets GLC bit if present in \p MI. Returns true if \p MI is
  /// modified, false otherwise.
  bool setGLC(const MachineBasicBlock::iterator &MI) const;

  /// \brief Removes all processed atomic pseudo instructions from the current
  /// function. Returns true if current function is modified, false otherwise.
  bool removeAtomicPseudoMIs();

  /// \brief Reports unknown synchronization scope used in \p MI to LLVM
  /// context.
  void reportUnknownSynchScope(const MachineBasicBlock::iterator &MI);

  /// \returns Atomic fence info if \p MI is an atomic fence operation,
  /// "None" otherwise.
  Optional<AtomicInfo> getAtomicFenceInfo(
      const MachineBasicBlock::iterator &MI) const;
  /// \returns Atomic load info if \p MI is an atomic load operation,
  /// "None" otherwise.
  Optional<AtomicInfo> getAtomicLoadInfo(
      const MachineBasicBlock::iterator &MI) const;
  /// \returns Atomic store info if \p MI is an atomic store operation,
  /// "None" otherwise.
  Optional<AtomicInfo> getAtomicStoreInfo(
      const MachineBasicBlock::iterator &MI) const;
  /// \returns Atomic cmpxchg info if \p MI is an atomic cmpxchg operation,
  /// "None" otherwise.
  Optional<AtomicInfo> getAtomicCmpxchgInfo(
      const MachineBasicBlock::iterator &MI) const;
  /// \returns Atomic rmw info if \p MI is an atomic rmw operation,
  /// "None" otherwise.
  Optional<AtomicInfo> getAtomicRmwInfo(
      const MachineBasicBlock::iterator &MI) const;

  /// \brief Expands atomic fence operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicFence(const AtomicInfo &AI,
                         MachineBasicBlock::iterator &MI);
  /// \brief Expands atomic load operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicLoad(const AtomicInfo &AI,
                        MachineBasicBlock::iterator &MI);
  /// \brief Expands atomic store operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicStore(const AtomicInfo &AI,
                         MachineBasicBlock::iterator &MI);
  /// \brief Expands atomic cmpxchg operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicCmpxchg(const AtomicInfo &AI,
                           MachineBasicBlock::iterator &MI);
  /// \brief Expands atomic rmw operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicRmw(const AtomicInfo &AI,
                       MachineBasicBlock::iterator &MI);

public:
  static char ID;

  SIMemoryLegalizer() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return PASS_NAME;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end namespace anonymous

bool SIMemoryLegalizer::insertBufferWbinvl1Vol(MachineBasicBlock::iterator &MI,
                                               bool Before) const {
  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (!Before)
    ++MI;

  BuildMI(MBB, MI, DL, TII->get(Wbinvl1Opcode));

  if (!Before)
    --MI;

  return true;
}

bool SIMemoryLegalizer::insertWaitcntVmcnt0(MachineBasicBlock::iterator &MI,
                                            bool Before) const {
  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (!Before)
    ++MI;

  BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT)).addImm(Vmcnt0Immediate);

  if (!Before)
    --MI;

  return true;
}

bool SIMemoryLegalizer::setGLC(const MachineBasicBlock::iterator &MI) const {
  int GLCIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::glc);
  if (GLCIdx == -1)
    return false;

  MachineOperand &GLC = MI->getOperand(GLCIdx);
  if (GLC.getImm() == 1)
    return false;

  GLC.setImm(1);
  return true;
}

bool SIMemoryLegalizer::removeAtomicPseudoMIs() {
  if (AtomicPseudoMIs.empty())
    return false;

  for (auto &MI : AtomicPseudoMIs)
    MI->eraseFromParent();

  AtomicPseudoMIs.clear();
  return true;
}

void SIMemoryLegalizer::reportUnknownSynchScope(
    const MachineBasicBlock::iterator &MI) {
  DiagnosticInfoUnsupported Diag(*MI->getParent()->getParent()->getFunction(),
                                 "Unsupported synchronization scope");
  CTX->diagnose(Diag);
}

Optional<SIMemoryLegalizer::AtomicInfo> SIMemoryLegalizer::getAtomicFenceInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (MI->getOpcode() != AMDGPU::ATOMIC_FENCE)
    return None;

  SyncScope::ID SSID =
      static_cast<SyncScope::ID>(MI->getOperand(1).getImm());
  AtomicOrdering Ordering =
      static_cast<AtomicOrdering>(MI->getOperand(0).getImm());
  return AtomicInfo(SSID, Ordering, AtomicOrdering::NotAtomic);
}

Optional<SIMemoryLegalizer::AtomicInfo> SIMemoryLegalizer::getAtomicLoadInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && !MI->mayStore()))
    return None;
  if (!MI->hasOneMemOperand())
    return AtomicInfo();

  const MachineMemOperand *MMO = *MI->memoperands_begin();
  if (!MMO->isAtomic())
    return None;

  return AtomicInfo(MMO);
}

Optional<SIMemoryLegalizer::AtomicInfo> SIMemoryLegalizer::getAtomicStoreInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(!MI->mayLoad() && MI->mayStore()))
    return None;
  if (!MI->hasOneMemOperand())
    return AtomicInfo();

  const MachineMemOperand *MMO = *MI->memoperands_begin();
  if (!MMO->isAtomic())
    return None;

  return AtomicInfo(MMO);
}

Optional<SIMemoryLegalizer::AtomicInfo> SIMemoryLegalizer::getAtomicCmpxchgInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && MI->mayStore()))
    return None;
  if (!MI->hasOneMemOperand())
    return AtomicInfo();

  const MachineMemOperand *MMO = *MI->memoperands_begin();
  if (!MMO->isAtomic())
    return None;
  if (MMO->getFailureOrdering() == AtomicOrdering::NotAtomic)
    return None;

  return AtomicInfo(MMO);
}

Optional<SIMemoryLegalizer::AtomicInfo> SIMemoryLegalizer::getAtomicRmwInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && MI->mayStore()))
    return None;
  if (!MI->hasOneMemOperand())
    return AtomicInfo();

  const MachineMemOperand *MMO = *MI->memoperands_begin();
  if (!MMO->isAtomic())
    return None;
  if (MMO->getFailureOrdering() != AtomicOrdering::NotAtomic)
    return None;

  return AtomicInfo(MMO);
}

bool SIMemoryLegalizer::expandAtomicFence(const AtomicInfo &AI,
                                          MachineBasicBlock::iterator &MI) {
  assert(MI->getOpcode() == AMDGPU::ATOMIC_FENCE);

  bool Changed = false;
  if (AI.SSID == SyncScope::System ||
      AI.SSID == MMI->getAgentSSID()) {
    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::Release ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertWaitcntVmcnt0(MI);

    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertBufferWbinvl1Vol(MI);

    AtomicPseudoMIs.push_back(MI);
    return Changed;
  } else if (AI.SSID == SyncScope::SingleThread ||
             AI.SSID == MMI->getWorkgroupSSID() ||
             AI.SSID == MMI->getWavefrontSSID()) {
    AtomicPseudoMIs.push_back(MI);
    return Changed;
  } else {
    reportUnknownSynchScope(MI);
    return Changed;
  }
}

bool SIMemoryLegalizer::expandAtomicLoad(const AtomicInfo &AI,
                                         MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && !MI->mayStore());

  bool Changed = false;
  if (AI.SSID == SyncScope::System ||
      AI.SSID == MMI->getAgentSSID()) {
    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= setGLC(MI);

    if (AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertWaitcntVmcnt0(MI);

    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent) {
      Changed |= insertWaitcntVmcnt0(MI, false);
      Changed |= insertBufferWbinvl1Vol(MI, false);
    }

    return Changed;
  } else if (AI.SSID == SyncScope::SingleThread ||
             AI.SSID == MMI->getWorkgroupSSID() ||
             AI.SSID == MMI->getWavefrontSSID()) {
    return Changed;
  } else {
    reportUnknownSynchScope(MI);
    return Changed;
  }
}

bool SIMemoryLegalizer::expandAtomicStore(const AtomicInfo &AI,
                                          MachineBasicBlock::iterator &MI) {
  assert(!MI->mayLoad() && MI->mayStore());

  bool Changed = false;
  if (AI.SSID == SyncScope::System ||
      AI.SSID == MMI->getAgentSSID()) {
    if (AI.Ordering == AtomicOrdering::Release ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertWaitcntVmcnt0(MI);

    return Changed;
  } else if (AI.SSID == SyncScope::SingleThread ||
             AI.SSID == MMI->getWorkgroupSSID() ||
             AI.SSID == MMI->getWavefrontSSID()) {
    return Changed;
  } else {
    reportUnknownSynchScope(MI);
    return Changed;
  }
}

bool SIMemoryLegalizer::expandAtomicCmpxchg(const AtomicInfo &AI,
                                            MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && MI->mayStore());

  bool Changed = false;
  if (AI.SSID == SyncScope::System ||
      AI.SSID == MMI->getAgentSSID()) {
    if (AI.Ordering == AtomicOrdering::Release ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent ||
        AI.FailureOrdering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertWaitcntVmcnt0(MI);

    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent ||
        AI.FailureOrdering == AtomicOrdering::Acquire ||
        AI.FailureOrdering == AtomicOrdering::SequentiallyConsistent) {
      Changed |= insertWaitcntVmcnt0(MI, false);
      Changed |= insertBufferWbinvl1Vol(MI, false);
    }

    return Changed;
  } else if (AI.SSID == SyncScope::SingleThread ||
             AI.SSID == MMI->getWorkgroupSSID() ||
             AI.SSID == MMI->getWavefrontSSID()) {
    Changed |= setGLC(MI);
    return Changed;
  } else {
    reportUnknownSynchScope(MI);
    return Changed;
  }
}

bool SIMemoryLegalizer::expandAtomicRmw(const AtomicInfo &AI,
                                        MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && MI->mayStore());

  bool Changed = false;
  if (AI.SSID == SyncScope::System ||
      AI.SSID == MMI->getAgentSSID()) {
    if (AI.Ordering == AtomicOrdering::Release ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent)
      Changed |= insertWaitcntVmcnt0(MI);

    if (AI.Ordering == AtomicOrdering::Acquire ||
        AI.Ordering == AtomicOrdering::AcquireRelease ||
        AI.Ordering == AtomicOrdering::SequentiallyConsistent) {
      Changed |= insertWaitcntVmcnt0(MI, false);
      Changed |= insertBufferWbinvl1Vol(MI, false);
    }

    return Changed;
  } else if (AI.SSID == SyncScope::SingleThread ||
             AI.SSID == MMI->getWorkgroupSSID() ||
             AI.SSID == MMI->getWavefrontSSID()) {
    Changed |= setGLC(MI);
    return Changed;
  } else {
    reportUnknownSynchScope(MI);
    return Changed;
  }
}

bool SIMemoryLegalizer::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const IsaInfo::IsaVersion IV = IsaInfo::getIsaVersion(ST.getFeatureBits());

  CTX = &MF.getFunction()->getContext();
  MMI = &MF.getMMI().getObjFileInfo<AMDGPUMachineModuleInfo>();
  TII = ST.getInstrInfo();

  Vmcnt0Immediate =
      AMDGPU::encodeWaitcnt(IV, 0, getExpcntBitMask(IV), getLgkmcntBitMask(IV));
  Wbinvl1Opcode = ST.getGeneration() <= AMDGPUSubtarget::SOUTHERN_ISLANDS ?
      AMDGPU::BUFFER_WBINVL1 : AMDGPU::BUFFER_WBINVL1_VOL;

  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (!(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic))
        continue;

      if (const auto &AI = getAtomicFenceInfo(MI))
        Changed |= expandAtomicFence(AI.getValue(), MI);
      else if (const auto &AI = getAtomicLoadInfo(MI))
        Changed |= expandAtomicLoad(AI.getValue(), MI);
      else if (const auto &AI = getAtomicStoreInfo(MI))
        Changed |= expandAtomicStore(AI.getValue(), MI);
      else if (const auto &AI = getAtomicCmpxchgInfo(MI))
        Changed |= expandAtomicCmpxchg(AI.getValue(), MI);
      else if (const auto &AI = getAtomicRmwInfo(MI))
        Changed |= expandAtomicRmw(AI.getValue(), MI);
    }
  }

  Changed |= removeAtomicPseudoMIs();
  return Changed;
}

INITIALIZE_PASS(SIMemoryLegalizer, DEBUG_TYPE, PASS_NAME, false, false)

char SIMemoryLegalizer::ID = 0;
char &llvm::SIMemoryLegalizerID = SIMemoryLegalizer::ID;

FunctionPass *llvm::createSIMemoryLegalizerPass() {
  return new SIMemoryLegalizer();
}
