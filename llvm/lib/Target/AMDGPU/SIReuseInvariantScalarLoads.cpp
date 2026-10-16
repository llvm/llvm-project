//===-- SIReuseInvariantScalarLoads.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reuse rematerialized invariant scalar loads after register allocation.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "si-reuse-invariant-scalar-loads"

static cl::opt<bool> CacheInvariantScalarLoads(
    "amdgpu-cache-invariant-scalar-loads", cl::Hidden, cl::init(true),
    cl::desc("Reuse invariant scalar loads in an unused EXEC-save register"));

namespace {

struct WordState {
  bool Overwritten = false;
  bool Valid = true;
  MachineOperand *Use = nullptr;
};

struct Candidate {
  MachineInstr *Load;
  MachineOperand *Use;
  MCRegister Replacement;
};

using LoadKey =
    std::tuple<unsigned, unsigned, int64_t, unsigned, int64_t, unsigned>;
using CandidateGroup = SmallVector<Candidate>;

class SIReuseInvariantScalarLoads {
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MCRegister Cache;
  SmallVector<MCRegister, 2> CacheWords;

  bool doesCacheFitSGPRUsage(const MachineFunction &MF) const;
  bool hasLocalExecCopyUses(const MachineFunction &MF) const;
  bool isSupportedUse(const MachineInstr &MI, unsigned OpNo) const;
  bool runOnBlock(MachineBasicBlock &MBB) const;

public:
  explicit SIReuseInvariantScalarLoads(MachineFunction &MF)
      : TRI(MF.getSubtarget<GCNSubtarget>().getRegisterInfo()),
        TII(MF.getSubtarget<GCNSubtarget>().getInstrInfo()),
        Cache(MF.getInfo<SIMachineFunctionInfo>()->getSGPRForEXECCopy()) {
    if (AMDGPU::SGPR_64RegClass.contains(Cache)) {
      CacheWords.push_back(TRI->getSubReg(Cache, AMDGPU::sub0));
      CacheWords.push_back(TRI->getSubReg(Cache, AMDGPU::sub1));
    } else if (AMDGPU::SGPR_32RegClass.contains(Cache)) {
      CacheWords.push_back(Cache);
    }
  }

  bool run(MachineFunction &MF) const;
};

class SIReuseInvariantScalarLoadsLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIReuseInvariantScalarLoadsLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SI reuse invariant scalar loads";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!CacheInvariantScalarLoads || skipFunction(MF.getFunction()))
      return false;
    return SIReuseInvariantScalarLoads(MF).run(MF);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(SIReuseInvariantScalarLoadsLegacy, DEBUG_TYPE,
                "SI reuse invariant scalar loads", false, false)

char SIReuseInvariantScalarLoadsLegacy::ID = 0;
char &llvm::SIReuseInvariantScalarLoadsLegacyID =
    SIReuseInvariantScalarLoadsLegacy::ID;

// Reserving the EXEC-save register does not itself include it in the kernel's
// SGPR allocation. Reusing it must not increase the highest emitted SGPR.
bool SIReuseInvariantScalarLoads::doesCacheFitSGPRUsage(
    const MachineFunction &MF) const {
  unsigned CacheEnd = TRI->getHWRegIndex(Cache) + CacheWords.size();
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB.instrs()) {
      unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(MI.getOpcode());
      if (MI.isMetaInstruction() || TII->pseudoToMCOpcode(Opcode) < 0)
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.isImplicit())
          continue;
        for (MCSubRegIterator R(MO.getReg(), TRI, true); R.isValid(); ++R) {
          if (AMDGPU::SGPR_32RegClass.contains(*R) &&
              TRI->getHWRegIndex(*R) + 1 >= CacheEnd)
            return true;
        }
      }
    }
  }
  return false;
}

// Reserved registers can be absent from live-ins. Require each existing
// EXEC-copy use to follow a local definition, as in WWM and frame lowering.
bool SIReuseInvariantScalarLoads::hasLocalExecCopyUses(
    const MachineFunction &MF) const {
  for (const MachineBasicBlock &MBB : MF) {
    bool Defined[2] = {false, false};
    for (const MachineInstr &MI : MBB.instrs()) {
      for (unsigned Word = 0; Word != CacheWords.size(); ++Word) {
        if (!Defined[Word] && MI.readsRegister(CacheWords[Word], TRI))
          return false;
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg() &&
              TRI->regsOverlap(MO.getReg(), CacheWords[Word]))
            Defined[Word] = true;
        }
      }
    }
  }
  return true;
}

bool SIReuseInvariantScalarLoads::isSupportedUse(const MachineInstr &MI,
                                                 unsigned OpNo) const {
  if (MI.getOpcode() != AMDGPU::V_MUL_LO_U32_e64 || MI.isBundled())
    return false;
  const MachineOperand &MO = MI.getOperand(OpNo);
  int Src0 = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src0);
  int Src1 = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src1);
  if (MO.isImplicit() || MO.isTied() || MO.isUndef() || MO.getSubReg() ||
      (int(OpNo) != Src0 && int(OpNo) != Src1))
    return false;
  // VALU sources are not renamable because of constant-bus constraints.
  // This pattern keeps the other operand in a VGPR and checks target legality.
  const MachineOperand &Other = MI.getOperand(int(OpNo) == Src0 ? Src1 : Src0);
  MachineOperand Replacement = MachineOperand::CreateReg(CacheWords[0], false);
  return Other.isReg() && !Other.getSubReg() &&
         AMDGPU::VGPR_32RegClass.contains(Other.getReg()) &&
         TII->isOperandLegal(MI, OpNo, &Replacement);
}

bool SIReuseInvariantScalarLoads::runOnBlock(MachineBasicBlock &MBB) const {
  LivePhysRegs Live(*TRI);
  Live.addLiveOuts(MBB);
  for (MCRegAliasIterator R(Cache, TRI, true); R.isValid(); ++R) {
    if (Live.contains(*R))
      return false;
  }

  SmallSet<MCRegister, 16> ModifiedWords;
  for (const MachineInstr &MI : MBB.instrs()) {
    if (MI.readsRegister(Cache, TRI) || MI.modifiesRegister(Cache, TRI) ||
        MI.isCall() || MI.isInlineAsm())
      return false;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask())
        return false;
      if (!MO.isReg() || !MO.isDef() || !MO.getReg())
        continue;
      for (MCSubRegIterator R(MO.getReg(), TRI, true); R.isValid(); ++R) {
        if (AMDGPU::SGPR_32RegClass.contains(*R))
          ModifiedWords.insert(*R);
      }
    }
  }

  DenseMap<MCRegister, WordState> Words;
  DenseMap<LoadKey, unsigned> GroupIndices;
  SmallVector<CandidateGroup> Groups;
  // Walking backwards records uses up to the next overwrite of each word,
  // without rescanning the instruction range for every rematerialized load.
  for (MachineInstr &MI : llvm::reverse(MBB.instrs())) {
    if (MI.isTerminator()) {
      Words.clear();
      Groups.clear();
      GroupIndices.clear();
      continue;
    }
    unsigned NumWords = 0;
    switch (MI.getOpcode()) {
    case AMDGPU::S_LOAD_DWORD_IMM:
      NumWords = 1;
      break;
    case AMDGPU::S_LOAD_DWORDX2_IMM:
    case AMDGPU::S_LOAD_DWORDX2_IMM_ec:
      NumWords = 2;
      break;
    }
    if (NumWords && NumWords <= CacheWords.size() && !MI.isBundled() &&
        MI.hasOneMemOperand() && MI.getNumOperands() == 4) {
      MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &Base = MI.getOperand(1);
      const MachineMemOperand *MMO = *MI.memoperands_begin();
      if (Dst.isReg() && Dst.isDef() && Dst.isRenamable() && !Dst.getSubReg() &&
          (NumWords == 1 ? AMDGPU::SGPR_32RegClass.contains(Dst.getReg())
                         : AMDGPU::SGPR_64RegClass.contains(Dst.getReg())) &&
          Base.isReg() && !Base.isUndef() && !Base.getSubReg() &&
          AMDGPU::SGPR_64RegClass.contains(Base.getReg()) &&
          MI.getOperand(2).isImm() && MI.getOperand(3).isImm() &&
          MMO->isLoad() && MMO->isInvariant() && !MMO->isVolatile() &&
          !MMO->isAtomic() &&
          MMO->getSize() == LocationSize::precise(4 * NumWords) &&
          !ModifiedWords.contains(
              TRI->getSubReg(Base.getReg(), AMDGPU::sub0)) &&
          !ModifiedWords.contains(
              TRI->getSubReg(Base.getReg(), AMDGPU::sub1))) {
        MachineOperand *Use = nullptr;
        MCRegister Replacement;
        bool Valid = true;
        for (unsigned Word = 0; Word != NumWords; ++Word) {
          MCRegister Reg =
              NumWords == 1
                  ? MCRegister(Dst.getReg())
                  : TRI->getSubReg(Dst.getReg(),
                                   Word == 0 ? AMDGPU::sub0 : AMDGPU::sub1);
          WordState State = Words.lookup(Reg);
          if (!State.Overwritten || !State.Valid || (Use && State.Use)) {
            Valid = false;
            break;
          }
          if (State.Use) {
            Use = State.Use;
            Replacement = CacheWords[Word];
          }
        }
        MachineOperand NewDst = MachineOperand::CreateReg(
            NumWords == 1 ? CacheWords[0] : Cache, true);
        if (Valid && Use && TII->isOperandLegal(MI, 0, &NewDst)) {
          LoadKey Key(NumWords, Base.getReg().id(), MI.getOperand(2).getImm(),
                      MI.getOperand(2).getTargetFlags(),
                      MI.getOperand(3).getImm(),
                      MI.getOperand(3).getTargetFlags());
          auto [It, Inserted] = GroupIndices.try_emplace(Key, Groups.size());
          if (Inserted)
            Groups.emplace_back();
          Groups[It->second].push_back({&MI, Use, Replacement});
        }
      }
    }

    // Uses precede definitions in forward execution, including tied operands.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg())
        continue;
      for (MCSubRegIterator R(MO.getReg(), TRI, true); R.isValid(); ++R) {
        if (AMDGPU::SGPR_32RegClass.contains(*R))
          Words[*R] = {true, true, nullptr};
      }
    }
    for (unsigned OpNo = 0; OpNo != MI.getNumOperands(); ++OpNo) {
      MachineOperand &MO = MI.getOperand(OpNo);
      if (!MO.isReg() || !MO.isUse() || !MO.getReg())
        continue;
      for (MCSubRegIterator R(MO.getReg(), TRI, true); R.isValid(); ++R) {
        if (!AMDGPU::SGPR_32RegClass.contains(*R))
          continue;
        WordState &State = Words[*R];
        if (MO.getReg() != *R || State.Use || !isSupportedUse(MI, OpNo)) {
          State.Valid = false;
        } else {
          State.Use = &MO;
        }
      }
    }
  }

  if (Groups.empty())
    return false;
  auto Best = llvm::max_element(
      Groups, [](const CandidateGroup &A, const CandidateGroup &B) {
        return A.size() < B.size();
      });
  if (Best->size() < 2)
    return false;
  MachineInstr &First = *Best->back().Load;
  First.getOperand(0).setReg(
      AMDGPU::SGPR_32RegClass.contains(First.getOperand(0).getReg())
          ? CacheWords[0]
          : Cache);
  First.getOperand(0).setIsDead(false);
  for (Candidate &C : *Best) {
    C.Use->setReg(C.Replacement);
    C.Use->setIsKill(false);
    if (C.Load == &First)
      continue;
    MBB.getParent()->substituteDebugValuesForInst(*C.Load, First, 1);
    C.Load->eraseFromParent();
  }
  return true;
}

bool SIReuseInvariantScalarLoads::run(MachineFunction &MF) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (!MFI->isEntryFunction() ||
      !MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::TracksLiveness) ||
      !MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::NoVRegs) ||
      CacheWords.empty() ||
      TRI->getHWRegIndex(Cache) + CacheWords.size() > ST.getMaxNumSGPRs(MF) ||
      (MFI->getLongBranchReservedReg() &&
       TRI->regsOverlap(MFI->getLongBranchReservedReg(), Cache)) ||
      !doesCacheFitSGPRUsage(MF) || !hasLocalExecCopyUses(MF))
    return false;
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnBlock(MBB);
  return Changed;
}

PreservedAnalyses
SIReuseInvariantScalarLoadsPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  if (!CacheInvariantScalarLoads || !SIReuseInvariantScalarLoads(MF).run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
