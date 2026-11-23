//===- AMDGPULDSProfiling.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Number the LDS instructions of a kernel and of its profiling clone, and
/// instrument the clone to measure LDS bank conflicts. See
/// AMDGPULDSProfiling.h for the output format.
///
/// This runs on SSA machine IR after all SSA optimizations, when the kernel
/// and its clone are still identical, so a site id names the same instruction
/// in both.
///
/// For each supported site the clone computes, per wave:
///
///   d[lane]  = (addr[lane] + offset) >> 2
///   active   = lanes which execute the access and are inside the allocation
///   reps     = one lane of active per distinct value of d
///   n        = the largest number of lanes of reps which share a bank, where
///              bank[lane] = d[lane] & (number of banks - 1)
///
/// n is the number of rounds the LDS needs to serve the access: lanes which
/// address the same dword are served together, lanes which address different
/// dwords of the same bank are not.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULDSProfiling.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;
using namespace llvm::AMDGPU::LDSProfiling;

#define DEBUG_TYPE "amdgpu-lds-profiling"

namespace {

struct Site {
  MachineInstr *MI;
  SiteKind Kind;
};

class AMDGPULDSProfilingImpl {
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  SIMachineFunctionInfo &MFI;

  bool isLDSAccess(const MachineInstr &MI) const;
  bool hasLivePhysRegAt(const MachineInstr &MI) const;
  SiteKind classify(const MachineInstr &MI) const;
  void instrument(MachineInstr &MI, unsigned SiteID,
                  const GlobalVariable &Counters);

public:
  AMDGPULDSProfilingImpl(MachineFunction &MF)
      : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()),
        MFI(*MF.getInfo<SIMachineFunctionInfo>()) {}

  bool run();
};

class AMDGPULDSProfilingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULDSProfilingLegacy() : MachineFunctionPass(ID) {}

  // This must not be skipped: the kernel and its clone have to be numbered
  // identically.
  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPULDSProfilingImpl(MF).run();
  }

  StringRef getPassName() const override { return "AMDGPU LDS Profiling"; }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPULDSProfilingLegacy, DEBUG_TYPE, "AMDGPU LDS Profiling",
                false, false)

char AMDGPULDSProfilingLegacy::ID = 0;

char &llvm::AMDGPULDSProfilingLegacyID = AMDGPULDSProfilingLegacy::ID;

PreservedAnalyses
AMDGPULDSProfilingPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  if (!AMDGPULDSProfilingImpl(MF).run())
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses();
}

/// \returns true if \p MI reads or writes LDS through an address operand.
bool AMDGPULDSProfilingImpl::isLDSAccess(const MachineInstr &MI) const {
  if (!TII.isDS(MI) || TII.isGWS(MI) || TII.isAlwaysGDS(MI.getOpcode()))
    return false;
  if (!MI.mayLoad() && !MI.mayStore())
    return false;
  if (!TII.getNamedOperand(MI, AMDGPU::OpName::addr))
    return false;
  const MachineOperand *GDS = TII.getNamedOperand(MI, AMDGPU::OpName::gds);
  return !GDS || GDS->getImm() == 0;
}

/// \returns true if a physical register other than SCC is live before \p MI
/// and used at or after it in its block. Splitting the block at \p MI would
/// need live-in lists which do not exist yet.
bool AMDGPULDSProfilingImpl::hasLivePhysRegAt(const MachineInstr &MI) const {
  const MachineBasicBlock &MBB = *MI.getParent();
  for (const MachineBasicBlock *Succ : MBB.successors())
    if (!Succ->livein_empty())
      return true;

  LiveRegUnits Defs(TRI);
  for (const MachineInstr &I : make_range(MI.getIterator(), MBB.instr_end())) {
    for (const MachineOperand &MO : I.all_uses()) {
      Register Reg = MO.getReg();
      if (!Reg.isPhysical() || Reg == AMDGPU::SCC || MRI.isReserved(Reg) ||
          MRI.isConstantPhysReg(Reg))
        continue;
      for (MCRegUnit Unit : TRI.regunits(Reg))
        if (!Defs.getBitVector().test(static_cast<unsigned>(Unit)))
          return true;
    }
    for (const MachineOperand &MO : I.all_defs())
      if (MO.getReg().isPhysical())
        Defs.addReg(MO.getReg());
    for (const MachineOperand &MO : I.operands())
      if (MO.isRegMask())
        Defs.addRegsInMask(MO.getRegMask());
  }
  return false;
}

SiteKind AMDGPULDSProfilingImpl::classify(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_WRITE_B32:
  case AMDGPU::DS_WRITE_B32_gfx9:
    break;
  default:
    return Unsupported;
  }

  const MachineOperand *Addr = TII.getNamedOperand(MI, AMDGPU::OpName::addr);
  const MachineOperand *Offset =
      TII.getNamedOperand(MI, AMDGPU::OpName::offset);
  if (!Addr->isReg() || !Addr->getReg().isVirtual() || Addr->getSubReg() ||
      !Offset || !Offset->isImm())
    return Unsupported;

  if (hasLivePhysRegAt(MI))
    return Unsupported;

  return OrdinaryB32;
}

/// Insert before \p MI the code which measures its bank conflict and adds the
/// result to the counters of site \p SiteID.
///
///   Head:      d, active, bank; skip to Remainder if no lane is active
///   Loop1:     reps
///   Loop2:     n
///   Publish:   counters[SiteID] += { n, 1, n, 1 }, from one lane
///   Remainder: MI and the rest of the block
void AMDGPULDSProfilingImpl::instrument(MachineInstr &MI, unsigned SiteID,
                                        const GlobalVariable &Counters) {
  MachineBasicBlock &Head = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  const TargetRegisterClass *SRC = &AMDGPU::SReg_32RegClass;
  const TargetRegisterClass *VRC = &AMDGPU::VGPR_32RegClass;
  const TargetRegisterClass *MaskRC = TRI.getWaveMaskRegClass();

  auto SReg = [&] { return MRI.createVirtualRegister(SRC); };
  auto VReg = [&] { return MRI.createVirtualRegister(VRC); };

  // The reduction overwrites SCC.
  Register SaveSCC;
  if (Head.computeRegisterLiveness(&TRI, AMDGPU::SCC, MI,
                                   std::numeric_limits<unsigned>::max()) !=
      MachineBasicBlock::LQR_Dead) {
    SaveSCC = SReg();
    BuildMI(Head, MI, DL, TII.get(AMDGPU::S_CSELECT_B32), SaveSCC)
        .addImm(1)
        .addImm(0);
  }

  MachineBasicBlock *Loop1 = MF.CreateMachineBasicBlock();
  MachineBasicBlock *Loop2 = MF.CreateMachineBasicBlock();
  MachineBasicBlock *Publish = MF.CreateMachineBasicBlock();
  MachineBasicBlock *Remainder = MF.CreateMachineBasicBlock();
  MachineFunction::iterator InsertPt = std::next(Head.getIterator());
  MF.insert(InsertPt, Loop1);
  MF.insert(InsertPt, Loop2);
  MF.insert(InsertPt, Publish);
  MF.insert(InsertPt, Remainder);

  Remainder->transferSuccessorsAndUpdatePHIs(&Head);
  Remainder->splice(Remainder->begin(), &Head, MI.getIterator(), Head.end());

  Head.addSuccessor(Loop1);
  Head.addSuccessor(Remainder);
  Loop1->addSuccessor(Loop1);
  Loop1->addSuccessor(Loop2);
  Loop2->addSuccessor(Loop2);
  Loop2->addSuccessor(Publish);
  Publish->addSuccessor(Remainder);

  // Head.
  Register Addr = TII.getNamedOperand(MI, AMDGPU::OpName::addr)->getReg();
  int64_t Offset = TII.getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();
  if (Offset) {
    Register Sum = VReg();
    BuildMI(&Head, DL, TII.get(AMDGPU::V_ADD_U32_e32), Sum)
        .addImm(Offset)
        .addReg(Addr);
    Addr = Sum;
  }

  Register DWord = VReg();
  BuildMI(&Head, DL, TII.get(AMDGPU::V_LSHRREV_B32_e32), DWord)
      .addImm(2)
      .addReg(Addr);

  // Lanes outside the allocation do not access a bank. The size of dynamic
  // LDS is not known here, so there is no filter for it.
  Register Active = MRI.createVirtualRegister(MaskRC);
  if (MFI.isDynamicLDSUsed()) {
    BuildMI(&Head, DL, TII.get(AMDGPU::S_MOV_B32), Active)
        .addReg(AMDGPU::EXEC_LO);
  } else {
    // The LDS address space is 512 KiB.
    uint32_t LDSWords = std::min<uint32_t>(MFI.getLDSSize(), 1u << 19) / 4;
    Register Limit = SReg();
    BuildMI(&Head, DL, TII.get(AMDGPU::S_MOV_B32), Limit).addImm(LDSWords);
    BuildMI(&Head, DL, TII.get(AMDGPU::V_CMP_LT_U32_e64), Active)
        .addReg(DWord)
        .addReg(Limit);
  }

  Register Bank = VReg();
  BuildMI(&Head, DL, TII.get(AMDGPU::V_AND_B32_e32), Bank)
      .addImm(ST.getLDSBankCount() - 1)
      .addReg(DWord);

  Register Zero = SReg();
  BuildMI(&Head, DL, TII.get(AMDGPU::S_MOV_B32), Zero).addImm(0);
  Register One = SReg();
  BuildMI(&Head, DL, TII.get(AMDGPU::S_MOV_B32), One).addImm(1);

  BuildMI(&Head, DL, TII.get(AMDGPU::S_CMP_EQ_U32)).addReg(Active).addImm(0);
  BuildMI(&Head, DL, TII.get(AMDGPU::S_CBRANCH_SCC1)).addMBB(Remainder);
  BuildMI(&Head, DL, TII.get(AMDGPU::S_BRANCH)).addMBB(Loop1);

  // One iteration of a reduction loop: select the lowest lane of Remaining,
  // and the lanes of Remaining with the same value of Val.
  auto BuildSelectLanes = [&](MachineBasicBlock *MBB, Register Remaining,
                              Register Val, Register &Lane, Register &Same) {
    Lane = SReg();
    BuildMI(MBB, DL, TII.get(AMDGPU::S_FF1_I32_B32), Lane).addReg(Remaining);
    Register LaneVal = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
    BuildMI(MBB, DL, TII.get(AMDGPU::V_READLANE_B32), LaneVal)
        .addReg(Val)
        .addReg(Lane);
    Register Equal = SReg();
    BuildMI(MBB, DL, TII.get(AMDGPU::V_CMP_EQ_U32_e64), Equal)
        .addReg(LaneVal)
        .addReg(Val);
    Same = SReg();
    BuildMI(MBB, DL, TII.get(AMDGPU::S_AND_B32), Same)
        .addReg(Equal)
        .addReg(Remaining)
        .setOperandDead(3); // SCC
  };

  // Remove Same from Remaining and loop while lanes remain.
  auto BuildLoopEnd = [&](MachineBasicBlock *MBB, Register Remaining,
                          Register Same, Register NextRemaining,
                          MachineBasicBlock *Exit) {
    BuildMI(MBB, DL, TII.get(AMDGPU::S_ANDN2_B32), NextRemaining)
        .addReg(Remaining)
        .addReg(Same)
        .setOperandDead(3); // SCC
    BuildMI(MBB, DL, TII.get(AMDGPU::S_CMP_LG_U32))
        .addReg(NextRemaining)
        .addImm(0);
    BuildMI(MBB, DL, TII.get(AMDGPU::S_CBRANCH_SCC1)).addMBB(MBB);
    BuildMI(MBB, DL, TII.get(AMDGPU::S_BRANCH)).addMBB(Exit);
  };

  // Loop1: one representative lane per distinct dword.
  Register Rem = SReg(), NextRem = SReg();
  Register Reps = SReg(), NextReps = SReg();
  BuildMI(Loop1, DL, TII.get(AMDGPU::PHI), Rem)
      .addReg(Active)
      .addMBB(&Head)
      .addReg(NextRem)
      .addMBB(Loop1);
  BuildMI(Loop1, DL, TII.get(AMDGPU::PHI), Reps)
      .addReg(Zero)
      .addMBB(&Head)
      .addReg(NextReps)
      .addMBB(Loop1);
  Register Lane, Same;
  BuildSelectLanes(Loop1, Rem, DWord, Lane, Same);
  Register LaneBit = SReg();
  BuildMI(Loop1, DL, TII.get(AMDGPU::S_LSHL_B32), LaneBit)
      .addReg(One)
      .addReg(Lane)
      .setOperandDead(3); // SCC
  BuildMI(Loop1, DL, TII.get(AMDGPU::S_OR_B32), NextReps)
      .addReg(Reps)
      .addReg(LaneBit)
      .setOperandDead(3); // SCC
  BuildLoopEnd(Loop1, Rem, Same, NextRem, Loop2);

  // Loop2: the largest number of representatives in one bank.
  Register Left = SReg(), NextLeft = SReg();
  Register NWay = SReg(), NextNWay = SReg();
  BuildMI(Loop2, DL, TII.get(AMDGPU::PHI), Left)
      .addReg(NextReps)
      .addMBB(Loop1)
      .addReg(NextLeft)
      .addMBB(Loop2);
  BuildMI(Loop2, DL, TII.get(AMDGPU::PHI), NWay)
      .addReg(Zero)
      .addMBB(Loop1)
      .addReg(NextNWay)
      .addMBB(Loop2);
  BuildSelectLanes(Loop2, Left, Bank, Lane, Same);
  Register Count = SReg();
  BuildMI(Loop2, DL, TII.get(AMDGPU::S_BCNT1_I32_B32), Count)
      .addReg(Same)
      .setOperandDead(2); // SCC
  BuildMI(Loop2, DL, TII.get(AMDGPU::S_MAX_U32), NextNWay)
      .addReg(NWay)
      .addReg(Count)
      .setOperandDead(3); // SCC
  BuildLoopEnd(Loop2, Left, Same, NextLeft, Publish);

  // Publish. A single lane does the adds. Every VGPR used by them has to be
  // written while only that lane is enabled, because it may not have been
  // active before. EXEC is saved and restored with moves, not copies, so that
  // they are not folded away.
  Register SaveExec = MRI.createVirtualRegister(MaskRC);
  BuildMI(Publish, DL, TII.get(AMDGPU::S_MOV_B32), SaveExec)
      .addReg(AMDGPU::EXEC_LO);
  BuildMI(Publish, DL, TII.get(AMDGPU::S_MOV_B32), AMDGPU::EXEC_LO).addImm(1);

  const int64_t RecordOffset = int64_t(SiteID) * SiteRecordSize;
  Register Base =
      MRI.createVirtualRegister(&AMDGPU::SReg_64_XEXEC_XNULLRegClass);
  BuildMI(Publish, DL, TII.get(AMDGPU::SI_PC_ADD_REL_OFFSET64), Base)
      .addGlobalAddress(&Counters, RecordOffset, SIInstrInfo::MO_REL64);
  Register VZero = VReg();
  BuildMI(Publish, DL, TII.get(AMDGPU::V_MOV_B32_e32), VZero).addImm(0);
  Register VNWay = VReg();
  BuildMI(Publish, DL, TII.get(TargetOpcode::COPY), VNWay).addReg(NextNWay);
  Register VOne = VReg();
  BuildMI(Publish, DL, TII.get(AMDGPU::V_MOV_B32_e32), VOne).addImm(1);

  // With a single service group, the total number of rounds is the N-way
  // conflict, and the baseline is one round.
  const Register Values[] = {VNWay, VOne, VNWay, VOne};
  for (unsigned I = 0; I != std::size(Values); ++I) {
    MachineMemOperand *MMO = MF.getMachineMemOperand(
        MachinePointerInfo(&Counters, RecordOffset + I * 4),
        MachineMemOperand::MOLoad | MachineMemOperand::MOStore, LLT::scalar(32),
        Align(4), MMOMetadata(), SyncScope::System, AtomicOrdering::Monotonic);
    BuildMI(Publish, DL, TII.get(AMDGPU::GLOBAL_ATOMIC_ADD_SADDR))
        .addReg(VZero)     // vaddr
        .addReg(Values[I]) // vdata
        .addReg(Base)      // saddr
        .addImm(I * 4)     // offset
        .addImm(0)         // cpol
        .addMemOperand(MMO);
  }

  BuildMI(Publish, DL, TII.get(AMDGPU::S_MOV_B32), AMDGPU::EXEC_LO)
      .addReg(SaveExec);
  BuildMI(Publish, DL, TII.get(AMDGPU::S_BRANCH)).addMBB(Remainder);

  // Remainder.
  if (SaveSCC) {
    BuildMI(*Remainder, Remainder->begin(), DL, TII.get(AMDGPU::S_CMP_LG_U32))
        .addReg(SaveSCC)
        .addImm(0);
  }
}

bool AMDGPULDSProfilingImpl::run() {
  const Function &F = MF.getFunction();
  bool IsClone = F.hasFnAttribute(CloneAttr);
  if (!IsClone && !F.hasFnAttribute(OriginalAttr))
    return false;

  // Number the sites. The kernel and its clone are identical here, so this
  // gives the same ids and kinds in both.
  SmallVector<Site, 16> Sites;
  SmallString<256> Signature;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isLDSAccess(MI))
        continue;
      SiteKind Kind = classify(MI);
      Sites.push_back({&MI, Kind});

      // Hash what identifies the access: the instruction and its offsets.
      Signature += TII.getName(MI.getOpcode());
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isImm()) {
          Signature += ' ';
          Signature += utostr(static_cast<uint64_t>(MO.getImm()));
        }
      }
      Signature += ' ';
      Signature += utostr(Kind);
      Signature += ';';
    }
  }

  MFI.setLDSProfilingSites(Sites.size(),
                           static_cast<uint32_t>(xxh3_64bits(Signature)));
  if (Sites.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  for (unsigned ID = 0, E = Sites.size(); ID != E; ++ID) {
    Metadata *Aux[] = {
        ConstantAsMetadata::get(ConstantInt::get(I32, ID)),
        ConstantAsMetadata::get(ConstantInt::get(I32, Sites[ID].Kind))};
    Metadata *Ops[] = {MDString::get(Ctx, SitesSection), MDNode::get(Ctx, Aux)};
    Sites[ID].MI->setPCSections(MF, MDNode::get(Ctx, Ops));
  }

  if (!IsClone)
    return true;

  const GlobalVariable *Counters =
      F.getParent()->getNamedGlobal((F.getName() + CountersSuffix).str());
  if (!Counters)
    return true;

  for (unsigned ID = 0, E = Sites.size(); ID != E; ++ID) {
    if (Sites[ID].Kind == Unsupported)
      continue;
    // The reduction loops have PHIs.
    MF.getProperties().resetNoPHIs();
    instrument(*Sites[ID].MI, ID, *Counters);
  }

  return true;
}
