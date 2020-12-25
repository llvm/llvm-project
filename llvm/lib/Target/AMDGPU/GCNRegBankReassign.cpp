//===-- GCNRegBankReassign.cpp - Reassign registers after regalloc --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Try to reassign registers on GFX10+ to reduce register bank
/// conflicts.
///
/// On GFX10 registers are organized in banks. VGPRs have 4 banks assigned in
/// a round-robin fashion: v0, v4, v8... belong to bank 0. v1, v5, v9... to
/// bank 1, etc. SGPRs have 8 banks and allocated in pairs, so that s0:s1,
/// s16:s17, s32:s33 are at bank 0. s2:s3, s18:s19, s34:s35 are at bank 1 etc.
///
/// The shader can read one dword from each of these banks once per cycle.
/// If an instruction has to read more register operands from the same bank
/// an additional cycle is needed. HW attempts to pre-load registers through
/// input operand gathering, but a stall cycle may occur if that fails. For
/// example V_FMA_F32 V111 = V0 + V4 * V8 will need 3 cycles to read operands,
/// potentially incuring 2 stall cycles.
///
/// The pass tries to reassign registers to reduce bank conflicts.
///
/// In this pass bank numbers 0-3 are VGPR banks and 4-11 are SGPR banks, so
/// that 4 has to be subtracted from an SGPR bank number to get the real value.
/// This also corresponds to bit numbers in bank masks used in the pass.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

static cl::opt<unsigned> VerifyStallCycles("amdgpu-verify-regbanks-reassign",
  cl::desc("Verify stall cycles in the regbanks reassign pass"),
  cl::value_desc("0|1|2"),
  cl::init(0), cl::Hidden);

#define DEBUG_TYPE "amdgpu-regbanks-reassign"

#define NUM_VGPR_BANKS 4
#define NUM_SGPR_BANKS 8
#define NUM_BANKS (NUM_VGPR_BANKS + NUM_SGPR_BANKS)
#define SGPR_BANK_OFFSET NUM_VGPR_BANKS
#define VGPR_BANK_MASK 0xf
#define SGPR_BANK_MASK 0xff0
#define SGPR_BANK_SHIFTED_MASK (SGPR_BANK_MASK >> SGPR_BANK_OFFSET)

STATISTIC(NumStallsDetected,
          "Number of operand read stalls detected");
STATISTIC(NumStallsRecovered,
          "Number of operand read stalls recovered");

namespace {

class GCNRegBankReassign : public MachineFunctionPass {

  class OperandMask {
  public:
    OperandMask(unsigned r, unsigned s, unsigned m)
      : Reg(r), SubReg(s), Mask(m) {}
    Register Reg;
    unsigned SubReg;
    unsigned Mask;
  };

  class Candidate {
  public:
    Candidate(MachineInstr *mi, Register reg, unsigned subreg,
              unsigned freebanks)
        : MI(mi), Reg(reg), SubReg(subreg), FreeBanks(freebanks) {}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump(const GCNRegBankReassign *P) const {
      MI->dump();
      dbgs() << P->printReg(Reg) << " to banks ";
      dumpFreeBanks(FreeBanks);
      dbgs() << '\n';
    }
#endif

    MachineInstr *MI;
    Register Reg;
    unsigned SubReg;
    unsigned FreeBanks;
  };

  class CandidateList : public std::map<unsigned, std::list<Candidate>> {
  public:
    void push(unsigned Weight, const Candidate&& C) {
      operator[](Weight).push_front(C);
    }

    Candidate &back() {
      return rbegin()->second.back();
    }

    void pop_back() {
      rbegin()->second.pop_back();
      if (rbegin()->second.empty())
        erase(rbegin()->first);
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump(const GCNRegBankReassign *P) const {
      dbgs() << "\nCandidates:\n\n";
      for (auto &B : *this) {
        dbgs() << " Weight " << B.first << ":\n";
        for (auto &C : B.second)
          C.dump(P);
      }
      dbgs() << "\n\n";
    }
#endif
  };

public:
  static char ID;

public:
  GCNRegBankReassign() : MachineFunctionPass(ID) {
    initializeGCNRegBankReassignPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "GCN RegBank Reassign"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<LiveIntervals>();
    AU.addRequired<VirtRegMap>();
    AU.addRequired<LiveRegMatrix>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const GCNSubtarget *ST;

  const MachineRegisterInfo *MRI;

  const SIRegisterInfo *TRI;

  MachineLoopInfo *MLI;

  VirtRegMap *VRM;

  LiveRegMatrix *LRM;

  LiveIntervals *LIS;

  unsigned MaxNumVGPRs;

  unsigned MaxNumSGPRs;

  BitVector RegsUsed;

  SmallVector<OperandMask, 8> OperandMasks;

  CandidateList Candidates;

  const MCPhysReg *CSRegs;

  // Returns bank for a phys reg.
  unsigned getPhysRegBank(Register Reg, unsigned SubReg) const;

  // Return a bit set for each register bank used. 4 banks for VGPRs and
  // 8 banks for SGPRs.
  // Registers already processed and recorded in RegsUsed are excluded.
  // If Bank is not -1 assume Reg:SubReg to belong to that Bank.
  uint32_t getRegBankMask(Register Reg, unsigned SubReg, int Bank);

  // Analyze one instruction returning the number of stalls and a mask of the
  // banks used by all operands.
  // If Reg and Bank are provided, assume all uses of Reg will be replaced with
  // a register chosen from Bank.
  std::pair<unsigned, unsigned> analyzeInst(const MachineInstr &MI,
                                            Register Reg = Register(),
                                            unsigned SubReg = 0, int Bank = -1);

  // Return true if register is regular VGPR or SGPR or their tuples.
  // Returns false for special registers like m0, vcc etc.
  bool isReassignable(Register Reg) const;

  // Check if registers' defs are old and may be pre-loaded.
  // Returns 0 if both registers are old enough, 1 or 2 if one or both
  // registers will not likely be pre-loaded.
  unsigned getOperandGatherWeight(const MachineInstr& MI,
                                  Register Reg1,
                                  Register Reg2,
                                  unsigned StallCycles) const;


  // Find all bank bits in UsedBanks where Mask can be relocated to.
  unsigned getFreeBanks(unsigned Mask, unsigned UsedBanks) const;

  // Find all bank bits in UsedBanks where Mask can be relocated to.
  // Bank is relative to the register and not its subregister component.
  // Returns 0 is a register is not reassignable.
  unsigned getFreeBanks(Register Reg, unsigned SubReg, unsigned Mask,
                        unsigned UsedBanks) const;

  // Add cadidate instruction to the work list.
  void collectCandidates(MachineInstr& MI, unsigned UsedBanks,
                         unsigned StallCycles);

  // Collect cadidate instructions across function. Returns a number stall
  // cycles detected. Only counts stalls if Collect is false.
  unsigned collectCandidates(MachineFunction &MF, bool Collect = true);

  // Remove all candidates that read specified register.
  void removeCandidates(Register Reg);

  // Compute stalls within the uses of SrcReg replaced by a register from
  // Bank. If Bank is -1 does not perform substitution. If Collect is set
  // candidates are collected and added to work list.
  unsigned computeStallCycles(Register SrcReg,
                              Register Reg = Register(),
                              unsigned SubReg = 0, int Bank = -1,
                              bool Collect = false);

  // Search for a register in Bank unused within LI.
  // Returns phys reg or NoRegister.
  MCRegister scavengeReg(LiveInterval &LI, unsigned Bank,
                         unsigned SubReg) const;

  // Try to reassign candidate. Returns number or stall cycles saved.
  unsigned tryReassign(Candidate &C);

  bool verifyCycles(MachineFunction &MF,
                    unsigned OriginalCycles, unsigned CyclesSaved);


#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
public:
  Printable printReg(Register Reg, unsigned SubReg = 0) const {
    return Printable([Reg, SubReg, this](raw_ostream &OS) {
      if (Reg.isPhysical()) {
        OS << llvm::printReg(Reg, TRI);
        return;
      }
      if (!VRM->isAssignedReg(Reg))
        OS << "<unassigned> " << llvm::printReg(Reg, TRI);
      else
        OS << llvm::printReg(Reg, TRI) << '('
           << llvm::printReg(VRM->getPhys(Reg), TRI) << ')';
      if (SubReg)
        OS << ':' << TRI->getSubRegIndexName(SubReg);
    });
  }

  static Printable printBank(unsigned Bank) {
    return Printable([Bank](raw_ostream &OS) {
      OS << ((Bank >= SGPR_BANK_OFFSET) ? Bank - SGPR_BANK_OFFSET : Bank);
    });
  }

  static void dumpFreeBanks(unsigned FreeBanks) {
    for (unsigned L = 0; L < NUM_BANKS; ++L)
      if (FreeBanks & (1 << L))
        dbgs() << printBank(L) << ' ';
  }
#endif
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNRegBankReassign, DEBUG_TYPE, "GCN RegBank Reassign",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(GCNRegBankReassign, DEBUG_TYPE, "GCN RegBank Reassign",
                    false, false)


char GCNRegBankReassign::ID = 0;

char &llvm::GCNRegBankReassignID = GCNRegBankReassign::ID;

unsigned GCNRegBankReassign::getPhysRegBank(Register Reg,
                                            unsigned SubReg) const {
  assert(Reg.isPhysical());

  const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
  unsigned Size = TRI->getRegSizeInBits(*RC);
  if (Size == 16)
    Reg = TRI->get32BitRegister(Reg);
  else if (Size > 32) {
    if (SubReg) {
      const TargetRegisterClass *SubRC = TRI->getSubRegClass(RC, SubReg);
      Reg = TRI->getSubReg(Reg, SubReg);
      if (TRI->getRegSizeInBits(*SubRC) > 32)
        Reg = TRI->getSubReg(Reg, AMDGPU::sub0);
    } else {
      Reg = TRI->getSubReg(Reg, AMDGPU::sub0);
    }
  }

  if (TRI->hasVGPRs(RC)) {
    unsigned RegNo = Reg - AMDGPU::VGPR0;
    return RegNo % NUM_VGPR_BANKS;
  }

  unsigned RegNo = TRI->getEncodingValue(AMDGPU::getMCReg(Reg, *ST)) / 2;
  return RegNo % NUM_SGPR_BANKS + SGPR_BANK_OFFSET;
}

uint32_t GCNRegBankReassign::getRegBankMask(Register Reg, unsigned SubReg,
                                            int Bank) {
  if (Reg.isVirtual()) {
    if (!VRM->isAssignedReg(Reg))
      return 0;

    Reg = VRM->getPhys(Reg);
    if (!Reg)
      return 0;
    if (SubReg)
      Reg = TRI->getSubReg(Reg, SubReg);
  }

  const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
  unsigned Size = TRI->getRegSizeInBits(*RC);

  if (Size == 16) {
    Reg = TRI->get32BitRegister(Reg);
    Size = 1;
  } else {
    Size /= 32;
    if (Size > 1)
      Reg = TRI->getSubReg(Reg, AMDGPU::sub0);
  }

  if (TRI->hasVGPRs(RC)) {
    // VGPRs have 4 banks assigned in a round-robin fashion.
    unsigned RegNo = Reg - AMDGPU::VGPR0;
    uint32_t Mask = maskTrailingOnes<uint32_t>(Size);
    unsigned Used = 0;
    // Bitmask lacks an extract method
    for (unsigned I = 0; I < Size; ++I)
      if (RegsUsed.test(RegNo + I))
        Used |= 1 << I;
    RegsUsed.set(RegNo, RegNo + Size);
    Mask &= ~Used;
    Mask <<= (Bank == -1) ? RegNo % NUM_VGPR_BANKS : uint32_t(Bank);
    return (Mask | (Mask >> NUM_VGPR_BANKS)) & VGPR_BANK_MASK;
  }

  // SGPRs have 8 banks holding 2 consequitive registers each.
  unsigned RegNo = TRI->getEncodingValue(AMDGPU::getMCReg(Reg, *ST)) / 2;
  unsigned StartBit = AMDGPU::VGPR_32RegClass.getNumRegs();
  if (RegNo + StartBit >= RegsUsed.size())
    return 0;

  if (Size > 1)
    Size /= 2;
  unsigned Mask = (1 << Size) - 1;
  unsigned Used = 0;
  for (unsigned I = 0; I < Size; ++I)
    if (RegsUsed.test(StartBit + RegNo + I))
      Used |= 1 << I;
  RegsUsed.set(StartBit + RegNo, StartBit + RegNo + Size);
  Mask &= ~Used;
  Mask <<= (Bank == -1) ? RegNo % NUM_SGPR_BANKS
                        : unsigned(Bank - SGPR_BANK_OFFSET);
  Mask = (Mask | (Mask >> NUM_SGPR_BANKS)) & SGPR_BANK_SHIFTED_MASK;
  // Reserve 4 bank ids for VGPRs.
  return Mask << SGPR_BANK_OFFSET;
}

std::pair<unsigned, unsigned>
GCNRegBankReassign::analyzeInst(const MachineInstr &MI, Register Reg,
                                unsigned SubReg, int Bank) {
  unsigned StallCycles = 0;
  unsigned UsedBanks = 0;

  if (MI.isDebugValue())
    return std::make_pair(StallCycles, UsedBanks);

  RegsUsed.reset();
  OperandMasks.clear();
  for (const auto& Op : MI.explicit_uses()) {
    // Undef can be assigned to any register, so two vregs can be assigned
    // the same phys reg within the same instruction.
    if (!Op.isReg() || Op.isUndef())
      continue;

    const Register R = Op.getReg();
    const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, R);

    // Do not compute stalls for AGPRs
    if (TRI->hasAGPRs(RC))
      continue;

    // Do not compute stalls if sub-register covers all banks
    if (Op.getSubReg()) {
      LaneBitmask LM = TRI->getSubRegIndexLaneMask(Op.getSubReg());
      if (TRI->hasVGPRs(RC)) {
        if (TRI->getNumCoveredRegs(LM) >= NUM_VGPR_BANKS)
          continue;
      } else {
        if (TRI->getNumCoveredRegs(LM) / 2 >= NUM_SGPR_BANKS)
          continue;
      }
    }

    unsigned ShiftedBank = Bank;

    if (Bank != -1 && R == Reg && (Op.getSubReg() || SubReg)) {
      unsigned RegOffset =
          TRI->getChannelFromSubReg(SubReg ? SubReg : (unsigned)AMDGPU::sub0);
      unsigned Offset = TRI->getChannelFromSubReg(
          Op.getSubReg() ? Op.getSubReg() : (unsigned)AMDGPU::sub0);
      if (Bank < NUM_VGPR_BANKS) {
        unsigned Shift = ((NUM_VGPR_BANKS + Offset) - RegOffset);
        ShiftedBank = (Bank + Shift) % NUM_VGPR_BANKS;
      } else if (Bank >= SGPR_BANK_OFFSET) {
        unsigned Shift = (NUM_SGPR_BANKS + (Offset >> 1)) - (RegOffset >> 1);
        ShiftedBank = SGPR_BANK_OFFSET +
                      (Bank - SGPR_BANK_OFFSET + Shift) % NUM_SGPR_BANKS;
      }
    }

    uint32_t Mask = getRegBankMask(R, Op.getSubReg(),
                                   (Reg == R) ? ShiftedBank : -1);
    StallCycles += countPopulation(UsedBanks & Mask);
    UsedBanks |= Mask;
    OperandMasks.push_back(OperandMask(Op.getReg(), Op.getSubReg(), Mask));
  }

  return std::make_pair(StallCycles, UsedBanks);
}

unsigned GCNRegBankReassign::getOperandGatherWeight(const MachineInstr& MI,
                                                    Register Reg1,
                                                    Register Reg2,
                                                    unsigned StallCycles) const
{
  unsigned Defs = 0;
  MachineBasicBlock::const_instr_iterator Def(MI.getIterator());
  MachineBasicBlock::const_instr_iterator B(MI.getParent()->instr_begin());
  for (unsigned S = StallCycles; S && Def != B && Defs != 3; --S) {
    if (MI.isDebugInstr())
      continue;
    --Def;
    if (Def->getOpcode() == TargetOpcode::IMPLICIT_DEF)
      continue;
    if (Def->modifiesRegister(Reg1, TRI))
      Defs |= 1;
    if (Def->modifiesRegister(Reg2, TRI))
      Defs |= 2;
  }
  return countPopulation(Defs);
}

bool GCNRegBankReassign::isReassignable(Register Reg) const {
  if (Reg.isPhysical() || !VRM->isAssignedReg(Reg))
    return false;

  const MachineInstr *Def = MRI->getUniqueVRegDef(Reg);

  Register PhysReg = VRM->getPhys(Reg);

  if (Def && Def->isCopy() && Def->getOperand(1).getReg() == PhysReg)
    return false;

  for (auto U : MRI->use_nodbg_operands(Reg)) {
    if (U.isImplicit())
      return false;
    const MachineInstr *UseInst = U.getParent();
    if (UseInst->isCopy() && UseInst->getOperand(0).getReg() == PhysReg)
      return false;
  }

  const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(PhysReg);
  unsigned Size = TRI->getRegSizeInBits(*RC);

  // TODO: Support 16 bit registers. Those needs to be moved with their
  //       parent VGPR_32 and potentially a sibling 16 bit sub-register.
  if (Size < 32)
    return false;

  if (TRI->hasVGPRs(RC))
    return true;

  if (Size == 16)
    return AMDGPU::SGPR_LO16RegClass.contains(PhysReg);

  if (Size > 32)
    PhysReg = TRI->getSubReg(PhysReg, AMDGPU::sub0);

  return AMDGPU::SGPR_32RegClass.contains(PhysReg);
}

unsigned GCNRegBankReassign::getFreeBanks(unsigned Mask,
                                          unsigned UsedBanks) const {
  unsigned Size = countPopulation(Mask);
  unsigned FreeBanks = 0;
  unsigned Bank = findFirstSet(Mask);

  UsedBanks &= ~Mask;

  // Find free VGPR banks
  if ((Mask & VGPR_BANK_MASK) && (Size < NUM_VGPR_BANKS)) {
    for (unsigned I = 0; I < NUM_VGPR_BANKS; ++I) {
      if (Bank == I)
        continue;
      unsigned NewMask = ((1 << Size) - 1) << I;
      NewMask = (NewMask | (NewMask >> NUM_VGPR_BANKS)) & VGPR_BANK_MASK;
      if (!(UsedBanks & NewMask))
        FreeBanks |= 1 << I;
    }
    return FreeBanks;
  }

  // Find free SGPR banks
  // SGPR tuples must be aligned, so step is size in banks it
  // crosses.
  Bank -= SGPR_BANK_OFFSET;
  for (unsigned I = 0; I < NUM_SGPR_BANKS; I += Size) {
    if (Bank == I)
      continue;
    unsigned NewMask = ((1 << Size) - 1) << I;
    NewMask = (NewMask | (NewMask >> NUM_SGPR_BANKS)) & SGPR_BANK_SHIFTED_MASK;
    if (!(UsedBanks & (NewMask << SGPR_BANK_OFFSET)))
      FreeBanks |= (1 << SGPR_BANK_OFFSET) << I;
  }

  return FreeBanks;
}

unsigned GCNRegBankReassign::getFreeBanks(Register Reg,
                                          unsigned SubReg,
                                          unsigned Mask,
                                          unsigned UsedBanks) const {
  if (!isReassignable(Reg))
    return 0;

  unsigned FreeBanks = getFreeBanks(Mask, UsedBanks);

  unsigned Offset = TRI->getChannelFromSubReg(SubReg);
  if (Offset && (Mask & VGPR_BANK_MASK)) {
    unsigned Shift = Offset;
    if (Shift >= NUM_VGPR_BANKS)
      return 0;
    unsigned VB = FreeBanks & VGPR_BANK_MASK;
    FreeBanks = ((VB >> Shift) | (VB << (NUM_VGPR_BANKS - Shift))) &
                VGPR_BANK_MASK;
  } else if (Offset > 1 && (Mask & SGPR_BANK_MASK)) {
    unsigned Shift = Offset >> 1;
    if (Shift >= NUM_SGPR_BANKS)
      return 0;
    unsigned SB = FreeBanks >> SGPR_BANK_OFFSET;
    FreeBanks = ((SB >> Shift) | (SB << (NUM_SGPR_BANKS - Shift))) &
                SGPR_BANK_SHIFTED_MASK;
    FreeBanks <<= SGPR_BANK_OFFSET;
  }

  LLVM_DEBUG(if (FreeBanks) {
          dbgs() << "Potential reassignments of " << printReg(Reg, SubReg)
                 << " to banks: "; dumpFreeBanks(FreeBanks);
          dbgs() << '\n'; });

  return FreeBanks;
}

void GCNRegBankReassign::collectCandidates(MachineInstr& MI,
                                           unsigned UsedBanks,
                                           unsigned StallCycles) {
  LLVM_DEBUG(MI.dump());

  if (!StallCycles)
    return;

  LLVM_DEBUG(dbgs() << "Stall cycles = " << StallCycles << '\n');

  for (unsigned I = 0, E = OperandMasks.size(); I + 1 < E; ++I) {
    for (unsigned J = I + 1; J != E; ++J) {
      if (!(OperandMasks[I].Mask & OperandMasks[J].Mask))
        continue;

      Register Reg1 = OperandMasks[I].Reg;
      Register Reg2 = OperandMasks[J].Reg;
      unsigned SubReg1 = OperandMasks[I].SubReg;
      unsigned SubReg2 = OperandMasks[J].SubReg;
      unsigned Mask1 = OperandMasks[I].Mask;
      unsigned Mask2 = OperandMasks[J].Mask;
      unsigned Size1 = countPopulation(Mask1);
      unsigned Size2 = countPopulation(Mask2);

      LLVM_DEBUG(dbgs() << "Conflicting operands: " << printReg(Reg1, SubReg1) <<
                      " and " << printReg(Reg2, SubReg2) << '\n');

      unsigned Weight = getOperandGatherWeight(MI, Reg1, Reg2, StallCycles);
      Weight += MLI->getLoopDepth(MI.getParent()) * 10;

      LLVM_DEBUG(dbgs() << "Stall weight = " << Weight << '\n');

      unsigned FreeBanks1 = getFreeBanks(Reg1, SubReg1, Mask1, UsedBanks);
      unsigned FreeBanks2 = getFreeBanks(Reg2, SubReg2, Mask2, UsedBanks);
      if (FreeBanks1)
        Candidates.push(Weight + ((Size2 > Size1) ? 1 : 0),
                        Candidate(&MI, Reg1, SubReg1, FreeBanks1));
      if (FreeBanks2)
        Candidates.push(Weight + ((Size1 > Size2) ? 1 : 0),
                        Candidate(&MI, Reg2, SubReg2, FreeBanks2));
    }
  }
}

unsigned GCNRegBankReassign::computeStallCycles(Register SrcReg, Register Reg,
                                                unsigned SubReg, int Bank,
                                                bool Collect) {
  unsigned TotalStallCycles = 0;
  SmallSet<const MachineInstr *, 16> Visited;

  for (auto &MI : MRI->use_nodbg_instructions(SrcReg)) {
    if (MI.isBundle())
      continue;
    if (!Visited.insert(&MI).second)
      continue;
    unsigned StallCycles;
    unsigned UsedBanks;
    std::tie(StallCycles, UsedBanks) = analyzeInst(MI, Reg, SubReg, Bank);
    TotalStallCycles += StallCycles;
    if (Collect)
      collectCandidates(MI, UsedBanks, StallCycles);
  }

  return TotalStallCycles;
}

MCRegister GCNRegBankReassign::scavengeReg(LiveInterval &LI, unsigned Bank,
                                           unsigned SubReg) const {
  const TargetRegisterClass *RC = MRI->getRegClass(LI.reg());
  unsigned MaxNumRegs = (Bank < NUM_VGPR_BANKS) ? MaxNumVGPRs
                                                : MaxNumSGPRs;
  unsigned MaxReg = MaxNumRegs + (Bank < NUM_VGPR_BANKS ? AMDGPU::VGPR0
                                                        : AMDGPU::SGPR0);

  for (MCRegister Reg : RC->getRegisters()) {
    // Check occupancy limit.
    if (TRI->isSubRegisterEq(Reg, MaxReg))
      break;

    if (!MRI->isAllocatable(Reg) || getPhysRegBank(Reg, SubReg) != Bank)
      continue;

    for (unsigned I = 0; CSRegs[I]; ++I)
      if (TRI->isSubRegisterEq(Reg, CSRegs[I]) &&
          !LRM->isPhysRegUsed(CSRegs[I]))
        return MCRegister::from(AMDGPU::NoRegister);

    LLVM_DEBUG(dbgs() << "Trying register " << printReg(Reg) << '\n');

    if (!LRM->checkInterference(LI, Reg))
      return Reg;
  }

  return MCRegister::from(AMDGPU::NoRegister);
}

unsigned GCNRegBankReassign::tryReassign(Candidate &C) {
  if (!LIS->hasInterval(C.Reg))
    return 0;

  LiveInterval &LI = LIS->getInterval(C.Reg);
  LLVM_DEBUG(dbgs() << "Try reassign " << printReg(C.Reg) << " in "; C.MI->dump();
             LI.dump());

  // For each candidate bank walk all instructions in the range of live
  // interval and check if replacing the register with one belonging to
  // the candidate bank reduces conflicts.

  unsigned OrigStalls = computeStallCycles(C.Reg);
  LLVM_DEBUG(dbgs() << "--- Stall cycles in range = " << OrigStalls << '\n');
  if (!OrigStalls)
    return 0;

  struct BankStall {
    BankStall(unsigned b, unsigned s) : Bank(b), Stalls(s) {};
    bool operator<(const BankStall &RHS) const {
      if (Stalls == RHS.Stalls)
        return Bank < RHS.Bank;
      return Stalls > RHS.Stalls;
    }
    unsigned Bank;
    unsigned Stalls;
  };
  SmallVector<BankStall, 8> BankStalls;

  for (int Bank = 0; Bank < NUM_BANKS; ++Bank) {
    if (C.FreeBanks & (1 << Bank)) {
      LLVM_DEBUG(dbgs() << "Trying bank " << printBank(Bank) << '\n');
      unsigned Stalls = computeStallCycles(C.Reg, C.Reg, C.SubReg, Bank);
      if (Stalls < OrigStalls) {
        LLVM_DEBUG(dbgs() << "With bank " << printBank(Bank) << " -> "
                     << Stalls << '\n');
        BankStalls.push_back(BankStall((unsigned)Bank, Stalls));
      }
    }
  }
  llvm::sort(BankStalls);

  MCRegister OrigReg = VRM->getPhys(C.Reg);
  LRM->unassign(LI);
  while (!BankStalls.empty()) {
    BankStall BS = BankStalls.pop_back_val();
    MCRegister Reg = scavengeReg(LI, BS.Bank, C.SubReg);
    if (Reg == AMDGPU::NoRegister) {
      LLVM_DEBUG(dbgs() << "No free registers in bank " << printBank(BS.Bank)
                   << '\n');
      continue;
    }
    LLVM_DEBUG(dbgs() << "Found free register " << printReg(Reg)
                 << (LRM->isPhysRegUsed(Reg) ? "" : " (new)")
                 << " in bank " << printBank(BS.Bank) << '\n');

    LRM->assign(LI, Reg);

    LLVM_DEBUG(dbgs() << "--- Cycles saved: " << OrigStalls - BS.Stalls << '\n');

    return OrigStalls - BS.Stalls;
  }
  LRM->assign(LI, OrigReg);

  return 0;
}

unsigned GCNRegBankReassign::collectCandidates(MachineFunction &MF,
                                               bool Collect) {
  unsigned TotalStallCycles = 0;

  for (MachineBasicBlock &MBB : MF) {

    LLVM_DEBUG(if (Collect) {
            if (MBB.getName().empty()) dbgs() << "bb." << MBB.getNumber();
            else dbgs() << MBB.getName(); dbgs() << ":\n";
          });

    for (MachineInstr &MI : MBB.instrs()) {
      if (MI.isBundle())
          continue; // we analyze the instructions inside the bundle individually

      unsigned StallCycles;
      unsigned UsedBanks;
      std::tie(StallCycles, UsedBanks) = analyzeInst(MI);

      if (Collect)
        collectCandidates(MI, UsedBanks, StallCycles);

      TotalStallCycles += StallCycles;
    }

    LLVM_DEBUG(if (Collect) { dbgs() << '\n'; });
  }

  return TotalStallCycles;
}

void GCNRegBankReassign::removeCandidates(Register Reg) {
  typename CandidateList::iterator Next;
  for (auto I = Candidates.begin(), E = Candidates.end(); I != E; I = Next) {
    Next = std::next(I);
    I->second.remove_if([Reg, this](const Candidate& C) {
      return C.MI->readsRegister(Reg, TRI);
    });
    if (I->second.empty())
      Candidates.erase(I);
  }
}

bool GCNRegBankReassign::verifyCycles(MachineFunction &MF,
                                      unsigned OriginalCycles,
                                      unsigned CyclesSaved) {
  unsigned StallCycles = collectCandidates(MF, false);
  LLVM_DEBUG(dbgs() << "=== After the pass " << StallCycles
               << " stall cycles left\n");
  return StallCycles + CyclesSaved == OriginalCycles;
}

bool GCNRegBankReassign::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->hasRegisterBanking() || skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  TRI = ST->getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfo>();
  VRM = &getAnalysis<VirtRegMap>();
  LRM = &getAnalysis<LiveRegMatrix>();
  LIS = &getAnalysis<LiveIntervals>();

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned Occupancy = MFI->getOccupancy();
  MaxNumVGPRs = ST->getMaxNumVGPRs(MF);
  MaxNumSGPRs = ST->getMaxNumSGPRs(MF);
  MaxNumVGPRs = std::min(ST->getMaxNumVGPRs(Occupancy), MaxNumVGPRs);
  MaxNumSGPRs = std::min(ST->getMaxNumSGPRs(Occupancy, true), MaxNumSGPRs);

  CSRegs = MRI->getCalleeSavedRegs();
  unsigned NumRegBanks = AMDGPU::VGPR_32RegClass.getNumRegs() +
                         // Not a tight bound
                         AMDGPU::SReg_32RegClass.getNumRegs() / 2 + 1;
  RegsUsed.resize(NumRegBanks);

  LLVM_DEBUG(dbgs() << "=== RegBanks reassign analysis on function " << MF.getName()
               << '\n');

  unsigned StallCycles = collectCandidates(MF);
  NumStallsDetected += StallCycles;

  LLVM_DEBUG(dbgs() << "=== " << StallCycles << " stall cycles detected in "
                  "function " << MF.getName() << '\n');

  LLVM_DEBUG(Candidates.dump(this));

  unsigned CyclesSaved = 0;
  while (!Candidates.empty()) {
    Candidate C = Candidates.back();
    unsigned LocalCyclesSaved = tryReassign(C);
    CyclesSaved += LocalCyclesSaved;

    if (VerifyStallCycles > 1 && !verifyCycles(MF, StallCycles, CyclesSaved))
      report_fatal_error("RegBank reassign stall cycles verification failed.");

    Candidates.pop_back();
    if (LocalCyclesSaved) {
      removeCandidates(C.Reg);
      computeStallCycles(C.Reg, AMDGPU::NoRegister, 0, -1, true);

      LLVM_DEBUG(Candidates.dump(this));
    }
  }
  NumStallsRecovered += CyclesSaved;

  LLVM_DEBUG(dbgs() << "=== After the pass " << CyclesSaved
               << " cycles saved in function " << MF.getName() << '\n');

  Candidates.clear();

  if (VerifyStallCycles == 1 && !verifyCycles(MF, StallCycles, CyclesSaved))
    report_fatal_error("RegBank reassign stall cycles verification failed.");

  RegsUsed.clear();

  return CyclesSaved > 0;
}
