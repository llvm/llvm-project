//===-- SIFixSGPRCopies.cpp - Remove potential VGPR => SGPR copies --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copies from VGPR to SGPR registers are illegal and the register coalescer
/// will sometimes generate these illegal copies in situations like this:
///
///  Register Class <vsrc> is the union of <vgpr> and <sgpr>
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///   %vreg1 <vsrc> = COPY %vreg0 <sgpr>
///    ...
///    BRANCH %cond BB1, BB2
///  BB1:
///    %vreg2 <vgpr> = VECTOR_INST
///    %vreg3 <vsrc> = COPY %vreg2 <vgpr>
///  BB2:
///    %vreg4 <vsrc> = PHI %vreg1 <vsrc>, <BB#0>, %vreg3 <vrsc>, <BB#1>
///    %vreg5 <vgpr> = VECTOR_INST %vreg4 <vsrc>
///
///
/// The coalescer will begin at BB0 and eliminate its copy, then the resulting
/// code will look like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <vsrc> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <vsrc>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now that the result of the PHI instruction is an SGPR, the register
/// allocator is now forced to constrain the register class of %vreg3 to
/// <sgpr> so we end up with final code like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <sgpr> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <sgpr>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now this code contains an illegal copy from a VGPR to an SGPR.
///
/// In order to avoid this problem, this pass searches for PHI instructions
/// which define a <vsrc> register and constrains its definition class to
/// <vgpr> if the user of the PHI's definition register is a vector instruction.
/// If the PHI's definition class is constrained to <vgpr> then the coalescer
/// will be unable to perform the COPY removal from the above example  which
/// ultimately led to the creation of an illegal COPY.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sgpr-copies"

static cl::opt<bool> EnableM0Merge(
  "amdgpu-enable-merge-m0",
  cl::desc("Merge and hoist M0 initializations"),
  cl::init(false));

namespace {

class SIFixSGPRCopies : public MachineFunctionPass {

  MachineDominatorTree *MDT;

public:
  static char ID;

  SIFixSGPRCopies() : MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Fix SGPR copies"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace

INITIALIZE_PASS_BEGIN(SIFixSGPRCopies, DEBUG_TYPE,
                     "SI Fix SGPR copies", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SIFixSGPRCopies, DEBUG_TYPE,
                     "SI Fix SGPR copies", false, false)


char SIFixSGPRCopies::ID = 0;

char &llvm::SIFixSGPRCopiesID = SIFixSGPRCopies::ID;

FunctionPass *llvm::createSIFixSGPRCopiesPass() {
  return new SIFixSGPRCopies();
}

static bool hasVGPROperands(const MachineInstr &MI, const SIRegisterInfo *TRI) {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    if (!MI.getOperand(i).isReg() ||
        !TargetRegisterInfo::isVirtualRegister(MI.getOperand(i).getReg()))
      continue;

    if (TRI->hasVGPRs(MRI.getRegClass(MI.getOperand(i).getReg())))
      return true;
  }
  return false;
}

static std::pair<const TargetRegisterClass *, const TargetRegisterClass *>
getCopyRegClasses(const MachineInstr &Copy,
                  const SIRegisterInfo &TRI,
                  const MachineRegisterInfo &MRI) {
  unsigned DstReg = Copy.getOperand(0).getReg();
  unsigned SrcReg = Copy.getOperand(1).getReg();

  const TargetRegisterClass *SrcRC =
    TargetRegisterInfo::isVirtualRegister(SrcReg) ?
    MRI.getRegClass(SrcReg) :
    TRI.getPhysRegClass(SrcReg);

  // We don't really care about the subregister here.
  // SrcRC = TRI.getSubRegClass(SrcRC, Copy.getOperand(1).getSubReg());

  const TargetRegisterClass *DstRC =
    TargetRegisterInfo::isVirtualRegister(DstReg) ?
    MRI.getRegClass(DstReg) :
    TRI.getPhysRegClass(DstReg);

  return std::make_pair(SrcRC, DstRC);
}

static bool isVGPRToSGPRCopy(const TargetRegisterClass *SrcRC,
                             const TargetRegisterClass *DstRC,
                             const SIRegisterInfo &TRI) {
  return TRI.isSGPRClass(DstRC) && TRI.hasVGPRs(SrcRC);
}

static bool isSGPRToVGPRCopy(const TargetRegisterClass *SrcRC,
                             const TargetRegisterClass *DstRC,
                             const SIRegisterInfo &TRI) {
  return TRI.isSGPRClass(SrcRC) && TRI.hasVGPRs(DstRC);
}

// Distribute an SGPR->VGPR copy of a REG_SEQUENCE into a VGPR REG_SEQUENCE.
//
// SGPRx = ...
// SGPRy = REG_SEQUENCE SGPRx, sub0 ...
// VGPRz = COPY SGPRy
//
// ==>
//
// VGPRx = COPY SGPRx
// VGPRz = REG_SEQUENCE VGPRx, sub0
//
// This exposes immediate folding opportunities when materializing 64-bit
// immediates.
static bool foldVGPRCopyIntoRegSequence(MachineInstr &MI,
                                        const SIRegisterInfo *TRI,
                                        const SIInstrInfo *TII,
                                        MachineRegisterInfo &MRI) {
  assert(MI.isRegSequence());

  unsigned DstReg = MI.getOperand(0).getReg();
  if (!TRI->isSGPRClass(MRI.getRegClass(DstReg)))
    return false;

  if (!MRI.hasOneUse(DstReg))
    return false;

  MachineInstr &CopyUse = *MRI.use_instr_begin(DstReg);
  if (!CopyUse.isCopy())
    return false;

  // It is illegal to have vreg inputs to a physreg defining reg_sequence.
  if (TargetRegisterInfo::isPhysicalRegister(CopyUse.getOperand(0).getReg()))
    return false;

  const TargetRegisterClass *SrcRC, *DstRC;
  std::tie(SrcRC, DstRC) = getCopyRegClasses(CopyUse, *TRI, MRI);

  if (!isSGPRToVGPRCopy(SrcRC, DstRC, *TRI))
    return false;

  // TODO: Could have multiple extracts?
  unsigned SubReg = CopyUse.getOperand(1).getSubReg();
  if (SubReg != AMDGPU::NoSubRegister)
    return false;

  MRI.setRegClass(DstReg, DstRC);

  // SGPRx = ...
  // SGPRy = REG_SEQUENCE SGPRx, sub0 ...
  // VGPRz = COPY SGPRy

  // =>
  // VGPRx = COPY SGPRx
  // VGPRz = REG_SEQUENCE VGPRx, sub0

  MI.getOperand(0).setReg(CopyUse.getOperand(0).getReg());

  for (unsigned I = 1, N = MI.getNumOperands(); I != N; I += 2) {
    unsigned SrcReg = MI.getOperand(I).getReg();
    unsigned SrcSubReg = MI.getOperand(I).getSubReg();

    const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);
    assert(TRI->isSGPRClass(SrcRC) &&
           "Expected SGPR REG_SEQUENCE to only have SGPR inputs");

    SrcRC = TRI->getSubRegClass(SrcRC, SrcSubReg);
    const TargetRegisterClass *NewSrcRC = TRI->getEquivalentVGPRClass(SrcRC);

    unsigned TmpReg = MRI.createVirtualRegister(NewSrcRC);

    BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY),
            TmpReg)
        .add(MI.getOperand(I));

    MI.getOperand(I).setReg(TmpReg);
  }

  CopyUse.eraseFromParent();
  return true;
}

static bool phiHasVGPROperands(const MachineInstr &PHI,
                               const MachineRegisterInfo &MRI,
                               const SIRegisterInfo *TRI,
                               const SIInstrInfo *TII) {

  for (unsigned i = 1; i < PHI.getNumOperands(); i += 2) {
    unsigned Reg = PHI.getOperand(i).getReg();
    if (TRI->hasVGPRs(MRI.getRegClass(Reg)))
      return true;
  }
  return false;
}
static bool phiHasBreakDef(const MachineInstr &PHI,
                           const MachineRegisterInfo &MRI,
                           SmallSet<unsigned, 8> &Visited) {

  for (unsigned i = 1; i < PHI.getNumOperands(); i += 2) {
    unsigned Reg = PHI.getOperand(i).getReg();
    if (Visited.count(Reg))
      continue;

    Visited.insert(Reg);

    MachineInstr *DefInstr = MRI.getUniqueVRegDef(Reg);
    assert(DefInstr);
    switch (DefInstr->getOpcode()) {
    default:
      break;
    case AMDGPU::SI_BREAK:
    case AMDGPU::SI_IF_BREAK:
    case AMDGPU::SI_ELSE_BREAK:
      return true;
    case AMDGPU::PHI:
      if (phiHasBreakDef(*DefInstr, MRI, Visited))
        return true;
    }
  }
  return false;
}

static bool hasTerminatorThatModifiesExec(const MachineBasicBlock &MBB,
                                          const TargetRegisterInfo &TRI) {
  for (MachineBasicBlock::const_iterator I = MBB.getFirstTerminator(),
       E = MBB.end(); I != E; ++I) {
    if (I->modifiesRegister(AMDGPU::EXEC, &TRI))
      return true;
  }
  return false;
}

static bool isSafeToFoldImmIntoCopy(const MachineInstr *Copy,
                                    const MachineInstr *MoveImm,
                                    const SIInstrInfo *TII,
                                    unsigned &SMovOp,
                                    int64_t &Imm) {

  if (!MoveImm->isMoveImmediate())
    return false;

  const MachineOperand *ImmOp =
      TII->getNamedOperand(*MoveImm, AMDGPU::OpName::src0);
  if (!ImmOp->isImm())
    return false;

  // FIXME: Handle copies with sub-regs.
  if (Copy->getOperand(0).getSubReg())
    return false;

  switch (MoveImm->getOpcode()) {
  default:
    return false;
  case AMDGPU::V_MOV_B32_e32:
    SMovOp = AMDGPU::S_MOV_B32;
    break;
  case AMDGPU::V_MOV_B64_PSEUDO:
    SMovOp = AMDGPU::S_MOV_B64;
    break;
  }
  Imm = ImmOp->getImm();
  return true;
}

template <class UnaryPredicate>
bool searchPredecessors(const MachineBasicBlock *MBB,
                        const MachineBasicBlock *CutOff,
                        UnaryPredicate Predicate) {

  if (MBB == CutOff)
    return false;

  DenseSet<const MachineBasicBlock*> Visited;
  SmallVector<MachineBasicBlock*, 4> Worklist(MBB->pred_begin(), 
                                              MBB->pred_end());

  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();

    if (!Visited.insert(MBB).second)
      continue;
    if (MBB == CutOff)
      continue;
    if (Predicate(MBB))
      return true;

    Worklist.append(MBB->pred_begin(), MBB->pred_end());
  }

  return false;
}

static bool predsHasDivergentTerminator(MachineBasicBlock *MBB,
                                        const TargetRegisterInfo *TRI) {
  return searchPredecessors(MBB, nullptr, [TRI](MachineBasicBlock *MBB) {
           return hasTerminatorThatModifiesExec(*MBB, *TRI); });
}

// Checks if there is potential path From instruction To instruction.
// If CutOff is specified and it sits in between of that path we ignore
// a higher portion of the path and report it is not reachable.
static bool isReachable(const MachineInstr *From,
                        const MachineInstr *To,
                        const MachineBasicBlock *CutOff,
                        MachineDominatorTree &MDT) {
  // If either From block dominates To block or instructions are in the same
  // block and From is higher.
  if (MDT.dominates(From, To))
    return true;

  const MachineBasicBlock *MBBFrom = From->getParent();
  const MachineBasicBlock *MBBTo = To->getParent();
  if (MBBFrom == MBBTo)
    return false;

  // Instructions are in different blocks, do predecessor search.
  // We should almost never get here since we do not usually produce M0 stores
  // other than -1.
  return searchPredecessors(MBBTo, CutOff, [MBBFrom]
           (const MachineBasicBlock *MBB) { return MBB == MBBFrom; });
}

// Hoist and merge identical SGPR initializations into a common predecessor.
// This is intended to combine M0 initializations, but can work with any
// SGPR. A VGPR cannot be processed since we cannot guarantee vector
// executioon.
static bool hoistAndMergeSGPRInits(unsigned Reg,
                                   const MachineRegisterInfo &MRI,
                                   MachineDominatorTree &MDT) {
  // List of inits by immediate value.
  typedef std::map<unsigned, std::list<MachineInstr*>> InitListMap;
  InitListMap Inits;
  // List of clobbering instructions.
  SmallVector<MachineInstr*, 8> Clobbers;
  bool Changed = false;

  for (auto &MI : MRI.def_instructions(Reg)) {
    MachineOperand *Imm = nullptr;
    for (auto &MO: MI.operands()) {
      if ((MO.isReg() && ((MO.isDef() && MO.getReg() != Reg) || !MO.isDef())) ||
          (!MO.isImm() && !MO.isReg()) || (MO.isImm() && Imm)) {
        Imm = nullptr;
        break;
      } else if (MO.isImm())
        Imm = &MO;
    }
    if (Imm)
      Inits[Imm->getImm()].push_front(&MI);
    else
      Clobbers.push_back(&MI);
  }

  for (auto &Init : Inits) {
    auto &Defs = Init.second;

    for (auto I1 = Defs.begin(), E = Defs.end(); I1 != E; ) {
      MachineInstr *MI1 = *I1;

      for (auto I2 = std::next(I1); I2 != E; ) {
        MachineInstr *MI2 = *I2;

        // Check any possible interference
        auto intereferes = [&](MachineBasicBlock::iterator From,
                               MachineBasicBlock::iterator To) -> bool {

          assert(MDT.dominates(&*To, &*From));

          auto interferes = [&MDT, From, To](MachineInstr* &Clobber) -> bool {
            const MachineBasicBlock *MBBFrom = From->getParent();
            const MachineBasicBlock *MBBTo = To->getParent();
            bool MayClobberFrom = isReachable(Clobber, &*From, MBBTo, MDT);
            bool MayClobberTo = isReachable(Clobber, &*To, MBBTo, MDT);
            if (!MayClobberFrom && !MayClobberTo)
              return false;
            if ((MayClobberFrom && !MayClobberTo) ||
                (!MayClobberFrom && MayClobberTo))
              return true;
            // Both can clobber, this is not an interference only if both are
            // dominated by Clobber and belong to the same block or if Clobber
            // properly dominates To, given that To >> From, so it dominates
            // both and located in a common dominator.
            return !((MBBFrom == MBBTo &&
                      MDT.dominates(Clobber, &*From) &&
                      MDT.dominates(Clobber, &*To)) ||
                     MDT.properlyDominates(Clobber->getParent(), MBBTo));
          };

          return (any_of(Clobbers, interferes)) ||
                 (any_of(Inits, [&](InitListMap::value_type &C) {
                    return C.first != Init.first && any_of(C.second, interferes);
                  }));
        };

        if (MDT.dominates(MI1, MI2)) {
          if (!intereferes(MI2, MI1)) {
            DEBUG(dbgs() << "Erasing from BB#" << MI2->getParent()->getNumber()
                         << " " << *MI2);
            MI2->eraseFromParent();
            Defs.erase(I2++);
            Changed = true;
            continue;
          }
        } else if (MDT.dominates(MI2, MI1)) {
          if (!intereferes(MI1, MI2)) {
            DEBUG(dbgs() << "Erasing from BB#" << MI1->getParent()->getNumber()
                         << " " << *MI1);
            MI1->eraseFromParent();
            Defs.erase(I1++);
            Changed = true;
            break;
          }
        } else {
          auto *MBB = MDT.findNearestCommonDominator(MI1->getParent(),
                                                     MI2->getParent());
          if (!MBB) {
            ++I2;
            continue;
          }

          MachineBasicBlock::iterator I = MBB->getFirstNonPHI();
          if (!intereferes(MI1, I) && !intereferes(MI2, I)) {
            DEBUG(dbgs() << "Erasing from BB#" << MI1->getParent()->getNumber()
                         << " " << *MI1 << "and moving from BB#"
                         << MI2->getParent()->getNumber() << " to BB#"
                         << I->getParent()->getNumber() << " " << *MI2);
            I->getParent()->splice(I, MI2->getParent(), MI2);
            MI1->eraseFromParent();
            Defs.erase(I1++);
            Changed = true;
            break;
          }
        }
        ++I2;
      }
      ++I1;
    }
  }

  if (Changed)
    MRI.clearKillFlags(Reg);

  return Changed;
}

bool SIFixSGPRCopies::runOnMachineFunction(MachineFunction &MF) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  MDT = &getAnalysis<MachineDominatorTree>();

  SmallVector<MachineInstr *, 16> Worklist;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      default:
        continue;
      case AMDGPU::COPY: {
        // If the destination register is a physical register there isn't really
        // much we can do to fix this.
        if (!TargetRegisterInfo::isVirtualRegister(MI.getOperand(0).getReg()))
          continue;

        const TargetRegisterClass *SrcRC, *DstRC;
        std::tie(SrcRC, DstRC) = getCopyRegClasses(MI, *TRI, MRI);
        if (isVGPRToSGPRCopy(SrcRC, DstRC, *TRI)) {
          MachineInstr *DefMI = MRI.getVRegDef(MI.getOperand(1).getReg());
          unsigned SMovOp;
          int64_t Imm;
          // If we are just copying an immediate, we can replace the copy with
          // s_mov_b32.
          if (isSafeToFoldImmIntoCopy(&MI, DefMI, TII, SMovOp, Imm)) {
            MI.getOperand(1).ChangeToImmediate(Imm);
            MI.addImplicitDefUseOperands(MF);
            MI.setDesc(TII->get(SMovOp));
            break;
          }
          TII->moveToVALU(MI);
        }

        break;
      }
      case AMDGPU::PHI: {
        unsigned Reg = MI.getOperand(0).getReg();
        if (!TRI->isSGPRClass(MRI.getRegClass(Reg)))
          break;

        // We don't need to fix the PHI if the common dominator of the
        // two incoming blocks terminates with a uniform branch.
        if (MI.getNumExplicitOperands() == 5) {
          MachineBasicBlock *MBB0 = MI.getOperand(2).getMBB();
          MachineBasicBlock *MBB1 = MI.getOperand(4).getMBB();

          if (!predsHasDivergentTerminator(MBB0, TRI) &&
              !predsHasDivergentTerminator(MBB1, TRI)) {
            DEBUG(dbgs() << "Not fixing PHI for uniform branch: " << MI << '\n');
            break;
          }
        }

        // If a PHI node defines an SGPR and any of its operands are VGPRs,
        // then we need to move it to the VALU.
        //
        // Also, if a PHI node defines an SGPR and has all SGPR operands
        // we must move it to the VALU, because the SGPR operands will
        // all end up being assigned the same register, which means
        // there is a potential for a conflict if different threads take
        // different control flow paths.
        //
        // For Example:
        //
        // sgpr0 = def;
        // ...
        // sgpr1 = def;
        // ...
        // sgpr2 = PHI sgpr0, sgpr1
        // use sgpr2;
        //
        // Will Become:
        //
        // sgpr2 = def;
        // ...
        // sgpr2 = def;
        // ...
        // use sgpr2
        //
        // The one exception to this rule is when one of the operands
        // is defined by a SI_BREAK, SI_IF_BREAK, or SI_ELSE_BREAK
        // instruction.  In this case, there we know the program will
        // never enter the second block (the loop) without entering
        // the first block (where the condition is computed), so there
        // is no chance for values to be over-written.

        SmallSet<unsigned, 8> Visited;
        if (phiHasVGPROperands(MI, MRI, TRI, TII) ||
            !phiHasBreakDef(MI, MRI, Visited)) {
          DEBUG(dbgs() << "Fixing PHI: " << MI);
          TII->moveToVALU(MI);
        }
        break;
      }
      case AMDGPU::REG_SEQUENCE: {
        if (TRI->hasVGPRs(TII->getOpRegClass(MI, 0)) ||
            !hasVGPROperands(MI, TRI)) {
          foldVGPRCopyIntoRegSequence(MI, TRI, TII, MRI);
          continue;
        }

        DEBUG(dbgs() << "Fixing REG_SEQUENCE: " << MI);

        TII->moveToVALU(MI);
        break;
      }
      case AMDGPU::INSERT_SUBREG: {
        const TargetRegisterClass *DstRC, *Src0RC, *Src1RC;
        DstRC = MRI.getRegClass(MI.getOperand(0).getReg());
        Src0RC = MRI.getRegClass(MI.getOperand(1).getReg());
        Src1RC = MRI.getRegClass(MI.getOperand(2).getReg());
        if (TRI->isSGPRClass(DstRC) &&
            (TRI->hasVGPRs(Src0RC) || TRI->hasVGPRs(Src1RC))) {
          DEBUG(dbgs() << " Fixing INSERT_SUBREG: " << MI);
          TII->moveToVALU(MI);
        }
        break;
      }
      }
    }
  }

  if (MF.getTarget().getOptLevel() > CodeGenOpt::None && EnableM0Merge)
    hoistAndMergeSGPRInits(AMDGPU::M0, MRI, *MDT);

  return true;
}
