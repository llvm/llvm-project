//===-- PeepholeOptimizer.cpp - Peephole Optimizations --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Perform peephole optimizations on the machine code:
//
// - Optimize Extensions
//
//     Optimization of sign / zero extension instructions. It may be extended to
//     handle other instructions with similar properties.
//
//     On some targets, some instructions, e.g. X86 sign / zero extension, may
//     leave the source value in the lower part of the result. This optimization
//     will replace some uses of the pre-extension value with uses of the
//     sub-register of the results.
//
// - Optimize Comparisons
//
//     Optimization of comparison instructions. For instance, in this code:
//
//       sub r1, 1
//       cmp r1, 0
//       bz  L1
//
//     If the "sub" instruction all ready sets (or could be modified to set) the
//     same flag that the "cmp" instruction sets and that "bz" uses, then we can
//     eliminate the "cmp" instruction.
//
//     Another instance, in this code:
//
//       sub r1, r3 | sub r1, imm
//       cmp r3, r1 or cmp r1, r3 | cmp r1, imm
//       bge L1
//
//     If the branch instruction can use flag from "sub", then we can replace
//     "sub" with "subs" and eliminate the "cmp" instruction.
//
// - Optimize Bitcast pairs:
//
//     v1 = bitcast v0
//     v2 = bitcast v1
//        = v2
//   =>
//     v1 = bitcast v0
//        = v0
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "peephole-opt"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

// Optimize Extensions
static cl::opt<bool>
Aggressive("aggressive-ext-opt", cl::Hidden,
           cl::desc("Aggressive extension optimization"));

static cl::opt<bool>
DisablePeephole("disable-peephole", cl::Hidden, cl::init(false),
                cl::desc("Disable the peephole optimizer"));

STATISTIC(NumReuse,      "Number of extension results reused");
STATISTIC(NumBitcasts,   "Number of bitcasts eliminated");
STATISTIC(NumCmps,       "Number of compares eliminated");
STATISTIC(NumImmFold,    "Number of move immediate folded");
STATISTIC(NumLoadFold,   "Number of loads folded");
STATISTIC(NumSelects,    "Number of selects optimized");

namespace {
  class PeepholeOptimizer : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineRegisterInfo   *MRI;
    MachineDominatorTree  *DT;  // Machine dominator tree

  public:
    static char ID; // Pass identification
    PeepholeOptimizer() : MachineFunctionPass(ID) {
      initializePeepholeOptimizerPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool optimizeBitcastInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool optimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                          SmallPtrSet<MachineInstr*, 8> &LocalMIs);
    bool optimizeSelect(MachineInstr *MI);
    bool isMoveImmediate(MachineInstr *MI,
                         SmallSet<unsigned, 4> &ImmDefRegs,
                         DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                       SmallSet<unsigned, 4> &ImmDefRegs,
                       DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool isLoadFoldable(MachineInstr *MI, unsigned &FoldAsLoadDefReg);
  };
}

char PeepholeOptimizer::ID = 0;
char &llvm::PeepholeOptimizerID = PeepholeOptimizer::ID;
INITIALIZE_PASS_BEGIN(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)

/// optimizeExtInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify the
/// source, and if the source value is preserved as a sub-register of the
/// result, then replace all reachable uses of the source with the subreg of the
/// result.
///
/// Do not generate an EXTRACT that is used only in a debug use, as this changes
/// the code. Since this code does not currently share EXTRACTs, just ignore all
/// debug uses.
bool PeepholeOptimizer::
optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                 SmallPtrSet<MachineInstr*, 8> &LocalMIs) {
  unsigned SrcReg, DstReg, SubIdx;
  if (!TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx))
    return false;

  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  if (MRI->hasOneNonDBGUse(SrcReg))
    // No other uses.
    return false;

  // Ensure DstReg can get a register class that actually supports
  // sub-registers. Don't change the class until we commit.
  const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
  DstRC = TM->getRegisterInfo()->getSubClassWithSubReg(DstRC, SubIdx);
  if (!DstRC)
    return false;

  // The ext instr may be operating on a sub-register of SrcReg as well.
  // PPC::EXTSW is a 32 -> 64-bit sign extension, but it reads a 64-bit
  // register.
  // If UseSrcSubIdx is Set, SubIdx also applies to SrcReg, and only uses of
  // SrcReg:SubIdx should be replaced.
  bool UseSrcSubIdx = TM->getRegisterInfo()->
    getSubClassWithSubReg(MRI->getRegClass(SrcReg), SubIdx) != 0;

  // The source has other uses. See if we can replace the other uses with use of
  // the result of the extension.
  SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
  for (MachineRegisterInfo::use_nodbg_iterator
       UI = MRI->use_nodbg_begin(DstReg), UE = MRI->use_nodbg_end();
       UI != UE; ++UI)
    ReachedBBs.insert(UI->getParent());

  // Uses that are in the same BB of uses of the result of the instruction.
  SmallVector<MachineOperand*, 8> Uses;

  // Uses that the result of the instruction can reach.
  SmallVector<MachineOperand*, 8> ExtendedUses;

  bool ExtendLife = true;
  for (MachineRegisterInfo::use_nodbg_iterator
       UI = MRI->use_nodbg_begin(SrcReg), UE = MRI->use_nodbg_end();
       UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    if (UseMI == MI)
      continue;

    if (UseMI->isPHI()) {
      ExtendLife = false;
      continue;
    }

    // Only accept uses of SrcReg:SubIdx.
    if (UseSrcSubIdx && UseMO.getSubReg() != SubIdx)
      continue;

    // It's an error to translate this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1026 = SUBREG_TO_REG 0, %reg1024, 4
    //
    // into this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1027 = COPY %reg1025:4
    //    %reg1026 = SUBREG_TO_REG 0, %reg1027, 4
    //
    // The problem here is that SUBREG_TO_REG is there to assert that an
    // implicit zext occurs. It doesn't insert a zext instruction. If we allow
    // the COPY here, it will give us the value after the <sext>, not the
    // original value of %reg1024 before <sext>.
    if (UseMI->getOpcode() == TargetOpcode::SUBREG_TO_REG)
      continue;

    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      // Local uses that come after the extension.
      if (!LocalMIs.count(UseMI))
        Uses.push_back(&UseMO);
    } else if (ReachedBBs.count(UseMBB)) {
      // Non-local uses where the result of the extension is used. Always
      // replace these unless it's a PHI.
      Uses.push_back(&UseMO);
    } else if (Aggressive && DT->dominates(MBB, UseMBB)) {
      // We may want to extend the live range of the extension result in order
      // to replace these uses.
      ExtendedUses.push_back(&UseMO);
    } else {
      // Both will be live out of the def MBB anyway. Don't extend live range of
      // the extension result.
      ExtendLife = false;
      break;
    }
  }

  if (ExtendLife && !ExtendedUses.empty())
    // Extend the liveness of the extension result.
    std::copy(ExtendedUses.begin(), ExtendedUses.end(),
              std::back_inserter(Uses));

  // Now replace all uses.
  bool Changed = false;
  if (!Uses.empty()) {
    SmallPtrSet<MachineBasicBlock*, 4> PHIBBs;

    // Look for PHI uses of the extended result, we don't want to extend the
    // liveness of a PHI input. It breaks all kinds of assumptions down
    // stream. A PHI use is expected to be the kill of its source values.
    for (MachineRegisterInfo::use_nodbg_iterator
         UI = MRI->use_nodbg_begin(DstReg), UE = MRI->use_nodbg_end();
         UI != UE; ++UI)
      if (UI->isPHI())
        PHIBBs.insert(UI->getParent());

    const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
    for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
      MachineOperand *UseMO = Uses[i];
      MachineInstr *UseMI = UseMO->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (PHIBBs.count(UseMBB))
        continue;

      // About to add uses of DstReg, clear DstReg's kill flags.
      if (!Changed) {
        MRI->clearKillFlags(DstReg);
        MRI->constrainRegClass(DstReg, DstRC);
      }

      unsigned NewVR = MRI->createVirtualRegister(RC);
      MachineInstr *Copy = BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY), NewVR)
        .addReg(DstReg, 0, SubIdx);
      // SubIdx applies to both SrcReg and DstReg when UseSrcSubIdx is set.
      if (UseSrcSubIdx) {
        Copy->getOperand(0).setSubReg(SubIdx);
        Copy->getOperand(0).setIsUndef();
      }
      UseMO->setReg(NewVR);
      ++NumReuse;
      Changed = true;
    }
  }

  return Changed;
}

/// optimizeBitcastInstr - If the instruction is a bitcast instruction A that
/// cannot be optimized away during isel (e.g. ARM::VMOVSR, which bitcast
/// a value cross register classes), and the source is defined by another
/// bitcast instruction B. And if the register class of source of B matches
/// the register class of instruction A, then it is legal to replace all uses
/// of the def of A with source of B. e.g.
///   %vreg0<def> = VMOVSR %vreg1
///   %vreg3<def> = VMOVRS %vreg0
///   Replace all uses of vreg3 with vreg1.

bool PeepholeOptimizer::optimizeBitcastInstr(MachineInstr *MI,
                                             MachineBasicBlock *MBB) {
  unsigned NumDefs = MI->getDesc().getNumDefs();
  unsigned NumSrcs = MI->getDesc().getNumOperands() - NumDefs;
  if (NumDefs != 1)
    return false;

  unsigned Def = 0;
  unsigned Src = 0;
  for (unsigned i = 0, e = NumDefs + NumSrcs; i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef())
      Def = Reg;
    else if (Src)
      // Multiple sources?
      return false;
    else
      Src = Reg;
  }

  assert(Def && Src && "Malformed bitcast instruction!");

  MachineInstr *DefMI = MRI->getVRegDef(Src);
  if (!DefMI || !DefMI->isBitcast())
    return false;

  unsigned SrcSrc = 0;
  NumDefs = DefMI->getDesc().getNumDefs();
  NumSrcs = DefMI->getDesc().getNumOperands() - NumDefs;
  if (NumDefs != 1)
    return false;
  for (unsigned i = 0, e = NumDefs + NumSrcs; i != e; ++i) {
    const MachineOperand &MO = DefMI->getOperand(i);
    if (!MO.isReg() || MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (!MO.isDef()) {
      if (SrcSrc)
        // Multiple sources?
        return false;
      else
        SrcSrc = Reg;
    }
  }

  if (MRI->getRegClass(SrcSrc) != MRI->getRegClass(Def))
    return false;

  MRI->replaceRegWith(Def, SrcSrc);
  MRI->clearKillFlags(SrcSrc);
  MI->eraseFromParent();
  ++NumBitcasts;
  return true;
}

/// optimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool PeepholeOptimizer::optimizeCmpInstr(MachineInstr *MI,
                                         MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg, SrcReg2;
  int CmpMask, CmpValue;
  if (!TII->analyzeCompare(MI, SrcReg, SrcReg2, CmpMask, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg) ||
      (SrcReg2 != 0 && TargetRegisterInfo::isPhysicalRegister(SrcReg2)))
    return false;

  // Attempt to optimize the comparison instruction.
  if (TII->optimizeCompareInstr(MI, SrcReg, SrcReg2, CmpMask, CmpValue, MRI)) {
    ++NumCmps;
    return true;
  }

  return false;
}

/// Optimize a select instruction.
bool PeepholeOptimizer::optimizeSelect(MachineInstr *MI) {
  unsigned TrueOp = 0;
  unsigned FalseOp = 0;
  bool Optimizable = false;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeSelect(MI, Cond, TrueOp, FalseOp, Optimizable))
    return false;
  if (!Optimizable)
    return false;
  if (!TII->optimizeSelect(MI))
    return false;
  MI->eraseFromParent();
  ++NumSelects;
  return true;
}

/// isLoadFoldable - Check whether MI is a candidate for folding into a later
/// instruction. We only fold loads to virtual registers and the virtual
/// register defined has a single use.
bool PeepholeOptimizer::isLoadFoldable(MachineInstr *MI,
                                       unsigned &FoldAsLoadDefReg) {
  if (!MI->canFoldAsLoad() || !MI->mayLoad())
    return false;
  const MCInstrDesc &MCID = MI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;

  unsigned Reg = MI->getOperand(0).getReg();
  // To reduce compilation time, we check MRI->hasOneUse when inserting
  // loads. It should be checked when processing uses of the load, since
  // uses can be removed during peephole.
  if (!MI->getOperand(0).getSubReg() &&
      TargetRegisterInfo::isVirtualRegister(Reg) &&
      MRI->hasOneUse(Reg)) {
    FoldAsLoadDefReg = Reg;
    return true;
  }
  return false;
}

bool PeepholeOptimizer::isMoveImmediate(MachineInstr *MI,
                                        SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MI->isMoveImmediate())
    return false;
  if (MCID.getNumDefs() != 1)
    return false;
  unsigned Reg = MI->getOperand(0).getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    ImmDefMIs.insert(std::make_pair(Reg, MI));
    ImmDefRegs.insert(Reg);
    return true;
  }

  return false;
}

/// foldImmediate - Try folding register operands that are defined by move
/// immediate instructions, i.e. a trivial constant folding optimization, if
/// and only if the def and use are in the same BB.
bool PeepholeOptimizer::foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                                      SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (ImmDefRegs.count(Reg) == 0)
      continue;
    DenseMap<unsigned, MachineInstr*>::iterator II = ImmDefMIs.find(Reg);
    assert(II != ImmDefMIs.end());
    if (TII->FoldImmediate(MI, II->second, Reg, MRI)) {
      ++NumImmFold;
      return true;
    }
  }
  return false;
}

bool PeepholeOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (DisablePeephole)
    return false;

  TM  = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT  = Aggressive ? &getAnalysis<MachineDominatorTree>() : 0;

  bool Changed = false;

  SmallPtrSet<MachineInstr*, 8> LocalMIs;
  SmallSet<unsigned, 4> ImmDefRegs;
  DenseMap<unsigned, MachineInstr*> ImmDefMIs;
  unsigned FoldAsLoadDefReg;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    bool SeenMoveImm = false;
    LocalMIs.clear();
    ImmDefRegs.clear();
    ImmDefMIs.clear();
    FoldAsLoadDefReg = 0;

    for (MachineBasicBlock::iterator
           MII = I->begin(), MIE = I->end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;
      // We may be erasing MI below, increment MII now.
      ++MII;
      LocalMIs.insert(MI);

      // If there exists an instruction which belongs to the following
      // categories, we will discard the load candidate.
      if (MI->isLabel() || MI->isPHI() || MI->isImplicitDef() ||
          MI->isKill() || MI->isInlineAsm() || MI->isDebugValue() ||
          MI->hasUnmodeledSideEffects()) {
        FoldAsLoadDefReg = 0;
        continue;
      }
      if (MI->mayStore() || MI->isCall())
        FoldAsLoadDefReg = 0;

      if ((MI->isBitcast() && optimizeBitcastInstr(MI, MBB)) ||
          (MI->isCompare() && optimizeCmpInstr(MI, MBB)) ||
          (MI->isSelect() && optimizeSelect(MI))) {
        // MI is deleted.
        LocalMIs.erase(MI);
        Changed = true;
        continue;
      }

      if (isMoveImmediate(MI, ImmDefRegs, ImmDefMIs)) {
        SeenMoveImm = true;
      } else {
        Changed |= optimizeExtInstr(MI, MBB, LocalMIs);
        // optimizeExtInstr might have created new instructions after MI
        // and before the already incremented MII. Adjust MII so that the
        // next iteration sees the new instructions.
        MII = MI;
        ++MII;
        if (SeenMoveImm)
          Changed |= foldImmediate(MI, MBB, ImmDefRegs, ImmDefMIs);
      }

      // Check whether MI is a load candidate for folding into a later
      // instruction. If MI is not a candidate, check whether we can fold an
      // earlier load into MI.
      if (!isLoadFoldable(MI, FoldAsLoadDefReg) && FoldAsLoadDefReg) {
        // We need to fold load after optimizeCmpInstr, since optimizeCmpInstr
        // can enable folding by converting SUB to CMP.
        MachineInstr *DefMI = 0;
        MachineInstr *FoldMI = TII->optimizeLoadInstr(MI, MRI,
                                                      FoldAsLoadDefReg, DefMI);
        if (FoldMI) {
          // Update LocalMIs since we replaced MI with FoldMI and deleted DefMI.
          LocalMIs.erase(MI);
          LocalMIs.erase(DefMI);
          LocalMIs.insert(FoldMI);
          MI->eraseFromParent();
          DefMI->eraseFromParent();
          ++NumLoadFold;

          // MI is replaced with FoldMI.
          Changed = true;
          continue;
        }
      }
    }
  }

  return Changed;
}
