//===-------------- PPCMIPeephole.cpp - MI Peephole Cleanups -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass performs peephole optimizations to clean up ugly code
// sequences at the MachineInstruction layer.  It runs at the end of
// the SSA phases, following VSX swap removal.  A pass of dead code
// elimination follows this one for quick clean-up of any dead
// instructions introduced here.  Although we could do this as callbacks
// from the generic peephole pass, this would have a couple of bad
// effects:  it might remove optimization opportunities for VSX swap
// removal, and it would miss cleanups made possible following VSX
// swap removal.
//
//===---------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCInstrInfo.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-mi-peepholes"

STATISTIC(RemoveTOCSave, "Number of TOC saves removed");
STATISTIC(MultiTOCSaves,
          "Number of functions with multiple TOC saves that must be kept");
STATISTIC(NumTOCSavesInPrologue, "Number of TOC saves placed in the prologue");
STATISTIC(NumEliminatedSExt, "Number of eliminated sign-extensions");
STATISTIC(NumEliminatedZExt, "Number of eliminated zero-extensions");
STATISTIC(NumOptADDLIs, "Number of optimized ADD instruction fed by LI");
STATISTIC(NumConvertedToImmediateForm,
          "Number of instructions converted to their immediate form");
STATISTIC(NumFunctionsEnteredInMIPeephole,
          "Number of functions entered in PPC MI Peepholes");
STATISTIC(NumFixedPointIterations,
          "Number of fixed-point iterations converting reg-reg instructions "
          "to reg-imm ones");
STATISTIC(NumRotatesCollapsed,
          "Number of pairs of rotate left, clear left/right collapsed");
STATISTIC(NumEXTSWAndSLDICombined,
          "Number of pairs of EXTSW and SLDI combined as EXTSWSLI");
STATISTIC(NumLoadImmZeroFoldedAndRemoved,
          "Number of LI(8) reg, 0 that are folded to r0 and removed");

static cl::opt<bool>
FixedPointRegToImm("ppc-reg-to-imm-fixed-point", cl::Hidden, cl::init(true),
                   cl::desc("Iterate to a fixed point when attempting to "
                            "convert reg-reg instructions to reg-imm"));

static cl::opt<bool>
ConvertRegReg("ppc-convert-rr-to-ri", cl::Hidden, cl::init(true),
              cl::desc("Convert eligible reg+reg instructions to reg+imm"));

static cl::opt<bool>
    EnableSExtElimination("ppc-eliminate-signext",
                          cl::desc("enable elimination of sign-extensions"),
                          cl::init(false), cl::Hidden);

static cl::opt<bool>
    EnableZExtElimination("ppc-eliminate-zeroext",
                          cl::desc("enable elimination of zero-extensions"),
                          cl::init(false), cl::Hidden);

static cl::opt<bool>
    EnableTrapOptimization("ppc-opt-conditional-trap",
                           cl::desc("enable optimization of conditional traps"),
                           cl::init(false), cl::Hidden);

namespace {

struct PPCMIPeephole : public MachineFunctionPass {

  static char ID;
  const PPCInstrInfo *TII;
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  PPCMIPeephole() : MachineFunctionPass(ID) {
    initializePPCMIPeepholePass(*PassRegistry::getPassRegistry());
  }

private:
  MachineDominatorTree *MDT;
  MachinePostDominatorTree *MPDT;
  MachineBlockFrequencyInfo *MBFI;
  uint64_t EntryFreq;

  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  // Perform peepholes.
  bool simplifyCode();

  // Perform peepholes.
  bool eliminateRedundantCompare();
  bool eliminateRedundantTOCSaves(std::map<MachineInstr *, bool> &TOCSaves);
  bool combineSEXTAndSHL(MachineInstr &MI, MachineInstr *&ToErase);
  bool emitRLDICWhenLoweringJumpTables(MachineInstr &MI);
  void UpdateTOCSaves(std::map<MachineInstr *, bool> &TOCSaves,
                      MachineInstr *MI);

public:

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addPreserved<MachinePostDominatorTree>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    initialize(MF);
    // At this point, TOC pointer should not be used in a function that uses
    // PC-Relative addressing.
    assert((MF.getRegInfo().use_empty(PPC::X2) ||
            !MF.getSubtarget<PPCSubtarget>().isUsingPCRelativeCalls()) &&
           "TOC pointer used in a function using PC-Relative addressing!");
    if (skipFunction(MF.getFunction()))
      return false;
    return simplifyCode();
  }
};

// Initialize class variables.
void PPCMIPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  MDT = &getAnalysis<MachineDominatorTree>();
  MPDT = &getAnalysis<MachinePostDominatorTree>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  EntryFreq = MBFI->getEntryFreq();
  TII = MF->getSubtarget<PPCSubtarget>().getInstrInfo();
  LLVM_DEBUG(dbgs() << "*** PowerPC MI peephole pass ***\n\n");
  LLVM_DEBUG(MF->dump());
}

static MachineInstr *getVRegDefOrNull(MachineOperand *Op,
                                      MachineRegisterInfo *MRI) {
  assert(Op && "Invalid Operand!");
  if (!Op->isReg())
    return nullptr;

  Register Reg = Op->getReg();
  if (!Register::isVirtualRegister(Reg))
    return nullptr;

  return MRI->getVRegDef(Reg);
}

// This function returns number of known zero bits in output of MI
// starting from the most significant bit.
static unsigned
getKnownLeadingZeroCount(MachineInstr *MI, const PPCInstrInfo *TII) {
  unsigned Opcode = MI->getOpcode();
  if (Opcode == PPC::RLDICL || Opcode == PPC::RLDICL_rec ||
      Opcode == PPC::RLDCL || Opcode == PPC::RLDCL_rec)
    return MI->getOperand(3).getImm();

  if ((Opcode == PPC::RLDIC || Opcode == PPC::RLDIC_rec) &&
      MI->getOperand(3).getImm() <= 63 - MI->getOperand(2).getImm())
    return MI->getOperand(3).getImm();

  if ((Opcode == PPC::RLWINM || Opcode == PPC::RLWINM_rec ||
       Opcode == PPC::RLWNM || Opcode == PPC::RLWNM_rec ||
       Opcode == PPC::RLWINM8 || Opcode == PPC::RLWNM8) &&
      MI->getOperand(3).getImm() <= MI->getOperand(4).getImm())
    return 32 + MI->getOperand(3).getImm();

  if (Opcode == PPC::ANDI_rec) {
    uint16_t Imm = MI->getOperand(2).getImm();
    return 48 + countLeadingZeros(Imm);
  }

  if (Opcode == PPC::CNTLZW || Opcode == PPC::CNTLZW_rec ||
      Opcode == PPC::CNTTZW || Opcode == PPC::CNTTZW_rec ||
      Opcode == PPC::CNTLZW8 || Opcode == PPC::CNTTZW8)
    // The result ranges from 0 to 32.
    return 58;

  if (Opcode == PPC::CNTLZD || Opcode == PPC::CNTLZD_rec ||
      Opcode == PPC::CNTTZD || Opcode == PPC::CNTTZD_rec)
    // The result ranges from 0 to 64.
    return 57;

  if (Opcode == PPC::LHZ   || Opcode == PPC::LHZX  ||
      Opcode == PPC::LHZ8  || Opcode == PPC::LHZX8 ||
      Opcode == PPC::LHZU  || Opcode == PPC::LHZUX ||
      Opcode == PPC::LHZU8 || Opcode == PPC::LHZUX8)
    return 48;

  if (Opcode == PPC::LBZ   || Opcode == PPC::LBZX  ||
      Opcode == PPC::LBZ8  || Opcode == PPC::LBZX8 ||
      Opcode == PPC::LBZU  || Opcode == PPC::LBZUX ||
      Opcode == PPC::LBZU8 || Opcode == PPC::LBZUX8)
    return 56;

  if (TII->isZeroExtended(*MI))
    return 32;

  return 0;
}

// This function maintains a map for the pairs <TOC Save Instr, Keep>
// Each time a new TOC save is encountered, it checks if any of the existing
// ones are dominated by the new one. If so, it marks the existing one as
// redundant by setting it's entry in the map as false. It then adds the new
// instruction to the map with either true or false depending on if any
// existing instructions dominated the new one.
void PPCMIPeephole::UpdateTOCSaves(
  std::map<MachineInstr *, bool> &TOCSaves, MachineInstr *MI) {
  assert(TII->isTOCSaveMI(*MI) && "Expecting a TOC save instruction here");
  // FIXME: Saving TOC in prologue hasn't been implemented well in AIX ABI part,
  // here only support it under ELFv2.
  if (MF->getSubtarget<PPCSubtarget>().isELFv2ABI()) {
    PPCFunctionInfo *FI = MF->getInfo<PPCFunctionInfo>();

    MachineBasicBlock *Entry = &MF->front();
    uint64_t CurrBlockFreq = MBFI->getBlockFreq(MI->getParent()).getFrequency();

    // If the block in which the TOC save resides is in a block that
    // post-dominates Entry, or a block that is hotter than entry (keep in mind
    // that early MachineLICM has already run so the TOC save won't be hoisted)
    // we can just do the save in the prologue.
    if (CurrBlockFreq > EntryFreq || MPDT->dominates(MI->getParent(), Entry))
      FI->setMustSaveTOC(true);

    // If we are saving the TOC in the prologue, all the TOC saves can be
    // removed from the code.
    if (FI->mustSaveTOC()) {
      for (auto &TOCSave : TOCSaves)
        TOCSave.second = false;
      // Add new instruction to map.
      TOCSaves[MI] = false;
      return;
    }
  }

  bool Keep = true;
  for (auto &I : TOCSaves) {
    MachineInstr *CurrInst = I.first;
    // If new instruction dominates an existing one, mark existing one as
    // redundant.
    if (I.second && MDT->dominates(MI, CurrInst))
      I.second = false;
    // Check if the new instruction is redundant.
    if (MDT->dominates(CurrInst, MI)) {
      Keep = false;
      break;
    }
  }
  // Add new instruction to map.
  TOCSaves[MI] = Keep;
}

// This function returns a list of all PHI nodes in the tree starting from
// the RootPHI node. We perform a BFS traversal to get an ordered list of nodes.
// The list initially only contains the root PHI. When we visit a PHI node, we
// add it to the list. We continue to look for other PHI node operands while
// there are nodes to visit in the list. The function returns false if the
// optimization cannot be applied on this tree.
static bool collectUnprimedAccPHIs(MachineRegisterInfo *MRI,
                                   MachineInstr *RootPHI,
                                   SmallVectorImpl<MachineInstr *> &PHIs) {
  PHIs.push_back(RootPHI);
  unsigned VisitedIndex = 0;
  while (VisitedIndex < PHIs.size()) {
    MachineInstr *VisitedPHI = PHIs[VisitedIndex];
    for (unsigned PHIOp = 1, NumOps = VisitedPHI->getNumOperands();
         PHIOp != NumOps; PHIOp += 2) {
      Register RegOp = VisitedPHI->getOperand(PHIOp).getReg();
      if (!Register::isVirtualRegister(RegOp))
        return false;
      MachineInstr *Instr = MRI->getVRegDef(RegOp);
      // While collecting the PHI nodes, we check if they can be converted (i.e.
      // all the operands are either copies, implicit defs or PHI nodes).
      unsigned Opcode = Instr->getOpcode();
      if (Opcode == PPC::COPY) {
        Register Reg = Instr->getOperand(1).getReg();
        if (!Register::isVirtualRegister(Reg) ||
            MRI->getRegClass(Reg) != &PPC::ACCRCRegClass)
          return false;
      } else if (Opcode != PPC::IMPLICIT_DEF && Opcode != PPC::PHI)
        return false;
      // If we detect a cycle in the PHI nodes, we exit. It would be
      // possible to change cycles as well, but that would add a lot
      // of complexity for a case that is unlikely to occur with MMA
      // code.
      if (Opcode != PPC::PHI)
        continue;
      if (llvm::is_contained(PHIs, Instr))
        return false;
      PHIs.push_back(Instr);
    }
    VisitedIndex++;
  }
  return true;
}

// This function changes the unprimed accumulator PHI nodes in the PHIs list to
// primed accumulator PHI nodes. The list is traversed in reverse order to
// change all the PHI operands of a PHI node before changing the node itself.
// We keep a map to associate each changed PHI node to its non-changed form.
static void convertUnprimedAccPHIs(const PPCInstrInfo *TII,
                                   MachineRegisterInfo *MRI,
                                   SmallVectorImpl<MachineInstr *> &PHIs,
                                   Register Dst) {
  DenseMap<MachineInstr *, MachineInstr *> ChangedPHIMap;
  for (MachineInstr *PHI : llvm::reverse(PHIs)) {
    SmallVector<std::pair<MachineOperand, MachineOperand>, 4> PHIOps;
    // We check if the current PHI node can be changed by looking at its
    // operands. If all the operands are either copies from primed
    // accumulators, implicit definitions or other unprimed accumulator
    // PHI nodes, we change it.
    for (unsigned PHIOp = 1, NumOps = PHI->getNumOperands(); PHIOp != NumOps;
         PHIOp += 2) {
      Register RegOp = PHI->getOperand(PHIOp).getReg();
      MachineInstr *PHIInput = MRI->getVRegDef(RegOp);
      unsigned Opcode = PHIInput->getOpcode();
      assert((Opcode == PPC::COPY || Opcode == PPC::IMPLICIT_DEF ||
              Opcode == PPC::PHI) &&
             "Unexpected instruction");
      if (Opcode == PPC::COPY) {
        assert(MRI->getRegClass(PHIInput->getOperand(1).getReg()) ==
                   &PPC::ACCRCRegClass &&
               "Unexpected register class");
        PHIOps.push_back({PHIInput->getOperand(1), PHI->getOperand(PHIOp + 1)});
      } else if (Opcode == PPC::IMPLICIT_DEF) {
        Register AccReg = MRI->createVirtualRegister(&PPC::ACCRCRegClass);
        BuildMI(*PHIInput->getParent(), PHIInput, PHIInput->getDebugLoc(),
                TII->get(PPC::IMPLICIT_DEF), AccReg);
        PHIOps.push_back({MachineOperand::CreateReg(AccReg, false),
                          PHI->getOperand(PHIOp + 1)});
      } else if (Opcode == PPC::PHI) {
        // We found a PHI operand. At this point we know this operand
        // has already been changed so we get its associated changed form
        // from the map.
        assert(ChangedPHIMap.count(PHIInput) == 1 &&
               "This PHI node should have already been changed.");
        MachineInstr *PrimedAccPHI = ChangedPHIMap.lookup(PHIInput);
        PHIOps.push_back({MachineOperand::CreateReg(
                              PrimedAccPHI->getOperand(0).getReg(), false),
                          PHI->getOperand(PHIOp + 1)});
      }
    }
    Register AccReg = Dst;
    // If the PHI node we are changing is the root node, the register it defines
    // will be the destination register of the original copy (of the PHI def).
    // For all other PHI's in the list, we need to create another primed
    // accumulator virtual register as the PHI will no longer define the
    // unprimed accumulator.
    if (PHI != PHIs[0])
      AccReg = MRI->createVirtualRegister(&PPC::ACCRCRegClass);
    MachineInstrBuilder NewPHI = BuildMI(
        *PHI->getParent(), PHI, PHI->getDebugLoc(), TII->get(PPC::PHI), AccReg);
    for (auto RegMBB : PHIOps)
      NewPHI.add(RegMBB.first).add(RegMBB.second);
    ChangedPHIMap[PHI] = NewPHI.getInstr();
  }
}

// Perform peephole optimizations.
bool PPCMIPeephole::simplifyCode() {
  bool Simplified = false;
  bool TrapOpt = false;
  MachineInstr* ToErase = nullptr;
  std::map<MachineInstr *, bool> TOCSaves;
  const TargetRegisterInfo *TRI = &TII->getRegisterInfo();
  NumFunctionsEnteredInMIPeephole++;
  if (ConvertRegReg) {
    // Fixed-point conversion of reg/reg instructions fed by load-immediate
    // into reg/imm instructions. FIXME: This is expensive, control it with
    // an option.
    bool SomethingChanged = false;
    do {
      NumFixedPointIterations++;
      SomethingChanged = false;
      for (MachineBasicBlock &MBB : *MF) {
        for (MachineInstr &MI : MBB) {
          if (MI.isDebugInstr())
            continue;

          if (TII->convertToImmediateForm(MI)) {
            // We don't erase anything in case the def has other uses. Let DCE
            // remove it if it can be removed.
            LLVM_DEBUG(dbgs() << "Converted instruction to imm form: ");
            LLVM_DEBUG(MI.dump());
            NumConvertedToImmediateForm++;
            SomethingChanged = true;
            Simplified = true;
            continue;
          }
        }
      }
    } while (SomethingChanged && FixedPointRegToImm);
  }

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {

      // If the previous instruction was marked for elimination,
      // remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }
      // If a conditional trap instruction got optimized to an
      // unconditional trap, eliminate all the instructions after
      // the trap.
      if (EnableTrapOptimization && TrapOpt) {
        ToErase = &MI;
        continue;
      }

      // Ignore debug instructions.
      if (MI.isDebugInstr())
        continue;

      // Per-opcode peepholes.
      switch (MI.getOpcode()) {

      default:
        break;
      case PPC::COPY: {
        Register Src = MI.getOperand(1).getReg();
        Register Dst = MI.getOperand(0).getReg();
        if (!Register::isVirtualRegister(Src) ||
            !Register::isVirtualRegister(Dst))
          break;
        if (MRI->getRegClass(Src) != &PPC::UACCRCRegClass ||
            MRI->getRegClass(Dst) != &PPC::ACCRCRegClass)
          break;

        // We are copying an unprimed accumulator to a primed accumulator.
        // If the input to the copy is a PHI that is fed only by (i) copies in
        // the other direction (ii) implicitly defined unprimed accumulators or
        // (iii) other PHI nodes satisfying (i) and (ii), we can change
        // the PHI to a PHI on primed accumulators (as long as we also change
        // its operands). To detect and change such copies, we first get a list
        // of all the PHI nodes starting from the root PHI node in BFS order.
        // We then visit all these PHI nodes to check if they can be changed to
        // primed accumulator PHI nodes and if so, we change them.
        MachineInstr *RootPHI = MRI->getVRegDef(Src);
        if (RootPHI->getOpcode() != PPC::PHI)
          break;

        SmallVector<MachineInstr *, 4> PHIs;
        if (!collectUnprimedAccPHIs(MRI, RootPHI, PHIs))
          break;

        convertUnprimedAccPHIs(TII, MRI, PHIs, Dst);

        ToErase = &MI;
        break;
      }
      case PPC::LI:
      case PPC::LI8: {
        // If we are materializing a zero, look for any use operands for which
        // zero means immediate zero. All such operands can be replaced with
        // PPC::ZERO.
        if (!MI.getOperand(1).isImm() || MI.getOperand(1).getImm() != 0)
          break;
        Register MIDestReg = MI.getOperand(0).getReg();
        for (MachineInstr& UseMI : MRI->use_instructions(MIDestReg))
          Simplified |= TII->onlyFoldImmediate(UseMI, MI, MIDestReg);
        if (MRI->use_nodbg_empty(MIDestReg)) {
          ++NumLoadImmZeroFoldedAndRemoved;
          ToErase = &MI;
        }
        break;
      }
      case PPC::STW:
      case PPC::STD: {
        MachineFrameInfo &MFI = MF->getFrameInfo();
        if (MFI.hasVarSizedObjects() ||
            (!MF->getSubtarget<PPCSubtarget>().isELFv2ABI() &&
             !MF->getSubtarget<PPCSubtarget>().isAIXABI()))
          break;
        // When encountering a TOC save instruction, call UpdateTOCSaves
        // to add it to the TOCSaves map and mark any existing TOC saves
        // it dominates as redundant.
        if (TII->isTOCSaveMI(MI))
          UpdateTOCSaves(TOCSaves, &MI);
        break;
      }
      case PPC::XXPERMDI: {
        // Perform simplifications of 2x64 vector swaps and splats.
        // A swap is identified by an immediate value of 2, and a splat
        // is identified by an immediate value of 0 or 3.
        int Immed = MI.getOperand(3).getImm();

        if (Immed == 1)
          break;

        // For each of these simplifications, we need the two source
        // regs to match.  Unfortunately, MachineCSE ignores COPY and
        // SUBREG_TO_REG, so for example we can see
        //   XXPERMDI t, SUBREG_TO_REG(s), SUBREG_TO_REG(s), immed.
        // We have to look through chains of COPY and SUBREG_TO_REG
        // to find the real source values for comparison.
        Register TrueReg1 =
          TRI->lookThruCopyLike(MI.getOperand(1).getReg(), MRI);
        Register TrueReg2 =
          TRI->lookThruCopyLike(MI.getOperand(2).getReg(), MRI);

        if (!(TrueReg1 == TrueReg2 && Register::isVirtualRegister(TrueReg1)))
          break;

        MachineInstr *DefMI = MRI->getVRegDef(TrueReg1);

        if (!DefMI)
          break;

        unsigned DefOpc = DefMI->getOpcode();

        // If this is a splat fed by a splatting load, the splat is
        // redundant. Replace with a copy. This doesn't happen directly due
        // to code in PPCDAGToDAGISel.cpp, but it can happen when converting
        // a load of a double to a vector of 64-bit integers.
        auto isConversionOfLoadAndSplat = [=]() -> bool {
          if (DefOpc != PPC::XVCVDPSXDS && DefOpc != PPC::XVCVDPUXDS)
            return false;
          Register FeedReg1 =
            TRI->lookThruCopyLike(DefMI->getOperand(1).getReg(), MRI);
          if (Register::isVirtualRegister(FeedReg1)) {
            MachineInstr *LoadMI = MRI->getVRegDef(FeedReg1);
            if (LoadMI && LoadMI->getOpcode() == PPC::LXVDSX)
              return true;
          }
          return false;
        };
        if ((Immed == 0 || Immed == 3) &&
            (DefOpc == PPC::LXVDSX || isConversionOfLoadAndSplat())) {
          LLVM_DEBUG(dbgs() << "Optimizing load-and-splat/splat "
                               "to load-and-splat/copy: ");
          LLVM_DEBUG(MI.dump());
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                  MI.getOperand(0).getReg())
              .add(MI.getOperand(1));
          ToErase = &MI;
          Simplified = true;
        }

        // If this is a splat or a swap fed by another splat, we
        // can replace it with a copy.
        if (DefOpc == PPC::XXPERMDI) {
          Register DefReg1 = DefMI->getOperand(1).getReg();
          Register DefReg2 = DefMI->getOperand(2).getReg();
          unsigned DefImmed = DefMI->getOperand(3).getImm();

          // If the two inputs are not the same register, check to see if
          // they originate from the same virtual register after only
          // copy-like instructions.
          if (DefReg1 != DefReg2) {
            Register FeedReg1 = TRI->lookThruCopyLike(DefReg1, MRI);
            Register FeedReg2 = TRI->lookThruCopyLike(DefReg2, MRI);

            if (!(FeedReg1 == FeedReg2 &&
                  Register::isVirtualRegister(FeedReg1)))
              break;
          }

          if (DefImmed == 0 || DefImmed == 3) {
            LLVM_DEBUG(dbgs() << "Optimizing splat/swap or splat/splat "
                                 "to splat/copy: ");
            LLVM_DEBUG(MI.dump());
            BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                    MI.getOperand(0).getReg())
                .add(MI.getOperand(1));
            ToErase = &MI;
            Simplified = true;
          }

          // If this is a splat fed by a swap, we can simplify modify
          // the splat to splat the other value from the swap's input
          // parameter.
          else if ((Immed == 0 || Immed == 3) && DefImmed == 2) {
            LLVM_DEBUG(dbgs() << "Optimizing swap/splat => splat: ");
            LLVM_DEBUG(MI.dump());
            MI.getOperand(1).setReg(DefReg1);
            MI.getOperand(2).setReg(DefReg2);
            MI.getOperand(3).setImm(3 - Immed);
            Simplified = true;
          }

          // If this is a swap fed by a swap, we can replace it
          // with a copy from the first swap's input.
          else if (Immed == 2 && DefImmed == 2) {
            LLVM_DEBUG(dbgs() << "Optimizing swap/swap => copy: ");
            LLVM_DEBUG(MI.dump());
            BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                    MI.getOperand(0).getReg())
                .add(DefMI->getOperand(1));
            ToErase = &MI;
            Simplified = true;
          }
        } else if ((Immed == 0 || Immed == 3 || Immed == 2) &&
                   DefOpc == PPC::XXPERMDIs &&
                   (DefMI->getOperand(2).getImm() == 0 ||
                    DefMI->getOperand(2).getImm() == 3)) {
          ToErase = &MI;
          Simplified = true;
          // Swap of a splat, convert to copy.
          if (Immed == 2) {
            LLVM_DEBUG(dbgs() << "Optimizing swap(splat) => copy(splat): ");
            LLVM_DEBUG(MI.dump());
            BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                    MI.getOperand(0).getReg())
                .add(MI.getOperand(1));
            break;
          }
          // Splat fed by another splat - switch the output of the first
          // and remove the second.
          DefMI->getOperand(0).setReg(MI.getOperand(0).getReg());
          LLVM_DEBUG(dbgs() << "Removing redundant splat: ");
          LLVM_DEBUG(MI.dump());
        }
        break;
      }
      case PPC::VSPLTB:
      case PPC::VSPLTH:
      case PPC::XXSPLTW: {
        unsigned MyOpcode = MI.getOpcode();
        unsigned OpNo = MyOpcode == PPC::XXSPLTW ? 1 : 2;
        Register TrueReg =
          TRI->lookThruCopyLike(MI.getOperand(OpNo).getReg(), MRI);
        if (!Register::isVirtualRegister(TrueReg))
          break;
        MachineInstr *DefMI = MRI->getVRegDef(TrueReg);
        if (!DefMI)
          break;
        unsigned DefOpcode = DefMI->getOpcode();
        auto isConvertOfSplat = [=]() -> bool {
          if (DefOpcode != PPC::XVCVSPSXWS && DefOpcode != PPC::XVCVSPUXWS)
            return false;
          Register ConvReg = DefMI->getOperand(1).getReg();
          if (!Register::isVirtualRegister(ConvReg))
            return false;
          MachineInstr *Splt = MRI->getVRegDef(ConvReg);
          return Splt && (Splt->getOpcode() == PPC::LXVWSX ||
            Splt->getOpcode() == PPC::XXSPLTW);
        };
        bool AlreadySplat = (MyOpcode == DefOpcode) ||
          (MyOpcode == PPC::VSPLTB && DefOpcode == PPC::VSPLTBs) ||
          (MyOpcode == PPC::VSPLTH && DefOpcode == PPC::VSPLTHs) ||
          (MyOpcode == PPC::XXSPLTW && DefOpcode == PPC::XXSPLTWs) ||
          (MyOpcode == PPC::XXSPLTW && DefOpcode == PPC::LXVWSX) ||
          (MyOpcode == PPC::XXSPLTW && DefOpcode == PPC::MTVSRWS)||
          (MyOpcode == PPC::XXSPLTW && isConvertOfSplat());
        // If the instruction[s] that feed this splat have already splat
        // the value, this splat is redundant.
        if (AlreadySplat) {
          LLVM_DEBUG(dbgs() << "Changing redundant splat to a copy: ");
          LLVM_DEBUG(MI.dump());
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                  MI.getOperand(0).getReg())
              .add(MI.getOperand(OpNo));
          ToErase = &MI;
          Simplified = true;
        }
        // Splat fed by a shift. Usually when we align value to splat into
        // vector element zero.
        if (DefOpcode == PPC::XXSLDWI) {
          Register ShiftRes = DefMI->getOperand(0).getReg();
          Register ShiftOp1 = DefMI->getOperand(1).getReg();
          Register ShiftOp2 = DefMI->getOperand(2).getReg();
          unsigned ShiftImm = DefMI->getOperand(3).getImm();
          unsigned SplatImm =
              MI.getOperand(MyOpcode == PPC::XXSPLTW ? 2 : 1).getImm();
          if (ShiftOp1 == ShiftOp2) {
            unsigned NewElem = (SplatImm + ShiftImm) & 0x3;
            if (MRI->hasOneNonDBGUse(ShiftRes)) {
              LLVM_DEBUG(dbgs() << "Removing redundant shift: ");
              LLVM_DEBUG(DefMI->dump());
              ToErase = DefMI;
            }
            Simplified = true;
            LLVM_DEBUG(dbgs() << "Changing splat immediate from " << SplatImm
                              << " to " << NewElem << " in instruction: ");
            LLVM_DEBUG(MI.dump());
            MI.getOperand(1).setReg(ShiftOp1);
            MI.getOperand(2).setImm(NewElem);
          }
        }
        break;
      }
      case PPC::XVCVDPSP: {
        // If this is a DP->SP conversion fed by an FRSP, the FRSP is redundant.
        Register TrueReg =
          TRI->lookThruCopyLike(MI.getOperand(1).getReg(), MRI);
        if (!Register::isVirtualRegister(TrueReg))
          break;
        MachineInstr *DefMI = MRI->getVRegDef(TrueReg);

        // This can occur when building a vector of single precision or integer
        // values.
        if (DefMI && DefMI->getOpcode() == PPC::XXPERMDI) {
          Register DefsReg1 =
            TRI->lookThruCopyLike(DefMI->getOperand(1).getReg(), MRI);
          Register DefsReg2 =
            TRI->lookThruCopyLike(DefMI->getOperand(2).getReg(), MRI);
          if (!Register::isVirtualRegister(DefsReg1) ||
              !Register::isVirtualRegister(DefsReg2))
            break;
          MachineInstr *P1 = MRI->getVRegDef(DefsReg1);
          MachineInstr *P2 = MRI->getVRegDef(DefsReg2);

          if (!P1 || !P2)
            break;

          // Remove the passed FRSP/XSRSP instruction if it only feeds this MI
          // and set any uses of that FRSP/XSRSP (in this MI) to the source of
          // the FRSP/XSRSP.
          auto removeFRSPIfPossible = [&](MachineInstr *RoundInstr) {
            unsigned Opc = RoundInstr->getOpcode();
            if ((Opc == PPC::FRSP || Opc == PPC::XSRSP) &&
                MRI->hasOneNonDBGUse(RoundInstr->getOperand(0).getReg())) {
              Simplified = true;
              Register ConvReg1 = RoundInstr->getOperand(1).getReg();
              Register FRSPDefines = RoundInstr->getOperand(0).getReg();
              MachineInstr &Use = *(MRI->use_instr_nodbg_begin(FRSPDefines));
              for (int i = 0, e = Use.getNumOperands(); i < e; ++i)
                if (Use.getOperand(i).isReg() &&
                    Use.getOperand(i).getReg() == FRSPDefines)
                  Use.getOperand(i).setReg(ConvReg1);
              LLVM_DEBUG(dbgs() << "Removing redundant FRSP/XSRSP:\n");
              LLVM_DEBUG(RoundInstr->dump());
              LLVM_DEBUG(dbgs() << "As it feeds instruction:\n");
              LLVM_DEBUG(MI.dump());
              LLVM_DEBUG(dbgs() << "Through instruction:\n");
              LLVM_DEBUG(DefMI->dump());
              RoundInstr->eraseFromParent();
            }
          };

          // If the input to XVCVDPSP is a vector that was built (even
          // partially) out of FRSP's, the FRSP(s) can safely be removed
          // since this instruction performs the same operation.
          if (P1 != P2) {
            removeFRSPIfPossible(P1);
            removeFRSPIfPossible(P2);
            break;
          }
          removeFRSPIfPossible(P1);
        }
        break;
      }
      case PPC::EXTSH:
      case PPC::EXTSH8:
      case PPC::EXTSH8_32_64: {
        if (!EnableSExtElimination) break;
        Register NarrowReg = MI.getOperand(1).getReg();
        if (!Register::isVirtualRegister(NarrowReg))
          break;

        MachineInstr *SrcMI = MRI->getVRegDef(NarrowReg);
        // If we've used a zero-extending load that we will sign-extend,
        // just do a sign-extending load.
        if (SrcMI->getOpcode() == PPC::LHZ ||
            SrcMI->getOpcode() == PPC::LHZX) {
          if (!MRI->hasOneNonDBGUse(SrcMI->getOperand(0).getReg()))
            break;
          auto is64Bit = [] (unsigned Opcode) {
            return Opcode == PPC::EXTSH8;
          };
          auto isXForm = [] (unsigned Opcode) {
            return Opcode == PPC::LHZX;
          };
          auto getSextLoadOp = [] (bool is64Bit, bool isXForm) {
            if (is64Bit)
              if (isXForm) return PPC::LHAX8;
              else         return PPC::LHA8;
            else
              if (isXForm) return PPC::LHAX;
              else         return PPC::LHA;
          };
          unsigned Opc = getSextLoadOp(is64Bit(MI.getOpcode()),
                                       isXForm(SrcMI->getOpcode()));
          LLVM_DEBUG(dbgs() << "Zero-extending load\n");
          LLVM_DEBUG(SrcMI->dump());
          LLVM_DEBUG(dbgs() << "and sign-extension\n");
          LLVM_DEBUG(MI.dump());
          LLVM_DEBUG(dbgs() << "are merged into sign-extending load\n");
          SrcMI->setDesc(TII->get(Opc));
          SrcMI->getOperand(0).setReg(MI.getOperand(0).getReg());
          ToErase = &MI;
          Simplified = true;
          NumEliminatedSExt++;
        }
        break;
      }
      case PPC::EXTSW:
      case PPC::EXTSW_32:
      case PPC::EXTSW_32_64: {
        if (!EnableSExtElimination) break;
        Register NarrowReg = MI.getOperand(1).getReg();
        if (!Register::isVirtualRegister(NarrowReg))
          break;

        MachineInstr *SrcMI = MRI->getVRegDef(NarrowReg);
        // If we've used a zero-extending load that we will sign-extend,
        // just do a sign-extending load.
        if (SrcMI->getOpcode() == PPC::LWZ ||
            SrcMI->getOpcode() == PPC::LWZX) {
          if (!MRI->hasOneNonDBGUse(SrcMI->getOperand(0).getReg()))
            break;
          auto is64Bit = [] (unsigned Opcode) {
            return Opcode == PPC::EXTSW || Opcode == PPC::EXTSW_32_64;
          };
          auto isXForm = [] (unsigned Opcode) {
            return Opcode == PPC::LWZX;
          };
          auto getSextLoadOp = [] (bool is64Bit, bool isXForm) {
            if (is64Bit)
              if (isXForm) return PPC::LWAX;
              else         return PPC::LWA;
            else
              if (isXForm) return PPC::LWAX_32;
              else         return PPC::LWA_32;
          };
          unsigned Opc = getSextLoadOp(is64Bit(MI.getOpcode()),
                                       isXForm(SrcMI->getOpcode()));
          LLVM_DEBUG(dbgs() << "Zero-extending load\n");
          LLVM_DEBUG(SrcMI->dump());
          LLVM_DEBUG(dbgs() << "and sign-extension\n");
          LLVM_DEBUG(MI.dump());
          LLVM_DEBUG(dbgs() << "are merged into sign-extending load\n");
          SrcMI->setDesc(TII->get(Opc));
          SrcMI->getOperand(0).setReg(MI.getOperand(0).getReg());
          ToErase = &MI;
          Simplified = true;
          NumEliminatedSExt++;
        } else if (MI.getOpcode() == PPC::EXTSW_32_64 &&
                   TII->isSignExtended(*SrcMI)) {
          // We can eliminate EXTSW if the input is known to be already
          // sign-extended.
          LLVM_DEBUG(dbgs() << "Removing redundant sign-extension\n");
          Register TmpReg =
              MF->getRegInfo().createVirtualRegister(&PPC::G8RCRegClass);
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::IMPLICIT_DEF),
                  TmpReg);
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::INSERT_SUBREG),
                  MI.getOperand(0).getReg())
              .addReg(TmpReg)
              .addReg(NarrowReg)
              .addImm(PPC::sub_32);
          ToErase = &MI;
          Simplified = true;
          NumEliminatedSExt++;
        }
        break;
      }
      case PPC::RLDICL: {
        // We can eliminate RLDICL (e.g. for zero-extension)
        // if all bits to clear are already zero in the input.
        // This code assume following code sequence for zero-extension.
        //   %6 = COPY %5:sub_32; (optional)
        //   %8 = IMPLICIT_DEF;
        //   %7<def,tied1> = INSERT_SUBREG %8<tied0>, %6, sub_32;
        if (!EnableZExtElimination) break;

        if (MI.getOperand(2).getImm() != 0)
          break;

        Register SrcReg = MI.getOperand(1).getReg();
        if (!Register::isVirtualRegister(SrcReg))
          break;

        MachineInstr *SrcMI = MRI->getVRegDef(SrcReg);
        if (!(SrcMI && SrcMI->getOpcode() == PPC::INSERT_SUBREG &&
              SrcMI->getOperand(0).isReg() && SrcMI->getOperand(1).isReg()))
          break;

        MachineInstr *ImpDefMI, *SubRegMI;
        ImpDefMI = MRI->getVRegDef(SrcMI->getOperand(1).getReg());
        SubRegMI = MRI->getVRegDef(SrcMI->getOperand(2).getReg());
        if (ImpDefMI->getOpcode() != PPC::IMPLICIT_DEF) break;

        SrcMI = SubRegMI;
        if (SubRegMI->getOpcode() == PPC::COPY) {
          Register CopyReg = SubRegMI->getOperand(1).getReg();
          if (Register::isVirtualRegister(CopyReg))
            SrcMI = MRI->getVRegDef(CopyReg);
        }

        unsigned KnownZeroCount = getKnownLeadingZeroCount(SrcMI, TII);
        if (MI.getOperand(3).getImm() <= KnownZeroCount) {
          LLVM_DEBUG(dbgs() << "Removing redundant zero-extension\n");
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                  MI.getOperand(0).getReg())
              .addReg(SrcReg);
          ToErase = &MI;
          Simplified = true;
          NumEliminatedZExt++;
        }
        break;
      }

      // TODO: Any instruction that has an immediate form fed only by a PHI
      // whose operands are all load immediate can be folded away. We currently
      // do this for ADD instructions, but should expand it to arithmetic and
      // binary instructions with immediate forms in the future.
      case PPC::ADD4:
      case PPC::ADD8: {
        auto isSingleUsePHI = [&](MachineOperand *PhiOp) {
          assert(PhiOp && "Invalid Operand!");
          MachineInstr *DefPhiMI = getVRegDefOrNull(PhiOp, MRI);

          return DefPhiMI && (DefPhiMI->getOpcode() == PPC::PHI) &&
                 MRI->hasOneNonDBGUse(DefPhiMI->getOperand(0).getReg());
        };

        auto dominatesAllSingleUseLIs = [&](MachineOperand *DominatorOp,
                                            MachineOperand *PhiOp) {
          assert(PhiOp && "Invalid Operand!");
          assert(DominatorOp && "Invalid Operand!");
          MachineInstr *DefPhiMI = getVRegDefOrNull(PhiOp, MRI);
          MachineInstr *DefDomMI = getVRegDefOrNull(DominatorOp, MRI);

          // Note: the vregs only show up at odd indices position of PHI Node,
          // the even indices position save the BB info.
          for (unsigned i = 1; i < DefPhiMI->getNumOperands(); i += 2) {
            MachineInstr *LiMI =
                getVRegDefOrNull(&DefPhiMI->getOperand(i), MRI);
            if (!LiMI ||
                (LiMI->getOpcode() != PPC::LI && LiMI->getOpcode() != PPC::LI8)
                || !MRI->hasOneNonDBGUse(LiMI->getOperand(0).getReg()) ||
                !MDT->dominates(DefDomMI, LiMI))
              return false;
          }

          return true;
        };

        MachineOperand Op1 = MI.getOperand(1);
        MachineOperand Op2 = MI.getOperand(2);
        if (isSingleUsePHI(&Op2) && dominatesAllSingleUseLIs(&Op1, &Op2))
          std::swap(Op1, Op2);
        else if (!isSingleUsePHI(&Op1) || !dominatesAllSingleUseLIs(&Op2, &Op1))
          break; // We don't have an ADD fed by LI's that can be transformed

        // Now we know that Op1 is the PHI node and Op2 is the dominator
        Register DominatorReg = Op2.getReg();

        const TargetRegisterClass *TRC = MI.getOpcode() == PPC::ADD8
                                             ? &PPC::G8RC_and_G8RC_NOX0RegClass
                                             : &PPC::GPRC_and_GPRC_NOR0RegClass;
        MRI->setRegClass(DominatorReg, TRC);

        // replace LIs with ADDIs
        MachineInstr *DefPhiMI = getVRegDefOrNull(&Op1, MRI);
        for (unsigned i = 1; i < DefPhiMI->getNumOperands(); i += 2) {
          MachineInstr *LiMI = getVRegDefOrNull(&DefPhiMI->getOperand(i), MRI);
          LLVM_DEBUG(dbgs() << "Optimizing LI to ADDI: ");
          LLVM_DEBUG(LiMI->dump());

          // There could be repeated registers in the PHI, e.g: %1 =
          // PHI %6, <%bb.2>, %8, <%bb.3>, %8, <%bb.6>; So if we've
          // already replaced the def instruction, skip.
          if (LiMI->getOpcode() == PPC::ADDI || LiMI->getOpcode() == PPC::ADDI8)
            continue;

          assert((LiMI->getOpcode() == PPC::LI ||
                  LiMI->getOpcode() == PPC::LI8) &&
                 "Invalid Opcode!");
          auto LiImm = LiMI->getOperand(1).getImm(); // save the imm of LI
          LiMI->removeOperand(1);                    // remove the imm of LI
          LiMI->setDesc(TII->get(LiMI->getOpcode() == PPC::LI ? PPC::ADDI
                                                              : PPC::ADDI8));
          MachineInstrBuilder(*LiMI->getParent()->getParent(), *LiMI)
              .addReg(DominatorReg)
              .addImm(LiImm); // restore the imm of LI
          LLVM_DEBUG(LiMI->dump());
        }

        // Replace ADD with COPY
        LLVM_DEBUG(dbgs() << "Optimizing ADD to COPY: ");
        LLVM_DEBUG(MI.dump());
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::COPY),
                MI.getOperand(0).getReg())
            .add(Op1);
        ToErase = &MI;
        Simplified = true;
        NumOptADDLIs++;
        break;
      }
      case PPC::RLDICR: {
        Simplified |= emitRLDICWhenLoweringJumpTables(MI) ||
                      combineSEXTAndSHL(MI, ToErase);
        break;
      }
      case PPC::RLWINM:
      case PPC::RLWINM_rec:
      case PPC::RLWINM8:
      case PPC::RLWINM8_rec: {
        Simplified = TII->combineRLWINM(MI, &ToErase);
        if (Simplified)
          ++NumRotatesCollapsed;
        break;
      }
      // We will replace TD/TW/TDI/TWI with an unconditional trap if it will
      // always trap, we will delete the node if it will never trap.
      case PPC::TDI:
      case PPC::TWI:
      case PPC::TD:
      case PPC::TW: {
        if (!EnableTrapOptimization) break;
        MachineInstr *LiMI1 = getVRegDefOrNull(&MI.getOperand(1), MRI);
        MachineInstr *LiMI2 = getVRegDefOrNull(&MI.getOperand(2), MRI);
        bool IsOperand2Immediate = MI.getOperand(2).isImm();
        // We can only do the optimization if we can get immediates
        // from both operands
        if (!(LiMI1 && (LiMI1->getOpcode() == PPC::LI ||
                        LiMI1->getOpcode() == PPC::LI8)))
          break;
        if (!IsOperand2Immediate &&
            !(LiMI2 && (LiMI2->getOpcode() == PPC::LI ||
                        LiMI2->getOpcode() == PPC::LI8)))
          break;

        auto ImmOperand0 = MI.getOperand(0).getImm();
        auto ImmOperand1 = LiMI1->getOperand(1).getImm();
        auto ImmOperand2 = IsOperand2Immediate ? MI.getOperand(2).getImm()
                                               : LiMI2->getOperand(1).getImm();

        // We will replace the MI with an unconditional trap if it will always
        // trap.
        if ((ImmOperand0 == 31) ||
            ((ImmOperand0 & 0x10) &&
             ((int64_t)ImmOperand1 < (int64_t)ImmOperand2)) ||
            ((ImmOperand0 & 0x8) &&
             ((int64_t)ImmOperand1 > (int64_t)ImmOperand2)) ||
            ((ImmOperand0 & 0x2) &&
             ((uint64_t)ImmOperand1 < (uint64_t)ImmOperand2)) ||
            ((ImmOperand0 & 0x1) &&
             ((uint64_t)ImmOperand1 > (uint64_t)ImmOperand2)) ||
            ((ImmOperand0 & 0x4) && (ImmOperand1 == ImmOperand2))) {
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(PPC::TRAP));
          TrapOpt = true;
        }
        // We will delete the MI if it will never trap.
        ToErase = &MI;
        Simplified = true;
        break;
      }
      }
    }

    // If the last instruction was marked for elimination,
    // remove it now.
    if (ToErase) {
      ToErase->eraseFromParent();
      ToErase = nullptr;
    }
    // Reset TrapOpt to false at the end of the basic block.
    if (EnableTrapOptimization)
      TrapOpt = false;
  }

  // Eliminate all the TOC save instructions which are redundant.
  Simplified |= eliminateRedundantTOCSaves(TOCSaves);
  PPCFunctionInfo *FI = MF->getInfo<PPCFunctionInfo>();
  if (FI->mustSaveTOC())
    NumTOCSavesInPrologue++;

  // We try to eliminate redundant compare instruction.
  Simplified |= eliminateRedundantCompare();

  return Simplified;
}

// helper functions for eliminateRedundantCompare
static bool isEqOrNe(MachineInstr *BI) {
  PPC::Predicate Pred = (PPC::Predicate)BI->getOperand(0).getImm();
  unsigned PredCond = PPC::getPredicateCondition(Pred);
  return (PredCond == PPC::PRED_EQ || PredCond == PPC::PRED_NE);
}

static bool isSupportedCmpOp(unsigned opCode) {
  return (opCode == PPC::CMPLD  || opCode == PPC::CMPD  ||
          opCode == PPC::CMPLW  || opCode == PPC::CMPW  ||
          opCode == PPC::CMPLDI || opCode == PPC::CMPDI ||
          opCode == PPC::CMPLWI || opCode == PPC::CMPWI);
}

static bool is64bitCmpOp(unsigned opCode) {
  return (opCode == PPC::CMPLD  || opCode == PPC::CMPD ||
          opCode == PPC::CMPLDI || opCode == PPC::CMPDI);
}

static bool isSignedCmpOp(unsigned opCode) {
  return (opCode == PPC::CMPD  || opCode == PPC::CMPW ||
          opCode == PPC::CMPDI || opCode == PPC::CMPWI);
}

static unsigned getSignedCmpOpCode(unsigned opCode) {
  if (opCode == PPC::CMPLD)  return PPC::CMPD;
  if (opCode == PPC::CMPLW)  return PPC::CMPW;
  if (opCode == PPC::CMPLDI) return PPC::CMPDI;
  if (opCode == PPC::CMPLWI) return PPC::CMPWI;
  return opCode;
}

// We can decrement immediate x in (GE x) by changing it to (GT x-1) or
// (LT x) to (LE x-1)
static unsigned getPredicateToDecImm(MachineInstr *BI, MachineInstr *CMPI) {
  uint64_t Imm = CMPI->getOperand(2).getImm();
  bool SignedCmp = isSignedCmpOp(CMPI->getOpcode());
  if ((!SignedCmp && Imm == 0) || (SignedCmp && Imm == 0x8000))
    return 0;

  PPC::Predicate Pred = (PPC::Predicate)BI->getOperand(0).getImm();
  unsigned PredCond = PPC::getPredicateCondition(Pred);
  unsigned PredHint = PPC::getPredicateHint(Pred);
  if (PredCond == PPC::PRED_GE)
    return PPC::getPredicate(PPC::PRED_GT, PredHint);
  if (PredCond == PPC::PRED_LT)
    return PPC::getPredicate(PPC::PRED_LE, PredHint);

  return 0;
}

// We can increment immediate x in (GT x) by changing it to (GE x+1) or
// (LE x) to (LT x+1)
static unsigned getPredicateToIncImm(MachineInstr *BI, MachineInstr *CMPI) {
  uint64_t Imm = CMPI->getOperand(2).getImm();
  bool SignedCmp = isSignedCmpOp(CMPI->getOpcode());
  if ((!SignedCmp && Imm == 0xFFFF) || (SignedCmp && Imm == 0x7FFF))
    return 0;

  PPC::Predicate Pred = (PPC::Predicate)BI->getOperand(0).getImm();
  unsigned PredCond = PPC::getPredicateCondition(Pred);
  unsigned PredHint = PPC::getPredicateHint(Pred);
  if (PredCond == PPC::PRED_GT)
    return PPC::getPredicate(PPC::PRED_GE, PredHint);
  if (PredCond == PPC::PRED_LE)
    return PPC::getPredicate(PPC::PRED_LT, PredHint);

  return 0;
}

// This takes a Phi node and returns a register value for the specified BB.
static unsigned getIncomingRegForBlock(MachineInstr *Phi,
                                       MachineBasicBlock *MBB) {
  for (unsigned I = 2, E = Phi->getNumOperands() + 1; I != E; I += 2) {
    MachineOperand &MO = Phi->getOperand(I);
    if (MO.getMBB() == MBB)
      return Phi->getOperand(I-1).getReg();
  }
  llvm_unreachable("invalid src basic block for this Phi node\n");
  return 0;
}

// This function tracks the source of the register through register copy.
// If BB1 and BB2 are non-NULL, we also track PHI instruction in BB2
// assuming that the control comes from BB1 into BB2.
static unsigned getSrcVReg(unsigned Reg, MachineBasicBlock *BB1,
                           MachineBasicBlock *BB2, MachineRegisterInfo *MRI) {
  unsigned SrcReg = Reg;
  while (true) {
    unsigned NextReg = SrcReg;
    MachineInstr *Inst = MRI->getVRegDef(SrcReg);
    if (BB1 && Inst->getOpcode() == PPC::PHI && Inst->getParent() == BB2) {
      NextReg = getIncomingRegForBlock(Inst, BB1);
      // We track through PHI only once to avoid infinite loop.
      BB1 = nullptr;
    }
    else if (Inst->isFullCopy())
      NextReg = Inst->getOperand(1).getReg();
    if (NextReg == SrcReg || !Register::isVirtualRegister(NextReg))
      break;
    SrcReg = NextReg;
  }
  return SrcReg;
}

static bool eligibleForCompareElimination(MachineBasicBlock &MBB,
                                          MachineBasicBlock *&PredMBB,
                                          MachineBasicBlock *&MBBtoMoveCmp,
                                          MachineRegisterInfo *MRI) {

  auto isEligibleBB = [&](MachineBasicBlock &BB) {
    auto BII = BB.getFirstInstrTerminator();
    // We optimize BBs ending with a conditional branch.
    // We check only for BCC here, not BCCLR, because BCCLR
    // will be formed only later in the pipeline.
    if (BB.succ_size() == 2 &&
        BII != BB.instr_end() &&
        (*BII).getOpcode() == PPC::BCC &&
        (*BII).getOperand(1).isReg()) {
      // We optimize only if the condition code is used only by one BCC.
      Register CndReg = (*BII).getOperand(1).getReg();
      if (!Register::isVirtualRegister(CndReg) || !MRI->hasOneNonDBGUse(CndReg))
        return false;

      MachineInstr *CMPI = MRI->getVRegDef(CndReg);
      // We assume compare and branch are in the same BB for ease of analysis.
      if (CMPI->getParent() != &BB)
        return false;

      // We skip this BB if a physical register is used in comparison.
      for (MachineOperand &MO : CMPI->operands())
        if (MO.isReg() && !Register::isVirtualRegister(MO.getReg()))
          return false;

      return true;
    }
    return false;
  };

  // If this BB has more than one successor, we can create a new BB and
  // move the compare instruction in the new BB.
  // So far, we do not move compare instruction to a BB having multiple
  // successors to avoid potentially increasing code size.
  auto isEligibleForMoveCmp = [](MachineBasicBlock &BB) {
    return BB.succ_size() == 1;
  };

  if (!isEligibleBB(MBB))
    return false;

  unsigned NumPredBBs = MBB.pred_size();
  if (NumPredBBs == 1) {
    MachineBasicBlock *TmpMBB = *MBB.pred_begin();
    if (isEligibleBB(*TmpMBB)) {
      PredMBB = TmpMBB;
      MBBtoMoveCmp = nullptr;
      return true;
    }
  }
  else if (NumPredBBs == 2) {
    // We check for partially redundant case.
    // So far, we support cases with only two predecessors
    // to avoid increasing the number of instructions.
    MachineBasicBlock::pred_iterator PI = MBB.pred_begin();
    MachineBasicBlock *Pred1MBB = *PI;
    MachineBasicBlock *Pred2MBB = *(PI+1);

    if (isEligibleBB(*Pred1MBB) && isEligibleForMoveCmp(*Pred2MBB)) {
      // We assume Pred1MBB is the BB containing the compare to be merged and
      // Pred2MBB is the BB to which we will append a compare instruction.
      // Hence we can proceed as is.
    }
    else if (isEligibleBB(*Pred2MBB) && isEligibleForMoveCmp(*Pred1MBB)) {
      // We need to swap Pred1MBB and Pred2MBB to canonicalize.
      std::swap(Pred1MBB, Pred2MBB);
    }
    else return false;

    // Here, Pred2MBB is the BB to which we need to append a compare inst.
    // We cannot move the compare instruction if operands are not available
    // in Pred2MBB (i.e. defined in MBB by an instruction other than PHI).
    MachineInstr *BI = &*MBB.getFirstInstrTerminator();
    MachineInstr *CMPI = MRI->getVRegDef(BI->getOperand(1).getReg());
    for (int I = 1; I <= 2; I++)
      if (CMPI->getOperand(I).isReg()) {
        MachineInstr *Inst = MRI->getVRegDef(CMPI->getOperand(I).getReg());
        if (Inst->getParent() == &MBB && Inst->getOpcode() != PPC::PHI)
          return false;
      }

    PredMBB = Pred1MBB;
    MBBtoMoveCmp = Pred2MBB;
    return true;
  }

  return false;
}

// This function will iterate over the input map containing a pair of TOC save
// instruction and a flag. The flag will be set to false if the TOC save is
// proven redundant. This function will erase from the basic block all the TOC
// saves marked as redundant.
bool PPCMIPeephole::eliminateRedundantTOCSaves(
    std::map<MachineInstr *, bool> &TOCSaves) {
  bool Simplified = false;
  int NumKept = 0;
  for (auto TOCSave : TOCSaves) {
    if (!TOCSave.second) {
      TOCSave.first->eraseFromParent();
      RemoveTOCSave++;
      Simplified = true;
    } else {
      NumKept++;
    }
  }

  if (NumKept > 1)
    MultiTOCSaves++;

  return Simplified;
}

// If multiple conditional branches are executed based on the (essentially)
// same comparison, we merge compare instructions into one and make multiple
// conditional branches on this comparison.
// For example,
//   if (a == 0) { ... }
//   else if (a < 0) { ... }
// can be executed by one compare and two conditional branches instead of
// two pairs of a compare and a conditional branch.
//
// This method merges two compare instructions in two MBBs and modifies the
// compare and conditional branch instructions if needed.
// For the above example, the input for this pass looks like:
//   cmplwi r3, 0
//   beq    0, .LBB0_3
//   cmpwi  r3, -1
//   bgt    0, .LBB0_4
// So, before merging two compares, we need to modify these instructions as
//   cmpwi  r3, 0       ; cmplwi and cmpwi yield same result for beq
//   beq    0, .LBB0_3
//   cmpwi  r3, 0       ; greather than -1 means greater or equal to 0
//   bge    0, .LBB0_4

bool PPCMIPeephole::eliminateRedundantCompare() {
  bool Simplified = false;

  for (MachineBasicBlock &MBB2 : *MF) {
    MachineBasicBlock *MBB1 = nullptr, *MBBtoMoveCmp = nullptr;

    // For fully redundant case, we select two basic blocks MBB1 and MBB2
    // as an optimization target if
    // - both MBBs end with a conditional branch,
    // - MBB1 is the only predecessor of MBB2, and
    // - compare does not take a physical register as a operand in both MBBs.
    // In this case, eligibleForCompareElimination sets MBBtoMoveCmp nullptr.
    //
    // As partially redundant case, we additionally handle if MBB2 has one
    // additional predecessor, which has only one successor (MBB2).
    // In this case, we move the compare instruction originally in MBB2 into
    // MBBtoMoveCmp. This partially redundant case is typically appear by
    // compiling a while loop; here, MBBtoMoveCmp is the loop preheader.
    //
    // Overview of CFG of related basic blocks
    // Fully redundant case        Partially redundant case
    //   --------                   ----------------  --------
    //   | MBB1 | (w/ 2 succ)       | MBBtoMoveCmp |  | MBB1 | (w/ 2 succ)
    //   --------                   ----------------  --------
    //      |    \                     (w/ 1 succ) \     |    \
    //      |     \                                 \    |     \
    //      |                                        \   |
    //   --------                                     --------
    //   | MBB2 | (w/ 1 pred                          | MBB2 | (w/ 2 pred
    //   -------- and 2 succ)                         -------- and 2 succ)
    //      |    \                                       |    \
    //      |     \                                      |     \
    //
    if (!eligibleForCompareElimination(MBB2, MBB1, MBBtoMoveCmp, MRI))
      continue;

    MachineInstr *BI1   = &*MBB1->getFirstInstrTerminator();
    MachineInstr *CMPI1 = MRI->getVRegDef(BI1->getOperand(1).getReg());

    MachineInstr *BI2   = &*MBB2.getFirstInstrTerminator();
    MachineInstr *CMPI2 = MRI->getVRegDef(BI2->getOperand(1).getReg());
    bool IsPartiallyRedundant = (MBBtoMoveCmp != nullptr);

    // We cannot optimize an unsupported compare opcode or
    // a mix of 32-bit and 64-bit comaprisons
    if (!isSupportedCmpOp(CMPI1->getOpcode()) ||
        !isSupportedCmpOp(CMPI2->getOpcode()) ||
        is64bitCmpOp(CMPI1->getOpcode()) != is64bitCmpOp(CMPI2->getOpcode()))
      continue;

    unsigned NewOpCode = 0;
    unsigned NewPredicate1 = 0, NewPredicate2 = 0;
    int16_t Imm1 = 0, NewImm1 = 0, Imm2 = 0, NewImm2 = 0;
    bool SwapOperands = false;

    if (CMPI1->getOpcode() != CMPI2->getOpcode()) {
      // Typically, unsigned comparison is used for equality check, but
      // we replace it with a signed comparison if the comparison
      // to be merged is a signed comparison.
      // In other cases of opcode mismatch, we cannot optimize this.

      // We cannot change opcode when comparing against an immediate
      // if the most significant bit of the immediate is one
      // due to the difference in sign extension.
      auto CmpAgainstImmWithSignBit = [](MachineInstr *I) {
        if (!I->getOperand(2).isImm())
          return false;
        int16_t Imm = (int16_t)I->getOperand(2).getImm();
        return Imm < 0;
      };

      if (isEqOrNe(BI2) && !CmpAgainstImmWithSignBit(CMPI2) &&
          CMPI1->getOpcode() == getSignedCmpOpCode(CMPI2->getOpcode()))
        NewOpCode = CMPI1->getOpcode();
      else if (isEqOrNe(BI1) && !CmpAgainstImmWithSignBit(CMPI1) &&
               getSignedCmpOpCode(CMPI1->getOpcode()) == CMPI2->getOpcode())
        NewOpCode = CMPI2->getOpcode();
      else continue;
    }

    if (CMPI1->getOperand(2).isReg() && CMPI2->getOperand(2).isReg()) {
      // In case of comparisons between two registers, these two registers
      // must be same to merge two comparisons.
      unsigned Cmp1Operand1 = getSrcVReg(CMPI1->getOperand(1).getReg(),
                                         nullptr, nullptr, MRI);
      unsigned Cmp1Operand2 = getSrcVReg(CMPI1->getOperand(2).getReg(),
                                         nullptr, nullptr, MRI);
      unsigned Cmp2Operand1 = getSrcVReg(CMPI2->getOperand(1).getReg(),
                                         MBB1, &MBB2, MRI);
      unsigned Cmp2Operand2 = getSrcVReg(CMPI2->getOperand(2).getReg(),
                                         MBB1, &MBB2, MRI);

      if (Cmp1Operand1 == Cmp2Operand1 && Cmp1Operand2 == Cmp2Operand2) {
        // Same pair of registers in the same order; ready to merge as is.
      }
      else if (Cmp1Operand1 == Cmp2Operand2 && Cmp1Operand2 == Cmp2Operand1) {
        // Same pair of registers in different order.
        // We reverse the predicate to merge compare instructions.
        PPC::Predicate Pred = (PPC::Predicate)BI2->getOperand(0).getImm();
        NewPredicate2 = (unsigned)PPC::getSwappedPredicate(Pred);
        // In case of partial redundancy, we need to swap operands
        // in another compare instruction.
        SwapOperands = true;
      }
      else continue;
    }
    else if (CMPI1->getOperand(2).isImm() && CMPI2->getOperand(2).isImm()) {
      // In case of comparisons between a register and an immediate,
      // the operand register must be same for two compare instructions.
      unsigned Cmp1Operand1 = getSrcVReg(CMPI1->getOperand(1).getReg(),
                                         nullptr, nullptr, MRI);
      unsigned Cmp2Operand1 = getSrcVReg(CMPI2->getOperand(1).getReg(),
                                         MBB1, &MBB2, MRI);
      if (Cmp1Operand1 != Cmp2Operand1)
        continue;

      NewImm1 = Imm1 = (int16_t)CMPI1->getOperand(2).getImm();
      NewImm2 = Imm2 = (int16_t)CMPI2->getOperand(2).getImm();

      // If immediate are not same, we try to adjust by changing predicate;
      // e.g. GT imm means GE (imm+1).
      if (Imm1 != Imm2 && (!isEqOrNe(BI2) || !isEqOrNe(BI1))) {
        int Diff = Imm1 - Imm2;
        if (Diff < -2 || Diff > 2)
          continue;

        unsigned PredToInc1 = getPredicateToIncImm(BI1, CMPI1);
        unsigned PredToDec1 = getPredicateToDecImm(BI1, CMPI1);
        unsigned PredToInc2 = getPredicateToIncImm(BI2, CMPI2);
        unsigned PredToDec2 = getPredicateToDecImm(BI2, CMPI2);
        if (Diff == 2) {
          if (PredToInc2 && PredToDec1) {
            NewPredicate2 = PredToInc2;
            NewPredicate1 = PredToDec1;
            NewImm2++;
            NewImm1--;
          }
        }
        else if (Diff == 1) {
          if (PredToInc2) {
            NewImm2++;
            NewPredicate2 = PredToInc2;
          }
          else if (PredToDec1) {
            NewImm1--;
            NewPredicate1 = PredToDec1;
          }
        }
        else if (Diff == -1) {
          if (PredToDec2) {
            NewImm2--;
            NewPredicate2 = PredToDec2;
          }
          else if (PredToInc1) {
            NewImm1++;
            NewPredicate1 = PredToInc1;
          }
        }
        else if (Diff == -2) {
          if (PredToDec2 && PredToInc1) {
            NewPredicate2 = PredToDec2;
            NewPredicate1 = PredToInc1;
            NewImm2--;
            NewImm1++;
          }
        }
      }

      // We cannot merge two compares if the immediates are not same.
      if (NewImm2 != NewImm1)
        continue;
    }

    LLVM_DEBUG(dbgs() << "Optimize two pairs of compare and branch:\n");
    LLVM_DEBUG(CMPI1->dump());
    LLVM_DEBUG(BI1->dump());
    LLVM_DEBUG(CMPI2->dump());
    LLVM_DEBUG(BI2->dump());

    // We adjust opcode, predicates and immediate as we determined above.
    if (NewOpCode != 0 && NewOpCode != CMPI1->getOpcode()) {
      CMPI1->setDesc(TII->get(NewOpCode));
    }
    if (NewPredicate1) {
      BI1->getOperand(0).setImm(NewPredicate1);
    }
    if (NewPredicate2) {
      BI2->getOperand(0).setImm(NewPredicate2);
    }
    if (NewImm1 != Imm1) {
      CMPI1->getOperand(2).setImm(NewImm1);
    }

    if (IsPartiallyRedundant) {
      // We touch up the compare instruction in MBB2 and move it to
      // a previous BB to handle partially redundant case.
      if (SwapOperands) {
        Register Op1 = CMPI2->getOperand(1).getReg();
        Register Op2 = CMPI2->getOperand(2).getReg();
        CMPI2->getOperand(1).setReg(Op2);
        CMPI2->getOperand(2).setReg(Op1);
      }
      if (NewImm2 != Imm2)
        CMPI2->getOperand(2).setImm(NewImm2);

      for (int I = 1; I <= 2; I++) {
        if (CMPI2->getOperand(I).isReg()) {
          MachineInstr *Inst = MRI->getVRegDef(CMPI2->getOperand(I).getReg());
          if (Inst->getParent() != &MBB2)
            continue;

          assert(Inst->getOpcode() == PPC::PHI &&
                 "We cannot support if an operand comes from this BB.");
          unsigned SrcReg = getIncomingRegForBlock(Inst, MBBtoMoveCmp);
          CMPI2->getOperand(I).setReg(SrcReg);
        }
      }
      auto I = MachineBasicBlock::iterator(MBBtoMoveCmp->getFirstTerminator());
      MBBtoMoveCmp->splice(I, &MBB2, MachineBasicBlock::iterator(CMPI2));

      DebugLoc DL = CMPI2->getDebugLoc();
      Register NewVReg = MRI->createVirtualRegister(&PPC::CRRCRegClass);
      BuildMI(MBB2, MBB2.begin(), DL,
              TII->get(PPC::PHI), NewVReg)
        .addReg(BI1->getOperand(1).getReg()).addMBB(MBB1)
        .addReg(BI2->getOperand(1).getReg()).addMBB(MBBtoMoveCmp);
      BI2->getOperand(1).setReg(NewVReg);
    }
    else {
      // We finally eliminate compare instruction in MBB2.
      BI2->getOperand(1).setReg(BI1->getOperand(1).getReg());
      CMPI2->eraseFromParent();
    }
    BI2->getOperand(1).setIsKill(true);
    BI1->getOperand(1).setIsKill(false);

    LLVM_DEBUG(dbgs() << "into a compare and two branches:\n");
    LLVM_DEBUG(CMPI1->dump());
    LLVM_DEBUG(BI1->dump());
    LLVM_DEBUG(BI2->dump());
    if (IsPartiallyRedundant) {
      LLVM_DEBUG(dbgs() << "The following compare is moved into "
                        << printMBBReference(*MBBtoMoveCmp)
                        << " to handle partial redundancy.\n");
      LLVM_DEBUG(CMPI2->dump());
    }

    Simplified = true;
  }

  return Simplified;
}

// We miss the opportunity to emit an RLDIC when lowering jump tables
// since ISEL sees only a single basic block. When selecting, the clear
// and shift left will be in different blocks.
bool PPCMIPeephole::emitRLDICWhenLoweringJumpTables(MachineInstr &MI) {
  if (MI.getOpcode() != PPC::RLDICR)
    return false;

  Register SrcReg = MI.getOperand(1).getReg();
  if (!Register::isVirtualRegister(SrcReg))
    return false;

  MachineInstr *SrcMI = MRI->getVRegDef(SrcReg);
  if (SrcMI->getOpcode() != PPC::RLDICL)
    return false;

  MachineOperand MOpSHSrc = SrcMI->getOperand(2);
  MachineOperand MOpMBSrc = SrcMI->getOperand(3);
  MachineOperand MOpSHMI = MI.getOperand(2);
  MachineOperand MOpMEMI = MI.getOperand(3);
  if (!(MOpSHSrc.isImm() && MOpMBSrc.isImm() && MOpSHMI.isImm() &&
        MOpMEMI.isImm()))
    return false;

  uint64_t SHSrc = MOpSHSrc.getImm();
  uint64_t MBSrc = MOpMBSrc.getImm();
  uint64_t SHMI = MOpSHMI.getImm();
  uint64_t MEMI = MOpMEMI.getImm();
  uint64_t NewSH = SHSrc + SHMI;
  uint64_t NewMB = MBSrc - SHMI;
  if (NewMB > 63 || NewSH > 63)
    return false;

  // The bits cleared with RLDICL are [0, MBSrc).
  // The bits cleared with RLDICR are (MEMI, 63].
  // After the sequence, the bits cleared are:
  // [0, MBSrc-SHMI) and (MEMI, 63).
  //
  // The bits cleared with RLDIC are [0, NewMB) and (63-NewSH, 63].
  if ((63 - NewSH) != MEMI)
    return false;

  LLVM_DEBUG(dbgs() << "Converting pair: ");
  LLVM_DEBUG(SrcMI->dump());
  LLVM_DEBUG(MI.dump());

  MI.setDesc(TII->get(PPC::RLDIC));
  MI.getOperand(1).setReg(SrcMI->getOperand(1).getReg());
  MI.getOperand(2).setImm(NewSH);
  MI.getOperand(3).setImm(NewMB);
  MI.getOperand(1).setIsKill(SrcMI->getOperand(1).isKill());
  SrcMI->getOperand(1).setIsKill(false);

  LLVM_DEBUG(dbgs() << "To: ");
  LLVM_DEBUG(MI.dump());
  NumRotatesCollapsed++;
  // If SrcReg has no non-debug use it's safe to delete its def SrcMI.
  if (MRI->use_nodbg_empty(SrcReg)) {
    assert(!SrcMI->hasImplicitDef() &&
           "Not expecting an implicit def with this instr.");
    SrcMI->eraseFromParent();
  }
  return true;
}

// For case in LLVM IR
// entry:
//   %iconv = sext i32 %index to i64
//   br i1 undef label %true, label %false
// true:
//   %ptr = getelementptr inbounds i32, i32* null, i64 %iconv
// ...
// PPCISelLowering::combineSHL fails to combine, because sext and shl are in
// different BBs when conducting instruction selection. We can do a peephole
// optimization to combine these two instructions into extswsli after
// instruction selection.
bool PPCMIPeephole::combineSEXTAndSHL(MachineInstr &MI,
                                      MachineInstr *&ToErase) {
  if (MI.getOpcode() != PPC::RLDICR)
    return false;

  if (!MF->getSubtarget<PPCSubtarget>().isISA3_0())
    return false;

  assert(MI.getNumOperands() == 4 && "RLDICR should have 4 operands");

  MachineOperand MOpSHMI = MI.getOperand(2);
  MachineOperand MOpMEMI = MI.getOperand(3);
  if (!(MOpSHMI.isImm() && MOpMEMI.isImm()))
    return false;

  uint64_t SHMI = MOpSHMI.getImm();
  uint64_t MEMI = MOpMEMI.getImm();
  if (SHMI + MEMI != 63)
    return false;

  Register SrcReg = MI.getOperand(1).getReg();
  if (!Register::isVirtualRegister(SrcReg))
    return false;

  MachineInstr *SrcMI = MRI->getVRegDef(SrcReg);
  if (SrcMI->getOpcode() != PPC::EXTSW &&
      SrcMI->getOpcode() != PPC::EXTSW_32_64)
    return false;

  // If the register defined by extsw has more than one use, combination is not
  // needed.
  if (!MRI->hasOneNonDBGUse(SrcReg))
    return false;

  assert(SrcMI->getNumOperands() == 2 && "EXTSW should have 2 operands");
  assert(SrcMI->getOperand(1).isReg() &&
         "EXTSW's second operand should be a register");
  if (!Register::isVirtualRegister(SrcMI->getOperand(1).getReg()))
    return false;

  LLVM_DEBUG(dbgs() << "Combining pair: ");
  LLVM_DEBUG(SrcMI->dump());
  LLVM_DEBUG(MI.dump());

  MachineInstr *NewInstr =
      BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(),
              SrcMI->getOpcode() == PPC::EXTSW ? TII->get(PPC::EXTSWSLI)
                                               : TII->get(PPC::EXTSWSLI_32_64),
              MI.getOperand(0).getReg())
          .add(SrcMI->getOperand(1))
          .add(MOpSHMI);
  (void)NewInstr;

  LLVM_DEBUG(dbgs() << "TO: ");
  LLVM_DEBUG(NewInstr->dump());
  ++NumEXTSWAndSLDICombined;
  ToErase = &MI;
  // SrcMI, which is extsw, is of no use now, erase it.
  SrcMI->eraseFromParent();
  return true;
}

} // end default namespace

INITIALIZE_PASS_BEGIN(PPCMIPeephole, DEBUG_TYPE,
                      "PowerPC MI Peephole Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(PPCMIPeephole, DEBUG_TYPE,
                    "PowerPC MI Peephole Optimization", false, false)

char PPCMIPeephole::ID = 0;
FunctionPass*
llvm::createPPCMIPeepholePass() { return new PPCMIPeephole(); }
