//====-- X86CmovConversion.cpp - Convert Cmov to Branch -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass that converts X86 cmov instructions into branch
/// when profitable. This pass is conservative, i.e., it applies transformation
/// if and only if it can gaurantee a gain with high confidence.
///
/// Thus, the optimization applies under the following conditions:
///   1. Consider as a candidate only CMOV in most inner loop, assuming that
///       most hotspots are represented by these loops.
///   2. Given a group of CMOV instructions, that are using same EFLAGS def
///      instruction:
///      a. Consider them as candidates only if all have same code condition or
///         opposite one, to prevent generating more than one conditional jump
///         per EFLAGS def instruction.
///      b. Consider them as candidates only if all are profitable to be
///         converted, assuming that one bad conversion may casue a degradation.
///   3. Apply conversion only for loop that are found profitable and only for
///      CMOV candidates that were found profitable.
///      a. Loop is considered profitable only if conversion will reduce its
///         depth cost by some thrishold.
///      b. CMOV is considered profitable if the cost of its condition is higher
///         than the average cost of its true-value and false-value by 25% of
///         branch-misprediction-penalty, this to assure no degredassion even
///         with 25% branch misprediction.
///
/// Note: This pass is assumed to run on SSA machine code.
//===----------------------------------------------------------------------===//
//
//  External interfaces:
//      FunctionPass *llvm::createX86CmovConverterPass();
//      bool X86CmovConverterPass::runOnMachineFunction(MachineFunction &MF);
//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "x86-cmov-converter"

STATISTIC(NumOfSkippedCmovGroups, "Number of unsupported CMOV-groups");
STATISTIC(NumOfCmovGroupCandidate, "Number of CMOV-group candidates");
STATISTIC(NumOfLoopCandidate, "Number of CMOV-conversion profitable loops");
STATISTIC(NumOfOptimizedCmovGroups, "Number of optimized CMOV-groups");

namespace {
// This internal switch can be used to turn off the cmov/branch optimization.
static cl::opt<bool>
    EnableCmovConverter("x86-cmov-converter",
                        cl::desc("Enable the X86 cmov-to-branch optimization."),
                        cl::init(true), cl::Hidden);

/// Converts X86 cmov instructions into branches when profitable.
class X86CmovConverterPass : public MachineFunctionPass {
public:
  X86CmovConverterPass() : MachineFunctionPass(ID) {}
  ~X86CmovConverterPass() {}

  StringRef getPassName() const override { return "X86 cmov Conversion"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  /// Pass identification, replacement for typeid.
  static char ID;

  const MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  TargetSchedModel TSchedModel;

  /// List of consecutive CMOV instructions.
  typedef SmallVector<MachineInstr *, 2> CmovGroup;
  typedef SmallVector<CmovGroup, 2> CmovGroups;

  /// Collect all CMOV-group-candidates in \p CurrLoop and update \p
  /// CmovInstGroups accordingly.
  ///
  /// \param CurrLoop Loop being processed.
  /// \param CmovInstGroups List of consecutive CMOV instructions in CurrLoop.
  /// \returns true iff it found any CMOV-group-candidate.
  bool collectCmovCandidates(MachineLoop *CurrLoop, CmovGroups &CmovInstGroups);

  /// Check if it is profitable to transform each CMOV-group-candidates into
  /// branch. Remove all groups that are not profitable from \p CmovInstGroups.
  ///
  /// \param CurrLoop Loop being processed.
  /// \param CmovInstGroups List of consecutive CMOV instructions in CurrLoop.
  /// \returns true iff any CMOV-group-candidate remain.
  bool checkForProfitableCmovCandidates(MachineLoop *CurrLoop,
                                        CmovGroups &CmovInstGroups);

  /// Convert the given list of consecutive CMOV instructions into a branch.
  ///
  /// \param Group Consecutive CMOV instructions to be converted into branch.
  void convertCmovInstsToBranches(SmallVectorImpl<MachineInstr *> &Group) const;
};

char X86CmovConverterPass::ID = 0;

void X86CmovConverterPass::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<MachineLoopInfo>();
}

bool X86CmovConverterPass::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;
  if (!EnableCmovConverter)
    return false;

  DEBUG(dbgs() << "********** " << getPassName() << " : " << MF.getName()
               << "**********\n");

  bool Changed = false;
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  MRI = &MF.getRegInfo();
  TII = STI.getInstrInfo();
  TSchedModel.init(STI.getSchedModel(), &STI, TII);

  //===--------------------------------------------------------------------===//
  // Algorithm
  // ---------
  //   For each inner most loop
  //     collectCmovCandidates() {
  //       Find all CMOV-group-candidates.
  //     }
  //
  //     checkForProfitableCmovCandidates() {
  //       * Calculate both loop-depth and optimized-loop-depth.
  //       * Use these depth to check for loop transformation profitability.
  //       * Check for CMOV-group-candidate transformation profitability.
  //     }
  //
  //     For each profitable CMOV-group-candidate
  //       convertCmovInstsToBranches() {
  //           * Create FalseBB, SinkBB, Conditional branch to SinkBB.
  //           * Replace each CMOV instruction with a PHI instruction in SinkBB.
  //       }
  //
  // Note: For more details, see each function description.
  //===--------------------------------------------------------------------===//
  for (MachineBasicBlock &MBB : MF) {
    MachineLoop *CurrLoop = MLI.getLoopFor(&MBB);

    // Optimize only inner most loops.
    if (!CurrLoop || CurrLoop->getHeader() != &MBB ||
        !CurrLoop->getSubLoops().empty())
      continue;

    // List of consecutive CMOV instructions to be processed.
    CmovGroups CmovInstGroups;

    if (!collectCmovCandidates(CurrLoop, CmovInstGroups))
      continue;

    if (!checkForProfitableCmovCandidates(CurrLoop, CmovInstGroups))
      continue;

    Changed = true;
    for (auto &Group : CmovInstGroups)
      convertCmovInstsToBranches(Group);
  }
  return Changed;
}

bool X86CmovConverterPass::collectCmovCandidates(MachineLoop *CurrLoop,
                                                 CmovGroups &CmovInstGroups) {
  //===--------------------------------------------------------------------===//
  // Collect all CMOV-group-candidates and add them into CmovInstGroups.
  //
  // CMOV-group:
  //   CMOV instructions, in same MBB, that uses same EFLAGS def instruction.
  //
  // CMOV-group-candidate:
  //   CMOV-group where all the CMOV instructions are
  //     1. consecutive.
  //     2. have same condition code or opposite one.
  //     3. have only operand registers (X86::CMOVrr).
  //===--------------------------------------------------------------------===//
  // List of possible improvement (TODO's):
  // --------------------------------------
  //   TODO: Add support for X86::CMOVrm instructions.
  //   TODO: Add support for X86::SETcc instructions.
  //   TODO: Add support for CMOV-groups with non consecutive CMOV instructions.
  //===--------------------------------------------------------------------===//

  // Current processed CMOV-Group.
  CmovGroup Group;
  for (auto *MBB : CurrLoop->getBlocks()) {
    Group.clear();
    // Condition code of first CMOV instruction current processed range and its
    // opposite condition code.
    X86::CondCode FirstCC, FirstOppCC;
    // Indicator of a non CMOVrr instruction in the current processed range.
    bool FoundNonCMOVInst = false;
    // Indicator for current processed CMOV-group if it should be skipped.
    bool SkipGroup = false;

    for (auto &I : *MBB) {
      X86::CondCode CC = X86::getCondFromCMovOpc(I.getOpcode());
      // Check if we found a X86::CMOVrr instruction.
      if (CC != X86::COND_INVALID && !I.mayLoad()) {
        if (Group.empty()) {
          // We found first CMOV in the range, reset flags.
          FirstCC = CC;
          FirstOppCC = X86::GetOppositeBranchCondition(CC);
          FoundNonCMOVInst = false;
          SkipGroup = false;
        }
        Group.push_back(&I);
        // Check if it is a non-consecutive CMOV instruction or it has different
        // condition code than FirstCC or FirstOppCC.
        if (FoundNonCMOVInst || (CC != FirstCC && CC != FirstOppCC))
          // Mark the SKipGroup indicator to skip current processed CMOV-Group.
          SkipGroup = true;
        continue;
      }
      // If Group is empty, keep looking for first CMOV in the range.
      if (Group.empty())
        continue;

      // We found a non X86::CMOVrr instruction.
      FoundNonCMOVInst = true;
      // Check if this instruction define EFLAGS, to determine end of processed
      // range, as there would be no more instructions using current EFLAGS def.
      if (I.definesRegister(X86::EFLAGS)) {
        // Check if current processed CMOV-group should not be skipped and add
        // it as a CMOV-group-candidate.
        if (!SkipGroup)
          CmovInstGroups.push_back(Group);
        else
          ++NumOfSkippedCmovGroups;
        Group.clear();
      }
    }
    // End of basic block is considered end of range, check if current processed
    // CMOV-group should not be skipped and add it as a CMOV-group-candidate.
    if (Group.empty())
      continue;
    if (!SkipGroup)
      CmovInstGroups.push_back(Group);
    else
      ++NumOfSkippedCmovGroups;
  }

  NumOfCmovGroupCandidate += CmovInstGroups.size();
  return !CmovInstGroups.empty();
}

/// \returns Depth of CMOV instruction as if it was converted into branch.
/// \param TrueOpDepth depth cost of CMOV true value operand.
/// \param FalseOpDepth depth cost of CMOV false value operand.
static unsigned getDepthOfOptCmov(unsigned TrueOpDepth, unsigned FalseOpDepth) {
  //===--------------------------------------------------------------------===//
  // With no info about branch weight, we assume 50% for each value operand.
  // Thus, depth of optimized CMOV instruction is the rounded up average of
  // its True-Operand-Value-Depth and False-Operand-Value-Depth.
  //===--------------------------------------------------------------------===//
  return (TrueOpDepth + FalseOpDepth + 1) / 2;
}

bool X86CmovConverterPass::checkForProfitableCmovCandidates(
    MachineLoop *CurrLoop, CmovGroups &CmovInstGroups) {
  struct DepthInfo {
    /// Depth of original loop.
    unsigned Depth;
    /// Depth of optimized loop.
    unsigned OptDepth;
  };
  /// Number of loop iterations to calculate depth for ?!
  static const unsigned LoopIterations = 2;
  DenseMap<MachineInstr *, DepthInfo> DepthMap;
  DepthInfo LoopDepth[LoopIterations] = {{0, 0}, {0, 0}};
  enum { PhyRegType = 0, VirRegType = 1, RegTypeNum = 2 };
  /// For each register type maps the register to its last def instruction.
  DenseMap<unsigned, MachineInstr *> RegDefMaps[RegTypeNum];
  /// Maps register operand to its def instruction, which can be nullptr if it
  /// is unknown (e.g., operand is defined outside the loop).
  DenseMap<MachineOperand *, MachineInstr *> OperandToDefMap;

  // Set depth of unknown instruction (i.e., nullptr) to zero.
  DepthMap[nullptr] = {0, 0};

  SmallPtrSet<MachineInstr *, 4> CmovInstructions;
  for (auto &Group : CmovInstGroups)
    CmovInstructions.insert(Group.begin(), Group.end());

  //===--------------------------------------------------------------------===//
  // Step 1: Calculate instruction depth and loop depth.
  // Optimized-Loop:
  //   loop with CMOV-group-candidates converted into branches.
  //
  // Instruction-Depth:
  //   instruction latency + max operand depth.
  //     * For CMOV instruction in optimized loop the depth is calculated as:
  //       CMOV latency + getDepthOfOptCmov(True-Op-Depth, False-Op-depth)
  // TODO: Find a better way to estimate the latency of the branch instruction
  //       rather than using the CMOV latency.
  //
  // Loop-Depth:
  //   max instruction depth of all instructions in the loop.
  // Note: instruction with max depth represents the critical-path in the loop.
  //
  // Loop-Depth[i]:
  //   Loop-Depth calculated for first `i` iterations.
  //   Note: it is enough to calculate depth for up to two iterations.
  //
  // Depth-Diff[i]:
  //   Number of cycles saved in first 'i` iterations by optimizing the loop.
  //===--------------------------------------------------------------------===//
  for (unsigned I = 0; I < LoopIterations; ++I) {
    DepthInfo &MaxDepth = LoopDepth[I];
    for (auto *MBB : CurrLoop->getBlocks()) {
      // Clear physical registers Def map.
      RegDefMaps[PhyRegType].clear();
      for (MachineInstr &MI : *MBB) {
        unsigned MIDepth = 0;
        unsigned MIDepthOpt = 0;
        bool IsCMOV = CmovInstructions.count(&MI);
        for (auto &MO : MI.uses()) {
          // Checks for "isUse()" as "uses()" returns also implicit definitions.
          if (!MO.isReg() || !MO.isUse())
            continue;
          unsigned Reg = MO.getReg();
          auto &RDM = RegDefMaps[TargetRegisterInfo::isVirtualRegister(Reg)];
          if (MachineInstr *DefMI = RDM.lookup(Reg)) {
            OperandToDefMap[&MO] = DefMI;
            DepthInfo Info = DepthMap.lookup(DefMI);
            MIDepth = std::max(MIDepth, Info.Depth);
            if (!IsCMOV)
              MIDepthOpt = std::max(MIDepthOpt, Info.OptDepth);
          }
        }

        if (IsCMOV)
          MIDepthOpt = getDepthOfOptCmov(
              DepthMap[OperandToDefMap.lookup(&MI.getOperand(1))].OptDepth,
              DepthMap[OperandToDefMap.lookup(&MI.getOperand(2))].OptDepth);

        // Iterates over all operands to handle implicit definitions as well.
        for (auto &MO : MI.operands()) {
          if (!MO.isReg() || !MO.isDef())
            continue;
          unsigned Reg = MO.getReg();
          RegDefMaps[TargetRegisterInfo::isVirtualRegister(Reg)][Reg] = &MI;
        }

        unsigned Latency = TSchedModel.computeInstrLatency(&MI);
        DepthMap[&MI] = {MIDepth += Latency, MIDepthOpt += Latency};
        MaxDepth.Depth = std::max(MaxDepth.Depth, MIDepth);
        MaxDepth.OptDepth = std::max(MaxDepth.OptDepth, MIDepthOpt);
      }
    }
  }

  unsigned Diff[LoopIterations] = {LoopDepth[0].Depth - LoopDepth[0].OptDepth,
                                   LoopDepth[1].Depth - LoopDepth[1].OptDepth};

  //===--------------------------------------------------------------------===//
  // Step 2: Check if Loop worth to be optimized.
  // Worth-Optimize-Loop:
  //   case 1: Diff[1] == Diff[0]
  //           Critical-path is iteration independent - there is no dependency
  //           of critical-path instructions on critical-path instructions of
  //           previous iteration.
  //           Thus, it is enough to check gain percent of 1st iteration -
  //           To be conservative, the optimized loop need to have a depth of
  //           12.5% cycles less than original loop, per iteration.
  //
  //   case 2: Diff[1] > Diff[0]
  //           Critical-path is iteration dependent - there is dependency of
  //           critical-path instructions on critical-path instructions of
  //           previous iteration.
  //           Thus, it is required to check the gradient of the gain - the
  //           change in Depth-Diff compared to the change in Loop-Depth between
  //           1st and 2nd iterations.
  //           To be conservative, the gradient need to be at least 50%.
  //
  // If loop is not worth optimizing, remove all CMOV-group-candidates.
  //===--------------------------------------------------------------------===//
  bool WorthOptLoop = false;
  if (Diff[1] == Diff[0])
    WorthOptLoop = Diff[0] * 8 >= LoopDepth[0].Depth;
  else if (Diff[1] > Diff[0])
    WorthOptLoop =
        (Diff[1] - Diff[0]) * 2 >= (LoopDepth[1].Depth - LoopDepth[0].Depth);

  if (!WorthOptLoop)
    return false;

  ++NumOfLoopCandidate;

  //===--------------------------------------------------------------------===//
  // Step 3: Check for each CMOV-group-candidate if it worth to be optimized.
  // Worth-Optimize-Group:
  //   Iff it worths to optimize all CMOV instructions in the group.
  //
  // Worth-Optimize-CMOV:
  //   Predicted branch is faster than CMOV by the difference between depth of
  //   condition operand and depth of taken (predicted) value operand.
  //   To be conservative, the gain of such CMOV transformation should cover at
  //   at least 25% of branch-misprediction-penalty.
  //===--------------------------------------------------------------------===//
  unsigned MispredictPenalty = TSchedModel.getMCSchedModel()->MispredictPenalty;
  CmovGroups TempGroups;
  std::swap(TempGroups, CmovInstGroups);
  for (auto &Group : TempGroups) {
    bool WorthOpGroup = true;
    for (auto *MI : Group) {
      // Avoid CMOV instruction which value is used as a pointer to load from.
      // This is another conservative check to avoid converting CMOV instruction
      // used with tree-search like algorithm, where the branch is unpredicted.
      auto UIs = MRI->use_instructions(MI->defs().begin()->getReg());
      if (UIs.begin() != UIs.end() && ++UIs.begin() == UIs.end()) {
        unsigned Op = UIs.begin()->getOpcode();
        if (Op == X86::MOV64rm || Op == X86::MOV32rm) {
          WorthOpGroup = false;
          break;
        }
      }

      unsigned CondCost =
          DepthMap[OperandToDefMap.lookup(&MI->getOperand(3))].Depth;
      unsigned ValCost = getDepthOfOptCmov(
          DepthMap[OperandToDefMap.lookup(&MI->getOperand(1))].Depth,
          DepthMap[OperandToDefMap.lookup(&MI->getOperand(2))].Depth);
      if (ValCost > CondCost || (CondCost - ValCost) * 4 < MispredictPenalty) {
        WorthOpGroup = false;
        break;
      }
    }

    if (WorthOpGroup)
      CmovInstGroups.push_back(Group);
  }

  return !CmovInstGroups.empty();
}

static bool checkEFLAGSLive(MachineInstr *MI) {
  if (MI->killsRegister(X86::EFLAGS))
    return false;

  // The EFLAGS operand of MI might be missing a kill marker.
  // Figure out whether EFLAGS operand should LIVE after MI instruction.
  MachineBasicBlock *BB = MI->getParent();
  MachineBasicBlock::iterator ItrMI = MI;

  // Scan forward through BB for a use/def of EFLAGS.
  for (auto I = std::next(ItrMI), E = BB->end(); I != E; ++I) {
    if (I->readsRegister(X86::EFLAGS))
      return true;
    if (I->definesRegister(X86::EFLAGS))
      return false;
  }

  // We hit the end of the block, check whether EFLAGS is live into a successor.
  for (auto I = BB->succ_begin(), E = BB->succ_end(); I != E; ++I) {
    if ((*I)->isLiveIn(X86::EFLAGS))
      return true;
  }

  return false;
}

void X86CmovConverterPass::convertCmovInstsToBranches(
    SmallVectorImpl<MachineInstr *> &Group) const {
  assert(!Group.empty() && "No CMOV instructions to convert");
  ++NumOfOptimizedCmovGroups;

  // To convert a CMOVcc instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.

  // Before
  // -----
  // MBB:
  //   cond = cmp ...
  //   v1 = CMOVge t1, f1, cond
  //   v2 = CMOVlt t2, f2, cond
  //   v3 = CMOVge v1, f3, cond
  //
  // After
  // -----
  // MBB:
  //   cond = cmp ...
  //   jge %SinkMBB
  //
  // FalseMBB:
  //   jmp %SinkMBB
  //
  // SinkMBB:
  //   %v1 = phi[%f1, %FalseMBB], [%t1, %MBB]
  //   %v2 = phi[%t2, %FalseMBB], [%f2, %MBB] ; For CMOV with OppCC switch
  //                                          ; true-value with false-value
  //   %v3 = phi[%f3, %FalseMBB], [%t1, %MBB] ; Phi instruction cannot use
  //                                          ; previous Phi instruction result

  MachineInstr &MI = *Group.front();
  MachineInstr *LastCMOV = Group.back();
  DebugLoc DL = MI.getDebugLoc();
  X86::CondCode CC = X86::CondCode(X86::getCondFromCMovOpc(MI.getOpcode()));
  X86::CondCode OppCC = X86::GetOppositeBranchCondition(CC);
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction::iterator It = ++MBB->getIterator();
  MachineFunction *F = MBB->getParent();
  const BasicBlock *BB = MBB->getBasicBlock();

  MachineBasicBlock *FalseMBB = F->CreateMachineBasicBlock(BB);
  MachineBasicBlock *SinkMBB = F->CreateMachineBasicBlock(BB);
  F->insert(It, FalseMBB);
  F->insert(It, SinkMBB);

  // If the EFLAGS register isn't dead in the terminator, then claim that it's
  // live into the sink and copy blocks.
  if (checkEFLAGSLive(LastCMOV)) {
    FalseMBB->addLiveIn(X86::EFLAGS);
    SinkMBB->addLiveIn(X86::EFLAGS);
  }

  // Transfer the remainder of BB and its successor edges to SinkMBB.
  SinkMBB->splice(SinkMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(LastCMOV)), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // Add the false and sink blocks as its successors.
  MBB->addSuccessor(FalseMBB);
  MBB->addSuccessor(SinkMBB);

  // Create the conditional branch instruction.
  BuildMI(MBB, DL, TII->get(X86::GetCondBranchFromCond(CC))).addMBB(SinkMBB);

  // Add the sink block to the false block successors.
  FalseMBB->addSuccessor(SinkMBB);

  MachineInstrBuilder MIB;
  MachineBasicBlock::iterator MIItBegin = MachineBasicBlock::iterator(MI);
  MachineBasicBlock::iterator MIItEnd =
      std::next(MachineBasicBlock::iterator(LastCMOV));
  MachineBasicBlock::iterator SinkInsertionPoint = SinkMBB->begin();
  // As we are creating the PHIs, we have to be careful if there is more than
  // one.  Later CMOVs may reference the results of earlier CMOVs, but later
  // PHIs have to reference the individual true/false inputs from earlier PHIs.
  // That also means that PHI construction must work forward from earlier to
  // later, and that the code must maintain a mapping from earlier PHI's
  // destination registers, and the registers that went into the PHI.
  DenseMap<unsigned, std::pair<unsigned, unsigned>> RegRewriteTable;

  for (MachineBasicBlock::iterator MIIt = MIItBegin; MIIt != MIItEnd; ++MIIt) {
    unsigned DestReg = MIIt->getOperand(0).getReg();
    unsigned Op1Reg = MIIt->getOperand(1).getReg();
    unsigned Op2Reg = MIIt->getOperand(2).getReg();

    // If this CMOV we are processing is the opposite condition from the jump we
    // generated, then we have to swap the operands for the PHI that is going to
    // be generated.
    if (X86::getCondFromCMovOpc(MIIt->getOpcode()) == OppCC)
      std::swap(Op1Reg, Op2Reg);

    auto Op1Itr = RegRewriteTable.find(Op1Reg);
    if (Op1Itr != RegRewriteTable.end())
      Op1Reg = Op1Itr->second.first;

    auto Op2Itr = RegRewriteTable.find(Op2Reg);
    if (Op2Itr != RegRewriteTable.end())
      Op2Reg = Op2Itr->second.second;

    //  SinkMBB:
    //   %Result = phi [ %FalseValue, FalseMBB ], [ %TrueValue, MBB ]
    //  ...
    MIB = BuildMI(*SinkMBB, SinkInsertionPoint, DL, TII->get(X86::PHI), DestReg)
              .addReg(Op1Reg)
              .addMBB(FalseMBB)
              .addReg(Op2Reg)
              .addMBB(MBB);
    (void)MIB;
    DEBUG(dbgs() << "\tFrom: "; MIIt->dump());
    DEBUG(dbgs() << "\tTo: "; MIB->dump());

    // Add this PHI to the rewrite table.
    RegRewriteTable[DestReg] = std::make_pair(Op1Reg, Op2Reg);
  }

  // Now remove the CMOV(s).
  MBB->erase(MIItBegin, MIItEnd);
}

} // End anonymous namespace.

FunctionPass *llvm::createX86CmovConverterPass() {
  return new X86CmovConverterPass();
}
