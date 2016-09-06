//===---------- AArch64CollectLOH.cpp - AArch64 collect LOH pass --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that collect the Linker Optimization Hint (LOH).
// This pass should be run at the very end of the compilation flow, just before
// assembly printer.
// To be useful for the linker, the LOH must be printed into the assembly file.
//
// A LOH describes a sequence of instructions that may be optimized by the
// linker.
// This same sequence cannot be optimized by the compiler because some of
// the information will be known at link time.
// For instance, consider the following sequence:
//     L1: adrp xA, sym@PAGE
//     L2: add xB, xA, sym@PAGEOFF
//     L3: ldr xC, [xB, #imm]
// This sequence can be turned into:
// A literal load if sym@PAGE + sym@PAGEOFF + #imm - address(L3) is < 1MB:
//     L3: ldr xC, sym+#imm
// It may also be turned into either the following more efficient
// code sequences:
// - If sym@PAGEOFF + #imm fits the encoding space of L3.
//     L1: adrp xA, sym@PAGE
//     L3: ldr xC, [xB, sym@PAGEOFF + #imm]
// - If sym@PAGE + sym@PAGEOFF - address(L1) < 1MB:
//     L1: adr xA, sym
//     L3: ldr xC, [xB, #imm]
//
// To be valid a LOH must meet all the requirements needed by all the related
// possible linker transformations.
// For instance, using the running example, the constraints to emit
// ".loh AdrpAddLdr" are:
// - L1, L2, and L3 instructions are of the expected type, i.e.,
//   respectively ADRP, ADD (immediate), and LD.
// - The result of L1 is used only by L2.
// - The register argument (xA) used in the ADD instruction is defined
//   only by L1.
// - The result of L2 is used only by L3.
// - The base address (xB) in L3 is defined only L2.
// - The ADRP in L1 and the ADD in L2 must reference the same symbol using
//   @PAGE/@PAGEOFF with no additional constants
//
// Currently supported LOHs are:
// * So called non-ADRP-related:
//   - .loh AdrpAddLdr L1, L2, L3:
//     L1: adrp xA, sym@PAGE
//     L2: add xB, xA, sym@PAGEOFF
//     L3: ldr xC, [xB, #imm]
//   - .loh AdrpLdrGotLdr L1, L2, L3:
//     L1: adrp xA, sym@GOTPAGE
//     L2: ldr xB, [xA, sym@GOTPAGEOFF]
//     L3: ldr xC, [xB, #imm]
//   - .loh AdrpLdr L1, L3:
//     L1: adrp xA, sym@PAGE
//     L3: ldr xC, [xA, sym@PAGEOFF]
//   - .loh AdrpAddStr L1, L2, L3:
//     L1: adrp xA, sym@PAGE
//     L2: add xB, xA, sym@PAGEOFF
//     L3: str xC, [xB, #imm]
//   - .loh AdrpLdrGotStr L1, L2, L3:
//     L1: adrp xA, sym@GOTPAGE
//     L2: ldr xB, [xA, sym@GOTPAGEOFF]
//     L3: str xC, [xB, #imm]
//   - .loh AdrpAdd L1, L2:
//     L1: adrp xA, sym@PAGE
//     L2: add xB, xA, sym@PAGEOFF
//   For all these LOHs, L1, L2, L3 form a simple chain:
//   L1 result is used only by L2 and L2 result by L3.
//   L3 LOH-related argument is defined only by L2 and L2 LOH-related argument
//   by L1.
// All these LOHs aim at using more efficient load/store patterns by folding
// some instructions used to compute the address directly into the load/store.
//
// * So called ADRP-related:
//  - .loh AdrpAdrp L2, L1:
//    L2: ADRP xA, sym1@PAGE
//    L1: ADRP xA, sym2@PAGE
//    L2 dominates L1 and xA is not redifined between L2 and L1
// This LOH aims at getting rid of redundant ADRP instructions.
//
// The overall design for emitting the LOHs is:
// 1. AArch64CollectLOH (this pass) records the LOHs in the AArch64FunctionInfo.
// 2. AArch64AsmPrinter reads the LOHs from AArch64FunctionInfo and it:
//     1. Associates them a label.
//     2. Emits them in a MCStreamer (EmitLOHDirective).
//         - The MCMachOStreamer records them into the MCAssembler.
//         - The MCAsmStreamer prints them.
//         - Other MCStreamers ignore them.
//     3. Closes the MCStreamer:
//         - The MachObjectWriter gets them from the MCAssembler and writes
//           them in the object file.
//         - Other ObjectWriters ignore them.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64-collect-loh"

static cl::opt<bool>
PreCollectRegister("aarch64-collect-loh-pre-collect-register", cl::Hidden,
                   cl::desc("Restrict analysis to registers invovled"
                            " in LOHs"),
                   cl::init(true));

static cl::opt<bool>
BasicBlockScopeOnly("aarch64-collect-loh-bb-only", cl::Hidden,
                    cl::desc("Restrict analysis at basic block scope"),
                    cl::init(true));

STATISTIC(NumADRPSimpleCandidate,
          "Number of simplifiable ADRP dominate by another");
STATISTIC(NumADRPComplexCandidate2,
          "Number of simplifiable ADRP reachable by 2 defs");
STATISTIC(NumADRPComplexCandidate3,
          "Number of simplifiable ADRP reachable by 3 defs");
STATISTIC(NumADRPComplexCandidateOther,
          "Number of simplifiable ADRP reachable by 4 or more defs");
STATISTIC(NumADDToSTRWithImm,
          "Number of simplifiable STR with imm reachable by ADD");
STATISTIC(NumLDRToSTRWithImm,
          "Number of simplifiable STR with imm reachable by LDR");
STATISTIC(NumADDToSTR, "Number of simplifiable STR reachable by ADD");
STATISTIC(NumLDRToSTR, "Number of simplifiable STR reachable by LDR");
STATISTIC(NumADDToLDRWithImm,
          "Number of simplifiable LDR with imm reachable by ADD");
STATISTIC(NumLDRToLDRWithImm,
          "Number of simplifiable LDR with imm reachable by LDR");
STATISTIC(NumADDToLDR, "Number of simplifiable LDR reachable by ADD");
STATISTIC(NumLDRToLDR, "Number of simplifiable LDR reachable by LDR");
STATISTIC(NumADRPToLDR, "Number of simplifiable LDR reachable by ADRP");
STATISTIC(NumCplxLvl1, "Number of complex case of level 1");
STATISTIC(NumTooCplxLvl1, "Number of too complex case of level 1");
STATISTIC(NumCplxLvl2, "Number of complex case of level 2");
STATISTIC(NumTooCplxLvl2, "Number of too complex case of level 2");
STATISTIC(NumADRSimpleCandidate, "Number of simplifiable ADRP + ADD");
STATISTIC(NumADRComplexCandidate, "Number of too complex ADRP + ADD");

#define AARCH64_COLLECT_LOH_NAME "AArch64 Collect Linker Optimization Hint (LOH)"

namespace {
struct AArch64CollectLOH : public MachineFunctionPass {
  static char ID;
  AArch64CollectLOH() : MachineFunctionPass(ID) {
    initializeAArch64CollectLOHPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::AllVRegsAllocated);
  }

  const char *getPassName() const override {
    return AARCH64_COLLECT_LOH_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTree>();
  }

private:
};

/// A set of MachineInstruction.
typedef SetVector<const MachineInstr *> SetOfMachineInstr;
/// Map a basic block to a set of instructions per register.
/// This is used to represent the exposed uses of a basic block
/// per register.
typedef MapVector<const MachineBasicBlock *,
                  std::unique_ptr<SetOfMachineInstr[]>>
BlockToSetOfInstrsPerColor;
/// Map a basic block to an instruction per register.
/// This is used to represent the live-out definitions of a basic block
/// per register.
typedef MapVector<const MachineBasicBlock *,
                  std::unique_ptr<const MachineInstr *[]>>
BlockToInstrPerColor;
/// Map an instruction to a set of instructions. Used to represent the
/// mapping def to reachable uses or use to definitions.
typedef MapVector<const MachineInstr *, SetOfMachineInstr> InstrToInstrs;
/// Map a basic block to a BitVector.
/// This is used to record the kill registers per basic block.
typedef MapVector<const MachineBasicBlock *, BitVector> BlockToRegSet;

/// Map a register to a dense id.
typedef DenseMap<unsigned, unsigned> MapRegToId;
/// Map a dense id to a register. Used for debug purposes.
typedef SmallVector<unsigned, 32> MapIdToReg;
} // end anonymous namespace.

char AArch64CollectLOH::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64CollectLOH, "aarch64-collect-loh",
                      AARCH64_COLLECT_LOH_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(AArch64CollectLOH, "aarch64-collect-loh",
                    AARCH64_COLLECT_LOH_NAME, false, false)

/// Given a couple (MBB, reg) get the corresponding set of instruction from
/// the given "sets".
/// If this couple does not reference any set, an empty set is added to "sets"
/// for this couple and returned.
/// \param nbRegs is used internally allocate some memory. It must be consistent
/// with the way sets is used.
static SetOfMachineInstr &getSet(BlockToSetOfInstrsPerColor &sets,
                                 const MachineBasicBlock &MBB, unsigned reg,
                                 unsigned nbRegs) {
  SetOfMachineInstr *result;
  BlockToSetOfInstrsPerColor::iterator it = sets.find(&MBB);
  if (it != sets.end())
    result = it->second.get();
  else
    result = (sets[&MBB] = make_unique<SetOfMachineInstr[]>(nbRegs)).get();

  return result[reg];
}

/// Given a couple (reg, MI) get the corresponding set of instructions from the
/// the given "sets".
/// This is used to get the uses record in sets of a definition identified by
/// MI and reg, i.e., MI defines reg.
/// If the couple does not reference anything, an empty set is added to
/// "sets[reg]".
/// \pre set[reg] is valid.
static SetOfMachineInstr &getUses(InstrToInstrs *sets, unsigned reg,
                                  const MachineInstr &MI) {
  return sets[reg][&MI];
}

/// Same as getUses but does not modify the input map: sets.
/// \return NULL if the couple (reg, MI) is not in sets.
static const SetOfMachineInstr *getUses(const InstrToInstrs *sets, unsigned reg,
                                        const MachineInstr &MI) {
  InstrToInstrs::const_iterator Res = sets[reg].find(&MI);
  if (Res != sets[reg].end())
    return &(Res->second);
  return nullptr;
}

/// Initialize the reaching definition algorithm:
/// For each basic block BB in MF, record:
/// - its kill set.
/// - its reachable uses (uses that are exposed to BB's predecessors).
/// - its the generated definitions.
/// \param DummyOp if not NULL, specifies a Dummy Operation to be added to
/// the list of uses of exposed defintions.
/// \param ADRPMode specifies to only consider ADRP instructions for generated
/// definition. It also consider definitions of ADRP instructions as uses and
/// ignore other uses. The ADRPMode is used to collect the information for LHO
/// that involve ADRP operation only.
static void initReachingDef(const MachineFunction &MF,
                            InstrToInstrs *ColorOpToReachedUses,
                            BlockToInstrPerColor &Gen, BlockToRegSet &Kill,
                            BlockToSetOfInstrsPerColor &ReachableUses,
                            const MapRegToId &RegToId,
                            const MachineInstr *DummyOp, bool ADRPMode) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned NbReg = RegToId.size();

  for (const MachineBasicBlock &MBB : MF) {
    auto &BBGen = Gen[&MBB];
    BBGen = make_unique<const MachineInstr *[]>(NbReg);
    std::fill(BBGen.get(), BBGen.get() + NbReg, nullptr);

    BitVector &BBKillSet = Kill[&MBB];
    BBKillSet.resize(NbReg);
    for (const MachineInstr &MI : MBB) {
      bool IsADRP = MI.getOpcode() == AArch64::ADRP;

      // Process uses first.
      if (IsADRP || !ADRPMode)
        for (const MachineOperand &MO : MI.operands()) {
          // Treat ADRP def as use, as the goal of the analysis is to find
          // ADRP defs reached by other ADRP defs.
          if (!MO.isReg() || (!ADRPMode && !MO.isUse()) ||
              (ADRPMode && (!IsADRP || !MO.isDef())))
            continue;
          unsigned CurReg = MO.getReg();
          MapRegToId::const_iterator ItCurRegId = RegToId.find(CurReg);
          if (ItCurRegId == RegToId.end())
            continue;
          CurReg = ItCurRegId->second;

          // if CurReg has not been defined, this use is reachable.
          if (!BBGen[CurReg] && !BBKillSet.test(CurReg))
            getSet(ReachableUses, MBB, CurReg, NbReg).insert(&MI);
          // current basic block definition for this color, if any, is in Gen.
          if (BBGen[CurReg])
            getUses(ColorOpToReachedUses, CurReg, *BBGen[CurReg]).insert(&MI);
        }

      // Process clobbers.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isRegMask())
          continue;
        // Clobbers kill the related colors.
        const uint32_t *PreservedRegs = MO.getRegMask();

        // Set generated regs.
        for (const auto &Entry : RegToId) {
          unsigned Reg = Entry.second;
          // Use the global register ID when querying APIs external to this
          // pass.
          if (MachineOperand::clobbersPhysReg(PreservedRegs, Entry.first)) {
            // Do not register clobbered definition for no ADRP.
            // This definition is not used anyway (otherwise register
            // allocation is wrong).
            BBGen[Reg] = ADRPMode ? &MI : nullptr;
            BBKillSet.set(Reg);
          }
        }
      }

      // Process register defs.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        unsigned CurReg = MO.getReg();
        MapRegToId::const_iterator ItCurRegId = RegToId.find(CurReg);
        if (ItCurRegId == RegToId.end())
          continue;

        for (MCRegAliasIterator AI(CurReg, TRI, true); AI.isValid(); ++AI) {
          MapRegToId::const_iterator ItRegId = RegToId.find(*AI);
          // If this alias has not been recorded, then it is not interesting
          // for the current analysis.
          // We can end up in this situation because of tuple registers.
          // E.g., Let say we are interested in S1. When we register
          // S1, we will also register its aliases and in particular
          // the tuple Q1_Q2.
          // Now, when we encounter Q1_Q2, we will look through its aliases
          // and will find that S2 is not registered.
          if (ItRegId == RegToId.end())
            continue;

          BBKillSet.set(ItRegId->second);
          BBGen[ItRegId->second] = &MI;
        }
        BBGen[ItCurRegId->second] = &MI;
      }
    }

    // If we restrict our analysis to basic block scope, conservatively add a
    // dummy
    // use for each generated value.
    if (!ADRPMode && DummyOp && !MBB.succ_empty())
      for (unsigned CurReg = 0; CurReg < NbReg; ++CurReg)
        if (BBGen[CurReg])
          getUses(ColorOpToReachedUses, CurReg, *BBGen[CurReg]).insert(DummyOp);
  }
}

/// Reaching def core algorithm:
/// while an Out has changed
///    for each bb
///       for each color
///           In[bb][color] = U Out[bb.predecessors][color]
///           insert reachableUses[bb][color] in each in[bb][color]
///                 op.reachedUses
///
///           Out[bb] = Gen[bb] U (In[bb] - Kill[bb])
static void reachingDefAlgorithm(const MachineFunction &MF,
                                 InstrToInstrs *ColorOpToReachedUses,
                                 BlockToSetOfInstrsPerColor &In,
                                 BlockToSetOfInstrsPerColor &Out,
                                 BlockToInstrPerColor &Gen, BlockToRegSet &Kill,
                                 BlockToSetOfInstrsPerColor &ReachableUses,
                                 unsigned NbReg) {
  bool HasChanged;
  do {
    HasChanged = false;
    for (const MachineBasicBlock &MBB : MF) {
      unsigned CurReg;
      for (CurReg = 0; CurReg < NbReg; ++CurReg) {
        SetOfMachineInstr &BBInSet = getSet(In, MBB, CurReg, NbReg);
        SetOfMachineInstr &BBReachableUses =
            getSet(ReachableUses, MBB, CurReg, NbReg);
        SetOfMachineInstr &BBOutSet = getSet(Out, MBB, CurReg, NbReg);
        unsigned Size = BBOutSet.size();
        //   In[bb][color] = U Out[bb.predecessors][color]
        for (const MachineBasicBlock *PredMBB : MBB.predecessors()) {
          SetOfMachineInstr &PredOutSet = getSet(Out, *PredMBB, CurReg, NbReg);
          BBInSet.insert(PredOutSet.begin(), PredOutSet.end());
        }
        //   insert reachableUses[bb][color] in each in[bb][color] op.reachedses
        for (const MachineInstr *MI : BBInSet) {
          SetOfMachineInstr &OpReachedUses =
              getUses(ColorOpToReachedUses, CurReg, *MI);
          OpReachedUses.insert(BBReachableUses.begin(), BBReachableUses.end());
        }
        //           Out[bb] = Gen[bb] U (In[bb] - Kill[bb])
        if (!Kill[&MBB].test(CurReg))
          BBOutSet.insert(BBInSet.begin(), BBInSet.end());
        if (Gen[&MBB][CurReg])
          BBOutSet.insert(Gen[&MBB][CurReg]);
        HasChanged |= BBOutSet.size() != Size;
      }
    }
  } while (HasChanged);
}

/// Reaching definition algorithm.
/// \param MF function on which the algorithm will operate.
/// \param[out] ColorOpToReachedUses will contain the result of the reaching
/// def algorithm.
/// \param ADRPMode specify whether the reaching def algorithm should be tuned
/// for ADRP optimization. \see initReachingDef for more details.
/// \param DummyOp if not NULL, the algorithm will work at
/// basic block scope and will set for every exposed definition a use to
/// @p DummyOp.
/// \pre ColorOpToReachedUses is an array of at least number of registers of
/// InstrToInstrs.
static void reachingDef(const MachineFunction &MF,
                        InstrToInstrs *ColorOpToReachedUses,
                        const MapRegToId &RegToId, bool ADRPMode = false,
                        const MachineInstr *DummyOp = nullptr) {
  // structures:
  // For each basic block.
  // Out: a set per color of definitions that reach the
  //      out boundary of this block.
  // In: Same as Out but for in boundary.
  // Gen: generated color in this block (one operation per color).
  // Kill: register set of killed color in this block.
  // ReachableUses: a set per color of uses (operation) reachable
  //                for "In" definitions.
  BlockToSetOfInstrsPerColor Out, In, ReachableUses;
  BlockToInstrPerColor Gen;
  BlockToRegSet Kill;

  // Initialize Gen, kill and reachableUses.
  initReachingDef(MF, ColorOpToReachedUses, Gen, Kill, ReachableUses, RegToId,
                  DummyOp, ADRPMode);

  // Algo.
  if (!DummyOp)
    reachingDefAlgorithm(MF, ColorOpToReachedUses, In, Out, Gen, Kill,
                         ReachableUses, RegToId.size());
}

#ifndef NDEBUG
/// print the result of the reaching definition algorithm.
static void printReachingDef(const InstrToInstrs *ColorOpToReachedUses,
                             unsigned NbReg, const TargetRegisterInfo *TRI,
                             const MapIdToReg &IdToReg) {
  unsigned CurReg;
  for (CurReg = 0; CurReg < NbReg; ++CurReg) {
    if (ColorOpToReachedUses[CurReg].empty())
      continue;
    DEBUG(dbgs() << "*** Reg " << PrintReg(IdToReg[CurReg], TRI) << " ***\n");

    for (const auto &DefsIt : ColorOpToReachedUses[CurReg]) {
      DEBUG(dbgs() << "Def:\n");
      DEBUG(DefsIt.first->print(dbgs()));
      DEBUG(dbgs() << "Reachable uses:\n");
      for (const MachineInstr *MI : DefsIt.second) {
        DEBUG(MI->print(dbgs()));
      }
    }
  }
}
#endif // NDEBUG

/// Answer the following question: Can Def be one of the definition
/// involved in a part of a LOH?
static bool canDefBePartOfLOH(const MachineInstr *Def) {
  unsigned Opc = Def->getOpcode();
  // Accept ADRP, ADDLow and LOADGot.
  switch (Opc) {
  default:
    return false;
  case AArch64::ADRP:
    return true;
  case AArch64::ADDXri:
    // Check immediate to see if the immediate is an address.
    switch (Def->getOperand(2).getType()) {
    default:
      return false;
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_JumpTableIndex:
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_BlockAddress:
      return true;
    }
  case AArch64::LDRXui:
    // Check immediate to see if the immediate is an address.
    switch (Def->getOperand(2).getType()) {
    default:
      return false;
    case MachineOperand::MO_GlobalAddress:
      return true;
    }
  }
  // Unreachable.
  return false;
}

/// Check whether the given instruction can the end of a LOH chain involving a
/// store.
static bool isCandidateStore(const MachineInstr *Instr) {
  switch (Instr->getOpcode()) {
  default:
    return false;
  case AArch64::STRBBui:
  case AArch64::STRHHui:
  case AArch64::STRBui:
  case AArch64::STRHui:
  case AArch64::STRWui:
  case AArch64::STRXui:
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:
    // In case we have str xA, [xA, #imm], this is two different uses
    // of xA and we cannot fold, otherwise the xA stored may be wrong,
    // even if #imm == 0.
    if (Instr->getOperand(0).getReg() != Instr->getOperand(1).getReg())
      return true;
  }
  return false;
}

/// Given the result of a reaching definition algorithm in ColorOpToReachedUses,
/// Build the Use to Defs information and filter out obvious non-LOH candidates.
/// In ADRPMode, non-LOH candidates are "uses" with non-ADRP definitions.
/// In non-ADRPMode, non-LOH candidates are "uses" with several definition,
/// i.e., no simple chain.
/// \param ADRPMode -- \see initReachingDef.
static void reachedUsesToDefs(InstrToInstrs &UseToReachingDefs,
                              const InstrToInstrs *ColorOpToReachedUses,
                              const MapRegToId &RegToId,
                              bool ADRPMode = false) {

  SetOfMachineInstr NotCandidate;
  unsigned NbReg = RegToId.size();
  MapRegToId::const_iterator EndIt = RegToId.end();
  for (unsigned CurReg = 0; CurReg < NbReg; ++CurReg) {
    // If this color is never defined, continue.
    if (ColorOpToReachedUses[CurReg].empty())
      continue;

    for (const auto &DefsIt : ColorOpToReachedUses[CurReg]) {
      for (const MachineInstr *MI : DefsIt.second) {
        const MachineInstr *Def = DefsIt.first;
        MapRegToId::const_iterator It;
        // if all the reaching defs are not adrp, this use will not be
        // simplifiable.
        if ((ADRPMode && Def->getOpcode() != AArch64::ADRP) ||
            (!ADRPMode && !canDefBePartOfLOH(Def)) ||
            (!ADRPMode && isCandidateStore(MI) &&
             // store are LOH candidate iff the end of the chain is used as
             // base.
             ((It = RegToId.find((MI)->getOperand(1).getReg())) == EndIt ||
              It->second != CurReg))) {
          NotCandidate.insert(MI);
          continue;
        }
        // Do not consider self reaching as a simplifiable case for ADRP.
        if (!ADRPMode || MI != DefsIt.first) {
          UseToReachingDefs[MI].insert(DefsIt.first);
          // If UsesIt has several reaching definitions, it is not
          // candidate for simplificaton in non-ADRPMode.
          if (!ADRPMode && UseToReachingDefs[MI].size() > 1)
            NotCandidate.insert(MI);
        }
      }
    }
  }
  for (const MachineInstr *Elem : NotCandidate) {
    DEBUG(dbgs() << "Too many reaching defs: " << *Elem << "\n");
    // It would have been better if we could just remove the entry
    // from the map.  Because of that, we have to filter the garbage
    // (second.empty) in the subsequence analysis.
    UseToReachingDefs[Elem].clear();
  }
}

/// Based on the use to defs information (in ADRPMode), compute the
/// opportunities of LOH ADRP-related.
static void computeADRP(const InstrToInstrs &UseToDefs,
                        AArch64FunctionInfo &AArch64FI,
                        const MachineDominatorTree *MDT) {
  DEBUG(dbgs() << "*** Compute LOH for ADRP\n");
  for (const auto &Entry : UseToDefs) {
    unsigned Size = Entry.second.size();
    if (Size == 0)
      continue;
    if (Size == 1) {
      const MachineInstr *L2 = *Entry.second.begin();
      const MachineInstr *L1 = Entry.first;
      if (!MDT->dominates(L2, L1)) {
        DEBUG(dbgs() << "Dominance check failed:\n" << *L2 << '\n' << *L1
                     << '\n');
        continue;
      }
      DEBUG(dbgs() << "Record AdrpAdrp:\n" << *L2 << '\n' << *L1 << '\n');
      AArch64FI.addLOHDirective(MCLOH_AdrpAdrp, {L2, L1});
      ++NumADRPSimpleCandidate;
    }
#ifdef DEBUG
    else if (Size == 2)
      ++NumADRPComplexCandidate2;
    else if (Size == 3)
      ++NumADRPComplexCandidate3;
    else
      ++NumADRPComplexCandidateOther;
#endif
    // if Size < 1, the use should have been removed from the candidates
    assert(Size >= 1 && "No reaching defs for that use!");
  }
}

/// Check whether the given instruction can be the end of a LOH chain
/// involving a load.
static bool isCandidateLoad(const MachineInstr *Instr) {
  switch (Instr->getOpcode()) {
  default:
    return false;
  case AArch64::LDRSBWui:
  case AArch64::LDRSBXui:
  case AArch64::LDRSHWui:
  case AArch64::LDRSHXui:
  case AArch64::LDRSWui:
  case AArch64::LDRBui:
  case AArch64::LDRHui:
  case AArch64::LDRWui:
  case AArch64::LDRXui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
    if (Instr->getOperand(2).getTargetFlags() & AArch64II::MO_GOT)
      return false;
    return true;
  }
  // Unreachable.
  return false;
}

/// Check whether the given instruction can load a litteral.
static bool supportLoadFromLiteral(const MachineInstr *Instr) {
  switch (Instr->getOpcode()) {
  default:
    return false;
  case AArch64::LDRSWui:
  case AArch64::LDRWui:
  case AArch64::LDRXui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
    return true;
  }
  // Unreachable.
  return false;
}

/// Check whether the given instruction is a LOH candidate.
/// \param UseToDefs is used to check that Instr is at the end of LOH supported
/// chain.
/// \pre UseToDefs contains only on def per use, i.e., obvious non candidate are
/// already been filtered out.
static bool isCandidate(const MachineInstr *Instr,
                        const InstrToInstrs &UseToDefs,
                        const MachineDominatorTree *MDT) {
  if (!isCandidateLoad(Instr) && !isCandidateStore(Instr))
    return false;

  const MachineInstr *Def = *UseToDefs.find(Instr)->second.begin();
  if (Def->getOpcode() != AArch64::ADRP) {
    // At this point, Def is ADDXri or LDRXui of the right type of
    // symbol, because we filtered out the uses that were not defined
    // by these kind of instructions (+ ADRP).

    // Check if this forms a simple chain: each intermediate node must
    // dominates the next one.
    if (!MDT->dominates(Def, Instr))
      return false;
    // Move one node up in the simple chain.
    if (UseToDefs.find(Def) ==
            UseToDefs.end()
            // The map may contain garbage we have to ignore.
        ||
        UseToDefs.find(Def)->second.empty())
      return false;
    Instr = Def;
    Def = *UseToDefs.find(Def)->second.begin();
  }
  // Check if we reached the top of the simple chain:
  // - top is ADRP.
  // - check the simple chain property: each intermediate node must
  // dominates the next one.
  if (Def->getOpcode() == AArch64::ADRP)
    return MDT->dominates(Def, Instr);
  return false;
}

static bool registerADRCandidate(const MachineInstr &Use,
                                 const InstrToInstrs &UseToDefs,
                                 const InstrToInstrs *DefsPerColorToUses,
                                 AArch64FunctionInfo &AArch64FI,
                                 SetOfMachineInstr *InvolvedInLOHs,
                                 const MapRegToId &RegToId) {
  // Look for opportunities to turn ADRP -> ADD or
  // ADRP -> LDR GOTPAGEOFF into ADR.
  // If ADRP has more than one use. Give up.
  if (Use.getOpcode() != AArch64::ADDXri &&
      (Use.getOpcode() != AArch64::LDRXui ||
       !(Use.getOperand(2).getTargetFlags() & AArch64II::MO_GOT)))
    return false;
  InstrToInstrs::const_iterator It = UseToDefs.find(&Use);
  // The map may contain garbage that we need to ignore.
  if (It == UseToDefs.end() || It->second.empty())
    return false;
  const MachineInstr &Def = **It->second.begin();
  if (Def.getOpcode() != AArch64::ADRP)
    return false;
  // Check the number of users of ADRP.
  const SetOfMachineInstr *Users =
      getUses(DefsPerColorToUses,
              RegToId.find(Def.getOperand(0).getReg())->second, Def);
  if (Users->size() > 1) {
    ++NumADRComplexCandidate;
    return false;
  }
  ++NumADRSimpleCandidate;
  assert((!InvolvedInLOHs || InvolvedInLOHs->insert(&Def)) &&
         "ADRP already involved in LOH.");
  assert((!InvolvedInLOHs || InvolvedInLOHs->insert(&Use)) &&
         "ADD already involved in LOH.");
  DEBUG(dbgs() << "Record AdrpAdd\n" << Def << '\n' << Use << '\n');

  AArch64FI.addLOHDirective(
      Use.getOpcode() == AArch64::ADDXri ? MCLOH_AdrpAdd : MCLOH_AdrpLdrGot,
      {&Def, &Use});
  return true;
}

/// Based on the use to defs information (in non-ADRPMode), compute the
/// opportunities of LOH non-ADRP-related
static void computeOthers(const InstrToInstrs &UseToDefs,
                          const InstrToInstrs *DefsPerColorToUses,
                          AArch64FunctionInfo &AArch64FI, const MapRegToId &RegToId,
                          const MachineDominatorTree *MDT) {
  SetOfMachineInstr *InvolvedInLOHs = nullptr;
#ifdef DEBUG
  SetOfMachineInstr InvolvedInLOHsStorage;
  InvolvedInLOHs = &InvolvedInLOHsStorage;
#endif // DEBUG
  DEBUG(dbgs() << "*** Compute LOH for Others\n");
  // ADRP -> ADD/LDR -> LDR/STR pattern.
  // Fall back to ADRP -> ADD pattern if we fail to catch the bigger pattern.

  // FIXME: When the statistics are not important,
  // This initial filtering loop can be merged into the next loop.
  // Currently, we didn't do it to have the same code for both DEBUG and
  // NDEBUG builds. Indeed, the iterator of the second loop would need
  // to be changed.
  SetOfMachineInstr PotentialCandidates;
  SetOfMachineInstr PotentialADROpportunities;
  for (auto &Use : UseToDefs) {
    // If no definition is available, this is a non candidate.
    if (Use.second.empty())
      continue;
    // Keep only instructions that are load or store and at the end of
    // a ADRP -> ADD/LDR/Nothing chain.
    // We already filtered out the no-chain cases.
    if (!isCandidate(Use.first, UseToDefs, MDT)) {
      PotentialADROpportunities.insert(Use.first);
      continue;
    }
    PotentialCandidates.insert(Use.first);
  }

  // Make the following distinctions for statistics as the linker does
  // know how to decode instructions:
  // - ADD/LDR/Nothing make there different patterns.
  // - LDR/STR make two different patterns.
  // Hence, 6 - 1 base patterns.
  // (because ADRP-> Nothing -> STR is not simplifiable)

  // The linker is only able to have a simple semantic, i.e., if pattern A
  // do B.
  // However, we want to see the opportunity we may miss if we were able to
  // catch more complex cases.

  // PotentialCandidates are result of a chain ADRP -> ADD/LDR ->
  // A potential candidate becomes a candidate, if its current immediate
  // operand is zero and all nodes of the chain have respectively only one user
#ifdef DEBUG
  SetOfMachineInstr DefsOfPotentialCandidates;
#endif
  for (const MachineInstr *Candidate : PotentialCandidates) {
    // Get the definition of the candidate i.e., ADD or LDR.
    const MachineInstr *Def = *UseToDefs.find(Candidate)->second.begin();
    // Record the elements of the chain.
    const MachineInstr *L1 = Def;
    const MachineInstr *L2 = nullptr;
    unsigned ImmediateDefOpc = Def->getOpcode();
    if (Def->getOpcode() != AArch64::ADRP) {
      // Check the number of users of this node.
      const SetOfMachineInstr *Users =
          getUses(DefsPerColorToUses,
                  RegToId.find(Def->getOperand(0).getReg())->second, *Def);
      if (Users->size() > 1) {
#ifdef DEBUG
        // if all the uses of this def are in potential candidate, this is
        // a complex candidate of level 2.
        bool IsLevel2 = true;
        for (const MachineInstr *MI : *Users) {
          if (!PotentialCandidates.count(MI)) {
            ++NumTooCplxLvl2;
            IsLevel2 = false;
            break;
          }
        }
        if (IsLevel2)
          ++NumCplxLvl2;
#endif // DEBUG
        PotentialADROpportunities.insert(Def);
        continue;
      }
      L2 = Def;
      Def = *UseToDefs.find(Def)->second.begin();
      L1 = Def;
    } // else the element in the middle of the chain is nothing, thus
      // Def already contains the first element of the chain.

    // Check the number of users of the first node in the chain, i.e., ADRP
    const SetOfMachineInstr *Users =
        getUses(DefsPerColorToUses,
                RegToId.find(Def->getOperand(0).getReg())->second, *Def);
    if (Users->size() > 1) {
#ifdef DEBUG
      // if all the uses of this def are in the defs of the potential candidate,
      // this is a complex candidate of level 1
      if (DefsOfPotentialCandidates.empty()) {
        // lazy init
        DefsOfPotentialCandidates = PotentialCandidates;
        for (const MachineInstr *Candidate : PotentialCandidates) {
          if (!UseToDefs.find(Candidate)->second.empty())
            DefsOfPotentialCandidates.insert(
                *UseToDefs.find(Candidate)->second.begin());
        }
      }
      bool Found = false;
      for (auto &Use : *Users) {
        if (!DefsOfPotentialCandidates.count(Use)) {
          ++NumTooCplxLvl1;
          Found = true;
          break;
        }
      }
      if (!Found)
        ++NumCplxLvl1;
#endif // DEBUG
      continue;
    }

    bool IsL2Add = (ImmediateDefOpc == AArch64::ADDXri);
    // If the chain is three instructions long and ldr is the second element,
    // then this ldr must load form GOT, otherwise this is not a correct chain.
    if (L2 && !IsL2Add &&
        !(L2->getOperand(2).getTargetFlags() & AArch64II::MO_GOT))
      continue;
    SmallVector<const MachineInstr *, 3> Args;
    MCLOHType Kind;
    if (isCandidateLoad(Candidate)) {
      if (!L2) {
        // At this point, the candidate LOH indicates that the ldr instruction
        // may use a direct access to the symbol. There is not such encoding
        // for loads of byte and half.
        if (!supportLoadFromLiteral(Candidate))
          continue;

        DEBUG(dbgs() << "Record AdrpLdr:\n" << *L1 << '\n' << *Candidate
                     << '\n');
        Kind = MCLOH_AdrpLdr;
        Args.push_back(L1);
        Args.push_back(Candidate);
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(L1)) &&
               "L1 already involved in LOH.");
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(Candidate)) &&
               "Candidate already involved in LOH.");
        ++NumADRPToLDR;
      } else {
        DEBUG(dbgs() << "Record Adrp" << (IsL2Add ? "Add" : "LdrGot")
                     << "Ldr:\n" << *L1 << '\n' << *L2 << '\n' << *Candidate
                     << '\n');

        Kind = IsL2Add ? MCLOH_AdrpAddLdr : MCLOH_AdrpLdrGotLdr;
        Args.push_back(L1);
        Args.push_back(L2);
        Args.push_back(Candidate);

        PotentialADROpportunities.remove(L2);
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(L1)) &&
               "L1 already involved in LOH.");
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(L2)) &&
               "L2 already involved in LOH.");
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(Candidate)) &&
               "Candidate already involved in LOH.");
#ifdef DEBUG
        // get the immediate of the load
        if (Candidate->getOperand(2).getImm() == 0)
          if (ImmediateDefOpc == AArch64::ADDXri)
            ++NumADDToLDR;
          else
            ++NumLDRToLDR;
        else if (ImmediateDefOpc == AArch64::ADDXri)
          ++NumADDToLDRWithImm;
        else
          ++NumLDRToLDRWithImm;
#endif // DEBUG
      }
    } else {
      if (ImmediateDefOpc == AArch64::ADRP)
        continue;
      else {

        DEBUG(dbgs() << "Record Adrp" << (IsL2Add ? "Add" : "LdrGot")
                     << "Str:\n" << *L1 << '\n' << *L2 << '\n' << *Candidate
                     << '\n');

        Kind = IsL2Add ? MCLOH_AdrpAddStr : MCLOH_AdrpLdrGotStr;
        Args.push_back(L1);
        Args.push_back(L2);
        Args.push_back(Candidate);

        PotentialADROpportunities.remove(L2);
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(L1)) &&
               "L1 already involved in LOH.");
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(L2)) &&
               "L2 already involved in LOH.");
        assert((!InvolvedInLOHs || InvolvedInLOHs->insert(Candidate)) &&
               "Candidate already involved in LOH.");
#ifdef DEBUG
        // get the immediate of the store
        if (Candidate->getOperand(2).getImm() == 0)
          if (ImmediateDefOpc == AArch64::ADDXri)
            ++NumADDToSTR;
          else
            ++NumLDRToSTR;
        else if (ImmediateDefOpc == AArch64::ADDXri)
          ++NumADDToSTRWithImm;
        else
          ++NumLDRToSTRWithImm;
#endif // DEBUG
      }
    }
    AArch64FI.addLOHDirective(Kind, Args);
  }

  // Now, we grabbed all the big patterns, check ADR opportunities.
  for (const MachineInstr *Candidate : PotentialADROpportunities)
    registerADRCandidate(*Candidate, UseToDefs, DefsPerColorToUses, AArch64FI,
                         InvolvedInLOHs, RegToId);
}

/// Look for every register defined by potential LOHs candidates.
/// Map these registers with dense id in @p RegToId and vice-versa in
/// @p IdToReg. @p IdToReg is populated only in DEBUG mode.
static void collectInvolvedReg(const MachineFunction &MF, MapRegToId &RegToId,
                               MapIdToReg &IdToReg,
                               const TargetRegisterInfo *TRI) {
  unsigned CurRegId = 0;
  if (!PreCollectRegister) {
    unsigned NbReg = TRI->getNumRegs();
    for (; CurRegId < NbReg; ++CurRegId) {
      RegToId[CurRegId] = CurRegId;
      DEBUG(IdToReg.push_back(CurRegId));
      DEBUG(assert(IdToReg[CurRegId] == CurRegId && "Reg index mismatches"));
    }
    return;
  }

  DEBUG(dbgs() << "** Collect Involved Register\n");
  for (const auto &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!canDefBePartOfLOH(&MI) &&
          !isCandidateLoad(&MI) && !isCandidateStore(&MI))
        continue;

      // Process defs
      for (MachineInstr::const_mop_iterator IO = MI.operands_begin(),
                                            IOEnd = MI.operands_end();
           IO != IOEnd; ++IO) {
        if (!IO->isReg() || !IO->isDef())
          continue;
        unsigned CurReg = IO->getReg();
        for (MCRegAliasIterator AI(CurReg, TRI, true); AI.isValid(); ++AI)
          if (RegToId.find(*AI) == RegToId.end()) {
            DEBUG(IdToReg.push_back(*AI);
                  assert(IdToReg[CurRegId] == *AI &&
                         "Reg index mismatches insertion index."));
            RegToId[*AI] = CurRegId++;
            DEBUG(dbgs() << "Register: " << PrintReg(*AI, TRI) << '\n');
          }
      }
    }
  }
}

bool AArch64CollectLOH::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();

  MapRegToId RegToId;
  MapIdToReg IdToReg;
  AArch64FunctionInfo *AArch64FI = MF.getInfo<AArch64FunctionInfo>();
  assert(AArch64FI && "No MachineFunctionInfo for this function!");

  DEBUG(dbgs() << "Looking for LOH in " << MF.getName() << '\n');

  collectInvolvedReg(MF, RegToId, IdToReg, TRI);
  if (RegToId.empty())
    return false;

  MachineInstr *DummyOp = nullptr;
  if (BasicBlockScopeOnly) {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    // For local analysis, create a dummy operation to record uses that are not
    // local.
    DummyOp = MF.CreateMachineInstr(TII->get(AArch64::COPY), DebugLoc());
  }

  unsigned NbReg = RegToId.size();
  bool Modified = false;

  // Start with ADRP.
  InstrToInstrs *ColorOpToReachedUses = new InstrToInstrs[NbReg];

  // Compute the reaching def in ADRP mode, meaning ADRP definitions
  // are first considered as uses.
  reachingDef(MF, ColorOpToReachedUses, RegToId, true, DummyOp);
  DEBUG(dbgs() << "ADRP reaching defs\n");
  DEBUG(printReachingDef(ColorOpToReachedUses, NbReg, TRI, IdToReg));

  // Translate the definition to uses map into a use to definitions map to ease
  // statistic computation.
  InstrToInstrs ADRPToReachingDefs;
  reachedUsesToDefs(ADRPToReachingDefs, ColorOpToReachedUses, RegToId, true);

  // Compute LOH for ADRP.
  computeADRP(ADRPToReachingDefs, *AArch64FI, MDT);
  delete[] ColorOpToReachedUses;

  // Continue with general ADRP -> ADD/LDR -> LDR/STR pattern.
  ColorOpToReachedUses = new InstrToInstrs[NbReg];

  // first perform a regular reaching def analysis.
  reachingDef(MF, ColorOpToReachedUses, RegToId, false, DummyOp);
  DEBUG(dbgs() << "All reaching defs\n");
  DEBUG(printReachingDef(ColorOpToReachedUses, NbReg, TRI, IdToReg));

  // Turn that into a use to defs to ease statistic computation.
  InstrToInstrs UsesToReachingDefs;
  reachedUsesToDefs(UsesToReachingDefs, ColorOpToReachedUses, RegToId, false);

  // Compute other than AdrpAdrp LOH.
  computeOthers(UsesToReachingDefs, ColorOpToReachedUses, *AArch64FI, RegToId,
                MDT);
  delete[] ColorOpToReachedUses;

  if (BasicBlockScopeOnly)
    MF.DeleteMachineInstr(DummyOp);

  return Modified;
}

/// createAArch64CollectLOHPass - returns an instance of the Statistic for
/// linker optimization pass.
FunctionPass *llvm::createAArch64CollectLOHPass() {
  return new AArch64CollectLOH();
}
