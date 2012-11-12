//===-- IfConversion.cpp - Machine code if conversion pass. ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine instruction level if-conversion pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ifcvt"
#include "BranchFolding.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

// Hidden options for help debugging.
static cl::opt<int> IfCvtFnStart("ifcvt-fn-start", cl::init(-1), cl::Hidden);
static cl::opt<int> IfCvtFnStop("ifcvt-fn-stop", cl::init(-1), cl::Hidden);
static cl::opt<int> IfCvtLimit("ifcvt-limit", cl::init(-1), cl::Hidden);
static cl::opt<bool> DisableSimple("disable-ifcvt-simple",
                                   cl::init(false), cl::Hidden);
static cl::opt<bool> DisableSimpleF("disable-ifcvt-simple-false",
                                    cl::init(false), cl::Hidden);
static cl::opt<bool> DisableTriangle("disable-ifcvt-triangle",
                                     cl::init(false), cl::Hidden);
static cl::opt<bool> DisableTriangleR("disable-ifcvt-triangle-rev",
                                      cl::init(false), cl::Hidden);
static cl::opt<bool> DisableTriangleF("disable-ifcvt-triangle-false",
                                      cl::init(false), cl::Hidden);
static cl::opt<bool> DisableTriangleFR("disable-ifcvt-triangle-false-rev",
                                       cl::init(false), cl::Hidden);
static cl::opt<bool> DisableDiamond("disable-ifcvt-diamond",
                                    cl::init(false), cl::Hidden);
static cl::opt<bool> IfCvtBranchFold("ifcvt-branch-fold",
                                     cl::init(true), cl::Hidden);

STATISTIC(NumSimple,       "Number of simple if-conversions performed");
STATISTIC(NumSimpleFalse,  "Number of simple (F) if-conversions performed");
STATISTIC(NumTriangle,     "Number of triangle if-conversions performed");
STATISTIC(NumTriangleRev,  "Number of triangle (R) if-conversions performed");
STATISTIC(NumTriangleFalse,"Number of triangle (F) if-conversions performed");
STATISTIC(NumTriangleFRev, "Number of triangle (F/R) if-conversions performed");
STATISTIC(NumDiamonds,     "Number of diamond if-conversions performed");
STATISTIC(NumIfConvBBs,    "Number of if-converted blocks");
STATISTIC(NumDupBBs,       "Number of duplicated blocks");
STATISTIC(NumUnpred,       "Number of true blocks of diamonds unpredicated");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum IfcvtKind {
      ICNotClassfied,  // BB data valid, but not classified.
      ICSimpleFalse,   // Same as ICSimple, but on the false path.
      ICSimple,        // BB is entry of an one split, no rejoin sub-CFG.
      ICTriangleFRev,  // Same as ICTriangleFalse, but false path rev condition.
      ICTriangleRev,   // Same as ICTriangle, but true path rev condition.
      ICTriangleFalse, // Same as ICTriangle, but on the false path.
      ICTriangle,      // BB is entry of a triangle sub-CFG.
      ICDiamond        // BB is entry of a diamond sub-CFG.
    };

    /// BBInfo - One per MachineBasicBlock, this is used to cache the result
    /// if-conversion feasibility analysis. This includes results from
    /// TargetInstrInfo::AnalyzeBranch() (i.e. TBB, FBB, and Cond), and its
    /// classification, and common tail block of its successors (if it's a
    /// diamond shape), its size, whether it's predicable, and whether any
    /// instruction can clobber the 'would-be' predicate.
    ///
    /// IsDone          - True if BB is not to be considered for ifcvt.
    /// IsBeingAnalyzed - True if BB is currently being analyzed.
    /// IsAnalyzed      - True if BB has been analyzed (info is still valid).
    /// IsEnqueued      - True if BB has been enqueued to be ifcvt'ed.
    /// IsBrAnalyzable  - True if AnalyzeBranch() returns false.
    /// HasFallThrough  - True if BB may fallthrough to the following BB.
    /// IsUnpredicable  - True if BB is known to be unpredicable.
    /// ClobbersPred    - True if BB could modify predicates (e.g. has
    ///                   cmp, call, etc.)
    /// NonPredSize     - Number of non-predicated instructions.
    /// ExtraCost       - Extra cost for multi-cycle instructions.
    /// ExtraCost2      - Some instructions are slower when predicated
    /// BB              - Corresponding MachineBasicBlock.
    /// TrueBB / FalseBB- See AnalyzeBranch().
    /// BrCond          - Conditions for end of block conditional branches.
    /// Predicate       - Predicate used in the BB.
    struct BBInfo {
      bool IsDone          : 1;
      bool IsBeingAnalyzed : 1;
      bool IsAnalyzed      : 1;
      bool IsEnqueued      : 1;
      bool IsBrAnalyzable  : 1;
      bool HasFallThrough  : 1;
      bool IsUnpredicable  : 1;
      bool CannotBeCopied  : 1;
      bool ClobbersPred    : 1;
      unsigned NonPredSize;
      unsigned ExtraCost;
      unsigned ExtraCost2;
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      SmallVector<MachineOperand, 4> BrCond;
      SmallVector<MachineOperand, 4> Predicate;
      BBInfo() : IsDone(false), IsBeingAnalyzed(false),
                 IsAnalyzed(false), IsEnqueued(false), IsBrAnalyzable(false),
                 HasFallThrough(false), IsUnpredicable(false),
                 CannotBeCopied(false), ClobbersPred(false), NonPredSize(0),
                 ExtraCost(0), ExtraCost2(0), BB(0), TrueBB(0), FalseBB(0) {}
    };

    /// IfcvtToken - Record information about pending if-conversions to attempt:
    /// BBI             - Corresponding BBInfo.
    /// Kind            - Type of block. See IfcvtKind.
    /// NeedSubsumption - True if the to-be-predicated BB has already been
    ///                   predicated.
    /// NumDups      - Number of instructions that would be duplicated due
    ///                   to this if-conversion. (For diamonds, the number of
    ///                   identical instructions at the beginnings of both
    ///                   paths).
    /// NumDups2     - For diamonds, the number of identical instructions
    ///                   at the ends of both paths.
    struct IfcvtToken {
      BBInfo &BBI;
      IfcvtKind Kind;
      bool NeedSubsumption;
      unsigned NumDups;
      unsigned NumDups2;
      IfcvtToken(BBInfo &b, IfcvtKind k, bool s, unsigned d, unsigned d2 = 0)
        : BBI(b), Kind(k), NeedSubsumption(s), NumDups(d), NumDups2(d2) {}
    };

    /// BBAnalysis - Results of if-conversion feasibility analysis indexed by
    /// basic block number.
    std::vector<BBInfo> BBAnalysis;

    const TargetLowering *TLI;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const InstrItineraryData *InstrItins;
    const MachineBranchProbabilityInfo *MBPI;
    MachineRegisterInfo *MRI;

    bool PreRegAlloc;
    bool MadeChange;
    int FnNum;
  public:
    static char ID;
    IfConverter() : MachineFunctionPass(ID), FnNum(-1) {
      initializeIfConverterPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineBranchProbabilityInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    bool ReverseBranchCondition(BBInfo &BBI);
    bool ValidSimple(BBInfo &TrueBBI, unsigned &Dups,
                     const BranchProbability &Prediction) const;
    bool ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                       bool FalseBranch, unsigned &Dups,
                       const BranchProbability &Prediction) const;
    bool ValidDiamond(BBInfo &TrueBBI, BBInfo &FalseBBI,
                      unsigned &Dups1, unsigned &Dups2) const;
    void ScanInstructions(BBInfo &BBI);
    BBInfo &AnalyzeBlock(MachineBasicBlock *BB,
                         std::vector<IfcvtToken*> &Tokens);
    bool FeasibilityAnalysis(BBInfo &BBI, SmallVectorImpl<MachineOperand> &Cond,
                             bool isTriangle = false, bool RevBranch = false);
    void AnalyzeBlocks(MachineFunction &MF, std::vector<IfcvtToken*> &Tokens);
    void InvalidatePreds(MachineBasicBlock *BB);
    void RemoveExtraEdges(BBInfo &BBI);
    bool IfConvertSimple(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertTriangle(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertDiamond(BBInfo &BBI, IfcvtKind Kind,
                          unsigned NumDups1, unsigned NumDups2);
    void PredicateBlock(BBInfo &BBI,
                        MachineBasicBlock::iterator E,
                        SmallVectorImpl<MachineOperand> &Cond,
                        SmallSet<unsigned, 4> &Redefs,
                        SmallSet<unsigned, 4> *LaterRedefs = 0);
    void CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                               SmallVectorImpl<MachineOperand> &Cond,
                               SmallSet<unsigned, 4> &Redefs,
                               bool IgnoreBr = false);
    void MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI, bool AddEdges = true);

    bool MeetIfcvtSizeLimit(MachineBasicBlock &BB,
                            unsigned Cycle, unsigned Extra,
                            const BranchProbability &Prediction) const {
      return Cycle > 0 && TII->isProfitableToIfCvt(BB, Cycle, Extra,
                                                   Prediction);
    }

    bool MeetIfcvtSizeLimit(MachineBasicBlock &TBB,
                            unsigned TCycle, unsigned TExtra,
                            MachineBasicBlock &FBB,
                            unsigned FCycle, unsigned FExtra,
                            const BranchProbability &Prediction) const {
      return TCycle > 0 && FCycle > 0 &&
        TII->isProfitableToIfCvt(TBB, TCycle, TExtra, FBB, FCycle, FExtra,
                                 Prediction);
    }

    // blockAlwaysFallThrough - Block ends without a terminator.
    bool blockAlwaysFallThrough(BBInfo &BBI) const {
      return BBI.IsBrAnalyzable && BBI.TrueBB == NULL;
    }

    // IfcvtTokenCmp - Used to sort if-conversion candidates.
    static bool IfcvtTokenCmp(IfcvtToken *C1, IfcvtToken *C2) {
      int Incr1 = (C1->Kind == ICDiamond)
        ? -(int)(C1->NumDups + C1->NumDups2) : (int)C1->NumDups;
      int Incr2 = (C2->Kind == ICDiamond)
        ? -(int)(C2->NumDups + C2->NumDups2) : (int)C2->NumDups;
      if (Incr1 > Incr2)
        return true;
      else if (Incr1 == Incr2) {
        // Favors subsumption.
        if (C1->NeedSubsumption == false && C2->NeedSubsumption == true)
          return true;
        else if (C1->NeedSubsumption == C2->NeedSubsumption) {
          // Favors diamond over triangle, etc.
          if ((unsigned)C1->Kind < (unsigned)C2->Kind)
            return true;
          else if (C1->Kind == C2->Kind)
            return C1->BBI.BB->getNumber() < C2->BBI.BB->getNumber();
        }
      }
      return false;
    }
  };

  char IfConverter::ID = 0;
}

char &llvm::IfConverterID = IfConverter::ID;

INITIALIZE_PASS_BEGIN(IfConverter, "if-converter", "If Converter", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_END(IfConverter, "if-converter", "If Converter", false, false)

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  TRI = MF.getTarget().getRegisterInfo();
  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MRI = &MF.getRegInfo();
  InstrItins = MF.getTarget().getInstrItineraryData();
  if (!TII) return false;

  PreRegAlloc = MRI->isSSA();

  bool BFChange = false;
  if (!PreRegAlloc) {
    // Tail merge tend to expose more if-conversion opportunities.
    BranchFolder BF(true, false);
    BFChange = BF.OptimizeFunction(MF, TII,
                                   MF.getTarget().getRegisterInfo(),
                                   getAnalysisIfAvailable<MachineModuleInfo>());
  }

  DEBUG(dbgs() << "\nIfcvt: function (" << ++FnNum <<  ") \'"
               << MF.getName() << "\'");

  if (FnNum < IfCvtFnStart || (IfCvtFnStop != -1 && FnNum > IfCvtFnStop)) {
    DEBUG(dbgs() << " skipped\n");
    return false;
  }
  DEBUG(dbgs() << "\n");

  MF.RenumberBlocks();
  BBAnalysis.resize(MF.getNumBlockIDs());

  std::vector<IfcvtToken*> Tokens;
  MadeChange = false;
  unsigned NumIfCvts = NumSimple + NumSimpleFalse + NumTriangle +
    NumTriangleRev + NumTriangleFalse + NumTriangleFRev + NumDiamonds;
  while (IfCvtLimit == -1 || (int)NumIfCvts < IfCvtLimit) {
    // Do an initial analysis for each basic block and find all the potential
    // candidates to perform if-conversion.
    bool Change = false;
    AnalyzeBlocks(MF, Tokens);
    while (!Tokens.empty()) {
      IfcvtToken *Token = Tokens.back();
      Tokens.pop_back();
      BBInfo &BBI = Token->BBI;
      IfcvtKind Kind = Token->Kind;
      unsigned NumDups = Token->NumDups;
      unsigned NumDups2 = Token->NumDups2;

      delete Token;

      // If the block has been evicted out of the queue or it has already been
      // marked dead (due to it being predicated), then skip it.
      if (BBI.IsDone)
        BBI.IsEnqueued = false;
      if (!BBI.IsEnqueued)
        continue;

      BBI.IsEnqueued = false;

      bool RetVal = false;
      switch (Kind) {
      default: llvm_unreachable("Unexpected!");
      case ICSimple:
      case ICSimpleFalse: {
        bool isFalse = Kind == ICSimpleFalse;
        if ((isFalse && DisableSimpleF) || (!isFalse && DisableSimple)) break;
        DEBUG(dbgs() << "Ifcvt (Simple" << (Kind == ICSimpleFalse ?
                                            " false" : "")
                     << "): BB#" << BBI.BB->getNumber() << " ("
                     << ((Kind == ICSimpleFalse)
                         ? BBI.FalseBB->getNumber()
                         : BBI.TrueBB->getNumber()) << ") ");
        RetVal = IfConvertSimple(BBI, Kind);
        DEBUG(dbgs() << (RetVal ? "succeeded!" : "failed!") << "\n");
        if (RetVal) {
          if (isFalse) ++NumSimpleFalse;
          else         ++NumSimple;
        }
       break;
      }
      case ICTriangle:
      case ICTriangleRev:
      case ICTriangleFalse:
      case ICTriangleFRev: {
        bool isFalse = Kind == ICTriangleFalse;
        bool isRev   = (Kind == ICTriangleRev || Kind == ICTriangleFRev);
        if (DisableTriangle && !isFalse && !isRev) break;
        if (DisableTriangleR && !isFalse && isRev) break;
        if (DisableTriangleF && isFalse && !isRev) break;
        if (DisableTriangleFR && isFalse && isRev) break;
        DEBUG(dbgs() << "Ifcvt (Triangle");
        if (isFalse)
          DEBUG(dbgs() << " false");
        if (isRev)
          DEBUG(dbgs() << " rev");
        DEBUG(dbgs() << "): BB#" << BBI.BB->getNumber() << " (T:"
                     << BBI.TrueBB->getNumber() << ",F:"
                     << BBI.FalseBB->getNumber() << ") ");
        RetVal = IfConvertTriangle(BBI, Kind);
        DEBUG(dbgs() << (RetVal ? "succeeded!" : "failed!") << "\n");
        if (RetVal) {
          if (isFalse) {
            if (isRev) ++NumTriangleFRev;
            else       ++NumTriangleFalse;
          } else {
            if (isRev) ++NumTriangleRev;
            else       ++NumTriangle;
          }
        }
        break;
      }
      case ICDiamond: {
        if (DisableDiamond) break;
        DEBUG(dbgs() << "Ifcvt (Diamond): BB#" << BBI.BB->getNumber() << " (T:"
                     << BBI.TrueBB->getNumber() << ",F:"
                     << BBI.FalseBB->getNumber() << ") ");
        RetVal = IfConvertDiamond(BBI, Kind, NumDups, NumDups2);
        DEBUG(dbgs() << (RetVal ? "succeeded!" : "failed!") << "\n");
        if (RetVal) ++NumDiamonds;
        break;
      }
      }

      Change |= RetVal;

      NumIfCvts = NumSimple + NumSimpleFalse + NumTriangle + NumTriangleRev +
        NumTriangleFalse + NumTriangleFRev + NumDiamonds;
      if (IfCvtLimit != -1 && (int)NumIfCvts >= IfCvtLimit)
        break;
    }

    if (!Change)
      break;
    MadeChange |= Change;
  }

  // Delete tokens in case of early exit.
  while (!Tokens.empty()) {
    IfcvtToken *Token = Tokens.back();
    Tokens.pop_back();
    delete Token;
  }

  Tokens.clear();
  BBAnalysis.clear();

  if (MadeChange && IfCvtBranchFold) {
    BranchFolder BF(false, false);
    BF.OptimizeFunction(MF, TII,
                        MF.getTarget().getRegisterInfo(),
                        getAnalysisIfAvailable<MachineModuleInfo>());
  }

  MadeChange |= BFChange;
  return MadeChange;
}

/// findFalseBlock - BB has a fallthrough. Find its 'false' successor given
/// its 'true' successor.
static MachineBasicBlock *findFalseBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *TrueBB) {
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    if (SuccBB != TrueBB)
      return SuccBB;
  }
  return NULL;
}

/// ReverseBranchCondition - Reverse the condition of the end of the block
/// branch. Swap block's 'true' and 'false' successors.
bool IfConverter::ReverseBranchCondition(BBInfo &BBI) {
  DebugLoc dl;  // FIXME: this is nowhere
  if (!TII->ReverseBranchCondition(BBI.BrCond)) {
    TII->RemoveBranch(*BBI.BB);
    TII->InsertBranch(*BBI.BB, BBI.FalseBB, BBI.TrueBB, BBI.BrCond, dl);
    std::swap(BBI.TrueBB, BBI.FalseBB);
    return true;
  }
  return false;
}

/// getNextBlock - Returns the next block in the function blocks ordering. If
/// it is the end, returns NULL.
static inline MachineBasicBlock *getNextBlock(MachineBasicBlock *BB) {
  MachineFunction::iterator I = BB;
  MachineFunction::iterator E = BB->getParent()->end();
  if (++I == E)
    return NULL;
  return I;
}

/// ValidSimple - Returns true if the 'true' block (along with its
/// predecessor) forms a valid simple shape for ifcvt. It also returns the
/// number of instructions that the ifcvt would need to duplicate if performed
/// in Dups.
bool IfConverter::ValidSimple(BBInfo &TrueBBI, unsigned &Dups,
                              const BranchProbability &Prediction) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed || TrueBBI.IsDone)
    return false;

  if (TrueBBI.IsBrAnalyzable)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied ||
        !TII->isProfitableToDupForIfCvt(*TrueBBI.BB, TrueBBI.NonPredSize,
                                        Prediction))
      return false;
    Dups = TrueBBI.NonPredSize;
  }

  return true;
}

/// ValidTriangle - Returns true if the 'true' and 'false' blocks (along
/// with their common predecessor) forms a valid triangle shape for ifcvt.
/// If 'FalseBranch' is true, it checks if 'true' block's false branch
/// branches to the 'false' block rather than the other way around. It also
/// returns the number of instructions that the ifcvt would need to duplicate
/// if performed in 'Dups'.
bool IfConverter::ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                                bool FalseBranch, unsigned &Dups,
                                const BranchProbability &Prediction) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed || TrueBBI.IsDone)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied)
      return false;

    unsigned Size = TrueBBI.NonPredSize;
    if (TrueBBI.IsBrAnalyzable) {
      if (TrueBBI.TrueBB && TrueBBI.BrCond.empty())
        // Ends with an unconditional branch. It will be removed.
        --Size;
      else {
        MachineBasicBlock *FExit = FalseBranch
          ? TrueBBI.TrueBB : TrueBBI.FalseBB;
        if (FExit)
          // Require a conditional branch
          ++Size;
      }
    }
    if (!TII->isProfitableToDupForIfCvt(*TrueBBI.BB, Size, Prediction))
      return false;
    Dups = Size;
  }

  MachineBasicBlock *TExit = FalseBranch ? TrueBBI.FalseBB : TrueBBI.TrueBB;
  if (!TExit && blockAlwaysFallThrough(TrueBBI)) {
    MachineFunction::iterator I = TrueBBI.BB;
    if (++I == TrueBBI.BB->getParent()->end())
      return false;
    TExit = I;
  }
  return TExit && TExit == FalseBBI.BB;
}

/// ValidDiamond - Returns true if the 'true' and 'false' blocks (along
/// with their common predecessor) forms a valid diamond shape for ifcvt.
bool IfConverter::ValidDiamond(BBInfo &TrueBBI, BBInfo &FalseBBI,
                               unsigned &Dups1, unsigned &Dups2) const {
  Dups1 = Dups2 = 0;
  if (TrueBBI.IsBeingAnalyzed || TrueBBI.IsDone ||
      FalseBBI.IsBeingAnalyzed || FalseBBI.IsDone)
    return false;

  MachineBasicBlock *TT = TrueBBI.TrueBB;
  MachineBasicBlock *FT = FalseBBI.TrueBB;

  if (!TT && blockAlwaysFallThrough(TrueBBI))
    TT = getNextBlock(TrueBBI.BB);
  if (!FT && blockAlwaysFallThrough(FalseBBI))
    FT = getNextBlock(FalseBBI.BB);
  if (TT != FT)
    return false;
  if (TT == NULL && (TrueBBI.IsBrAnalyzable || FalseBBI.IsBrAnalyzable))
    return false;
  if  (TrueBBI.BB->pred_size() > 1 || FalseBBI.BB->pred_size() > 1)
    return false;

  // FIXME: Allow true block to have an early exit?
  if (TrueBBI.FalseBB || FalseBBI.FalseBB ||
      (TrueBBI.ClobbersPred && FalseBBI.ClobbersPred))
    return false;

  // Count duplicate instructions at the beginning of the true and false blocks.
  MachineBasicBlock::iterator TIB = TrueBBI.BB->begin();
  MachineBasicBlock::iterator FIB = FalseBBI.BB->begin();
  MachineBasicBlock::iterator TIE = TrueBBI.BB->end();
  MachineBasicBlock::iterator FIE = FalseBBI.BB->end();
  while (TIB != TIE && FIB != FIE) {
    // Skip dbg_value instructions. These do not count.
    if (TIB->isDebugValue()) {
      while (TIB != TIE && TIB->isDebugValue())
        ++TIB;
      if (TIB == TIE)
        break;
    }
    if (FIB->isDebugValue()) {
      while (FIB != FIE && FIB->isDebugValue())
        ++FIB;
      if (FIB == FIE)
        break;
    }
    if (!TIB->isIdenticalTo(FIB))
      break;
    ++Dups1;
    ++TIB;
    ++FIB;
  }

  // Now, in preparation for counting duplicate instructions at the ends of the
  // blocks, move the end iterators up past any branch instructions.
  while (TIE != TIB) {
    --TIE;
    if (!TIE->isBranch())
      break;
  }
  while (FIE != FIB) {
    --FIE;
    if (!FIE->isBranch())
      break;
  }

  // If Dups1 includes all of a block, then don't count duplicate
  // instructions at the end of the blocks.
  if (TIB == TIE || FIB == FIE)
    return true;

  // Count duplicate instructions at the ends of the blocks.
  while (TIE != TIB && FIE != FIB) {
    // Skip dbg_value instructions. These do not count.
    if (TIE->isDebugValue()) {
      while (TIE != TIB && TIE->isDebugValue())
        --TIE;
      if (TIE == TIB)
        break;
    }
    if (FIE->isDebugValue()) {
      while (FIE != FIB && FIE->isDebugValue())
        --FIE;
      if (FIE == FIB)
        break;
    }
    if (!TIE->isIdenticalTo(FIE))
      break;
    ++Dups2;
    --TIE;
    --FIE;
  }

  return true;
}

/// ScanInstructions - Scan all the instructions in the block to determine if
/// the block is predicable. In most cases, that means all the instructions
/// in the block are isPredicable(). Also checks if the block contains any
/// instruction which can clobber a predicate (e.g. condition code register).
/// If so, the block is not predicable unless it's the last instruction.
void IfConverter::ScanInstructions(BBInfo &BBI) {
  if (BBI.IsDone)
    return;

  bool AlreadyPredicated = !BBI.Predicate.empty();
  // First analyze the end of BB branches.
  BBI.TrueBB = BBI.FalseBB = NULL;
  BBI.BrCond.clear();
  BBI.IsBrAnalyzable =
    !TII->AnalyzeBranch(*BBI.BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
  BBI.HasFallThrough = BBI.IsBrAnalyzable && BBI.FalseBB == NULL;

  if (BBI.BrCond.size()) {
    // No false branch. This BB must end with a conditional branch and a
    // fallthrough.
    if (!BBI.FalseBB)
      BBI.FalseBB = findFalseBlock(BBI.BB, BBI.TrueBB);
    if (!BBI.FalseBB) {
      // Malformed bcc? True and false blocks are the same?
      BBI.IsUnpredicable = true;
      return;
    }
  }

  // Then scan all the instructions.
  BBI.NonPredSize = 0;
  BBI.ExtraCost = 0;
  BBI.ExtraCost2 = 0;
  BBI.ClobbersPred = false;
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    if (I->isDebugValue())
      continue;

    if (I->isNotDuplicable())
      BBI.CannotBeCopied = true;

    bool isPredicated = TII->isPredicated(I);
    bool isCondBr = BBI.IsBrAnalyzable && I->isConditionalBranch();

    if (!isCondBr) {
      if (!isPredicated) {
        BBI.NonPredSize++;
        unsigned ExtraPredCost = 0;
        unsigned NumCycles = TII->getInstrLatency(InstrItins, &*I,
                                                  &ExtraPredCost);
        if (NumCycles > 1)
          BBI.ExtraCost += NumCycles-1;
        BBI.ExtraCost2 += ExtraPredCost;
      } else if (!AlreadyPredicated) {
        // FIXME: This instruction is already predicated before the
        // if-conversion pass. It's probably something like a conditional move.
        // Mark this block unpredicable for now.
        BBI.IsUnpredicable = true;
        return;
      }
    }

    if (BBI.ClobbersPred && !isPredicated) {
      // Predicate modification instruction should end the block (except for
      // already predicated instructions and end of block branches).
      if (isCondBr) {
        // A conditional branch is not predicable, but it may be eliminated.
        continue;
      }

      // Predicate may have been modified, the subsequent (currently)
      // unpredicated instructions cannot be correctly predicated.
      BBI.IsUnpredicable = true;
      return;
    }

    // FIXME: Make use of PredDefs? e.g. ADDC, SUBC sets predicates but are
    // still potentially predicable.
    std::vector<MachineOperand> PredDefs;
    if (TII->DefinesPredicate(I, PredDefs))
      BBI.ClobbersPred = true;

    if (!TII->isPredicable(I)) {
      BBI.IsUnpredicable = true;
      return;
    }
  }
}

/// FeasibilityAnalysis - Determine if the block is a suitable candidate to be
/// predicated by the specified predicate.
bool IfConverter::FeasibilityAnalysis(BBInfo &BBI,
                                      SmallVectorImpl<MachineOperand> &Pred,
                                      bool isTriangle, bool RevBranch) {
  // If the block is dead or unpredicable, then it cannot be predicated.
  if (BBI.IsDone || BBI.IsUnpredicable)
    return false;

  // If it is already predicated, check if its predicate subsumes the new
  // predicate.
  if (BBI.Predicate.size() && !TII->SubsumesPredicate(BBI.Predicate, Pred))
    return false;

  if (BBI.BrCond.size()) {
    if (!isTriangle)
      return false;

    // Test predicate subsumption.
    SmallVector<MachineOperand, 4> RevPred(Pred.begin(), Pred.end());
    SmallVector<MachineOperand, 4> Cond(BBI.BrCond.begin(), BBI.BrCond.end());
    if (RevBranch) {
      if (TII->ReverseBranchCondition(Cond))
        return false;
    }
    if (TII->ReverseBranchCondition(RevPred) ||
        !TII->SubsumesPredicate(Cond, RevPred))
      return false;
  }

  return true;
}

/// AnalyzeBlock - Analyze the structure of the sub-CFG starting from
/// the specified block. Record its successors and whether it looks like an
/// if-conversion candidate.
IfConverter::BBInfo &IfConverter::AnalyzeBlock(MachineBasicBlock *BB,
                                             std::vector<IfcvtToken*> &Tokens) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.IsAnalyzed || BBI.IsBeingAnalyzed)
    return BBI;

  BBI.BB = BB;
  BBI.IsBeingAnalyzed = true;

  ScanInstructions(BBI);

  // Unanalyzable or ends with fallthrough or unconditional branch, or if is not
  // considered for ifcvt anymore.
  if (!BBI.IsBrAnalyzable || BBI.BrCond.empty() || BBI.IsDone) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  // Do not ifcvt if either path is a back edge to the entry block.
  if (BBI.TrueBB == BB || BBI.FalseBB == BB) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  // Do not ifcvt if true and false fallthrough blocks are the same.
  if (!BBI.FalseBB) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  BBInfo &TrueBBI  = AnalyzeBlock(BBI.TrueBB, Tokens);
  BBInfo &FalseBBI = AnalyzeBlock(BBI.FalseBB, Tokens);

  if (TrueBBI.IsDone && FalseBBI.IsDone) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  SmallVector<MachineOperand, 4> RevCond(BBI.BrCond.begin(), BBI.BrCond.end());
  bool CanRevCond = !TII->ReverseBranchCondition(RevCond);

  unsigned Dups = 0;
  unsigned Dups2 = 0;
  bool TNeedSub = !TrueBBI.Predicate.empty();
  bool FNeedSub = !FalseBBI.Predicate.empty();
  bool Enqueued = false;

  BranchProbability Prediction = MBPI->getEdgeProbability(BB, TrueBBI.BB);

  if (CanRevCond && ValidDiamond(TrueBBI, FalseBBI, Dups, Dups2) &&
      MeetIfcvtSizeLimit(*TrueBBI.BB, (TrueBBI.NonPredSize - (Dups + Dups2) +
                                       TrueBBI.ExtraCost), TrueBBI.ExtraCost2,
                         *FalseBBI.BB, (FalseBBI.NonPredSize - (Dups + Dups2) +
                                        FalseBBI.ExtraCost),FalseBBI.ExtraCost2,
                         Prediction) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond) &&
      FeasibilityAnalysis(FalseBBI, RevCond)) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //  TailBB
    // Note TailBB can be empty.
    Tokens.push_back(new IfcvtToken(BBI, ICDiamond, TNeedSub|FNeedSub, Dups,
                                    Dups2));
    Enqueued = true;
  }

  if (ValidTriangle(TrueBBI, FalseBBI, false, Dups, Prediction) &&
      MeetIfcvtSizeLimit(*TrueBBI.BB, TrueBBI.NonPredSize + TrueBBI.ExtraCost,
                         TrueBBI.ExtraCost2, Prediction) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond, true)) {
    // Triangle:
    //   EBB
    //   | \_
    //   |  |
    //   | TBB
    //   |  /
    //   FBB
    Tokens.push_back(new IfcvtToken(BBI, ICTriangle, TNeedSub, Dups));
    Enqueued = true;
  }

  if (ValidTriangle(TrueBBI, FalseBBI, true, Dups, Prediction) &&
      MeetIfcvtSizeLimit(*TrueBBI.BB, TrueBBI.NonPredSize + TrueBBI.ExtraCost,
                         TrueBBI.ExtraCost2, Prediction) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond, true, true)) {
    Tokens.push_back(new IfcvtToken(BBI, ICTriangleRev, TNeedSub, Dups));
    Enqueued = true;
  }

  if (ValidSimple(TrueBBI, Dups, Prediction) &&
      MeetIfcvtSizeLimit(*TrueBBI.BB, TrueBBI.NonPredSize + TrueBBI.ExtraCost,
                         TrueBBI.ExtraCost2, Prediction) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond)) {
    // Simple (split, no rejoin):
    //   EBB
    //   | \_
    //   |  |
    //   | TBB---> exit
    //   |
    //   FBB
    Tokens.push_back(new IfcvtToken(BBI, ICSimple, TNeedSub, Dups));
    Enqueued = true;
  }

  if (CanRevCond) {
    // Try the other path...
    if (ValidTriangle(FalseBBI, TrueBBI, false, Dups,
                      Prediction.getCompl()) &&
        MeetIfcvtSizeLimit(*FalseBBI.BB,
                           FalseBBI.NonPredSize + FalseBBI.ExtraCost,
                           FalseBBI.ExtraCost2, Prediction.getCompl()) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFalse, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidTriangle(FalseBBI, TrueBBI, true, Dups,
                      Prediction.getCompl()) &&
        MeetIfcvtSizeLimit(*FalseBBI.BB,
                           FalseBBI.NonPredSize + FalseBBI.ExtraCost,
                           FalseBBI.ExtraCost2, Prediction.getCompl()) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFRev, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidSimple(FalseBBI, Dups, Prediction.getCompl()) &&
        MeetIfcvtSizeLimit(*FalseBBI.BB,
                           FalseBBI.NonPredSize + FalseBBI.ExtraCost,
                           FalseBBI.ExtraCost2, Prediction.getCompl()) &&
        FeasibilityAnalysis(FalseBBI, RevCond)) {
      Tokens.push_back(new IfcvtToken(BBI, ICSimpleFalse, FNeedSub, Dups));
      Enqueued = true;
    }
  }

  BBI.IsEnqueued = Enqueued;
  BBI.IsBeingAnalyzed = false;
  BBI.IsAnalyzed = true;
  return BBI;
}

/// AnalyzeBlocks - Analyze all blocks and find entries for all if-conversion
/// candidates.
void IfConverter::AnalyzeBlocks(MachineFunction &MF,
                                std::vector<IfcvtToken*> &Tokens) {
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *BB = I;
    AnalyzeBlock(BB, Tokens);
  }

  // Sort to favor more complex ifcvt scheme.
  std::stable_sort(Tokens.begin(), Tokens.end(), IfcvtTokenCmp);
}

/// canFallThroughTo - Returns true either if ToBB is the next block after BB or
/// that all the intervening blocks are empty (given BB can fall through to its
/// next block).
static bool canFallThroughTo(MachineBasicBlock *BB, MachineBasicBlock *ToBB) {
  MachineFunction::iterator PI = BB;
  MachineFunction::iterator I = llvm::next(PI);
  MachineFunction::iterator TI = ToBB;
  MachineFunction::iterator E = BB->getParent()->end();
  while (I != TI) {
    // Check isSuccessor to avoid case where the next block is empty, but
    // it's not a successor.
    if (I == E || !I->empty() || !PI->isSuccessor(I))
      return false;
    PI = I++;
  }
  return true;
}

/// InvalidatePreds - Invalidate predecessor BB info so it would be re-analyzed
/// to determine if it can be if-converted. If predecessor is already enqueued,
/// dequeue it!
void IfConverter::InvalidatePreds(MachineBasicBlock *BB) {
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI) {
    BBInfo &PBBI = BBAnalysis[(*PI)->getNumber()];
    if (PBBI.IsDone || PBBI.BB == BB)
      continue;
    PBBI.IsAnalyzed = false;
    PBBI.IsEnqueued = false;
  }
}

/// InsertUncondBranch - Inserts an unconditional branch from BB to ToBB.
///
static void InsertUncondBranch(MachineBasicBlock *BB, MachineBasicBlock *ToBB,
                               const TargetInstrInfo *TII) {
  DebugLoc dl;  // FIXME: this is nowhere
  SmallVector<MachineOperand, 0> NoCond;
  TII->InsertBranch(*BB, ToBB, NULL, NoCond, dl);
}

/// RemoveExtraEdges - Remove true / false edges if either / both are no longer
/// successors.
void IfConverter::RemoveExtraEdges(BBInfo &BBI) {
  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->AnalyzeBranch(*BBI.BB, TBB, FBB, Cond))
    BBI.BB->CorrectExtraCFGEdges(TBB, FBB, !Cond.empty());
}

/// InitPredRedefs / UpdatePredRedefs - Defs by predicated instructions are
/// modeled as read + write (sort like two-address instructions). These
/// routines track register liveness and add implicit uses to if-converted
/// instructions to conform to the model.
static void InitPredRedefs(MachineBasicBlock *BB, SmallSet<unsigned,4> &Redefs,
                           const TargetRegisterInfo *TRI) {
  for (MachineBasicBlock::livein_iterator I = BB->livein_begin(),
         E = BB->livein_end(); I != E; ++I) {
    unsigned Reg = *I;
    Redefs.insert(Reg);
    for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
      Redefs.insert(*SubRegs);
  }
}

static void UpdatePredRedefs(MachineInstr *MI, SmallSet<unsigned,4> &Redefs,
                             const TargetRegisterInfo *TRI,
                             bool AddImpUse = false) {
  SmallVector<unsigned, 4> Defs;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef())
      Defs.push_back(Reg);
    else if (MO.isKill()) {
      Redefs.erase(Reg);
      for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
        Redefs.erase(*SubRegs);
    }
  }
  for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
    unsigned Reg = Defs[i];
    if (!Redefs.insert(Reg)) {
      if (AddImpUse)
        // Treat predicated update as read + write.
        MI->addOperand(MachineOperand::CreateReg(Reg, false/*IsDef*/,
                                              true/*IsImp*/,false/*IsKill*/,
                                              false/*IsDead*/,true/*IsUndef*/));
    } else {
      for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
        Redefs.insert(*SubRegs);
    }
  }
}

static void UpdatePredRedefs(MachineBasicBlock::iterator I,
                             MachineBasicBlock::iterator E,
                             SmallSet<unsigned,4> &Redefs,
                             const TargetRegisterInfo *TRI) {
  while (I != E) {
    UpdatePredRedefs(I, Redefs, TRI);
    ++I;
  }
}

/// IfConvertSimple - If convert a simple (split, no rejoin) sub-CFG.
///
bool IfConverter::IfConvertSimple(BBInfo &BBI, IfcvtKind Kind) {
  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;

  SmallVector<MachineOperand, 4> Cond(BBI.BrCond.begin(), BBI.BrCond.end());
  if (Kind == ICSimpleFalse)
    std::swap(CvtBBI, NextBBI);

  if (CvtBBI->IsDone ||
      (CvtBBI->CannotBeCopied && CvtBBI->BB->pred_size() > 1)) {
    // Something has changed. It's no longer safe to predicate this block.
    BBI.IsAnalyzed = false;
    CvtBBI->IsAnalyzed = false;
    return false;
  }

  if (Kind == ICSimpleFalse)
    if (TII->ReverseBranchCondition(Cond))
      llvm_unreachable("Unable to reverse branch condition!");

  // Initialize liveins to the first BB. These are potentiall redefined by
  // predicated instructions.
  SmallSet<unsigned, 4> Redefs;
  InitPredRedefs(CvtBBI->BB, Redefs, TRI);
  InitPredRedefs(NextBBI->BB, Redefs, TRI);

  if (CvtBBI->BB->pred_size() > 1) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them, and add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond, Redefs);
  } else {
    PredicateBlock(*CvtBBI, CvtBBI->BB->end(), Cond, Redefs);

    // Merge converted block into entry block.
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, *CvtBBI);
  }

  bool IterIfcvt = true;
  if (!canFallThroughTo(BBI.BB, NextBBI->BB)) {
    InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
    BBI.HasFallThrough = false;
    // Now ifcvt'd block will look like this:
    // BB:
    // ...
    // t, f = cmp
    // if t op
    // b BBf
    //
    // We cannot further ifcvt this block because the unconditional branch
    // will have to be predicated on the new condition, that will not be
    // available if cmp executes.
    IterIfcvt = false;
  }

  RemoveExtraEdges(BBI);

  // Update block info. BB can be iteratively if-converted.
  if (!IterIfcvt)
    BBI.IsDone = true;
  InvalidatePreds(BBI.BB);
  CvtBBI->IsDone = true;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertTriangle - If convert a triangle sub-CFG.
///
bool IfConverter::IfConvertTriangle(BBInfo &BBI, IfcvtKind Kind) {
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;
  DebugLoc dl;  // FIXME: this is nowhere

  SmallVector<MachineOperand, 4> Cond(BBI.BrCond.begin(), BBI.BrCond.end());
  if (Kind == ICTriangleFalse || Kind == ICTriangleFRev)
    std::swap(CvtBBI, NextBBI);

  if (CvtBBI->IsDone ||
      (CvtBBI->CannotBeCopied && CvtBBI->BB->pred_size() > 1)) {
    // Something has changed. It's no longer safe to predicate this block.
    BBI.IsAnalyzed = false;
    CvtBBI->IsAnalyzed = false;
    return false;
  }

  if (Kind == ICTriangleFalse || Kind == ICTriangleFRev)
    if (TII->ReverseBranchCondition(Cond))
      llvm_unreachable("Unable to reverse branch condition!");

  if (Kind == ICTriangleRev || Kind == ICTriangleFRev) {
    if (ReverseBranchCondition(*CvtBBI)) {
      // BB has been changed, modify its predecessors (except for this
      // one) so they don't get ifcvt'ed based on bad intel.
      for (MachineBasicBlock::pred_iterator PI = CvtBBI->BB->pred_begin(),
             E = CvtBBI->BB->pred_end(); PI != E; ++PI) {
        MachineBasicBlock *PBB = *PI;
        if (PBB == BBI.BB)
          continue;
        BBInfo &PBBI = BBAnalysis[PBB->getNumber()];
        if (PBBI.IsEnqueued) {
          PBBI.IsAnalyzed = false;
          PBBI.IsEnqueued = false;
        }
      }
    }
  }

  // Initialize liveins to the first BB. These are potentially redefined by
  // predicated instructions.
  SmallSet<unsigned, 4> Redefs;
  InitPredRedefs(CvtBBI->BB, Redefs, TRI);
  InitPredRedefs(NextBBI->BB, Redefs, TRI);

  bool HasEarlyExit = CvtBBI->FalseBB != NULL;
  if (CvtBBI->BB->pred_size() > 1) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them, and add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond, Redefs, true);
  } else {
    // Predicate the 'true' block after removing its branch.
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    PredicateBlock(*CvtBBI, CvtBBI->BB->end(), Cond, Redefs);

    // Now merge the entry of the triangle with the true block.
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, *CvtBBI, false);
  }

  // If 'true' block has a 'false' successor, add an exit branch to it.
  if (HasEarlyExit) {
    SmallVector<MachineOperand, 4> RevCond(CvtBBI->BrCond.begin(),
                                           CvtBBI->BrCond.end());
    if (TII->ReverseBranchCondition(RevCond))
      llvm_unreachable("Unable to reverse branch condition!");
    TII->InsertBranch(*BBI.BB, CvtBBI->FalseBB, NULL, RevCond, dl);
    BBI.BB->addSuccessor(CvtBBI->FalseBB);
  }

  // Merge in the 'false' block if the 'false' block has no other
  // predecessors. Otherwise, add an unconditional branch to 'false'.
  bool FalseBBDead = false;
  bool IterIfcvt = true;
  bool isFallThrough = canFallThroughTo(BBI.BB, NextBBI->BB);
  if (!isFallThrough) {
    // Only merge them if the true block does not fallthrough to the false
    // block. By not merging them, we make it possible to iteratively
    // ifcvt the blocks.
    if (!HasEarlyExit &&
        NextBBI->BB->pred_size() == 1 && !NextBBI->HasFallThrough) {
      MergeBlocks(BBI, *NextBBI);
      FalseBBDead = true;
    } else {
      InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
      BBI.HasFallThrough = false;
    }
    // Mixed predicated and unpredicated code. This cannot be iteratively
    // predicated.
    IterIfcvt = false;
  }

  RemoveExtraEdges(BBI);

  // Update block info. BB can be iteratively if-converted.
  if (!IterIfcvt)
    BBI.IsDone = true;
  InvalidatePreds(BBI.BB);
  CvtBBI->IsDone = true;
  if (FalseBBDead)
    NextBBI->IsDone = true;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertDiamond - If convert a diamond sub-CFG.
///
bool IfConverter::IfConvertDiamond(BBInfo &BBI, IfcvtKind Kind,
                                   unsigned NumDups1, unsigned NumDups2) {
  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  MachineBasicBlock *TailBB = TrueBBI.TrueBB;
  // True block must fall through or end with an unanalyzable terminator.
  if (!TailBB) {
    if (blockAlwaysFallThrough(TrueBBI))
      TailBB = FalseBBI.TrueBB;
    assert((TailBB || !TrueBBI.IsBrAnalyzable) && "Unexpected!");
  }

  if (TrueBBI.IsDone || FalseBBI.IsDone ||
      TrueBBI.BB->pred_size() > 1 ||
      FalseBBI.BB->pred_size() > 1) {
    // Something has changed. It's no longer safe to predicate these blocks.
    BBI.IsAnalyzed = false;
    TrueBBI.IsAnalyzed = false;
    FalseBBI.IsAnalyzed = false;
    return false;
  }

  // Put the predicated instructions from the 'true' block before the
  // instructions from the 'false' block, unless the true block would clobber
  // the predicate, in which case, do the opposite.
  BBInfo *BBI1 = &TrueBBI;
  BBInfo *BBI2 = &FalseBBI;
  SmallVector<MachineOperand, 4> RevCond(BBI.BrCond.begin(), BBI.BrCond.end());
  if (TII->ReverseBranchCondition(RevCond))
    llvm_unreachable("Unable to reverse branch condition!");
  SmallVector<MachineOperand, 4> *Cond1 = &BBI.BrCond;
  SmallVector<MachineOperand, 4> *Cond2 = &RevCond;

  // Figure out the more profitable ordering.
  bool DoSwap = false;
  if (TrueBBI.ClobbersPred && !FalseBBI.ClobbersPred)
    DoSwap = true;
  else if (TrueBBI.ClobbersPred == FalseBBI.ClobbersPred) {
    if (TrueBBI.NonPredSize > FalseBBI.NonPredSize)
      DoSwap = true;
  }
  if (DoSwap) {
    std::swap(BBI1, BBI2);
    std::swap(Cond1, Cond2);
  }

  // Remove the conditional branch from entry to the blocks.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);

  // Initialize liveins to the first BB. These are potentially redefined by
  // predicated instructions.
  SmallSet<unsigned, 4> Redefs;
  InitPredRedefs(BBI1->BB, Redefs, TRI);

  // Remove the duplicated instructions at the beginnings of both paths.
  MachineBasicBlock::iterator DI1 = BBI1->BB->begin();
  MachineBasicBlock::iterator DI2 = BBI2->BB->begin();
  MachineBasicBlock::iterator DIE1 = BBI1->BB->end();
  MachineBasicBlock::iterator DIE2 = BBI2->BB->end();
  // Skip dbg_value instructions
  while (DI1 != DIE1 && DI1->isDebugValue())
    ++DI1;
  while (DI2 != DIE2 && DI2->isDebugValue())
    ++DI2;
  BBI1->NonPredSize -= NumDups1;
  BBI2->NonPredSize -= NumDups1;

  // Skip past the dups on each side separately since there may be
  // differing dbg_value entries.
  for (unsigned i = 0; i < NumDups1; ++DI1) {
    if (!DI1->isDebugValue())
      ++i;
  }
  while (NumDups1 != 0) {
    ++DI2;
    if (!DI2->isDebugValue())
      --NumDups1;
  }

  UpdatePredRedefs(BBI1->BB->begin(), DI1, Redefs, TRI);
  BBI.BB->splice(BBI.BB->end(), BBI1->BB, BBI1->BB->begin(), DI1);
  BBI2->BB->erase(BBI2->BB->begin(), DI2);

  // Remove branch from 'true' block and remove duplicated instructions.
  BBI1->NonPredSize -= TII->RemoveBranch(*BBI1->BB);
  DI1 = BBI1->BB->end();
  for (unsigned i = 0; i != NumDups2; ) {
    // NumDups2 only counted non-dbg_value instructions, so this won't
    // run off the head of the list.
    assert (DI1 != BBI1->BB->begin());
    --DI1;
    // skip dbg_value instructions
    if (!DI1->isDebugValue())
      ++i;
  }
  BBI1->BB->erase(DI1, BBI1->BB->end());

  // Remove 'false' block branch and find the last instruction to predicate.
  BBI2->NonPredSize -= TII->RemoveBranch(*BBI2->BB);
  DI2 = BBI2->BB->end();
  while (NumDups2 != 0) {
    // NumDups2 only counted non-dbg_value instructions, so this won't
    // run off the head of the list.
    assert (DI2 != BBI2->BB->begin());
    --DI2;
    // skip dbg_value instructions
    if (!DI2->isDebugValue())
      --NumDups2;
  }

  // Remember which registers would later be defined by the false block.
  // This allows us not to predicate instructions in the true block that would
  // later be re-defined. That is, rather than
  //   subeq  r0, r1, #1
  //   addne  r0, r1, #1
  // generate:
  //   sub    r0, r1, #1
  //   addne  r0, r1, #1
  SmallSet<unsigned, 4> RedefsByFalse;
  SmallSet<unsigned, 4> ExtUses;
  if (TII->isProfitableToUnpredicate(*BBI1->BB, *BBI2->BB)) {
    for (MachineBasicBlock::iterator FI = BBI2->BB->begin(); FI != DI2; ++FI) {
      if (FI->isDebugValue())
        continue;
      SmallVector<unsigned, 4> Defs;
      for (unsigned i = 0, e = FI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = FI->getOperand(i);
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        if (MO.isDef()) {
          Defs.push_back(Reg);
        } else if (!RedefsByFalse.count(Reg)) {
          // These are defined before ctrl flow reach the 'false' instructions.
          // They cannot be modified by the 'true' instructions.
          ExtUses.insert(Reg);
          for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
            ExtUses.insert(*SubRegs);
        }
      }

      for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
        unsigned Reg = Defs[i];
        if (!ExtUses.count(Reg)) {
          RedefsByFalse.insert(Reg);
          for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
            RedefsByFalse.insert(*SubRegs);
        }
      }
    }
  }

  // Predicate the 'true' block.
  PredicateBlock(*BBI1, BBI1->BB->end(), *Cond1, Redefs, &RedefsByFalse);

  // Predicate the 'false' block.
  PredicateBlock(*BBI2, DI2, *Cond2, Redefs);

  // Merge the true block into the entry of the diamond.
  MergeBlocks(BBI, *BBI1, TailBB == 0);
  MergeBlocks(BBI, *BBI2, TailBB == 0);

  // If the if-converted block falls through or unconditionally branches into
  // the tail block, and the tail block does not have other predecessors, then
  // fold the tail block in as well. Otherwise, unless it falls through to the
  // tail, add a unconditional branch to it.
  if (TailBB) {
    BBInfo &TailBBI = BBAnalysis[TailBB->getNumber()];
    bool CanMergeTail = !TailBBI.HasFallThrough;
    // There may still be a fall-through edge from BBI1 or BBI2 to TailBB;
    // check if there are any other predecessors besides those.
    unsigned NumPreds = TailBB->pred_size();
    if (NumPreds > 1)
      CanMergeTail = false;
    else if (NumPreds == 1 && CanMergeTail) {
      MachineBasicBlock::pred_iterator PI = TailBB->pred_begin();
      if (*PI != BBI1->BB && *PI != BBI2->BB)
        CanMergeTail = false;
    }
    if (CanMergeTail) {
      MergeBlocks(BBI, TailBBI);
      TailBBI.IsDone = true;
    } else {
      BBI.BB->addSuccessor(TailBB);
      InsertUncondBranch(BBI.BB, TailBB, TII);
      BBI.HasFallThrough = false;
    }
  }

  // RemoveExtraEdges won't work if the block has an unanalyzable branch,
  // which can happen here if TailBB is unanalyzable and is merged, so
  // explicitly remove BBI1 and BBI2 as successors.
  BBI.BB->removeSuccessor(BBI1->BB);
  BBI.BB->removeSuccessor(BBI2->BB);
  RemoveExtraEdges(BBI);

  // Update block info.
  BBI.IsDone = TrueBBI.IsDone = FalseBBI.IsDone = true;
  InvalidatePreds(BBI.BB);

  // FIXME: Must maintain LiveIns.
  return true;
}

static bool MaySpeculate(const MachineInstr *MI,
                         SmallSet<unsigned, 4> &LaterRedefs,
                         const TargetInstrInfo *TII) {
  bool SawStore = true;
  if (!MI->isSafeToMove(TII, 0, SawStore))
    return false;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef() && !LaterRedefs.count(Reg))
      return false;
  }

  return true;
}

/// PredicateBlock - Predicate instructions from the start of the block to the
/// specified end with the specified condition.
void IfConverter::PredicateBlock(BBInfo &BBI,
                                 MachineBasicBlock::iterator E,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 SmallSet<unsigned, 4> &Redefs,
                                 SmallSet<unsigned, 4> *LaterRedefs) {
  bool AnyUnpred = false;
  bool MaySpec = LaterRedefs != 0;
  for (MachineBasicBlock::iterator I = BBI.BB->begin(); I != E; ++I) {
    if (I->isDebugValue() || TII->isPredicated(I))
      continue;
    // It may be possible not to predicate an instruction if it's the 'true'
    // side of a diamond and the 'false' side may re-define the instruction's
    // defs.
    if (MaySpec && MaySpeculate(I, *LaterRedefs, TII)) {
      AnyUnpred = true;
      continue;
    }
    // If any instruction is predicated, then every instruction after it must
    // be predicated.
    MaySpec = false;
    if (!TII->PredicateInstruction(I, Cond)) {
#ifndef NDEBUG
      dbgs() << "Unable to predicate " << *I << "!\n";
#endif
      llvm_unreachable(0);
    }

    // If the predicated instruction now redefines a register as the result of
    // if-conversion, add an implicit kill.
    UpdatePredRedefs(I, Redefs, TRI, true);
  }

  std::copy(Cond.begin(), Cond.end(), std::back_inserter(BBI.Predicate));

  BBI.IsAnalyzed = false;
  BBI.NonPredSize = 0;

  ++NumIfConvBBs;
  if (AnyUnpred)
    ++NumUnpred;
}

/// CopyAndPredicateBlock - Copy and predicate instructions from source BB to
/// the destination block. Skip end of block branches if IgnoreBr is true.
void IfConverter::CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                                        SmallVectorImpl<MachineOperand> &Cond,
                                        SmallSet<unsigned, 4> &Redefs,
                                        bool IgnoreBr) {
  MachineFunction &MF = *ToBBI.BB->getParent();

  for (MachineBasicBlock::iterator I = FromBBI.BB->begin(),
         E = FromBBI.BB->end(); I != E; ++I) {
    // Do not copy the end of the block branches.
    if (IgnoreBr && I->isBranch())
      break;

    MachineInstr *MI = MF.CloneMachineInstr(I);
    ToBBI.BB->insert(ToBBI.BB->end(), MI);
    ToBBI.NonPredSize++;
    unsigned ExtraPredCost = 0;
    unsigned NumCycles = TII->getInstrLatency(InstrItins, &*I, &ExtraPredCost);
    if (NumCycles > 1)
      ToBBI.ExtraCost += NumCycles-1;
    ToBBI.ExtraCost2 += ExtraPredCost;

    if (!TII->isPredicated(I) && !MI->isDebugValue()) {
      if (!TII->PredicateInstruction(MI, Cond)) {
#ifndef NDEBUG
        dbgs() << "Unable to predicate " << *I << "!\n";
#endif
        llvm_unreachable(0);
      }
    }

    // If the predicated instruction now redefines a register as the result of
    // if-conversion, add an implicit kill.
    UpdatePredRedefs(MI, Redefs, TRI, true);
  }

  if (!IgnoreBr) {
    std::vector<MachineBasicBlock *> Succs(FromBBI.BB->succ_begin(),
                                           FromBBI.BB->succ_end());
    MachineBasicBlock *NBB = getNextBlock(FromBBI.BB);
    MachineBasicBlock *FallThrough = FromBBI.HasFallThrough ? NBB : NULL;

    for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
      MachineBasicBlock *Succ = Succs[i];
      // Fallthrough edge can't be transferred.
      if (Succ == FallThrough)
        continue;
      ToBBI.BB->addSuccessor(Succ);
    }
  }

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  std::copy(Cond.begin(), Cond.end(), std::back_inserter(ToBBI.Predicate));

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.IsAnalyzed = false;

  ++NumDupBBs;
}

/// MergeBlocks - Move all instructions from FromBB to the end of ToBB.
/// This will leave FromBB as an empty block, so remove all of its
/// successor edges except for the fall-through edge.  If AddEdges is true,
/// i.e., when FromBBI's branch is being moved, add those successor edges to
/// ToBBI.
void IfConverter::MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI, bool AddEdges) {
  ToBBI.BB->splice(ToBBI.BB->end(),
                   FromBBI.BB, FromBBI.BB->begin(), FromBBI.BB->end());

  std::vector<MachineBasicBlock *> Succs(FromBBI.BB->succ_begin(),
                                         FromBBI.BB->succ_end());
  MachineBasicBlock *NBB = getNextBlock(FromBBI.BB);
  MachineBasicBlock *FallThrough = FromBBI.HasFallThrough ? NBB : NULL;

  for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
    MachineBasicBlock *Succ = Succs[i];
    // Fallthrough edge can't be transferred.
    if (Succ == FallThrough)
      continue;
    FromBBI.BB->removeSuccessor(Succ);
    if (AddEdges)
      ToBBI.BB->addSuccessor(Succ);
  }

  // Now FromBBI always falls through to the next block!
  if (NBB && !FromBBI.BB->isSuccessor(NBB))
    FromBBI.BB->addSuccessor(NBB);

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  FromBBI.Predicate.clear();

  ToBBI.NonPredSize += FromBBI.NonPredSize;
  ToBBI.ExtraCost += FromBBI.ExtraCost;
  ToBBI.ExtraCost2 += FromBBI.ExtraCost2;
  FromBBI.NonPredSize = 0;
  FromBBI.ExtraCost = 0;
  FromBBI.ExtraCost2 = 0;

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.HasFallThrough = FromBBI.HasFallThrough;
  ToBBI.IsAnalyzed = false;
  FromBBI.IsAnalyzed = false;
}
