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
#include "llvm/Function.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
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

STATISTIC(NumSimple,       "Number of simple if-conversions performed");
STATISTIC(NumSimpleFalse,  "Number of simple (F) if-conversions performed");
STATISTIC(NumTriangle,     "Number of triangle if-conversions performed");
STATISTIC(NumTriangleRev,  "Number of triangle (R) if-conversions performed");
STATISTIC(NumTriangleFalse,"Number of triangle (F) if-conversions performed");
STATISTIC(NumTriangleFRev, "Number of triangle (F/R) if-conversions performed");
STATISTIC(NumDiamonds,     "Number of diamond if-conversions performed");
STATISTIC(NumIfConvBBs,    "Number of if-converted blocks");
STATISTIC(NumDupBBs,       "Number of duplicated blocks");

namespace {
  class VISIBILITY_HIDDEN IfConverter : public MachineFunctionPass {
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
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      SmallVector<MachineOperand, 4> BrCond;
      SmallVector<MachineOperand, 4> Predicate;
      BBInfo() : IsDone(false), IsBeingAnalyzed(false),
                 IsAnalyzed(false), IsEnqueued(false), IsBrAnalyzable(false),
                 HasFallThrough(false), IsUnpredicable(false),
                 CannotBeCopied(false), ClobbersPred(false), NonPredSize(0),
                 BB(0), TrueBB(0), FalseBB(0) {}
    };

    /// IfcvtToken - Record information about pending if-conversions to attemp:
    /// BBI             - Corresponding BBInfo.
    /// Kind            - Type of block. See IfcvtKind.
    /// NeedSubsumsion  - True if the to be predicated BB has already been
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
      bool NeedSubsumsion;
      unsigned NumDups;
      unsigned NumDups2;
      IfcvtToken(BBInfo &b, IfcvtKind k, bool s, unsigned d, unsigned d2 = 0)
        : BBI(b), Kind(k), NeedSubsumsion(s), NumDups(d), NumDups2(d2) {}
    };

    /// Roots - Basic blocks that do not have successors. These are the starting
    /// points of Graph traversal.
    std::vector<MachineBasicBlock*> Roots;

    /// BBAnalysis - Results of if-conversion feasibility analysis indexed by
    /// basic block number.
    std::vector<BBInfo> BBAnalysis;

    const TargetLowering *TLI;
    const TargetInstrInfo *TII;
    bool MadeChange;
  public:
    static char ID;
    IfConverter() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "If Converter"; }

  private:
    bool ReverseBranchCondition(BBInfo &BBI);
    bool ValidSimple(BBInfo &TrueBBI, unsigned &Dups) const;
    bool ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                       bool FalseBranch, unsigned &Dups) const;
    bool ValidDiamond(BBInfo &TrueBBI, BBInfo &FalseBBI,
                      unsigned &Dups1, unsigned &Dups2) const;
    void ScanInstructions(BBInfo &BBI);
    BBInfo &AnalyzeBlock(MachineBasicBlock *BB,
                         std::vector<IfcvtToken*> &Tokens);
    bool FeasibilityAnalysis(BBInfo &BBI, SmallVectorImpl<MachineOperand> &Cond,
                             bool isTriangle = false, bool RevBranch = false);
    bool AnalyzeBlocks(MachineFunction &MF,
                       std::vector<IfcvtToken*> &Tokens);
    void InvalidatePreds(MachineBasicBlock *BB);
    void RemoveExtraEdges(BBInfo &BBI);
    bool IfConvertSimple(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertTriangle(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertDiamond(BBInfo &BBI, IfcvtKind Kind,
                          unsigned NumDups1, unsigned NumDups2);
    void PredicateBlock(BBInfo &BBI,
                        MachineBasicBlock::iterator E,
                        SmallVectorImpl<MachineOperand> &Cond);
    void CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                               SmallVectorImpl<MachineOperand> &Cond,
                               bool IgnoreBr = false);
    void MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI);

    bool MeetIfcvtSizeLimit(unsigned Size) const {
      return Size > 0 && Size <= TLI->getIfCvtBlockSizeLimit();
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
        // Favors subsumsion.
        if (C1->NeedSubsumsion == false && C2->NeedSubsumsion == true)
          return true;
        else if (C1->NeedSubsumsion == C2->NeedSubsumsion) {
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

static RegisterPass<IfConverter>
X("if-converter", "If Converter");

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  static int FnNum = -1;
  DOUT << "\nIfcvt: function (" << ++FnNum <<  ") \'"
       << MF.getFunction()->getName() << "\'";

  if (FnNum < IfCvtFnStart || (IfCvtFnStop != -1 && FnNum > IfCvtFnStop)) {
    DOUT << " skipped\n";
    return false;
  }
  DOUT << "\n";

  MF.RenumberBlocks();
  BBAnalysis.resize(MF.getNumBlockIDs());

  // Look for root nodes, i.e. blocks without successors.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->succ_empty())
      Roots.push_back(I);

  std::vector<IfcvtToken*> Tokens;
  MadeChange = false;
  unsigned NumIfCvts = NumSimple + NumSimpleFalse + NumTriangle +
    NumTriangleRev + NumTriangleFalse + NumTriangleFRev + NumDiamonds;
  while (IfCvtLimit == -1 || (int)NumIfCvts < IfCvtLimit) {
    // Do an intial analysis for each basic block and finding all the potential
    // candidates to perform if-convesion.
    bool Change = AnalyzeBlocks(MF, Tokens);
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
      default: assert(false && "Unexpected!");
        break;
      case ICSimple:
      case ICSimpleFalse: {
        bool isFalse = Kind == ICSimpleFalse;
        if ((isFalse && DisableSimpleF) || (!isFalse && DisableSimple)) break;
        DOUT << "Ifcvt (Simple" << (Kind == ICSimpleFalse ? " false" :"")
             << "): BB#" << BBI.BB->getNumber() << " ("
             << ((Kind == ICSimpleFalse)
                 ? BBI.FalseBB->getNumber()
                 : BBI.TrueBB->getNumber()) << ") ";
        RetVal = IfConvertSimple(BBI, Kind);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) {
          if (isFalse) NumSimpleFalse++;
          else         NumSimple++;
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
        DOUT << "Ifcvt (Triangle";
        if (isFalse)
          DOUT << " false";
        if (isRev)
          DOUT << " rev";
        DOUT << "): BB#" << BBI.BB->getNumber() << " (T:"
             << BBI.TrueBB->getNumber() << ",F:"
             << BBI.FalseBB->getNumber() << ") ";
        RetVal = IfConvertTriangle(BBI, Kind);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) {
          if (isFalse) {
            if (isRev) NumTriangleFRev++;
            else       NumTriangleFalse++;
          } else {
            if (isRev) NumTriangleRev++;
            else       NumTriangle++;
          }
        }
        break;
      }
      case ICDiamond: {
        if (DisableDiamond) break;
        DOUT << "Ifcvt (Diamond): BB#" << BBI.BB->getNumber() << " (T:"
             << BBI.TrueBB->getNumber() << ",F:"
             << BBI.FalseBB->getNumber() << ") ";
        RetVal = IfConvertDiamond(BBI, Kind, NumDups, NumDups2);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) NumDiamonds++;
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
  Roots.clear();
  BBAnalysis.clear();

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
/// branchs. Swap block's 'true' and 'false' successors.
bool IfConverter::ReverseBranchCondition(BBInfo &BBI) {
  if (!TII->ReverseBranchCondition(BBI.BrCond)) {
    TII->RemoveBranch(*BBI.BB);
    TII->InsertBranch(*BBI.BB, BBI.FalseBB, BBI.TrueBB, BBI.BrCond);
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
bool IfConverter::ValidSimple(BBInfo &TrueBBI, unsigned &Dups) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed || TrueBBI.IsDone)
    return false;

  if (TrueBBI.IsBrAnalyzable)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied ||
        TrueBBI.NonPredSize > TLI->getIfCvtDupBlockSizeLimit())
      return false;
    Dups = TrueBBI.NonPredSize;
  }

  return true;
}

/// ValidTriangle - Returns true if the 'true' and 'false' blocks (along
/// with their common predecessor) forms a valid triangle shape for ifcvt.
/// If 'FalseBranch' is true, it checks if 'true' block's false branch
/// branches to the false branch rather than the other way around. It also
/// returns the number of instructions that the ifcvt would need to duplicate
/// if performed in 'Dups'.
bool IfConverter::ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                                bool FalseBranch, unsigned &Dups) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed || TrueBBI.IsDone)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied)
      return false;

    unsigned Size = TrueBBI.NonPredSize;
    if (TrueBBI.IsBrAnalyzable) {
      if (TrueBBI.TrueBB && TrueBBI.BrCond.empty())
        // End with an unconditional branch. It will be removed.
        --Size;
      else {
        MachineBasicBlock *FExit = FalseBranch
          ? TrueBBI.TrueBB : TrueBBI.FalseBB;
        if (FExit)
          // Require a conditional branch
          ++Size;
      }
    }
    if (Size > TLI->getIfCvtDupBlockSizeLimit())
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

static
MachineBasicBlock::iterator firstNonBranchInst(MachineBasicBlock *BB,
                                               const TargetInstrInfo *TII) {
  MachineBasicBlock::iterator I = BB->end();
  while (I != BB->begin()) {
    --I;
    if (!I->getDesc().isBranch())
      break;
  }
  return I;
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

  MachineBasicBlock::iterator TI = TrueBBI.BB->begin();
  MachineBasicBlock::iterator FI = FalseBBI.BB->begin();
  while (TI != TrueBBI.BB->end() && FI != FalseBBI.BB->end()) {
    if (!TI->isIdenticalTo(FI))
      break;
    ++Dups1;
    ++TI;
    ++FI;
  }

  TI = firstNonBranchInst(TrueBBI.BB, TII);
  FI = firstNonBranchInst(FalseBBI.BB, TII);
  while (TI != TrueBBI.BB->begin() && FI != FalseBBI.BB->begin()) {
    if (!TI->isIdenticalTo(FI))
      break;
    ++Dups2;
    --TI;
    --FI;
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

  bool AlreadyPredicated = BBI.Predicate.size() > 0;
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
    assert(BBI.FalseBB && "Expected to find the fallthrough block!");
  }

  // Then scan all the instructions.
  BBI.NonPredSize = 0;
  BBI.ClobbersPred = false;
  bool SeenCondBr = false;
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    const TargetInstrDesc &TID = I->getDesc();
    if (TID.isNotDuplicable())
      BBI.CannotBeCopied = true;

    bool isPredicated = TII->isPredicated(I);
    bool isCondBr = BBI.IsBrAnalyzable && TID.isConditionalBranch();

    if (!isCondBr) {
      if (!isPredicated)
        BBI.NonPredSize++;
      else if (!AlreadyPredicated) {
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
        SeenCondBr = true;

        // Conditional branches is not predicable. But it may be eliminated.
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

    if (!TID.isPredicable()) {
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

    // Test predicate subsumsion.
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

  // Unanalyable or ends with fallthrough or unconditional branch.
  if (!BBI.IsBrAnalyzable || BBI.BrCond.empty()) {
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
  bool TNeedSub = TrueBBI.Predicate.size() > 0;
  bool FNeedSub = FalseBBI.Predicate.size() > 0;
  bool Enqueued = false;
  if (CanRevCond && ValidDiamond(TrueBBI, FalseBBI, Dups, Dups2) &&
      MeetIfcvtSizeLimit(TrueBBI.NonPredSize - (Dups + Dups2)) &&
      MeetIfcvtSizeLimit(FalseBBI.NonPredSize - (Dups + Dups2)) &&
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

  if (ValidTriangle(TrueBBI, FalseBBI, false, Dups) &&
      MeetIfcvtSizeLimit(TrueBBI.NonPredSize) &&
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
  
  if (ValidTriangle(TrueBBI, FalseBBI, true, Dups) &&
      MeetIfcvtSizeLimit(TrueBBI.NonPredSize) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond, true, true)) {
    Tokens.push_back(new IfcvtToken(BBI, ICTriangleRev, TNeedSub, Dups));
    Enqueued = true;
  }

  if (ValidSimple(TrueBBI, Dups) &&
      MeetIfcvtSizeLimit(TrueBBI.NonPredSize) &&
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
    if (ValidTriangle(FalseBBI, TrueBBI, false, Dups) &&
        MeetIfcvtSizeLimit(FalseBBI.NonPredSize) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFalse, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidTriangle(FalseBBI, TrueBBI, true, Dups) &&
        MeetIfcvtSizeLimit(FalseBBI.NonPredSize) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFRev, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidSimple(FalseBBI, Dups) &&
        MeetIfcvtSizeLimit(FalseBBI.NonPredSize) &&
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
/// candidates. It returns true if any CFG restructuring is done to expose more
/// if-conversion opportunities.
bool IfConverter::AnalyzeBlocks(MachineFunction &MF,
                                std::vector<IfcvtToken*> &Tokens) {
  bool Change = false;
  std::set<MachineBasicBlock*> Visited;
  for (unsigned i = 0, e = Roots.size(); i != e; ++i) {
    for (idf_ext_iterator<MachineBasicBlock*> I=idf_ext_begin(Roots[i],Visited),
           E = idf_ext_end(Roots[i], Visited); I != E; ++I) {
      MachineBasicBlock *BB = *I;
      AnalyzeBlock(BB, Tokens);
    }
  }

  // Sort to favor more complex ifcvt scheme.
  std::stable_sort(Tokens.begin(), Tokens.end(), IfcvtTokenCmp);

  return Change;
}

/// canFallThroughTo - Returns true either if ToBB is the next block after BB or
/// that all the intervening blocks are empty (given BB can fall through to its
/// next block).
static bool canFallThroughTo(MachineBasicBlock *BB, MachineBasicBlock *ToBB) {
  MachineFunction::iterator I = BB;
  MachineFunction::iterator TI = ToBB;
  MachineFunction::iterator E = BB->getParent()->end();
  while (++I != TI)
    if (I == E || !I->empty())
      return false;
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
  SmallVector<MachineOperand, 0> NoCond;
  TII->InsertBranch(*BB, ToBB, NULL, NoCond);
}

/// RemoveExtraEdges - Remove true / false edges if either / both are no longer
/// successors.
void IfConverter::RemoveExtraEdges(BBInfo &BBI) {
  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->AnalyzeBranch(*BBI.BB, TBB, FBB, Cond))
    BBI.BB->CorrectExtraCFGEdges(TBB, FBB, !Cond.empty());
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
      assert(false && "Unable to reverse branch condition!");

  if (CvtBBI->BB->pred_size() > 1) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond);
  } else {
    PredicateBlock(*CvtBBI, CvtBBI->BB->end(), Cond);

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
      assert(false && "Unable to reverse branch condition!");

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

  bool HasEarlyExit = CvtBBI->FalseBB != NULL;
  bool DupBB = CvtBBI->BB->pred_size() > 1;
  if (DupBB) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond, true);
  } else {
    // Predicate the 'true' block after removing its branch.
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    PredicateBlock(*CvtBBI, CvtBBI->BB->end(), Cond);
  }

  if (!DupBB) {
    // Now merge the entry of the triangle with the true block.
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, *CvtBBI);
  }

  // If 'true' block has a 'false' successor, add an exit branch to it.
  if (HasEarlyExit) {
    SmallVector<MachineOperand, 4> RevCond(CvtBBI->BrCond.begin(),
                                           CvtBBI->BrCond.end());
    if (TII->ReverseBranchCondition(RevCond))
      assert(false && "Unable to reverse branch condition!");
    TII->InsertBranch(*BBI.BB, CvtBBI->FalseBB, NULL, RevCond);
    BBI.BB->addSuccessor(CvtBBI->FalseBB);
  }

  // Merge in the 'false' block if the 'false' block has no other
  // predecessors. Otherwise, add a unconditional branch from to 'false'.
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
  // True block must fall through or ended with unanalyzable terminator.
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

  // Merge the 'true' and 'false' blocks by copying the instructions
  // from the 'false' block to the 'true' block. That is, unless the true
  // block would clobber the predicate, in that case, do the opposite.
  BBInfo *BBI1 = &TrueBBI;
  BBInfo *BBI2 = &FalseBBI;
  SmallVector<MachineOperand, 4> RevCond(BBI.BrCond.begin(), BBI.BrCond.end());
  if (TII->ReverseBranchCondition(RevCond))
    assert(false && "Unable to reverse branch condition!");
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

  // Remove the duplicated instructions at the beginnings of both paths.
  MachineBasicBlock::iterator DI1 = BBI1->BB->begin();
  MachineBasicBlock::iterator DI2 = BBI2->BB->begin();
  BBI1->NonPredSize -= NumDups1;
  BBI2->NonPredSize -= NumDups1;
  while (NumDups1 != 0) {
    ++DI1;
    ++DI2;
    --NumDups1;
  }
  BBI.BB->splice(BBI.BB->end(), BBI1->BB, BBI1->BB->begin(), DI1);
  BBI2->BB->erase(BBI2->BB->begin(), DI2);

  // Predicate the 'true' block after removing its branch.
  BBI1->NonPredSize -= TII->RemoveBranch(*BBI1->BB);
  DI1 = BBI1->BB->end();
  for (unsigned i = 0; i != NumDups2; ++i)
    --DI1;
  BBI1->BB->erase(DI1, BBI1->BB->end());
  PredicateBlock(*BBI1, BBI1->BB->end(), *Cond1);

  // Predicate the 'false' block.
  BBI2->NonPredSize -= TII->RemoveBranch(*BBI2->BB);
  DI2 = BBI2->BB->end();
  while (NumDups2 != 0) {
    --DI2;
    --NumDups2;
  }
  PredicateBlock(*BBI2, DI2, *Cond2);

  // Merge the true block into the entry of the diamond.
  MergeBlocks(BBI, *BBI1);
  MergeBlocks(BBI, *BBI2);

  // If the if-converted block fallthrough or unconditionally branch into the
  // tail block, and the tail block does not have other predecessors, then
  // fold the tail block in as well. Otherwise, unless it falls through to the
  // tail, add a unconditional branch to it.
  if (TailBB) {
    BBInfo TailBBI = BBAnalysis[TailBB->getNumber()];
    if (TailBB->pred_size() == 1 && !TailBBI.HasFallThrough) {
      BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
      MergeBlocks(BBI, TailBBI);
      TailBBI.IsDone = true;
    } else {
      InsertUncondBranch(BBI.BB, TailBB, TII);
      BBI.HasFallThrough = false;
    }
  }

  RemoveExtraEdges(BBI);

  // Update block info.
  BBI.IsDone = TrueBBI.IsDone = FalseBBI.IsDone = true;
  InvalidatePreds(BBI.BB);

  // FIXME: Must maintain LiveIns.
  return true;
}

/// PredicateBlock - Predicate instructions from the start of the block to the
/// specified end with the specified condition.
void IfConverter::PredicateBlock(BBInfo &BBI,
                                 MachineBasicBlock::iterator E,
                                 SmallVectorImpl<MachineOperand> &Cond) {
  for (MachineBasicBlock::iterator I = BBI.BB->begin(); I != E; ++I) {
    if (TII->isPredicated(I))
      continue;
    if (!TII->PredicateInstruction(I, Cond)) {
      cerr << "Unable to predicate " << *I << "!\n";
      abort();
    }
  }

  std::copy(Cond.begin(), Cond.end(), std::back_inserter(BBI.Predicate));

  BBI.IsAnalyzed = false;
  BBI.NonPredSize = 0;

  NumIfConvBBs++;
}

/// CopyAndPredicateBlock - Copy and predicate instructions from source BB to
/// the destination block. Skip end of block branches if IgnoreBr is true.
void IfConverter::CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                                        SmallVectorImpl<MachineOperand> &Cond,
                                        bool IgnoreBr) {
  MachineFunction &MF = *ToBBI.BB->getParent();

  for (MachineBasicBlock::iterator I = FromBBI.BB->begin(),
         E = FromBBI.BB->end(); I != E; ++I) {
    const TargetInstrDesc &TID = I->getDesc();
    bool isPredicated = TII->isPredicated(I);
    // Do not copy the end of the block branches.
    if (IgnoreBr && !isPredicated && TID.isBranch())
      break;

    MachineInstr *MI = MF.CloneMachineInstr(I);
    ToBBI.BB->insert(ToBBI.BB->end(), MI);
    ToBBI.NonPredSize++;

    if (!isPredicated)
      if (!TII->PredicateInstruction(MI, Cond)) {
        cerr << "Unable to predicate " << *MI << "!\n";
        abort();
      }
  }

  std::vector<MachineBasicBlock *> Succs(FromBBI.BB->succ_begin(),
                                         FromBBI.BB->succ_end());
  MachineBasicBlock *NBB = getNextBlock(FromBBI.BB);
  MachineBasicBlock *FallThrough = FromBBI.HasFallThrough ? NBB : NULL;

  for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
    MachineBasicBlock *Succ = Succs[i];
    // Fallthrough edge can't be transferred.
    if (Succ == FallThrough)
      continue;
    if (!ToBBI.BB->isSuccessor(Succ))
      ToBBI.BB->addSuccessor(Succ);
  }

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  std::copy(Cond.begin(), Cond.end(), std::back_inserter(ToBBI.Predicate));

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.IsAnalyzed = false;

  NumDupBBs++;
}

/// MergeBlocks - Move all instructions from FromBB to the end of ToBB.
///
void IfConverter::MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI) {
  ToBBI.BB->splice(ToBBI.BB->end(),
                   FromBBI.BB, FromBBI.BB->begin(), FromBBI.BB->end());

  // Redirect all branches to FromBB to ToBB.
  std::vector<MachineBasicBlock *> Preds(FromBBI.BB->pred_begin(),
                                         FromBBI.BB->pred_end());
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    MachineBasicBlock *Pred = Preds[i];
    if (Pred == ToBBI.BB)
      continue;
    Pred->ReplaceUsesOfBlockWith(FromBBI.BB, ToBBI.BB);
  }
 
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
    if (!ToBBI.BB->isSuccessor(Succ))
      ToBBI.BB->addSuccessor(Succ);
  }

  // Now FromBBI always fall through to the next block!
  if (NBB && !FromBBI.BB->isSuccessor(NBB))
    FromBBI.BB->addSuccessor(NBB);

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  FromBBI.Predicate.clear();

  ToBBI.NonPredSize += FromBBI.NonPredSize;
  FromBBI.NonPredSize = 0;

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.HasFallThrough = FromBBI.HasFallThrough;
  ToBBI.IsAnalyzed = false;
  FromBBI.IsAnalyzed = false;
}
