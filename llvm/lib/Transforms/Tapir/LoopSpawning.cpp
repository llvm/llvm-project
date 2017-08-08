//===- LoopSpawning.cpp - Spawn loop iterations efficiently ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modify Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>

using std::make_pair;

using namespace llvm;

#define LS_NAME "loop-spawning"
#define DEBUG_TYPE LS_NAME

STATISTIC(LoopsAnalyzed, "Number of Tapir loops analyzed");
STATISTIC(LoopsConvertedToDAC,
          "Number of Tapir loops converted to divide-and-conquer iteration spawning");
STATISTIC(LoopsConvertedToCilkABI,
          "Number of Tapir loops converted to use the Cilk ABI for loops");

namespace {
// Forward declarations.
class LoopSpawningHints;

// /// \brief This modifies LoopAccessReport to initialize message with
// /// tapir-loop-specific part.
// class LoopSpawningReport : public LoopAccessReport {
// public:
//   LoopSpawningReport(Instruction *I = nullptr)
//       : LoopAccessReport("loop-spawning: ", I) {}

//   /// \brief This allows promotion of the loop-access analysis report into the
//   /// loop-spawning report.  It modifies the message to add the
//   /// loop-spawning-specific part of the message.
//   explicit LoopSpawningReport(const LoopAccessReport &R)
//       : LoopAccessReport(Twine("loop-spawning: ") + R.str(),
//                          R.getInstr()) {}
// };


/// Utility class for getting and setting loop spawning hints in the form
/// of loop metadata.
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class LoopSpawningHints {
  enum HintKind { HK_STRATEGY };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val) {
      switch (Kind) {
      case HK_STRATEGY:
        return (Val < ST_END);
      }
      return false;
    }
  };

  /// Spawning strategy
  Hint Strategy;

  /// Return the loop metadata prefix.
  static StringRef Prefix() { return "tapir.loop."; }

public:
  enum SpawningStrategy {
    ST_SEQ,
    ST_DAC,
    ST_END,
  };

  static std::string printStrategy(enum SpawningStrategy Strat) {
    switch(Strat) {
    case LoopSpawningHints::ST_SEQ:
      return "Spawn iterations sequentially";
    case LoopSpawningHints::ST_DAC:
      return "Use divide-and-conquer";
    case LoopSpawningHints::ST_END:
    default:
      return "Unknown";
    }
  }

  LoopSpawningHints(const Loop *L, OptimizationRemarkEmitter &ORE)
      : Strategy("spawn.strategy", ST_SEQ, HK_STRATEGY),
        TheLoop(L), ORE(ORE) {
    // Populate values with existing loop metadata.
    getHintsFromMetadata();
  }

  // /// Dumps all the hint information.
  // std::string emitRemark() const {
  //   LoopSpawningReport R;
  //   R << "Strategy = " << printStrategy(getStrategy());

  //   return R.str();
  // }

  enum SpawningStrategy getStrategy() const {
    return (SpawningStrategy)Strategy.Value;
  }

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata() {
    MDNode *LoopID = TheLoop->getLoopID();
    if (!LoopID)
      return;

    // First operand should refer to the loop id itself.
    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      const MDString *S = nullptr;
      SmallVector<Metadata *, 4> Args;

      // The expected hint is either a MDString or a MDNode with the first
      // operand a MDString.
      if (const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i))) {
        if (!MD || MD->getNumOperands() == 0)
          continue;
        S = dyn_cast<MDString>(MD->getOperand(0));
        for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
          Args.push_back(MD->getOperand(i));
      } else {
        S = dyn_cast<MDString>(LoopID->getOperand(i));
        assert(Args.size() == 0 && "too many arguments for MDString");
      }

      if (!S)
        continue;

      // Check if the hint starts with the loop metadata prefix.
      StringRef Name = S->getString();
      if (Args.size() == 1)
        setHint(Name, Args[0]);
    }
  }

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg) {
    if (!Name.startswith(Prefix()))
      return;
    Name = Name.substr(Prefix().size(), StringRef::npos);

    const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
    if (!C)
      return;
    unsigned Val = C->getZExtValue();

    Hint *Hints[] = {&Strategy};
    for (auto H : Hints) {
      if (Name == H->Name) {
        if (H->validate(Val))
          H->Value = Val;
        else
          DEBUG(dbgs() << LS_NAME << " ignoring invalid hint '" <<
                Name << "'\n");
        break;
      }
    }
  }

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const {
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    Metadata *MDs[] = {MDString::get(Context, Name),
                       ConstantAsMetadata::get(
                           ConstantInt::get(Type::getInt32Ty(Context), V))};
    return MDNode::get(Context, MDs);
  }

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes) {
    MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
    if (!Name)
      return false;

    for (auto H : HintTypes)
      if (Name->getString().endswith(H.Name))
        return true;
    return false;
  }

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes) {
    if (HintTypes.size() == 0)
      return;

    // Reserve the first element to LoopID (see below).
    SmallVector<Metadata *, 4> MDs(1);
    // If the loop already has metadata, then ignore the existing operands.
    MDNode *LoopID = TheLoop->getLoopID();
    if (LoopID) {
      for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
        MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
        // If node in update list, ignore old value.
        if (!matchesHintMetadataName(Node, HintTypes))
          MDs.push_back(Node);
      }
    }

    // Now, add the missing hints.
    for (auto H : HintTypes)
      MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

    // Replace current metadata node with new one.
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);

    TheLoop->setLoopID(NewLoopID);
  }

  /// The loop these hints belong to.
  const Loop *TheLoop;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;
};

// static void emitAnalysisDiag(const Loop *TheLoop,
//                              OptimizationRemarkEmitter &ORE,
//                              const LoopAccessReport &Message) {
//   const char *Name = LS_NAME;
//   LoopAccessReport::emitAnalysis(Message, TheLoop, Name, ORE);
// }

static void emitMissedWarning(Function *F, Loop *L,
                              const LoopSpawningHints &LH,
                              OptimizationRemarkEmitter *ORE) {
  // ORE->emit(OptimizationRemarkMissed(
  //               LS_NAME, "LSHint", L->getStartLoc(), L->getHeader())
  //           << "Strategy = "
  //           << LoopSpawningHints::printStrategy(LH.getStrategy()));
  switch (LH.getStrategy()) {
  case LoopSpawningHints::ST_DAC:
    ORE->emit(DiagnosticInfoOptimizationFailure(
                  DEBUG_TYPE, "FailedRequestedSpawning",
                  L->getStartLoc(), L->getHeader())
              << "Tapir loop not transformed: "
              << "failed to use divide-and-conquer loop spawning");
    break;
  case LoopSpawningHints::ST_SEQ:
    ORE->emit(DiagnosticInfoOptimizationFailure(
                  DEBUG_TYPE, "SpawningDisabled",
                  L->getStartLoc(), L->getHeader())
              << "Tapir loop not transformed: "
              << "loop-spawning transformation disabled");
    break;
  case LoopSpawningHints::ST_END:
    ORE->emit(DiagnosticInfoOptimizationFailure(
                  DEBUG_TYPE, "FailedRequestedSpawning",
                  L->getStartLoc(), L->getHeader())
              << "Tapir loop not transformed: "
              << "unknown loop-spawning strategy");
    break;
  }
}

/// LoopOutline serves as a base class for different variants of LoopSpawning.
/// LoopOutline implements common parts of LoopSpawning transformations, namely,
/// lifting a Tapir loop into a separate helper function.
class LoopOutline {
public:

  LoopOutline(Loop *OrigLoop, ScalarEvolution &SE,
              LoopInfo *LI, DominatorTree *DT,
              AssumptionCache *AC,
              OptimizationRemarkEmitter &ORE)
      : OrigLoop(OrigLoop), SE(SE), LI(LI), DT(DT), AC(AC), ORE(ORE),
        ExitBlock(nullptr)
  {
    // Use the loop latch to determine the canonical exit block for this loop.
    TerminatorInst *TI = OrigLoop->getLoopLatch()->getTerminator();
    if (2 != TI->getNumSuccessors())
      return;
    ExitBlock = TI->getSuccessor(0);
    if (ExitBlock == OrigLoop->getHeader())
      ExitBlock = TI->getSuccessor(1);
  }

  virtual bool processLoop() = 0;

  virtual ~LoopOutline() {}

protected:
  PHINode* canonicalizeIVs(Type *Ty);
  Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit);
  void unlinkLoop();

  /// The original loop.
  Loop *OrigLoop;

  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  // PredicatedScalarEvolution &PSE;
  ScalarEvolution &SE;
  /// Loop info.
  LoopInfo *LI;
  /// Dominator tree.
  DominatorTree *DT;
  /// Assumption cache.
  AssumptionCache *AC;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;

  /// The exit block of this loop.  We compute our own exit block, based on the
  /// latch, and handle other exit blocks (i.e., for exception handling) in a
  /// special manner.
  BasicBlock *ExitBlock;

// private:
//   /// Report an analysis message to assist the user in diagnosing loops that are
//   /// not transformed.  These are handled as LoopAccessReport rather than
//   /// VectorizationReport because the << operator of LoopSpawningReport returns
//   /// LoopAccessReport.
//   void emitAnalysis(const LoopAccessReport &Message) const {
//     emitAnalysisDiag(OrigLoop, *ORE, Message);
//   }
};

/// DACLoopSpawning implements the transformation to spawn the iterations of a
/// Tapir loop in a recursive divide-and-conquer fashion.
class DACLoopSpawning : public LoopOutline {
public:
  // DACLoopSpawning(Loop *OrigLoop, ScalarEvolution &SE,
  //                 LoopInfo *LI, DominatorTree *DT,
  //                 const TargetLibraryInfo *TLI,
  //                 const TargetTransformInfo *TTI,
  //                 OptimizationRemarkEmitter *ORE)
  //     : OrigLoop(OrigLoop), SE(SE), LI(LI), DT(DT),
  //       TLI(TLI), TTI(TTI), ORE(ORE)
  // {}

  DACLoopSpawning(Loop *OrigLoop, ScalarEvolution &SE,
                  LoopInfo *LI, DominatorTree *DT,
                  AssumptionCache *AC,
                  OptimizationRemarkEmitter &ORE)
      : LoopOutline(OrigLoop, SE, LI, DT, AC, ORE)
  {}

  bool processLoop();

  virtual ~DACLoopSpawning() {}

protected:
  Value* computeGrainsize(Value *Limit);
  void implementDACIterSpawnOnHelper(Function *Helper,
                                     BasicBlock *Preheader,
                                     BasicBlock *Header,
                                     PHINode *CanonicalIV,
                                     Argument *Limit,
                                     Argument *Grainsize,
                                     Instruction *SyncRegion,
                                     DominatorTree *DT,
                                     LoopInfo *LI,
                                     bool CanonicalIVFlagNUW = false,
                                     bool CanonicalIVFlagNSW = false);

// private:
//   /// Report an analysis message to assist the user in diagnosing loops that are
//   /// not transformed.  These are handled as LoopAccessReport rather than
//   /// VectorizationReport because the << operator of LoopSpawningReport returns
//   /// LoopAccessReport.
//   void emitAnalysis(const LoopAccessReport &Message) const {
//     emitAnalysisDiag(OrigLoop, *ORE, Message);
//   }
};

/// CilkABILoopSpawning uses the Cilk Plus ABI to handle Tapir loops.
class CilkABILoopSpawning : public LoopOutline {
public:
  CilkABILoopSpawning(Loop *OrigLoop, ScalarEvolution &SE,
                      LoopInfo *LI, DominatorTree *DT,
                      AssumptionCache *AC,
                      OptimizationRemarkEmitter &ORE)
      : LoopOutline(OrigLoop, SE, LI, DT, AC, ORE)
  {}

  bool processLoop();

  virtual ~CilkABILoopSpawning() {}

protected:
  // PHINode* canonicalizeIVs(Type *Ty);
  Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit);

// private:
//   /// Report an analysis message to assist the user in diagnosing loops that are
//   /// not transformed.  These are handled as LoopAccessReport rather than
//   /// VectorizationReport because the << operator of LoopSpawningReport returns
//   /// LoopAccessReport.
//   void emitAnalysis(const LoopAccessReport &Message) const {
//     emitAnalysisDiag(OrigLoop, *ORE, Message);
//   }
};

struct LoopSpawningImpl {
  // LoopSpawningImpl(Function &F, LoopInfo &LI, ScalarEvolution &SE,
  //                  DominatorTree &DT,
  //                  const TargetTransformInfo &TTI,
  //                  const TargetLibraryInfo *TLI,
  //                  AliasAnalysis &AA, AssumptionCache &AC,
  //                  OptimizationRemarkEmitter &ORE)
  //     : F(&F), LI(&LI), SE(&SE), DT(&DT), TTI(&TTI), TLI(TLI),
  //       AA(&AA), AC(&AC), ORE(&ORE) {}
  // LoopSpawningImpl(Function &F,
  //                  function_ref<LoopInfo &(Function &)> GetLI,
  //                  function_ref<ScalarEvolution &(Function &)> GetSE,
  //                  function_ref<DominatorTree &(Function &)> GetDT,
  //                  OptimizationRemarkEmitter &ORE)
  //     : F(F), GetLI(GetLI), LI(nullptr), GetSE(GetSE), GetDT(GetDT),
  //       ORE(ORE)
  // {}
  LoopSpawningImpl(Function &F,
                   LoopInfo &LI,
                   ScalarEvolution &SE,
                   DominatorTree &DT,
                   AssumptionCache &AC,
                   OptimizationRemarkEmitter &ORE)
      : F(F), LI(LI), SE(SE), DT(DT), AC(AC), ORE(ORE) {}

  bool run();

private:
  void addTapirLoop(Loop *L, SmallVectorImpl<Loop *> &V);
  bool isTapirLoop(const Loop *L);
  bool processLoop(Loop *L);

  Function &F;
  // function_ref<LoopInfo &(Function &)> GetLI;
  LoopInfo &LI;
  // function_ref<ScalarEvolution &(Function &)> GetSE;
  // function_ref<DominatorTree &(Function &)> GetDT;
  ScalarEvolution &SE;
  DominatorTree &DT;
  // const TargetTransformInfo *TTI;
  // const TargetLibraryInfo *TLI;
  // AliasAnalysis *AA;
  AssumptionCache &AC;
  OptimizationRemarkEmitter &ORE;
};
} // end anonymous namespace

/// Canonicalize the induction variables in the loop.  Return the canonical
/// induction variable created or inserted by the scalar evolution expander.
PHINode* LoopOutline::canonicalizeIVs(Type *Ty) {
  Loop *L = OrigLoop;

  BasicBlock* Header = L->getHeader();
  Module* M = Header->getParent()->getParent();

  SCEVExpander Exp(SE, M->getDataLayout(), "ls");

  PHINode *CanonicalIV = Exp.getOrInsertCanonicalInductionVariable(L, Ty);
  DEBUG(dbgs() << "LS Canonical induction variable " << *CanonicalIV << "\n");

  SmallVector<WeakTrackingVH, 16> DeadInsts;
  Exp.replaceCongruentIVs(L, DT, DeadInsts);
  for (WeakTrackingVH V : DeadInsts) {
    DEBUG(dbgs() << "LS erasing dead inst " << *V << "\n");
    Instruction *I = cast<Instruction>(V);
    I->eraseFromParent();
  }

  return CanonicalIV;
}

/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
///
/// This method assumes that the loop has a single loop latch.
Value* LoopOutline::canonicalizeLoopLatch(PHINode *IV, Value *Limit) {
  Loop *L = OrigLoop;

  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition = Builder.CreateICmpULT(IV, Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  Builder.CreateCondBr(NewCondition, Header, ExitBlock);

  // Erase the old conditional branch.
  Value *OldCond = LatchBr->getCondition();
  LatchBr->eraseFromParent();
  if (!OldCond->hasNUsesOrMore(1))
    if (Instruction *OldCondInst = dyn_cast<Instruction>(OldCond))
      OldCondInst->eraseFromParent();

  return NewCondition;
}

/// Unlink the specified loop, and update analysis accordingly.  The heavy
/// lifting of deleting the loop is carried out by a run of LoopDeletion after
/// this pass.
void LoopOutline::unlinkLoop() {
  Loop *L = OrigLoop;

  // Get components of the old loop.
  BasicBlock *Preheader = L->getLoopPreheader();
  assert(Preheader && "Loop does not have a unique preheader.");
  BasicBlock *Latch = L->getLoopLatch();

  // Invalidate the analysis of the old loop.
  SE.forgetLoop(L);

  // Redirect the preheader to branch directly to loop exit.
  assert(1 == Preheader->getTerminator()->getNumSuccessors() &&
         "Preheader does not have a unique successor.");
  Preheader->getTerminator()->replaceUsesOfWith(L->getHeader(),
                                                ExitBlock);

  // Rewrite phis in the exit block to get their inputs from
  // the preheader instead of the exiting block.
  BasicBlock::iterator BI = ExitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    int j = P->getBasicBlockIndex(Latch);
    assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
    P->setIncomingBlock(j, Preheader);
    P->removeIncomingValue(Latch);
    ++BI;
  }

  // Rewrite phis in the header block to not receive an input from
  // the preheader.
  BI = L->getHeader()->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    P->removeIncomingValue(Preheader);
    ++BI;
  }
}

/// \brief Compute the grainsize of the loop, based on the limit.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
///
/// TODO: This method is the only method that depends on the CilkABI.
/// Generalize this method for other grainsize calculations and to query TLI.
Value* DACLoopSpawning::computeGrainsize(Value *Limit) {
  Loop *L = OrigLoop;

  Value *Grainsize;
  BasicBlock *Preheader = L->getLoopPreheader();
  assert(Preheader && "No Preheader found for loop.");

  IRBuilder<> Builder(Preheader->getTerminator());

  // Get 8 * workers
  Value *Workers8 = Builder.CreateIntCast(cilk::GetOrCreateWorker8(*Preheader->getParent()),
                                          Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) = (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, Workers8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       Workers8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  return Grainsize;
}

/// \brief Method to help convertLoopToDACIterSpawn convert the Tapir
/// loop cloned into function Helper to spawn its iterations in a
/// parallel divide-and-conquer fashion.
///
/// Example: Suppose that Helper contains the following Tapir loop:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     spawn { ... loop body ... };
///   } while (i++ < end);
///   sync;
/// }
///
/// Then this method transforms Helper into the following form:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
/// recur:
///   iter_t itercount = end - start;
///   if (itercount > grain) {
///     // Invariant: itercount >= 2
///     count_t miditer = start + itercount / 2;
///     spawn Helper(start, miditer, grain, ...);
///     start = miditer + 1;
///     goto recur;
///   }
///
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     ... Loop Body ...
///   } while (i++ < end);
///   sync;
/// }
///
void DACLoopSpawning::implementDACIterSpawnOnHelper(Function *Helper,
                                                    BasicBlock *Preheader,
                                                    BasicBlock *Header,
                                                    PHINode *CanonicalIV,
                                                    Argument *Limit,
                                                    Argument *Grainsize,
                                                    Instruction *SyncRegion,
                                                    DominatorTree *DT,
                                                    LoopInfo *LI,
                                                    bool CanonicalIVFlagNUW,
                                                    bool CanonicalIVFlagNSW) {
  // Serialize the cloned copy of the loop.
  assert(Preheader->getParent() == Helper &&
         "Preheader does not belong to helper function.");
  assert(Header->getParent() == Helper &&
         "Header does not belong to helper function.");
  assert(CanonicalIV->getParent() == Header &&
         "CanonicalIV does not belong to header");
  assert(isa<DetachInst>(Header->getTerminator()) &&
         "Cloned header is not terminated by a detach.");
  DetachInst *DI = dyn_cast<DetachInst>(Header->getTerminator());
  SerializeDetachedCFG(DI, DT);

  // Convert the cloned loop into the strip-mined loop body.

  BasicBlock *DACHead = Preheader;
  if (&(Helper->getEntryBlock()) == Preheader)
    // Split the entry block.  We'll want to create a backedge into
    // the split block later.
    DACHead = SplitBlock(Preheader, Preheader->getTerminator(), DT, LI);

  BasicBlock *RecurHead, *RecurDet, *RecurCont;
  Value *IterCount;
  Value *CanonicalIVInput;
  PHINode *CanonicalIVStart;
  {
    Instruction *PreheaderOrigFront = &(DACHead->front());
    IRBuilder<> Builder(PreheaderOrigFront);
    // Create branch based on grainsize.
    DEBUG(dbgs() << "LS CanonicalIV: " << *CanonicalIV << "\n");
    CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(DACHead);
    CanonicalIVStart = Builder.CreatePHI(CanonicalIV->getType(), 2,
                                         CanonicalIV->getName()+".dac");
    CanonicalIVInput->replaceAllUsesWith(CanonicalIVStart);
    IterCount = Builder.CreateSub(Limit, CanonicalIVStart,
                                  "itercount");
    Value *IterCountCmp = Builder.CreateICmpUGT(IterCount, Grainsize);
    TerminatorInst *RecurTerm =
      SplitBlockAndInsertIfThen(IterCountCmp, PreheaderOrigFront,
                                /*Unreachable=*/false,
                                /*BranchWeights=*/nullptr,
                                DT);
    RecurHead = RecurTerm->getParent();
    // Create skeleton of divide-and-conquer recursion:
    // DACHead -> RecurHead -> RecurDet -> RecurCont -> DACHead
    RecurDet = SplitBlock(RecurHead, RecurHead->getTerminator(),
                          DT, LI);
    RecurCont = SplitBlock(RecurDet, RecurDet->getTerminator(),
                           DT, LI);
    RecurCont->getTerminator()->replaceUsesOfWith(RecurTerm->getSuccessor(0),
                                                  DACHead);
  }

  // Compute mid iteration in RecurHead.
  Value *MidIter, *MidIterPlusOne;
  {
    IRBuilder<> Builder(&(RecurHead->front()));
    MidIter = Builder.CreateAdd(CanonicalIVStart,
                                Builder.CreateLShr(IterCount, 1,
                                                   "halfcount"),
                                "miditer",
                                CanonicalIVFlagNUW, CanonicalIVFlagNSW);
  }

  // Create recursive call in RecurDet.
  {
    // Create input array for recursive call.
    IRBuilder<> Builder(&(RecurDet->front()));
    SetVector<Value*> RecurInputs;
    Function::arg_iterator AI = Helper->arg_begin();
    assert(cast<Argument>(CanonicalIVInput) == &*AI &&
           "First argument does not match original input to canonical IV.");
    RecurInputs.insert(CanonicalIVStart);
    ++AI;
    assert(Limit == &*AI &&
           "Second argument does not match original input to the loop limit.");
    RecurInputs.insert(MidIter);
    ++AI;
    for (Function::arg_iterator AE = Helper->arg_end();
         AI != AE;  ++AI)
        RecurInputs.insert(&*AI);
    // RecurInputs.insert(CanonicalIVStart);
    // // for (PHINode *IV : IVs)
    // //   RecurInputs.insert(DACStart[IV]);
    // RecurInputs.insert(Limit);
    // RecurInputs.insert(Grainsize);
    // for (Value *V : BodyInputs)
    //   RecurInputs.insert(VMap[V]);
    DEBUG({
        dbgs() << "RecurInputs: ";
        for (Value *Input : RecurInputs)
          dbgs() << *Input << ", ";
        dbgs() << "\n";
      });

    // Create call instruction.
    CallInst *RecurCall = Builder.CreateCall(Helper, RecurInputs.getArrayRef());
    RecurCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // Use a fast calling convention for the helper.
    RecurCall->setCallingConv(CallingConv::Fast);
    // RecurCall->setCallingConv(Helper->getCallingConv());
    // // Update CG graph with the recursive call we just added.
    // CG[Helper]->addCalledFunction(RecurCall, CG[Helper]);
  }

  // Set up continuation of detached recursive call.  We effectively
  // inline this tail call automatically.
  {
    IRBuilder<> Builder(&(RecurCont->front()));
    MidIterPlusOne = Builder.CreateAdd(MidIter,
                                       ConstantInt::get(Limit->getType(), 1),
                                       "miditerplusone",
                                       CanonicalIVFlagNUW,
                                       CanonicalIVFlagNSW);
  }

  // Finish setup of new phi node for canonical IV.
  {
    CanonicalIVStart->addIncoming(CanonicalIVInput, Preheader);
    CanonicalIVStart->addIncoming(MidIterPlusOne, RecurCont);
  }

  /// Make the recursive DAC parallel.
  {
    IRBuilder<> Builder(RecurHead->getTerminator());
    // Create the detach.
    DetachInst *DI = Builder.CreateDetach(RecurDet, RecurCont, SyncRegion);
    DI->setDebugLoc(Header->getTerminator()->getDebugLoc());
    RecurHead->getTerminator()->eraseFromParent();
    // Create the reattach.
    Builder.SetInsertPoint(RecurDet->getTerminator());
    ReattachInst *RI = Builder.CreateReattach(RecurCont, SyncRegion);
    RI->setDebugLoc(Header->getTerminator()->getDebugLoc());
    RecurDet->getTerminator()->eraseFromParent();
  }
}

/// Helper routine to get all exit blocks of a loop that are unreachable.
static void getEHExits(Loop *L, const BasicBlock *DesignatedExitBlock,
                       SmallVectorImpl<BasicBlock *> &EHExits) {
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  SmallVector<BasicBlock *, 4> WorkList;
  for (BasicBlock *Exit : ExitBlocks) {
    if (Exit == DesignatedExitBlock) continue;
    EHExits.push_back(Exit);
    WorkList.push_back(Exit);
  }

  // Traverse the CFG from these frontier blocks to find all blocks involved in
  // exception-handling exit code.
  SmallPtrSet<BasicBlock *, 4> Visited;
  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    // Check that the exception handling blocks do not reenter the loop.
    assert(!L->contains(BB) &&
           "Exception handling blocks re-enter loop.");

    for (BasicBlock *Succ : successors(BB)) {
      EHExits.push_back(Succ);
      WorkList.push_back(Succ);
    }
  }
}

/// Top-level call to convert loop to spawn its iterations in a
/// divide-and-conquer fashion.
bool DACLoopSpawning::processLoop() {
  Loop *L = OrigLoop;

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();

  DEBUG({
      LoopBlocksDFS DFS(L);
      DFS.perform(LI);
      dbgs() << "Blocks in loop (from DFS):\n";
      for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO()))
        dbgs() << *BB;
    });

  using namespace ore;

  // Check that this loop has a valid exit block after the latch.
  if (!ExitBlock) {
    DEBUG(dbgs() << "LS loop does not contain valid exit block after latch.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "InvalidLatchExit",
                                        L->getStartLoc(),
                                        Header)
             << "invalid latch exit");
    return false;
  }

  // Get special exits from this loop.
  SmallVector<BasicBlock *, 4> EHExits;
  getEHExits(L, ExitBlock, EHExits);

  // Check the exit blocks of the loop.
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  for (const BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    if (Exit->isLandingPad()) {
      DEBUG({
          const LandingPadInst *LPI = Exit->getLandingPadInst();
          dbgs() << "landing pad found: " << *LPI << "\n";
          for (const User *U : LPI->users())
            dbgs() << "\tuser " << *U << "\n";
        });
    }
  }
  SmallPtrSet<BasicBlock *, 4> HandledExits;
  for (BasicBlock *BB : EHExits)
    HandledExits.insert(BB);
  for (BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    if (!HandledExits.count(Exit)) {
      DEBUG(dbgs() << "LS loop contains a bad exit block " << *Exit);
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "BadExit",
                                          L->getStartLoc(),
                                          Header)
               << "bad exit block found");
      return false;
    }
  }

  Function *F = Header->getParent();
  Module* M = F->getParent();

  DEBUG(dbgs() << "LS loop header:" << *Header);
  DEBUG(dbgs() << "LS loop latch:" << *Latch);
  DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");

  /// Get loop limit.
  const SCEV *Limit = SE.getExitCount(L, Latch);
  DEBUG(dbgs() << "LS Loop limit: " << *Limit << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getExitCount(L, Latch);
  // DEBUG(dbgs() << "LS predicated loop limit: " << *PLimit << "\n");
  // emitAnalysis(LoopSpawningReport()
  //              << "computed loop limit " << *Limit << "\n");
  if (SE.getCouldNotCompute() == Limit) {
    DEBUG(dbgs() << "SE could not compute loop limit.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "UnknownLoopLimit",
                                        L->getStartLoc(),
                                        Header)
             << "could not compute limit");
    return false;
  }
  // ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "LoopLimit", L->getStartLoc(),
  //                                     Header)
  //          << "loop limit: " << NV("Limit", Limit));
  /// Clean up the loop's induction variables.
  PHINode *CanonicalIV = canonicalizeIVs(Limit->getType());
  if (!CanonicalIV) {
    DEBUG(dbgs() << "Could not get canonical IV.\n");
    // emitAnalysis(LoopSpawningReport()
    //              << "Could not get a canonical IV.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoCanonicalIV",
                                        L->getStartLoc(),
                                        Header)
             << "could not find or create canonical IV");
    return false;
  }
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

  // Remove all IV's other than CanonicalIV.
  // First, check that we can do this.
  bool CanRemoveIVs = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    if (CanonicalIV == PN) continue;
    // dbgs() << "IV " << *PN;
    const SCEV *S = SE.getSCEV(PN);
    // dbgs() << " SCEV " << *S << "\n";
    if (SE.getCouldNotCompute() == S) {
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Could not compute the scalar evolution.\n");
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoSCEV", PN)
               << "could not compute scalar evolution of "
               << NV("PHINode", PN));
      CanRemoveIVs = false;
    }
  }

  if (!CanRemoveIVs) {
    DEBUG(dbgs() << "Could not compute scalar evolutions for all IV's.\n");
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  // We now have everything we need to extract the loop.  It's time to
  // do some surgery.

  SCEVExpander Exp(SE, M->getDataLayout(), "ls");

  // Remove the IV's (other than CanonicalIV) and replace them with
  // their stronger forms.
  //
  // TODO?: We can probably adapt this loop->DAC process such that we
  // don't require all IV's to be canonical.
  {
    SmallVector<PHINode*, 8> IVsToRemove;
    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV) continue;
      const SCEV *S = SE.getSCEV(PN);
      DEBUG(dbgs() << "Removing the IV " << *PN << " (" << *S << ")\n");
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "RemoveIV", PN)
               << "removing the IV "
               << NV("PHINode", PN));
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
      PN->replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(PN);
    }
    for (PHINode *PN : IVsToRemove)
      PN->eraseFromParent();
  }

  // All remaining IV's should be canonical.  Collect them.
  //
  // TODO?: We can probably adapt this loop->DAC process such that we
  // don't require all IV's to be canonical.
  SmallVector<PHINode*, 8> IVs;
  bool AllCanonical = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    DEBUG({
        const SCEVAddRecExpr *PNSCEV =
          dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
        assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr");
        assert(PNSCEV->getStart()->isZero() &&
               "PHINode SCEV does not start at 0");
        dbgs() << "LS step recurrence for SCEV " << *PNSCEV << " is "
               << *(PNSCEV->getStepRecurrence(SE)) << "\n";
        assert(PNSCEV->getStepRecurrence(SE)->isOne() &&
               "PHINode SCEV step is not 1");
      });
    if (ConstantInt *C =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Preheader))) {
      if (C->isZero()) {
        DEBUG({
            if (PN != CanonicalIV) {
              const SCEVAddRecExpr *PNSCEV =
                dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
              dbgs() << "Saving the canonical IV " << *PN << " (" << *PNSCEV << ")\n";
            }
          });
        if (PN != CanonicalIV)
          ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "SaveIV", PN)
                   << "saving the canonical the IV "
                   << NV("PHINode", PN));
        IVs.push_back(PN);
      }
    } else {
      AllCanonical = false;
      DEBUG(dbgs() << "Remaining non-canonical PHI Node found: " << *PN <<
            "\n");
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Found a remaining non-canonical IV.\n");
      ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "NonCanonicalIV", PN)
               << "found a remaining noncanonical IV");
    }
  }
  if (!AllCanonical)
    return false;

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      Preheader->getTerminator());
  DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

  // Canonicalize the loop latch.
  assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                        CanonicalSCEV, Limit) &&
         "Loop backedge is not guarded by canonical comparison with limit.");
  Value *NewCond = canonicalizeLoopLatch(CanonicalIV, LimitVar);

  // Insert computation of grainsize into the Preheader.
  // For debugging:
  // Value *GrainVar = ConstantInt::get(Limit->getType(), 2);
  Value *GrainVar = computeGrainsize(LimitVar);
  DEBUG(dbgs() << "GrainVar: " << *GrainVar << "\n");
  // emitAnalysis(LoopSpawningReport()
  //              << "grainsize value " << *GrainVar << "\n");
  // ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "UsingGrainsize",
  //                                     L->getStartLoc(), Header)
  //          << "grainsize: " << NV("Grainsize", GrainVar));

  /// Clone the loop into a new function.

  // Get the inputs and outputs for the Loop blocks.
  SetVector<Value *> Inputs, Outputs;
  SetVector<Value *> BodyInputs, BodyOutputs;
  ValueToValueMapTy VMap, InputMap;
  std::vector<BasicBlock *> LoopBlocks;
  SmallPtrSet<BasicBlock *, 4> ExitsToSplit;

  // Get the sync region containing this Tapir loop.
  const Instruction *InputSyncRegion;
  {
    const DetachInst *DI = cast<DetachInst>(Header->getTerminator());
    InputSyncRegion = cast<Instruction>(DI->getSyncRegion());
  }

  // Add start iteration, end iteration, and grainsize to inputs.
  {
    LoopBlocks = L->getBlocks();
    // // Add exit blocks terminated by unreachable.  There should not be any other
    // // exit blocks in the loop.
    // SmallSet<BasicBlock *, 4> UnreachableExits;
    // for (BasicBlock *Exit : ExitBlocks) {
    //   if (Exit == ExitBlock) continue;
    //   assert(isa<UnreachableInst>(Exit->getTerminator()) &&
    //          "Found problematic exit block.");
    //   UnreachableExits.insert(Exit);
    // }

    // Add unreachable and exception-handling exits to the set of loop blocks to
    // clone.
    DEBUG({
        dbgs() << "Handled exits of loop:";
        for (BasicBlock *HE : HandledExits)
          dbgs() << *HE;
        dbgs() << "\n";
      });
    for (BasicBlock *HE : HandledExits)
      LoopBlocks.push_back(HE);
    {
      const DetachInst *DI = cast<DetachInst>(Header->getTerminator());
      BasicBlockEdge DetachEdge(Header, DI->getDetached());
      for (BasicBlock *HE : HandledExits)
        if (!DT || !DT->dominates(DetachEdge, HE))
          ExitsToSplit.insert(HE);
      DEBUG({
          dbgs() << "Loop exits to split:";
          for (BasicBlock *ETS : ExitsToSplit)
            dbgs() << *ETS;
          dbgs() << "\n";
        });
    }

    // DEBUG({
    //     dbgs() << "LoopBlocks: ";
    //     for (BasicBlock *LB : LoopBlocks)
    //       dbgs() << LB->getName() << "("
    //              << *(LB->getTerminator()) << "), ";
    //     dbgs() << "\n";
    //   });

    // Get the inputs and outputs for the loop body.
    {
      // CodeExtractor Ext(LoopBlocks, DT);
      // Ext.findInputsOutputs(BodyInputs, BodyOutputs);
      SmallPtrSet<BasicBlock *, 32> Blocks;
      for (BasicBlock *BB : LoopBlocks)
        Blocks.insert(BB);
      findInputsOutputs(Blocks, BodyInputs, BodyOutputs, &ExitsToSplit);
    }

    // Add argument for start of CanonicalIV.
    DEBUG({
        Value *CanonicalIVInput =
          CanonicalIV->getIncomingValueForBlock(Preheader);
        // CanonicalIVInput should be the constant 0.
        assert(isa<Constant>(CanonicalIVInput) &&
               "Input to canonical IV from preheader is not constant.");
      });
    Argument *StartArg = new Argument(CanonicalIV->getType(),
                                      CanonicalIV->getName()+".start");
    Inputs.insert(StartArg);
    InputMap[CanonicalIV] = StartArg;

    // Add argument for end.
    //
    // In the general case, the loop limit is the result of some computation
    // that the pass added to the loop's preheader.  In this case, the variable
    // storing the loop limit is used exactly once, in the canonicalized loop
    // latch.  In this case, the pass wants to prevent outlining from passing
    // the loop-limit variable as an arbitrary argument to the outlined
    // function.  Hence, this pass adds the loop-limit variable as an argument
    // manually.
    //
    // There are two special cases to consider: the loop limit is a constant, or
    // the loop limit is used elsewhere within the loop.  To handle these two
    // cases, this pass adds an explict argument for the end of the loop, to
    // supports the subsequent transformation to using recursive
    // divide-and-conquer.  After the loop is outlined, this pass will rewrite
    // the latch in the outlined loop to use this explicit argument.
    // Furthermore, this pass does not prevent outliner from recognizing the
    // loop limit as a potential argument to the function.
    if (isa<Constant>(LimitVar) || !LimitVar->hasOneUse()) {
      Argument *EndArg = new Argument(LimitVar->getType(), "end");
      Inputs.insert(EndArg);
      InputMap[LimitVar] = EndArg;
    } else {
      // If the limit var is not constant and has exactly one use, then the
      // limit var is the result of some nontrivial computation, and that one
      // use is the new condition inserted.
      Inputs.insert(LimitVar);
      InputMap[LimitVar] = LimitVar;
    }

    // Add argument for grainsize.
    if (isa<Constant>(GrainVar)) {
      Argument *GrainArg = new Argument(GrainVar->getType(), "grainsize");
      Inputs.insert(GrainArg);
      InputMap[GrainVar] = GrainArg;
    } else {
      Inputs.insert(GrainVar);
      InputMap[GrainVar] = GrainVar;
    }

    // Put all of the inputs together, and clear redundant inputs from
    // the set for the loop body.
    SmallVector<Value *, 8> BodyInputsToRemove;
    for (Value *V : BodyInputs)
      if (V == InputSyncRegion)
        BodyInputsToRemove.push_back(V);
      else if (!Inputs.count(V))
        Inputs.insert(V);
      else
        BodyInputsToRemove.push_back(V);
    for (Value *V : BodyInputsToRemove)
      BodyInputs.remove(V);
    DEBUG({
        for (Value *V : BodyInputs)
          dbgs() << "Remaining body input: " << *V << "\n";
      });
    for (Value *V : BodyOutputs)
      dbgs() << "EL output: " << *V << "\n";
    assert(0 == BodyOutputs.size() &&
           "All results from parallel loop should be passed by memory already.");
  }
  DEBUG({
      for (Value *V : Inputs)
        dbgs() << "EL input: " << *V << "\n";
      for (Value *V : Outputs)
        dbgs() << "EL output: " << *V << "\n";
    });

  // Clone the loop blocks into a new helper function.
  Function *Helper;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.

    // LowerDbgDeclare(*(Header->getParent()));

    Helper = CreateHelper(Inputs, Outputs, LoopBlocks,
                          Header, Preheader, ExitBlock,
                          VMap, M,
                          F->getSubprogram() != nullptr, Returns, ".ls",
                          &ExitsToSplit, InputSyncRegion,
                          nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning loop.");

    // Use a fast calling convention for the helper.
    Helper->setCallingConv(CallingConv::Fast);
    // Helper->setCallingConv(Header->getParent()->getCallingConv());
  }

  // Add a sync to the helper's return.
  BasicBlock *HelperHeader = cast<BasicBlock>(VMap[Header]);
  {
    BasicBlock *HelperExit = cast<BasicBlock>(VMap[ExitBlock]);
    assert(isa<ReturnInst>(HelperExit->getTerminator()));
    BasicBlock *NewHelperExit = SplitBlock(HelperExit,
                                           HelperExit->getTerminator(),
                                           DT, LI);
    IRBuilder<> Builder(&(HelperExit->front()));
    SyncInst *NewSync = Builder.CreateSync(
        NewHelperExit,
        cast<Instruction>(VMap[InputSyncRegion]));
    // Set debug info of new sync to match that of terminator of the header of
    // the cloned loop.
    NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());
    HelperExit->getTerminator()->eraseFromParent();
  }

  // // Add syncs to the helper's cloned resume blocks.
  // for (BasicBlock *BB : Resumes) {
  //   BasicBlock *HelperResume = cast<BasicBlock>(VMap[BB]);
  //   assert(isa<ResumeInst>(HelperResume->getTerminator()));
  //   BasicBlock *NewHelperResume = SplitBlock(HelperResume,
  //                                            HelperResume->getTerminator(),
  //                                            DT, LI);
  //   IRBuilder<> Builder(&(HelperResume->front()));
  //   SyncInst *NewSync = Builder.CreateSync(NewHelperResume);
  //   // Set debug info of new sync to match that of terminator of the header of
  //   // the cloned loop.
  //   NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());
  //   HelperResume->getTerminator()->eraseFromParent();
  // }

  BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
  PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

  // Rewrite the cloned IV's to start at the start iteration argument.
  {
    // Rewrite clone of canonical IV to start at the start iteration
    // argument.
    Argument *NewCanonicalIVStart = cast<Argument>(VMap[InputMap[CanonicalIV]]);
    {
      int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned canonical IV does not inherit a constant value from cloned preheader.");
      NewCanonicalIV->setIncomingValue(NewPreheaderIdx, NewCanonicalIVStart);
    }

    // Rewrite other cloned IV's to start at their value at the start
    // iteration.
    const SCEV *StartIterSCEV = SE.getSCEV(NewCanonicalIVStart);
    DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
    for (PHINode *IV : IVs) {
      if (CanonicalIV == IV) continue;

      // Get the value of the IV at the start iteration.
      DEBUG(dbgs() << "IV " << *IV);
      const SCEV *IVSCEV = SE.getSCEV(IV);
      DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
      const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
      const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
      DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
            " to " << *IVAtIter << "\n");

      // NOTE: Expanded code should not refer to other IV's.
      Value *IVStart = Exp.expandCodeFor(IVAtIter, IVAtIter->getType(),
                                         NewPreheader->getTerminator());


      // Set the value that the cloned IV inherits from the cloned preheader.
      PHINode *NewIV = cast<PHINode>(VMap[IV]);
      int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned IV does not inherit a constant value from cloned preheader.");
      NewIV->setIncomingValue(NewPreheaderIdx, IVStart);
    }

    // Remap the newly added instructions in the new preheader to use
    // values local to the helper.
    for (Instruction &II : *NewPreheader)
      RemapInstruction(&II, VMap, RF_IgnoreMissingLocals,
                       /*TypeMapper=*/nullptr, /*Materializer=*/nullptr);
  }

  // The loop has been outlined by this point.  To handle the special cases
  // where the loop limit was constant or used elsewhere within the loop, this
  // pass rewrites the outlined loop-latch condition to use the explicit
  // end-iteration argument.
  if (isa<Constant>(LimitVar) || !LimitVar->hasOneUse()) {
    CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
    assert(((isa<Constant>(LimitVar) &&
             HelperCond->getOperand(1) == LimitVar) ||
            (!LimitVar->hasOneUse() &&
             HelperCond->getOperand(1) == VMap[LimitVar])) &&
           "Unexpected condition in loop latch.");
    IRBuilder<> Builder(HelperCond);
    Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                 VMap[InputMap[LimitVar]]);
    HelperCond->replaceAllUsesWith(NewHelperCond);
    HelperCond->eraseFromParent();
    DEBUG(dbgs() << "Rewritten Latch: " <<
          *(cast<Instruction>(NewHelperCond)->getParent()));
  }

  // DEBUGGING: Simply serialize the cloned loop.
  // BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  // SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
  implementDACIterSpawnOnHelper(Helper, NewPreheader,
                                cast<BasicBlock>(VMap[Header]),
                                cast<PHINode>(VMap[CanonicalIV]),
                                cast<Argument>(VMap[InputMap[LimitVar]]),
                                cast<Argument>(VMap[InputMap[GrainVar]]),
                                cast<Instruction>(VMap[InputSyncRegion]),
                                /*DT=*/nullptr, /*LI=*/nullptr,
                                CanonicalSCEV->getNoWrapFlags(SCEV::FlagNUW),
                                CanonicalSCEV->getNoWrapFlags(SCEV::FlagNSW));

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Update allocas in cloned loop body.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (pred_iterator PI = pred_begin(Latch), PE = pred_end(Latch);
         PI != PE; ++PI) {
      BasicBlock *Pred = *PI;
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (L->contains(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }
    // The cloned loop should be serialized by this point.
    BasicBlock *ClonedLoopBodyEntry =
      cast<BasicBlock>(VMap[Header])->getSingleSuccessor();
    assert(ClonedLoopBodyEntry &&
           "Head of cloned loop body has multiple successors.");
    bool ContainsDynamicAllocas =
      MoveStaticAllocasInBlock(&Helper->getEntryBlock(), ClonedLoopBodyEntry,
                               ReattachPoints);

    // If the cloned loop contained dynamic alloca instructions, wrap the cloned
    // loop with llvm.stacksave/llvm.stackrestore intrinsics.
    if (ContainsDynamicAllocas) {
      Module *M = Helper->getParent();
      // Get the two intrinsics we care about.
      Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
      Function *StackRestore =
        Intrinsic::getDeclaration(M,Intrinsic::stackrestore);

      // Insert the llvm.stacksave.
      CallInst *SavedPtr = IRBuilder<>(&*ClonedLoopBodyEntry,
                                       ClonedLoopBodyEntry->begin())
                             .CreateCall(StackSave, {}, "savedstack");

      // Insert a call to llvm.stackrestore before the reattaches in the
      // original Tapir loop.
      for (Instruction *ExitPoint : ReattachPoints)
        IRBuilder<>(ExitPoint).CreateCall(StackRestore, SavedPtr);
    }
  }

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(F, Inputs, VMap,
                          Preheader->getTerminator(), AC, DT);

  // Add call to new helper function in original function.
  {
    // Setup arguments for call.
    SmallVector<Value *, 4> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.push_back(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add loop limit.
    TopCallArgs.push_back(LimitVar);
    // Add grainsize.
    TopCallArgs.push_back(GrainVar);
    // Add the rest of the arguments.
    for (Value *V : BodyInputs)
      TopCallArgs.push_back(V);
    DEBUG({
        for (Value *TCArg : TopCallArgs)
          dbgs() << "Top call arg: " << *TCArg << "\n";
      });

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());
    CallInst *TopCall = Builder.CreateCall(Helper,
                                           ArrayRef<Value *>(TopCallArgs));

    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    // TopCall->setCallingConv(Helper->getCallingConv());
    TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // // Update CG graph with the call we just added.
    // CG[F]->addCalledFunction(TopCall, CG[Helper]);
  }

  // Remove sync of loop in parent.
  {
    // Get the sync region for this loop's detached iterations.
    DetachInst *HeadDetach = cast<DetachInst>(Header->getTerminator());
    Value *SyncRegion = HeadDetach->getSyncRegion();
    // Check the Tapir instructions contained in this sync region.  Look for a
    // single sync instruction among those Tapir instructions.  Meanwhile,
    // verify that the only detach instruction in this sync region is the detach
    // in theloop header.  If these conditions are met, then we assume that the
    // sync applies to this loop.  Otherwise, something more complicated is
    // going on, and we give up.
    SyncInst *LoopSync = nullptr;
    bool SingleSyncJustForLoop = true;
    for (User *U : SyncRegion->users()) {
      // Skip the detach in the loop header.
      if (HeadDetach == U) continue;
      // Remember the first sync instruction we find.  If we find multiple sync
      // instructions, then something nontrivial is going on.
      if (SyncInst *SI = dyn_cast<SyncInst>(U)) {
        if (!LoopSync)
          LoopSync = SI;
        else
          SingleSyncJustForLoop = false;
      }
      // If we find a detach instruction that is not the loop header's, then
      // something nontrivial is going on.
      if (isa<DetachInst>(U))
        SingleSyncJustForLoop = false;
    }
    if (LoopSync && SingleSyncJustForLoop)
      // Replace the sync with a branch.
      ReplaceInstWithInst(LoopSync,
                          BranchInst::Create(LoopSync->getSuccessor(0)));
    else if (!LoopSync)
      DEBUG(dbgs() << "No sync found for this loop.");
    else
      DEBUG(dbgs() << "No single sync found that only affects this loop.");
  }

  ++LoopsConvertedToDAC;

  unlinkLoop();

  return Helper;
}

/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
///
/// This method assumes that the loop has a single loop latch.
Value* CilkABILoopSpawning::canonicalizeLoopLatch(PHINode *IV, Value *Limit) {
  Loop *L = OrigLoop;

  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition =
    Builder.CreateICmpULT(Builder.CreateAdd(IV,
                                            ConstantInt::get(IV->getType(), 1)),
                          Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  Builder.CreateCondBr(NewCondition, Header, ExitBlock);

  // Erase the old conditional branch.
  LatchBr->eraseFromParent();

  return NewCondition;
}

/// Top-level call to convert a Tapir loop to be processed using an appropriate
/// Cilk ABI call.
bool CilkABILoopSpawning::processLoop() {
  Loop *L = OrigLoop;

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();

  using namespace ore;

  // Check the exit blocks of the loop.
  if (!ExitBlock) {
    DEBUG(dbgs() << "LS loop does not contain valid exit block after latch.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "InvalidLatchExit",
                                        L->getStartLoc(),
                                        Header)
             << "invalid latch exit");
    return false;
  }

  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  for (const BasicBlock *Exit : ExitBlocks) {
    if (Exit == ExitBlock) continue;
    if (!isa<UnreachableInst>(Exit->getTerminator())) {
      DEBUG(dbgs() << "LS loop contains a bad exit block " << *Exit);
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "BadExit",
                                          L->getStartLoc(),
                                          Header)
               << "bad exit block found");
      return false;
    }
  }

  Function *F = Header->getParent();
  Module* M = F->getParent();

  DEBUG(dbgs() << "LS loop header:" << *Header);
  DEBUG(dbgs() << "LS loop latch:" << *Latch);

  // DEBUG(dbgs() << "LS SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
  // DEBUG(dbgs() << "LS SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");

  /// Get loop limit.
  const SCEV *BETC = SE.getExitCount(L, Latch);
  const SCEV *Limit = SE.getAddExpr(BETC, SE.getOne(BETC->getType()));
  DEBUG(dbgs() << "LS Loop limit: " << *Limit << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getExitCount(L, Latch);
  // DEBUG(dbgs() << "LS predicated loop limit: " << *PLimit << "\n");
  // emitAnalysis(LoopSpawningReport()
  //              << "computed loop limit " << *Limit << "\n");
  if (SE.getCouldNotCompute() == Limit) {
    DEBUG(dbgs() << "SE could not compute loop limit.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "UnknownLoopLimit",
                                        L->getStartLoc(),
                                        Header)
             << "could not compute limit");
    return false;
  }
  // ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "LoopLimit", L->getStartLoc(),
  //                                     Header)
  //          << "loop limit: " << NV("Limit", Limit));
  /// Clean up the loop's induction variables.
  PHINode *CanonicalIV = canonicalizeIVs(Limit->getType());
  if (!CanonicalIV) {
    DEBUG(dbgs() << "Could not get canonical IV.\n");
    // emitAnalysis(LoopSpawningReport()
    //              << "Could not get a canonical IV.\n");
    ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoCanonicalIV",
                                        L->getStartLoc(),
                                        Header)
             << "could not find or create canonical IV");
    return false;
  }
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

  // Remove all IV's other can CanonicalIV.
  // First, check that we can do this.
  bool CanRemoveIVs = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    if (CanonicalIV == PN) continue;
    // dbgs() << "IV " << *PN;
    const SCEV *S = SE.getSCEV(PN);
    // dbgs() << " SCEV " << *S << "\n";
    if (SE.getCouldNotCompute() == S) {
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Could not compute the scalar evolution.\n");
      ORE.emit(OptimizationRemarkAnalysis(LS_NAME, "NoSCEV", PN)
               << "could not compute scalar evolution of "
               << NV("PHINode", PN));
      CanRemoveIVs = false;
    }
  }

  if (!CanRemoveIVs) {
    DEBUG(dbgs() << "Could not compute scalar evolutions for all IV's.\n");
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  // We now have everything we need to extract the loop.  It's time to
  // do some surgery.

  SCEVExpander Exp(SE, M->getDataLayout(), "ls");

  // Remove the IV's (other than CanonicalIV) and replace them with
  // their stronger forms.
  //
  // TODO?: We can probably adapt this process such that we don't require all
  // IV's to be canonical.
  {
    SmallVector<PHINode*, 8> IVsToRemove;
    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV) continue;
      const SCEV *S = SE.getSCEV(PN);
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
      PN->replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(PN);
    }
    for (PHINode *PN : IVsToRemove)
      PN->eraseFromParent();
  }

  // All remaining IV's should be canonical.  Collect them.
  //
  // TODO?: We can probably adapt this process such that we don't require all
  // IV's to be canonical.
  SmallVector<PHINode*, 8> IVs;
  bool AllCanonical = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    DEBUG({
        const SCEVAddRecExpr *PNSCEV =
          dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
        assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr");
        assert(PNSCEV->getStart()->isZero() &&
               "PHINode SCEV does not start at 0");
        dbgs() << "LS step recurrence for SCEV " << *PNSCEV << " is "
               << *(PNSCEV->getStepRecurrence(SE)) << "\n";
        assert(PNSCEV->getStepRecurrence(SE)->isOne() &&
               "PHINode SCEV step is not 1");
      });
    if (ConstantInt *C =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Preheader))) {
      if (C->isZero())
        IVs.push_back(PN);
    } else {
      AllCanonical = false;
      DEBUG(dbgs() << "Remaining non-canonical PHI Node found: " << *PN << "\n");
      // emitAnalysis(LoopSpawningReport(PN)
      //              << "Found a remaining non-canonical IV.\n");
      ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "NonCanonicalIV", PN)
               << "found a remaining noncanonical IV");
    }
  }
  if (!AllCanonical)
    return false;

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      Preheader->getTerminator());
  DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

  // Canonicalize the loop latch.
  Value *NewCond = canonicalizeLoopLatch(CanonicalIV, LimitVar);

  /// Clone the loop into a new function.

  // Get the inputs and outputs for the Loop blocks.
  SetVector<Value*> Inputs, Outputs;
  SetVector<Value*> BodyInputs, BodyOutputs;
  ValueToValueMapTy VMap, InputMap;
  std::vector<BasicBlock *> LoopBlocks;
  AllocaInst* closure;
  // Add start iteration, end iteration, and grainsize to inputs.
  {
    LoopBlocks = L->getBlocks();
    // // Add exit blocks terminated by unreachable.  There should not be any other
    // // exit blocks in the loop.
    // SmallSet<BasicBlock *, 4> UnreachableExits;
    // for (BasicBlock *Exit : ExitBlocks) {
    //   if (Exit == ExitBlock) continue;
    //   assert(isa<UnreachableInst>(Exit->getTerminator()) &&
    //          "Found problematic exit block.");
    //   UnreachableExits.insert(Exit);
    // }

    // // Add unreachable and exception-handling exits to the set of loop blocks to
    // // clone.
    // for (BasicBlock *BB : UnreachableExits)
    //   LoopBlocks.push_back(BB);
    // for (BasicBlock *BB : EHExits)
    //   LoopBlocks.push_back(BB);

    // DEBUG({
    //     dbgs() << "LoopBlocks: ";
    //     for (BasicBlock *LB : LoopBlocks)
    //       dbgs() << LB->getName() << "("
    //              << *(LB->getTerminator()) << "), ";
    //     dbgs() << "\n";
    //   });

    // Get the inputs and outputs for the loop body.
    {
      // CodeExtractor Ext(LoopBlocks, DT);
      // Ext.findInputsOutputs(BodyInputs, BodyOutputs);
      SmallPtrSet<BasicBlock *, 32> Blocks;
      for (BasicBlock *BB : LoopBlocks)
        Blocks.insert(BB);
      findInputsOutputs(Blocks, BodyInputs, BodyOutputs);
    }

    // Add argument for start of CanonicalIV.
    DEBUG({
        Value *CanonicalIVInput =
          CanonicalIV->getIncomingValueForBlock(Preheader);
        // CanonicalIVInput should be the constant 0.
        assert(isa<Constant>(CanonicalIVInput) &&
               "Input to canonical IV from preheader is not constant.");
      });
    Argument *StartArg = new Argument(CanonicalIV->getType(),
                                      CanonicalIV->getName()+".start");
    Inputs.insert(StartArg);
    InputMap[CanonicalIV] = StartArg;

    // Add argument for end.
    Value* ea;
    if (isa<Constant>(LimitVar)) {
      Argument *EndArg = new Argument(LimitVar->getType(), "end");
      Inputs.insert(EndArg);
      ea = InputMap[LimitVar] = EndArg;
    } else {
      Inputs.insert(LimitVar);
      ea = InputMap[LimitVar] = LimitVar;
    }

    // Put all of the inputs together, and clear redundant inputs from
    // the set for the loop body.
    SmallVector<Value*, 8> BodyInputsToRemove;
    SmallVector<Value*, 8> StructInputs;
    SmallVector<Type*, 8> StructIT;
    for (Value *V : BodyInputs) {
      if (!Inputs.count(V)) {
        StructInputs.push_back(V);
        StructIT.push_back(V->getType());
      }
      else
        BodyInputsToRemove.push_back(V);
    }
    StructType* ST = StructType::create(StructIT);
    IRBuilder<> B(L->getLoopPreheader()->getTerminator());
    IRBuilder<> B2(L->getHeader()->getFirstNonPHIOrDbgOrLifetime());
    closure = B.CreateAlloca(ST);
    for(unsigned i=0; i<StructInputs.size(); i++) {
      B.CreateStore(StructInputs[i], B.CreateConstGEP2_32(ST, closure, 0, i));
      auto l2 = B2.CreateLoad(B2.CreateConstGEP2_32(ST, closure, 0, i));
      auto UI = StructInputs[i]->use_begin(), E = StructInputs[i]->use_end();
      for (; UI != E;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (Usr && !L->contains(Usr->getParent()))
          continue;
        U.set(l2);
      }
    }
    Inputs.insert(closure);
    //llvm::errs() << "<B>\n";
    //for(auto& a : Inputs) a->dump();
    //llvm::errs() << "</B>\n";
    //StartArg->dump();
    //ea->dump();
    Inputs.remove(StartArg);
    Inputs.insert(StartArg);
    Inputs.remove(ea);
    Inputs.insert(ea);
    //llvm::errs() << "<A>\n";
    //for(auto& a : Inputs) a->dump();
    //llvm::errs() << "</A>\n";
    for (Value *V : BodyInputsToRemove)
      BodyInputs.remove(V);
    assert(0 == BodyOutputs.size() &&
           "All results from parallel loop should be passed by memory already.");
  }
  DEBUG({
      for (Value *V : Inputs)
        dbgs() << "EL input: " << *V << "\n";
      for (Value *V : Outputs)
        dbgs() << "EL output: " << *V << "\n";
    });


  Function *Helper;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.

    // LowerDbgDeclare(*(Header->getParent()));

    Helper = CreateHelper(Inputs, Outputs, L->getBlocks(),
                          Header, Preheader, ExitBlock/*L->getExitBlock()*/,
                          VMap, M,
                          F->getSubprogram() != nullptr, Returns, ".ls",
                          nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning loop.");

    // Use a fast calling convention for the helper.
    //Helper->setCallingConv(CallingConv::Fast);
    // Helper->setCallingConv(Header->getParent()->getCallingConv());
  }

  BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
  PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

  // Rewrite the cloned IV's to start at the start iteration argument.
  {
    // Rewrite clone of canonical IV to start at the start iteration
    // argument.
    Argument *NewCanonicalIVStart = cast<Argument>(VMap[InputMap[CanonicalIV]]);
    {
      int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned canonical IV does not inherit a constant value from cloned preheader.");
      NewCanonicalIV->setIncomingValue(NewPreheaderIdx, NewCanonicalIVStart);
    }

    // Rewrite other cloned IV's to start at their value at the start
    // iteration.
    const SCEV *StartIterSCEV = SE.getSCEV(NewCanonicalIVStart);
    DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
    for (PHINode *IV : IVs) {
      if (CanonicalIV == IV) continue;

      // Get the value of the IV at the start iteration.
      DEBUG(dbgs() << "IV " << *IV);
      const SCEV *IVSCEV = SE.getSCEV(IV);
      DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
      const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
      const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
      DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
            " to " << *IVAtIter << "\n");

      // NOTE: Expanded code should not refer to other IV's.
      Value *IVStart = Exp.expandCodeFor(IVAtIter, IVAtIter->getType(),
                                         NewPreheader->getTerminator());


      // Set the value that the cloned IV inherits from the cloned preheader.
      PHINode *NewIV = cast<PHINode>(VMap[IV]);
      int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned IV does not inherit a constant value from cloned preheader.");
      NewIV->setIncomingValue(NewPreheaderIdx, IVStart);
    }

    // Remap the newly added instructions in the new preheader to use
    // values local to the helper.
    for (Instruction &II : *NewPreheader)
      RemapInstruction(&II, VMap, RF_IgnoreMissingLocals,
                       /*TypeMapper=*/nullptr, /*Materializer=*/nullptr);
  }

  // If the loop limit is constant, then rewrite the loop latch
  // condition to use the end-iteration argument.
  if (isa<Constant>(LimitVar)) {
    CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
    assert(HelperCond->getOperand(1) == LimitVar);
    IRBuilder<> Builder(HelperCond);
    Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                 VMap[InputMap[LimitVar]]);
    HelperCond->replaceAllUsesWith(NewHelperCond);
    HelperCond->eraseFromParent();
  }

  // For debugging:
  BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
  {
    Value* v = &*Helper->arg_begin();
    auto UI = v->use_begin(), E = v->use_end();
    for (; UI != E;) {
      Use &U = *UI;
      ++UI;
      auto *Usr = dyn_cast<Instruction>(U.getUser());
      Usr->moveBefore(Helper->getEntryBlock().getTerminator());

      auto UI2 = Usr->use_begin(), E2 = Usr->use_end();
      for (; UI2 != E2;) {
        Use &U2 = *UI2;
        ++UI2;
        auto *Usr2 = dyn_cast<Instruction>(U2.getUser());
        Usr2->moveBefore(Helper->getEntryBlock().getTerminator());
      }
    }
  }

  if (verifyFunction(*Helper, &dbgs()))
    return false;

  // Add call to new helper function in original function.
  {
    // Setup arguments for call.
    SetVector<Value*> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.insert(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add loop limit.
    TopCallArgs.insert(LimitVar);
    // Add grainsize.
    //TopCallArgs.insert(GrainVar);
    // Add the rest of the arguments.
    for (Value *V : BodyInputs)
      TopCallArgs.insert(V);

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());

    llvm::Function* F;
    if( ((llvm::IntegerType*)LimitVar->getType())->getBitWidth() == 32 )
      F = CILKRTS_FUNC(cilk_for_32, *M);
    else {
      assert( ((llvm::IntegerType*)LimitVar->getType())->getBitWidth() == 64 );
      F = CILKRTS_FUNC(cilk_for_64, *M);
    }
    llvm::Value* args[] = {
      Builder.CreatePointerCast(Helper, F->getFunctionType()->getParamType(0)),
      Builder.CreatePointerCast(closure, F->getFunctionType()->getParamType(1)),
      LimitVar,
      ConstantInt::get(IntegerType::get(F->getContext(), sizeof(int)*8),0)
    };

    /*CallInst *TopCall = */Builder.CreateCall(F, args);

    // Use a fast calling convention for the helper.
    //TopCall->setCallingConv(CallingConv::Fast);
    // TopCall->setCallingConv(Helper->getCallingConv());
    //TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // // Update CG graph with the call we just added.
    // CG[F]->addCalledFunction(TopCall, CG[Helper]);
  }

  ++LoopsConvertedToCilkABI;

  unlinkLoop();

  return Helper;
}

/// Checks if this loop is a Tapir loop.  Right now we check that the loop is
/// in a canonical form:
/// 1) The header detaches the body.
/// 2) The loop contains a single latch.
/// 3) The body reattaches to the latch (which is necessary for a valid
///    detached CFG).
/// 4) The loop only branches to the exit block from the header or the latch.
bool LoopSpawningImpl::isTapirLoop(const Loop *L) {
  const BasicBlock *Header = L->getHeader();
  const BasicBlock *Latch = L->getLoopLatch();
  // const BasicBlock *Exit = L->getExitBlock();

  // DEBUG(dbgs() << "LS checking if Tapir loop: " << *L);

  // Header must be terminated by a detach.
  if (!isa<DetachInst>(Header->getTerminator())) {
    DEBUG(dbgs() << "LS loop header is not terminated by a detach: " << *L << "\n");
    return false;
  }

  // Loop must have a unique latch.
  if (nullptr == Latch) {
    DEBUG(dbgs() << "LS loop does not have a unique latch: " << *L << "\n");
    return false;
  }

  // // Loop must have a unique exit block.
  // if (nullptr == Exit) {
  //   DEBUG(dbgs() << "LS loop does not have a unique exit block: " << *L << "\n");
  //   SmallVector<BasicBlock *, 4> ExitBlocks;
  //   L->getUniqueExitBlocks(ExitBlocks);
  //   for (BasicBlock *Exit : ExitBlocks)
  //     DEBUG(dbgs() << *Exit);
  //   return false;
  // }

  // Continuation of header terminator must be the latch.
  const DetachInst *HeaderDetach = cast<DetachInst>(Header->getTerminator());
  const BasicBlock *Continuation = HeaderDetach->getContinue();
  if (Continuation != Latch) {
    DEBUG(dbgs() << "LS continuation of detach in header is not the latch: "
                 << *L << "\n");
    return false;
  }

  // All other predecessors of Latch are terminated by reattach instructions.
  for (auto PI = pred_begin(Latch), PE = pred_end(Latch);  PI != PE; ++PI) {
    const BasicBlock *Pred = *PI;
    if (Header == Pred) continue;
    if (!isa<ReattachInst>(Pred->getTerminator())) {
      DEBUG(dbgs() << "LS Latch has a predecessor that is not terminated "
                   << "by a reattach: " << *L << "\n");
      return false;
    }
  }

  // Get the exit block from Latch.
  BasicBlock *Exit = Latch->getTerminator()->getSuccessor(0);
  if (Header == Exit)
    Exit = Latch->getTerminator()->getSuccessor(1);

  // The only predecessors of Exit inside the loop are Header and Latch.
  for (auto PI = pred_begin(Exit), PE = pred_end(Exit);  PI != PE; ++PI) {
    const BasicBlock *Pred = *PI;
    if (!L->contains(Pred))
      continue;
    if (Header != Pred && Latch != Pred) {
      DEBUG(dbgs() << "LS Loop branches to exit block from a block "
                   << "other than the header or latch" << *L << "\n");
      return false;
    }
  }

  return true;
}

/// This routine recursively examines all descendants of the specified loop and
/// adds all Tapir loops in that tree to the vector.  This routine performs a
/// pre-order traversal of the tree of loops and pushes each Tapir loop found
/// onto the end of the vector.
void LoopSpawningImpl::addTapirLoop(Loop *L, SmallVectorImpl<Loop *> &V) {
  if (isTapirLoop(L)) {
    V.push_back(L);
    return;
  }

  LoopSpawningHints Hints(L, ORE);

  DEBUG(dbgs() << "LS: Loop hints:"
               << " strategy = " << Hints.printStrategy(Hints.getStrategy())
               << "\n");

  using namespace ore;

  if (LoopSpawningHints::ST_SEQ != Hints.getStrategy()) {
    DEBUG(dbgs() << "LS: Marked loop is not a valid Tapir loop.\n"
          << "\tLoop hints:"
          << " strategy = " << Hints.printStrategy(Hints.getStrategy())
          << "\n");
    ORE.emit(OptimizationRemarkMissed(LS_NAME, "NotTapir",
                                      L->getStartLoc(), L->getHeader())
             << "marked loop is not a valid Tapir loop");
  }

  for (Loop *InnerL : *L)
    addTapirLoop(InnerL, V);
}

#ifndef NDEBUG
/// \return string containing a file name and a line # for the given loop.
static std::string getDebugLocString(const Loop *L) {
  std::string Result;
  if (L) {
    raw_string_ostream OS(Result);
    if (const DebugLoc LoopDbgLoc = L->getStartLoc())
      LoopDbgLoc.print(OS);
    else
      // Just print the module name.
      OS << L->getHeader()->getParent()->getParent()->getModuleIdentifier();
    OS.flush();
  }
  return Result;
}
#endif

bool LoopSpawningImpl::run() {
  // Build up a worklist of inner-loops to vectorize. This is necessary as
  // the act of vectorizing or partially unrolling a loop creates new loops
  // and can invalidate iterators across the loops.
  SmallVector<Loop *, 8> Worklist;

  // Examine all top-level loops in this function, and call addTapirLoop to push
  // those loops onto the work list.
  for (Loop *L : LI)
    addTapirLoop(L, Worklist);

  LoopsAnalyzed += Worklist.size();

  // Now walk the identified inner loops.
  bool Changed = false;
  while (!Worklist.empty())
    // Process the work list of loops backwards.  For each tree of loops in this
    // function, addTapirLoop pushed those loops onto the work list according to
    // a pre-order tree traversal.  Therefore, processing the work list
    // backwards leads us to process innermost loops first.
    Changed |= processLoop(Worklist.pop_back_val());

  // Process each loop nest in the function.
  return Changed;
}

// Top-level routine to process a given loop.
bool LoopSpawningImpl::processLoop(Loop *L) {
#ifndef NDEBUG
  const std::string DebugLocStr = getDebugLocString(L);
#endif /* NDEBUG */

  // Function containing loop
  Function *F = L->getHeader()->getParent();

  DEBUG(dbgs() << "\nLS: Checking a Tapir loop in \""
               << L->getHeader()->getParent()->getName() << "\" from "
        << DebugLocStr << ": " << *L << "\n");

  LoopSpawningHints Hints(L, ORE);

  DEBUG(dbgs() << "LS: Loop hints:"
               << " strategy = " << Hints.printStrategy(Hints.getStrategy())
               << "\n");

  using namespace ore;

  // Get the loop preheader.  LoopSimplify should guarantee that the loop
  // preheader is not terminated by a sync.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    DEBUG(dbgs() << "LS: Loop lacks a preheader.\n");
    ORE.emit(OptimizationRemarkMissed(LS_NAME, "NoPreheader",
                                      L->getStartLoc(), L->getHeader())
             << "loop lacks a preheader");
    emitMissedWarning(F, L, Hints, &ORE);
    return false;
  } else if (!isa<BranchInst>(Preheader->getTerminator())) {
    DEBUG(dbgs() << "LS: Loop preheader is not terminated by a branch.\n");
    ORE.emit(OptimizationRemarkMissed(LS_NAME, "ComplexPreheader",
                                      L->getStartLoc(), L->getHeader())
             << "loop preheader not terminated by a branch");
    emitMissedWarning(F, L, Hints, &ORE);
    return false;
  }

  switch(Hints.getStrategy()) {
  case LoopSpawningHints::ST_SEQ:
    DEBUG(dbgs() << "LS: Hints dictate sequential spawning.\n");
    break;
  case LoopSpawningHints::ST_DAC:
    DEBUG(dbgs() << "LS: Hints dictate DAC spawning.\n");
    {
      DebugLoc DLoc = L->getStartLoc();
      BasicBlock *Header = L->getHeader();
      DACLoopSpawning DLS(L, SE, &LI, &DT, &AC, ORE);
      // CilkABILoopSpawning DLS(L, SE, &LI, &DT, &AC, ORE);
      // DACLoopSpawning DLS(L, SE, LI, DT, TLI, TTI, ORE);
      if (DLS.processLoop()) {
        DEBUG({
            if (verifyFunction(*L->getHeader()->getParent())) {
              dbgs() << "Transformed function is invalid.\n";
              return false;
            }
          });
        // Report success.
        ORE.emit(OptimizationRemark(LS_NAME, "DACSpawning", DLoc, Header)
                 << "spawning iterations using divide-and-conquer");
        return true;
      } else {
        // Report failure.
        ORE.emit(OptimizationRemarkMissed(LS_NAME, "NoDACSpawning", DLoc,
                                          Header)
                 << "cannot spawn iterations using divide-and-conquer");
        emitMissedWarning(F, L, Hints, &ORE);
        return false;
      }
    }
    break;
  case LoopSpawningHints::ST_END:
    dbgs() << "LS: Hints specify unknown spawning strategy.\n";
    break;
  }
  return false;
}

// PreservedAnalyses LoopSpawningPass::run(Module &M, ModuleAnalysisManager &AM) {
//   // Find functions that detach for processing.
//   SmallVector<Function *, 4> WorkList;
//   for (Function &F : M)
//     for (BasicBlock &BB : F)
//       if (isa<DetachInst>(BB.getTerminator()))
//         WorkList.push_back(&F);

//   if (WorkList.empty())
//     return PreservedAnalyses::all();

//   bool Changed = false;
//   while (!WorkList.empty()) {
//     Function *F = WorkList.back();
//     auto &TLI = AM.getResult<TargetLibraryAnalysis>(M);
//     auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
//     auto &LI = FAM.getResult<LoopAnalysis>(*F);
//     auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(*F);
//     auto &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
//     auto &TTI = FAM.getResult<TargetIRAnalysis>(*F);
//     auto &AA = FAM.getResult<AAManager>(*F);
//     auto &AC = FAM.getResult<AssumptionAnalysis>(*F);
//     auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
//     LoopSpawningImpl Impl(*F, LI, SE, DT, TTI, &TLI, AA, AC, ORE);
//     Changed |= Impl.run();
//     WorkList.pop_back();
//   }

//   if (Changed)
//     return PreservedAnalyses::none();
//   return PreservedAnalyses::all();
// }

PreservedAnalyses LoopSpawningPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  // Determine if function detaches.
  bool DetachingFunction = false;
  for (BasicBlock &BB : F)
    if (isa<DetachInst>(BB.getTerminator()))
      DetachingFunction = true;

  if (!DetachingFunction)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  // auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  // auto &TLI = AM.getResult<TargetLibraryAnalysis>(M);
  // auto &AA = AM.getResult<AAManager>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &ORE =
    AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  // OptimizationRemarkEmitter ORE(F);

  bool Changed = LoopSpawningImpl(F, LI, SE, DT, AC, ORE).run();

  AM.invalidate<ScalarEvolutionAnalysis>(F);

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

namespace {
struct LoopSpawning : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopSpawning() : FunctionPass(ID) {
    initializeLoopSpawningPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    bool DetachingFunction = false;
    for (BasicBlock &BB : F)
      if (isa<DetachInst>(BB.getTerminator()))
        DetachingFunction = true;

    if (!DetachingFunction)
      return false;

    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    // auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);
    // auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    // auto *TLI = TLIP ? &TLIP->getTLI() : nullptr;
    // auto *TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    // auto *AA = &getAnalysis<AAResultsWrapperPass>(*F).getAAResults();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &ORE =
      getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    // OptimizationRemarkEmitter ORE(F);
    return LoopSpawningImpl(F, LI, SE, DT, AC, ORE).run();
  }

  // bool runOnModule(Module &M) override {
  //   if (skipModule(M))
  //     return false;

  //   // Find functions that detach for processing.
  //   SmallVector<Function *, 4> WorkList;
  //   for (Function &F : M)
  //     for (BasicBlock &BB : F)
  //       if (isa<DetachInst>(BB.getTerminator()))
  //         WorkList.push_back(&F);

  //   if (WorkList.empty())
  //     return false;

  //   auto GetLI = [this](Function &F) -> LoopInfo & {
  //     return getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  //   };
  //   auto GetSE = [this](Function &F) -> ScalarEvolution & {
  //     return getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  //   };
  //   auto GetDT = [this](Function &F) -> DominatorTree & {
  //     return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  //   };

  //   bool Changed = false;
  //   while (!WorkList.empty()) {
  //     // Process the next function.
  //     Function *F = WorkList.back();
  //     // auto *LI = &getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
  //     // auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>(*F).getSE();
  //     // auto *DT = &getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
  //     // auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);
  //     // auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  //     // auto *TLI = TLIP ? &TLIP->getTLI() : nullptr;
  //     // auto *TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  //     // auto *AA = &getAnalysis<AAResultsWrapperPass>(*F).getAAResults();
  //     // auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);
  //     auto &ORE =
  //       getAnalysis<OptimizationRemarkEmitterWrapperPass>(*F).getORE();
  //     // OptimizationRemarkEmitter ORE(F);
  //     // LoopSpawningImpl Impl(*F, GetLI, GetSE, GetDT, *TTI, TLI, *AA, *AC, ORE);
  //     LoopSpawningImpl Impl(*F, GetLI, GetSE, GetDT, ORE);
  //     Changed |= Impl.run();

  //     WorkList.pop_back();
  //   }
  //   return Changed;
  // }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    // AU.addRequired<LoopAccessLegacyAnalysis>();
    // getAAResultsAnalysisUsage(AU);
    // AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};
}

char LoopSpawning::ID = 0;
// static RegisterPass<LoopSpawning> X(LS_NAME, "Transform Tapir loops to spawn iterations efficiently", false, false);
static const char ls_name[] = "Loop Spawning";
INITIALIZE_PASS_BEGIN(LoopSpawning, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
// INITIALIZE_PASS_DEPENDENCY(LoopAccessLegacyAnalysis)
// INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LoopSpawning, LS_NAME, ls_name, false, false)

namespace llvm {
Pass *createLoopSpawningPass() {
  return new LoopSpawning();
}
}
