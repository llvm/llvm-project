//===- LoopInterchange.cpp - Loop interchange pass-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Pass handles loop interchange transform.
// This pass interchanges loops to provide a more cache-friendly memory access
// patterns.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopInterchange.h"
#include "LoopInterchangeUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopCacheAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-interchange"

STATISTIC(LoopsInterchanged, "Number of loops interchanged");
STATISTIC(OuterEpiloguesDistributed,
          "Number of outer-loop epilogues distributed into their own loop "
          "before interchange");

static cl::opt<int> LoopInterchangeCostThreshold(
    "loop-interchange-threshold", cl::init(0), cl::Hidden,
    cl::desc("Interchange if you gain more than this number"));

static cl::opt<unsigned int> MaxMemInstrRatio(
    "loop-interchange-max-mem-instr-ratio", cl::init(4), cl::Hidden,
    cl::desc("Maximum number of load/store instructions squared in relation to "
             "the total number of instructions. Higher value may lead to more "
             "interchanges at the cost of compile-time"));

namespace {

using LoopVector = SmallVector<Loop *, 8>;

/// A list of direction vectors. Each entry represents a direction vector
/// corresponding to one or more dependencies existing in the loop nest. The
/// length of all direction vectors is equal and is N + 1, where N is the depth
/// of the loop nest. The first N elements correspond to the dependency
/// direction of each N loops. The last one indicates whether this entry is
/// forward dependency ('<') or not ('*'). The term "forward" aligns with what
/// is defined in LoopAccessAnalysis.
// TODO: Check if we can use a sparse matrix here.
using CharMatrix = std::vector<std::vector<char>>;

/// Types of rules used in profitability check.
enum class RuleTy {
  PerLoopCacheAnalysis,
  PerInstrOrderCost,
  ForVectorization,
  Ignore
};

enum class DependenceColumns {
  SelectedSubnest,
  AbsoluteAncestors,
};

} // end anonymous namespace

// Minimum loop depth supported.
static cl::opt<unsigned int> MinLoopNestDepth(
    "loop-interchange-min-loop-nest-depth", cl::init(2), cl::Hidden,
    cl::desc("Minimum depth of loop nest considered for the transform"));

// Maximum loop depth supported.
static cl::opt<unsigned int> MaxLoopNestDepth(
    "loop-interchange-max-loop-nest-depth", cl::init(10), cl::Hidden,
    cl::desc("Maximum depth of loop nest considered for the transform"));

// We prefer cache cost to vectorization by default.
static cl::list<RuleTy> Profitabilities(
    "loop-interchange-profitabilities", cl::MiscFlags::CommaSeparated,
    cl::Hidden,
    cl::desc("List of profitability heuristics to be used. They are applied in "
             "the given order"),
    cl::list_init<RuleTy>({RuleTy::PerInstrOrderCost,
                           RuleTy::ForVectorization}),
    cl::values(clEnumValN(RuleTy::PerLoopCacheAnalysis, "cache",
                          "Prioritize loop cache cost"),
               clEnumValN(RuleTy::PerInstrOrderCost, "instorder",
                          "Prioritize the IVs order of each instruction"),
               clEnumValN(RuleTy::ForVectorization, "vectorize",
                          "Prioritize vectorization"),
               clEnumValN(RuleTy::Ignore, "ignore",
                          "Ignore profitability, force interchange (does not "
                          "work with other options)")));

// Support for the inner-loop reduction pattern.
static cl::opt<bool> EnableReduction2Memory(
    "loop-interchange-reduction-to-mem", cl::init(false), cl::Hidden,
    cl::desc("Support for the inner-loop reduction pattern."));

static cl::opt<bool> EnableOuterEpilogueFission(
    "loop-interchange-outer-epilogue-fission", cl::init(false), cl::Hidden,
    cl::desc("Prepare a provably independent outer-loop epilogue for "
             "distribution before interchange"));

static cl::opt<unsigned int> MaxOuterEpilogueFissionCandidates(
    "loop-interchange-max-outer-epilogue-fission-candidates", cl::init(10),
    cl::Hidden,
    cl::desc("Maximum number of eligible outer-epilogue fission candidates "
             "prepared"));

static constexpr StringLiteral RuntimeVersionedLoopMarker =
    "llvm.loop.interchange.runtime_versioned";

// Outer-epilogue runtime versioning: opt-in, effective only with
// outer-epilogue fission enabled.
static cl::opt<bool> EnableOuterEpilogueRuntimeVersioning(
    "loop-interchange-outer-epilogue-runtime-versioning", cl::init(false),
    cl::Hidden,
    cl::desc("Version runtime-bounded outer-epilogue fission candidates"));

static cl::opt<unsigned int> MaxRuntimeVersioningBlocks(
    "loop-interchange-max-runtime-versioning-blocks", cl::init(16), cl::Hidden,
    cl::desc("Maximum number of blocks in an outer loop selected for runtime "
             "versioning"));

static cl::opt<unsigned int> MaxRuntimeVersioningInstructions(
    "loop-interchange-max-runtime-versioning-instructions", cl::init(128),
    cl::Hidden,
    cl::desc("Maximum number of non-debug instructions in an outer loop "
             "selected for runtime versioning"));

static cl::opt<unsigned int> RuntimeTripExpansionBudget(
    "loop-interchange-runtime-trip-expansion-budget", cl::init(8), cl::Hidden,
    cl::desc("Maximum SCEV expansion cost for a runtime outer-trip guard"));

static cl::opt<bool> PrintPreparedEpiloguePlans(
    "loop-interchange-print-prepared-plan", cl::init(false), cl::Hidden,
    cl::desc("Print deterministic outer-epilogue preparation decisions"));

#ifndef NDEBUG
static bool noDuplicateRulesAndIgnore(ArrayRef<RuleTy> Rules) {
  SmallSet<RuleTy, 4> Set;
  for (RuleTy Rule : Rules) {
    if (!Set.insert(Rule).second)
      return false;
    if (Rule == RuleTy::Ignore)
      return false;
  }
  return true;
}

static void printDepMatrix(CharMatrix &DepMatrix) {
  for (auto &Row : DepMatrix) {
    // Drop the last element because it is a flag indicating whether this is
    // forward dependency or not, which doesn't affect the legality check.
    for (char D : drop_end(Row))
      LLVM_DEBUG(dbgs() << D << " ");
    LLVM_DEBUG(dbgs() << "\n");
  }
}

/// Return true if \p Src appears before \p Dst in the same basic block.
/// Precondition: \p Src and \Dst are distinct instructions within the same
/// basic block.
static bool inThisOrder(const Instruction *Src, const Instruction *Dst) {
  assert(Src->getParent() == Dst->getParent() && Src != Dst &&
         "Expected Src and Dst to be different instructions in the same BB");

  bool FoundSrc = false;
  for (const Instruction &I : *(Src->getParent())) {
    if (&I == Src) {
      FoundSrc = true;
      continue;
    }
    if (&I == Dst)
      return FoundSrc;
  }

  llvm_unreachable("Dst not found");
}
#endif

namespace {

/// A borrowed view of the instructions omitted by virtual extraction.
/// Ordinary interchange uses the empty view.
class ExtractedEpilogueView {
  const SmallPtrSetImpl<Instruction *> *Excluded = nullptr;

public:
  ExtractedEpilogueView() = default;
  explicit ExtractedEpilogueView(const SmallPtrSetImpl<Instruction *> *Excluded)
      : Excluded(Excluded) {}

  bool excludes(const Instruction *I) const {
    return Excluded && Excluded->contains(const_cast<Instruction *>(I));
  }

  explicit operator bool() const { return Excluded != nullptr; }

  const SmallPtrSetImpl<Instruction *> *getExcludedInstructions() const {
    return Excluded;
  }
};

/// Follow the ordinary empty-block walk while treating the extracted slice as
/// absent. Forwarding PHIs remain in the nest.
static const BasicBlock &
skipVirtuallyEmptyBlockUntil(const BasicBlock *From, const BasicBlock *End,
                             ExtractedEpilogueView View) {
  if (!View)
    return LoopNest::skipEmptyBlockUntil(From, End);

  assert(From && End && "expected valid path endpoints");
  if (From == End || !From->getUniqueSuccessor())
    return *From;

  auto IsVirtuallyEmpty = [View](const BasicBlock *BB) {
    return all_of(*BB, [View](const Instruction &I) {
      return I.isTerminator() || View.excludes(&I);
    });
  };

  SmallPtrSet<const BasicBlock *, 4> Visited;
  const BasicBlock *BB = From->getUniqueSuccessor();
  const BasicBlock *PredBB = From;
  while (BB && BB != End && IsVirtuallyEmpty(BB) && !Visited.contains(BB)) {
    Visited.insert(BB);
    PredBB = BB;
    BB = BB->getUniqueSuccessor();
  }
  return BB == End ? *End : *PredBB;
}

} // end anonymous namespace

static bool populateDependencyMatrix(
    CharMatrix &DepMatrix, unsigned Level, Loop *L, DependenceInfo *DI,
    ScalarEvolution *SE, OptimizationRemarkEmitter *ORE,
    DependenceColumns Columns = DependenceColumns::SelectedSubnest,
    ExtractedEpilogueView View = {}) {
  using ValueVector = SmallVector<Value *, 16>;

  ValueVector MemInstr;
  unsigned NumInsts = 0;

  // For each block.
  for (BasicBlock *BB : L->blocks()) {
    // Scan the BB and collect legal loads and stores.
    for (Instruction &I : *BB) {
      if (View.excludes(&I))
        continue;
      NumInsts++;
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        if (!Ld->isSimple())
          return false;
        MemInstr.push_back(&I);
      } else if (auto *St = dyn_cast<StoreInst>(&I)) {
        if (!St->isSimple())
          return false;
        MemInstr.push_back(&I);
      }
    }
  }

  // To populate the dependence matrix, we perform dependence test for each pair
  // of memory instructions, which has O(NumMemInstr^2) complexity. This implies
  // that even if the number of memory instructions is small, the analysis can
  // still be expensive if the most of the instructions in the loop are memory
  // instructions. On the other hand, if the number of memory instructions is
  // not small, but the loop is large (i.e., it contains many non-memory
  // instructions), the analysis can still be affordable.
  unsigned NumMemInstr = MemInstr.size();
  LLVM_DEBUG(dbgs() << "Found " << NumMemInstr
                    << " Loads and Stores to analyze\n");
  if (MaxMemInstrRatio * NumInsts < NumMemInstr * NumMemInstr) {
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedLoop",
                                        L->getStartLoc(), L->getHeader())
               << "Number of loads/stores exceeded, the supported maximum can "
                  "be "
                  "increased with option "
                  "-loop-interchange-max-mem-instr-ratio.";
      });
    return false;
  }
  ValueVector::iterator I, IE, J, JE;

  // Manage direction vectors that are already seen. Map each direction vector
  // to an index of DepMatrix at which it is stored.
  StringMap<unsigned> Seen;

  for (I = MemInstr.begin(), IE = MemInstr.end(); I != IE; ++I) {
    for (J = I, JE = MemInstr.end(); J != JE; ++J) {
      std::vector<char> Dep;
      Instruction *Src = cast<Instruction>(*I);
      Instruction *Dst = cast<Instruction>(*J);
      // Ignore Input dependencies.
      if (isa<LoadInst>(Src) && isa<LoadInst>(Dst))
        continue;
      // Track Output, Flow, and Anti dependencies.
      if (auto D = DI->depends(Src, Dst)) {
        assert(D->isOrdered() && "Expected an output, flow or anti dep.");
        // If the direction vector is negative, normalize it to
        // make it non-negative.
        if (D->normalize(SE))
          LLVM_DEBUG(dbgs() << "Negative dependence vector normalized.\n");
        LLVM_DEBUG(StringRef DepType =
                       D->isFlow() ? "flow" : D->isAnti() ? "anti" : "output";
                   dbgs() << "Found " << DepType
                          << " dependency between Src and Dst\n"
                          << " Src:" << *Src << "\n Dst:" << *Dst << '\n');
        unsigned Levels = D->getLevels();
        char Direction;
        for (unsigned II = 1; II <= Levels; ++II) {
          // `DVEntry::LE` is converted to `*`. This is because `LE` means `<`
          // or `=`, for which we don't have an equivalent representation, so
          // that the conservative approximation is necessary. The same goes for
          // `DVEntry::GE`.
          // TODO: Use of fine-grained expressions allows for more accurate
          // analysis.
          unsigned Dir = D->getDirection(II);
          if (Dir == Dependence::DVEntry::LT)
            Direction = '<';
          else if (Dir == Dependence::DVEntry::GT)
            Direction = '>';
          else if (Dir == Dependence::DVEntry::EQ)
            Direction = '=';
          else
            Direction = '*';
          Dep.push_back(Direction);
        }

        // If the Dependence object doesn't have any information, fill the
        // dependency vector with '*'.
        unsigned AbsoluteDepth = L->getLoopDepth() + Level - 1;
        if (D->isConfused()) {
          assert(Dep.empty() && "Expected empty dependency vector");
          Dep.assign(AbsoluteDepth, '*');
        }

        // Absolute mode cannot represent levels below the selected inner loop.
        if (Columns == DependenceColumns::AbsoluteAncestors &&
            Dep.size() > AbsoluteDepth)
          return false;

        while (Dep.size() < AbsoluteDepth) {
          Dep.push_back('I');
        }

        // Dependence analysis reports levels for the full enclosing loop nest.
        // Keep only the suffix that corresponds to the selected perfect
        // subnest.
        if (Columns == DependenceColumns::SelectedSubnest && Dep.size() > Level)
          Dep.erase(Dep.begin(), Dep.end() - Level);

        // If all the elements of any direction vector have only '*', legality
        // can't be proven. Exit early to save compile time.
        if (all_of(Dep, equal_to('*'))) {
          if (ORE)
            ORE->emit([&]() {
              return OptimizationRemarkMissed(DEBUG_TYPE, "Dependence",
                                              L->getStartLoc(), L->getHeader())
                     << "All loops have dependencies in all directions.";
            });
          return false;
        }

        // Test whether the dependency is forward or not.
        bool IsKnownForward = true;
        if (Src->getParent() != Dst->getParent()) {
          // In general, when Src and Dst are in different BBs, the execution
          // order of them within a single iteration is not guaranteed. Treat
          // conservatively as not-forward dependency in this case.
          IsKnownForward = false;
        } else {
          // Src and Dst are in the same BB. If they are the different
          // instructions, Src should appear before Dst in the BB as they are
          // stored to MemInstr in that order.
          assert((Src == Dst || inThisOrder(Src, Dst)) &&
                 "Unexpected instructions");

          // If the Dependence object is reversed (due to normalization), it
          // represents the dependency from Dst to Src, meaning it is a backward
          // dependency. Otherwise it should be a forward dependency.
          bool IsReversed = D->getSrc() != Src;
          if (IsReversed)
            IsKnownForward = false;
        }

        // Initialize the last element. Assume forward dependencies only; it
        // will be updated later if there is any non-forward dependency.
        Dep.push_back('<');

        // The last element should express the "summary" among one or more
        // direction vectors whose first N elements are the same (where N is
        // the depth of the loop nest). Hence we exclude the last element from
        // the Seen map.
        auto [Ite, Inserted] = Seen.try_emplace(
            StringRef(Dep.data(), Dep.size() - 1), DepMatrix.size());

        // Make sure we only add unique entries to the dependency matrix.
        if (Inserted)
          DepMatrix.push_back(Dep);

        // If we cannot prove that this dependency is forward, change the last
        // element of the corresponding entry. Since a `[... *]` dependency
        // includes a `[... <]` dependency, we do not need to keep both and
        // change the existing entry instead.
        if (!IsKnownForward)
          DepMatrix[Ite->second].back() = '*';
      }
    }
  }

  return true;
}

// A loop is moved from index 'from' to an index 'to'. Update the Dependence
// matrix by exchanging the two columns.
static void interChangeDependencies(CharMatrix &DepMatrix, unsigned FromIndx,
                                    unsigned ToIndx) {
  for (auto &Row : DepMatrix)
    std::swap(Row[ToIndx], Row[FromIndx]);
}

// Check if a direction vector is lexicographically positive. Return true if it
// is positive, nullopt if it is "zero", otherwise false.
// [Theorem] A permutation of the loops in a perfect nest is legal if and only
// if the direction matrix, after the same permutation is applied to its
// columns, has no ">" direction as the leftmost non-"=" direction in any row.
static std::optional<bool>
isLexicographicallyPositive(ArrayRef<char> DV, unsigned Begin, unsigned End) {
  for (unsigned char Direction : DV.slice(Begin, End - Begin)) {
    if (Direction == '<')
      return true;
    if (Direction == '>' || Direction == '*')
      return false;
  }
  return std::nullopt;
}

// Checks if it is legal to interchange 2 loops.
static bool isLegalToInterChangeLoops(CharMatrix &DepMatrix,
                                      unsigned InnerLoopId,
                                      unsigned OuterLoopId) {
  unsigned NumRows = DepMatrix.size();
  std::vector<char> Cur;
  // For each row check if it is valid to interchange.
  for (unsigned Row = 0; Row < NumRows; ++Row) {
    // Create temporary DepVector check its lexicographical order
    // before and after swapping OuterLoop vs InnerLoop
    Cur = DepMatrix[Row];

    // If the surrounding loops already ensure that the direction vector is
    // lexicographically positive, nothing within the loop will be able to break
    // the dependence. In such a case we can skip the subsequent check.
    if (isLexicographicallyPositive(Cur, 0, OuterLoopId) == true)
      continue;

    // Check if the direction vector is lexicographically positive (or zero)
    // for both before/after exchanged. Ignore the last element because it
    // doesn't affect the legality.
    if (isLexicographicallyPositive(Cur, OuterLoopId, Cur.size() - 1) == false)
      return false;
    std::swap(Cur[InnerLoopId], Cur[OuterLoopId]);
    if (isLexicographicallyPositive(Cur, OuterLoopId, Cur.size() - 1) == false)
      return false;
  }
  return true;
}

/// True when no row of \p DepMatrix has a '>' or '*' before any '<' in the
/// columns before \p OuterLoopId, the loops enclosing the selected outer loop.
static bool hasDecisiveOrEqualAncestorPrefix(const CharMatrix &DepMatrix,
                                             unsigned OuterLoopId) {
  for (const std::vector<char> &Row : DepMatrix)
    if (isLexicographicallyPositive(Row, 0, OuterLoopId) == false)
      return false;
  return true;
}

static void populateWorklist(Loop &L, LoopVector &LoopList) {
  LLVM_DEBUG(dbgs() << "Calling populateWorklist on Func: "
                    << L.getHeader()->getParent()->getName() << " Loop: %"
                    << L.getHeader()->getName() << '\n');
  assert(LoopList.empty() && "LoopList should initially be empty!");
  Loop *CurrentLoop = &L;
  const std::vector<Loop *> *Vec = &CurrentLoop->getSubLoops();
  while (!Vec->empty()) {
    // The current loop has multiple subloops in it hence it is not tightly
    // nested.
    // Discard all loops above it added into Worklist.
    if (Vec->size() != 1) {
      LoopList = {};
      return;
    }

    LoopList.push_back(CurrentLoop);
    CurrentLoop = Vec->front();
    Vec = &CurrentLoop->getSubLoops();
  }
  LoopList.push_back(CurrentLoop);
}

static bool hasSupportedLoopDepth(ArrayRef<Loop *> LoopList) {
  unsigned LoopNestDepth = LoopList.size();
  return LoopNestDepth >= MinLoopNestDepth && LoopNestDepth <= MaxLoopNestDepth;
}

static bool hasSupportedLoopDepth(ArrayRef<Loop *> LoopList,
                                  OptimizationRemarkEmitter &ORE) {
  if (!hasSupportedLoopDepth(LoopList)) {
    LLVM_DEBUG(dbgs() << "Unsupported depth of loop nest " << LoopList.size()
                      << ", the supported range is [" << MinLoopNestDepth
                      << ", " << MaxLoopNestDepth << "].\n");
    Loop *OuterLoop = LoopList.front();
    ORE.emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedLoopNestDepth",
                                      OuterLoop->getStartLoc(),
                                      OuterLoop->getHeader())
             << "Unsupported depth of loop nest, the supported range is ["
             << std::to_string(MinLoopNestDepth) << ", "
             << std::to_string(MaxLoopNestDepth) << "].\n";
    });
    return false;
  }
  return true;
}

static bool isComputableLoopNest(ScalarEvolution *SE,
                                 ArrayRef<Loop *> LoopList) {
  for (Loop *L : LoopList) {
    const SCEV *ExitCountOuter = SE->getBackedgeTakenCount(L);
    if (isa<SCEVCouldNotCompute>(ExitCountOuter)) {
      LLVM_DEBUG(dbgs() << "Couldn't compute backedge count\n");
      return false;
    }
    if (L->getNumBackEdges() != 1) {
      LLVM_DEBUG(dbgs() << "NumBackEdges is not equal to 1\n");
      return false;
    }
    if (!L->getExitingBlock()) {
      LLVM_DEBUG(dbgs() << "Loop doesn't have unique exit block\n");
      return false;
    }
  }
  return true;
}

namespace {

/// LoopInterchangeLegality checks if it is legal to interchange the loop.
class LoopInterchangeLegality {
public:
  LoopInterchangeLegality(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                          OptimizationRemarkEmitter *ORE, DominatorTree *DT,
                          ExtractedEpilogueView View = {})
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), DT(DT), ORE(ORE),
        HasExtractedEpilogueView(bool(View)) {
    if (const auto *Excluded = View.getExcludedInstructions())
      ExcludedEpilogueInstructions.insert(Excluded->begin(), Excluded->end());
  }

  /// Check if the loops can be interchanged.
  bool canInterchangeLoops(unsigned InnerLoopId, unsigned OuterLoopId,
                           CharMatrix &DepMatrix);

  /// Check if the loop structure is understood. We do not handle triangular
  /// loops for now.
  bool isLoopStructureUnderstood();

  bool currentLimitations();

  const SmallPtrSetImpl<PHINode *> &getOuterInnerReductions() const {
    return OuterInnerReductions;
  }

  const ArrayRef<PHINode *> getInnerLoopInductions() const {
    return InnerLoopInductions;
  }

  ArrayRef<Instruction *> getHasNoWrapReductions() const {
    return HasNoWrapReductions;
  }

  ArrayRef<Instruction *> getHasNoInfInsts() const { return HasNoInfInsts; }

  bool isPreparedFor(
      Loop *Outer, Loop *Inner,
      const SmallPtrSetImpl<Instruction *> &ExcludedInstructions) const {
    return OuterLoop == Outer && InnerLoop == Inner &&
           HasExtractedEpilogueView &&
           ExcludedEpilogueInstructions.size() == ExcludedInstructions.size() &&
           all_of(ExcludedInstructions, [&](Instruction *I) {
             return ExcludedEpilogueInstructions.contains(I);
           });
  }

  /// Record reductions in the inner loop. Currently supported reductions:
  /// - initialized from a constant.
  /// - reduction PHI node has only one user.
  /// - located in the innermost loop.
  struct InnerReduction {
    /// The reduction itself.
    PHINode *Reduction;
    Value *Init;
    Value *Next;
    /// The Lcssa PHI.
    PHINode *LcssaPhi;
    /// Store reduction result into memory object.
    StoreInst *LcssaStore;
    /// The memory Location.
    Value *MemRef;
    Type *ElemTy;
  };

  ArrayRef<InnerReduction> getInnerReductions() const {
    return InnerReductions;
  }

private:
  bool tightlyNested(Loop *Outer, Loop *Inner);
  bool containsUnsafeInstructions(BasicBlock *BB, Instruction *Skip);
  bool isVirtuallyExtracted(const Instruction *I) const {
    return HasExtractedEpilogueView &&
           ExcludedEpilogueInstructions.contains(const_cast<Instruction *>(I));
  }
  ExtractedEpilogueView getExtractedEpilogueView() {
    if (!HasExtractedEpilogueView)
      return {};
    return ExtractedEpilogueView(&ExcludedEpilogueInstructions);
  }

  /// Traverse all PHI nodes in the header of each loop in the loop nest
  /// starting from \p OuterLoop, and perform the following checks:
  ///
  /// - Identify induction variables in the child loop of \p OuterLoop.
  /// - Check for reductions across the inner loop and \p OuterLoop.
  /// - Detect unsupported PHI nodes.
  ///
  /// Return false if any unsupported PHI node is found or if no induction
  /// variable is found in the child loop of \p OuterLoop. Otherwise return
  /// true.
  bool checkInductionsAndReductions(Loop *OuterLoop);

  /// Detect and record the reduction of the inner loop. Add them to
  /// InnerReductions.
  ///
  ///    innerloop:
  ///        Re = phi<0.0, Next>
  ///        Next = Re op ...
  ///    OuterLoopLatch:
  ///        Lcssa = phi<Next>    ; lcssa phi
  ///        store Lcssa, MemRef  ; LcssaStore
  ///
  bool isInnerReduction(Loop *L, PHINode *Phi,
                        SmallVectorImpl<Instruction *> &HasNoWrapInsts);

  Loop *OuterLoop;
  Loop *InnerLoop;

  ScalarEvolution *SE;
  DominatorTree *DT;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// Own the exclusion set so moving a prepared plan cannot invalidate it.
  SmallPtrSet<Instruction *, 32> ExcludedEpilogueInstructions;
  bool HasExtractedEpilogueView = false;

  /// Set of reduction PHIs taking part of a reduction across the inner and
  /// outer loop.
  SmallPtrSet<PHINode *, 4> OuterInnerReductions;

  /// Set of inner loop induction PHIs
  SmallVector<PHINode *, 8> InnerLoopInductions;

  /// Hold instructions that have nuw/nsw flags and involved in reductions,
  /// like integer addition/multiplication. Those flags must be dropped when
  /// interchanging the loops.
  SmallVector<Instruction *, 4> HasNoWrapReductions;

  /// Hold instructions that have ninf flags and involved in reductions. Those
  /// flags must be dropped when interchanging the loops.
  SmallVector<Instruction *, 4> HasNoInfInsts;

  /// Vector of reductions in the inner loop.
  SmallVector<InnerReduction, 8> InnerReductions;
};

/// Manages information utilized by the profitability check for cache. The main
/// purpose of this class is to delay the computation of CacheCost until it is
/// actually needed.
class CacheCostManager {
  Loop *OutermostLoop;
  LoopStandardAnalysisResults *AR;
  DependenceInfo *DI;

  /// CacheCost for \ref OutermostLoop. Once it is computed, it is cached. Note
  /// that the result can be nullptr.
  std::optional<std::unique_ptr<CacheCost>> CC;

  /// Maps each loop to an index representing the optimal position within the
  /// loop-nest, as determined by the cache cost analysis.
  DenseMap<const Loop *, unsigned> CostMap;

  void computeIfUnitinialized();

public:
  CacheCostManager(Loop *OutermostLoop, LoopStandardAnalysisResults *AR,
                   DependenceInfo *DI)
      : OutermostLoop(OutermostLoop), AR(AR), DI(DI) {}
  CacheCost *getCacheCost();
  const DenseMap<const Loop *, unsigned> &getCostMap();
};

/// LoopInterchangeProfitability checks if it is profitable to interchange the
/// loop.
class LoopInterchangeProfitability {
public:
  LoopInterchangeProfitability(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                               OptimizationRemarkEmitter *ORE)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), ORE(ORE) {}

  /// Check if the loop interchange is profitable.
  bool isProfitable(const Loop *InnerLoop, const Loop *OuterLoop,
                    unsigned InnerLoopId, unsigned OuterLoopId,
                    CharMatrix &DepMatrix, CacheCostManager &CCM);

private:
  int getInstrOrderCost();
  std::optional<bool> isProfitablePerLoopCacheAnalysis(
      const DenseMap<const Loop *, unsigned> &CostMap, CacheCost *CC);
  std::optional<bool> isProfitablePerInstrOrderCost();
  std::optional<bool> isProfitableForVectorization(unsigned InnerLoopId,
                                                   unsigned OuterLoopId,
                                                   CharMatrix &DepMatrix);
  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;
};

/// LoopInterchangeTransform interchanges the loop.
class LoopInterchangeTransform {
public:
  LoopInterchangeTransform(Loop *Outer, Loop *Inner, ScalarEvolution *SE,
                           LoopInfo *LI, DominatorTree *DT,
                           const LoopInterchangeLegality &LIL)
      : OuterLoop(Outer), InnerLoop(Inner), SE(SE), LI(LI), DT(DT), LIL(LIL) {}

  /// Interchange OuterLoop and InnerLoop.
  void transform(ArrayRef<Instruction *> DropNoWrapInsts,
                 ArrayRef<Instruction *> DropNoInfInsts);
  void reduction2Memory();
  void restructureLoops(Loop *NewInner, Loop *NewOuter,
                        BasicBlock *OrigInnerPreHeader,
                        BasicBlock *OrigOuterPreHeader);
  void removeChildLoop(Loop *OuterLoop, Loop *InnerLoop);

private:
  void adjustLoopBranches();

  Loop *OuterLoop;
  Loop *InnerLoop;

  /// Scev analysis.
  ScalarEvolution *SE;

  LoopInfo *LI;
  DominatorTree *DT;

  const LoopInterchangeLegality &LIL;
};

struct PreparedOuterIVControl {
  PHINode *Induction = nullptr;
  Value *InitialValue = nullptr;
  Instruction *NextValue = nullptr;
  ICmpInst *LatchCompare = nullptr;
  CondBrInst *LatchBranch = nullptr;
  SmallVector<Instruction *, 4> LatchInstructions;
};

/// An outer-loop epilogue that can be distributed into its own loop. Debug
/// output abbreviates it as E and the retained nest as N. Instructions holds
/// the closed slice in program order. Forwarding PHIs remain with the nest.
struct DistributableOuterEpilogue {
  Loop *Outer = nullptr;
  Loop *Inner = nullptr;
  SmallVector<BasicBlock *, 4> Path;
  SmallVector<Instruction *, 16> Instructions;
  SmallPtrSet<Instruction *, 32> InstructionSet;
  SmallVector<Instruction *, 8> MemoryInstructions;
  SmallVector<Instruction *, 8> OuterIVDerivedInstructions;
  PreparedOuterIVControl OuterControl;
};

/// Dependence objects are query temporaries. Retain only the proof summary
/// and the ID of any bound requirements that discharge it.
struct PreparedCrossPartitionDependence {
  bool UsedByteOffsetProof = false;
  unsigned RequirementId = 0;
};

enum class PreparedBoundOutcome { StaticBound, RuntimeBound };
enum class BoundRequirementKind { ModularOuterSpan, ObjectContainment };

/// One requirement: trip(DomainLoop) u<= Limit.
struct PreparedBoundRequirement {
  Loop *DomainLoop = nullptr;
  APInt Limit;
  BoundRequirementKind Kind = BoundRequirementKind::ModularOuterSpan;
  unsigned RequirementId = 0;
  const SCEV *ExactTrip = nullptr;
  bool Runtime = false;
};

struct PreparedTripBound {
  PreparedBoundOutcome Outcome = PreparedBoundOutcome::StaticBound;
  SmallVector<PreparedBoundRequirement, 2> Requirements;
  const SCEV *RuntimeExactTrip = nullptr;
  APInt Wmin;
};

/// Pre-mutation state recorded by prepareRuntimeVersioning for versioning a
/// runtime-bounded selected outer loop. The vectors record the loop's structure
/// at preparation time. applyPreparedRuntimeInterchange rechecks them against
/// the live loop immediately before mutation and clears them before cloning, so
/// the structure never holds a post-version preheader, exit, or join.
struct PreparedRuntimeVersioning {
  Instruction *CheckPoint = nullptr;
  Loop *ParentLoop = nullptr;
  SmallVector<Loop *, 2> SubLoops;
  SmallVector<BasicBlock *, 16> Blocks;
  SmallVector<Instruction *, 128> Instructions;
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 32> Edges;
  SmallVector<Instruction *, 8> DefsUsedOutside;
};

/// All state needed to validate one collected-chain tail without mutating IR.
struct PreparedInterchangePlan {
  DistributableOuterEpilogue Epilogue;
  SmallVector<Loop *, 8> RoutingChain;
  SmallVector<Loop *, 8> AbsoluteAncestors;
  unsigned AbsoluteOuterLoopId = 0;
  unsigned AbsoluteInnerLoopId = 0;
  unsigned RoutingOuterLoopId = 0;
  unsigned RoutingInnerLoopId = 0;
  SmallVector<Instruction *, 16> NestMemoryInstructions;
  SmallVector<PreparedCrossPartitionDependence, 8> CrossDependences;
  PreparedTripBound TripBound;
  std::optional<PreparedRuntimeVersioning> RuntimeVersioning;
  CharMatrix FissionContextMatrix;
  CharMatrix RoutingMatrix;
  Loop *FissionContextScanRoot = nullptr;
  Loop *RoutingScanRoot = nullptr;
  unsigned FissionContextLevel = 0;
  unsigned RoutingLevel = 0;
  DependenceColumns FissionContextColumns =
      DependenceColumns::AbsoluteAncestors;
  DependenceColumns RoutingColumns = DependenceColumns::SelectedSubnest;
  SmallVector<Instruction *, 4> DropNoWrap;
  SmallVector<Instruction *, 4> DropNoInf;
  std::unique_ptr<LoopInterchangeLegality> Legality;
  bool FissionLegal = false;
  bool FissionContextMatrixComplete = false;
  bool FissionContextLegal = false;
  bool RoutingMatrixComplete = false;
  bool InterchangeLegal = false;
  bool Profitable = false;
};

static bool hasRuntimeVersionedLoopMarker(const Loop *L) {
  MDNode *LoopID = L->getLoopID();
  return LoopID &&
         findOptionMDForLoopID(LoopID, RuntimeVersionedLoopMarker) != nullptr;
}

static void markRuntimeVersionedLoop(Loop *L) {
  assert(!L->getLoopID() && "prepared loop must not carry input loop metadata");
  LLVMContext &Ctx = L->getHeader()->getContext();
  Metadata *MarkerOps[] = {MDString::get(Ctx, RuntimeVersionedLoopMarker)};
  Metadata *Marker = MDNode::get(Ctx, MarkerOps);
  SmallVector<Metadata *, 2> MDs = {nullptr, Marker};
  MDNode *LoopID = MDNode::get(Ctx, MDs);
  LoopID->replaceOperandWith(0, LoopID);
  L->setLoopID(LoopID);
}

static void debugPreparationReject(Loop *Outer, Loop *Inner, StringRef Reason) {
  LLVM_DEBUG(dbgs() << "loop-interchange: outer-epilogue preparation rejected "
                       "candidate in function '"
                    << Outer->getHeader()->getParent()->getName()
                    << "': Outer '" << Outer->getName() << "', Inner '"
                    << Inner->getName() << "': " << Reason << "\n");
  if (PrintPreparedEpiloguePlans)
    errs() << "loop-interchange: rejected function="
           << Outer->getHeader()->getParent()->getName()
           << " outer=" << Outer->getName() << " inner=" << Inner->getName()
           << " reason=" << Reason << "\n";
}

/// Emit a feature-specific remark only after discovering a material epilogue.
/// Speculative calls to ordinary legality and profitability use a null emitter.
static void rejectPreparedEpilogue(OptimizationRemarkEmitter *ORE, Loop *Outer,
                                   Loop *Inner, StringRef Reason) {
  debugPreparationReject(Outer, Inner, Reason);
  if (!ORE)
    return;
  ORE->emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, "OuterEpilogueNotDistributed",
                                    Outer->getStartLoc(), Outer->getHeader())
           << "did not distribute the discovered outer-loop epilogue: "
           << Reason;
  });
}

static bool isSupportedEpilogueValueInstruction(const Instruction &I) {
  return isa<BinaryOperator, UnaryOperator, CastInst, CmpInst, SelectInst,
             GetElementPtrInst>(I);
}

static bool hasSupportedEpilogueMetadata(const Instruction &I) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  I.getAllMetadataOtherThanDebugLoc(Metadata);
  return all_of(Metadata, [](const auto &Entry) {
    switch (Entry.first) {
    case LLVMContext::MD_tbaa:
    case LLVMContext::MD_fpmath:
    case LLVMContext::MD_range:
    case LLVMContext::MD_tbaa_struct:
    case LLVMContext::MD_invariant_load:
    case LLVMContext::MD_alias_scope:
    case LLVMContext::MD_noalias:
    case LLVMContext::MD_nontemporal:
    case LLVMContext::MD_nonnull:
    case LLVMContext::MD_dereferenceable:
    case LLVMContext::MD_dereferenceable_or_null:
    case LLVMContext::MD_prof:
    case LLVMContext::MD_align:
    case LLVMContext::MD_noundef:
    case LLVMContext::MD_noalias_addrspace:
    case LLVMContext::MD_nofpclass:
    case LLVMContext::MD_mem_cache_hint:
      return true;
    default:
      return false;
    }
  });
}

static bool isSupportedEpilogueInstruction(const Instruction &I) {
  if (!hasSupportedEpilogueMetadata(I))
    return false;
  if (const auto *Load = dyn_cast<LoadInst>(&I))
    return Load->isSimple();
  if (const auto *Store = dyn_cast<StoreInst>(&I))
    return Store->isSimple();
  return isSupportedEpilogueValueInstruction(I) && !I.mayReadFromMemory() &&
         !I.mayHaveSideEffects();
}

enum class EpilogueOperandClass {
  Invalid,
  Invariant,
  OuterIVDerived,
};

/// Classify the operand closure outside the epilogue E and record supported
/// loop-local definitions in operand-before-user order.
class EpilogueOperandClassifier {
  Loop *Outer;
  Loop *Inner;
  PHINode *OuterIV;
  const SmallPtrSetImpl<Instruction *> &EpilogueInstructions;
  SmallVectorImpl<Instruction *> &Rematerialize;
  DenseMap<Value *, EpilogueOperandClass> Cache;
  SmallPtrSet<Value *, 8> Visiting;
  SmallPtrSet<Instruction *, 8> Recorded;

public:
  EpilogueOperandClassifier(
      Loop *Outer, Loop *Inner, PHINode *OuterIV,
      const SmallPtrSetImpl<Instruction *> &EpilogueInstructions,
      SmallVectorImpl<Instruction *> &Rematerialize)
      : Outer(Outer), Inner(Inner), OuterIV(OuterIV),
        EpilogueInstructions(EpilogueInstructions),
        Rematerialize(Rematerialize) {}

  EpilogueOperandClass classify(Value *V) {
    SmallVector<Frame, 16> Stack;
    if (std::optional<EpilogueOperandClass> Resolved = enter(V, Stack))
      return *Resolved;

    while (true) {
      // Index the top rather than hold a reference: entering an operand can
      // push and reallocate.
      size_t Top = Stack.size() - 1;
      if (Stack[Top].Result != EpilogueOperandClass::Invalid &&
          Stack[Top].NextOperand < Stack[Top].I->getNumOperands()) {
        Value *Operand = Stack[Top].I->getOperand(Stack[Top].NextOperand++);
        // A pushed operand becomes the top and finalizes first, so operands
        // are visited in order and the first Invalid one stops the rest.
        if (std::optional<EpilogueOperandClass> Resolved =
                enter(Operand, Stack))
          merge(Stack[Top].Result, *Resolved);
        continue;
      }

      Frame F = Stack.pop_back_val();
      Visiting.erase(F.I);

      // Even mathematically invariant loop-local definitions need independent
      // clones. The originals remain with the nest.
      if (F.Result != EpilogueOperandClass::Invalid &&
          Recorded.insert(F.I).second)
        Rematerialize.push_back(F.I);
      Cache[F.I] = F.Result;

      if (Stack.empty())
        return F.Result;
      merge(Stack.back().Result, F.Result);
    }
  }

private:
  /// One suspended operand walk. The heap holds the traversal, so call-stack
  /// use is independent of operand-chain length.
  struct Frame {
    Instruction *I;
    /// Index of the next operand to enter.
    unsigned NextOperand;
    /// Accumulated class. Invalid overrides OuterIVDerived, which overrides
    /// Invariant.
    EpilogueOperandClass Result;
  };

  /// Resolve a leaf, a cached value, or a structurally invalid value. Otherwise
  /// push a frame and report that the caller must run it.
  std::optional<EpilogueOperandClass> enter(Value *V,
                                            SmallVectorImpl<Frame> &Stack) {
    if (V == OuterIV)
      return EpilogueOperandClass::OuterIVDerived;
    // Retention is decided by loop invariance alone; in particular, do not
    // follow an enclosing-loop PHI through its recurrence.
    if (Outer->isLoopInvariant(V))
      return EpilogueOperandClass::Invariant;

    auto CacheIt = Cache.find(V);
    if (CacheIt != Cache.end())
      return CacheIt->second;

    auto *I = dyn_cast<Instruction>(V);
    if (!I || EpilogueInstructions.contains(I) ||
        Inner->contains(I->getParent()) || isa<PHINode>(I) ||
        !isSupportedEpilogueValueInstruction(*I) ||
        !hasSupportedEpilogueMetadata(*I) || I->mayReadFromMemory() ||
        I->mayHaveSideEffects() || !Visiting.insert(V).second) {
      Cache[V] = EpilogueOperandClass::Invalid;
      return EpilogueOperandClass::Invalid;
    }

    Stack.push_back({I, 0, EpilogueOperandClass::Invariant});
    return std::nullopt;
  }

  static void merge(EpilogueOperandClass &Acc, EpilogueOperandClass C) {
    if (C == EpilogueOperandClass::Invalid)
      Acc = EpilogueOperandClass::Invalid;
    else if (C == EpilogueOperandClass::OuterIVDerived &&
             Acc != EpilogueOperandClass::Invalid)
      Acc = EpilogueOperandClass::OuterIVDerived;
  }
};

static std::optional<DistributableOuterEpilogue>
discoverDistributableOuterEpilogue(Loop *Outer, Loop *Inner,
                                   ScalarEvolution *SE, DominatorTree *DT) {
  auto Reject =
      [&](StringRef Reason) -> std::optional<DistributableOuterEpilogue> {
    debugPreparationReject(Outer, Inner, Reason);
    return std::nullopt;
  };

  if (hasRuntimeVersionedLoopMarker(Outer) ||
      hasRuntimeVersionedLoopMarker(Inner))
    return Reject("loop was already runtime-versioned");

  if (Outer->getSubLoops().size() != 1 ||
      Outer->getSubLoops().front() != Inner ||
      Inner->getParentLoop() != Outer || !Inner->isInnermost())
    return Reject("not a direct single-child/leaf-inner pair");

  SmallVector<Loop *, 2> Pair = {Outer, Inner};
  if (!Outer->isLoopSimplifyForm() || !Inner->isLoopSimplifyForm() ||
      !Outer->hasDedicatedExits() || !Inner->hasDedicatedExits() ||
      !Outer->isLCSSAForm(*DT) || !Inner->isLCSSAForm(*DT) ||
      !isComputableLoopNest(SE, Pair))
    return Reject("pair is not canonical LoopSimplify/LCSSA with computable "
                  "single exits");

  BasicBlock *OuterLatch = Outer->getLoopLatch();
  BasicBlock *InnerLatch = Inner->getLoopLatch();
  BasicBlock *InnerExit = Inner->getUniqueExitBlock();
  if (!OuterLatch || !InnerLatch || !InnerExit ||
      Outer->getExitingBlock() != OuterLatch ||
      Inner->getExitingBlock() != InnerLatch ||
      !isa<CondBrInst>(OuterLatch->getTerminator()) ||
      !isa<CondBrInst>(InnerLatch->getTerminator()))
    return Reject("latch/exit control is not the supported canonical form");

  const SCEV *InnerBTC = SE->getBackedgeTakenCount(Inner);
  if (isa<SCEVCouldNotCompute>(InnerBTC) ||
      !SE->isLoopInvariant(InnerBTC, Outer))
    return Reject("child range is not rectangular with respect to the outer "
                  "loop");

  if (Outer->getLoopID() || Inner->getLoopID())
    return Reject("loop carries metadata without a fission policy");

  PHINode *OuterIV = Outer->getInductionVariable(*SE);
  std::optional<Loop::LoopBounds> Bounds = Outer->getBounds(*SE);
  auto *OuterLatchBranch = dyn_cast<CondBrInst>(OuterLatch->getTerminator());
  ICmpInst *OuterLatchCompare = Outer->getLatchCmpInst();
  if (!OuterIV || !Bounds || !OuterLatchBranch || !OuterLatchCompare ||
      !Bounds->getStepValue())
    return Reject("outer IV/latch control cannot be mapped exactly");

  Value *InitialValue =
      OuterIV->getIncomingValueForBlock(Outer->getLoopPreheader());
  auto *NextValue =
      dyn_cast<Instruction>(OuterIV->getIncomingValueForBlock(OuterLatch));
  if (!InitialValue || !NextValue || NextValue->getParent() != OuterLatch ||
      OuterLatchCompare->getParent() != OuterLatch)
    return Reject("outer IV initial/update mapping is incomplete");

  SmallPtrSet<Value *, 4> MappedOuterControl = {OuterIV, NextValue,
                                                OuterLatchCompare};
  for (Instruction *Control :
       {NextValue, static_cast<Instruction *>(OuterLatchCompare)})
    for (Value *Operand : Control->operands())
      if (!MappedOuterControl.contains(Operand) &&
          !Outer->isLoopInvariant(Operand))
        return Reject(
            "outer latch control depends on an unsupported loop-local value");
  if (!hasSupportedEpilogueMetadata(*OuterIV) ||
      !hasSupportedEpilogueMetadata(*NextValue) ||
      !hasSupportedEpilogueMetadata(*OuterLatchCompare) ||
      !hasSupportedEpilogueMetadata(*OuterLatchBranch))
    return Reject("outer IV/latch control carries unsupported metadata");

  DistributableOuterEpilogue Epilogue;
  Epilogue.Outer = Outer;
  Epilogue.Inner = Inner;
  Epilogue.OuterControl.Induction = OuterIV;
  Epilogue.OuterControl.InitialValue = InitialValue;
  Epilogue.OuterControl.NextValue = NextValue;
  Epilogue.OuterControl.LatchCompare = OuterLatchCompare;
  Epilogue.OuterControl.LatchBranch = OuterLatchBranch;

  SmallPtrSet<BasicBlock *, 8> Visited;
  BasicBlock *Previous = InnerLatch;
  BasicBlock *Current = InnerExit;
  while (Current != OuterLatch) {
    if (!Visited.insert(Current).second || !Outer->contains(Current) ||
        Inner->contains(Current) || !DT->dominates(InnerExit, Current))
      return Reject("exit-to-latch region is not one dominated acyclic path");
    if (Current != InnerExit && Current->hasAddressTaken())
      return Reject(
          "post-inner path block cannot be collapsed after extraction");
    if (Current->getUniquePredecessor() != Previous)
      return Reject("exit-to-latch path block does not have its expected "
                    "unique predecessor");

    auto *Branch = dyn_cast<UncondBrInst>(Current->getTerminator());
    if (!Branch || Current->getUniqueSuccessor() != Branch->getSuccessor(0))
      return Reject("exit-to-latch path contains conditional or non-straight-"
                    "line control");
    if (!hasSupportedEpilogueMetadata(*Branch))
      return Reject("exit-to-latch control carries unsupported metadata");

    for (PHINode &Phi : Current->phis())
      if (Phi.getNumIncomingValues() != 1 ||
          Phi.getIncomingBlock(0) != Previous ||
          !hasSupportedEpilogueMetadata(Phi))
        return Reject("post-inner forwarding PHI is not single-input or "
                      "carries unsupported metadata");

    Epilogue.Path.push_back(Current);
    Previous = Current;
    Current = Branch->getSuccessor(0);
  }

  if (OuterLatch->getUniquePredecessor() != Previous)
    return Reject(
        "the post-inner path does not uniquely enter the outer latch");

  for (PHINode &Phi : OuterLatch->phis())
    if (Phi.getNumIncomingValues() != 1 ||
        Phi.getIncomingBlock(0) != Previous ||
        !hasSupportedEpilogueMetadata(Phi))
      return Reject("outer-latch forwarding PHI is not single-input or carries "
                    "unsupported metadata");

  for (BasicBlock *BB : Epilogue.Path)
    for (Instruction &I : *BB) {
      if (isa<PHINode>(I) || I.isTerminator())
        continue;
      if (!isSupportedEpilogueInstruction(I))
        return Reject("epilogue contains an unsupported operation or metadata");
      Epilogue.Instructions.push_back(&I);
      Epilogue.InstructionSet.insert(&I);
      if (isa<LoadInst, StoreInst>(I))
        Epilogue.MemoryInstructions.push_back(&I);
    }

  bool SawOuterControl = false;
  for (Instruction &I : *OuterLatch) {
    if (isa<PHINode>(I) || I.isTerminator())
      continue;
    if (&I == NextValue || &I == OuterLatchCompare) {
      SawOuterControl = true;
      Epilogue.OuterControl.LatchInstructions.push_back(&I);
      continue;
    }
    if (SawOuterControl)
      return Reject("outer latch has material epilogue instructions after its "
                    "IV/update/compare control slice");
    if (!isSupportedEpilogueInstruction(I))
      return Reject("epilogue contains an unsupported operation or metadata");
    Epilogue.Instructions.push_back(&I);
    Epilogue.InstructionSet.insert(&I);
    if (isa<LoadInst, StoreInst>(I))
      Epilogue.MemoryInstructions.push_back(&I);
  }
  if (Epilogue.OuterControl.LatchInstructions.size() != 2 ||
      Epilogue.OuterControl.LatchInstructions[0] != NextValue ||
      Epilogue.OuterControl.LatchInstructions[1] != OuterLatchCompare)
    return Reject("outer latch does not end in the exact IV-update/compare "
                  "control order");

  if (Epilogue.Instructions.empty() || Epilogue.MemoryInstructions.empty())
    return Reject("post-inner region has no material epilogue memory slice");

  EpilogueOperandClassifier Classifier(Outer, Inner, OuterIV,
                                       Epilogue.InstructionSet,
                                       Epilogue.OuterIVDerivedInstructions);
  for (Instruction *I : Epilogue.Instructions)
    for (Value *Operand : I->operands()) {
      if (auto *OperandI = dyn_cast<Instruction>(Operand);
          OperandI && Epilogue.InstructionSet.contains(OperandI))
        continue;
      if (Classifier.classify(Operand) == EpilogueOperandClass::Invalid)
        return Reject("epilogue consumes an inner/reduction or unsupported "
                      "external SSA value");
    }

  for (Instruction *I : Epilogue.Instructions)
    for (User *U : I->users()) {
      if (auto *UserI = dyn_cast<Instruction>(U);
          UserI && Epilogue.InstructionSet.contains(UserI))
        continue;
      if (U->isDroppable())
        continue;
      return Reject("epilogue-defined SSA value escapes the closed slice");
    }

  LLVM_DEBUG(dbgs() << "loop-interchange: discovered a closed outer-loop "
                       "epilogue in function '"
                    << Outer->getHeader()->getParent()->getName() << "' with "
                    << Epilogue.Path.size() << " path block(s), "
                    << Epilogue.Instructions.size() << " instruction(s), and "
                    << Epilogue.OuterIVDerivedInstructions.size()
                    << " rematerialized loop-local definition(s).\n");
  return std::optional<DistributableOuterEpilogue>(std::move(Epilogue));
}

static bool
collectAndCheckPreparationMemory(const DistributableOuterEpilogue &Epilogue,
                                 SmallVectorImpl<Instruction *> &NestMemory) {
  uint64_t NumInstructions = 0;
  uint64_t NumMemory = 0;

  // Use the same LoopInfo block order as validatePreparedPlan, which rebuilds
  // and compares this list.
  for (BasicBlock *BB : Epilogue.Outer->blocks())
    for (Instruction &I : *BB) {
      ++NumInstructions;
      if (!isa<LoadInst, StoreInst>(I))
        continue;
      ++NumMemory;
      if (!Epilogue.InstructionSet.contains(&I))
        NestMemory.push_back(&I);
    }

  bool LeftOverflow = false;
  bool RightOverflow = false;
  uint64_t Allowed = SaturatingMultiply<uint64_t>(
      static_cast<uint64_t>(MaxMemInstrRatio), NumInstructions, &LeftOverflow);
  uint64_t Required =
      SaturatingMultiply<uint64_t>(NumMemory, NumMemory, &RightOverflow);
  LLVM_DEBUG(dbgs() << "loop-interchange: outer-epilogue preparation memory "
                       "budget: "
                    << NumMemory << " loads/stores over " << NumInstructions
                    << " instructions.\n");
  if (LeftOverflow || RightOverflow || Allowed < Required)
    return false;

  auto IsSimpleMemory = [](Instruction *I) {
    if (auto *Load = dyn_cast<LoadInst>(I))
      return Load->isSimple();
    return cast<StoreInst>(I)->isSimple();
  };
  return all_of(NestMemory, IsSimpleMemory) &&
         all_of(Epilogue.MemoryInstructions, IsSimpleMemory);
}

/// Reject retained convergent calls, because distributing the epilogue changes
/// their control flow without a convergence proof. Convergence-control
/// intrinsics are convergent calls, so the scan also finds them.
static bool
hasConvergentOperationInRetainedNest(const DistributableOuterEpilogue &Epi) {
  for (BasicBlock *BB : Epi.Outer->blocks())
    for (Instruction &I : *BB) {
      if (Epi.InstructionSet.contains(&I))
        continue;
      if (const auto *Call = dyn_cast<CallBase>(&I);
          Call && Call->isConvergent())
        return true;
    }
  return false;
}

struct AffineArrayIndex {
  int64_t Constant = 0;
  int64_t OuterCoefficient = 0;
  int64_t InnerCoefficient = 0;

  bool add(const AffineArrayIndex &Other) {
    int64_t NewConstant;
    int64_t NewOuter;
    int64_t NewInner;
    if (AddOverflow(Constant, Other.Constant, NewConstant) ||
        AddOverflow(OuterCoefficient, Other.OuterCoefficient, NewOuter) ||
        AddOverflow(InnerCoefficient, Other.InnerCoefficient, NewInner))
      return false;
    Constant = NewConstant;
    OuterCoefficient = NewOuter;
    InnerCoefficient = NewInner;
    return true;
  }

  bool multiply(int64_t Factor) {
    int64_t NewConstant;
    int64_t NewOuter;
    int64_t NewInner;
    if (MulOverflow(Constant, Factor, NewConstant) ||
        MulOverflow(OuterCoefficient, Factor, NewOuter) ||
        MulOverflow(InnerCoefficient, Factor, NewInner))
      return false;
    Constant = NewConstant;
    OuterCoefficient = NewOuter;
    InnerCoefficient = NewInner;
    return true;
  }

  bool operator==(const AffineArrayIndex &Other) const {
    return Constant == Other.Constant &&
           OuterCoefficient == Other.OuterCoefficient &&
           InnerCoefficient == Other.InnerCoefficient;
  }
};

/// The accessible byte extent proves storage containment, not row stride.
struct FlattenedByteAccess {
  Value *Base = nullptr;
  Type *ElementType = nullptr;
  unsigned AddressSpace = 0;
  uint64_t ElementSize = 0;
  uint64_t KnownAccessibleBytes = 0;
  AffineArrayIndex ByteOffset;
};

static std::optional<int64_t> getSignedSCEVConstant(const SCEV *S) {
  const auto *C = dyn_cast<SCEVConstant>(S);
  if (!C)
    return std::nullopt;
  return C->getAPInt().trySExtValue();
}

/// Decompose an exact signed-i64 affine expression over the selected outer and
/// inner iteration numbers.
static std::optional<AffineArrayIndex>
decomposeAffineByteOffset(const SCEV *S, Loop *Outer, Loop *Inner,
                          ScalarEvolution &SE) {
  if (!S->getType()->isIntegerTy(64))
    return std::nullopt;
  if (std::optional<int64_t> C = getSignedSCEVConstant(S))
    return AffineArrayIndex{*C, 0, 0};

  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (!AR->isAffine() || (AR->getLoop() != Outer && AR->getLoop() != Inner))
      return std::nullopt;
    std::optional<AffineArrayIndex> Start =
        decomposeAffineByteOffset(AR->getStart(), Outer, Inner, SE);
    std::optional<AffineArrayIndex> Step =
        decomposeAffineByteOffset(AR->getStepRecurrence(SE), Outer, Inner, SE);
    if (!Start || !Step || Step->OuterCoefficient != 0 ||
        Step->InnerCoefficient != 0)
      return std::nullopt;
    AffineArrayIndex IterationTerm;
    if (AR->getLoop() == Outer)
      IterationTerm.OuterCoefficient = Step->Constant;
    else
      IterationTerm.InnerCoefficient = Step->Constant;
    if (!Start->add(IterationTerm))
      return std::nullopt;
    return Start;
  }

  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    AffineArrayIndex Result;
    for (const SCEV *Operand : Add->operands()) {
      std::optional<AffineArrayIndex> Part =
          decomposeAffineByteOffset(Operand, Outer, Inner, SE);
      if (!Part || !Result.add(*Part))
        return std::nullopt;
    }
    return Result;
  }

  if (const auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
    int64_t ConstantFactor = 1;
    std::optional<AffineArrayIndex> VaryingFactor;
    for (const SCEV *Operand : Mul->operands()) {
      if (std::optional<int64_t> C = getSignedSCEVConstant(Operand)) {
        int64_t Product;
        if (MulOverflow(ConstantFactor, *C, Product))
          return std::nullopt;
        ConstantFactor = Product;
        continue;
      }
      if (VaryingFactor)
        return std::nullopt;
      VaryingFactor = decomposeAffineByteOffset(Operand, Outer, Inner, SE);
      if (!VaryingFactor)
        return std::nullopt;
    }
    if (!VaryingFactor)
      return AffineArrayIndex{ConstantFactor, 0, 0};
    if (!VaryingFactor->multiply(ConstantFactor))
      return std::nullopt;
    return VaryingFactor;
  }

  return std::nullopt;
}

static std::optional<uint64_t> getKnownAccessibleBytes(Value *Base,
                                                       const DataLayout &DL) {
  if (auto *GV = dyn_cast<GlobalVariable>(Base)) {
    if (!GV->getValueType()->isSized() || GV->hasExternalWeakLinkage())
      return std::nullopt;
    uint64_t DeclaredMinimumBytes = GV->getGlobalSize(DL);
    if (DeclaredMinimumBytes == 0)
      return std::nullopt;
    return DeclaredMinimumBytes;
  }

  ObjectSizeOpts Opts;
  Opts.NullIsUnknownSize = true;
  std::optional<TypeSize> Size =
      getBaseObjectSize(Base, DL, /*TLI=*/nullptr, Opts);
  if (Size && !Size->isScalable() && Size->getFixedValue() != 0)
    return Size->getFixedValue();

  bool CanBeNull = true;
  uint64_t DereferenceableBytes = Base->getPointerDereferenceableBytes(
      DL, CanBeNull, /*CanBeFreed=*/nullptr);
  if (DereferenceableBytes == 0 || CanBeNull)
    return std::nullopt;
  return DereferenceableBytes;
}

/// Verify that collectOffset's modulo-2^64 result represents the exact
/// mathematical byte contribution of this GEP.
static bool collectExactGEPByteOffset(
    GEPOperator &GEP, const DataLayout &DL,
    SmallMapVector<Value *, int64_t, 4> &ExactVariableOffsets,
    int64_t &ExactConstantOffset) {
  SmallMapVector<Value *, APInt, 4> CollectedVariableOffsets;
  APInt CollectedConstantOffset(64, 0);
  if (!GEP.collectOffset(DL, 64, CollectedVariableOffsets,
                         CollectedConstantOffset))
    return false;

  ExactConstantOffset = 0;
  for (gep_type_iterator GTI = gep_type_begin(&GEP), GTE = gep_type_end(&GEP);
       GTI != GTE; ++GTI) {
    Value *Index = GTI.getOperand();
    if (StructType *Struct = GTI.getStructTypeOrNull()) {
      auto *ConstantIndex = dyn_cast<ConstantInt>(Index);
      if (!ConstantIndex)
        return false;
      uint64_t FieldOffset = DL.getStructLayout(Struct)->getElementOffset(
          ConstantIndex->getZExtValue());
      if (FieldOffset >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return false;
      int64_t NewConstant;
      if (AddOverflow(ExactConstantOffset, static_cast<int64_t>(FieldOffset),
                      NewConstant))
        return false;
      ExactConstantOffset = NewConstant;
      continue;
    }

    TypeSize StrideSize = GTI.getSequentialElementStride(DL);
    if (StrideSize.isScalable() ||
        StrideSize.getFixedValue() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    int64_t Stride = static_cast<int64_t>(StrideSize.getFixedValue());
    if (auto *ConstantIndex = dyn_cast<ConstantInt>(Index)) {
      std::optional<int64_t> SignedIndex =
          ConstantIndex->getValue().trySExtValue();
      int64_t Term;
      int64_t NewConstant;
      if (!SignedIndex || MulOverflow(*SignedIndex, Stride, Term) ||
          AddOverflow(ExactConstantOffset, Term, NewConstant))
        return false;
      ExactConstantOffset = NewConstant;
      continue;
    }

    if (Stride == 0)
      continue;
    auto It = ExactVariableOffsets.insert({Index, 0}).first;
    int64_t NewScale;
    if (AddOverflow(It->second, Stride, NewScale))
      return false;
    It->second = NewScale;
  }

  std::optional<int64_t> CollectedConstant =
      CollectedConstantOffset.trySExtValue();
  if (!CollectedConstant || *CollectedConstant != ExactConstantOffset ||
      CollectedVariableOffsets.size() != ExactVariableOffsets.size())
    return false;
  for (const auto &[Variable, ExactScale] : ExactVariableOffsets) {
    auto It = CollectedVariableOffsets.find(Variable);
    if (It == CollectedVariableOffsets.end())
      return false;
    std::optional<int64_t> CollectedScale = It->second.trySExtValue();
    if (!CollectedScale || *CollectedScale != ExactScale)
      return false;
  }
  return true;
}

/// Parse instruction and constant-expression GEPs into one exact byte offset.
static std::optional<FlattenedByteAccess>
parseFlattenedByteAccess(Instruction *MemoryInstruction, Loop *Outer,
                         Loop *Inner, ScalarEvolution &SE) {
  Value *Pointer = getLoadStorePointerOperand(MemoryInstruction);
  if (!Pointer || !Pointer->getType()->isPointerTy())
    return std::nullopt;

  const DataLayout &DL = MemoryInstruction->getFunction()->getDataLayout();
  unsigned AddressSpace = getLoadStoreAddressSpace(MemoryInstruction);
  if (DL.getIndexTypeSizeInBits(Pointer->getType()) != 64)
    return std::nullopt;

  Value *OriginalPointer = Pointer;
  AffineArrayIndex ByteOffset;
  unsigned GEPCount = 0;
  while (auto *GEP = dyn_cast<GEPOperator>(Pointer)) {
    if (++GEPCount > 16 || GEP->getPointerAddressSpace() != AddressSpace ||
        DL.getIndexTypeSizeInBits(GEP->getPointerOperand()->getType()) != 64)
      return std::nullopt;

    SmallMapVector<Value *, int64_t, 4> VariableOffsets;
    int64_t ConstantOffset;
    if (!collectExactGEPByteOffset(*GEP, DL, VariableOffsets, ConstantOffset) ||
        !ByteOffset.add(AffineArrayIndex{ConstantOffset, 0, 0}))
      return std::nullopt;

    for (const auto &[Variable, Scale] : VariableOffsets) {
      std::optional<AffineArrayIndex> Part =
          decomposeAffineByteOffset(SE.getSCEV(Variable), Outer, Inner, SE);
      if (!Part || !Part->multiply(Scale) || !ByteOffset.add(*Part))
        return std::nullopt;
    }
    Pointer = GEP->getPointerOperand();
  }

  if (GEPCount == 0 || !Pointer->getType()->isPointerTy() ||
      Pointer->getType()->getPointerAddressSpace() != AddressSpace ||
      !Outer->isLoopInvariant(Pointer) ||
      getUnderlyingObject(OriginalPointer, /*MaxLookup=*/16) != Pointer)
    return std::nullopt;

  Type *ElementType = getLoadStoreType(MemoryInstruction);
  if (!ElementType->isSized())
    return std::nullopt;
  TypeSize StoreSize = DL.getTypeStoreSize(ElementType);
  TypeSize AllocSize = DL.getTypeAllocSize(ElementType);
  if (StoreSize.isScalable() || AllocSize.isScalable() ||
      StoreSize.getFixedValue() == 0 ||
      StoreSize.getFixedValue() != AllocSize.getFixedValue())
    return std::nullopt;

  std::optional<uint64_t> AccessibleBytes =
      getKnownAccessibleBytes(Pointer, DL);
  if (!AccessibleBytes || *AccessibleBytes == 0)
    return std::nullopt;

  FlattenedByteAccess Access;
  Access.Base = Pointer;
  Access.ElementType = ElementType;
  Access.AddressSpace = AddressSpace;
  Access.ElementSize = AllocSize.getFixedValue();
  Access.KnownAccessibleBytes = *AccessibleBytes;
  Access.ByteOffset = ByteOffset;
  return Access;
}

static std::optional<std::pair<int64_t, int64_t>>
getAffineIndexRange(const AffineArrayIndex &Index, uint64_t OuterTripCount,
                    uint64_t InnerTripCount) {
  if (OuterTripCount == 0 || InnerTripCount == 0 ||
      OuterTripCount >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      InnerTripCount >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;

  int64_t Minimum = Index.Constant;
  int64_t Maximum = Index.Constant;
  auto Accumulate = [&](int64_t Coefficient, uint64_t TripCount) {
    int64_t LastIteration = static_cast<int64_t>(TripCount - 1);
    int64_t Term;
    if (MulOverflow(Coefficient, LastIteration, Term))
      return false;
    if (Coefficient < 0)
      return !AddOverflow(Minimum, Term, Minimum);
    return !AddOverflow(Maximum, Term, Maximum);
  };
  if (!Accumulate(Index.OuterCoefficient, OuterTripCount) ||
      !Accumulate(Index.InnerCoefficient, InnerTripCount))
    return std::nullopt;
  return std::pair<int64_t, int64_t>(Minimum, Maximum);
}

static bool isByteAccessRangeWithinObject(const FlattenedByteAccess &Access,
                                          uint64_t OuterTripCount,
                                          uint64_t InnerTripCount) {
  std::optional<std::pair<int64_t, int64_t>> Range =
      getAffineIndexRange(Access.ByteOffset, OuterTripCount, InnerTripCount);
  if (!Range)
    return false;
  return loop_interchange_utils::isFixedByteRangeWithinObject(
      Range->first, Range->second, Access.ElementSize,
      Access.KnownAccessibleBytes);
}

/// Divide raw byte coefficients by element size exactly once.
static std::optional<AffineArrayIndex>
normalizeByteOffset(const FlattenedByteAccess &Access) {
  if (Access.ElementSize == 0 ||
      Access.ElementSize >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  int64_t ElementSize = static_cast<int64_t>(Access.ElementSize);
  const AffineArrayIndex &Bytes = Access.ByteOffset;
  if (Bytes.Constant % ElementSize != 0 ||
      Bytes.OuterCoefficient % ElementSize != 0 ||
      Bytes.InnerCoefficient % ElementSize != 0)
    return std::nullopt;
  return AffineArrayIndex{Bytes.Constant / ElementSize,
                          Bytes.OuterCoefficient / ElementSize,
                          Bytes.InnerCoefficient / ElementSize};
}

static int64_t positiveModulo(int64_t Value, int64_t Modulus) {
  assert(Modulus > 0 && "expected a positive modulus");
  int64_t Result = Value % Modulus;
  return Result < 0 ? Result + Modulus : Result;
}

/// Prove that the nest access N and the epilogue access E can collide only
/// within one outer iteration. N steps by +/-1 element per outer iteration and
/// by +/-W per inner iteration. E is inner-invariant. Both share an object and
/// agree modulo W in outer coefficient and constant. Equal addresses then
/// imply equal outer iterations modulo W, hence equal iterations when the
/// outer trip count is at most W. Containment is checked with W trips in each
/// loop, and both trip bounds are recorded as requirements.
static bool proveSameOuterIterationByByteOffset(
    Instruction *NestInstruction, Instruction *EpilogueInstruction, Loop *Outer,
    Loop *Inner, ScalarEvolution &SE, unsigned RequirementId,
    SmallVectorImpl<PreparedBoundRequirement> &Requirements) {
  std::optional<FlattenedByteAccess> Nest =
      parseFlattenedByteAccess(NestInstruction, Outer, Inner, SE);
  std::optional<FlattenedByteAccess> Epilogue =
      parseFlattenedByteAccess(EpilogueInstruction, Outer, Inner, SE);
  if (!Nest || !Epilogue || Nest->Base != Epilogue->Base ||
      Nest->ElementType != Epilogue->ElementType ||
      Nest->AddressSpace != Epilogue->AddressSpace ||
      Nest->ElementSize != Epilogue->ElementSize)
    return false;

  std::optional<AffineArrayIndex> NestElements = normalizeByteOffset(*Nest);
  std::optional<AffineArrayIndex> EpilogueElements =
      normalizeByteOffset(*Epilogue);
  if (!NestElements || !EpilogueElements)
    return false;
  std::optional<uint64_t> AbsInner =
      loop_interchange_utils::checkedAbsToUnsigned(
          NestElements->InnerCoefficient);
  if (!AbsInner || *AbsInner == 0)
    return false;
  uint64_t Width = *AbsInner;
  if (Width > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  int64_t SignedWidth = static_cast<int64_t>(Width);

  if ((NestElements->OuterCoefficient != 1 &&
       NestElements->OuterCoefficient != -1) ||
      (NestElements->InnerCoefficient != SignedWidth &&
       NestElements->InnerCoefficient != -SignedWidth) ||
      EpilogueElements->InnerCoefficient != 0)
    return false;

  if (positiveModulo(EpilogueElements->OuterCoefficient, SignedWidth) !=
          positiveModulo(NestElements->OuterCoefficient, SignedWidth) ||
      positiveModulo(EpilogueElements->Constant, SignedWidth) !=
          positiveModulo(NestElements->Constant, SignedWidth))
    return false;

  if (!isByteAccessRangeWithinObject(*Nest, Width, Width) ||
      !isByteAccessRangeWithinObject(*Epilogue, Width, Width))
    return false;

  Requirements.push_back({Outer, APInt(64, Width),
                          BoundRequirementKind::ModularOuterSpan, RequirementId,
                          nullptr, false});
  Requirements.push_back({Inner, APInt(64, Width),
                          BoundRequirementKind::ObjectContainment,
                          RequirementId, nullptr, false});

  LLVM_DEBUG(
      dbgs() << "loop-interchange: flattened byte-offset proof accepted an N/E "
                "pair:\n"
             << " N: " << *NestInstruction << "\n"
             << " E: " << *EpilogueInstruction << "\n"
             << " width = " << Width
             << ", element bytes = " << Nest->ElementSize
             << ", bounded modular equality implies equal outer iteration.\n");
  return true;
}

/// Query raw dependences from the retained nest N to the epilogue E,
/// independently of both normalized matrices.
static bool proveNestToEpilogueDependences(
    ArrayRef<Instruction *> NestMemory, ArrayRef<Instruction *> EpilogueMemory,
    Loop *Outer, Loop *Inner, unsigned AbsoluteOuterLoopId, DependenceInfo *DI,
    ScalarEvolution *SE,
    SmallVectorImpl<PreparedCrossPartitionDependence> &Prepared,
    SmallVectorImpl<PreparedBoundRequirement> &Requirements) {
  const unsigned OuterLevel = AbsoluteOuterLoopId + 1;
  unsigned NextRequirementId = 0;
  for (Instruction *NestI : NestMemory)
    for (Instruction *EpilogueI : EpilogueMemory) {
      if (isa<LoadInst>(NestI) && isa<LoadInst>(EpilogueI))
        continue;

      std::unique_ptr<Dependence> D =
          DI->depends(NestI, EpilogueI, /*UnderRuntimeAssumptions=*/false);
      if (!D)
        continue;

      // Keep the raw N-to-E order. Normalization can swap the endpoints.
      PreparedCrossPartitionDependence Summary;
      if (!D->isOrdered() || D->isConfused() ||
          !D->getRuntimeAssumptions().isAlwaysTrue() || D->getSrc() != NestI ||
          D->getDst() != EpilogueI || D->isDirectionNegative() ||
          D->getLevels() < OuterLevel) {
        LLVM_DEBUG(
            dbgs() << "loop-interchange: rejected an unknown, reversed, "
                      "confused, or assumption-bearing N/E dependence.\n");
        return false;
      }

      bool DecisiveForward = false;
      bool NeedsByteOffsetProof = false;
      for (unsigned Level = 1; Level <= OuterLevel; ++Level) {
        unsigned Direction = D->getDirection(Level);
        if (Level < OuterLevel) {
          if (Direction == Dependence::DVEntry::LT) {
            DecisiveForward = true;
            break;
          }
          if (Direction == Dependence::DVEntry::GT ||
              Direction == Dependence::DVEntry::GE)
            return false;
          // EQ, ALL, LE, and NE preserve ancestor iteration order.
          if (Direction != Dependence::DVEntry::EQ &&
              Direction != Dependence::DVEntry::ALL &&
              Direction != Dependence::DVEntry::LE &&
              Direction != Dependence::DVEntry::NE)
            return false;
          continue;
        }

        if (Direction == Dependence::DVEntry::EQ)
          continue;
        if (Direction == Dependence::DVEntry::LT) {
          DecisiveForward = true;
          break;
        }
        if (Direction == Dependence::DVEntry::ALL &&
            D->getLevels() == OuterLevel) {
          NeedsByteOffsetProof = true;
          break;
        }
        return false;
      }

      if (!DecisiveForward && !NeedsByteOffsetProof &&
          D->getLevels() != OuterLevel)
        return false;
      if (NeedsByteOffsetProof) {
        Summary.RequirementId = NextRequirementId;
        if (!proveSameOuterIterationByByteOffset(NestI, EpilogueI, Outer, Inner,
                                                 *SE, NextRequirementId,
                                                 Requirements))
          return false;
        Summary.UsedByteOffsetProof = true;
        ++NextRequirementId;
      }
      Prepared.push_back(Summary);
    }
  return true;
}

/// Compare trip counts, not backedge counts. Every requirement not discharged
/// by the constant maximum or a known predicate must have the selected outer
/// loop's exact trip count.
static std::optional<PreparedTripBound> resolvePreparedTripBound(
    SmallVectorImpl<PreparedBoundRequirement> &Requirements,
    Loop *SelectedOuter, ScalarEvolution &SE, const TargetTransformInfo *TTI,
    Instruction *CheckPoint, const char *&RejectReason) {
  PreparedTripBound Bound;
  LLVMContext &Ctx = SE.getContext();
  SCEVExpander Expander(SE, "loop-interchange-bound");

  const SCEV *RuntimeTrip = nullptr;
  const SCEV *SelectedOuterExactTrip = nullptr;
  bool HasRuntime = false;
  APInt Wmin;

  auto MatchWidth =
      [&](const SCEV *Trip,
          const APInt &W) -> std::pair<const SCEV *, const SCEV *> {
    unsigned TripWidth = Trip->getType()->getIntegerBitWidth();
    unsigned CommonWidth = std::max(TripWidth, W.getBitWidth());
    const SCEV *WideTrip =
        CommonWidth > TripWidth
            ? SE.getZeroExtendExpr(Trip, Type::getIntNTy(Ctx, CommonWidth))
            : Trip;
    return {WideTrip, SE.getConstant(W.zext(CommonWidth))};
  };

  for (PreparedBoundRequirement &Req : Requirements) {
    Loop *L = Req.DomainLoop;
    const APInt &W = Req.Limit;

    const SCEV *MaxExit = SE.getConstantMaxBackedgeTakenCount(L);
    if (!isa<SCEVCouldNotCompute>(MaxExit)) {
      const SCEV *MaxTrip = SE.getTripCountFromExitCount(MaxExit);
      if (const auto *C = dyn_cast<SCEVConstant>(MaxTrip))
        if (loop_interchange_utils::unsignedLEWithZeroExtend(C->getAPInt(), W))
          continue;
    }

    const SCEV *ExactExit = SE.getBackedgeTakenCount(L);
    if (isa<SCEVCouldNotCompute>(ExactExit)) {
      RejectReason = "exact-backedge-not-computable";
      return std::nullopt;
    }
    const SCEV *ExactTrip = SE.getTripCountFromExitCount(ExactExit);
    Req.ExactTrip = ExactTrip;
    auto [WideTrip, WideW] = MatchWidth(ExactTrip, W);

    if (SE.isKnownPredicateAt(ICmpInst::ICMP_ULE, WideTrip, WideW, CheckPoint))
      continue;
    if (SE.isKnownPredicateAt(ICmpInst::ICMP_UGT, WideTrip, WideW,
                              CheckPoint)) {
      RejectReason = "known-unsafe-trip-exceeds-bound";
      return std::nullopt;
    }

    if (!SE.isLoopInvariant(ExactTrip, SelectedOuter) ||
        !SE.isAvailableAtLoopEntry(ExactTrip, SelectedOuter)) {
      RejectReason = "runtime-trip-not-available-at-preheader";
      return std::nullopt;
    }
    if (!Expander.isSafeToExpandAt(ExactTrip, CheckPoint)) {
      RejectReason = "runtime-trip-unsafe-to-expand";
      return std::nullopt;
    }
    if (!TTI ||
        Expander.isHighCostExpansion({ExactTrip}, L, RuntimeTripExpansionBudget,
                                     TTI, CheckPoint)) {
      RejectReason = "runtime-trip-expansion-too-costly";
      return std::nullopt;
    }
    Req.Runtime = true;

    if (!SelectedOuterExactTrip) {
      if (L == SelectedOuter) {
        SelectedOuterExactTrip = ExactTrip;
      } else {
        const SCEV *SelectedOuterExit = SE.getBackedgeTakenCount(SelectedOuter);
        if (isa<SCEVCouldNotCompute>(SelectedOuterExit)) {
          RejectReason = "selected-outer-exact-backedge-not-computable";
          return std::nullopt;
        }
        SelectedOuterExactTrip =
            SE.getTripCountFromExitCount(SelectedOuterExit);
      }
    }
    if (ExactTrip != SelectedOuterExactTrip) {
      RejectReason = Req.Kind == BoundRequirementKind::ObjectContainment
                         ? "second-distinct-runtime-trip-object-containment"
                         : "second-distinct-runtime-trip-modular-outer-span";
      return std::nullopt;
    }

    if (HasRuntime) {
      if (RuntimeTrip != ExactTrip) {
        RejectReason = Req.Kind == BoundRequirementKind::ObjectContainment
                           ? "second-distinct-runtime-trip-object-containment"
                           : "second-distinct-runtime-trip-modular-outer-span";
        return std::nullopt;
      }
      Wmin = APIntOps::umin(Wmin, W);
    } else {
      HasRuntime = true;
      RuntimeTrip = SelectedOuterExactTrip;
      Wmin = W;
    }
  }

  Bound.Requirements.assign(Requirements.begin(), Requirements.end());
  if (HasRuntime) {
    Bound.Outcome = PreparedBoundOutcome::RuntimeBound;
    Bound.RuntimeExactTrip = RuntimeTrip;
    Bound.Wmin = Wmin;
  }
  return Bound;
}

/// Dependence levels are absolute, including ancestors outside the current
/// LoopNest root.
static std::optional<SmallVector<Loop *, 8>>
buildCandidateAncestorChain(Loop *Outer, Loop *Inner) {
  SmallVector<Loop *, 8> InnerFirst;
  for (Loop *L = Inner; L; L = L->getParentLoop())
    InnerFirst.push_back(L);

  SmallVector<Loop *, 8> Ancestors(InnerFirst.rbegin(), InnerFirst.rend());
  if (Ancestors.size() < 2 || Ancestors.back() != Inner ||
      Ancestors[Ancestors.size() - 2] != Outer)
    return std::nullopt;
  return Ancestors;
}

static bool hasSafeRuntimeVersioningControl(const Loop *L) {
  for (BasicBlock *BB : L->blocks()) {
    if (!isa<CondBrInst, UncondBrInst>(BB->getTerminator()))
      return false;
    for (Instruction &I : *BB) {
      if (I.getType()->isTokenLikeTy())
        return false;
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        if (!Load->isSimple())
          return false;
        continue;
      }
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        if (!Store->isSimple())
          return false;
        continue;
      }
      if (auto *Call = dyn_cast<CallBase>(&I)) {
        if (!isa<CallInst>(Call) || Call->cannotDuplicate() ||
            Call->isConvergent() || Call->hasDeoptState() || Call->mayThrow() ||
            !Call->willReturn() || Call->mayReadOrWriteMemory() ||
            Call->mayHaveSideEffects())
          return false;
        continue;
      }
      if (I.mayReadOrWriteMemory() || I.mayThrow())
        return false;
    }
  }
  return true;
}

static bool
runtimeRequirementsAreConsistent(const PreparedTripBound &TripBound) {
  if (TripBound.Outcome != PreparedBoundOutcome::RuntimeBound ||
      !TripBound.RuntimeExactTrip || TripBound.Wmin.isZero())
    return false;

  bool SawRuntime = false;
  APInt ComputedWmin;
  for (const PreparedBoundRequirement &Req : TripBound.Requirements) {
    if (!Req.Runtime)
      continue;
    if (!Req.ExactTrip || Req.ExactTrip != TripBound.RuntimeExactTrip ||
        Req.Limit.isZero() ||
        Req.Limit.getBitWidth() != TripBound.Wmin.getBitWidth())
      return false;
    if (SawRuntime)
      ComputedWmin = APIntOps::umin(ComputedWmin, Req.Limit);
    else {
      SawRuntime = true;
      ComputedWmin = Req.Limit;
    }
  }
  return SawRuntime && ComputedWmin == TripBound.Wmin;
}

static bool hasIrreducibleRuntimeVersioningControl(Loop *L,
                                                   const LoopInfo &LI) {
  LoopBlocksRPO RPOT(L);
  RPOT.perform(&LI);
  return containsIrreducibleCFG<const BasicBlock *>(RPOT, LI);
}

/// Return true unless a recorded live-out has an unreachable use outside the
/// loop. findDefsUsedOutsideOfLoop records a definition as live-out when any
/// user lies outside the loop, while Loop::isLCSSAForm attributes a PHI use to
/// its incoming block and ignores uses in blocks unreachable from entry. A
/// definition whose outside uses are all ignored by LCSSA lacks an exit PHI,
/// so LoopVersioning would synthesize a join PHI whose incoming value need not
/// dominate the exiting block. The predicate applies LCSSA's own attribution
/// here, so the set passed to versionLoop is exactly the set covered by LCSSA
/// exit PHIs. When this predicate returns false, the caller rejects the
/// candidate before any mutation.
static bool liveOutsMatchLCSSA(ArrayRef<Instruction *> Defs, const Loop *L,
                               const DominatorTree &DT) {
  for (Instruction *Def : Defs) {
    for (const Use &U : Def->uses()) {
      auto *UserInst = cast<Instruction>(U.getUser());
      BasicBlock *UserBB = UserInst->getParent();
      if (auto *PN = dyn_cast<PHINode>(UserInst))
        UserBB = PN->getIncomingBlock(U);
      if (!L->contains(UserBB) && !DT.isReachableFromEntry(UserBB))
        return false;
    }
  }
  return true;
}

static bool prepareRuntimeVersioning(PreparedInterchangePlan &Plan,
                                     ScalarEvolution &SE, LoopInfo &LI,
                                     DominatorTree &DT,
                                     const TargetTransformInfo &TTI,
                                     const char *&RejectReason) {
  assert(Plan.TripBound.Outcome == PreparedBoundOutcome::RuntimeBound &&
         "runtime versioning checks require a runtime-bound plan");
  Loop *Outer = Plan.Epilogue.Outer;
  Loop *Inner = Plan.Epilogue.Inner;
  if (!Outer || !Inner || !runtimeRequirementsAreConsistent(Plan.TripBound)) {
    RejectReason = "runtime-bound requirements are inconsistent";
    return false;
  }
  if (!EnableOuterEpilogueRuntimeVersioning) {
    RejectReason = "runtime outer-epilogue versioning is disabled";
    return false;
  }
  if (Outer->getHeader()->getParent()->hasOptSize()) {
    RejectReason =
        "runtime versioning is disabled for size-optimized functions";
    return false;
  }
  if (hasRuntimeVersionedLoopMarker(Outer) ||
      hasRuntimeVersionedLoopMarker(Inner)) {
    RejectReason = "loop was already runtime-versioned";
    return false;
  }
  if (!Outer->isSafeToCloneConditionally(DT)) {
    RejectReason = "selected outer loop is not safe to clone";
    return false;
  }

  BasicBlock *Preheader = Outer->getLoopPreheader();
  if (!Preheader || !Outer->getUniqueExitBlock() || !Outer->getExitingBlock() ||
      !Outer->isLoopSimplifyForm() || !Outer->hasDedicatedExits() ||
      !Outer->isRecursivelyLCSSAForm(DT, LI) ||
      hasIrreducibleRuntimeVersioningControl(Outer, LI)) {
    RejectReason =
        "selected outer loop is not canonical single-exit recursive LCSSA";
    return false;
  }
  if (!hasSafeRuntimeVersioningControl(Outer)) {
    RejectReason = "selected outer loop contains unsafe versioning control";
    return false;
  }
  if (Outer->getNumBlocks() > MaxRuntimeVersioningBlocks) {
    RejectReason = "selected outer loop exceeds the runtime block limit";
    return false;
  }

  Instruction *CheckPoint = Preheader->getTerminator();
  const SCEV *ExactTrip = Plan.TripBound.RuntimeExactTrip;
  SCEVExpander Expander(SE, "loop-interchange-bound");
  if (!SE.isLoopInvariant(ExactTrip, Outer) ||
      !SE.isAvailableAtLoopEntry(ExactTrip, Outer) ||
      !Expander.isSafeToExpandAt(ExactTrip, CheckPoint)) {
    RejectReason = "runtime trip is unavailable or unsafe to expand";
    return false;
  }
  if (Expander.isHighCostExpansion(
          {ExactTrip}, Outer, RuntimeTripExpansionBudget, &TTI, CheckPoint)) {
    RejectReason = "runtime trip expansion exceeds the cost budget";
    return false;
  }

  PreparedRuntimeVersioning Runtime;
  Runtime.CheckPoint = CheckPoint;
  Runtime.ParentLoop = Outer->getParentLoop();
  Runtime.SubLoops.append(Outer->getSubLoops().begin(),
                          Outer->getSubLoops().end());

  unsigned NonDebugInstructions = 0;
  for (BasicBlock *BB : Outer->blocks()) {
    Runtime.Blocks.push_back(BB);
    for (Instruction &I : *BB) {
      Runtime.Instructions.push_back(&I);
      if (!I.isDebugOrPseudoInst() &&
          ++NonDebugInstructions > MaxRuntimeVersioningInstructions) {
        RejectReason =
            "selected outer loop exceeds the runtime instruction limit";
        return false;
      }
    }
    for (BasicBlock *Successor : successors(BB))
      Runtime.Edges.emplace_back(BB, Successor);
  }
  Runtime.DefsUsedOutside = findDefsUsedOutsideOfLoop(Outer);
  if (!liveOutsMatchLCSSA(Runtime.DefsUsedOutside, Outer, DT)) {
    RejectReason =
        "selected outer loop has a live-out used in unreachable code";
    return false;
  }
  Plan.RuntimeVersioning = std::move(Runtime);
  return true;
}

static bool isRuntimeVersioningStateCurrent(const PreparedInterchangePlan &Plan,
                                            const DominatorTree &DT,
                                            const LoopInfo &LI) {
  if (!Plan.RuntimeVersioning)
    return false;
  const PreparedRuntimeVersioning &Runtime = *Plan.RuntimeVersioning;
  Loop *Outer = Plan.Epilogue.Outer;
  Loop *Inner = Plan.Epilogue.Inner;
  if (!EnableOuterEpilogueRuntimeVersioning || !Outer || !Inner ||
      Runtime.CheckPoint == nullptr ||
      Outer->getHeader()->getParent()->hasOptSize() ||
      Runtime.ParentLoop != Outer->getParentLoop() ||
      hasRuntimeVersionedLoopMarker(Outer) ||
      hasRuntimeVersionedLoopMarker(Inner) ||
      !Outer->isSafeToCloneConditionally(DT) || !Outer->isLoopSimplifyForm() ||
      !Outer->hasDedicatedExits() || !Outer->getUniqueExitBlock() ||
      !Outer->getExitingBlock() || !Outer->isRecursivelyLCSSAForm(DT, LI) ||
      hasIrreducibleRuntimeVersioningControl(Outer, LI))
    return false;

  BasicBlock *Preheader = Outer->getLoopPreheader();
  if (!Preheader || Runtime.CheckPoint != Preheader->getTerminator() ||
      !llvm::equal(Runtime.SubLoops, Outer->getSubLoops()) ||
      Outer->getNumBlocks() > MaxRuntimeVersioningBlocks ||
      !hasSafeRuntimeVersioningControl(Outer))
    return false;

  unsigned NonDebugInstructions = 0;
  unsigned BlockIndex = 0;
  unsigned InstructionIndex = 0;
  unsigned EdgeIndex = 0;
  for (BasicBlock *BB : Outer->blocks()) {
    if (BlockIndex == Runtime.Blocks.size() ||
        Runtime.Blocks[BlockIndex++] != BB)
      return false;
    for (Instruction &I : *BB) {
      if (InstructionIndex == Runtime.Instructions.size() ||
          Runtime.Instructions[InstructionIndex++] != &I)
        return false;
      if (!I.isDebugOrPseudoInst() &&
          ++NonDebugInstructions > MaxRuntimeVersioningInstructions)
        return false;
    }
    for (BasicBlock *Successor : successors(BB)) {
      if (EdgeIndex == Runtime.Edges.size() ||
          Runtime.Edges[EdgeIndex++] != std::make_pair(BB, Successor))
        return false;
    }
  }
  if (BlockIndex != Runtime.Blocks.size() ||
      InstructionIndex != Runtime.Instructions.size() ||
      EdgeIndex != Runtime.Edges.size())
    return false;
  SmallVector<Instruction *, 8> CurrentDefs = findDefsUsedOutsideOfLoop(Outer);
  return liveOutsMatchLCSSA(CurrentDefs, Outer, DT) &&
         llvm::equal(Runtime.DefsUsedOutside, CurrentDefs);
}

static void forgetAffectedTopmostLoops(ScalarEvolution &SE,
                                       ArrayRef<Loop *> Loops) {
  SmallPtrSet<Loop *, 4> TopmostLoops;
  for (Loop *L : Loops) {
    if (!L)
      continue;
    while (L->getParentLoop())
      L = L->getParentLoop();
    TopmostLoops.insert(L);
  }
  for (Loop *L : TopmostLoops)
    SE.forgetTopmostLoop(L);
  SE.forgetBlockAndLoopDispositions();
}

/// Revalidate the plan without IR mutation or SCEV/DependenceInfo queries.
static bool validatePreparedPlan(const PreparedInterchangePlan &Plan,
                                 const DominatorTree &DT, const LoopInfo &LI) {
  const DistributableOuterEpilogue &Epi = Plan.Epilogue;
  if (!Epi.Outer || !Epi.Inner || !Plan.FissionLegal ||
      !Plan.FissionContextMatrixComplete || !Plan.FissionContextLegal ||
      !Plan.RoutingMatrixComplete || !Plan.InterchangeLegal ||
      !Plan.Profitable || !Plan.Legality)
    return false;

  if (Plan.AbsoluteAncestors.size() < 2 ||
      Plan.AbsoluteAncestors.size() != Epi.Inner->getLoopDepth() ||
      Plan.AbsoluteInnerLoopId != Plan.AbsoluteAncestors.size() - 1 ||
      Plan.AbsoluteOuterLoopId + 1 != Plan.AbsoluteInnerLoopId ||
      Plan.AbsoluteOuterLoopId != Epi.Outer->getLoopDepth() - 1 ||
      Plan.AbsoluteInnerLoopId != Epi.Inner->getLoopDepth() - 1 ||
      Plan.AbsoluteAncestors[Plan.AbsoluteOuterLoopId] != Epi.Outer ||
      Plan.AbsoluteAncestors[Plan.AbsoluteInnerLoopId] != Epi.Inner)
    return false;
  if (!Plan.AbsoluteAncestors.front() ||
      Plan.AbsoluteAncestors.front()->getParentLoop())
    return false;
  for (auto [Index, L] : enumerate(Plan.AbsoluteAncestors))
    if (!L || L->getLoopDepth() != Index + 1 ||
        (Index != 0 && L->getParentLoop() != Plan.AbsoluteAncestors[Index - 1]))
      return false;

  if (Plan.RoutingChain.size() < 2 ||
      Plan.RoutingInnerLoopId != Plan.RoutingChain.size() - 1 ||
      Plan.RoutingOuterLoopId + 1 != Plan.RoutingInnerLoopId ||
      Plan.RoutingChain[Plan.RoutingOuterLoopId] != Epi.Outer ||
      Plan.RoutingChain[Plan.RoutingInnerLoopId] != Epi.Inner)
    return false;
  for (unsigned I = 1; I < Plan.RoutingChain.size(); ++I)
    if (!Plan.RoutingChain[I] ||
        Plan.RoutingChain[I]->getParentLoop() != Plan.RoutingChain[I - 1])
      return false;

  if (Plan.RoutingChain.size() > Plan.AbsoluteAncestors.size())
    return false;
  unsigned SuffixStart =
      Plan.AbsoluteAncestors.size() - Plan.RoutingChain.size();
  for (unsigned I = 0; I < Plan.RoutingChain.size(); ++I)
    if (Plan.RoutingChain[I] != Plan.AbsoluteAncestors[SuffixStart + I])
      return false;

  if (Epi.Outer->getSubLoops().size() != 1 ||
      Epi.Outer->getSubLoops().front() != Epi.Inner ||
      Epi.Inner->getParentLoop() != Epi.Outer || !Epi.Inner->isInnermost())
    return false;

  if (Plan.FissionContextScanRoot != Epi.Outer ||
      Plan.FissionContextLevel != 2 ||
      Plan.FissionContextColumns != DependenceColumns::AbsoluteAncestors ||
      Plan.RoutingScanRoot != Plan.RoutingChain.front() ||
      Plan.RoutingLevel != Plan.RoutingChain.size() ||
      Plan.RoutingColumns != DependenceColumns::SelectedSubnest)
    return false;

  auto HasWidth = [](const CharMatrix &Matrix, unsigned Width) {
    return all_of(Matrix, [Width](const std::vector<char> &Row) {
      return Row.size() == Width + 1 &&
             (Row.back() == '<' || Row.back() == '*');
    });
  };
  CharMatrix ValidatedRoutingMatrix = Plan.RoutingMatrix;
  if (!HasWidth(Plan.FissionContextMatrix, Plan.AbsoluteAncestors.size()) ||
      !HasWidth(Plan.RoutingMatrix, Plan.RoutingChain.size()) ||
      !hasDecisiveOrEqualAncestorPrefix(Plan.FissionContextMatrix,
                                        Plan.AbsoluteOuterLoopId) ||
      !isLegalToInterChangeLoops(ValidatedRoutingMatrix,
                                 Plan.RoutingInnerLoopId,
                                 Plan.RoutingOuterLoopId))
    return false;

  if (Epi.Instructions.empty() || Epi.MemoryInstructions.empty() ||
      Epi.InstructionSet.size() != Epi.Instructions.size())
    return false;
  for (Instruction *I : Epi.Instructions)
    if (!I || !Epi.InstructionSet.contains(I) ||
        !isSupportedEpilogueInstruction(*I))
      return false;

  BasicBlock *OuterLatch = Epi.Outer->getLoopLatch();
  BasicBlock *InnerLatch = Epi.Inner->getLoopLatch();
  BasicBlock *Current = Epi.Inner->getUniqueExitBlock();
  BasicBlock *OuterPreheader = Epi.Outer->getLoopPreheader();
  BasicBlock *Previous = InnerLatch;
  if (!OuterLatch || !InnerLatch || !Current || !OuterPreheader)
    return false;
  for (BasicBlock *BB : Epi.Path) {
    if (!BB || BB != Current || BB == OuterLatch ||
        BB->getUniquePredecessor() != Previous)
      return false;
    auto *Branch = dyn_cast<UncondBrInst>(BB->getTerminator());
    if (!Branch || BB->getUniqueSuccessor() != Branch->getSuccessor(0))
      return false;
    Previous = BB;
    Current = Branch->getSuccessor(0);
  }
  if (Current != OuterLatch || OuterLatch->getUniquePredecessor() != Previous)
    return false;

  SmallVector<Instruction *, 16> ExpectedInstructions;
  SmallVector<Instruction *, 8> ExpectedMemory;
  for (BasicBlock *BB : Epi.Path)
    for (Instruction &I : *BB) {
      if (isa<PHINode>(I) || I.isTerminator())
        continue;
      ExpectedInstructions.push_back(&I);
      if (isa<LoadInst, StoreInst>(I))
        ExpectedMemory.push_back(&I);
    }
  for (Instruction &I : *OuterLatch) {
    if (&I == Epi.OuterControl.NextValue || &I == Epi.OuterControl.LatchCompare)
      break;
    if (isa<PHINode>(I) || I.isTerminator())
      continue;
    ExpectedInstructions.push_back(&I);
    if (isa<LoadInst, StoreInst>(I))
      ExpectedMemory.push_back(&I);
  }
  if (ExpectedInstructions != Epi.Instructions ||
      ExpectedMemory != Epi.MemoryInstructions)
    return false;

  for (Instruction *I : Epi.Instructions)
    for (User *U : I->users()) {
      if (auto *UserI = dyn_cast<Instruction>(U);
          UserI && Epi.InstructionSet.contains(UserI))
        continue;
      if (!U->isDroppable())
        return false;
    }

  SmallVector<Instruction *, 8> ExpectedRematerialized;
  EpilogueOperandClassifier Classifier(
      Epi.Outer, Epi.Inner, Epi.OuterControl.Induction, Epi.InstructionSet,
      ExpectedRematerialized);
  for (Instruction *I : Epi.Instructions)
    for (Value *Operand : I->operands()) {
      if (auto *OperandI = dyn_cast<Instruction>(Operand);
          OperandI && Epi.InstructionSet.contains(OperandI))
        continue;
      if (Classifier.classify(Operand) == EpilogueOperandClass::Invalid)
        return false;
    }
  if (ExpectedRematerialized != Epi.OuterIVDerivedInstructions)
    return false;

  for (Instruction *I : Epi.MemoryInstructions)
    if (!I || !Epi.InstructionSet.contains(I) || !isa<LoadInst, StoreInst>(I))
      return false;
  // Recheck convergence rather than trust the preparation result.
  if (hasConvergentOperationInRetainedNest(Epi))
    return false;

  SmallVector<Instruction *, 16> ExpectedNestMemory;
  for (BasicBlock *BB : Epi.Outer->blocks())
    for (Instruction &I : *BB)
      if (isa<LoadInst, StoreInst>(I) && !Epi.InstructionSet.contains(&I))
        ExpectedNestMemory.push_back(&I);
  if (ExpectedNestMemory != Plan.NestMemoryInstructions)
    return false;
  for (Instruction *I : Plan.NestMemoryInstructions)
    if (!I || Epi.InstructionSet.contains(I) ||
        !Epi.Outer->contains(I->getParent()) || !isa<LoadInst, StoreInst>(I))
      return false;

  const PreparedOuterIVControl &Control = Epi.OuterControl;
  if (!Control.Induction || !Control.InitialValue || !Control.NextValue ||
      !Control.LatchCompare || !Control.LatchBranch ||
      Control.LatchInstructions.size() != 2 ||
      Control.LatchInstructions[0] != Control.NextValue ||
      Control.LatchInstructions[1] != Control.LatchCompare ||
      Control.Induction->getParent() != Epi.Outer->getHeader() ||
      Control.NextValue->getParent() != OuterLatch ||
      Control.LatchCompare->getParent() != OuterLatch ||
      Control.LatchBranch != OuterLatch->getTerminator() ||
      Control.InitialValue !=
          Control.Induction->getIncomingValueForBlock(OuterPreheader) ||
      Control.NextValue !=
          Control.Induction->getIncomingValueForBlock(OuterLatch))
    return false;

  SmallSet<unsigned, 8> RequirementIds;
  for (const PreparedCrossPartitionDependence &Dep : Plan.CrossDependences) {
    if (!Dep.UsedByteOffsetProof)
      continue;
    if (!RequirementIds.insert(Dep.RequirementId).second)
      return false;
    bool HasOuter = false;
    bool HasContainment = false;
    unsigned MatchingRequirements = 0;
    for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements) {
      if (Req.RequirementId != Dep.RequirementId)
        continue;
      ++MatchingRequirements;
      HasOuter |= Req.Kind == BoundRequirementKind::ModularOuterSpan;
      HasContainment |= Req.Kind == BoundRequirementKind::ObjectContainment;
    }
    if (MatchingRequirements != 2 || !HasOuter || !HasContainment)
      return false;
  }

  for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements) {
    if (!Req.DomainLoop || Req.Limit.isZero() || Req.Limit.getBitWidth() != 64)
      return false;
    if (Req.Kind == BoundRequirementKind::ModularOuterSpan) {
      if (Req.DomainLoop != Epi.Outer)
        return false;
    } else if (Req.DomainLoop != Epi.Inner) {
      return false;
    }
    if (none_of(Plan.CrossDependences,
                [&](const PreparedCrossPartitionDependence &Dep) {
                  return Dep.UsedByteOffsetProof &&
                         Dep.RequirementId == Req.RequirementId;
                }))
      return false;
  }

  if (Plan.TripBound.Outcome == PreparedBoundOutcome::RuntimeBound) {
    if (!Plan.TripBound.RuntimeExactTrip || Plan.TripBound.Wmin.isZero() ||
        !isRuntimeVersioningStateCurrent(Plan, DT, LI))
      return false;
    bool SawRuntime = false;
    APInt ExpectedWmin;
    for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements) {
      if (!Req.Runtime)
        continue;
      if (!Req.ExactTrip || Req.ExactTrip != Plan.TripBound.RuntimeExactTrip)
        return false;
      if (SawRuntime)
        ExpectedWmin = APIntOps::umin(ExpectedWmin, Req.Limit);
      else
        ExpectedWmin = Req.Limit;
      SawRuntime = true;
    }
    if (!SawRuntime || ExpectedWmin != Plan.TripBound.Wmin)
      return false;
  } else {
    if (Plan.TripBound.RuntimeExactTrip || Plan.RuntimeVersioning ||
        any_of(Plan.TripBound.Requirements,
               [](const PreparedBoundRequirement &Req) { return Req.Runtime; }))
      return false;
  }

  ArrayRef<Instruction *> LegalDropNoWrap =
      Plan.Legality->getHasNoWrapReductions();
  ArrayRef<Instruction *> LegalDropNoInf = Plan.Legality->getHasNoInfInsts();
  if (!Plan.Legality->isPreparedFor(Epi.Outer, Epi.Inner, Epi.InstructionSet) ||
      Plan.DropNoWrap.size() != LegalDropNoWrap.size() ||
      Plan.DropNoInf.size() != LegalDropNoInf.size() ||
      !std::equal(Plan.DropNoWrap.begin(), Plan.DropNoWrap.end(),
                  LegalDropNoWrap.begin()) ||
      !std::equal(Plan.DropNoInf.begin(), Plan.DropNoInf.end(),
                  LegalDropNoInf.begin()))
    return false;

  return true;
}

static void printPreparedPlanDiagnostic(const PreparedInterchangePlan &Plan) {
  if (!PrintPreparedEpiloguePlans)
    return;
  const DistributableOuterEpilogue &Epi = Plan.Epilogue;
  Loop *Outer = Epi.Outer;
  Loop *Inner = Epi.Inner;
  assert(Plan.Legality && "a printed plan carries interchange legality");
  unsigned ByteOffsetProofs = count_if(
      Plan.CrossDependences, [](const PreparedCrossPartitionDependence &Dep) {
        return Dep.UsedByteOffsetProof;
      });

  bool Static = Plan.TripBound.Outcome == PreparedBoundOutcome::StaticBound;
  APInt MaxLimit(64, 0);
  for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements)
    if (Req.Limit.ugt(MaxLimit))
      MaxLimit = Req.Limit;

  errs() << "loop-interchange: prepared function="
         << Outer->getHeader()->getParent()->getName()
         << " outer=" << Outer->getName() << " inner=" << Inner->getName()
         << " path-blocks=" << Epi.Path.size()
         << " epilogue-insts=" << Epi.Instructions.size()
         << " rematerialized=" << Epi.OuterIVDerivedInstructions.size()
         << " absolute-depth=" << Plan.AbsoluteAncestors.size()
         << " routing-depth=" << Plan.RoutingChain.size()
         << " fission-rows=" << Plan.FissionContextMatrix.size()
         << " routing-rows=" << Plan.RoutingMatrix.size()
         << " cross-deps=" << Plan.CrossDependences.size()
         << " reductions=" << Plan.Legality->getOuterInnerReductions().size()
         << " byte-offset-proofs=" << ByteOffsetProofs
         << " requirements=" << Plan.TripBound.Requirements.size()
         << " bound=" << (Static ? "static-bound" : "runtime-bound");
  if (Static && Plan.TripBound.Requirements.empty())
    errs() << ", no-bound-requirement";
  else if (Static)
    errs() << " W=" << MaxLimit.getZExtValue();
  else
    errs() << " Wmin=" << Plan.TripBound.Wmin.getZExtValue();
  errs() << " requirement-ids=";
  bool First = true;
  for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements) {
    if (!First)
      errs() << ",";
    First = false;
    errs() << Req.RequirementId << ":"
           << (Req.Kind == BoundRequirementKind::ModularOuterSpan
                   ? "modular-outer-span"
                   : "object-containment")
           << "/" << (Req.Runtime ? "runtime" : "static") << "/"
           << (Req.Runtime     ? "runtime"
               : Req.ExactTrip ? "by-context-exact"
                               : "by-constant-max");
  }
  errs() << "\n";
}

static std::optional<PreparedInterchangePlan>
prepareInterchangeWithOuterEpilogue(ArrayRef<Loop *> RoutingChain,
                                    ScalarEvolution *SE, LoopInfo *LI,
                                    DependenceInfo *DI, DominatorTree *DT,
                                    LoopStandardAnalysisResults *AR,
                                    OptimizationRemarkEmitter *ORE) {
  if (RoutingChain.size() < 2)
    return std::nullopt;
  Loop *Outer = RoutingChain[RoutingChain.size() - 2];
  Loop *Inner = RoutingChain.back();

  std::optional<DistributableOuterEpilogue> Discovered =
      discoverDistributableOuterEpilogue(Outer, Inner, SE, DT);
  if (!Discovered)
    return std::nullopt;

  std::optional<SmallVector<Loop *, 8>> AbsoluteAncestors =
      buildCandidateAncestorChain(Outer, Inner);
  if (!AbsoluteAncestors) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "true ancestor-chain context is unavailable");
    return std::nullopt;
  }

  PreparedInterchangePlan Plan;
  Plan.Epilogue = std::move(*Discovered);
  Plan.RoutingChain.assign(RoutingChain.begin(), RoutingChain.end());
  Plan.AbsoluteAncestors = std::move(*AbsoluteAncestors);
  Plan.AbsoluteInnerLoopId =
      static_cast<unsigned>(Plan.AbsoluteAncestors.size() - 1);
  Plan.AbsoluteOuterLoopId = Plan.AbsoluteInnerLoopId - 1;
  Plan.RoutingInnerLoopId = static_cast<unsigned>(Plan.RoutingChain.size() - 1);
  Plan.RoutingOuterLoopId = Plan.RoutingInnerLoopId - 1;

  if (!collectAndCheckPreparationMemory(Plan.Epilogue,
                                        Plan.NestMemoryInstructions)) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "union N/E memory budget or simple-access check "
                           "failed");
    return std::nullopt;
  }

  if (hasConvergentOperationInRetainedNest(Plan.Epilogue)) {
    rejectPreparedEpilogue(
        ORE, Outer, Inner,
        "retained nest contains an unsupported convergent operation");
    return std::nullopt;
  }

  SmallVector<PreparedBoundRequirement, 4> Requirements;
  if (!proveNestToEpilogueDependences(Plan.NestMemoryInstructions,
                                      Plan.Epilogue.MemoryInstructions, Outer,
                                      Inner, Plan.AbsoluteOuterLoopId, DI, SE,
                                      Plan.CrossDependences, Requirements)) {
    rejectPreparedEpilogue(
        ORE, Outer, Inner,
        "cross-partition dependence is not statically proved N-to-E");
    return std::nullopt;
  }
  Plan.FissionLegal = true;

  BasicBlock *Preheader = Outer->getLoopPreheader();
  Instruction *CheckPoint = Preheader ? Preheader->getTerminator() : nullptr;
  if (!CheckPoint) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "selected outer loop has no preheader terminator");
    return std::nullopt;
  }
  const char *BoundRejectReason = "array-bound-requirement-unresolved";
  std::optional<PreparedTripBound> TripBound = resolvePreparedTripBound(
      Requirements, Outer, *SE, &AR->TTI, CheckPoint, BoundRejectReason);
  if (!TripBound) {
    rejectPreparedEpilogue(ORE, Outer, Inner, BoundRejectReason);
    return std::nullopt;
  }
  Plan.TripBound = std::move(*TripBound);

  ExtractedEpilogueView View(&Plan.Epilogue.InstructionSet);

  // Fission uses only the selected Outer's memory and absolute ancestry.
  Plan.FissionContextScanRoot = Outer;
  Plan.FissionContextLevel = 2;
  Plan.FissionContextColumns = DependenceColumns::AbsoluteAncestors;
  if (!populateDependencyMatrix(
          Plan.FissionContextMatrix, Plan.FissionContextLevel,
          Plan.FissionContextScanRoot, DI, SE,
          /*ORE=*/nullptr, Plan.FissionContextColumns, View)) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "fission-context dependence matrix is unavailable");
    return std::nullopt;
  }
  Plan.FissionContextMatrixComplete = true;
  if (!hasDecisiveOrEqualAncestorPrefix(Plan.FissionContextMatrix,
                                        Plan.AbsoluteOuterLoopId)) {
    rejectPreparedEpilogue(
        ORE, Outer, Inner,
        "virtual extracted nest has unknown surrounding dependence context");
    return std::nullopt;
  }
  Plan.FissionContextLegal = true;

  // Ordinary legality and profitability need the collected chain's row domain
  // and suffix projection, not a projection of the fission matrix.
  Plan.RoutingScanRoot = Plan.RoutingChain.front();
  Plan.RoutingLevel = static_cast<unsigned>(Plan.RoutingChain.size());
  Plan.RoutingColumns = DependenceColumns::SelectedSubnest;
  if (!populateDependencyMatrix(Plan.RoutingMatrix, Plan.RoutingLevel,
                                Plan.RoutingScanRoot, DI, SE,
                                /*ORE=*/nullptr, Plan.RoutingColumns, View)) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "routing dependence matrix is unavailable");
    return std::nullopt;
  }
  Plan.RoutingMatrixComplete = true;

  Plan.Legality = std::make_unique<LoopInterchangeLegality>(
      Outer, Inner, SE, /*ORE=*/nullptr, DT, View);
  if (!Plan.Legality->canInterchangeLoops(Plan.RoutingInnerLoopId,
                                          Plan.RoutingOuterLoopId,
                                          Plan.RoutingMatrix)) {
    rejectPreparedEpilogue(
        ORE, Outer, Inner,
        "virtual extracted nest failed existing interchange legality");
    return std::nullopt;
  }
  Plan.InterchangeLegal = true;

  CacheCostManager CCM(Plan.RoutingChain.front(), AR, DI);
  LoopInterchangeProfitability Profitability(Outer, Inner, SE, /*ORE=*/nullptr);
  if (!Profitability.isProfitable(Inner, Outer, Plan.RoutingInnerLoopId,
                                  Plan.RoutingOuterLoopId, Plan.RoutingMatrix,
                                  CCM)) {
    rejectPreparedEpilogue(
        ORE, Outer, Inner,
        "virtual extracted nest failed existing default profitability");
    return std::nullopt;
  }
  Plan.Profitable = true;

  ArrayRef<Instruction *> DropNoWrap = Plan.Legality->getHasNoWrapReductions();
  Plan.DropNoWrap.append(DropNoWrap.begin(), DropNoWrap.end());
  ArrayRef<Instruction *> DropNoInf = Plan.Legality->getHasNoInfInsts();
  Plan.DropNoInf.append(DropNoInf.begin(), DropNoInf.end());

  if (Plan.TripBound.Outcome == PreparedBoundOutcome::RuntimeBound) {
    const char *RuntimeRejectReason = "runtime versioning checks failed";
    if (!prepareRuntimeVersioning(Plan, *SE, *LI, *DT, AR->TTI,
                                  RuntimeRejectReason)) {
      rejectPreparedEpilogue(ORE, Outer, Inner, RuntimeRejectReason);
      return std::nullopt;
    }
  }

  if (!validatePreparedPlan(Plan, *DT, *LI)) {
    rejectPreparedEpilogue(ORE, Outer, Inner,
                           "prepared plan failed pure re-validation");
    return std::nullopt;
  }

  LLVM_DEBUG(dbgs() << "loop-interchange: prepared a complete outer-epilogue "
                       "fission plan in function '"
                    << Outer->getHeader()->getParent()->getName()
                    << "': Outer '" << Outer->getName() << "', Inner '"
                    << Inner->getName() << "', absolute columns "
                    << Plan.AbsoluteOuterLoopId << "/"
                    << Plan.AbsoluteInnerLoopId << ", routing columns "
                    << Plan.RoutingOuterLoopId << "/" << Plan.RoutingInnerLoopId
                    << ", " << Plan.CrossDependences.size()
                    << " cross dependence(s).\n");
  return std::optional<PreparedInterchangePlan>(std::move(Plan));
}

struct LoopInterchange {
  ScalarEvolution *SE = nullptr;
  LoopInfo *LI = nullptr;
  DependenceInfo *DI = nullptr;
  DominatorTree *DT = nullptr;
  LoopStandardAnalysisResults *AR = nullptr;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  LoopInterchange(ScalarEvolution *SE, LoopInfo *LI, DependenceInfo *DI,
                  DominatorTree *DT, LoopStandardAnalysisResults *AR,
                  OptimizationRemarkEmitter *ORE)
      : SE(SE), LI(LI), DI(DI), DT(DT), AR(AR), ORE(ORE) {}

  bool run(Loop *L) {
    if (L->getParentLoop())
      return false;
    SmallVector<Loop *, 8> LoopList;
    populateWorklist(*L, LoopList);
    return processLoopList(LoopList);
  }

  /// Consider below kernel:
  /// for(int i=0; i<n; i++){  // Loop 1
  ///     for(int j=0; j<m; j++){  // Loop 2
  ///         for(int r=0; r<m; r++){  // Loop 3
  ///         // Do something
  ///         }
  ///     }
  ///     for(int k=0; k<p; k++){  // Loop 4
  ///         for(int l=0; l<p; l++){  // Loop 5
  ///             // Do something
  ///         }
  ///     }
  /// }
  /// Then collectPerfectNests() will return:
  /// - [Loop2, Loop3]
  /// - [Loop4, Loop5]
  static SmallVector<SmallVector<Loop *, 8>, 4>
  collectPerfectNests(LoopNest &LN) {
    SmallVector<SmallVector<Loop *, 8>, 4> LoopLists;
    for (Loop *L : LN.getLoops()) {
      if (!L->isInnermost())
        continue;

      SmallVector<Loop *, 8> LoopList;
      Loop *Current = L;
      while (true) {
        LoopList.push_back(Current);
        Loop *Parent = Current->getParentLoop();
        if (!Parent || Parent->getSubLoops().size() != 1)
          break;
        Current = Parent;
      }
      std::reverse(LoopList.begin(), LoopList.end());
      if (LoopList.size() >= 2)
        LoopLists.push_back(std::move(LoopList));
    }
    return LoopLists;
  }

  bool isFissionEligibleChain(ArrayRef<Loop *> Chain) {
    if (Chain.size() < 2 || !hasSupportedLoopDepth(Chain) ||
        !isComputableLoopNest(SE, Chain))
      return false;

    Loop *Outer = Chain[Chain.size() - 2];
    Loop *Inner = Chain.back();
    if (hasRuntimeVersionedLoopMarker(Outer) ||
        hasRuntimeVersionedLoopMarker(Inner)) {
      LLVM_DEBUG(dbgs() << "loop-interchange: outer-epilogue candidate "
                           "discovery skipped chain Outer '"
                        << Outer->getName() << "', Inner '" << Inner->getName()
                        << "': one loop is already versioned.\n");
      return false;
    }
    Loop *Pair[] = {Outer, Inner};
    if (!hasSupportedLoopDepth(ArrayRef<Loop *>(Pair)))
      return false;

    return Outer->getSubLoops().size() == 1 &&
           Outer->getSubLoops().front() == Inner &&
           Inner->getParentLoop() == Outer && Inner->isInnermost();
  }

  /// Ineligible chains consume no slot. Rejections and runtime bounds consume
  /// one attempt each, and the search continues while the budget allows.
  std::optional<PreparedInterchangePlan>
  tryPrepareOuterEpilogueFission(ArrayRef<SmallVector<Loop *, 8>> LoopLists) {
    const unsigned Budget = MaxOuterEpilogueFissionCandidates;
    if (Budget == 0)
      return std::nullopt;

    unsigned Attempts = 0;
    std::optional<PreparedInterchangePlan> FirstRuntime;
    for (const SmallVector<Loop *, 8> &CollectedChain : LoopLists) {
      ArrayRef<Loop *> LoopList = CollectedChain;
      if (!isFissionEligibleChain(LoopList))
        continue;
      if (Attempts == Budget)
        break;
      ++Attempts;

      std::optional<PreparedInterchangePlan> Plan =
          prepareInterchangeWithOuterEpilogue(LoopList, SE, LI, DI, DT, AR,
                                              ORE);
      if (!Plan)
        continue;
      if (Plan->TripBound.Outcome == PreparedBoundOutcome::StaticBound) {
        if (!validatePreparedPlan(*Plan, *DT, *LI)) {
          rejectPreparedEpilogue(
              ORE, Plan->Epilogue.Outer, Plan->Epilogue.Inner,
              "prepared plan failed pure re-validation before selection");
          continue;
        }
        printPreparedPlanDiagnostic(*Plan);
        return Plan;
      }
      if (!FirstRuntime)
        FirstRuntime = std::move(Plan);
    }

    if (FirstRuntime) {
      if (!validatePreparedPlan(*FirstRuntime, *DT, *LI)) {
        rejectPreparedEpilogue(
            ORE, FirstRuntime->Epilogue.Outer, FirstRuntime->Epilogue.Inner,
            "prepared plan failed pure re-validation before selection");
        return std::nullopt;
      }
      printPreparedPlanDiagnostic(*FirstRuntime);
    }
    return FirstRuntime;
  }

  /// Materialize the prepared outer-loop epilogue as its own sibling loop and
  /// perform the interchange decided during preparation. Below, E is that
  /// epilogue slice and N the retained nest. A statically bounded plan reaches
  /// this routine directly. A runtime-bounded plan reaches it only after the
  /// caller has created and marked a whole-loop versioned and fallback pair,
  /// cleared the prepared trip SCEV and the pre-version state, and passed the
  /// fallback loop in \p RuntimeFallback. The selected outer loop may be
  /// top-level or nested, and the epilogue and fallback loops are registered
  /// accordingly. The mutation invalidates the IR pointers recorded in
  /// \p Plan, so the caller must not read the plan afterwards.
  void applyPreparedInterchange(PreparedInterchangePlan &Plan, LPMUpdater &U,
                                Loop *RuntimeFallback = nullptr) {
    bool IsRuntime =
        Plan.TripBound.Outcome == PreparedBoundOutcome::RuntimeBound;
    assert(IsRuntime == (RuntimeFallback != nullptr) &&
           "runtime materialization requires its fallback loop");
    assert(!Plan.TripBound.RuntimeExactTrip &&
           "the runtime trip SCEV must be absent before apply");
    assert(!Plan.RuntimeVersioning &&
           "pre-version structural state must be retired before apply");
    DistributableOuterEpilogue &Epi = Plan.Epilogue;
    Loop *Outer = Epi.Outer;
    Loop *Inner = Epi.Inner;
    assert(Outer && Inner && "prepared plan must carry its candidate pair");
    assert(Plan.FissionLegal && Plan.InterchangeLegal && Plan.Profitable &&
           Plan.Legality && "apply requires a complete prepared plan");

    // The interchange below replaces the selected outer loop in place, so
    // capture its parent first. A null parent makes the epilogue a new
    // top-level loop.
    Loop *OuterParent = Outer->getParentLoop();

    Function &F = *Outer->getHeader()->getParent();
    LLVMContext &Ctx = F.getContext();

    PreparedOuterIVControl &Ctrl = Epi.OuterControl;
    PHINode *OuterIV = Ctrl.Induction;
    BasicBlock *OuterExit = Outer->getExitBlock();
    assert(OuterExit && "prepared outer loop must have a unique exit block");

    // The epilogue latch mirrors the outer latch's branch orientation so both
    // loops run the same trip count.
    CondBrInst *OuterLatchBranch = Ctrl.LatchBranch;
    assert(OuterLatchBranch &&
           "outer latch control was proved conditional during preparation");
    bool ExitIsTrueEdge = OuterLatchBranch->getSuccessor(0) == OuterExit;
    assert((ExitIsTrueEdge || OuterLatchBranch->getSuccessor(1) == OuterExit) &&
           "outer latch must branch to the unique exit");

    // Capture the interchange remark anchor before the transform repurposes the
    // inner loop's blocks.
    DebugLoc InnerLoc = Inner->getStartLoc();
    BasicBlock *InnerHeader = Inner->getHeader();

    DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Eager);

    // (1) Split the outer exit after its LCSSA PHIs. The continuation stays
    // after the epilogue loop, preserving the original N, E, continuation
    // order even when memory aliases.
    BasicBlock::iterator ContinuationStart = OuterExit->getFirstNonPHIIt();
    assert(ContinuationStart != OuterExit->end() &&
           "outer exit must contain a terminator");
    BasicBlock *ExitCont =
        SplitBlock(OuterExit, ContinuationStart, &DTU, LI, /*MSSAU=*/nullptr,
                   OuterExit->getName() + ".cont");

    // (2) Build the epilogue loop: a preheader, a header for the rematerialized
    // outer-IV chain and the cloned E slice, and a latch for the cloned outer
    // IV update and exit compare.
    BasicBlock *EpiPreheader =
        BasicBlock::Create(Ctx, "epilogue.preheader", &F, ExitCont);
    BasicBlock *EpiHeader =
        BasicBlock::Create(Ctx, "epilogue.header", &F, ExitCont);
    BasicBlock *EpiLatch =
        BasicBlock::Create(Ctx, "epilogue.latch", &F, ExitCont);

    // A fresh outer induction variable for the epilogue loop.
    PHINode *EpiIV =
        PHINode::Create(OuterIV->getType(), 2, "epilogue.iv", EpiHeader);
    EpiIV->setDebugLoc(OuterIV->getDebugLoc());

    // Clone every instruction before remapping any of them, so debug records
    // attached to earlier instructions can refer to later definitions.
    ValueToValueMapTy VMap;
    VMap[OuterIV] = EpiIV;
    SmallPtrSet<Value *, 32> ExplicitlyMappedValues;
    ExplicitlyMappedValues.insert(OuterIV);
    if (const DebugLoc &DL = OuterIV->getDebugLoc())
      mapAtomInstance(DL, VMap);

    SmallVector<std::pair<Instruction *, Instruction *>, 32> ClonedInstructions;
    auto CloneWithMap = [&](Instruction *Orig, BasicBlock *Into,
                            bool OriginalSurvives) -> Instruction * {
      Instruction *Clone = Orig->clone();
      if (Orig->hasName())
        Clone->setName(Orig->getName() + ".epil");
      Clone->insertInto(Into, Into->end());
      VMap[Orig] = Clone;
      ExplicitlyMappedValues.insert(Orig);
      ClonedInstructions.emplace_back(Orig, Clone);
      if (OriginalSurvives)
        if (const DebugLoc &DL = Orig->getDebugLoc())
          mapAtomInstance(DL, VMap);
      return Clone;
    };

    // (2a) Clone loop-local definitions before their users, then E in program
    // order. Both lists exclude outer-header arithmetic used only by N.
    for (Instruction *Def : Epi.OuterIVDerivedInstructions)
      CloneWithMap(Def, EpiHeader, /*OriginalSurvives=*/true);
    for (Instruction *I : Epi.Instructions)
      CloneWithMap(I, EpiHeader, /*OriginalSurvives=*/false);
    UncondBrInst *EpiHeaderBranch = UncondBrInst::Create(EpiLatch, EpiHeader);
    if (const DebugLoc &DL = OuterLatchBranch->getDebugLoc())
      EpiHeaderBranch->setDebugLoc(DebugLoc(DL.get()->getWithoutAtom()));

    // (2b) Clone the outer-IV update and exit compare into the latch.
    for (Instruction *I : Ctrl.LatchInstructions)
      CloneWithMap(I, EpiLatch, /*OriginalSurvives=*/true);
    Value *EpiNext = VMap[Ctrl.NextValue];
    Value *EpiCond = VMap[Ctrl.LatchCompare];
    assert(EpiNext && EpiCond &&
           "outer IV update/compare must have been remapped");
    CondBrInst *EpiLatchBranch;
    if (ExitIsTrueEdge)
      EpiLatchBranch =
          CondBrInst::Create(EpiCond, ExitCont, EpiHeader, EpiLatch);
    else
      EpiLatchBranch =
          CondBrInst::Create(EpiCond, EpiHeader, ExitCont, EpiLatch);
    EpiLatchBranch->setDebugLoc(OuterLatchBranch->getDebugLoc());
    EpiLatchBranch->copyMetadata(*OuterLatchBranch, {LLVMContext::MD_prof});
    VMap[OuterLatchBranch] = EpiLatchBranch;
    ExplicitlyMappedValues.insert(OuterLatchBranch);
    ClonedInstructions.emplace_back(OuterLatchBranch, EpiLatchBranch);
    if (const DebugLoc &DL = OuterLatchBranch->getDebugLoc())
      mapAtomInstance(DL, VMap);

    // (2c) Preheader branch and induction closure.
    UncondBrInst *EpiPreheaderBranch =
        UncondBrInst::Create(EpiHeader, EpiPreheader);
    if (const DebugLoc &DL = OuterLatchBranch->getDebugLoc())
      EpiPreheaderBranch->setDebugLoc(DebugLoc(DL.get()->getWithoutAtom()));
    EpiIV->addIncoming(Ctrl.InitialValue, EpiPreheader);
    EpiIV->addIncoming(EpiNext, EpiLatch);

    // Remap every clone now that the atom map is complete, so a source atom
    // shared by N and E is never live in both loops.
    RemapSourceAtom(EpiIV, VMap);
    for (auto [Orig, Clone] : ClonedInstructions)
      RemapInstruction(Clone, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

    // Preparation proved that no E instruction uses the latch control.
    for (Instruction &I :
         make_range(EpiHeader->getFirstNonPHIIt(), EpiHeader->end()))
      for (Value *Operand : I.operands())
        assert((!isa<Instruction>(Operand) ||
                cast<Instruction>(Operand)->getParent() != EpiLatch) &&
               "epilogue body depends on later latch control");

    // (3) Route the nest exit through the epilogue loop. It runs only after
    // the nest exits, so it inherits any zero-trip guard.
    OuterExit->getTerminator()->replaceSuccessorWith(ExitCont, EpiPreheader);

    // (4) Register the epilogue loop's control edges with the dominator tree.
    DTU.applyUpdatesPermissive({
        {DominatorTree::Delete, OuterExit, ExitCont},
        {DominatorTree::Insert, OuterExit, EpiPreheader},
        {DominatorTree::Insert, EpiPreheader, EpiHeader},
        {DominatorTree::Insert, EpiHeader, EpiLatch},
        {DominatorTree::Insert, EpiLatch, EpiHeader},
        {DominatorTree::Insert, EpiLatch, ExitCont},
    });

    // Debug records attach to source positions, not to the values they
    // describe. Map each cloned instruction to its clone and each omitted path
    // terminator to the clone of the first E instruction from a later block,
    // or to the new header branch when there is none.
    DenseMap<Instruction *, Instruction *> DebugAnchorMap;
    for (auto [Orig, Clone] : ClonedInstructions)
      DebugAnchorMap[Orig] = Clone;

    DenseMap<BasicBlock *, unsigned> PathOrder;
    for (auto [Index, BB] : enumerate(Epi.Path))
      PathOrder[BB] = static_cast<unsigned>(Index);

    // A backward suffix scan over each block's first E position computes the
    // terminator mapping in linear time. The outer latch sorts last.
    const unsigned PathBlockCount = static_cast<unsigned>(Epi.Path.size());
    BasicBlock *LatchBlock = Ctrl.NextValue->getParent();
    constexpr size_t NoPosition = std::numeric_limits<size_t>::max();
    SmallVector<Instruction *, 8> FirstOfOrder(PathBlockCount + 1, nullptr);
    SmallVector<size_t, 8> FirstOfOrderPos(PathBlockCount + 1, NoPosition);
    for (auto [Position, I] : enumerate(Epi.Instructions)) {
      BasicBlock *SourceBB = I->getParent();
      unsigned Order;
      if (SourceBB == LatchBlock) {
        Order = PathBlockCount;
      } else {
        auto It = PathOrder.find(SourceBB);
        if (It == PathOrder.end())
          continue;
        Order = It->second;
      }
      if (Position < FirstOfOrderPos[Order]) {
        FirstOfOrderPos[Order] = Position;
        FirstOfOrder[Order] = I;
      }
    }
    // EarliestFrom[Rank] is the earliest slice instruction whose order is at
    // least Rank, so the terminator at path index I reads EarliestFrom[I + 1].
    SmallVector<Instruction *, 8> EarliestFrom(PathBlockCount + 2, nullptr);
    SmallVector<size_t, 8> EarliestFromPos(PathBlockCount + 2, NoPosition);
    for (unsigned Rank = PathBlockCount + 1; Rank-- > 0;) {
      EarliestFrom[Rank] = EarliestFrom[Rank + 1];
      EarliestFromPos[Rank] = EarliestFromPos[Rank + 1];
      if (FirstOfOrderPos[Rank] < EarliestFromPos[Rank]) {
        EarliestFromPos[Rank] = FirstOfOrderPos[Rank];
        EarliestFrom[Rank] = FirstOfOrder[Rank];
      }
    }

    SmallPtrSet<Instruction *, 4> PathTerminators;
    for (auto [Index, BB] : enumerate(Epi.Path)) {
      Instruction *Source = EarliestFrom[Index + 1];
      Instruction *Dest =
          Source ? cast<Instruction>(VMap[Source]) : EpiHeaderBranch;
      Instruction *Terminator = BB->getTerminator();
      DebugAnchorMap[Terminator] = Dest;
      PathTerminators.insert(Terminator);
    }

    // PHI records sit at the header's first insertion point. Give that anchor
    // an E destination so the outer IV's record can follow the epilogue IV;
    // reduction-PHI records fail the mapped-value filter below and stay in N.
    Instruction *OuterHeaderDebugAnchor =
        &*Outer->getHeader()->getFirstInsertionPt();
    if (!DebugAnchorMap.contains(OuterHeaderDebugAnchor))
      DebugAnchorMap[OuterHeaderDebugAnchor] =
          &*EpiHeader->getFirstInsertionPt();

    SmallVector<Instruction *, 32> SourceDebugAnchors;
    SmallPtrSet<Instruction *, 32> SeenDebugAnchors;
    auto AddDebugAnchor = [&](Instruction *I) {
      if (SeenDebugAnchors.insert(I).second)
        SourceDebugAnchors.push_back(I);
    };
    AddDebugAnchor(OuterHeaderDebugAnchor);
    for (Instruction *I : Epi.OuterIVDerivedInstructions)
      AddDebugAnchor(I);
    for (BasicBlock *BB : Epi.Path) {
      for (Instruction &I : *BB)
        if (Epi.InstructionSet.contains(&I))
          AddDebugAnchor(&I);
      AddDebugAnchor(BB->getTerminator());
    }
    for (Instruction &I : *Ctrl.NextValue->getParent())
      if (Epi.InstructionSet.contains(&I) ||
          is_contained(Ctrl.LatchInstructions, &I) || &I == OuterLatchBranch)
        AddDebugAnchor(&I);

    auto IsAvailableInEpilogue = [&](Value *V) {
      if (ExplicitlyMappedValues.contains(V))
        return true;
      if (!Outer->isLoopInvariant(V))
        return false;
      auto *I = dyn_cast<Instruction>(V);
      return !I || DT->dominates(I->getParent(), EpiPreheader);
    };
    auto ClassifyLocation = [&](DbgVariableRecord &DVR, bool &HasMapped,
                                bool &HasTransplanted) {
      bool AllAvailable = true;
      auto ClassifyValue = [&](Value *V) {
        if (!V)
          return;
        HasMapped |= ExplicitlyMappedValues.contains(V);
        if (auto *I = dyn_cast<Instruction>(V))
          HasTransplanted |= Epi.InstructionSet.contains(I);
        AllAvailable &= IsAvailableInEpilogue(V);
      };
      for (Value *V : DVR.location_ops())
        ClassifyValue(V);
      return AllAvailable;
    };

    // Transfer only records whose whole location is available in E: duplicate
    // those describing surviving values, move those describing moved E values.
    // Assignment-tracking records take the salvage path in step 6.
    for (Instruction *Source : SourceDebugAnchors) {
      Instruction *Dest = DebugAnchorMap.lookup(Source);
      assert(Dest && "every debug source position must have an E destination");
      DbgMarker *DestMarker =
          Dest->getParent()->createMarker(Dest->getIterator());
      bool SourcePositionMoves = Epi.InstructionSet.contains(Source) ||
                                 PathTerminators.contains(Source);
      for (DbgRecord &DR : make_early_inc_range(Source->getDbgRecordRange())) {
        bool EraseOriginal = false;
        if (auto *DVR = dyn_cast<DbgVariableRecord>(&DR)) {
          if (DVR->isDbgAssign())
            continue;
          bool HasMapped = false;
          bool HasTransplanted = false;
          bool AllAvailable =
              ClassifyLocation(*DVR, HasMapped, HasTransplanted);
          if (!HasMapped)
            continue;
          if (!AllAvailable) {
            // A location mixing an E value with an outer-variant N value has
            // no meaning in E; drop it rather than leave an erased operand.
            if (HasTransplanted)
              DR.eraseFromParent();
            continue;
          }
          EraseOriginal = HasTransplanted;
        } else {
          // Labels identify a program point rather than an SSA value.
          if (!SourcePositionMoves && !ExplicitlyMappedValues.contains(Source))
            continue;
          EraseOriginal = SourcePositionMoves;
        }

        DbgRecord *Clone = DR.clone();
        DestMarker->insertDbgRecord(Clone, /*InsertAtHead=*/false);
        RemapDbgRecord(F.getParent(), Clone, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
        if (EraseOriginal)
          DR.eraseFromParent();
      }
    }

    // (5) Register the epilogue loop as the sibling following the selected
    // outer loop. Runtime materialization also moves LoopVersioning's appended
    // fallback into the program order N, E, fallback. The two LoopInfo
    // containers store siblings in opposite orders.
    Loop *EpiLoop = LI->AllocateLoop();
    if (!OuterParent) {
      // TopLevelLoops is in reverse program order, so E precedes Outer there.
      // Static storage is E then N. Runtime storage is fallback, E, then N.
      SmallVector<Loop *, 4> ExistingTopLevelLoops =
          LI->takeChildrenIf(/*Parent=*/nullptr, [](Loop *) { return true; });
      bool FoundOuter = false;
      bool FoundFallback = false;
      for (Loop *L : ExistingTopLevelLoops) {
        if (IsRuntime && L == RuntimeFallback) {
          assert(!FoundFallback && "fallback loop appears more than once");
          FoundFallback = true;
          continue;
        }
        if (L == Outer) {
          assert(!FoundOuter && "selected outer appears more than once");
          if (IsRuntime)
            LI->addTopLevelLoop(RuntimeFallback);
          LI->addTopLevelLoop(EpiLoop);
          LI->addTopLevelLoop(Outer);
          FoundOuter = true;
          continue;
        }
        LI->addTopLevelLoop(L);
      }
      assert(FoundOuter &&
             "prepared top-level outer must be present in LoopInfo");
      assert((!IsRuntime || FoundFallback) &&
             "runtime fallback must be present in top-level LoopInfo");
      (void)FoundOuter;
      (void)FoundFallback;
    } else {
      // SubLoops is in program order, so the static order is N then E and the
      // runtime order is N, E, then fallback. The interchange later replaces
      // Outer in place (LI->replaceLoop), which keeps this order.
      SmallVector<Loop *, 4> ExistingChildren =
          LI->takeChildrenIf(OuterParent, [](Loop *) { return true; });
      bool FoundOuter = false;
      bool FoundFallback = false;
      for (Loop *L : ExistingChildren) {
        if (IsRuntime && L == RuntimeFallback) {
          assert(!FoundFallback && "fallback loop appears more than once");
          FoundFallback = true;
          continue;
        }
        if (L == Outer) {
          assert(!FoundOuter && "selected outer appears more than once");
          OuterParent->addChildLoop(Outer);
          OuterParent->addChildLoop(EpiLoop);
          if (IsRuntime)
            OuterParent->addChildLoop(RuntimeFallback);
          FoundOuter = true;
          continue;
        }
        OuterParent->addChildLoop(L);
      }
      assert(FoundOuter &&
             "prepared nested outer must be present in its parent's subloops");
      assert((!IsRuntime || FoundFallback) &&
             "runtime fallback must be present in parent subloops");
      (void)FoundOuter;
      (void)FoundFallback;
      // The epilogue's dedicated preheader lives inside the parent loop but
      // outside E, so it belongs to the parent and its ancestors.
      OuterParent->addBasicBlockToLoop(EpiPreheader, *LI);
    }
    // addBasicBlockToLoop also registers ancestors, so E's parent is set first.
    EpiLoop->addBasicBlockToLoop(EpiHeader, *LI);
    EpiLoop->addBasicBlockToLoop(EpiLatch, *LI);

    // (6) Erase the original E slice in reverse order. Droppable uses (assumes)
    // go first; the slice is otherwise closed.
    for (Instruction *I : Epi.Instructions)
      I->dropDroppableUses();
    for (Instruction *I : reverse(Epi.Instructions)) {
      // Salvage debug uses not moved by the transfer above. Records attached
      // here move to the next surviving instruction on erase.
      salvageDebugInfo(*I);
      assert(I->use_empty() &&
             "epilogue value still used after cloning; slice was not closed");
      I->eraseFromParent();
    }

    // (6a) Merge the now-empty intermediate path blocks, keeping the first
    // (the inner exit with its forwarding PHIs) and the outer latch. A lazy
    // updater batches the dominator-tree updates, which an eager one would
    // apply once per merge.
    DomTreeUpdater LazyDTU(*DT, DomTreeUpdater::UpdateStrategy::Lazy);
    for (unsigned K = 1, PathSize = Epi.Path.size(); K < PathSize; ++K) {
      bool Merged = MergeBlockIntoPredecessor(Epi.Path[K], &LazyDTU, LI);
      (void)Merged;
      assert(Merged && "straight-line epilogue path must collapse");
    }
    LazyDTU.flush();

    // (7) Cached SCEV expressions and dispositions may still name the erased
    // path blocks. Drop them before the LCSSA formation and the interchange,
    // which both query SCEV.
    if (IsRuntime)
      forgetAffectedTopmostLoops(*SE, {Outer, Inner, EpiLoop, RuntimeFallback});
    else {
      SE->forgetLoop(Outer);
      SE->forgetBlockAndLoopDispositions();
    }

    // The epilogue loop is closed (no live-outs); keep it in LCSSA defensively
    // before the interchange updates analyses.
    formLCSSARecursively(*EpiLoop, *DT, LI, SE);

    // (8) Interchange with the stored legality and flag-drop decisions.
    LoopInterchangeTransform LIT(Outer, Inner, SE, LI, DT, *Plan.Legality);
    LIT.transform(Plan.DropNoWrap, Plan.DropNoInf);
    if (IsRuntime) {
      // The interchange is the final CFG mutation. Drop the cached SCEV and
      // dispositions for every affected topmost loop before rebuilding
      // recursive LCSSA from the updated loop tree.
      forgetAffectedTopmostLoops(*SE, {Outer, Inner, EpiLoop, RuntimeFallback});
      if (OuterParent) {
        Loop *AffectedRoot = OuterParent;
        while (AffectedRoot->getParentLoop())
          AffectedRoot = AffectedRoot->getParentLoop();
        formLCSSARecursively(*AffectedRoot, *DT, LI, SE);
      } else {
        formLCSSARecursively(*Inner, *DT, LI, SE);
        formLCSSARecursively(*EpiLoop, *DT, LI, SE);
        formLCSSARecursively(*RuntimeFallback, *DT, LI, SE);
      }
      forgetAffectedTopmostLoops(*SE, {Outer, Inner, EpiLoop, RuntimeFallback});
    } else {
      llvm::formLCSSARecursively(*Outer, *DT, LI, SE);
      // The parent gained a subloop and the interchange moved blocks between
      // loops, so drop the parent's cached SCEV and the dispositions again.
      if (OuterParent)
        SE->forgetLoop(OuterParent);
      SE->forgetBlockAndLoopDispositions();
    }
    LoopsInterchanged++;
    OuterEpiloguesDistributed++;

    // The remark names the largest accepted bound limit W, or says that the
    // plan did not need a bound requirement. A runtime plan instead names Wmin,
    // the smallest of its runtime bound limits, and its guard enforces that
    // limit.
    std::string BoundNote = "bound=static-bound, no-bound-requirement";
    if (IsRuntime) {
      BoundNote = ("bound=runtime-bound, Wmin=" +
                   Twine(Plan.TripBound.Wmin.getZExtValue()))
                      .str();
    } else if (!Plan.TripBound.Requirements.empty()) {
      APInt StaticW(64, 0);
      for (const PreparedBoundRequirement &Req : Plan.TripBound.Requirements)
        if (Req.Limit.ugt(StaticW))
          StaticW = Req.Limit;
      BoundNote =
          ("bound=static-bound, W=" + Twine(StaticW.getZExtValue())).str();
    }

    // (9) Emit the interchange and distribution remarks.
    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Interchanged", InnerLoc,
                                InnerHeader)
             << "Loop interchanged with enclosing loop.";
    });
    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "OuterEpilogueDistributed",
                                EpiLoop->getStartLoc(), EpiHeader)
             << "Distributed a proven outer-loop epilogue into its own loop "
                "before interchanging the reduction nest; "
             << BoundNote << ".";
    });

    // (10) In loop-nest mode the updater accepts only new top-level siblings.
    // A nested epilogue loop is found by the next loop-nest walk after run()
    // reports the change.
    if (!OuterParent) {
      if (IsRuntime)
        U.addSiblingLoops({RuntimeFallback, EpiLoop});
      else
        U.addSiblingLoops({EpiLoop});
    }

    LLVM_DEBUG(
        dbgs() << "loop-interchange: materialized outer-epilogue fission + "
                  "interchange, "
               << (OuterParent ? "nested sibling" : "top-level sibling")
               << ", in function '" << F.getName() << "'.\n");
  }

  /// Guard the selected outer trip, version the whole selected outer loop, and
  /// materialize the prepared transform in the versioned loop only. Every
  /// rejection precedes the guard expansion, the first mutation. After that
  /// point only assertions check the transform.
  bool applyPreparedRuntimeInterchange(PreparedInterchangePlan &Plan,
                                       LPMUpdater &U) {
    assert(Plan.TripBound.Outcome == PreparedBoundOutcome::RuntimeBound &&
           Plan.RuntimeVersioning &&
           "runtime apply requires a completely prepared runtime plan");
    Loop *Outer = Plan.Epilogue.Outer;
    Loop *Inner = Plan.Epilogue.Inner;

    // LoopVersioning composes the LoopAccessInfo predicate union into the
    // condition replaced below, so the union must be empty. LoopAccessAnalysis
    // rejects non-innermost loops before recording predicates or convergent
    // operations, so the requirement holds in the current implementation. The
    // check enforces the LoopVersioning precondition if the analysis gains
    // outer-loop support.
    LoopAccessInfoManager LAIM(*SE, AR->AA, *DT, *LI, &AR->TTI, &AR->TLI,
                               &AR->AC);
    const LoopAccessInfo &LAI = LAIM.getInfo(*Outer);
    const SCEVPredicate &PSEPredicate = LAI.getPSE().getPredicate();
    const auto *PSEUnion = dyn_cast<SCEVUnionPredicate>(&PSEPredicate);
    if (!PSEUnion || !PSEUnion->isAlwaysTrue() ||
        !PSEUnion->getPredicates().empty() || LAI.hasConvergentOp()) {
      rejectPreparedEpilogue(
          ORE, Outer, Inner,
          "loop access predicates are not an empty always-true union");
      return false;
    }

    bool StillValid = validatePreparedPlan(Plan, *DT, *LI);
    assert(StillValid && "a prepared runtime plan must stay valid until apply");
    if (!StillValid) {
      rejectPreparedEpilogue(
          ORE, Outer, Inner,
          "prepared runtime plan failed re-validation before apply");
      return false;
    }

    PreparedRuntimeVersioning &Runtime = *Plan.RuntimeVersioning;
    BasicBlock *CheckBlock = Outer->getLoopPreheader();
    assert(CheckBlock && Runtime.CheckPoint == CheckBlock->getTerminator() &&
           "runtime check point must be the selected outer preheader");
    Loop *OuterParent = Outer->getParentLoop();
    assert(OuterParent == Runtime.ParentLoop &&
           "selected outer parent changed after preparation");
    (void)OuterParent;
    SmallVector<Instruction *, 8> DefsUsedOutside = Runtime.DefsUsedOutside;

    LoopVersioning LVer(LAI, /*Checks=*/{}, Outer, LI, DT, SE);
    Value *Bad = nullptr;
    {
      const SCEV *ExactTrip = Plan.TripBound.RuntimeExactTrip;
      const APInt &Wmin = Plan.TripBound.Wmin;
      unsigned TripWidth = ExactTrip->getType()->getIntegerBitWidth();
      unsigned CompareWidth = std::max(TripWidth, Wmin.getBitWidth());
      Type *CompareType = Type::getIntNTy(SE->getContext(), CompareWidth);
      const SCEV *CompareTrip =
          CompareWidth == TripWidth
              ? ExactTrip
              : SE->getZeroExtendExpr(ExactTrip, CompareType);
      const SCEV *CompareLimit = SE->getConstant(Wmin.zext(CompareWidth));
      const SCEVPredicate *Predicate = SE->getComparePredicate(
          ICmpInst::ICMP_ULE, CompareTrip, CompareLimit);
      SCEVExpander Expander(*SE, "loop-interchange-bound");
      // SCEV expansion may drop poison-generating annotations on a reused
      // instruction that dominates the check point. Both versions share that
      // instruction, and only the versioned body is interchanged afterwards.
      Bad = Expander.expandComparePredicate(
          cast<SCEVComparePredicate>(Predicate), CheckBlock->getTerminator());
    }

    // Clear the prepared trip SCEVs and the recorded structure before cloning.
    // The rest of the plan stays valid. After expansion, the guard needs only
    // the expanded predicate, and Wmin remains available for the remark.
    Plan.TripBound.RuntimeExactTrip = nullptr;
    for (PreparedBoundRequirement &Req : Plan.TripBound.Requirements)
      Req.ExactTrip = nullptr;
    Plan.RuntimeVersioning.reset();

    LVer.versionLoop(DefsUsedOutside);
    assert(LVer.getVersionedLoop() == Outer &&
           "the supplied loop must remain the versioned loop");
    Loop *Fallback = LVer.getNonVersionedLoop();
    assert(Fallback && Fallback != Outer &&
           "loop versioning must create a fallback clone");

    auto *Branch = cast<CondBrInst>(CheckBlock->getTerminator());
    assert(match(Branch->getCondition(), m_Zero()) &&
           "empty checks and predicates must emit a false condition");
    assert(Branch->getSuccessor(0) == Fallback->getLoopPreheader() &&
           Branch->getSuccessor(1) == Outer->getLoopPreheader() &&
           "the failure edge must select the fallback, and the success edge "
           "must select the versioned loop");
    Branch->setCondition(Bad);

    markRuntimeVersionedLoop(Outer);
    markRuntimeVersionedLoop(Fallback);

    // Resolve the exits again from the loop objects. The versioned loop and
    // the clone have distinct dedicated exits feeding one shared join. Apply
    // inserts the versioned loop's epilogue only between the versioned exit
    // and that join.
    BasicBlock *VersionedExit = Outer->getExitBlock();
    BasicBlock *FallbackExit = Fallback->getExitBlock();
    assert(VersionedExit && FallbackExit && VersionedExit != FallbackExit &&
           VersionedExit->getSingleSuccessor() &&
           VersionedExit->getSingleSuccessor() ==
               FallbackExit->getSingleSuccessor() &&
           "versioned loops must feed one shared join through dedicated exits");
    (void)VersionedExit;
    (void)FallbackExit;
    assert(Outer->getParentLoop() == OuterParent &&
           Fallback->getParentLoop() == OuterParent &&
           Inner->getParentLoop() == Outer &&
           "versioning must preserve the selected pair in the versioned loop "
           "and create a sibling clone");
    assert(Outer->isLoopSimplifyForm() && Fallback->isLoopSimplifyForm() &&
           Outer->isRecursivelyLCSSAForm(*DT, *LI) &&
           Fallback->isRecursivelyLCSSAForm(*DT, *LI) &&
           "both versions must remain canonical recursive LCSSA loops");

    // LoopVersioning changed the CFG and live-out PHIs. Invalidate every
    // distinct affected topmost loop before fission's first SCEV-aware query.
    forgetAffectedTopmostLoops(*SE, {Outer, Fallback});

    applyPreparedInterchange(Plan, U, Fallback);
    return true;
  }

  bool run(LoopNest &LN, LPMUpdater &U) {
    SmallVector<SmallVector<Loop *, 8>, 4> LoopLists = collectPerfectNests(LN);
    if (LoopLists.empty()) {
      LLVM_DEBUG(dbgs() << "No Valid candidates for loop interchange.\n");
      return false;
    }

    // Try fission on the ordinary parent/leaf-child candidates. A StaticBound
    // plan extracts a sibling epilogue loop and interchanges the retained nest
    // without a runtime guard or clone. A RuntimeBound plan versions the whole
    // selected outer loop behind one guard on its trip. Only the versioned
    // loop is distributed and interchanged, and the clone is the fallback.
    // Size-optimized functions skip fission but still use ordinary routing.
    if (EnableOuterEpilogueFission && !LN.getParent()->hasOptSize()) {
      if (std::optional<PreparedInterchangePlan> Prepared =
              tryPrepareOuterEpilogueFission(LoopLists)) {
        if (Prepared->TripBound.Outcome == PreparedBoundOutcome::StaticBound) {
          // No IR has changed since preparation. Assert on validation failure.
          // A release build reports the rejection and uses ordinary routing.
          bool StillValid = validatePreparedPlan(*Prepared, *DT, *LI);
          assert(StillValid && "a prepared plan must stay valid until apply");
          if (StillValid) {
            applyPreparedInterchange(*Prepared, U);
            return true;
          }
          rejectPreparedEpilogue(
              ORE, Prepared->Epilogue.Outer, Prepared->Epilogue.Inner,
              "prepared plan failed re-validation immediately before apply");
        } else if (applyPreparedRuntimeInterchange(*Prepared, U)) {
          return true;
        }
        // An unapplied plan is destroyed here, before ordinary routing mutates
        // the chains it points into.
      }
    }

    bool Changed = false;
    for (SmallVector<Loop *, 8> &LoopList : LoopLists) {
      // Ensure minimum depth of the loop nest to do the interchange.
      if (!hasSupportedLoopDepth(LoopList, *ORE))
        continue;
      // Ensure computable loop nest.
      if (!isComputableLoopNest(&AR->SE, LoopList)) {
        LLVM_DEBUG(dbgs() << "Not valid loop candidate for interchange\n");
        continue;
      }
      Changed |= processLoopList(LoopList);
    }
    return Changed;
  }

  unsigned selectLoopForInterchange(ArrayRef<Loop *> LoopList) {
    // TODO: Add a better heuristic to select the loop to be interchanged based
    // on the dependence matrix. Currently we select the innermost loop.
    return LoopList.size() - 1;
  }

  bool processLoopList(SmallVectorImpl<Loop *> &LoopList) {
    bool Changed = false;

    // Ensure proper loop nest depth.
    assert(hasSupportedLoopDepth(LoopList, *ORE) &&
           "Unsupported depth of loop nest.");

    unsigned LoopNestDepth = LoopList.size();

    LLVM_DEBUG({
      dbgs() << "Processing LoopList of size = " << LoopNestDepth
             << " containing the following loops:\n";
      for (auto *L : LoopList) {
        dbgs() << "  - ";
        L->print(dbgs());
      }
    });

    CharMatrix DependencyMatrix;
    Loop *OuterMostLoop = *(LoopList.begin());
    if (!populateDependencyMatrix(DependencyMatrix, LoopNestDepth,
                                  OuterMostLoop, DI, SE, ORE)) {
      LLVM_DEBUG(dbgs() << "Populating dependency matrix failed\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "Dependency matrix before interchange:\n";
               printDepMatrix(DependencyMatrix));

    // Get the Outermost loop exit.
    BasicBlock *LoopNestExit = OuterMostLoop->getExitBlock();
    if (!LoopNestExit) {
      LLVM_DEBUG(dbgs() << "OuterMostLoop '" << OuterMostLoop->getName()
                        << "' needs an unique exit block");
      return false;
    }

    unsigned SelecLoopId = selectLoopForInterchange(LoopList);
    CacheCostManager CCM(LoopList[0], AR, DI);
    // We try to achieve the globally optimal memory access for the loopnest,
    // and do interchange based on a bubble-sort fasion. We start from
    // the innermost loop, move it outwards to the best possible position
    // and repeat this process.
    for (unsigned j = SelecLoopId; j > 0; j--) {
      bool ChangedPerIter = false;
      for (unsigned i = SelecLoopId; i > SelecLoopId - j; i--) {
        bool Interchanged =
            processLoop(LoopList, i, i - 1, DependencyMatrix, CCM);
        ChangedPerIter |= Interchanged;
        Changed |= Interchanged;
      }
      // Early abort if there was no interchange during an entire round of
      // moving loops outwards.
      if (!ChangedPerIter)
        break;
    }
    return Changed;
  }

  bool processLoop(SmallVectorImpl<Loop *> &LoopList, unsigned InnerLoopId,
                   unsigned OuterLoopId,
                   std::vector<std::vector<char>> &DependencyMatrix,
                   CacheCostManager &CCM) {
    Loop *OuterLoop = LoopList[OuterLoopId];
    Loop *InnerLoop = LoopList[InnerLoopId];
    LLVM_DEBUG(dbgs() << "Processing InnerLoopId = " << InnerLoopId
                      << " and OuterLoopId = " << OuterLoopId << "\n");
    LoopInterchangeLegality LIL(OuterLoop, InnerLoop, SE, ORE, DT);
    if (!LIL.canInterchangeLoops(InnerLoopId, OuterLoopId, DependencyMatrix)) {
      LLVM_DEBUG(dbgs() << "Cannot prove legality, not interchanging loops '"
                        << OuterLoop->getName() << "' and '"
                        << InnerLoop->getName() << "'\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "Loops '" << OuterLoop->getName() << "' and '"
                      << InnerLoop->getName()
                      << "' are legal to interchange\n");
    LoopInterchangeProfitability LIP(OuterLoop, InnerLoop, SE, ORE);
    if (!LIP.isProfitable(InnerLoop, OuterLoop, InnerLoopId, OuterLoopId,
                          DependencyMatrix, CCM)) {
      LLVM_DEBUG(dbgs() << "Interchanging loops '" << OuterLoop->getName()
                        << "' and '" << InnerLoop->getName()
                        << "' not profitable.\n");
      return false;
    }

    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Interchanged",
                                InnerLoop->getStartLoc(),
                                InnerLoop->getHeader())
             << "Loop interchanged with enclosing loop.";
    });

    LoopInterchangeTransform LIT(OuterLoop, InnerLoop, SE, LI, DT, LIL);
    LIT.transform(LIL.getHasNoWrapReductions(), LIL.getHasNoInfInsts());
    LLVM_DEBUG(dbgs() << "Loops interchanged: outer loop '"
                      << OuterLoop->getName() << "' and inner loop '"
                      << InnerLoop->getName() << "'\n");
    LoopsInterchanged++;

    llvm::formLCSSARecursively(*OuterLoop, *DT, LI, SE);

    // Loops interchanged, update LoopList accordingly.
    std::swap(LoopList[OuterLoopId], LoopList[InnerLoopId]);
    // Update the DependencyMatrix
    interChangeDependencies(DependencyMatrix, InnerLoopId, OuterLoopId);

    LLVM_DEBUG(dbgs() << "Dependency matrix after interchange:\n";
               printDepMatrix(DependencyMatrix));

    return true;
  }
};

} // end anonymous namespace

bool LoopInterchangeLegality::containsUnsafeInstructions(BasicBlock *BB,
                                                         Instruction *Skip) {
  return any_of(*BB, [this, Skip](const Instruction &I) {
    if (&I == Skip || isVirtuallyExtracted(&I))
      return false;
    return I.mayHaveSideEffects() || I.mayReadFromMemory();
  });
}

static FreezeInst *findFreezeInReNestedBlocks(Loop *OuterLoop,
                                              Loop *InnerLoop) {
  // adjustLoopBranches swaps the preheader bodies after changing their loop
  // roles, so the original outer-preheader body remains outside the new outer
  // loop and retains its execution count.
  BasicBlock *Blocks[] = {
      OuterLoop->getHeader(),
      OuterLoop->getLoopLatch(),
      InnerLoop->getLoopPreheader(),
      InnerLoop->getExitBlock(),
  };
  for (BasicBlock *BB : Blocks)
    if (BB)
      for (Instruction &I : *BB)
        if (auto *Freeze = dyn_cast<FreezeInst>(&I))
          return Freeze;
  return nullptr;
}

static FreezeInst *
findFreezeInInnerLatchCloneSet(Loop *InnerLoop,
                               ArrayRef<PHINode *> InnerLoopInductions) {
  // Mirror the latch-condition and induction-update operand closure cloned by
  // MoveInstructions in LoopInterchangeTransform::transform.
  SmallSetVector<Instruction *, 8> Worklist;
  auto IsDirectInnerLoopBlock = [InnerLoop](BasicBlock *BB) {
    return InnerLoop->contains(BB) &&
           none_of(InnerLoop->getSubLoops(),
                   [BB](Loop *SubLoop) { return SubLoop->contains(BB); });
  };
  auto *LatchBranch =
      dyn_cast<CondBrInst>(InnerLoop->getLoopLatch()->getTerminator());
  if (LatchBranch)
    if (auto *Condition = dyn_cast<Instruction>(LatchBranch->getCondition()))
      Worklist.insert(Condition);

  for (PHINode *Induction : InnerLoopInductions) {
    auto *Incoming = dyn_cast<Instruction>(
        Induction->getIncomingValueForBlock(InnerLoop->getLoopLatch()));
    if (Incoming && !is_contained(InnerLoopInductions, Incoming))
      Worklist.insert(Incoming);
  }

  for (unsigned I = 0; I < Worklist.size(); ++I) {
    Instruction *Current = Worklist[I];
    if (auto *Freeze = dyn_cast<FreezeInst>(Current))
      return Freeze;
    for (Value *Operand : Current->operands()) {
      auto *OperandI = dyn_cast<Instruction>(Operand);
      if (!OperandI || !IsDirectInnerLoopBlock(OperandI->getParent()) ||
          is_contained(InnerLoopInductions, OperandI))
        continue;
      Worklist.insert(OperandI);
    }
  }
  return nullptr;
}

bool LoopInterchangeLegality::tightlyNested(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  LLVM_DEBUG(dbgs() << "Checking if loops '" << OuterLoop->getName()
                    << "' and '" << InnerLoop->getName()
                    << "' are tightly nested\n");

  // In a perfectly nested loop the outer header branches only into the inner
  // loop. If it can also reach the outer latch, it conditionally guards the
  // inner loop (an imperfect nest), so the inner loop runs on only a subset of
  // the outer iterations. Interchanging such a nest would run the inner loop on
  // every outer iteration, including the guarded-off ones, which is illegal
  // when the inner loop relies on the guard to terminate (e.g. an eq/ne exit
  // whose trip count is degenerate once the guard is false). Reject by allowing
  // the outer header to branch only into the inner loop.
  //
  // TODO: This is conservative. A guarded nest is still safe to interchange
  // when the inner loop has a computable trip count that is empty exactly when
  // the guard is false, e.g.:
  //   for (i = 0; i < N; i++)
  //     if (M > 0)                  // loop-invariant guard
  //       for (j = 0; j < M; j++)   // empty when M <= 0
  //         A[j][i] = ...;
  // Interchanging is legal here because the inner loop runs zero times on the
  // guarded-off iterations.
  for (BasicBlock *Succ : successors(OuterLoopHeader))
    if (Succ != InnerLoopPreHeader && Succ != InnerLoop->getHeader())
      return false;

  LLVM_DEBUG(dbgs() << "Checking instructions in Loop header and Loop latch\n");

  // The inner loop reduction pattern requires storing the LCSSA PHI in
  // the OuterLoop Latch. Therefore, when reduction2Memory is enabled, skip
  // that store during checks.
  Instruction *Skip = nullptr;
  assert(InnerReductions.size() <= 1 &&
         "So far we only support at most one reduction.");
  if (InnerReductions.size() == 1)
    Skip = InnerReductions[0].LcssaStore;

  // We do not have any basic block in between now make sure the outer header
  // and outer loop latch doesn't contain any unsafe instructions.
  if (containsUnsafeInstructions(OuterLoopHeader, Skip) ||
      containsUnsafeInstructions(OuterLoopLatch, Skip))
    return false;

  // Also make sure the inner loop preheader does not contain any unsafe
  // instructions. Note that all instructions in the preheader will be moved to
  // the outer loop header when interchanging.
  if (InnerLoopPreHeader != OuterLoopHeader &&
      containsUnsafeInstructions(InnerLoopPreHeader, Skip))
    return false;

  BasicBlock *InnerLoopExit = InnerLoop->getExitBlock();
  // Ensure the inner loop exit block flows to the outer loop latch possibly
  // through empty blocks.
  const BasicBlock &SuccInner = skipVirtuallyEmptyBlockUntil(
      InnerLoopExit, OuterLoopLatch, getExtractedEpilogueView());
  if (&SuccInner != OuterLoopLatch) {
    LLVM_DEBUG(dbgs() << "Inner loop exit block " << *InnerLoopExit
                      << " does not lead to the outer loop latch.\n";);
    return false;
  }
  // The inner loop exit block does flow to the outer loop latch and not some
  // other BBs, now make sure it contains safe instructions, since it will be
  // moved into the (new) inner loop after interchange.
  if (containsUnsafeInstructions(InnerLoopExit, Skip))
    return false;

  LLVM_DEBUG(dbgs() << "Loops are perfectly nested\n");
  // We have a perfect loop nest.
  return true;
}

bool LoopInterchangeLegality::isLoopStructureUnderstood() {
  BasicBlock *InnerLoopPreheader = InnerLoop->getLoopPreheader();
  for (PHINode *InnerInduction : InnerLoopInductions) {
    unsigned Num = InnerInduction->getNumOperands();
    for (unsigned i = 0; i < Num; ++i) {
      Value *Val = InnerInduction->getOperand(i);
      if (isa<Constant>(Val))
        continue;
      Instruction *I = dyn_cast<Instruction>(Val);
      if (!I)
        return false;
      // TODO: Handle triangular loops.
      // e.g. for(int i=0;i<N;i++)
      //        for(int j=i;j<N;j++)
      unsigned IncomBlockIndx = PHINode::getIncomingValueNumForOperand(i);
      if (InnerInduction->getIncomingBlock(IncomBlockIndx) ==
              InnerLoopPreheader &&
          !OuterLoop->isLoopInvariant(I)) {
        return false;
      }
    }
  }

  // TODO: Handle triangular loops of another form.
  // e.g. for(int i=0;i<N;i++)
  //        for(int j=0;j<i;j++)
  // or,
  //      for(int i=0;i<N;i++)
  //        for(int j=0;j*i<N;j++)
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  CondBrInst *InnerLoopLatchBI =
      dyn_cast<CondBrInst>(InnerLoopLatch->getTerminator());
  if (!InnerLoopLatchBI)
    return false;

  CmpInst *InnerLoopCmp = dyn_cast<CmpInst>(InnerLoopLatchBI->getCondition());
  if (!InnerLoopCmp)
    return false;

  Value *Op0 = InnerLoopCmp->getOperand(0);
  Value *Op1 = InnerLoopCmp->getOperand(1);

  // LHS and RHS of the inner loop exit condition, e.g.,
  // in "for(int j=0;j<i;j++)", LHS is j and RHS is i.
  Value *Left = nullptr;
  Value *Right = nullptr;

  // Check if V only involves inner loop induction variable.
  // Return true if V is InnerInduction, or a cast from
  // InnerInduction, or a binary operator that involves
  // InnerInduction and a constant.
  std::function<bool(Value *)> IsPathToInnerIndVar;
  IsPathToInnerIndVar = [this, &IsPathToInnerIndVar](const Value *V) -> bool {
    if (llvm::is_contained(InnerLoopInductions, V))
      return true;
    if (isa<Constant>(V))
      return true;
    const Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;
    if (isa<CastInst>(I))
      return IsPathToInnerIndVar(I->getOperand(0));
    if (isa<BinaryOperator>(I))
      return IsPathToInnerIndVar(I->getOperand(0)) &&
             IsPathToInnerIndVar(I->getOperand(1));
    return false;
  };

  // In case of multiple inner loop indvars, it is okay if LHS and RHS
  // are both inner indvar related variables.
  if (IsPathToInnerIndVar(Op0) && IsPathToInnerIndVar(Op1))
    return true;

  // Otherwise we check if the cmp instruction compares an inner indvar
  // related variable (Left) with a outer loop invariant (Right).
  if (IsPathToInnerIndVar(Op0) && !isa<Constant>(Op0)) {
    Left = Op0;
    Right = Op1;
  } else if (IsPathToInnerIndVar(Op1) && !isa<Constant>(Op1)) {
    Left = Op1;
    Right = Op0;
  }

  if (Left == nullptr)
    return false;

  const SCEV *S = SE->getSCEV(Right);
  if (!SE->isLoopInvariant(S, OuterLoop))
    return false;

  return true;
}

// If SV is a LCSSA PHI node with a single incoming value, return the incoming
// value.
static Value *followLCSSA(Value *SV) {
  PHINode *PHI = dyn_cast<PHINode>(SV);
  if (!PHI)
    return SV;

  if (PHI->getNumIncomingValues() != 1)
    return SV;
  return followLCSSA(PHI->getIncomingValue(0));
}

static bool checkReductionKind(Loop *L, PHINode *PHI,
                               SmallVectorImpl<Instruction *> &HasNoWrapInsts,
                               SmallVectorImpl<Instruction *> &HasNoInfInsts) {
  RecurrenceDescriptor RD;
  if (RecurrenceDescriptor::isReductionPHI(PHI, L, RD)) {
    // Detect floating point reduction only when it can be reordered.
    if (RD.getExactFPMathInst() != nullptr)
      return false;

    // The extra uses of a reduction phi outside of its reduction chain make
    // the order in which the elements are visited observable.
    if (RD.hasUsesOutsideReductionChain())
      return false;

    RecurKind RK = RD.getRecurrenceKind();
    switch (RK) {
    case RecurKind::Or:
    case RecurKind::And:
    case RecurKind::Xor:
    case RecurKind::SMin:
    case RecurKind::SMax:
    case RecurKind::UMin:
    case RecurKind::UMax:
      return true;

    // Interchanging the loops that contain AnyOf reduction is not always legal.
    // Especially, when the result value of the AnyOf is not loop-invariant with
    // respect to the outer loop, interchanging may change the semantics. The
    // following is an example of such case:
    //   int A = {{ 1, 0 }, { 0, 1 }};
    //   int red = 0;
    //   for (int i = 0; i < 2; i++)
    //     for (int j = 0; j < 2; j++)
    //       red = (A[j][i] == 0) ? i + 1 : red;
    //
    // TODO: We may be able to support interchanging loops with AnyOf reduction
    // by checking the operand of the reduction is loop-invariant with respect
    // to the outer loop as well.
    case RecurKind::AnyOf:
      return false;

    // Changing the order of floating-point operations may alter the results. If
    // a certain instruction has the ninf flag, it means that reordering can
    // produce a poison value, which may lead to undefined behavior. To prevent
    // this, we must drop the ninf flags if we decide to apply the
    // transformation.
    case RecurKind::FAdd:
    case RecurKind::FMul:
    case RecurKind::FMin:
    case RecurKind::FMax:
    case RecurKind::FMinimum:
    case RecurKind::FMaximum:
    case RecurKind::FMinimumNum:
    case RecurKind::FMaximumNum:
    case RecurKind::FMulAdd:
      for (Instruction *I : RD.getReductionOpChain(PHI, L))
        if (isa<FPMathOperator>(I) && I->hasNoInfs())
          HasNoInfInsts.push_back(I);
      return true;

    // Change the order of integer addition/multiplication may change the
    // semantics. Consider the following case:
    //
    //  int A[2][2] = {{ INT_MAX, INT_MAX }, { INT_MIN, INT_MIN }};
    //  int sum = 0;
    //  for (int i = 0; i < 2; i++)
    //    for (int j = 0; j < 2; j++)
    //      sum += A[j][i];
    //
    // If the above loops are exchanged, the addition will cause an
    // overflow. To prevent this, we must drop the nuw/nsw flags from the
    // addition/multiplication instructions when we actually exchanges the
    // loops.
    case RecurKind::Add:
    case RecurKind::Mul: {
      unsigned OpCode = RecurrenceDescriptor::getOpcode(RK);
      SmallVector<Instruction *, 4> Ops = RD.getReductionOpChain(PHI, L);

      // Bail out when we fail to collect reduction instructions chain.
      if (Ops.empty())
        return false;

      for (Instruction *I : Ops) {
        assert(I->getOpcode() == OpCode &&
               "Expected the instruction to be the reduction operation");
        (void)OpCode;

        // If the instruction has nuw/nsw flags, we must drop them when the
        // transformation is actually performed.
        if (I->hasNoSignedWrap() || I->hasNoUnsignedWrap())
          HasNoWrapInsts.push_back(I);
      }
      return true;
    }

    default:
      return false;
    }
  } else
    return false;
}

// Check V's users to see if it is involved in a reduction in L.
static PHINode *
findInnerReductionPhi(Loop *L, Value *V,
                      SmallVectorImpl<Instruction *> &HasNoWrapInsts,
                      SmallVectorImpl<Instruction *> &HasNoInfInsts) {
  // Reduction variables cannot be constants.
  if (isa<Constant>(V))
    return nullptr;

  for (Value *User : V->users()) {
    if (PHINode *PHI = dyn_cast<PHINode>(User)) {
      if (PHI->getNumIncomingValues() == 1)
        continue;

      if (checkReductionKind(L, PHI, HasNoWrapInsts, HasNoInfInsts))
        return PHI;
      else
        return nullptr;
    }
  }

  return nullptr;
}

bool LoopInterchangeLegality::isInnerReduction(
    Loop *L, PHINode *Phi, SmallVectorImpl<Instruction *> &HasNoWrapInsts) {

  // Only support reduction2Mem when the loop nest to be interchanged is
  // the innermost two loops.
  if (!L->isInnermost()) {
    LLVM_DEBUG(dbgs() << "Only supported when the loop is the innermost.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerReduction",
                                        L->getStartLoc(), L->getHeader())
               << "Only supported when the loop is the innermost.";
      });
    return false;
  }

  if (Phi->getNumIncomingValues() != 2)
    return false;

  Value *Init = Phi->getIncomingValueForBlock(L->getLoopPreheader());
  Value *Next = Phi->getIncomingValueForBlock(L->getLoopLatch());

  // So far only supports constant initial value.
  if (!isa<Constant>(Init)) {
    LLVM_DEBUG(
        dbgs()
        << "Only supported for the reduction with a constant initial value.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerReduction",
                                        L->getStartLoc(), L->getHeader())
               << "Only supported for the reduction with a constant initial "
                  "value.";
      });
    return false;
  }

  // The reduction result must live in the inner loop.
  if (Instruction *I = dyn_cast<Instruction>(Next)) {
    BasicBlock *BB = I->getParent();
    if (!L->contains(BB))
      return false;
  }

  // The reduction should have only one user.
  if (!Phi->hasOneUser())
    return false;

  // Check the reduction kind.
  if (!checkReductionKind(L, Phi, HasNoWrapInsts, HasNoInfInsts))
    return false;

  // Find lcssa_phi in OuterLoop's Latch
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock)
    return false;

  PHINode *Lcssa = NULL;
  for (auto *U : Next->users()) {
    if (auto *P = dyn_cast<PHINode>(U)) {
      if (P == Phi)
        continue;

      if (Lcssa == NULL && P->getParent() == ExitBlock &&
          P->getIncomingValueForBlock(L->getLoopLatch()) == Next)
        Lcssa = P;
      else
        return false;
    } else
      return false;
  }
  if (!Lcssa)
    return false;

  if (!Lcssa->hasOneUser()) {
    LLVM_DEBUG(dbgs() << "Only supported when the reduction is used once in "
                         "the outer loop.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerReduction",
                                        L->getStartLoc(), L->getHeader())
               << "Only supported when the reduction is used once in the outer "
                  "loop.";
      });
    return false;
  }

  StoreInst *LcssaStore =
      dyn_cast<StoreInst>(Lcssa->getUniqueUndroppableUser());
  if (!LcssaStore || LcssaStore->getParent() != ExitBlock)
    return false;

  Value *MemRef = LcssaStore->getOperand(1);
  Type *ElemTy = LcssaStore->getOperand(0)->getType();

  // LcssaStore stores the reduction result in BB.
  // When the reduction is initialized from a constant value, we need to load
  // from the memory object into the target basic block of the inner loop. This
  // means the memory reference was used prematurely. So we must ensure that the
  // memory reference does not dominate the target basic block.
  // TODO: Move the memory reference definition into the loop header.
  if (!DT->dominates(dyn_cast<Instruction>(MemRef), L->getHeader())) {
    LLVM_DEBUG(dbgs() << "Only supported when memory reference dominate "
                         "the inner loop.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerReduction",
                                        L->getStartLoc(), L->getHeader())
               << "Only supported when memory reference dominate the inner "
                  "loop.";
      });
    return false;
  }

  // Found a reduction in the inner loop.
  InnerReduction SR;
  SR.Reduction = Phi;
  SR.Init = Init;
  SR.Next = Next;
  SR.LcssaPhi = Lcssa;
  SR.LcssaStore = LcssaStore;
  SR.MemRef = MemRef;
  SR.ElemTy = ElemTy;

  InnerReductions.push_back(SR);
  return true;
}

bool LoopInterchangeLegality::checkInductionsAndReductions(Loop *OuterLoop) {
  auto ChildLoop = [](Loop *L) {
    assert(L->getSubLoops().size() <= 1 &&
           "Expect at most one child loop for now.");
    return L->getSubLoops().empty() ? nullptr : L->getSubLoops().front();
  };

  Loop *InnerLoop = ChildLoop(OuterLoop);
  for (Loop *CurLoop = OuterLoop; CurLoop; CurLoop = ChildLoop(CurLoop)) {
    for (PHINode &PHI : CurLoop->getHeader()->phis()) {
      InductionDescriptor ID;
      if (InductionDescriptor::isInductionPHI(&PHI, CurLoop, SE, ID)) {
        if (CurLoop == InnerLoop) {
          const SCEV *Step = ID.getStep();
          if (!SE->isLoopInvariant(Step, OuterLoop))
            return false;
          InnerLoopInductions.push_back(&PHI);
        }
        continue;
      }

      if (CurLoop == OuterLoop) {
        // PHIs in inner loops need to be part of a reduction in the outer loop,
        if (PHI.getNumIncomingValues() != 2) {
          LLVM_DEBUG(dbgs() << "Only PHI nodes in the outer loop header with 2 "
                               "incoming values are supported.\n");
          return false;
        }
        // Check if we have a PHI node in the outer loop that has a reduction
        // result from the inner loop as an incoming value.
        Value *V = followLCSSA(
            PHI.getIncomingValueForBlock(OuterLoop->getLoopLatch()));
        PHINode *InnerRedPhi = findInnerReductionPhi(
            InnerLoop, V, HasNoWrapReductions, HasNoInfInsts);

        // Reject if PHI has users other than InnerRedPhi. The typical case is
        // as follows:
        //
        //   o.header:
        //     %red.o = phi [ 0, ... ], [ %red.next, %o.latch ]
        //     br label %i.header
        //
        //   i.header:
        //     %red.i = phi [ %red.o, %o.header ], [ %red.next, %i.latch ]
        //     br label %i.body
        //
        //   i.body:
        //     store %red.o to %mem
        //     ...
        //
        if (!InnerRedPhi ||
            !llvm::is_contained(InnerRedPhi->incoming_values(), &PHI) ||
            !all_of(PHI.users(),
                    [InnerRedPhi](User *U) { return U == InnerRedPhi; })) {
          LLVM_DEBUG(
              dbgs()
              << "Failed to recognize PHI as an induction or reduction.\n");
          if (ORE)
            ORE->emit([&]() {
              return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIOuter",
                                              OuterLoop->getStartLoc(),
                                              OuterLoop->getHeader())
                     << "Only outer loops with induction or reduction PHI "
                        "nodes "
                        "can be interchanged currently.";
            });
          return false;
        }

        OuterInnerReductions.insert(&PHI);
        OuterInnerReductions.insert(InnerRedPhi);
      } else {
        if (OuterInnerReductions.count(&PHI)) {
          LLVM_DEBUG(dbgs() << "Found a reduction across the outer loop.\n");
        } else if (EnableReduction2Memory &&
                   isInnerReduction(CurLoop, &PHI, HasNoWrapReductions)) {
          LLVM_DEBUG(dbgs() << "Found a reduction in the inner loop: \n"
                            << PHI << '\n');
        } else {
          if (ORE)
            ORE->emit([&]() {
              return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedPHIInner",
                                              CurLoop->getStartLoc(),
                                              CurLoop->getHeader())
                     << "Only inner loops with induction or reduction PHI "
                        "nodes "
                        "can be interchanged currently.";
            });
          return false;
        }
      }
    }

    // For now we only support at most one reduction.
    if (InnerReductions.size() > 1) {
      LLVM_DEBUG(dbgs() << "Only supports at most one reduction.\n");
      if (ORE)
        ORE->emit([&]() {
          return OptimizationRemarkMissed(
                     DEBUG_TYPE, "UnsupportedInnerReduction",
                     CurLoop->getStartLoc(), CurLoop->getHeader())
                 << "Only supports at most one reduction.";
        });
      return false;
    }
  }

  return !InnerLoopInductions.empty();
}

// This function indicates the current limitations in the transform as a result
// of which we do not proceed.
bool LoopInterchangeLegality::currentLimitations() {
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();

  // transform currently expects the loop latches to also be the exiting
  // blocks.
  if (InnerLoop->getExitingBlock() != InnerLoopLatch ||
      OuterLoop->getExitingBlock() != OuterLoop->getLoopLatch() ||
      !isa<CondBrInst>(InnerLoopLatch->getTerminator()) ||
      !isa<CondBrInst>(OuterLoop->getLoopLatch()->getTerminator())) {
    LLVM_DEBUG(
        dbgs() << "Loops where the latch is not the exiting block are not"
               << " supported currently.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "ExitingNotLatch",
                                        OuterLoop->getStartLoc(),
                                        OuterLoop->getHeader())
               << "Loops where the latch is not the exiting block cannot be"
                  " interchange currently.";
      });
    return true;
  }

  // TODO: Triangular loops are not handled for now.
  if (!isLoopStructureUnderstood()) {
    LLVM_DEBUG(dbgs() << "Loop structure not understood by pass\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedStructureInner",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Inner loop structure not understood currently.";
      });
    return true;
  }

  // Currently, we do not support loops that have a predecessor entering the
  // loop via an indirectbr.
  for (Loop *L : {OuterLoop, InnerLoop}) {
    BasicBlock *Header = L->getHeader();
    for (BasicBlock *Pred : predecessors(Header)) {
      if (L->contains(Pred))
        continue;
      if (isa<IndirectBrInst>(Pred->getTerminator())) {
        LLVM_DEBUG(
            dbgs() << "Indirect branch found in the loop predecessor.\n");
        if (ORE)
          ORE->emit([&]() {
            return OptimizationRemarkMissed(DEBUG_TYPE,
                                            "IndirectBranchPreheader",
                                            L->getStartLoc(), L->getHeader())
                   << "Indirect branch found in the loop predecessor.";
          });
        return true;
      }
    }
  }

  // Currently, we do not support loops where the inner loop header has
  // duplicate successors.
  SmallPtrSet<BasicBlock *, 2> InnerLoopHeaderSuccs;
  for (BasicBlock *Succ : successors(InnerLoop->getHeader()))
    if (!InnerLoopHeaderSuccs.insert(Succ).second)
      return true;

  return false;
}

/// We currently only support LCSSA PHI nodes in the inner loop exit if their
/// users are either of the following:
///
/// - Reduction PHIs
/// - PHIs outside the outer loop
/// - PHIs belonging to the latch of the outer loop
///
/// These conditions mean that we are only interested in the final value after
/// the inner loop.
static bool
areInnerLoopExitPHIsSupported(Loop *OuterL, Loop *InnerL,
                              SmallPtrSetImpl<PHINode *> &Reductions,
                              PHINode *LcssaReduction) {
  BasicBlock *InnerExit = InnerL->getUniqueExitBlock();
  for (PHINode &PHI : InnerExit->phis()) {
    // The reduction LCSSA PHI will have only one incoming block, which comes
    // from the loop latch.
    if (PHI.getNumIncomingValues() > 1)
      return false;
    // The reduction LCSSA PHI's store user is rewritten by reduction2Memory();
    // skip its user-check but keep validating the remaining LCSSA PHIs.
    if (&PHI == LcssaReduction)
      continue;
    if (any_of(PHI.users(), [&Reductions, OuterL](User *U) {
          PHINode *PN = dyn_cast<PHINode>(U);
          if (!PN)
            return true;
          if (Reductions.count(PN))
            return false;
          BasicBlock *PB = PN->getParent();
          if (!OuterL->contains(PB))
            return false;
          return PB != OuterL->getLoopLatch();
        }))
      return false;
  }
  return true;
}

// We currently support LCSSA PHI nodes in the outer loop exit, if their
// incoming values do not come from the outer loop latch or if the
// outer loop latch has a single predecessor. In that case, the value will
// be available if both the inner and outer loop conditions are true, which
// will still be true after interchanging. If we have multiple predecessor,
// that may not be the case, e.g. because the outer loop latch may be executed
// if the inner loop is not executed.
static bool areOuterLoopExitPHIsSupported(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *LoopNestExit = OuterLoop->getUniqueExitBlock();
  for (PHINode &PHI : LoopNestExit->phis()) {
    for (Value *Incoming : PHI.incoming_values()) {
      Instruction *IncomingI = dyn_cast<Instruction>(Incoming);
      if (!IncomingI || IncomingI->getParent() != OuterLoop->getLoopLatch())
        continue;

      // The incoming value is defined in the outer loop latch. Currently we
      // only support that in case the outer loop latch has a single predecessor.
      // This guarantees that the outer loop latch is executed if and only if
      // the inner loop is executed (because tightlyNested() guarantees that the
      // outer loop header only branches to the inner loop or the outer loop
      // latch).
      // FIXME: We could weaken this logic and allow multiple predecessors,
      //        if the values are produced outside the loop latch. We would need
      //        additional logic to update the PHI nodes in the exit block as
      //        well.
      if (OuterLoop->getLoopLatch()->getUniquePredecessor() == nullptr)
        return false;
    }
  }
  return true;
}

/// The transform partially clones the inner loop's latch block, but PHI nodes
/// cannot be cloned this way. This function follows the instruction trees that
/// would be cloned and checks whether any PHI node other than the induction
/// PHIs feeds them. If such a PHI is found, the interchange is rejected.
///
/// TODO: This check strongly depends on the current implementation of the
/// transform. Ideally, the transform should be able to handle such PHI nodes in
/// the inner loop latch.
static bool areInnerLoopLatchPHIsSupported(Loop *InnerLoop,
                                           ArrayRef<PHINode *> InductionPHIs) {
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();

  // Seed the worklist with the roots of the use-def chains the transform
  // clones: the latch's exit condition and the incoming values of the induction
  // PHIs from the latch.
  SmallSetVector<Instruction *, 8> Worklist;
  if (auto *LatchBI = dyn_cast<CondBrInst>(InnerLoopLatch->getTerminator()))
    if (auto *CondI = dyn_cast<Instruction>(LatchBI->getCondition()))
      Worklist.insert(CondI);
  for (PHINode *InductionPHI : InductionPHIs) {
    if (auto *IncomingI = dyn_cast<Instruction>(
            InductionPHI->getIncomingValueForBlock(InnerLoopLatch)))
      if (!is_contained(InductionPHIs, IncomingI))
        Worklist.insert(IncomingI);
  }

  // Bail if a PHI node other than the induction PHIs feeds the cloned
  // instructions, walking the operand trees within the inner loop.
  SmallPtrSet<Instruction *, 4> InductionPHISet(InductionPHIs.begin(),
                                                InductionPHIs.end());
  for (unsigned I = 0; I < Worklist.size(); ++I) {
    Instruction *Cur = Worklist[I];
    if (isa<PHINode>(Cur) && !InductionPHISet.contains(Cur))
      return false;
    for (Value *Op : Cur->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        if (InnerLoop->contains(OpI))
          Worklist.insert(OpI);
  }
  return true;
}

bool LoopInterchangeLegality::canInterchangeLoops(unsigned InnerLoopId,
                                                  unsigned OuterLoopId,
                                                  CharMatrix &DepMatrix) {
  if (!isLegalToInterChangeLoops(DepMatrix, InnerLoopId, OuterLoopId)) {
    LLVM_DEBUG(dbgs() << "Failed interchange InnerLoopId = " << InnerLoopId
                      << " and OuterLoopId = " << OuterLoopId
                      << " due to dependence\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "Dependence",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Cannot interchange loops due to dependences.";
      });
    return false;
  }
  // Check if outer and inner loop contain legal instructions only.
  for (auto *BB : OuterLoop->blocks())
    for (Instruction &I : *BB) {
      if (isVirtuallyExtracted(&I))
        continue;
      // Loads and stores are checked separately, so we can skip them here.
      if (isa<LoadInst, StoreInst, PseudoProbeInst>(&I))
        continue;

      // We cannot ignore potential memory reads, e.g., loads inside the called
      // function.
      if (!I.mayHaveSideEffects() && !I.mayReadFromMemory())
        continue;

      LLVM_DEBUG(
          dbgs()
          << "Loops contain instructions that cannot be safely interchanged\n");
      if (ORE)
        ORE->emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "UnsafeInst",
                                          I.getDebugLoc(), I.getParent())
                 << "Cannot interchange loops due to instruction that is "
                    "potentially unsafe to interchange.";
        });

      return false;
    }

  if (!checkInductionsAndReductions(OuterLoop)) {
    LLVM_DEBUG(dbgs() << "Failed to find inner loop inductions or found "
                         "unsupported reductions.\n");
    return false;
  }

  if (!areInnerLoopLatchPHIsSupported(InnerLoop, InnerLoopInductions)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in inner loop latch.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedInnerLatchPHI",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Cannot interchange loops because unsupported PHI nodes "
                  "found "
                  "in inner loop latch.";
      });
    return false;
  }

  FreezeInst *Freeze = findFreezeInReNestedBlocks(OuterLoop, InnerLoop);
  if (!Freeze)
    Freeze = findFreezeInInnerLatchCloneSet(InnerLoop, InnerLoopInductions);
  if (Freeze) {
    LLVM_DEBUG(dbgs() << "Interchange would re-nest or duplicate freeze\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsafeInst",
                                        Freeze->getDebugLoc(),
                                        Freeze->getParent())
               << "Cannot interchange loops because re-nesting or duplicating "
                  "freeze may change its sampling behavior.";
      });
    return false;
  }

  // TODO: The loops could not be interchanged due to current limitations in the
  // transform module.
  if (currentLimitations()) {
    LLVM_DEBUG(dbgs() << "Not legal because of current transform limitation\n");
    return false;
  }

  // Check if the loops are tightly nested.
  if (!tightlyNested(OuterLoop, InnerLoop)) {
    LLVM_DEBUG(dbgs() << "Loops not tightly nested\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "NotTightlyNested",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Cannot interchange loops because they are not tightly "
                  "nested.";
      });
    return false;
  }

  // The LCSSA PHI for the reduction has passed checks before; its user
  // is a store instruction.
  PHINode *LcssaReduction = nullptr;
  assert(InnerReductions.size() <= 1 &&
         "So far we only support at most one reduction.");
  if (InnerReductions.size() == 1)
    LcssaReduction = InnerReductions[0].LcssaPhi;

  if (!areInnerLoopExitPHIsSupported(OuterLoop, InnerLoop, OuterInnerReductions,
                                     LcssaReduction)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in inner loop exit.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedExitPHI",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Found unsupported PHI node in loop exit.";
      });
    return false;
  }

  if (!areOuterLoopExitPHIsSupported(OuterLoop, InnerLoop)) {
    LLVM_DEBUG(dbgs() << "Found unsupported PHI nodes in outer loop exit.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedExitPHI",
                                        OuterLoop->getStartLoc(),
                                        OuterLoop->getHeader())
               << "Found unsupported PHI node in loop exit.";
      });
    return false;
  }

  if (any_of(OuterLoop->getLoopLatch()->phis(),
             [](PHINode &PHI) { return PHI.getNumIncomingValues() != 1; })) {
    LLVM_DEBUG(dbgs() << "Only outer loop latch PHI nodes with one incoming "
                         "value are supported.\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedLatchPHI",
                                        OuterLoop->getStartLoc(),
                                        OuterLoop->getHeader())
               << "Only outer loop latch PHI nodes with one incoming value are "
                  "supported.";
      });
    return false;
  }

  // Regarding def-use chains that begin at an LCSSA PHI in the inner loop exit
  // and end at any instruction in the outer loop latch, we currently support
  // only the case where the chain contains only PHI nodes. Since we already
  // call `tightlyNested()`, we know that if there is a def-use chain that we
  // don't support (i.e., a chain that contains a non-PHI user), then the
  // non-PHI user must be in the outer loop latch.
  if (InnerLoop->getExitBlock() != OuterLoop->getLoopLatch())
    for (PHINode &PHI : OuterLoop->getLoopLatch()->phis())
      if (any_of(PHI.users(), [](const User *U) { return !isa<PHINode>(U); })) {
        LLVM_DEBUG(dbgs() << "Outer loop latch PHI has a non-PHI user.\n");
        if (ORE)
          ORE->emit([&]() {
            return OptimizationRemarkMissed(DEBUG_TYPE, "UnsupportedLatchPHI",
                                            OuterLoop->getStartLoc(),
                                            OuterLoop->getHeader())
                   << "Cannot interchange loops because an outer loop latch "
                      "PHI "
                      "node has a non-PHI user.";
          });
        return false;
      }

  return true;
}

void CacheCostManager::computeIfUnitinialized() {
  if (CC.has_value())
    return;

  LLVM_DEBUG(dbgs() << "Compute CacheCost.\n");
  CC = CacheCost::getCacheCost(*OutermostLoop, *AR, *DI);
  // Obtain the loop vector returned from loop cache analysis beforehand,
  // and put each <Loop, index> pair into a map for constant time query
  // later. Indices in loop vector reprsent the optimal order of the
  // corresponding loop, e.g., given a loopnest with depth N, index 0
  // indicates the loop should be placed as the outermost loop and index N
  // indicates the loop should be placed as the innermost loop.
  //
  // For the old pass manager CacheCost would be null.
  if (*CC != nullptr)
    for (const auto &[Idx, Cost] : enumerate((*CC)->getLoopCosts()))
      CostMap[Cost.first] = Idx;
}

CacheCost *CacheCostManager::getCacheCost() {
  computeIfUnitinialized();
  return CC->get();
}

const DenseMap<const Loop *, unsigned> &CacheCostManager::getCostMap() {
  computeIfUnitinialized();
  return CostMap;
}

/// If \S contains an affine addrec for \p L, return the step recurrence of it.
/// If \S is loop invariant with respect to \p L, return nullptr. Otherwise,
/// return std::nullopt, which indicates we cannot determine the coefficient of
/// the addrec for \p L in \S.
/// TODO: Handle more complex cases. Maybe using SCEVTraversal is a good way to
/// do that.
static std::optional<const SCEV *>
getAddRecCoefficient(ScalarEvolution &SE, const SCEV *S, const Loop *L) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR) {
    if (SE.isLoopInvariant(S, L))
      return nullptr;
    return std::nullopt;
  }

  if (!AR->isAffine()) {
    LLVM_DEBUG(dbgs() << "Unexpected non-affine addrec\n");
    return std::nullopt;
  }

  std::optional<const SCEV *> Coeff =
      getAddRecCoefficient(SE, AR->getStart(), L);
  if (!Coeff.has_value())
    return std::nullopt;

  if (AR->getLoop() == L) {
    assert(!*Coeff && "Found more than one addrec for the same loop");
    Coeff = AR->getStepRecurrence(SE);
  }
  return Coeff;
}

int LoopInterchangeProfitability::getInstrOrderCost() {
  SmallPtrSet<const SCEV *, 4> GoodBasePtrs, BadBasePtrs;
  for (BasicBlock *BB : InnerLoop->blocks()) {
    for (Instruction &Ins : *BB) {
      if (!isa<LoadInst, StoreInst>(&Ins))
        continue;
      const SCEV *Access = SE->getSCEV(getLoadStorePointerOperand(&Ins));
      const SCEV *BasePtr = SE->getPointerBase(Access);
      std::optional<const SCEV *> OuterCoeff =
          getAddRecCoefficient(*SE, Access, OuterLoop);
      std::optional<const SCEV *> InnerCoeff =
          getAddRecCoefficient(*SE, Access, InnerLoop);

      if (!OuterCoeff.has_value() || !*OuterCoeff || !InnerCoeff.has_value() ||
          !*InnerCoeff)
        continue;

      // This heuristic assumes that a smaller step recurrence implies that the
      // induction variable corresponding to the loop is used in the inner
      // dimension of the array. Placing such a loop in the inner position would
      // be beneficial in terms of locality. If the array access is of the form
      // like `A[3*i + 2*j]`, this heuristic may lead to an unprofitable
      // interchange, but we expect such cases to be rare.
      const SCEV *OuterStep = SE->getAbsExpr(*OuterCoeff, /*IsNSW=*/false);
      const SCEV *InnerStep = SE->getAbsExpr(*InnerCoeff, /*IsNSW=*/false);
      // If we find the inner induction after an outer induction e.g.
      //
      //   for(int i=0;i<N;i++)
      //     for(int j=0;j<N;j++)
      //       A[i][j] = A[i-1][j-1]+k;
      //
      //
      // then it is a good order. If we find the outer induction after an inner
      // induction e.g.
      //
      //   for(int i=0;i<N;i++)
      //     for(int j=0;j<N;j++)
      //       A[j][i] = A[j-1][i-1]+k;
      //
      // then it is a bad order.
      //
      // To avoid counting the same base pointers multiple times, we deduplicate
      // them by using a set of base pointers.
      if (SE->isKnownPredicate(ICmpInst::ICMP_SLT, InnerStep, OuterStep))
        GoodBasePtrs.insert(BasePtr);
      else if (SE->isKnownPredicate(ICmpInst::ICMP_SLT, OuterStep, InnerStep))
        BadBasePtrs.insert(BasePtr);
    }
  }

  int GoodOrder = GoodBasePtrs.size();
  int BadOrder = BadBasePtrs.size();
  return GoodOrder - BadOrder;
}

std::optional<bool>
LoopInterchangeProfitability::isProfitablePerLoopCacheAnalysis(
    const DenseMap<const Loop *, unsigned> &CostMap, CacheCost *CC) {
  // This is the new cost model returned from loop cache analysis.
  // A smaller index means the loop should be placed an outer loop, and vice
  // versa.
  auto InnerLoopIt = CostMap.find(InnerLoop);
  if (InnerLoopIt == CostMap.end())
    return std::nullopt;
  auto OuterLoopIt = CostMap.find(OuterLoop);
  if (OuterLoopIt == CostMap.end())
    return std::nullopt;

  if (CC->getLoopCost(*OuterLoop) == CC->getLoopCost(*InnerLoop))
    return std::nullopt;
  unsigned InnerIndex = InnerLoopIt->second;
  unsigned OuterIndex = OuterLoopIt->second;
  LLVM_DEBUG(dbgs() << "InnerIndex = " << InnerIndex
                    << ", OuterIndex = " << OuterIndex << "\n");
  assert(InnerIndex != OuterIndex && "CostMap should assign unique "
                                     "numbers to each loop");
  return std::optional<bool>(InnerIndex < OuterIndex);
}

std::optional<bool>
LoopInterchangeProfitability::isProfitablePerInstrOrderCost() {
  // Legacy cost model: this is rough cost estimation algorithm. It counts the
  // good and bad order of induction variables in the instruction and allows
  // reordering if number of bad orders is more than good.
  int Cost = getInstrOrderCost();
  LLVM_DEBUG(dbgs() << "Cost = " << Cost << "\n");
  if (Cost < 0 && Cost < LoopInterchangeCostThreshold)
    return std::optional<bool>(true);

  return std::nullopt;
}

/// Return true if we can vectorize the loop specified by \p LoopId.
static bool canVectorize(const CharMatrix &DepMatrix, unsigned LoopId) {
  for (const auto &Dep : DepMatrix) {
    char Dir = Dep[LoopId];
    char DepType = Dep.back();
    assert((DepType == '<' || DepType == '*') &&
           "Unexpected element in dependency vector");

    // There are no loop-carried dependencies.
    if (Dir == '=' || Dir == 'I')
      continue;

    // DepType being '<' means that this direction vector represents a forward
    // dependency. In principle, a loop with '<' direction can be vectorized in
    // this case.
    if (Dir == '<' && DepType == '<')
      continue;

    // We cannot prove that the loop is vectorizable.
    return false;
  }
  return true;
}

std::optional<bool> LoopInterchangeProfitability::isProfitableForVectorization(
    unsigned InnerLoopId, unsigned OuterLoopId, CharMatrix &DepMatrix) {
  // If the outer loop cannot be vectorized, it is not profitable to move this
  // to inner position.
  if (!canVectorize(DepMatrix, OuterLoopId))
    return false;

  // If the inner loop cannot be vectorized but the outer loop can be, then it
  // is profitable to interchange to enable inner loop parallelism.
  if (!canVectorize(DepMatrix, InnerLoopId))
    return true;

  // If both the inner and the outer loop can be vectorized, it is necessary to
  // check the cost of each vectorized loop for profitability decision. At this
  // time we do not have a cost model to estimate them, so return nullopt.
  // TODO: Estimate the cost of vectorized loop when both the outer and the
  // inner loop can be vectorized.
  return std::nullopt;
}

bool LoopInterchangeProfitability::isProfitable(
    const Loop *InnerLoop, const Loop *OuterLoop, unsigned InnerLoopId,
    unsigned OuterLoopId, CharMatrix &DepMatrix, CacheCostManager &CCM) {
  // Do not consider loops with a backedge that isn't taken, e.g. an
  // unconditional branch true/false, as candidates for interchange.
  // TODO: when interchange is forced, we should probably also allow
  // interchange for these loops, and thus this logic should be moved just
  // below the cost-model ignore check below. But this check is done first
  // to avoid the issue in #163954.
  const SCEV *InnerBTC = SE->getBackedgeTakenCount(InnerLoop);
  const SCEV *OuterBTC = SE->getBackedgeTakenCount(OuterLoop);
  if (InnerBTC && InnerBTC->isZero()) {
    LLVM_DEBUG(dbgs() << "Inner loop back-edge isn't taken, rejecting "
                         "single iteration loop\n");
    return false;
  }
  if (OuterBTC && OuterBTC->isZero()) {
    LLVM_DEBUG(dbgs() << "Outer loop back-edge isn't taken, rejecting "
                         "single iteration loop\n");
    return false;
  }

  // Return true if interchange is forced and the cost-model ignored.
  if (Profitabilities.size() == 1 && Profitabilities[0] == RuleTy::Ignore)
    return true;
  assert(noDuplicateRulesAndIgnore(Profitabilities) &&
         "Duplicate rules and option 'ignore' are not allowed");

  // isProfitable() is structured to avoid endless loop interchange. If the
  // highest priority rule (isProfitablePerLoopCacheAnalysis by default) could
  // decide the profitability then, profitability check will stop and return the
  // analysis result. If it failed to determine it (e.g., cache analysis failed
  // to analyze the loopnest due to delinearization issues) then go ahead the
  // second highest priority rule (isProfitablePerInstrOrderCost by default).
  // Likewise, if it failed to analysis the profitability then only, the last
  // rule (isProfitableForVectorization by default) will decide.
  std::optional<bool> shouldInterchange;
  for (RuleTy RT : Profitabilities) {
    switch (RT) {
    case RuleTy::PerLoopCacheAnalysis: {
      CacheCost *CC = CCM.getCacheCost();
      const DenseMap<const Loop *, unsigned> &CostMap = CCM.getCostMap();
      shouldInterchange = isProfitablePerLoopCacheAnalysis(CostMap, CC);
      break;
    }
    case RuleTy::PerInstrOrderCost:
      shouldInterchange = isProfitablePerInstrOrderCost();
      break;
    case RuleTy::ForVectorization:
      shouldInterchange =
          isProfitableForVectorization(InnerLoopId, OuterLoopId, DepMatrix);
      break;
    case RuleTy::Ignore:
      llvm_unreachable("Option 'ignore' is not supported with other options");
      break;
    }

    // If this rule could determine the profitability, don't call subsequent
    // rules.
    if (shouldInterchange.has_value())
      break;
  }

  if (!shouldInterchange.has_value()) {
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "InterchangeNotProfitable",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Insufficient information to calculate the cost of loop for "
                  "interchange.";
      });
    return false;
  } else if (!shouldInterchange.value()) {
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "InterchangeNotProfitable",
                                        InnerLoop->getStartLoc(),
                                        InnerLoop->getHeader())
               << "Interchanging loops is not considered to improve cache "
                  "locality nor vectorization.";
      });
    return false;
  }
  return true;
}

void LoopInterchangeTransform::removeChildLoop(Loop *OuterLoop,
                                               Loop *InnerLoop) {
  for (Loop *L : *OuterLoop)
    if (L == InnerLoop) {
      OuterLoop->removeChildLoop(L);
      return;
    }
  llvm_unreachable("Couldn't find loop");
}

/// Update LoopInfo, after interchanging. NewInner and NewOuter refer to the
/// new inner and outer loop after interchanging: NewInner is the original
/// outer loop and NewOuter is the original inner loop.
///
/// Before interchanging, we have the following structure
/// Outer preheader
//  Outer header
//    Inner preheader
//    Inner header
//      Inner body
//      Inner latch
//   outer bbs
//   Outer latch
//
// After interchanging:
// Inner preheader
// Inner header
//   Outer preheader
//   Outer header
//     Inner body
//     outer bbs
//     Outer latch
//   Inner latch
void LoopInterchangeTransform::restructureLoops(
    Loop *NewInner, Loop *NewOuter, BasicBlock *OrigInnerPreHeader,
    BasicBlock *OrigOuterPreHeader) {
  Loop *OuterLoopParent = OuterLoop->getParentLoop();
  // The original inner loop preheader moves from the new inner loop to
  // the parent loop, if there is one.
  NewInner->removeBlockFromLoop(OrigInnerPreHeader);
  LI->changeLoopFor(OrigInnerPreHeader, OuterLoopParent);

  // Switch the loop levels.
  removeChildLoop(NewInner, NewOuter);
  // Replace NewInner with NewOuter in place, preserving sibling order.
  LI->replaceLoop(NewInner, NewOuter);

  while (!NewOuter->isInnermost())
    NewInner->addChildLoop(NewOuter->removeChildLoop(NewOuter->begin()));
  NewOuter->addChildLoop(NewInner);

  // BBs from the original inner loop.
  SmallVector<BasicBlock *, 8> OrigInnerBBs(NewOuter->blocks());

  // Add BBs from the original outer loop to the original inner loop (excluding
  // BBs already in inner loop)
  for (BasicBlock *BB : NewInner->blocks())
    if (LI->getLoopFor(BB) == NewInner)
      NewOuter->addBlockEntry(BB);

  // Now remove inner loop header and latch from the new inner loop and move
  // other BBs (the loop body) to the new inner loop.
  BasicBlock *OuterHeader = NewOuter->getHeader();
  BasicBlock *OuterLatch = NewOuter->getLoopLatch();
  for (BasicBlock *BB : OrigInnerBBs) {
    // Nothing will change for BBs in child loops.
    if (LI->getLoopFor(BB) != NewOuter)
      continue;
    // Remove the new outer loop header and latch from the new inner loop.
    if (BB == OuterHeader || BB == OuterLatch)
      NewInner->removeBlockFromLoop(BB);
    else
      LI->changeLoopFor(BB, NewInner);
  }

  // The preheader of the original outer loop becomes part of the new
  // outer loop.
  NewOuter->addBlockEntry(OrigOuterPreHeader);
  LI->changeLoopFor(OrigOuterPreHeader, NewOuter);

  // Tell SE that we move the loops around.
  SE->forgetLoop(NewOuter);
}

///  User can write, or optimizers can generate the reduction for inner loop.
///  To make the interchange valid, apply Reduction2Mem by moving the
///  initializer and store instructions into the inner loop. So far we only
///  handle cases where the reduction variable is initialized to a constant.
///  For example, below code:
///
///  loop:
///    re = phi<0.0, next>
///    next = re op ...
///  endloop
///  reduc_sum = phi<next>       // lcssa phi
///  MEM_REF[idx] = reduc_sum    // LcssaStore
///
///  is transformed into:
///
///  loop:
///    tmp = MEM_REF[idx];
///    new_var = !first_iteration ? tmp : 0.0;
///    next = new_var op ...
///    MEM_REF[idx] = next;		// after moving
///  endloop
///
///  In this way the initial const is used in the first iteration of loop.
void LoopInterchangeTransform::reduction2Memory() {
  ArrayRef<LoopInterchangeLegality::InnerReduction> InnerReductions =
      LIL.getInnerReductions();

  assert(InnerReductions.size() == 1 &&
         "So far we only support at most one reduction.");

  LoopInterchangeLegality::InnerReduction SR = InnerReductions[0];
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  IRBuilder<> Builder(InnerLoopHeader, InnerLoopHeader->getFirstNonPHIIt());

  // Check if it's the first iteration.
  LLVMContext &Context = InnerLoopHeader->getContext();
  PHINode *FirstIter =
      Builder.CreatePHI(Type::getInt1Ty(Context), 2, "first.iter");
  FirstIter->addIncoming(ConstantInt::get(Type::getInt1Ty(Context), 1),
                         InnerLoop->getLoopPreheader());
  FirstIter->addIncoming(ConstantInt::get(Type::getInt1Ty(Context), 0),
                         InnerLoop->getLoopLatch());
  assert(FirstIter->isComplete() && "The FirstIter PHI node is not complete.");

  // When the reduction is initialized from a constant value, we need to add
  // a stmt loading from the memory object to target basic block in inner
  // loop.
  Instruction *LoadMem = Builder.CreateLoad(SR.ElemTy, SR.MemRef);

  // Init new_var to MEM_REF or CONST depending on if it is the first iteration.
  Value *NewVar = Builder.CreateSelect(FirstIter, SR.Init, LoadMem, "new.var");

  // Replace all uses of the reduction variable with a new variable.
  SR.Reduction->replaceAllUsesWith(NewVar);

  // Move store instruction into inner loop, just after reduction next's
  // definition.
  SR.LcssaStore->setOperand(0, SR.Next);
  SR.LcssaStore->moveAfter(dyn_cast<Instruction>(SR.Next));
}

void LoopInterchangeTransform::transform(
    ArrayRef<Instruction *> DropNoWrapInsts,
    ArrayRef<Instruction *> DropNoInfInsts) {

  ArrayRef<LoopInterchangeLegality::InnerReduction> InnerReductions =
      LIL.getInnerReductions();
  if (InnerReductions.size() == 1)
    reduction2Memory();

  LLVM_DEBUG(dbgs() << "Splitting the inner loop latch\n");
  auto &InductionPHIs = LIL.getInnerLoopInductions();
  assert(!InductionPHIs.empty() &&
         "Expected at least one induction variable in the inner loop");

  SmallVector<Instruction *, 8> InnerIndexVarList;
  for (PHINode *CurInductionPHI : InductionPHIs) {
    Instruction *IncomingValue = dyn_cast<Instruction>(
        CurInductionPHI->getIncomingValueForBlock(InnerLoop->getLoopLatch()));
    assert(IncomingValue &&
           "Incoming value from loop latch isn't an instruction");
    if (is_contained(InductionPHIs, IncomingValue))
      continue;
    InnerIndexVarList.push_back(IncomingValue);
  }

  // Create a new latch block for the inner loop. We split at the
  // current latch's terminator and then move the condition and all
  // operands that are not either loop-invariant or the induction PHI into the
  // new latch block.
  BasicBlock *NewLatch =
      SplitBlock(InnerLoop->getLoopLatch(),
                 InnerLoop->getLoopLatch()->getTerminator(), DT, LI);

  // Keep these seeds and the operand filter aligned with
  // findFreezeInInnerLatchCloneSet.
  SmallSetVector<Instruction *, 4> WorkList;
  unsigned i = 0;
  auto MoveInstructions = [&i, &WorkList, this, &InductionPHIs, NewLatch]() {
    for (; i < WorkList.size(); i++) {
      // PHI nodes cannot be cloned and moved here; the legality check
      // (areInnerLoopLatchPHIsSupported) ensures none reach the worklist.
      assert(!isa<PHINode>(WorkList[i]) &&
             "MoveInstructions does not support PHI nodes");
      // Duplicate instruction and move it to the new latch. Update uses that
      // have been moved.
      Instruction *NewI = WorkList[i]->clone();
      NewI->insertBefore(NewLatch->getFirstNonPHIIt());
      assert(!NewI->mayHaveSideEffects() &&
             "Moving instructions with side-effects may change behavior of "
             "the loop nest!");
      for (Use &U : llvm::make_early_inc_range(WorkList[i]->uses())) {
        Instruction *UserI = cast<Instruction>(U.getUser());
        if (!InnerLoop->contains(UserI->getParent()) ||
            UserI->getParent() == NewLatch ||
            llvm::is_contained(InductionPHIs, UserI))
          U.set(NewI);
      }
      // Add operands of moved instruction to the worklist, except if they are
      // outside the inner loop or are the induction PHI.
      for (Value *Op : WorkList[i]->operands()) {
        Instruction *OpI = dyn_cast<Instruction>(Op);
        if (!OpI || this->LI->getLoopFor(OpI->getParent()) != this->InnerLoop ||
            llvm::is_contained(InductionPHIs, OpI))
          continue;
        WorkList.insert(OpI);
      }
    }
  };

  // FIXME: Should we interchange when we have a constant condition?
  Instruction *CondI = dyn_cast<Instruction>(
      cast<CondBrInst>(InnerLoop->getLoopLatch()->getTerminator())
          ->getCondition());
  if (CondI)
    WorkList.insert(CondI);
  MoveInstructions();
  for (Instruction *InnerIndexVar : InnerIndexVarList)
    WorkList.insert(cast<Instruction>(InnerIndexVar));
  MoveInstructions();

  // Split the inner header so that it has a unique successor.
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  SplitBlock(InnerLoopHeader, InnerLoopHeader->getFirstNonPHIIt(), DT, LI);
  LLVM_DEBUG(dbgs() << "splitting InnerLoopHeader done\n");

  // Instructions in the original inner loop preheader may depend on values
  // defined in the outer loop header. Move them there, because the original
  // inner loop preheader will become the entry into the interchanged loop nest.
  // Currently we move all instructions and rely on LICM to move invariant
  // instructions outside the loop nest.
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();

  if (InnerLoopPreHeader != OuterLoopHeader) {
    // Eliminate PHIs in the inner-loop preheader.
    for (PHINode &P : make_early_inc_range(InnerLoopPreHeader->phis())) {
      assert(all_equal(P.incoming_values()) &&
             "Expected equivalent incoming values in inner loop preheader");
      P.replaceAllUsesWith(P.getIncomingValue(0));
      P.eraseFromParent();
    }
    for (Instruction &I :
         make_early_inc_range(make_range(InnerLoopPreHeader->begin(),
                                         std::prev(InnerLoopPreHeader->end()))))
      I.moveBeforePreserving(OuterLoopHeader->getTerminator()->getIterator());
  }

  adjustLoopBranches();

  // Finally, drop the nsw/nuw/ninf flags from the instructions for reduction
  // calculations.
  for (Instruction *Reduction : DropNoWrapInsts) {
    Reduction->setHasNoSignedWrap(false);
    Reduction->setHasNoUnsignedWrap(false);
  }
  for (Instruction *I : DropNoInfInsts)
    I->setHasNoInfs(false);
}

/// \brief Move all instructions except the terminator from FromBB right before
/// InsertBefore
static void moveBBContents(BasicBlock *FromBB, Instruction *InsertBefore) {
  BasicBlock *ToBB = InsertBefore->getParent();

  ToBB->splice(InsertBefore->getIterator(), FromBB, FromBB->begin(),
               FromBB->getTerminator()->getIterator());
}

/// Swap instructions between \p BB1 and \p BB2 but keep terminators intact.
static void swapBBContents(BasicBlock *BB1, BasicBlock *BB2) {
  // Save all non-terminator instructions of BB1 into TempInstrs and unlink them
  // from BB1 afterwards.
  auto Iter = map_range(*BB1, [](Instruction &I) { return &I; });
  SmallVector<Instruction *, 4> TempInstrs(Iter.begin(), std::prev(Iter.end()));
  for (Instruction *I : TempInstrs)
    I->removeFromParent();

  // Move instructions from BB2 to BB1.
  moveBBContents(BB2, BB1->getTerminator());

  // Move instructions from TempInstrs to BB2.
  for (Instruction *I : TempInstrs)
    I->insertBefore(BB2->getTerminator()->getIterator());
}

// Update BI to jump to NewBB instead of OldBB. Records updates to the
// dominator tree in DTUpdates. If \p MustUpdateOnce is true, assert that
// \p OldBB  is exactly once in BI's successor list.
static void updateSuccessor(Instruction *Term, BasicBlock *OldBB,
                            BasicBlock *NewBB,
                            std::vector<DominatorTree::UpdateType> &DTUpdates,
                            bool MustUpdateOnce = true) {
  assert((!MustUpdateOnce || llvm::count(successors(Term), OldBB) == 1) &&
         "BI must jump to OldBB exactly once.");
  bool Changed = false;
  for (Use &Op : Term->operands())
    if (Op == OldBB) {
      Op.set(NewBB);
      Changed = true;
    }

  if (Changed) {
    DTUpdates.push_back(
        {DominatorTree::UpdateKind::Insert, Term->getParent(), NewBB});
    DTUpdates.push_back(
        {DominatorTree::UpdateKind::Delete, Term->getParent(), OldBB});
  }
  assert(Changed && "Expected a successor to be updated");
}

// Move Lcssa PHIs to the right place.
static void moveLCSSAPhis(BasicBlock *InnerExit, BasicBlock *InnerHeader,
                          BasicBlock *InnerLatch, BasicBlock *OuterHeader,
                          BasicBlock *OuterLatch, BasicBlock *OuterExit,
                          Loop *InnerLoop, LoopInfo *LI) {

  // Deal with LCSSA PHI nodes in the exit block of the inner loop, that are
  // defined either in the header or latch. Those blocks will become header and
  // latch of the new outer loop, and the only possible users can PHI nodes
  // in the exit block of the loop nest or the outer loop header (reduction
  // PHIs, in that case, the incoming value must be defined in the inner loop
  // header). We can just substitute the user with the incoming value and remove
  // the PHI.
  for (PHINode &P : make_early_inc_range(InnerExit->phis())) {
    assert(P.getNumIncomingValues() == 1 &&
           "Only loops with a single exit are supported!");

    Value *IncomingValue = P.getIncomingValueForBlock(InnerLatch);
    auto *IncI = dyn_cast<Instruction>(IncomingValue);
    if (!IncI) {
      // If the incoming value is not an instruction, it must be loop invariant.
      // In that case, we can just replace the PHI with the incoming value and
      // remove the PHI.
      assert(InnerLoop->isLoopInvariant(IncomingValue) &&
             "Expected non-instruction incoming value to be loop invariant");
      P.replaceAllUsesWith(IncomingValue);
      P.eraseFromParent();
      continue;
    }

    // In case of multi-level nested loops, follow LCSSA to find the incoming
    // value defined from the innermost loop.
    auto *IncIInnerMost = dyn_cast<Instruction>(followLCSSA(IncI));
    // Skip phis when:
    // - they are not an instruction, e.g. incoming values are constants.
    // - Incomming values from the inner loop body, excluding the header and
    //   latch.
    if (!IncIInnerMost || (IncIInnerMost->getParent() != InnerLatch &&
                           IncIInnerMost->getParent() != InnerHeader))
      continue;

    assert(all_of(P.users(),
                  [OuterHeader, OuterExit, IncI, InnerHeader](User *U) {
                    return (cast<PHINode>(U)->getParent() == OuterHeader &&
                            IncI->getParent() == InnerHeader) ||
                           cast<PHINode>(U)->getParent() == OuterExit;
                  }) &&
           "Can only replace phis iff the uses are in the loop nest exit or "
           "the incoming value is defined in the inner header (it will "
           "dominate all loop blocks after interchanging)");
    P.replaceAllUsesWith(IncI);
    P.eraseFromParent();
  }

  SmallVector<PHINode *, 8> LcssaInnerExit(
      llvm::make_pointer_range(InnerExit->phis()));

  SmallVector<PHINode *, 8> LcssaInnerLatch(
      llvm::make_pointer_range(InnerLatch->phis()));

  // Lcssa PHIs for values used outside the inner loop are in InnerExit.
  // If a PHI node has users outside of InnerExit, it has a use outside the
  // interchanged loop and we have to preserve it. We move these to
  // InnerLatch, which will become the new exit block for the innermost
  // loop after interchanging.
  for (PHINode *P : LcssaInnerExit)
    P->moveBefore(InnerLatch->getFirstNonPHIIt());

  // If the inner loop latch contains LCSSA PHIs, those come from a child loop
  // and we have to move them to the new inner latch.
  for (PHINode *P : LcssaInnerLatch)
    P->moveBefore(InnerExit->getFirstNonPHIIt());

  // Deal with LCSSA PHI nodes in the loop nest exit block. For PHIs that have
  // incoming values defined in the outer loop, we have to add a new PHI
  // in the inner loop latch, which became the exit block of the outer loop,
  // after interchanging.
  if (OuterExit) {
    for (PHINode &P : OuterExit->phis()) {
      if (P.getNumIncomingValues() != 1)
        continue;
      // Skip Phis with incoming values defined in the inner loop. Those should
      // already have been updated.
      auto I = dyn_cast<Instruction>(P.getIncomingValue(0));
      if (!I || LI->getLoopFor(I->getParent()) == InnerLoop)
        continue;

      PHINode *NewPhi = dyn_cast<PHINode>(P.clone());
      NewPhi->setIncomingValue(0, P.getIncomingValue(0));
      NewPhi->setIncomingBlock(0, OuterLatch);
      // We might have incoming edges from other BBs, i.e., the original outer
      // header.
      for (auto *Pred : predecessors(InnerLatch)) {
        if (Pred == OuterLatch)
          continue;
        NewPhi->addIncoming(P.getIncomingValue(0), Pred);
      }
      NewPhi->insertBefore(InnerLatch->getFirstNonPHIIt());
      P.setIncomingValue(0, NewPhi);
    }
  }

  // Now adjust the incoming blocks for the LCSSA PHIs.
  // For PHIs moved from Inner's exit block, we need to replace Inner's latch
  // with the new latch.
  InnerLatch->replacePhiUsesWith(InnerLatch, OuterLatch);
}

/// This deals with a corner case when a LCSSA phi node appears in a non-exit
/// block: the outer loop latch block does not need to be exit block of the
/// inner loop. Consider a loop that was in LCSSA form, but then some
/// transformation like loop-unswitch comes along and creates an empty block,
/// where BB5 in this example is the outer loop latch block:
///
///   BB4:
///     br label %BB5
///   BB5:
///     %old.cond.lcssa = phi i16 [ %cond, %BB4 ]
///     br outer.header
///
/// Interchange then brings it in LCSSA form again resulting in this chain of
/// single-input phi nodes:
///
///   BB4:
///     %new.cond.lcssa = phi i16 [ %cond, %BB3 ]
///     br label %BB5
///   BB5:
///     %old.cond.lcssa = phi i16 [ %new.cond.lcssa, %BB4 ]
///
/// The problem is that interchange can reoder blocks BB4 and BB5 placing the
/// use before the def if we don't check this. The solution is to simplify
/// lcssa phi nodes (remove) if they appear in non-exit blocks.
///
static void simplifyLCSSAPhis(Loop *OuterLoop, Loop *InnerLoop) {
  BasicBlock *InnerLoopExit = InnerLoop->getExitBlock();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();

  // Do not modify lcssa phis where they actually belong, i.e. in exit blocks.
  if (OuterLoopLatch == InnerLoopExit)
    return;

  // Collect and remove phis in non-exit blocks if they have 1 input.
  SmallVector<PHINode *, 8> Phis(
      llvm::make_pointer_range(OuterLoopLatch->phis()));
  for (PHINode *Phi : Phis) {
    assert(Phi->getNumIncomingValues() == 1 && "Single input phi expected");
    LLVM_DEBUG(dbgs() << "Removing 1-input phi in non-exit block: " << *Phi
                      << "\n");
    Phi->replaceAllUsesWith(Phi->getIncomingValue(0));
    Phi->eraseFromParent();
  }
}

void LoopInterchangeTransform::adjustLoopBranches() {
  LLVM_DEBUG(dbgs() << "adjustLoopBranches called\n");
  std::vector<DominatorTree::UpdateType> DTUpdates;

  BasicBlock *OuterLoopPreHeader = OuterLoop->getLoopPreheader();
  BasicBlock *InnerLoopPreHeader = InnerLoop->getLoopPreheader();

  assert(OuterLoopPreHeader != OuterLoop->getHeader() &&
         InnerLoopPreHeader != InnerLoop->getHeader() && OuterLoopPreHeader &&
         InnerLoopPreHeader && "Guaranteed by loop-simplify form");

  simplifyLCSSAPhis(OuterLoop, InnerLoop);

  // Ensure that both preheaders do not contain PHI nodes and have single
  // predecessors. This allows us to move them easily. We use
  // InsertPreHeaderForLoop to create an 'extra' preheader, if the existing
  // preheaders do not satisfy those conditions.
  if (isa<PHINode>(OuterLoopPreHeader->begin()) ||
      !OuterLoopPreHeader->getUniquePredecessor())
    OuterLoopPreHeader =
        InsertPreheaderForLoop(OuterLoop, DT, LI, nullptr, true);
  if (InnerLoopPreHeader == OuterLoop->getHeader())
    InnerLoopPreHeader =
        InsertPreheaderForLoop(InnerLoop, DT, LI, nullptr, true);

  // Adjust the loop preheader
  BasicBlock *InnerLoopHeader = InnerLoop->getHeader();
  BasicBlock *OuterLoopHeader = OuterLoop->getHeader();
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  BasicBlock *OuterLoopLatch = OuterLoop->getLoopLatch();
  BasicBlock *OuterLoopPredecessor = OuterLoopPreHeader->getUniquePredecessor();
  BasicBlock *InnerLoopLatchPredecessor =
      InnerLoopLatch->getUniquePredecessor();
  BasicBlock *InnerLoopLatchSuccessor;
  BasicBlock *OuterLoopLatchSuccessor;

  CondBrInst *OuterLoopLatchBI =
      dyn_cast<CondBrInst>(OuterLoopLatch->getTerminator());
  CondBrInst *InnerLoopLatchBI =
      dyn_cast<CondBrInst>(InnerLoopLatch->getTerminator());
  Instruction *OuterLoopHeaderBI = OuterLoopHeader->getTerminator();
  Instruction *InnerLoopHeaderBI = InnerLoopHeader->getTerminator();

  assert(OuterLoopPredecessor && InnerLoopLatchPredecessor &&
         "Failed to find a unique predecessor");
  assert(OuterLoopLatchBI && InnerLoopLatchBI &&
         "Failed to find a conditional branch");

  Instruction *InnerLoopLatchPredecessorBI =
      InnerLoopLatchPredecessor->getTerminator();
  Instruction *OuterLoopPredecessorBI = OuterLoopPredecessor->getTerminator();

  BasicBlock *InnerLoopHeaderSuccessor = InnerLoopHeader->getUniqueSuccessor();
  assert(InnerLoopHeaderSuccessor &&
         "Failed to find a unique successor for the inner loop header");

  // Adjust Loop Preheader and headers.
  // The branches in the outer loop predecessor and the outer loop header can
  // be unconditional branches or conditional branches with duplicates. Consider
  // this when updating the successors.
  updateSuccessor(OuterLoopPredecessorBI, OuterLoopPreHeader,
                  InnerLoopPreHeader, DTUpdates, /*MustUpdateOnce=*/false);
  // The outer loop header might or might not branch to the outer latch.
  // We are guaranteed to branch to the inner loop preheader.
  if (llvm::is_contained(successors(OuterLoopHeaderBI), OuterLoopLatch)) {
    // In this case the outerLoopHeader should branch to the InnerLoopLatch.
    updateSuccessor(OuterLoopHeaderBI, OuterLoopLatch, InnerLoopLatch,
                    DTUpdates,
                    /*MustUpdateOnce=*/false);
  }
  updateSuccessor(OuterLoopHeaderBI, InnerLoopPreHeader,
                  InnerLoopHeaderSuccessor, DTUpdates,
                  /*MustUpdateOnce=*/false);

  // Adjust reduction PHI's now that the incoming block has changed.
  InnerLoopHeaderSuccessor->replacePhiUsesWith(InnerLoopHeader,
                                               OuterLoopHeader);

  updateSuccessor(InnerLoopHeaderBI, InnerLoopHeaderSuccessor,
                  OuterLoopPreHeader, DTUpdates);

  // -------------Adjust loop latches-----------
  if (InnerLoopLatchBI->getSuccessor(0) == InnerLoopHeader)
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(1);
  else
    InnerLoopLatchSuccessor = InnerLoopLatchBI->getSuccessor(0);

  updateSuccessor(InnerLoopLatchPredecessorBI, InnerLoopLatch,
                  InnerLoopLatchSuccessor, DTUpdates);

  if (OuterLoopLatchBI->getSuccessor(0) == OuterLoopHeader)
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(1);
  else
    OuterLoopLatchSuccessor = OuterLoopLatchBI->getSuccessor(0);

  updateSuccessor(InnerLoopLatchBI, InnerLoopLatchSuccessor,
                  OuterLoopLatchSuccessor, DTUpdates);
  updateSuccessor(OuterLoopLatchBI, OuterLoopLatchSuccessor, InnerLoopLatch,
                  DTUpdates);

  DT->applyUpdates(DTUpdates);
  restructureLoops(OuterLoop, InnerLoop, InnerLoopPreHeader,
                   OuterLoopPreHeader);

  moveLCSSAPhis(InnerLoopLatchSuccessor, InnerLoopHeader, InnerLoopLatch,
                OuterLoopHeader, OuterLoopLatch, InnerLoop->getExitBlock(),
                InnerLoop, LI);
  // For PHIs in the exit block of the outer loop, outer's latch has been
  // replaced by Inners'.
  OuterLoopLatchSuccessor->replacePhiUsesWith(OuterLoopLatch, InnerLoopLatch);

  auto &OuterInnerReductions = LIL.getOuterInnerReductions();
  // Now update the reduction PHIs in the inner and outer loop headers.
  SmallVector<PHINode *, 4> InnerLoopPHIs, OuterLoopPHIs;
  for (PHINode &PHI : InnerLoopHeader->phis())
    if (OuterInnerReductions.contains(&PHI))
      InnerLoopPHIs.push_back(&PHI);

  for (PHINode &PHI : OuterLoopHeader->phis())
    if (OuterInnerReductions.contains(&PHI))
      OuterLoopPHIs.push_back(&PHI);

  // Now move the remaining reduction PHIs from outer to inner loop header and
  // vice versa. The PHI nodes must be part of a reduction across the inner and
  // outer loop and all the remains to do is and updating the incoming blocks.
  for (PHINode *PHI : OuterLoopPHIs) {
    LLVM_DEBUG(dbgs() << "Outer loop reduction PHIs:\n"; PHI->dump(););
    PHI->moveBefore(InnerLoopHeader->getFirstNonPHIIt());
    assert(OuterInnerReductions.count(PHI) && "Expected a reduction PHI node");
  }
  for (PHINode *PHI : InnerLoopPHIs) {
    LLVM_DEBUG(dbgs() << "Inner loop reduction PHIs:\n"; PHI->dump(););
    PHI->moveBefore(OuterLoopHeader->getFirstNonPHIIt());
    assert(OuterInnerReductions.count(PHI) && "Expected a reduction PHI node");
  }

  // Update the incoming blocks for moved PHI nodes.
  OuterLoopHeader->replacePhiUsesWith(InnerLoopPreHeader, OuterLoopPreHeader);
  OuterLoopHeader->replacePhiUsesWith(InnerLoopLatch, OuterLoopLatch);
  InnerLoopHeader->replacePhiUsesWith(OuterLoopPreHeader, InnerLoopPreHeader);
  InnerLoopHeader->replacePhiUsesWith(OuterLoopLatch, InnerLoopLatch);

  // Swap the preheader contents so each definition sits in the preheader of the
  // loop it now belongs to. This runs before the LCSSA rebuild below so that
  // any definition referenced across the interchanged levels dominates its uses
  // when formLCSSAForInstructions runs.
  swapBBContents(OuterLoop->getLoopPreheader(), InnerLoop->getLoopPreheader());

  // Values defined in the outer loop header could be used in the inner loop
  // latch. In that case, we need to create LCSSA phis for them, because after
  // interchanging they will be defined in the new inner loop and used in the
  // new outer loop.
  SmallVector<Instruction *, 4> MayNeedLCSSAPhis;
  for (Instruction &I :
       make_range(OuterLoopHeader->begin(), std::prev(OuterLoopHeader->end())))
    MayNeedLCSSAPhis.push_back(&I);

#ifndef NDEBUG
  assert(!verifyFunction(*OuterLoopHeader->getParent(), &errs()) &&
         "LoopInterchange handed dominance-broken IR to LCSSA rebuild");
#endif

  formLCSSAForInstructions(MayNeedLCSSAPhis, *DT, *LI, SE);
}

PreservedAnalyses LoopInterchangePass::run(LoopNest &LN,
                                           LoopAnalysisManager &AM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &U) {
  Function &F = *LN.getParent();

  OptimizationRemarkEmitter ORE(&F);

  ORE.emit([&]() {
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "Dependence",
                                      LN.getOutermostLoop().getStartLoc(),
                                      LN.getOutermostLoop().getHeader())
           << "Computed dependence info, invoking the transform.";
  });

  DependenceInfo DI(&F, &AR.AA, &AR.SE, &AR.LI);
  if (!LoopInterchange(&AR.SE, &AR.LI, &DI, &AR.DT, &AR, &ORE).run(LN, U))
    return PreservedAnalyses::all();
  U.markLoopNestChanged(true);
  return getLoopPassPreservedAnalyses();
}
