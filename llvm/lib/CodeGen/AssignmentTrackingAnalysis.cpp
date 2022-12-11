#include "llvm/CodeGen/AssignmentTrackingAnalysis.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Analysis/Interval.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <assert.h>
#include <cstdint>
#include <optional>
#include <sstream>
#include <unordered_map>

using namespace llvm;
#define DEBUG_TYPE "debug-ata"

STATISTIC(NumDefsScanned, "Number of dbg locs that get scanned for removal");
STATISTIC(NumDefsRemoved, "Number of dbg locs removed");
STATISTIC(NumWedgesScanned, "Number of dbg wedges scanned");
STATISTIC(NumWedgesChanged, "Number of dbg wedges changed");

static cl::opt<unsigned>
    MaxNumBlocks("debug-ata-max-blocks", cl::init(10000),
                 cl::desc("Maximum num basic blocks before debug info dropped"),
                 cl::Hidden);
/// Option for debugging the pass, determines if the memory location fragment
/// filling happens after generating the variable locations.
static cl::opt<bool> EnableMemLocFragFill("mem-loc-frag-fill", cl::init(true),
                                          cl::Hidden);
/// Print the results of the analysis. Respects -filter-print-funcs.
static cl::opt<bool> PrintResults("print-debug-ata", cl::init(false),
                                  cl::Hidden);

// Implicit conversions are disabled for enum class types, so unfortunately we
// need to create a DenseMapInfo wrapper around the specified underlying type.
template <> struct llvm::DenseMapInfo<VariableID> {
  using Wrapped = DenseMapInfo<unsigned>;
  static inline VariableID getEmptyKey() {
    return static_cast<VariableID>(Wrapped::getEmptyKey());
  }
  static inline VariableID getTombstoneKey() {
    return static_cast<VariableID>(Wrapped::getTombstoneKey());
  }
  static unsigned getHashValue(const VariableID &Val) {
    return Wrapped::getHashValue(static_cast<unsigned>(Val));
  }
  static bool isEqual(const VariableID &LHS, const VariableID &RHS) {
    return LHS == RHS;
  }
};

/// Helper class to build FunctionVarLocs, since that class isn't easy to
/// modify. TODO: There's not a great deal of value in the split, it could be
/// worth merging the two classes.
class FunctionVarLocsBuilder {
  friend FunctionVarLocs;
  UniqueVector<DebugVariable> Variables;
  // Use an unordered_map so we don't invalidate iterators after
  // insert/modifications.
  std::unordered_map<const Instruction *, SmallVector<VarLocInfo>>
      VarLocsBeforeInst;

  SmallVector<VarLocInfo> SingleLocVars;

public:
  /// Find or insert \p V and return the ID.
  VariableID insertVariable(DebugVariable V) {
    return static_cast<VariableID>(Variables.insert(V));
  }

  /// Get a variable from its \p ID.
  const DebugVariable &getVariable(VariableID ID) const {
    return Variables[static_cast<unsigned>(ID)];
  }

  /// Return ptr to wedge of defs or nullptr if no defs come just before /p
  /// Before.
  const SmallVectorImpl<VarLocInfo> *getWedge(const Instruction *Before) const {
    auto R = VarLocsBeforeInst.find(Before);
    if (R == VarLocsBeforeInst.end())
      return nullptr;
    return &R->second;
  }

  /// Replace the defs that come just before /p Before with /p Wedge.
  void setWedge(const Instruction *Before, SmallVector<VarLocInfo> &&Wedge) {
    VarLocsBeforeInst[Before] = std::move(Wedge);
  }

  /// Add a def for a variable that is valid for its lifetime.
  void addSingleLocVar(DebugVariable Var, DIExpression *Expr, DebugLoc DL,
                       Value *V) {
    VarLocInfo VarLoc;
    VarLoc.VariableID = insertVariable(Var);
    VarLoc.Expr = Expr;
    VarLoc.DL = DL;
    VarLoc.V = V;
    SingleLocVars.emplace_back(VarLoc);
  }

  /// Add a def to the wedge of defs just before /p Before.
  void addVarLoc(Instruction *Before, DebugVariable Var, DIExpression *Expr,
                 DebugLoc DL, Value *V) {
    VarLocInfo VarLoc;
    VarLoc.VariableID = insertVariable(Var);
    VarLoc.Expr = Expr;
    VarLoc.DL = DL;
    VarLoc.V = V;
    VarLocsBeforeInst[Before].emplace_back(VarLoc);
  }
};

void FunctionVarLocs::print(raw_ostream &OS, const Function &Fn) const {
  // Print the variable table first. TODO: Sorting by variable could make the
  // output more stable?
  unsigned Counter = -1;
  OS << "=== Variables ===\n";
  for (const DebugVariable &V : Variables) {
    ++Counter;
    // Skip first entry because it is a dummy entry.
    if (Counter == 0) {
      continue;
    }
    OS << "[" << Counter << "] " << V.getVariable()->getName();
    if (auto F = V.getFragment())
      OS << " bits [" << F->OffsetInBits << ", "
         << F->OffsetInBits + F->SizeInBits << ")";
    if (const auto *IA = V.getInlinedAt())
      OS << " inlined-at " << *IA;
    OS << "\n";
  }

  auto PrintLoc = [&OS](const VarLocInfo &Loc) {
    OS << "DEF Var=[" << (unsigned)Loc.VariableID << "]"
       << " Expr=" << *Loc.Expr << " V=" << *Loc.V << "\n";
  };

  // Print the single location variables.
  OS << "=== Single location vars ===\n";
  for (auto It = single_locs_begin(), End = single_locs_end(); It != End;
       ++It) {
    PrintLoc(*It);
  }

  // Print the non-single-location defs in line with IR.
  OS << "=== In-line variable defs ===";
  for (const BasicBlock &BB : Fn) {
    OS << "\n" << BB.getName() << ":\n";
    for (const Instruction &I : BB) {
      for (auto It = locs_begin(&I), End = locs_end(&I); It != End; ++It) {
        PrintLoc(*It);
      }
      OS << I << "\n";
    }
  }
}

void FunctionVarLocs::init(FunctionVarLocsBuilder &Builder) {
  // Add the single-location variables first.
  for (const auto &VarLoc : Builder.SingleLocVars)
    VarLocRecords.emplace_back(VarLoc);
  // Mark the end of the section.
  SingleVarLocEnd = VarLocRecords.size();

  // Insert a contiguous block of VarLocInfos for each instruction, mapping it
  // to the start and end position in the vector with VarLocsBeforeInst.
  for (auto &P : Builder.VarLocsBeforeInst) {
    unsigned BlockStart = VarLocRecords.size();
    for (const VarLocInfo &VarLoc : P.second)
      VarLocRecords.emplace_back(VarLoc);
    unsigned BlockEnd = VarLocRecords.size();
    // Record the start and end indices.
    if (BlockEnd != BlockStart)
      VarLocsBeforeInst[P.first] = {BlockStart, BlockEnd};
  }

  // Copy the Variables vector from the builder's UniqueVector.
  assert(Variables.empty() && "Expect clear before init");
  // UniqueVectors IDs are one-based (which means the VarLocInfo VarID values
  // are one-based) so reserve an extra and insert a dummy.
  Variables.reserve(Builder.Variables.size() + 1);
  Variables.push_back(DebugVariable(nullptr, std::nullopt, nullptr));
  Variables.append(Builder.Variables.begin(), Builder.Variables.end());
}

void FunctionVarLocs::clear() {
  Variables.clear();
  VarLocRecords.clear();
  VarLocsBeforeInst.clear();
  SingleVarLocEnd = 0;
}

/// Walk backwards along constant GEPs and bitcasts to the base storage from \p
/// Start as far as possible. Prepend \Expression with the offset and append it
/// with a DW_OP_deref that haes been implicit until now. Returns the walked-to
/// value and modified expression.
static std::pair<Value *, DIExpression *>
walkToAllocaAndPrependOffsetDeref(const DataLayout &DL, Value *Start,
                                  DIExpression *Expression) {
  APInt OffsetInBytes(DL.getTypeSizeInBits(Start->getType()), false);
  Value *End =
      Start->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetInBytes);
  SmallVector<uint64_t, 3> Ops;
  if (OffsetInBytes.getBoolValue()) {
    Ops = {dwarf::DW_OP_plus_uconst, OffsetInBytes.getZExtValue()};
    Expression = DIExpression::prependOpcodes(
        Expression, Ops, /*StackValue=*/false, /*EntryValue=*/false);
  }
  Expression = DIExpression::append(Expression, {dwarf::DW_OP_deref});
  return {End, Expression};
}

/// Extract the offset used in \p DIExpr. Returns std::nullopt if the expression
/// doesn't explicitly describe a memory location with DW_OP_deref or if the
/// expression is too complex to interpret.
static Optional<int64_t> getDerefOffsetInBytes(const DIExpression *DIExpr) {
  int64_t Offset = 0;
  const unsigned NumElements = DIExpr->getNumElements();
  const auto Elements = DIExpr->getElements();
  unsigned NextElement = 0;
  // Extract the offset.
  if (NumElements > 2 && Elements[0] == dwarf::DW_OP_plus_uconst) {
    Offset = Elements[1];
    NextElement = 2;
  } else if (NumElements > 3 && Elements[0] == dwarf::DW_OP_constu) {
    NextElement = 3;
    if (Elements[2] == dwarf::DW_OP_plus)
      Offset = Elements[1];
    else if (Elements[2] == dwarf::DW_OP_minus)
      Offset = -Elements[1];
    else
      return std::nullopt;
  }

  // If that's all there is it means there's no deref.
  if (NextElement >= NumElements)
    return std::nullopt;

  // Check the next element is DW_OP_deref - otherwise this is too complex or
  // isn't a deref expression.
  if (Elements[NextElement] != dwarf::DW_OP_deref)
    return std::nullopt;

  // Check the final operation is either the DW_OP_deref or is a fragment.
  if (NumElements == NextElement + 1)
    return Offset; // Ends with deref.
  else if (NumElements == NextElement + 3 &&
           Elements[NextElement] == dwarf::DW_OP_LLVM_fragment)
    return Offset; // Ends with deref + fragment.

  // Don't bother trying to interpret anything more complex.
  return std::nullopt;
}

/// A whole (unfragmented) source variable.
using DebugAggregate = std::pair<const DILocalVariable *, const DILocation *>;
static DebugAggregate getAggregate(const DbgVariableIntrinsic *DII) {
  return DebugAggregate(DII->getVariable(), DII->getDebugLoc().getInlinedAt());
}
static DebugAggregate getAggregate(const DebugVariable &Var) {
  return DebugAggregate(Var.getVariable(), Var.getInlinedAt());
}

/// In dwarf emission, the following sequence
///    1. dbg.value ... Fragment(0, 64)
///    2. dbg.value ... Fragment(0, 32)
/// effectively sets Fragment(32, 32) to undef (each def sets all bits not in
/// the intersection of the fragments to having "no location"). This makes
/// sense for implicit location values because splitting the computed values
/// could be troublesome, and is probably quite uncommon.  When we convert
/// dbg.assigns to dbg.value+deref this kind of thing is common, and describing
/// a location (memory) rather than a value means we don't need to worry about
/// splitting any values, so we try to recover the rest of the fragment
/// location here.
/// This class performs a(nother) dataflow analysis over the function, adding
/// variable locations so that any bits of a variable with a memory location
/// have that location explicitly reinstated at each subsequent variable
/// location definition that that doesn't overwrite those bits. i.e. after a
/// variable location def, insert new defs for the memory location with
/// fragments for the difference of "all bits currently in memory" and "the
/// fragment of the second def".
class MemLocFragmentFill {
  Function &Fn;
  FunctionVarLocsBuilder *FnVarLocs;
  const DenseSet<DebugAggregate> *VarsWithStackSlot;

  // 0 = no memory location.
  using BaseAddress = unsigned;
  using OffsetInBitsTy = unsigned;
  using FragTraits = IntervalMapHalfOpenInfo<OffsetInBitsTy>;
  using FragsInMemMap = IntervalMap<
      OffsetInBitsTy, BaseAddress,
      IntervalMapImpl::NodeSizer<OffsetInBitsTy, BaseAddress>::LeafSize,
      FragTraits>;
  FragsInMemMap::Allocator IntervalMapAlloc;
  using VarFragMap = DenseMap<unsigned, FragsInMemMap>;

  /// IDs for memory location base addresses in maps. Use 0 to indicate that
  /// there's no memory location.
  UniqueVector<Value *> Bases;
  UniqueVector<DebugAggregate> Aggregates;
  DenseMap<const BasicBlock *, VarFragMap> LiveIn;
  DenseMap<const BasicBlock *, VarFragMap> LiveOut;

  struct FragMemLoc {
    unsigned Var;
    unsigned Base;
    unsigned OffsetInBits;
    unsigned SizeInBits;
    DebugLoc DL;
  };
  using InsertMap = MapVector<Instruction *, SmallVector<FragMemLoc>>;

  /// BBInsertBeforeMap holds a description for the set of location defs to be
  /// inserted after the analysis is complete. It is updated during the dataflow
  /// and the entry for a block is CLEARED each time it is (re-)visited. After
  /// the dataflow is complete, each block entry will contain the set of defs
  /// calculated during the final (fixed-point) iteration.
  DenseMap<const BasicBlock *, InsertMap> BBInsertBeforeMap;

  static bool intervalMapsAreEqual(const FragsInMemMap &A,
                                   const FragsInMemMap &B) {
    auto AIt = A.begin(), AEnd = A.end();
    auto BIt = B.begin(), BEnd = B.end();
    for (; AIt != AEnd; ++AIt, ++BIt) {
      if (BIt == BEnd)
        return false; // B has fewer elements than A.
      if (AIt.start() != BIt.start() || AIt.stop() != BIt.stop())
        return false; // Interval is different.
      if (AIt.value() != BIt.value())
        return false; // Value at interval is different.
    }
    // AIt == AEnd. Check BIt is also now at end.
    return BIt == BEnd;
  }

  static bool varFragMapsAreEqual(const VarFragMap &A, const VarFragMap &B) {
    if (A.size() != B.size())
      return false;
    for (const auto &APair : A) {
      auto BIt = B.find(APair.first);
      if (BIt == B.end())
        return false;
      if (!intervalMapsAreEqual(APair.second, BIt->second))
        return false;
    }
    return true;
  }

  /// Return a string for the value that \p BaseID represents.
  std::string toString(unsigned BaseID) {
    if (BaseID)
      return Bases[BaseID]->getName().str();
    else
      return "None";
  }

  /// Format string describing an FragsInMemMap (IntervalMap) interval.
  std::string toString(FragsInMemMap::const_iterator It, bool Newline = true) {
    std::string String;
    std::stringstream S(String);
    if (It.valid()) {
      S << "[" << It.start() << ", " << It.stop()
        << "): " << toString(It.value());
    } else {
      S << "invalid iterator (end)";
    }
    if (Newline)
      S << "\n";
    return S.str();
  };

  FragsInMemMap meetFragments(const FragsInMemMap &A, const FragsInMemMap &B) {
    FragsInMemMap Result(IntervalMapAlloc);
    for (auto AIt = A.begin(), AEnd = A.end(); AIt != AEnd; ++AIt) {
      LLVM_DEBUG(dbgs() << "a " << toString(AIt));
      // This is basically copied from process() and inverted (process is
      // performing something like a union whereas this is more of an
      // intersect).

      // There's no work to do if interval `a` overlaps no fragments in map `B`.
      if (!B.overlaps(AIt.start(), AIt.stop()))
        continue;

      // Does StartBit intersect an existing fragment?
      auto FirstOverlap = B.find(AIt.start());
      assert(FirstOverlap != B.end());
      bool IntersectStart = FirstOverlap.start() < AIt.start();
      LLVM_DEBUG(dbgs() << "- FirstOverlap " << toString(FirstOverlap, false)
                        << ", IntersectStart: " << IntersectStart << "\n");

      // Does EndBit intersect an existing fragment?
      auto LastOverlap = B.find(AIt.stop());
      bool IntersectEnd =
          LastOverlap != B.end() && LastOverlap.start() < AIt.stop();
      LLVM_DEBUG(dbgs() << "- LastOverlap " << toString(LastOverlap, false)
                        << ", IntersectEnd: " << IntersectEnd << "\n");

      // Check if both ends of `a` intersect the same interval `b`.
      if (IntersectStart && IntersectEnd && FirstOverlap == LastOverlap) {
        // Insert `a` (`a` is contained in `b`) if the values match.
        // [ a ]
        // [ - b - ]
        // -
        // [ r ]
        LLVM_DEBUG(dbgs() << "- a is contained within "
                          << toString(FirstOverlap));
        if (AIt.value() && AIt.value() == FirstOverlap.value())
          Result.insert(AIt.start(), AIt.stop(), AIt.value());
      } else {
        // There's an overlap but `a` is not fully contained within
        // `b`. Shorten any end-point intersections.
        //     [ - a - ]
        // [ - b - ]
        // -
        //     [ r ]
        auto Next = FirstOverlap;
        if (IntersectStart) {
          LLVM_DEBUG(dbgs() << "- insert intersection of a and "
                            << toString(FirstOverlap));
          if (AIt.value() && AIt.value() == FirstOverlap.value())
            Result.insert(AIt.start(), FirstOverlap.stop(), AIt.value());
          ++Next;
        }
        // [ - a - ]
        //     [ - b - ]
        // -
        //     [ r ]
        if (IntersectEnd) {
          LLVM_DEBUG(dbgs() << "- insert intersection of a and "
                            << toString(LastOverlap));
          if (AIt.value() && AIt.value() == LastOverlap.value())
            Result.insert(LastOverlap.start(), AIt.stop(), AIt.value());
        }

        // Insert all intervals in map `B` that are contained within interval
        // `a` where the values match.
        // [ -  - a -  - ]
        // [ b1 ]   [ b2 ]
        // -
        // [ r1 ]   [ r2 ]
        while (Next != B.end() && Next.start() < AIt.stop() &&
               Next.stop() <= AIt.stop()) {
          LLVM_DEBUG(dbgs()
                     << "- insert intersection of a and " << toString(Next));
          if (AIt.value() && AIt.value() == Next.value())
            Result.insert(Next.start(), Next.stop(), Next.value());
          ++Next;
        }
      }
    }
    return Result;
  }

  /// Meet \p A and \p B, storing the result in \p A.
  void meetVars(VarFragMap &A, const VarFragMap &B) {
    // Meet A and B.
    //
    // Result = meet(a, b) for a in A, b in B where Var(a) == Var(b)
    for (auto It = A.begin(), End = A.end(); It != End; ++It) {
      unsigned AVar = It->first;
      FragsInMemMap &AFrags = It->second;
      auto BIt = B.find(AVar);
      if (BIt == B.end()) {
        A.erase(It);
        continue; // Var has no bits defined in B.
      }
      LLVM_DEBUG(dbgs() << "meet fragment maps for "
                        << Aggregates[AVar].first->getName() << "\n");
      AFrags = meetFragments(AFrags, BIt->second);
    }
  }

  bool meet(const BasicBlock &BB,
            const SmallPtrSet<BasicBlock *, 16> &Visited) {
    LLVM_DEBUG(dbgs() << "meet block info from preds of " << BB.getName()
                      << "\n");

    VarFragMap BBLiveIn;
    bool FirstMeet = true;
    // LiveIn locs for BB is the meet of the already-processed preds' LiveOut
    // locs.
    for (auto I = pred_begin(&BB), E = pred_end(&BB); I != E; I++) {
      // Ignore preds that haven't been processed yet. This is essentially the
      // same as initialising all variables to implicit top value (⊤) which is
      // the identity value for the meet operation.
      const BasicBlock *Pred = *I;
      if (!Visited.count(Pred))
        continue;

      auto PredLiveOut = LiveOut.find(Pred);
      assert(PredLiveOut != LiveOut.end());

      if (FirstMeet) {
        LLVM_DEBUG(dbgs() << "BBLiveIn = " << Pred->getName() << "\n");
        BBLiveIn = PredLiveOut->second;
        FirstMeet = false;
      } else {
        LLVM_DEBUG(dbgs() << "BBLiveIn = meet BBLiveIn, " << Pred->getName()
                          << "\n");
        meetVars(BBLiveIn, PredLiveOut->second);
      }

      // An empty set is ⊥ for the intersect-like meet operation. If we've
      // already got ⊥ there's no need to run the code - we know the result is
      // ⊥ since `meet(a, ⊥) = ⊥`.
      if (BBLiveIn.size() == 0)
        break;
    }

    auto CurrentLiveInEntry = LiveIn.find(&BB);
    // If there's no LiveIn entry for the block yet, add it.
    if (CurrentLiveInEntry == LiveIn.end()) {
      LLVM_DEBUG(dbgs() << "change=true (first) on meet on " << BB.getName()
                        << "\n");
      LiveIn[&BB] = std::move(BBLiveIn);
      return /*Changed=*/true;
    }

    // If the LiveIn set has changed (expensive check) update it and return
    // true.
    if (!varFragMapsAreEqual(BBLiveIn, CurrentLiveInEntry->second)) {
      LLVM_DEBUG(dbgs() << "change=true on meet on " << BB.getName() << "\n");
      CurrentLiveInEntry->second = std::move(BBLiveIn);
      return /*Changed=*/true;
    }

    LLVM_DEBUG(dbgs() << "change=false on meet on " << BB.getName() << "\n");
    return /*Changed=*/false;
  }

  void insertMemLoc(BasicBlock &BB, Instruction &Before, unsigned Var,
                    unsigned StartBit, unsigned EndBit, unsigned Base,
                    DebugLoc DL) {
    assert(StartBit < EndBit && "Cannot create fragment of size <= 0");
    if (!Base)
      return;
    FragMemLoc Loc;
    Loc.Var = Var;
    Loc.OffsetInBits = StartBit;
    Loc.SizeInBits = EndBit - StartBit;
    assert(Base && "Expected a non-zero ID for Base address");
    Loc.Base = Base;
    Loc.DL = DL;
    BBInsertBeforeMap[&BB][&Before].push_back(Loc);
    LLVM_DEBUG(dbgs() << "Add mem def for " << Aggregates[Var].first->getName()
                      << " bits [" << StartBit << ", " << EndBit << ")\n");
  }

  void addDef(const VarLocInfo &VarLoc, Instruction &Before, BasicBlock &BB,
              VarFragMap &LiveSet) {
    DebugVariable DbgVar = FnVarLocs->getVariable(VarLoc.VariableID);
    if (skipVariable(DbgVar.getVariable()))
      return;
    // Don't bother doing anything for this variables if we know it's fully
    // promoted. We're only interested in variables that (sometimes) live on
    // the stack here.
    if (!VarsWithStackSlot->count(getAggregate(DbgVar)))
      return;
    unsigned Var = Aggregates.insert(
        DebugAggregate(DbgVar.getVariable(), VarLoc.DL.getInlinedAt()));

    // [StartBit: EndBit) are the bits affected by this def.
    const DIExpression *DIExpr = VarLoc.Expr;
    unsigned StartBit;
    unsigned EndBit;
    if (auto Frag = DIExpr->getFragmentInfo()) {
      StartBit = Frag->OffsetInBits;
      EndBit = StartBit + Frag->SizeInBits;
    } else {
      assert(static_cast<bool>(DbgVar.getVariable()->getSizeInBits()));
      StartBit = 0;
      EndBit = *DbgVar.getVariable()->getSizeInBits();
    }

    // We will only fill fragments for simple memory-describing dbg.value
    // intrinsics. If the fragment offset is the same as the offset from the
    // base pointer, do The Thing, otherwise fall back to normal dbg.value
    // behaviour. AssignmentTrackingLowering has generated DIExpressions
    // written in terms of the base pointer.
    // TODO: Remove this condition since the fragment offset doesn't always
    // equal the offset from base pointer (e.g. for a SROA-split variable).
    const auto DerefOffsetInBytes = getDerefOffsetInBytes(DIExpr);
    const unsigned Base =
        DerefOffsetInBytes && *DerefOffsetInBytes * 8 == StartBit
            ? Bases.insert(VarLoc.V)
            : 0;
    LLVM_DEBUG(dbgs() << "DEF " << DbgVar.getVariable()->getName() << " ["
                      << StartBit << ", " << EndBit << "): " << toString(Base)
                      << "\n");

    // First of all, any locs that use mem that are disrupted need reinstating.
    // Unfortunately, IntervalMap doesn't let us insert intervals that overlap
    // with existing intervals so this code involves a lot of fiddling around
    // with intervals to do that manually.
    auto FragIt = LiveSet.find(Var);

    // Check if the variable does not exist in the map.
    if (FragIt == LiveSet.end()) {
      // Add this variable to the BB map.
      auto P = LiveSet.try_emplace(Var, FragsInMemMap(IntervalMapAlloc));
      assert(P.second && "Var already in map?");
      // Add the interval to the fragment map.
      P.first->second.insert(StartBit, EndBit, Base);
      return;
    }
    // The variable has an entry in the map.

    FragsInMemMap &FragMap = FragIt->second;
    // First check the easy case: the new fragment `f` doesn't overlap with any
    // intervals.
    if (!FragMap.overlaps(StartBit, EndBit)) {
      LLVM_DEBUG(dbgs() << "- No overlaps\n");
      FragMap.insert(StartBit, EndBit, Base);
      return;
    }
    // There is at least one overlap.

    // Does StartBit intersect an existing fragment?
    auto FirstOverlap = FragMap.find(StartBit);
    assert(FirstOverlap != FragMap.end());
    bool IntersectStart = FirstOverlap.start() < StartBit;

    // Does EndBit intersect an existing fragment?
    auto LastOverlap = FragMap.find(EndBit);
    bool IntersectEnd = LastOverlap.valid() && LastOverlap.start() < EndBit;

    // Check if both ends of `f` intersect the same interval `i`.
    if (IntersectStart && IntersectEnd && FirstOverlap == LastOverlap) {
      LLVM_DEBUG(dbgs() << "- Intersect single interval @ both ends\n");
      // Shorten `i` so that there's space to insert `f`.
      //      [ f ]
      // [  -   i   -  ]
      // +
      // [ i ][ f ][ i ]
      auto EndBitOfOverlap = FirstOverlap.stop();
      FirstOverlap.setStop(StartBit);
      insertMemLoc(BB, Before, Var, FirstOverlap.start(), StartBit,
                   FirstOverlap.value(), VarLoc.DL);

      // Insert a new interval to represent the end part.
      FragMap.insert(EndBit, EndBitOfOverlap, FirstOverlap.value());
      insertMemLoc(BB, Before, Var, EndBit, EndBitOfOverlap,
                   FirstOverlap.value(), VarLoc.DL);

      // Insert the new (middle) fragment now there is space.
      FragMap.insert(StartBit, EndBit, Base);
    } else {
      // There's an overlap but `f` may not be fully contained within
      // `i`. Shorten any end-point intersections so that we can then
      // insert `f`.
      //      [ - f - ]
      // [ - i - ]
      // |   |
      // [ i ]
      // Shorten any end-point intersections.
      if (IntersectStart) {
        LLVM_DEBUG(dbgs() << "- Intersect interval at start\n");
        // Split off at the intersection.
        FirstOverlap.setStop(StartBit);
        insertMemLoc(BB, Before, Var, FirstOverlap.start(), StartBit,
                     FirstOverlap.value(), VarLoc.DL);
      }
      // [ - f - ]
      //      [ - i - ]
      //          |   |
      //          [ i ]
      if (IntersectEnd) {
        LLVM_DEBUG(dbgs() << "- Intersect interval at end\n");
        // Split off at the intersection.
        LastOverlap.setStart(EndBit);
        insertMemLoc(BB, Before, Var, EndBit, LastOverlap.stop(),
                     LastOverlap.value(), VarLoc.DL);
      }

      LLVM_DEBUG(dbgs() << "- Erase intervals contained within\n");
      // FirstOverlap and LastOverlap have been shortened such that they're
      // no longer overlapping with [StartBit, EndBit). Delete any overlaps
      // that remain (these will be fully contained within `f`).
      // [ - f - ]       }
      //      [ - i - ]  } Intersection shortening that has happened above.
      //          |   |  }
      //          [ i ]  }
      // -----------------
      // [i2 ]           } Intervals fully contained within `f` get erased.
      // -----------------
      // [ - f - ][ i ]  } Completed insertion.
      auto It = FirstOverlap;
      if (IntersectStart)
        ++It; // IntersectStart: first overlap has been shortened.
      while (It.valid() && It.start() >= StartBit && It.stop() <= EndBit) {
        LLVM_DEBUG(dbgs() << "- Erase " << toString(It));
        It.erase(); // This increments It after removing the interval.
      }
      // We've dealt with all the overlaps now!
      assert(!FragMap.overlaps(StartBit, EndBit));
      LLVM_DEBUG(dbgs() << "- Insert DEF into now-empty space\n");
      FragMap.insert(StartBit, EndBit, Base);
    }
  }

  bool skipVariable(const DILocalVariable *V) { return !V->getSizeInBits(); }

  void process(BasicBlock &BB, VarFragMap &LiveSet) {
    BBInsertBeforeMap[&BB].clear();
    for (auto &I : BB) {
      if (const auto *Locs = FnVarLocs->getWedge(&I)) {
        for (const VarLocInfo &Loc : *Locs) {
          addDef(Loc, I, *I.getParent(), LiveSet);
        }
      }
    }
  }

public:
  MemLocFragmentFill(Function &Fn,
                     const DenseSet<DebugAggregate> *VarsWithStackSlot)
      : Fn(Fn), VarsWithStackSlot(VarsWithStackSlot) {}

  /// Add variable locations to \p FnVarLocs so that any bits of a variable
  /// with a memory location have that location explicitly reinstated at each
  /// subsequent variable location definition that that doesn't overwrite those
  /// bits. i.e. after a variable location def, insert new defs for the memory
  /// location with fragments for the difference of "all bits currently in
  /// memory" and "the fragment of the second def". e.g.
  ///
  ///     Before:
  ///
  ///     var x bits 0 to 63:  value in memory
  ///     more instructions
  ///     var x bits 0 to 31:  value is %0
  ///
  ///     After:
  ///
  ///     var x bits 0 to 63:  value in memory
  ///     more instructions
  ///     var x bits 0 to 31:  value is %0
  ///     var x bits 32 to 61: value in memory ; <-- new loc def
  ///
  void run(FunctionVarLocsBuilder *FnVarLocs) {
    if (!EnableMemLocFragFill)
      return;

    this->FnVarLocs = FnVarLocs;

    // Prepare for traversal.
    //
    ReversePostOrderTraversal<Function *> RPOT(&Fn);
    std::priority_queue<unsigned int, std::vector<unsigned int>,
                        std::greater<unsigned int>>
        Worklist;
    std::priority_queue<unsigned int, std::vector<unsigned int>,
                        std::greater<unsigned int>>
        Pending;
    DenseMap<unsigned int, BasicBlock *> OrderToBB;
    DenseMap<BasicBlock *, unsigned int> BBToOrder;
    { // Init OrderToBB and BBToOrder.
      unsigned int RPONumber = 0;
      for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
        OrderToBB[RPONumber] = *RI;
        BBToOrder[*RI] = RPONumber;
        Worklist.push(RPONumber);
        ++RPONumber;
      }
      LiveIn.init(RPONumber);
      LiveOut.init(RPONumber);
    }

    // Perform the traversal.
    //
    // This is a standard "intersect of predecessor outs" dataflow problem. To
    // solve it, we perform meet() and process() using the two worklist method
    // until the LiveIn data for each block becomes unchanging.
    //
    // This dataflow is essentially working on maps of sets and at each meet we
    // intersect the maps and the mapped sets. So, initialized live-in maps
    // monotonically decrease in value throughout the dataflow.
    SmallPtrSet<BasicBlock *, 16> Visited;
    while (!Worklist.empty() || !Pending.empty()) {
      // We track what is on the pending worklist to avoid inserting the same
      // thing twice.  We could avoid this with a custom priority queue, but
      // this is probably not worth it.
      SmallPtrSet<BasicBlock *, 16> OnPending;
      LLVM_DEBUG(dbgs() << "Processing Worklist\n");
      while (!Worklist.empty()) {
        BasicBlock *BB = OrderToBB[Worklist.top()];
        LLVM_DEBUG(dbgs() << "\nPop BB " << BB->getName() << "\n");
        Worklist.pop();
        bool InChanged = meet(*BB, Visited);
        // Always consider LiveIn changed on the first visit.
        InChanged |= Visited.insert(BB).second;
        if (InChanged) {
          LLVM_DEBUG(dbgs()
                     << BB->getName() << " has new InLocs, process it\n");
          //  Mutate a copy of LiveIn while processing BB. Once we've processed
          //  the terminator LiveSet is the LiveOut set for BB.
          //  This is an expensive copy!
          VarFragMap LiveSet = LiveIn[BB];

          // Process the instructions in the block.
          process(*BB, LiveSet);

          // Relatively expensive check: has anything changed in LiveOut for BB?
          if (!varFragMapsAreEqual(LiveOut[BB], LiveSet)) {
            LLVM_DEBUG(dbgs() << BB->getName()
                              << " has new OutLocs, add succs to worklist: [ ");
            LiveOut[BB] = std::move(LiveSet);
            for (auto I = succ_begin(BB), E = succ_end(BB); I != E; I++) {
              if (OnPending.insert(*I).second) {
                LLVM_DEBUG(dbgs() << I->getName() << " ");
                Pending.push(BBToOrder[*I]);
              }
            }
            LLVM_DEBUG(dbgs() << "]\n");
          }
        }
      }
      Worklist.swap(Pending);
      // At this point, pending must be empty, since it was just the empty
      // worklist
      assert(Pending.empty() && "Pending should be empty");
    }

    // Insert new location defs.
    for (auto Pair : BBInsertBeforeMap) {
      InsertMap &Map = Pair.second;
      for (auto Pair : Map) {
        Instruction *InsertBefore = Pair.first;
        assert(InsertBefore && "should never be null");
        auto FragMemLocs = Pair.second;
        auto &Ctx = Fn.getContext();

        for (auto FragMemLoc : FragMemLocs) {
          DIExpression *Expr = DIExpression::get(Ctx, std::nullopt);
          Expr = *DIExpression::createFragmentExpression(
              Expr, FragMemLoc.OffsetInBits, FragMemLoc.SizeInBits);
          Expr = DIExpression::prepend(Expr, DIExpression::DerefAfter,
                                       FragMemLoc.OffsetInBits / 8);
          DebugVariable Var(Aggregates[FragMemLoc.Var].first, Expr,
                            FragMemLoc.DL.getInlinedAt());
          FnVarLocs->addVarLoc(InsertBefore, Var, Expr, FragMemLoc.DL,
                               Bases[FragMemLoc.Base]);
        }
      }
    }
  }
};

/// AssignmentTrackingLowering encapsulates a dataflow analysis over a function
/// that interprets assignment tracking debug info metadata and stores in IR to
/// create a map of variable locations.
class AssignmentTrackingLowering {
public:
  /// The kind of location in use for a variable, where Mem is the stack home,
  /// Val is an SSA value or const, and None means that there is not one single
  /// kind (either because there are multiple or because there is none; it may
  /// prove useful to split this into two values in the future).
  ///
  /// LocKind is a join-semilattice with the partial order:
  /// None > Mem, Val
  ///
  /// i.e.
  /// join(Mem, Mem)   = Mem
  /// join(Val, Val)   = Val
  /// join(Mem, Val)   = None
  /// join(None, Mem)  = None
  /// join(None, Val)  = None
  /// join(None, None) = None
  ///
  /// Note: the order is not `None > Val > Mem` because we're using DIAssignID
  /// to name assignments and are not tracking the actual stored values.
  /// Therefore currently there's no way to ensure that Mem values and Val
  /// values are the same. This could be a future extension, though it's not
  /// clear that many additional locations would be recovered that way in
  /// practice as the likelihood of this sitation arising naturally seems
  /// incredibly low.
  enum class LocKind { Mem, Val, None };

  /// An abstraction of the assignment of a value to a variable or memory
  /// location.
  ///
  /// An Assignment is Known or NoneOrPhi. A Known Assignment means we have a
  /// DIAssignID ptr that represents it. NoneOrPhi means that we don't (or
  /// can't) know the ID of the last assignment that took place.
  ///
  /// The Status of the Assignment (Known or NoneOrPhi) is another
  /// join-semilattice. The partial order is:
  /// NoneOrPhi > Known {id_0, id_1, ...id_N}
  ///
  /// i.e. for all values x and y where x != y:
  /// join(x, x) = x
  /// join(x, y) = NoneOrPhi
  struct Assignment {
    enum S { Known, NoneOrPhi } Status;
    /// ID of the assignment. nullptr if Status is not Known.
    DIAssignID *ID;
    /// The dbg.assign that marks this dbg-def. Mem-defs don't use this field.
    /// May be nullptr.
    DbgAssignIntrinsic *Source;

    bool isSameSourceAssignment(const Assignment &Other) const {
      // Don't include Source in the equality check. Assignments are
      // defined by their ID, not debug intrinsic(s).
      return std::tie(Status, ID) == std::tie(Other.Status, Other.ID);
    }
    void dump(raw_ostream &OS) {
      static const char *LUT[] = {"Known", "NoneOrPhi"};
      OS << LUT[Status] << "(id=";
      if (ID)
        OS << ID;
      else
        OS << "null";
      OS << ", s=";
      if (Source)
        OS << *Source;
      else
        OS << "null";
      OS << ")";
    }

    static Assignment make(DIAssignID *ID, DbgAssignIntrinsic *Source) {
      return Assignment(Known, ID, Source);
    }
    static Assignment makeFromMemDef(DIAssignID *ID) {
      return Assignment(Known, ID, nullptr);
    }
    static Assignment makeNoneOrPhi() {
      return Assignment(NoneOrPhi, nullptr, nullptr);
    }
    // Again, need a Top value?
    Assignment()
        : Status(NoneOrPhi), ID(nullptr), Source(nullptr) {
    } // Can we delete this?
    Assignment(S Status, DIAssignID *ID, DbgAssignIntrinsic *Source)
        : Status(Status), ID(ID), Source(Source) {
      // If the Status is Known then we expect there to be an assignment ID.
      assert(Status == NoneOrPhi || ID);
    }
  };

  using AssignmentMap = DenseMap<VariableID, Assignment>;
  using LocMap = DenseMap<VariableID, LocKind>;
  using OverlapMap = DenseMap<VariableID, SmallVector<VariableID, 4>>;
  using UntaggedStoreAssignmentMap =
      DenseMap<const Instruction *,
               SmallVector<std::pair<VariableID, at::AssignmentInfo>>>;

private:
  /// Map a variable to the set of variables that it fully contains.
  OverlapMap VarContains;
  /// Map untagged stores to the variable fragments they assign to. Used by
  /// processUntaggedInstruction.
  UntaggedStoreAssignmentMap UntaggedStoreVars;

  // Machinery to defer inserting dbg.values.
  using InsertMap = MapVector<Instruction *, SmallVector<VarLocInfo>>;
  InsertMap InsertBeforeMap;
  /// Clear the location definitions currently cached for insertion after /p
  /// After.
  void resetInsertionPoint(Instruction &After);
  void emitDbgValue(LocKind Kind, const DbgVariableIntrinsic *Source,
                    Instruction *After);

  static bool mapsAreEqual(const AssignmentMap &A, const AssignmentMap &B) {
    if (A.size() != B.size())
      return false;
    for (const auto &Pair : A) {
      VariableID Var = Pair.first;
      const Assignment &AV = Pair.second;
      auto R = B.find(Var);
      // Check if this entry exists in B, otherwise ret false.
      if (R == B.end())
        return false;
      // Check that the assignment value is the same.
      if (!AV.isSameSourceAssignment(R->second))
        return false;
    }
    return true;
  }

  /// Represents the stack and debug assignments in a block. Used to describe
  /// the live-in and live-out values for blocks, as well as the "current"
  /// value as we process each instruction in a block.
  struct BlockInfo {
    /// Dominating assignment to memory for each variable.
    AssignmentMap StackHomeValue;
    /// Dominating assignemnt to each variable.
    AssignmentMap DebugValue;
    /// Location kind for each variable. LiveLoc indicates whether the
    /// dominating assignment in StackHomeValue (LocKind::Mem), DebugValue
    /// (LocKind::Val), or neither (LocKind::None) is valid, in that order of
    /// preference. This cannot be derived by inspecting DebugValue and
    /// StackHomeValue due to the fact that there's no distinction in
    /// Assignment (the class) between whether an assignment is unknown or a
    /// merge of multiple assignments (both are Status::NoneOrPhi). In other
    /// words, the memory location may well be valid while both DebugValue and
    /// StackHomeValue contain Assignments that have a Status of NoneOrPhi.
    LocMap LiveLoc;

    /// Compare every element in each map to determine structural equality
    /// (slow).
    bool operator==(const BlockInfo &Other) const {
      return LiveLoc == Other.LiveLoc &&
             mapsAreEqual(StackHomeValue, Other.StackHomeValue) &&
             mapsAreEqual(DebugValue, Other.DebugValue);
    }
    bool operator!=(const BlockInfo &Other) const { return !(*this == Other); }
    bool isValid() {
      return LiveLoc.size() == DebugValue.size() &&
             LiveLoc.size() == StackHomeValue.size();
    }
  };

  Function &Fn;
  const DataLayout &Layout;
  const DenseSet<DebugAggregate> *VarsWithStackSlot;
  FunctionVarLocsBuilder *FnVarLocs;
  DenseMap<const BasicBlock *, BlockInfo> LiveIn;
  DenseMap<const BasicBlock *, BlockInfo> LiveOut;

  /// Helper for process methods to track variables touched each frame.
  DenseSet<VariableID> VarsTouchedThisFrame;

  /// The set of variables that sometimes are not located in their stack home.
  DenseSet<DebugAggregate> NotAlwaysStackHomed;

  VariableID getVariableID(const DebugVariable &Var) {
    return static_cast<VariableID>(FnVarLocs->insertVariable(Var));
  }

  /// Join the LiveOut values of preds that are contained in \p Visited into
  /// LiveIn[BB]. Return True if LiveIn[BB] has changed as a result. LiveIn[BB]
  /// values monotonically increase. See the @link joinMethods join methods
  /// @endlink documentation for more info.
  bool join(const BasicBlock &BB, const SmallPtrSet<BasicBlock *, 16> &Visited);
  ///@name joinMethods
  /// Functions that implement `join` (the least upper bound) for the
  /// join-semilattice types used in the dataflow. There is an explicit bottom
  /// value (⊥) for some types and and explicit top value (⊤) for all types.
  /// By definition:
  ///
  ///     Join(A, B) >= A && Join(A, B) >= B
  ///     Join(A, ⊥) = A
  ///     Join(A, ⊤) = ⊤
  ///
  /// These invariants are important for monotonicity.
  ///
  /// For the map-type functions, all unmapped keys in an empty map are
  /// associated with a bottom value (⊥). This represents their values being
  /// unknown. Unmapped keys in non-empty maps (joining two maps with a key
  /// only present in one) represents either a variable going out of scope or
  /// dropped debug info. It is assumed the key is associated with a top value
  /// (⊤) in this case (unknown location / assignment).
  ///@{
  static LocKind joinKind(LocKind A, LocKind B);
  static LocMap joinLocMap(const LocMap &A, const LocMap &B);
  static Assignment joinAssignment(const Assignment &A, const Assignment &B);
  static AssignmentMap joinAssignmentMap(const AssignmentMap &A,
                                         const AssignmentMap &B);
  static BlockInfo joinBlockInfo(const BlockInfo &A, const BlockInfo &B);
  ///@}

  /// Process the instructions in \p BB updating \p LiveSet along the way. \p
  /// LiveSet must be initialized with the current live-in locations before
  /// calling this.
  void process(BasicBlock &BB, BlockInfo *LiveSet);
  ///@name processMethods
  /// Methods to process instructions in order to update the LiveSet (current
  /// location information).
  ///@{
  void processNonDbgInstruction(Instruction &I, BlockInfo *LiveSet);
  void processDbgInstruction(Instruction &I, BlockInfo *LiveSet);
  /// Update \p LiveSet after encountering an instruction with a DIAssignID
  /// attachment, \p I.
  void processTaggedInstruction(Instruction &I, BlockInfo *LiveSet);
  /// Update \p LiveSet after encountering an instruciton without a DIAssignID
  /// attachment, \p I.
  void processUntaggedInstruction(Instruction &I, BlockInfo *LiveSet);
  void processDbgAssign(DbgAssignIntrinsic &DAI, BlockInfo *LiveSet);
  void processDbgValue(DbgValueInst &DVI, BlockInfo *LiveSet);
  /// Add an assignment to memory for the variable /p Var.
  void addMemDef(BlockInfo *LiveSet, VariableID Var, const Assignment &AV);
  /// Add an assignment to the variable /p Var.
  void addDbgDef(BlockInfo *LiveSet, VariableID Var, const Assignment &AV);
  ///@}

  /// Set the LocKind for \p Var.
  void setLocKind(BlockInfo *LiveSet, VariableID Var, LocKind K);
  /// Get the live LocKind for a \p Var. Requires addMemDef or addDbgDef to
  /// have been called for \p Var first.
  LocKind getLocKind(BlockInfo *LiveSet, VariableID Var);
  /// Return true if \p Var has an assignment in \p M matching \p AV.
  bool hasVarWithAssignment(VariableID Var, const Assignment &AV,
                            const AssignmentMap &M);

  /// Emit info for variables that are fully promoted.
  bool emitPromotedVarLocs(FunctionVarLocsBuilder *FnVarLocs);

public:
  AssignmentTrackingLowering(Function &Fn, const DataLayout &Layout,
                             const DenseSet<DebugAggregate> *VarsWithStackSlot)
      : Fn(Fn), Layout(Layout), VarsWithStackSlot(VarsWithStackSlot) {}
  /// Run the analysis, adding variable location info to \p FnVarLocs. Returns
  /// true if any variable locations have been added to FnVarLocs.
  bool run(FunctionVarLocsBuilder *FnVarLocs);
};

void AssignmentTrackingLowering::setLocKind(BlockInfo *LiveSet, VariableID Var,
                                            LocKind K) {
  auto SetKind = [this](BlockInfo *LiveSet, VariableID Var, LocKind K) {
    VarsTouchedThisFrame.insert(Var);
    LiveSet->LiveLoc[Var] = K;
  };
  SetKind(LiveSet, Var, K);

  // Update the LocKind for all fragments contained within Var.
  for (VariableID Frag : VarContains[Var])
    SetKind(LiveSet, Frag, K);
}

AssignmentTrackingLowering::LocKind
AssignmentTrackingLowering::getLocKind(BlockInfo *LiveSet, VariableID Var) {
  auto Pair = LiveSet->LiveLoc.find(Var);
  assert(Pair != LiveSet->LiveLoc.end());
  return Pair->second;
}

void AssignmentTrackingLowering::addMemDef(BlockInfo *LiveSet, VariableID Var,
                                           const Assignment &AV) {
  auto AddDef = [](BlockInfo *LiveSet, VariableID Var, Assignment AV) {
    LiveSet->StackHomeValue[Var] = AV;
    // Add default (Var -> ⊤) to DebugValue if Var isn't in DebugValue yet.
    LiveSet->DebugValue.insert({Var, Assignment::makeNoneOrPhi()});
    // Add default (Var -> ⊤) to LiveLocs if Var isn't in LiveLocs yet. Callers
    // of addMemDef will call setLocKind to override.
    LiveSet->LiveLoc.insert({Var, LocKind::None});
  };
  AddDef(LiveSet, Var, AV);

  // Use this assigment for all fragments contained within Var, but do not
  // provide a Source because we cannot convert Var's value to a value for the
  // fragment.
  Assignment FragAV = AV;
  FragAV.Source = nullptr;
  for (VariableID Frag : VarContains[Var])
    AddDef(LiveSet, Frag, FragAV);
}

void AssignmentTrackingLowering::addDbgDef(BlockInfo *LiveSet, VariableID Var,
                                           const Assignment &AV) {
  auto AddDef = [](BlockInfo *LiveSet, VariableID Var, Assignment AV) {
    LiveSet->DebugValue[Var] = AV;
    // Add default (Var -> ⊤) to StackHome if Var isn't in StackHome yet.
    LiveSet->StackHomeValue.insert({Var, Assignment::makeNoneOrPhi()});
    // Add default (Var -> ⊤) to LiveLocs if Var isn't in LiveLocs yet. Callers
    // of addDbgDef will call setLocKind to override.
    LiveSet->LiveLoc.insert({Var, LocKind::None});
  };
  AddDef(LiveSet, Var, AV);

  // Use this assigment for all fragments contained within Var, but do not
  // provide a Source because we cannot convert Var's value to a value for the
  // fragment.
  Assignment FragAV = AV;
  FragAV.Source = nullptr;
  for (VariableID Frag : VarContains[Var])
    AddDef(LiveSet, Frag, FragAV);
}

static DIAssignID *getIDFromInst(const Instruction &I) {
  return cast<DIAssignID>(I.getMetadata(LLVMContext::MD_DIAssignID));
}

static DIAssignID *getIDFromMarker(const DbgAssignIntrinsic &DAI) {
  return cast<DIAssignID>(DAI.getAssignID());
}

/// Return true if \p Var has an assignment in \p M matching \p AV.
bool AssignmentTrackingLowering::hasVarWithAssignment(VariableID Var,
                                                      const Assignment &AV,
                                                      const AssignmentMap &M) {
  auto AssignmentIsMapped = [](VariableID Var, const Assignment &AV,
                               const AssignmentMap &M) {
    auto R = M.find(Var);
    if (R == M.end())
      return false;
    return AV.isSameSourceAssignment(R->second);
  };

  if (!AssignmentIsMapped(Var, AV, M))
    return false;

  // Check all the frags contained within Var as these will have all been
  // mapped to AV at the last store to Var.
  for (VariableID Frag : VarContains[Var])
    if (!AssignmentIsMapped(Frag, AV, M))
      return false;
  return true;
}

const char *locStr(AssignmentTrackingLowering::LocKind Loc) {
  using LocKind = AssignmentTrackingLowering::LocKind;
  switch (Loc) {
  case LocKind::Val:
    return "Val";
  case LocKind::Mem:
    return "Mem";
  case LocKind::None:
    return "None";
  };
  llvm_unreachable("unknown LocKind");
}

void AssignmentTrackingLowering::emitDbgValue(
    AssignmentTrackingLowering::LocKind Kind,
    const DbgVariableIntrinsic *Source, Instruction *After) {

  DILocation *DL = Source->getDebugLoc();
  auto Emit = [this, Source, After, DL](Value *Val, DIExpression *Expr) {
    assert(Expr);
    // It's possible that getVariableLocationOp(0) is null. Occurs in
    // llvm/test/DebugInfo/Generic/2010-05-03-OriginDIE.ll Treat it as undef.
    if (!Val)
      Val = UndefValue::get(Type::getInt1Ty(Source->getContext()));

    // Find a suitable insert point.
    Instruction *InsertBefore = After->getNextNode();
    assert(InsertBefore && "Shouldn't be inserting after a terminator");

    VariableID Var = getVariableID(DebugVariable(Source));
    VarLocInfo VarLoc;
    VarLoc.VariableID = static_cast<VariableID>(Var);
    VarLoc.Expr = Expr;
    VarLoc.V = Val;
    VarLoc.DL = DL;
    // Insert it into the map for later.
    InsertBeforeMap[InsertBefore].push_back(VarLoc);
  };

  // NOTE: This block can mutate Kind.
  if (Kind == LocKind::Mem) {
    const auto *DAI = cast<DbgAssignIntrinsic>(Source);
    // Check the address hasn't been dropped (e.g. the debug uses may not have
    // been replaced before deleting a Value).
    if (Value *Val = DAI->getAddress()) {
      DIExpression *Expr = DAI->getAddressExpression();
      assert(!Expr->getFragmentInfo() &&
             "fragment info should be stored in value-expression only");
      // Copy the fragment info over from the value-expression to the new
      // DIExpression.
      if (auto OptFragInfo = Source->getExpression()->getFragmentInfo()) {
        auto FragInfo = OptFragInfo.value();
        Expr = *DIExpression::createFragmentExpression(
            Expr, FragInfo.OffsetInBits, FragInfo.SizeInBits);
      }
      // The address-expression has an implicit deref, add it now.
      std::tie(Val, Expr) =
          walkToAllocaAndPrependOffsetDeref(Layout, Val, Expr);
      Emit(Val, Expr);
      return;
    } else {
      // The address isn't valid so treat this as a non-memory def.
      Kind = LocKind::Val;
    }
  }

  if (Kind == LocKind::Val) {
    /// Get the value component, converting to Undef if it is variadic.
    Value *Val =
        Source->hasArgList()
            ? UndefValue::get(Source->getVariableLocationOp(0)->getType())
            : Source->getVariableLocationOp(0);
    Emit(Val, Source->getExpression());
    return;
  }

  if (Kind == LocKind::None) {
    Value *Val = UndefValue::get(Source->getVariableLocationOp(0)->getType());
    Emit(Val, Source->getExpression());
    return;
  }
}

void AssignmentTrackingLowering::processNonDbgInstruction(
    Instruction &I, AssignmentTrackingLowering::BlockInfo *LiveSet) {
  if (I.hasMetadata(LLVMContext::MD_DIAssignID))
    processTaggedInstruction(I, LiveSet);
  else
    processUntaggedInstruction(I, LiveSet);
}

void AssignmentTrackingLowering::processUntaggedInstruction(
    Instruction &I, AssignmentTrackingLowering::BlockInfo *LiveSet) {
  // Interpret stack stores that are not tagged as an assignment in memory for
  // the variables associated with that address. These stores may not be tagged
  // because a) the store cannot be represented using dbg.assigns (non-const
  // length or offset) or b) the tag was accidentally dropped during
  // optimisations. For these stores we fall back to assuming that the stack
  // home is a valid location for the variables. The benefit is that this
  // prevents us missing an assignment and therefore incorrectly maintaining
  // earlier location definitions, and in many cases it should be a reasonable
  // assumption. However, this will occasionally lead to slight
  // inaccuracies. The value of a hoisted untagged store will be visible
  // "early", for example.
  assert(!I.hasMetadata(LLVMContext::MD_DIAssignID));
  auto It = UntaggedStoreVars.find(&I);
  if (It == UntaggedStoreVars.end())
    return; // No variables associated with the store destination.

  LLVM_DEBUG(dbgs() << "processUntaggedInstruction on UNTAGGED INST " << I
                    << "\n");
  // Iterate over the variables that this store affects, add a NoneOrPhi dbg
  // and mem def, set lockind to Mem, and emit a location def for each.
  for (auto [Var, Info] : It->second) {
    // This instruction is treated as both a debug and memory assignment,
    // meaning the memory location should be used. We don't have an assignment
    // ID though so use Assignment::makeNoneOrPhi() to create an imaginary one.
    addMemDef(LiveSet, Var, Assignment::makeNoneOrPhi());
    addDbgDef(LiveSet, Var, Assignment::makeNoneOrPhi());
    setLocKind(LiveSet, Var, LocKind::Mem);
    LLVM_DEBUG(dbgs() << "  setting Stack LocKind to: " << locStr(LocKind::Mem)
                      << "\n");
    // Build the dbg location def to insert.
    //
    // DIExpression: Add fragment and offset.
    DebugVariable V = FnVarLocs->getVariable(Var);
    DIExpression *DIE = DIExpression::get(I.getContext(), std::nullopt);
    if (auto Frag = V.getFragment()) {
      auto R = DIExpression::createFragmentExpression(DIE, Frag->OffsetInBits,
                                                      Frag->SizeInBits);
      assert(R && "unexpected createFragmentExpression failure");
      DIE = R.value();
    }
    SmallVector<uint64_t, 3> Ops;
    if (Info.OffsetInBits)
      Ops = {dwarf::DW_OP_plus_uconst, Info.OffsetInBits / 8};
    Ops.push_back(dwarf::DW_OP_deref);
    DIE = DIExpression::prependOpcodes(DIE, Ops, /*StackValue=*/false,
                                       /*EntryValue=*/false);
    // Find a suitable insert point.
    Instruction *InsertBefore = I.getNextNode();
    assert(InsertBefore && "Shouldn't be inserting after a terminator");

    // Get DILocation for this unrecorded assignment.
    DILocation *InlinedAt = const_cast<DILocation *>(V.getInlinedAt());
    const DILocation *DILoc = DILocation::get(
        Fn.getContext(), 0, 0, V.getVariable()->getScope(), InlinedAt);

    VarLocInfo VarLoc;
    VarLoc.VariableID = static_cast<VariableID>(Var);
    VarLoc.Expr = DIE;
    VarLoc.V = const_cast<AllocaInst *>(Info.Base);
    VarLoc.DL = DILoc;
    // 3. Insert it into the map for later.
    InsertBeforeMap[InsertBefore].push_back(VarLoc);
  }
}

void AssignmentTrackingLowering::processTaggedInstruction(
    Instruction &I, AssignmentTrackingLowering::BlockInfo *LiveSet) {
  auto Linked = at::getAssignmentMarkers(&I);
  // No dbg.assign intrinsics linked.
  // FIXME: All vars that have a stack slot this store modifies that don't have
  // a dbg.assign linked to it should probably treat this like an untagged
  // store.
  if (Linked.empty())
    return;

  LLVM_DEBUG(dbgs() << "processTaggedInstruction on " << I << "\n");
  for (DbgAssignIntrinsic *DAI : Linked) {
    VariableID Var = getVariableID(DebugVariable(DAI));
    // Something has gone wrong if VarsWithStackSlot doesn't contain a variable
    // that is linked to a store.
    assert(VarsWithStackSlot->count(getAggregate(DAI)) &&
           "expected DAI's variable to have stack slot");

    Assignment AV = Assignment::makeFromMemDef(getIDFromInst(I));
    addMemDef(LiveSet, Var, AV);

    LLVM_DEBUG(dbgs() << "   linked to " << *DAI << "\n");
    LLVM_DEBUG(dbgs() << "   LiveLoc " << locStr(getLocKind(LiveSet, Var))
                      << " -> ");

    // The last assignment to the stack is now AV. Check if the last debug
    // assignment has a matching Assignment.
    if (hasVarWithAssignment(Var, AV, LiveSet->DebugValue)) {
      // The StackHomeValue and DebugValue for this variable match so we can
      // emit a stack home location here.
      LLVM_DEBUG(dbgs() << "Mem, Stack matches Debug program\n";);
      LLVM_DEBUG(dbgs() << "   Stack val: "; AV.dump(dbgs()); dbgs() << "\n");
      LLVM_DEBUG(dbgs() << "   Debug val: ";
                 LiveSet->DebugValue[Var].dump(dbgs()); dbgs() << "\n");
      setLocKind(LiveSet, Var, LocKind::Mem);
      emitDbgValue(LocKind::Mem, DAI, &I);
      continue;
    }

    // The StackHomeValue and DebugValue for this variable do not match. I.e.
    // The value currently stored in the stack is not what we'd expect to
    // see, so we cannot use emit a stack home location here. Now we will
    // look at the live LocKind for the variable and determine an appropriate
    // dbg.value to emit.
    LocKind PrevLoc = getLocKind(LiveSet, Var);
    switch (PrevLoc) {
    case LocKind::Val: {
      // The value in memory in memory has changed but we're not currently
      // using the memory location. Do nothing.
      LLVM_DEBUG(dbgs() << "Val, (unchanged)\n";);
      setLocKind(LiveSet, Var, LocKind::Val);
    } break;
    case LocKind::Mem: {
      // There's been an assignment to memory that we were using as a
      // location for this variable, and the Assignment doesn't match what
      // we'd expect to see in memory.
      if (LiveSet->DebugValue[Var].Status == Assignment::NoneOrPhi) {
        // We need to terminate any previously open location now.
        LLVM_DEBUG(dbgs() << "None, No Debug value available\n";);
        setLocKind(LiveSet, Var, LocKind::None);
        emitDbgValue(LocKind::None, DAI, &I);
      } else {
        // The previous DebugValue Value can be used here.
        LLVM_DEBUG(dbgs() << "Val, Debug value is Known\n";);
        setLocKind(LiveSet, Var, LocKind::Val);
        Assignment PrevAV = LiveSet->DebugValue.lookup(Var);
        if (PrevAV.Source) {
          emitDbgValue(LocKind::Val, PrevAV.Source, &I);
        } else {
          // PrevAV.Source is nullptr so we must emit undef here.
          emitDbgValue(LocKind::None, DAI, &I);
        }
      }
    } break;
    case LocKind::None: {
      // There's been an assignment to memory and we currently are
      // not tracking a location for the variable. Do not emit anything.
      LLVM_DEBUG(dbgs() << "None, (unchanged)\n";);
      setLocKind(LiveSet, Var, LocKind::None);
    } break;
    }
  }
}

void AssignmentTrackingLowering::processDbgAssign(DbgAssignIntrinsic &DAI,
                                                  BlockInfo *LiveSet) {
  // Only bother tracking variables that are at some point stack homed. Other
  // variables can be dealt with trivially later.
  if (!VarsWithStackSlot->count(getAggregate(&DAI)))
    return;

  VariableID Var = getVariableID(DebugVariable(&DAI));
  Assignment AV = Assignment::make(getIDFromMarker(DAI), &DAI);
  addDbgDef(LiveSet, Var, AV);

  LLVM_DEBUG(dbgs() << "processDbgAssign on " << DAI << "\n";);
  LLVM_DEBUG(dbgs() << "   LiveLoc " << locStr(getLocKind(LiveSet, Var))
                    << " -> ");

  // Check if the DebugValue and StackHomeValue both hold the same
  // Assignment.
  if (hasVarWithAssignment(Var, AV, LiveSet->StackHomeValue)) {
    // They match. We can use the stack home because the debug intrinsics state
    // that an assignment happened here, and we know that specific assignment
    // was the last one to take place in memory for this variable.
    LocKind Kind;
    if (isa<UndefValue>(DAI.getAddress())) {
      // Address may be undef to indicate that although the store does take
      // place, this part of the original store has been elided.
      LLVM_DEBUG(
          dbgs() << "Val, Stack matches Debug program but address is undef\n";);
      Kind = LocKind::Val;
    } else {
      LLVM_DEBUG(dbgs() << "Mem, Stack matches Debug program\n";);
      Kind = LocKind::Mem;
    };
    setLocKind(LiveSet, Var, Kind);
    emitDbgValue(Kind, &DAI, &DAI);
  } else {
    // The last assignment to the memory location isn't the one that we want to
    // show to the user so emit a dbg.value(Value). Value may be undef.
    LLVM_DEBUG(dbgs() << "Val, Stack contents is unknown\n";);
    setLocKind(LiveSet, Var, LocKind::Val);
    emitDbgValue(LocKind::Val, &DAI, &DAI);
  }
}

void AssignmentTrackingLowering::processDbgValue(DbgValueInst &DVI,
                                                 BlockInfo *LiveSet) {
  // Only other tracking variables that are at some point stack homed.
  // Other variables can be dealt with trivally later.
  if (!VarsWithStackSlot->count(getAggregate(&DVI)))
    return;

  VariableID Var = getVariableID(DebugVariable(&DVI));
  // We have no ID to create an Assignment with so we mark this assignment as
  // NoneOrPhi. Note that the dbg.value still exists, we just cannot determine
  // the assignment responsible for setting this value.
  // This is fine; dbg.values are essentially interchangable with unlinked
  // dbg.assigns, and some passes such as mem2reg and instcombine add them to
  // PHIs for promoted variables.
  Assignment AV = Assignment::makeNoneOrPhi();
  addDbgDef(LiveSet, Var, AV);

  LLVM_DEBUG(dbgs() << "processDbgValue on " << DVI << "\n";);
  LLVM_DEBUG(dbgs() << "   LiveLoc " << locStr(getLocKind(LiveSet, Var))
                    << " -> Val, dbg.value override");

  setLocKind(LiveSet, Var, LocKind::Val);
  emitDbgValue(LocKind::Val, &DVI, &DVI);
}

void AssignmentTrackingLowering::processDbgInstruction(
    Instruction &I, AssignmentTrackingLowering::BlockInfo *LiveSet) {
  assert(!isa<DbgAddrIntrinsic>(&I) && "unexpected dbg.addr");
  if (auto *DAI = dyn_cast<DbgAssignIntrinsic>(&I))
    processDbgAssign(*DAI, LiveSet);
  else if (auto *DVI = dyn_cast<DbgValueInst>(&I))
    processDbgValue(*DVI, LiveSet);
}

void AssignmentTrackingLowering::resetInsertionPoint(Instruction &After) {
  assert(!After.isTerminator() && "Can't insert after a terminator");
  auto R = InsertBeforeMap.find(After.getNextNode());
  if (R == InsertBeforeMap.end())
    return;
  R->second.clear();
}

void AssignmentTrackingLowering::process(BasicBlock &BB, BlockInfo *LiveSet) {
  for (auto II = BB.begin(), EI = BB.end(); II != EI;) {
    assert(VarsTouchedThisFrame.empty());
    // Process the instructions in "frames". A "frame" includes a single
    // non-debug instruction followed any debug instructions before the
    // next non-debug instruction.
    if (!isa<DbgInfoIntrinsic>(&*II)) {
      if (II->isTerminator())
        break;
      resetInsertionPoint(*II);
      processNonDbgInstruction(*II, LiveSet);
      assert(LiveSet->isValid());
      ++II;
    }
    while (II != EI) {
      if (!isa<DbgInfoIntrinsic>(&*II))
        break;
      resetInsertionPoint(*II);
      processDbgInstruction(*II, LiveSet);
      assert(LiveSet->isValid());
      ++II;
    }

    // We've processed everything in the "frame". Now determine which variables
    // cannot be represented by a dbg.declare.
    for (auto Var : VarsTouchedThisFrame) {
      LocKind Loc = getLocKind(LiveSet, Var);
      // If a variable's LocKind is anything other than LocKind::Mem then we
      // must note that it cannot be represented with a dbg.declare.
      // Note that this check is enough without having to check the result of
      // joins() because for join to produce anything other than Mem after
      // we've already seen a Mem we'd be joining None or Val with Mem. In that
      // case, we've already hit this codepath when we set the LocKind to Val
      // or None in that block.
      if (Loc != LocKind::Mem) {
        DebugVariable DbgVar = FnVarLocs->getVariable(Var);
        DebugAggregate Aggr{DbgVar.getVariable(), DbgVar.getInlinedAt()};
        NotAlwaysStackHomed.insert(Aggr);
      }
    }
    VarsTouchedThisFrame.clear();
  }
}

AssignmentTrackingLowering::LocKind
AssignmentTrackingLowering::joinKind(LocKind A, LocKind B) {
  // Partial order:
  // None > Mem, Val
  return A == B ? A : LocKind::None;
}

AssignmentTrackingLowering::LocMap
AssignmentTrackingLowering::joinLocMap(const LocMap &A, const LocMap &B) {
  // Join A and B.
  //
  // U = join(a, b) for a in A, b in B where Var(a) == Var(b)
  // D = join(x, ⊤) for x where Var(x) is in A xor B
  // Join = U ∪ D
  //
  // This is achieved by performing a join on elements from A and B with
  // variables common to both A and B (join elements indexed by var intersect),
  // then adding LocKind::None elements for vars in A xor B. The latter part is
  // equivalent to performing join on elements with variables in A xor B with
  // LocKind::None (⊤) since join(x, ⊤) = ⊤.
  LocMap Join;
  SmallVector<VariableID, 16> SymmetricDifference;
  // Insert the join of the elements with common vars into Join. Add the
  // remaining elements to into SymmetricDifference.
  for (const auto &[Var, Loc] : A) {
    // If this Var doesn't exist in B then add it to the symmetric difference
    // set.
    auto R = B.find(Var);
    if (R == B.end()) {
      SymmetricDifference.push_back(Var);
      continue;
    }
    // There is an entry for Var in both, join it.
    Join[Var] = joinKind(Loc, R->second);
  }
  unsigned IntersectSize = Join.size();
  (void)IntersectSize;

  // Add the elements in B with variables that are not in A into
  // SymmetricDifference.
  for (const auto &Pair : B) {
    VariableID Var = Pair.first;
    if (A.count(Var) == 0)
      SymmetricDifference.push_back(Var);
  }

  // Add SymmetricDifference elements to Join and return the result.
  for (const auto &Var : SymmetricDifference)
    Join.insert({Var, LocKind::None});

  assert(Join.size() == (IntersectSize + SymmetricDifference.size()));
  assert(Join.size() >= A.size() && Join.size() >= B.size());
  return Join;
}

AssignmentTrackingLowering::Assignment
AssignmentTrackingLowering::joinAssignment(const Assignment &A,
                                           const Assignment &B) {
  // Partial order:
  // NoneOrPhi(null, null) > Known(v, ?s)

  // If either are NoneOrPhi the join is NoneOrPhi.
  // If either value is different then the result is
  // NoneOrPhi (joining two values is a Phi).
  if (!A.isSameSourceAssignment(B))
    return Assignment::makeNoneOrPhi();
  if (A.Status == Assignment::NoneOrPhi)
    return Assignment::makeNoneOrPhi();

  // Source is used to lookup the value + expression in the debug program if
  // the stack slot gets assigned a value earlier than expected. Because
  // we're only tracking the one dbg.assign, we can't capture debug PHIs.
  // It's unlikely that we're losing out on much coverage by avoiding that
  // extra work.
  // The Source may differ in this situation:
  // Pred.1:
  //   dbg.assign i32 0, ..., !1, ...
  // Pred.2:
  //   dbg.assign i32 1, ..., !1, ...
  // Here the same assignment (!1) was performed in both preds in the source,
  // but we can't use either one unless they are identical (e.g. .we don't
  // want to arbitrarily pick between constant values).
  auto JoinSource = [&]() -> DbgAssignIntrinsic * {
    if (A.Source == B.Source)
      return A.Source;
    if (A.Source == nullptr || B.Source == nullptr)
      return nullptr;
    if (A.Source->isIdenticalTo(B.Source))
      return A.Source;
    return nullptr;
  };
  DbgAssignIntrinsic *Source = JoinSource();
  assert(A.Status == B.Status && A.Status == Assignment::Known);
  assert(A.ID == B.ID);
  return Assignment::make(A.ID, Source);
}

AssignmentTrackingLowering::AssignmentMap
AssignmentTrackingLowering::joinAssignmentMap(const AssignmentMap &A,
                                              const AssignmentMap &B) {
  // Join A and B.
  //
  // U = join(a, b) for a in A, b in B where Var(a) == Var(b)
  // D = join(x, ⊤) for x where Var(x) is in A xor B
  // Join = U ∪ D
  //
  // This is achieved by performing a join on elements from A and B with
  // variables common to both A and B (join elements indexed by var intersect),
  // then adding LocKind::None elements for vars in A xor B. The latter part is
  // equivalent to performing join on elements with variables in A xor B with
  // Status::NoneOrPhi (⊤) since join(x, ⊤) = ⊤.
  AssignmentMap Join;
  SmallVector<VariableID, 16> SymmetricDifference;
  // Insert the join of the elements with common vars into Join. Add the
  // remaining elements to into SymmetricDifference.
  for (const auto &[Var, AV] : A) {
    // If this Var doesn't exist in B then add it to the symmetric difference
    // set.
    auto R = B.find(Var);
    if (R == B.end()) {
      SymmetricDifference.push_back(Var);
      continue;
    }
    // There is an entry for Var in both, join it.
    Join[Var] = joinAssignment(AV, R->second);
  }
  unsigned IntersectSize = Join.size();
  (void)IntersectSize;

  // Add the elements in B with variables that are not in A into
  // SymmetricDifference.
  for (const auto &Pair : B) {
    VariableID Var = Pair.first;
    if (A.count(Var) == 0)
      SymmetricDifference.push_back(Var);
  }

  // Add SymmetricDifference elements to Join and return the result.
  for (auto Var : SymmetricDifference)
    Join.insert({Var, Assignment::makeNoneOrPhi()});

  assert(Join.size() == (IntersectSize + SymmetricDifference.size()));
  assert(Join.size() >= A.size() && Join.size() >= B.size());
  return Join;
}

AssignmentTrackingLowering::BlockInfo
AssignmentTrackingLowering::joinBlockInfo(const BlockInfo &A,
                                          const BlockInfo &B) {
  BlockInfo Join;
  Join.LiveLoc = joinLocMap(A.LiveLoc, B.LiveLoc);
  Join.StackHomeValue = joinAssignmentMap(A.StackHomeValue, B.StackHomeValue);
  Join.DebugValue = joinAssignmentMap(A.DebugValue, B.DebugValue);
  assert(Join.isValid());
  return Join;
}

bool AssignmentTrackingLowering::join(
    const BasicBlock &BB, const SmallPtrSet<BasicBlock *, 16> &Visited) {
  BlockInfo BBLiveIn;
  bool FirstJoin = true;
  // LiveIn locs for BB is the join of the already-processed preds' LiveOut
  // locs.
  for (auto I = pred_begin(&BB), E = pred_end(&BB); I != E; I++) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block. This
    // is essentially the same as initialising all LocKinds and Assignments to
    // an implicit ⊥ value which is the identity value for the join operation.
    const BasicBlock *Pred = *I;
    if (!Visited.count(Pred))
      continue;

    auto PredLiveOut = LiveOut.find(Pred);
    // Pred must have been processed already. See comment at start of this loop.
    assert(PredLiveOut != LiveOut.end());

    // Perform the join of BBLiveIn (current live-in info) and PrevLiveOut.
    if (FirstJoin)
      BBLiveIn = PredLiveOut->second;
    else
      BBLiveIn = joinBlockInfo(std::move(BBLiveIn), PredLiveOut->second);
    FirstJoin = false;
  }

  auto CurrentLiveInEntry = LiveIn.find(&BB);
  // Check if there isn't an entry, or there is but the LiveIn set has changed
  // (expensive check).
  if (CurrentLiveInEntry == LiveIn.end() ||
      BBLiveIn != CurrentLiveInEntry->second) {
    LiveIn[&BB] = std::move(BBLiveIn);
    // A change has occured.
    return true;
  }
  // No change.
  return false;
}

/// Return true if A fully contains B.
static bool fullyContains(DIExpression::FragmentInfo A,
                          DIExpression::FragmentInfo B) {
  auto ALeft = A.OffsetInBits;
  auto BLeft = B.OffsetInBits;
  if (BLeft < ALeft)
    return false;

  auto ARight = ALeft + A.SizeInBits;
  auto BRight = BLeft + B.SizeInBits;
  if (BRight > ARight)
    return false;
  return true;
}

static std::optional<at::AssignmentInfo>
getUntaggedStoreAssignmentInfo(const Instruction &I, const DataLayout &Layout) {
  // Don't bother checking if this is an AllocaInst. We know this
  // instruction has no tag which means there are no variables associated
  // with it.
  if (const auto *SI = dyn_cast<StoreInst>(&I))
    return at::getAssignmentInfo(Layout, SI);
  if (const auto *MI = dyn_cast<MemIntrinsic>(&I))
    return at::getAssignmentInfo(Layout, MI);
  // Alloca or non-store-like inst.
  return std::nullopt;
}

/// Build a map of {Variable x: Variables y} where all variable fragments
/// contained within the variable fragment x are in set y. This means that
/// y does not contain all overlaps because partial overlaps are excluded.
///
/// While we're iterating over the function, add single location defs for
/// dbg.declares to \p FnVarLocs
///
/// Finally, populate UntaggedStoreVars with a mapping of untagged stores to
/// the stored-to variable fragments.
///
/// These tasks are bundled together to reduce the number of times we need
/// to iterate over the function as they can be achieved together in one pass.
static AssignmentTrackingLowering::OverlapMap buildOverlapMapAndRecordDeclares(
    Function &Fn, FunctionVarLocsBuilder *FnVarLocs,
    AssignmentTrackingLowering::UntaggedStoreAssignmentMap &UntaggedStoreVars) {
  DenseSet<DebugVariable> Seen;
  // Map of Variable: [Fragments].
  DenseMap<DebugAggregate, SmallVector<DebugVariable, 8>> FragmentMap;
  // Iterate over all instructions:
  // - dbg.declare    -> add single location variable record
  // - dbg.*          -> Add fragments to FragmentMap
  // - untagged store -> Add fragments to FragmentMap and update
  //                     UntaggedStoreVars.
  // We need to add fragments for untagged stores too so that we can correctly
  // clobber overlapped fragment locations later.
  for (auto &BB : Fn) {
    for (auto &I : BB) {
      if (auto *DDI = dyn_cast<DbgDeclareInst>(&I)) {
        FnVarLocs->addSingleLocVar(DebugVariable(DDI), DDI->getExpression(),
                                   DDI->getDebugLoc(), DDI->getAddress());
      } else if (auto *DII = dyn_cast<DbgVariableIntrinsic>(&I)) {
        DebugVariable DV = DebugVariable(DII);
        DebugAggregate DA = {DV.getVariable(), DV.getInlinedAt()};
        if (Seen.insert(DV).second)
          FragmentMap[DA].push_back(DV);
      } else if (auto Info = getUntaggedStoreAssignmentInfo(
                     I, Fn.getParent()->getDataLayout())) {
        // Find markers linked to this alloca.
        for (DbgAssignIntrinsic *DAI : at::getAssignmentMarkers(Info->Base)) {
          // Discard the fragment if it covers the entire variable.
          std::optional<DIExpression::FragmentInfo> FragInfo =
              [&Info, DAI]() -> std::optional<DIExpression::FragmentInfo> {
            DIExpression::FragmentInfo F;
            F.OffsetInBits = Info->OffsetInBits;
            F.SizeInBits = Info->SizeInBits;
            if (auto ExistingFrag = DAI->getExpression()->getFragmentInfo())
              F.OffsetInBits += ExistingFrag->OffsetInBits;
            if (auto Sz = DAI->getVariable()->getSizeInBits()) {
              if (F.OffsetInBits == 0 && F.SizeInBits == *Sz)
                return std::nullopt;
            }
            return F;
          }();

          DebugVariable DV = DebugVariable(DAI->getVariable(), FragInfo,
                                           DAI->getDebugLoc().getInlinedAt());
          DebugAggregate DA = {DV.getVariable(), DV.getInlinedAt()};

          // Cache this info for later.
          UntaggedStoreVars[&I].push_back(
              {FnVarLocs->insertVariable(DV), *Info});

          if (Seen.insert(DV).second)
            FragmentMap[DA].push_back(DV);
        }
      }
    }
  }

  // Sort the fragment map for each DebugAggregate in non-descending
  // order of fragment size. Assert no entries are duplicates.
  for (auto &Pair : FragmentMap) {
    SmallVector<DebugVariable, 8> &Frags = Pair.second;
    std::sort(
        Frags.begin(), Frags.end(), [](DebugVariable Next, DebugVariable Elmt) {
          assert(!(Elmt.getFragmentOrDefault() == Next.getFragmentOrDefault()));
          return Elmt.getFragmentOrDefault().SizeInBits >
                 Next.getFragmentOrDefault().SizeInBits;
        });
  }

  // Build the map.
  AssignmentTrackingLowering::OverlapMap Map;
  for (auto Pair : FragmentMap) {
    auto &Frags = Pair.second;
    for (auto It = Frags.begin(), IEnd = Frags.end(); It != IEnd; ++It) {
      DIExpression::FragmentInfo Frag = It->getFragmentOrDefault();
      // Find the frags that this is contained within.
      //
      // Because Frags is sorted by size and none have the same offset and
      // size, we know that this frag can only be contained by subsequent
      // elements.
      SmallVector<DebugVariable, 8>::iterator OtherIt = It;
      ++OtherIt;
      VariableID ThisVar = FnVarLocs->insertVariable(*It);
      for (; OtherIt != IEnd; ++OtherIt) {
        DIExpression::FragmentInfo OtherFrag = OtherIt->getFragmentOrDefault();
        VariableID OtherVar = FnVarLocs->insertVariable(*OtherIt);
        if (fullyContains(OtherFrag, Frag))
          Map[OtherVar].push_back(ThisVar);
      }
    }
  }

  return Map;
}

bool AssignmentTrackingLowering::run(FunctionVarLocsBuilder *FnVarLocsBuilder) {
  if (Fn.size() > MaxNumBlocks) {
    LLVM_DEBUG(dbgs() << "[AT] Dropping var locs in: " << Fn.getName()
                      << ": too many blocks (" << Fn.size() << ")\n");
    at::deleteAll(&Fn);
    return false;
  }

  FnVarLocs = FnVarLocsBuilder;

  // The general structure here is inspired by VarLocBasedImpl.cpp
  // (LiveDebugValues).

  // Build the variable fragment overlap map.
  // Note that this pass doesn't handle partial overlaps correctly (FWIW
  // neither does LiveDebugVariables) because that is difficult to do and
  // appears to be rare occurance.
  VarContains =
      buildOverlapMapAndRecordDeclares(Fn, FnVarLocs, UntaggedStoreVars);

  // Prepare for traversal.
  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;
  DenseMap<unsigned int, BasicBlock *> OrderToBB;
  DenseMap<BasicBlock *, unsigned int> BBToOrder;
  { // Init OrderToBB and BBToOrder.
    unsigned int RPONumber = 0;
    for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
      OrderToBB[RPONumber] = *RI;
      BBToOrder[*RI] = RPONumber;
      Worklist.push(RPONumber);
      ++RPONumber;
    }
    LiveIn.init(RPONumber);
    LiveOut.init(RPONumber);
  }

  // Perform the traversal.
  //
  // This is a standard "union of predecessor outs" dataflow problem. To solve
  // it, we perform join() and process() using the two worklist method until
  // the LiveIn data for each block becomes unchanging. The "proof" that this
  // terminates can be put together by looking at the comments around LocKind,
  // Assignment, and the various join methods, which show that all the elements
  // involved are made up of join-semilattices; LiveIn(n) can only
  // monotonically increase in value throughout the dataflow.
  //
  SmallPtrSet<BasicBlock *, 16> Visited;
  while (!Worklist.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.
    SmallPtrSet<BasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      BasicBlock *BB = OrderToBB[Worklist.top()];
      LLVM_DEBUG(dbgs() << "\nPop BB " << BB->getName() << "\n");
      Worklist.pop();
      bool InChanged = join(*BB, Visited);
      // Always consider LiveIn changed on the first visit.
      InChanged |= Visited.insert(BB).second;
      if (InChanged) {
        LLVM_DEBUG(dbgs() << BB->getName() << " has new InLocs, process it\n");
        // Mutate a copy of LiveIn while processing BB. After calling process
        // LiveSet is the LiveOut set for BB.
        BlockInfo LiveSet = LiveIn[BB];

        // Process the instructions in the block.
        process(*BB, &LiveSet);

        // Relatively expensive check: has anything changed in LiveOut for BB?
        if (LiveOut[BB] != LiveSet) {
          LLVM_DEBUG(dbgs() << BB->getName()
                            << " has new OutLocs, add succs to worklist: [ ");
          LiveOut[BB] = std::move(LiveSet);
          for (auto I = succ_begin(BB), E = succ_end(BB); I != E; I++) {
            if (OnPending.insert(*I).second) {
              LLVM_DEBUG(dbgs() << I->getName() << " ");
              Pending.push(BBToOrder[*I]);
            }
          }
          LLVM_DEBUG(dbgs() << "]\n");
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // That's the hard part over. Now we just have some admin to do.

  // Record whether we inserted any intrinsics.
  bool InsertedAnyIntrinsics = false;

  // Identify and add defs for single location variables.
  //
  // Go through all of the defs that we plan to add. If the aggregate variable
  // it's a part of is not in the NotAlwaysStackHomed set we can emit a single
  // location def and omit the rest. Add an entry to AlwaysStackHomed so that
  // we can identify those uneeded defs later.
  DenseSet<DebugAggregate> AlwaysStackHomed;
  for (const auto &Pair : InsertBeforeMap) {
    const auto &Vec = Pair.second;
    for (VarLocInfo VarLoc : Vec) {
      DebugVariable Var = FnVarLocs->getVariable(VarLoc.VariableID);
      DebugAggregate Aggr{Var.getVariable(), Var.getInlinedAt()};

      // Skip this Var if it's not always stack homed.
      if (NotAlwaysStackHomed.contains(Aggr))
        continue;

      // Skip complex cases such as when different fragments of a variable have
      // been split into different allocas. Skipping in this case means falling
      // back to using a list of defs (which could reduce coverage, but is no
      // less correct).
      bool Simple =
          VarLoc.Expr->getNumElements() == 1 && VarLoc.Expr->startsWithDeref();
      if (!Simple) {
        NotAlwaysStackHomed.insert(Aggr);
        continue;
      }

      // All source assignments to this variable remain and all stores to any
      // part of the variable store to the same address (with varying
      // offsets). We can just emit a single location for the whole variable.
      //
      // Unless we've already done so, create the single location def now.
      if (AlwaysStackHomed.insert(Aggr).second) {
        assert(isa<AllocaInst>(VarLoc.V));
        // TODO: When more complex cases are handled VarLoc.Expr should be
        // built appropriately rather than always using an empty DIExpression.
        // The assert below is a reminder.
        assert(Simple);
        VarLoc.Expr = DIExpression::get(Fn.getContext(), std::nullopt);
        DebugVariable Var = FnVarLocs->getVariable(VarLoc.VariableID);
        FnVarLocs->addSingleLocVar(Var, VarLoc.Expr, VarLoc.DL, VarLoc.V);
        InsertedAnyIntrinsics = true;
      }
    }
  }

  // Insert the other DEFs.
  for (const auto &[InsertBefore, Vec] : InsertBeforeMap) {
    SmallVector<VarLocInfo> NewDefs;
    for (const VarLocInfo &VarLoc : Vec) {
      DebugVariable Var = FnVarLocs->getVariable(VarLoc.VariableID);
      DebugAggregate Aggr{Var.getVariable(), Var.getInlinedAt()};
      // If this variable is always stack homed then we have already inserted a
      // dbg.declare and deleted this dbg.value.
      if (AlwaysStackHomed.contains(Aggr))
        continue;
      NewDefs.push_back(VarLoc);
      InsertedAnyIntrinsics = true;
    }

    FnVarLocs->setWedge(InsertBefore, std::move(NewDefs));
  }

  InsertedAnyIntrinsics |= emitPromotedVarLocs(FnVarLocs);

  return InsertedAnyIntrinsics;
}

bool AssignmentTrackingLowering::emitPromotedVarLocs(
    FunctionVarLocsBuilder *FnVarLocs) {
  bool InsertedAnyIntrinsics = false;
  // Go through every block, translating debug intrinsics for fully promoted
  // variables into FnVarLocs location defs. No analysis required for these.
  for (auto &BB : Fn) {
    for (auto &I : BB) {
      // Skip instructions other than dbg.values and dbg.assigns.
      auto *DVI = dyn_cast<DbgValueInst>(&I);
      if (!DVI)
        continue;
      // Skip variables that haven't been promoted - we've dealt with those
      // already.
      if (VarsWithStackSlot->contains(getAggregate(DVI)))
        continue;
      // Wrapper to get a single value (or undef) from DVI.
      auto GetValue = [DVI]() -> Value * {
        // Conditions for undef: Any operand undef, zero operands or single
        // operand is nullptr. We also can't handle variadic DIExpressions yet.
        // Some of those conditions don't have a type we can pick for
        // undef. Use i32.
        if (DVI->isUndef() || DVI->getValue() == nullptr || DVI->hasArgList())
          return UndefValue::get(Type::getInt32Ty(DVI->getContext()));
        return DVI->getValue();
      };
      Instruction *InsertBefore = I.getNextNode();
      assert(InsertBefore && "Unexpected: debug intrinsics after a terminator");
      FnVarLocs->addVarLoc(InsertBefore, DebugVariable(DVI),
                           DVI->getExpression(), DVI->getDebugLoc(),
                           GetValue());
      InsertedAnyIntrinsics = true;
    }
  }
  return InsertedAnyIntrinsics;
}

/// Remove redundant definitions within sequences of consecutive location defs.
/// This is done using a backward scan to keep the last def describing a
/// specific variable/fragment.
///
/// This implements removeRedundantDbgInstrsUsingBackwardScan from
/// lib/Transforms/Utils/BasicBlockUtils.cpp for locations described with
/// FunctionVarLocsBuilder instead of with intrinsics.
static bool
removeRedundantDbgLocsUsingBackwardScan(const BasicBlock *BB,
                                        FunctionVarLocsBuilder &FnVarLocs) {
  bool Changed = false;
  SmallDenseSet<DebugVariable> VariableSet;

  // Scan over the entire block, not just over the instructions mapped by
  // FnVarLocs, because wedges in FnVarLocs may only be seperated by debug
  // instructions.
  for (const Instruction &I : reverse(*BB)) {
    if (!isa<DbgVariableIntrinsic>(I)) {
      // Sequence of consecutive defs ended. Clear map for the next one.
      VariableSet.clear();
    }

    // Get the location defs that start just before this instruction.
    const auto *Locs = FnVarLocs.getWedge(&I);
    if (!Locs)
      continue;

    NumWedgesScanned++;
    bool ChangedThisWedge = false;
    // The new pruned set of defs, reversed because we're scanning backwards.
    SmallVector<VarLocInfo> NewDefsReversed;

    // Iterate over the existing defs in reverse.
    for (auto RIt = Locs->rbegin(), REnd = Locs->rend(); RIt != REnd; ++RIt) {
      NumDefsScanned++;
      const DebugVariable &Key = FnVarLocs.getVariable(RIt->VariableID);
      bool FirstDefOfFragment = VariableSet.insert(Key).second;

      // If the same variable fragment is described more than once it is enough
      // to keep the last one (i.e. the first found in this reverse iteration).
      if (FirstDefOfFragment) {
        // New def found: keep it.
        NewDefsReversed.push_back(*RIt);
      } else {
        // Redundant def found: throw it away. Since the wedge of defs is being
        // rebuilt, doing nothing is the same as deleting an entry.
        ChangedThisWedge = true;
        NumDefsRemoved++;
      }
      continue;
    }

    // Un-reverse the defs and replace the wedge with the pruned version.
    if (ChangedThisWedge) {
      std::reverse(NewDefsReversed.begin(), NewDefsReversed.end());
      FnVarLocs.setWedge(&I, std::move(NewDefsReversed));
      NumWedgesChanged++;
      Changed = true;
    }
  }

  return Changed;
}

/// Remove redundant location defs using a forward scan. This can remove a
/// location definition that is redundant due to indicating that a variable has
/// the same value as is already being indicated by an earlier def.
///
/// This implements removeRedundantDbgInstrsUsingForwardScan from
/// lib/Transforms/Utils/BasicBlockUtils.cpp for locations described with
/// FunctionVarLocsBuilder instead of with intrinsics
static bool
removeRedundantDbgLocsUsingForwardScan(const BasicBlock *BB,
                                       FunctionVarLocsBuilder &FnVarLocs) {
  bool Changed = false;
  DenseMap<DebugVariable, std::pair<Value *, DIExpression *>> VariableMap;

  // Scan over the entire block, not just over the instructions mapped by
  // FnVarLocs, because wedges in FnVarLocs may only be seperated by debug
  // instructions.
  for (const Instruction &I : *BB) {
    // Get the defs that come just before this instruction.
    const auto *Locs = FnVarLocs.getWedge(&I);
    if (!Locs)
      continue;

    NumWedgesScanned++;
    bool ChangedThisWedge = false;
    // The new pruned set of defs.
    SmallVector<VarLocInfo> NewDefs;

    // Iterate over the existing defs.
    for (const VarLocInfo &Loc : *Locs) {
      NumDefsScanned++;
      DebugVariable Key(FnVarLocs.getVariable(Loc.VariableID).getVariable(),
                        std::nullopt, Loc.DL.getInlinedAt());
      auto VMI = VariableMap.find(Key);

      // Update the map if we found a new value/expression describing the
      // variable, or if the variable wasn't mapped already.
      if (VMI == VariableMap.end() || VMI->second.first != Loc.V ||
          VMI->second.second != Loc.Expr) {
        VariableMap[Key] = {Loc.V, Loc.Expr};
        NewDefs.push_back(Loc);
        continue;
      }

      // Did not insert this Loc, which is the same as removing it.
      ChangedThisWedge = true;
      NumDefsRemoved++;
    }

    // Replace the existing wedge with the pruned version.
    if (ChangedThisWedge) {
      FnVarLocs.setWedge(&I, std::move(NewDefs));
      NumWedgesChanged++;
      Changed = true;
    }
  }

  return Changed;
}

static bool
removeUndefDbgLocsFromEntryBlock(const BasicBlock *BB,
                                 FunctionVarLocsBuilder &FnVarLocs) {
  assert(BB->isEntryBlock());
  // Do extra work to ensure that we remove semantically unimportant undefs.
  //
  // This is to work around the fact that SelectionDAG will hoist dbg.values
  // using argument values to the top of the entry block. That can move arg
  // dbg.values before undef and constant dbg.values which they previously
  // followed. The easiest thing to do is to just try to feed SelectionDAG
  // input it's happy with.
  //
  // Map of {Variable x: Fragments y} where the fragments y of variable x have
  // have at least one non-undef location defined already. Don't use directly,
  // instead call DefineBits and HasDefinedBits.
  SmallDenseMap<DebugAggregate, SmallDenseSet<DIExpression::FragmentInfo>>
      VarsWithDef;
  // Specify that V (a fragment of A) has a non-undef location.
  auto DefineBits = [&VarsWithDef](DebugAggregate A, DebugVariable V) {
    VarsWithDef[A].insert(V.getFragmentOrDefault());
  };
  // Return true if a non-undef location has been defined for V (a fragment of
  // A). Doesn't imply that the location is currently non-undef, just that a
  // non-undef location has been seen previously.
  auto HasDefinedBits = [&VarsWithDef](DebugAggregate A, DebugVariable V) {
    auto FragsIt = VarsWithDef.find(A);
    if (FragsIt == VarsWithDef.end())
      return false;
    return llvm::any_of(FragsIt->second, [V](auto Frag) {
      return DIExpression::fragmentsOverlap(Frag, V.getFragmentOrDefault());
    });
  };

  bool Changed = false;
  DenseMap<DebugVariable, std::pair<Value *, DIExpression *>> VariableMap;

  // Scan over the entire block, not just over the instructions mapped by
  // FnVarLocs, because wedges in FnVarLocs may only be seperated by debug
  // instructions.
  for (const Instruction &I : *BB) {
    // Get the defs that come just before this instruction.
    const auto *Locs = FnVarLocs.getWedge(&I);
    if (!Locs)
      continue;

    NumWedgesScanned++;
    bool ChangedThisWedge = false;
    // The new pruned set of defs.
    SmallVector<VarLocInfo> NewDefs;

    // Iterate over the existing defs.
    for (const VarLocInfo &Loc : *Locs) {
      NumDefsScanned++;
      DebugAggregate Aggr{FnVarLocs.getVariable(Loc.VariableID).getVariable(),
                          Loc.DL.getInlinedAt()};
      DebugVariable Var = FnVarLocs.getVariable(Loc.VariableID);

      // Remove undef entries that are encountered before any non-undef
      // intrinsics from the entry block.
      if (isa<UndefValue>(Loc.V) && !HasDefinedBits(Aggr, Var)) {
        // Did not insert this Loc, which is the same as removing it.
        NumDefsRemoved++;
        ChangedThisWedge = true;
        continue;
      }

      DefineBits(Aggr, Var);
      NewDefs.push_back(Loc);
    }

    // Replace the existing wedge with the pruned version.
    if (ChangedThisWedge) {
      FnVarLocs.setWedge(&I, std::move(NewDefs));
      NumWedgesChanged++;
      Changed = true;
    }
  }

  return Changed;
}

static bool removeRedundantDbgLocs(const BasicBlock *BB,
                                   FunctionVarLocsBuilder &FnVarLocs) {
  bool MadeChanges = false;
  MadeChanges |= removeRedundantDbgLocsUsingBackwardScan(BB, FnVarLocs);
  if (BB->isEntryBlock())
    MadeChanges |= removeUndefDbgLocsFromEntryBlock(BB, FnVarLocs);
  MadeChanges |= removeRedundantDbgLocsUsingForwardScan(BB, FnVarLocs);

  if (MadeChanges)
    LLVM_DEBUG(dbgs() << "Removed redundant dbg locs from: " << BB->getName()
                      << "\n");
  return MadeChanges;
}

static DenseSet<DebugAggregate> findVarsWithStackSlot(Function &Fn) {
  DenseSet<DebugAggregate> Result;
  for (auto &BB : Fn) {
    for (auto &I : BB) {
      // Any variable linked to an instruction is considered
      // interesting. Ideally we only need to check Allocas, however, a
      // DIAssignID might get dropped from an alloca but not stores. In that
      // case, we need to consider the variable interesting for NFC behaviour
      // with this change. TODO: Consider only looking at allocas.
      for (DbgAssignIntrinsic *DAI : at::getAssignmentMarkers(&I)) {
        Result.insert({DAI->getVariable(), DAI->getDebugLoc().getInlinedAt()});
      }
    }
  }
  return Result;
}

static void analyzeFunction(Function &Fn, const DataLayout &Layout,
                            FunctionVarLocsBuilder *FnVarLocs) {
  // The analysis will generate location definitions for all variables, but we
  // only need to perform a dataflow on the set of variables which have a stack
  // slot. Find those now.
  DenseSet<DebugAggregate> VarsWithStackSlot = findVarsWithStackSlot(Fn);

  bool Changed = false;

  // Use a scope block to clean up AssignmentTrackingLowering before running
  // MemLocFragmentFill to reduce peak memory consumption.
  {
    AssignmentTrackingLowering Pass(Fn, Layout, &VarsWithStackSlot);
    Changed = Pass.run(FnVarLocs);
  }

  if (Changed) {
    MemLocFragmentFill Pass(Fn, &VarsWithStackSlot);
    Pass.run(FnVarLocs);

    // Remove redundant entries. As well as reducing memory consumption and
    // avoiding waiting cycles later by burning some now, this has another
    // important job. That is to work around some SelectionDAG quirks. See
    // removeRedundantDbgLocsUsingForwardScan comments for more info on that.
    for (auto &BB : Fn)
      removeRedundantDbgLocs(&BB, *FnVarLocs);
  }
}

bool AssignmentTrackingAnalysis::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "AssignmentTrackingAnalysis run on " << F.getName()
                    << "\n");
  auto DL = std::make_unique<DataLayout>(F.getParent());

  // Clear previous results.
  Results->clear();

  FunctionVarLocsBuilder Builder;
  analyzeFunction(F, *DL.get(), &Builder);

  // Save these results.
  Results->init(Builder);

  if (PrintResults && isFunctionInPrintList(F.getName()))
    Results->print(errs(), F);

  // Return false because this pass does not modify the function.
  return false;
}

AssignmentTrackingAnalysis::AssignmentTrackingAnalysis()
    : FunctionPass(ID), Results(std::make_unique<FunctionVarLocs>()) {}

char AssignmentTrackingAnalysis::ID = 0;

INITIALIZE_PASS(AssignmentTrackingAnalysis, DEBUG_TYPE,
                "Assignment Tracking Analysis", false, true)
