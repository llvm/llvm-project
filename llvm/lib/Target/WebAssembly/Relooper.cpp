//===-- Relooper.cpp - Top-level interface for WebAssembly  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
///
/// \file
/// \brief This implements the Relooper algorithm. This implementation includes
/// optimizations added since the original academic paper [1] was published.
///
/// [1] Alon Zakai. 2011. Emscripten: an LLVM-to-JavaScript compiler. In
/// Proceedings of the ACM international conference companion on Object
/// oriented programming systems languages and applications companion
/// (SPLASH '11). ACM, New York, NY, USA, 301-312. DOI=10.1145/2048147.2048224
/// http://doi.acm.org/10.1145/2048147.2048224
///
//===-------------------------------------------------------------------===//

#include "Relooper.h"
#include "WebAssembly.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <cstdlib>
#include <functional>
#include <list>
#include <stack>
#include <string>

#define DEBUG_TYPE "relooper"

using namespace llvm;
using namespace Relooper;

static cl::opt<int> RelooperSplittingFactor(
    "relooper-splitting-factor",
    cl::desc(
        "How much to discount code size when deciding whether to split a node"),
    cl::init(5));

static cl::opt<unsigned> RelooperMultipleSwitchThreshold(
    "relooper-multiple-switch-threshold",
    cl::desc(
        "How many entries to allow in a multiple before we use a switch"),
    cl::init(10));

static cl::opt<unsigned> RelooperNestingLimit(
    "relooper-nesting-limit",
    cl::desc(
        "How much nesting is acceptable"),
    cl::init(20));


namespace {
///
/// Implements the relooper algorithm for a function's blocks.
///
/// Implementation details: The Relooper instance has
/// ownership of the blocks and shapes, and frees them when done.
///
struct RelooperAlgorithm {
  std::deque<Block *> Blocks;
  std::deque<Shape *> Shapes;
  Shape *Root;
  bool MinSize;
  int BlockIdCounter;
  int ShapeIdCounter;

  RelooperAlgorithm();
  ~RelooperAlgorithm();

  void AddBlock(Block *New, int Id = -1);

  // Calculates the shapes
  void Calculate(Block *Entry);

  // Sets us to try to minimize size
  void SetMinSize(bool MinSize_) { MinSize = MinSize_; }
};

struct RelooperAnalysis final : public FunctionPass {
  static char ID;
  RelooperAnalysis() : FunctionPass(ID) {}
  const char *getPassName() const override { return "relooper"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
  bool runOnFunction(Function &F) override;
};
}

// RelooperAnalysis

char RelooperAnalysis::ID = 0;
FunctionPass *llvm::createWebAssemblyRelooper() {
  return new RelooperAnalysis();
}

bool RelooperAnalysis::runOnFunction(Function &F) {
  DEBUG(dbgs() << "Relooping function '" << F.getName() << "'\n");
  RelooperAlgorithm R;
  // FIXME: remove duplication between relooper's and LLVM's BBs.
  std::map<const BasicBlock *, Block *> BB2B;
  std::map<const Block *, const BasicBlock *> B2BB;
  for (const BasicBlock &BB : F) {
    // FIXME: getName is wrong here, Code is meant to represent amount of code.
    // FIXME: use BranchVarInit for switch.
    Block *B = new Block(BB.getName().str().data(), /*BranchVarInit=*/nullptr);
    R.AddBlock(B);
    assert(BB2B.find(&BB) == BB2B.end() && "Inserting the same block twice");
    assert(B2BB.find(B) == B2BB.end() && "Inserting the same block twice");
    BB2B[&BB] = B;
    B2BB[B] = &BB;
  }
  for (Block *B : R.Blocks) {
    const BasicBlock *BB = B2BB[B];
    for (const BasicBlock *Successor : successors(BB))
      // FIXME: add branch's Condition and Code below.
      B->AddBranchTo(BB2B[Successor], /*Condition=*/nullptr, /*Code=*/nullptr);
  }
  R.Calculate(BB2B[&F.getEntryBlock()]);
  return false; // Analysis passes don't modify anything.
}

// Helpers

typedef MapVector<Block *, BlockSet> BlockBlockSetMap;
typedef std::list<Block *> BlockList;

template <class T, class U>
static bool contains(const T &container, const U &contained) {
  return container.count(contained);
}


// Branch

Branch::Branch(const char *ConditionInit, const char *CodeInit)
    : Ancestor(nullptr), Labeled(true) {
  // FIXME: move from char* to LLVM data structures
  Condition = ConditionInit ? strdup(ConditionInit) : nullptr;
  Code = CodeInit ? strdup(CodeInit) : nullptr;
}

Branch::~Branch() {
  // FIXME: move from char* to LLVM data structures
  free(static_cast<void *>(const_cast<char *>(Condition)));
  free(static_cast<void *>(const_cast<char *>(Code)));
}

// Block

Block::Block(const char *CodeInit, const char *BranchVarInit)
    : Parent(nullptr), Id(-1), IsCheckedMultipleEntry(false) {
  // FIXME: move from char* to LLVM data structures
  Code = strdup(CodeInit);
  BranchVar = BranchVarInit ? strdup(BranchVarInit) : nullptr;
}

Block::~Block() {
  // FIXME: move from char* to LLVM data structures
  free(static_cast<void *>(const_cast<char *>(Code)));
  free(static_cast<void *>(const_cast<char *>(BranchVar)));
}

void Block::AddBranchTo(Block *Target, const char *Condition,
                        const char *Code) {
  assert(!contains(BranchesOut, Target) &&
         "cannot add more than one branch to the same target");
  BranchesOut[Target] = make_unique<Branch>(Condition, Code);
}

// Relooper

RelooperAlgorithm::RelooperAlgorithm()
    : Root(nullptr), MinSize(false), BlockIdCounter(1),
      ShapeIdCounter(0) { // block ID 0 is reserved for clearings
}

RelooperAlgorithm::~RelooperAlgorithm() {
  for (auto Curr : Blocks)
    delete Curr;
  for (auto Curr : Shapes)
    delete Curr;
}

void RelooperAlgorithm::AddBlock(Block *New, int Id) {
  New->Id = Id == -1 ? BlockIdCounter++ : Id;
  Blocks.push_back(New);
}

struct RelooperRecursor {
  RelooperAlgorithm *Parent;
  RelooperRecursor(RelooperAlgorithm *ParentInit) : Parent(ParentInit) {}
};

void RelooperAlgorithm::Calculate(Block *Entry) {
  // Scan and optimize the input
  struct PreOptimizer : public RelooperRecursor {
    PreOptimizer(RelooperAlgorithm *Parent) : RelooperRecursor(Parent) {}
    BlockSet Live;

    void FindLive(Block *Root) {
      BlockList ToInvestigate;
      ToInvestigate.push_back(Root);
      while (!ToInvestigate.empty()) {
        Block *Curr = ToInvestigate.front();
        ToInvestigate.pop_front();
        if (contains(Live, Curr))
          continue;
        Live.insert(Curr);
        for (const auto &iter : Curr->BranchesOut)
          ToInvestigate.push_back(iter.first);
      }
    }

    // If a block has multiple entries but no exits, and it is small enough, it
    // is useful to split it. A common example is a C++ function where
    // everything ends up at a final exit block and does some RAII cleanup.
    // Without splitting, we will be forced to introduce labelled loops to
    // allow reaching the final block
    void SplitDeadEnds() {
      unsigned TotalCodeSize = 0;
      for (const auto &Curr : Live) {
        TotalCodeSize += strlen(Curr->Code);
      }
      BlockSet Splits;
      BlockSet Removed;
      for (const auto &Original : Live) {
        if (Original->BranchesIn.size() <= 1 ||
            !Original->BranchesOut.empty())
          continue; // only dead ends, for now
        if (contains(Original->BranchesOut, Original))
          continue; // cannot split a looping node
        if (strlen(Original->Code) * (Original->BranchesIn.size() - 1) >
            TotalCodeSize / RelooperSplittingFactor)
          continue; // if splitting increases raw code size by a significant
                    // amount, abort
        // Split the node (for simplicity, we replace all the blocks, even
        // though we could have reused the original)
        DEBUG(dbgs() << "  Splitting '" << Original->Code << "'\n");
        for (const auto &Prior : Original->BranchesIn) {
          Block *Split = new Block(Original->Code, Original->BranchVar);
          Parent->AddBlock(Split, Original->Id);
          Split->BranchesIn.insert(Prior);
          std::unique_ptr<Branch> Details;
          Details.swap(Prior->BranchesOut[Original]);
          Prior->BranchesOut[Split] = make_unique<Branch>(Details->Condition,
                                                          Details->Code);
          for (const auto &iter : Original->BranchesOut) {
            Block *Post = iter.first;
            Branch *Details = iter.second.get();
            Split->BranchesOut[Post] = make_unique<Branch>(Details->Condition,
                                                           Details->Code);
            Post->BranchesIn.insert(Split);
          }
          Splits.insert(Split);
          Removed.insert(Original);
        }
        for (const auto &iter : Original->BranchesOut) {
          Block *Post = iter.first;
          Post->BranchesIn.remove(Original);
        }
      }
      for (const auto &iter : Splits)
        Live.insert(iter);
      for (const auto &iter : Removed)
        Live.remove(iter);
    }
  };
  PreOptimizer Pre(this);
  Pre.FindLive(Entry);

  // Add incoming branches from live blocks, ignoring dead code
  for (unsigned i = 0; i < Blocks.size(); i++) {
    Block *Curr = Blocks[i];
    if (!contains(Pre.Live, Curr))
      continue;
    for (const auto &iter : Curr->BranchesOut)
      iter.first->BranchesIn.insert(Curr);
  }

  if (!MinSize)
    Pre.SplitDeadEnds();

  // Recursively process the graph

  struct Analyzer : public RelooperRecursor {
    Analyzer(RelooperAlgorithm *Parent) : RelooperRecursor(Parent) {}

    // Add a shape to the list of shapes in this Relooper calculation
    void Notice(Shape *New) {
      New->Id = Parent->ShapeIdCounter++;
      Parent->Shapes.push_back(New);
    }

    // Create a list of entries from a block. If LimitTo is provided, only
    // results in that set will appear
    void GetBlocksOut(Block *Source, BlockSet &Entries,
                      BlockSet *LimitTo = nullptr) {
      for (const auto &iter : Source->BranchesOut)
        if (!LimitTo || contains(*LimitTo, iter.first))
          Entries.insert(iter.first);
    }

    // Converts/processes all branchings to a specific target
    void Solipsize(Block *Target, Branch::FlowType Type, Shape *Ancestor,
                   BlockSet &From) {
      DEBUG(dbgs() << "  Solipsize '" << Target->Code << "' type " << Type
                   << "\n");
      for (auto iter = Target->BranchesIn.begin();
           iter != Target->BranchesIn.end();) {
        Block *Prior = *iter;
        if (!contains(From, Prior)) {
          iter++;
          continue;
        }
        std::unique_ptr<Branch> PriorOut;
        PriorOut.swap(Prior->BranchesOut[Target]);
        PriorOut->Ancestor = Ancestor;
        PriorOut->Type = Type;
        if (MultipleShape *Multiple = dyn_cast<MultipleShape>(Ancestor))
          Multiple->Breaks++; // We are breaking out of this Multiple, so need a
                              // loop
        iter++; // carefully increment iter before erasing
        Target->BranchesIn.remove(Prior);
        Target->ProcessedBranchesIn.insert(Prior);
        Prior->ProcessedBranchesOut[Target].swap(PriorOut);
      }
    }

    Shape *MakeSimple(BlockSet &Blocks, Block *Inner, BlockSet &NextEntries) {
      DEBUG(dbgs() << "  MakeSimple inner block '" << Inner->Code << "'\n");
      SimpleShape *Simple = new SimpleShape;
      Notice(Simple);
      Simple->Inner = Inner;
      Inner->Parent = Simple;
      if (Blocks.size() > 1) {
        Blocks.remove(Inner);
        GetBlocksOut(Inner, NextEntries, &Blocks);
        BlockSet JustInner;
        JustInner.insert(Inner);
        for (const auto &iter : NextEntries)
          Solipsize(iter, Branch::Direct, Simple, JustInner);
      }
      return Simple;
    }

    Shape *MakeLoop(BlockSet &Blocks, BlockSet &Entries,
                    BlockSet &NextEntries) {
      // Find the inner blocks in this loop. Proceed backwards from the entries
      // until
      // you reach a seen block, collecting as you go.
      BlockSet InnerBlocks;
      BlockSet Queue = Entries;
      while (!Queue.empty()) {
        Block *Curr = *(Queue.begin());
        Queue.remove(*Queue.begin());
        if (!contains(InnerBlocks, Curr)) {
          // This element is new, mark it as inner and remove from outer
          InnerBlocks.insert(Curr);
          Blocks.remove(Curr);
          // Add the elements prior to it
          for (const auto &iter : Curr->BranchesIn)
            Queue.insert(iter);
        }
      }
      assert(!InnerBlocks.empty());

      for (const auto &Curr : InnerBlocks) {
        for (const auto &iter : Curr->BranchesOut) {
          Block *Possible = iter.first;
          if (!contains(InnerBlocks, Possible))
            NextEntries.insert(Possible);
        }
      }

      LoopShape *Loop = new LoopShape();
      Notice(Loop);

      // Solipsize the loop, replacing with break/continue and marking branches
      // as Processed (will not affect later calculations)
      // A. Branches to the loop entries become a continue to this shape
      for (const auto &iter : Entries)
        Solipsize(iter, Branch::Continue, Loop, InnerBlocks);
      // B. Branches to outside the loop (a next entry) become breaks on this
      // shape
      for (const auto &iter : NextEntries)
        Solipsize(iter, Branch::Break, Loop, InnerBlocks);
      // Finish up
      Shape *Inner = Process(InnerBlocks, Entries, nullptr);
      Loop->Inner = Inner;
      return Loop;
    }

    // For each entry, find the independent group reachable by it. The
    // independent group is the entry itself, plus all the blocks it can
    // reach that cannot be directly reached by another entry. Note that we
    // ignore directly reaching the entry itself by another entry.
    //   @param Ignore - previous blocks that are irrelevant
    void FindIndependentGroups(BlockSet &Entries,
                               BlockBlockSetMap &IndependentGroups,
                               BlockSet *Ignore = nullptr) {
      typedef std::map<Block *, Block *> BlockBlockMap;

      struct HelperClass {
        BlockBlockSetMap &IndependentGroups;
        BlockBlockMap Ownership; // For each block, which entry it belongs to.
                                 // We have reached it from there.

        HelperClass(BlockBlockSetMap &IndependentGroupsInit)
            : IndependentGroups(IndependentGroupsInit) {}
        void InvalidateWithChildren(Block *New) {
          // Being in the list means you need to be invalidated
          BlockList ToInvalidate;
          ToInvalidate.push_back(New);
          while (!ToInvalidate.empty()) {
            Block *Invalidatee = ToInvalidate.front();
            ToInvalidate.pop_front();
            Block *Owner = Ownership[Invalidatee];
            // Owner may have been invalidated, do not add to
            // IndependentGroups!
            if (contains(IndependentGroups, Owner))
              IndependentGroups[Owner].remove(Invalidatee);
            if (Ownership[Invalidatee]) { // may have been seen before and
                                          // invalidated already
              Ownership[Invalidatee] = nullptr;
              for (const auto &iter : Invalidatee->BranchesOut) {
                Block *Target = iter.first;
                BlockBlockMap::iterator Known = Ownership.find(Target);
                if (Known != Ownership.end()) {
                  Block *TargetOwner = Known->second;
                  if (TargetOwner)
                    ToInvalidate.push_back(Target);
                }
              }
            }
          }
        }
      };
      HelperClass Helper(IndependentGroups);

      // We flow out from each of the entries, simultaneously.
      // When we reach a new block, we add it as belonging to the one we got to
      // it from.
      // If we reach a new block that is already marked as belonging to someone,
      // it is reachable by two entries and is not valid for any of them.
      // Remove it and all it can reach that have been visited.

      // Being in the queue means we just added this item, and
      // we need to add its children
      BlockList Queue;
      for (const auto &Entry : Entries) {
        Helper.Ownership[Entry] = Entry;
        IndependentGroups[Entry].insert(Entry);
        Queue.push_back(Entry);
      }
      while (!Queue.empty()) {
        Block *Curr = Queue.front();
        Queue.pop_front();
        Block *Owner = Helper.Ownership[Curr]; // Curr must be in the ownership
                                               // map if we are in the queue
        if (!Owner)
          continue; // we have been invalidated meanwhile after being reached
                    // from two entries
        // Add all children
        for (const auto &iter : Curr->BranchesOut) {
          Block *New = iter.first;
          BlockBlockMap::iterator Known = Helper.Ownership.find(New);
          if (Known == Helper.Ownership.end()) {
            // New node. Add it, and put it in the queue
            Helper.Ownership[New] = Owner;
            IndependentGroups[Owner].insert(New);
            Queue.push_back(New);
            continue;
          }
          Block *NewOwner = Known->second;
          if (!NewOwner)
            continue; // We reached an invalidated node
          if (NewOwner != Owner)
            // Invalidate this and all reachable that we have seen - we reached
            // this from two locations
            Helper.InvalidateWithChildren(New);
          // otherwise, we have the same owner, so do nothing
        }
      }

      // Having processed all the interesting blocks, we remain with just one
      // potential issue:
      // If a->b, and a was invalidated, but then b was later reached by
      // someone else, we must invalidate b. To check for this, we go over all
      // elements in the independent groups, if an element has a parent which
      // does *not* have the same owner, we/ must remove it and all its
      // children.

      for (const auto &iter : Entries) {
        BlockSet &CurrGroup = IndependentGroups[iter];
        BlockList ToInvalidate;
        for (const auto &iter : CurrGroup) {
          Block *Child = iter;
          for (const auto &iter : Child->BranchesIn) {
            Block *Parent = iter;
            if (Ignore && contains(*Ignore, Parent))
              continue;
            if (Helper.Ownership[Parent] != Helper.Ownership[Child])
              ToInvalidate.push_back(Child);
          }
        }
        while (!ToInvalidate.empty()) {
          Block *Invalidatee = ToInvalidate.front();
          ToInvalidate.pop_front();
          Helper.InvalidateWithChildren(Invalidatee);
        }
      }

      // Remove empty groups
      for (const auto &iter : Entries)
        if (IndependentGroups[iter].empty())
          IndependentGroups.erase(iter);
    }

    Shape *MakeMultiple(BlockSet &Blocks, BlockSet &Entries,
                        BlockBlockSetMap &IndependentGroups, Shape *Prev,
                        BlockSet &NextEntries) {
      bool Fused = isa<SimpleShape>(Prev);
      MultipleShape *Multiple = new MultipleShape();
      Notice(Multiple);
      BlockSet CurrEntries;
      for (auto &iter : IndependentGroups) {
        Block *CurrEntry = iter.first;
        BlockSet &CurrBlocks = iter.second;
        // Create inner block
        CurrEntries.clear();
        CurrEntries.insert(CurrEntry);
        for (const auto &CurrInner : CurrBlocks) {
          // Remove the block from the remaining blocks
          Blocks.remove(CurrInner);
          // Find new next entries and fix branches to them
          for (auto iter = CurrInner->BranchesOut.begin();
               iter != CurrInner->BranchesOut.end();) {
            Block *CurrTarget = iter->first;
            auto Next = iter;
            Next++;
            if (!contains(CurrBlocks, CurrTarget)) {
              NextEntries.insert(CurrTarget);
              Solipsize(CurrTarget, Branch::Break, Multiple, CurrBlocks);
            }
            iter = Next; // increment carefully because Solipsize can remove us
          }
        }
        Multiple->InnerMap[CurrEntry->Id] =
            Process(CurrBlocks, CurrEntries, nullptr);
        // If we are not fused, then our entries will actually be checked
        if (!Fused)
          CurrEntry->IsCheckedMultipleEntry = true;
      }
      // Add entries not handled as next entries, they are deferred
      for (const auto &Entry : Entries)
        if (!contains(IndependentGroups, Entry))
          NextEntries.insert(Entry);
      // The multiple has been created, we can decide how to implement it
      if (Multiple->InnerMap.size() >= RelooperMultipleSwitchThreshold) {
        Multiple->UseSwitch = true;
        Multiple->Breaks++; // switch captures breaks
      }
      return Multiple;
    }

    // Main function.
    // Process a set of blocks with specified entries, returns a shape
    // The Make* functions receive a NextEntries. If they fill it with data,
    // those are the entries for the ->Next block on them, and the blocks
    // are what remains in Blocks (which Make* modify). In this way
    // we avoid recursing on Next (imagine a long chain of Simples, if we
    // recursed we could blow the stack).
    Shape *Process(BlockSet &Blocks, BlockSet &InitialEntries, Shape *Prev) {
      BlockSet *Entries = &InitialEntries;
      BlockSet TempEntries[2];
      int CurrTempIndex = 0;
      BlockSet *NextEntries;
      Shape *Ret = nullptr;

      auto Make = [&](Shape *Temp) {
        if (Prev)
          Prev->Next = Temp;
        if (!Ret)
          Ret = Temp;
        Prev = Temp;
        Entries = NextEntries;
      };

      while (1) {
        CurrTempIndex = 1 - CurrTempIndex;
        NextEntries = &TempEntries[CurrTempIndex];
        NextEntries->clear();

        if (Entries->empty())
          return Ret;
        if (Entries->size() == 1) {
          Block *Curr = *(Entries->begin());
          if (Curr->BranchesIn.empty()) {
            // One entry, no looping ==> Simple
            Make(MakeSimple(Blocks, Curr, *NextEntries));
            if (NextEntries->empty())
              return Ret;
            continue;
          }
          // One entry, looping ==> Loop
          Make(MakeLoop(Blocks, *Entries, *NextEntries));
          if (NextEntries->empty())
            return Ret;
          continue;
        }

        // More than one entry, try to eliminate through a Multiple groups of
        // independent blocks from an entry/ies. It is important to remove
        // through multiples as opposed to looping since the former is more
        // performant.
        BlockBlockSetMap IndependentGroups;
        FindIndependentGroups(*Entries, IndependentGroups);

        if (!IndependentGroups.empty()) {
          // We can handle a group in a multiple if its entry cannot be reached
          // by another group.
          // Note that it might be reachable by itself - a loop. But that is
          // fine, we will create a loop inside the multiple block (which
          // is the performant order to do it).
          for (auto iter = IndependentGroups.begin();
               iter != IndependentGroups.end();) {
            Block *Entry = iter->first;
            BlockSet &Group = iter->second;
            auto curr = iter++; // iterate carefully, we may delete
            for (BlockSet::iterator iterBranch = Entry->BranchesIn.begin();
                 iterBranch != Entry->BranchesIn.end(); iterBranch++) {
              Block *Origin = *iterBranch;
              if (!contains(Group, Origin)) {
                // Reached from outside the group, so we cannot handle this
                IndependentGroups.erase(curr);
                break;
              }
            }
          }

          // As an optimization, if we have 2 independent groups, and one is a
          // small dead end, we can handle only that dead end.
          // The other then becomes a Next - without nesting in the code and
          // recursion in the analysis.
          // TODO: if the larger is the only dead end, handle that too
          // TODO: handle >2 groups
          // TODO: handle not just dead ends, but also that do not branch to the
          // NextEntries. However, must be careful there since we create a
          // Next, and that Next can prevent eliminating a break (since we no
          // longer naturally reach the same place), which may necessitate a
          // one-time loop, which makes the unnesting pointless.
          if (IndependentGroups.size() == 2) {
            // Find the smaller one
            auto iter = IndependentGroups.begin();
            Block *SmallEntry = iter->first;
            auto SmallSize = iter->second.size();
            iter++;
            Block *LargeEntry = iter->first;
            auto LargeSize = iter->second.size();
            if (SmallSize != LargeSize) { // ignore the case where they are
                                          // identical - keep things symmetrical
                                          // there
              if (SmallSize > LargeSize) {
                Block *Temp = SmallEntry;
                SmallEntry = LargeEntry;
                LargeEntry = Temp; // Note: we did not flip the Sizes too, they
                                   // are now invalid. TODO: use the smaller
                                   // size as a limit?
              }
              // Check if dead end
              bool DeadEnd = true;
              BlockSet &SmallGroup = IndependentGroups[SmallEntry];
              for (const auto &Curr : SmallGroup) {
                for (const auto &iter : Curr->BranchesOut) {
                  Block *Target = iter.first;
                  if (!contains(SmallGroup, Target)) {
                    DeadEnd = false;
                    break;
                  }
                }
                if (!DeadEnd)
                  break;
              }
              if (DeadEnd)
                IndependentGroups.erase(LargeEntry);
            }
          }

          if (!IndependentGroups.empty())
            // Some groups removable ==> Multiple
            Make(MakeMultiple(Blocks, *Entries, IndependentGroups, Prev,
                              *NextEntries));
            if (NextEntries->empty())
              return Ret;
            continue;
        }
        // No independent groups, must be loopable ==> Loop
        Make(MakeLoop(Blocks, *Entries, *NextEntries));
        if (NextEntries->empty())
          return Ret;
        continue;
      }
    }
  };

  // Main

  BlockSet AllBlocks;
  for (const auto &Curr : Pre.Live) {
    AllBlocks.insert(Curr);
  }

  BlockSet Entries;
  Entries.insert(Entry);
  Root = Analyzer(this).Process(AllBlocks, Entries, nullptr);
  assert(Root);

  ///
  /// Relooper post-optimizer
  ///
  struct PostOptimizer {
    RelooperAlgorithm *Parent;
    std::stack<Shape *> LoopStack;

    PostOptimizer(RelooperAlgorithm *ParentInit) : Parent(ParentInit) {}

    void ShapeSwitch(Shape* var,
                     std::function<void (SimpleShape*)> simple,
                     std::function<void (MultipleShape*)> multiple,
                     std::function<void (LoopShape*)> loop) {
      switch (var->getKind()) {
        case Shape::SK_Simple: {
          simple(cast<SimpleShape>(var));
          break;
        }
        case Shape::SK_Multiple: {
          multiple(cast<MultipleShape>(var));
          break;
        }
        case Shape::SK_Loop: {
          loop(cast<LoopShape>(var));
          break;
        }
        default: llvm_unreachable("invalid shape");
      }
    }

    // Find the blocks that natural control flow can get us directly to, or
    // through a multiple that we ignore
    void FollowNaturalFlow(Shape *S, BlockSet &Out) {
      ShapeSwitch(S, [&](SimpleShape* Simple) {
        Out.insert(Simple->Inner);
      }, [&](MultipleShape* Multiple) {
        for (const auto &iter : Multiple->InnerMap) {
          FollowNaturalFlow(iter.second, Out);
        }
        FollowNaturalFlow(Multiple->Next, Out);
      }, [&](LoopShape* Loop) {
        FollowNaturalFlow(Loop->Inner, Out);
      });
    }

    void FindNaturals(Shape *Root, Shape *Otherwise = nullptr) {
      if (Root->Next) {
        Root->Natural = Root->Next;
        FindNaturals(Root->Next, Otherwise);
      } else {
        Root->Natural = Otherwise;
      }

      ShapeSwitch(Root, [](SimpleShape* Simple) {
      }, [&](MultipleShape* Multiple) {
        for (const auto &iter : Multiple->InnerMap) {
          FindNaturals(iter.second, Root->Natural);
        }
      }, [&](LoopShape* Loop){
        FindNaturals(Loop->Inner, Loop->Inner);
      });
    }

    // Remove unneeded breaks and continues.
    // A flow operation is trivially unneeded if the shape we naturally get to
    // by normal code execution is the same as the flow forces us to.
    void RemoveUnneededFlows(Shape *Root, Shape *Natural = nullptr,
                             LoopShape *LastLoop = nullptr,
                             unsigned Depth = 0) {
      BlockSet NaturalBlocks;
      FollowNaturalFlow(Natural, NaturalBlocks);
      Shape *Next = Root;
      while (Next) {
        Root = Next;
        Next = nullptr;
        ShapeSwitch(
            Root,
            [&](SimpleShape* Simple) {
              if (Simple->Inner->BranchVar)
                LastLoop =
                    nullptr; // a switch clears out the loop (TODO: only for
                             // breaks, not continue)

              if (Simple->Next) {
                if (!Simple->Inner->BranchVar &&
                    Simple->Inner->ProcessedBranchesOut.size() == 2 &&
                    Depth < RelooperNestingLimit) {
                  // If there is a next block, we already know at Simple
                  // creation time to make direct branches, and we can do
                  // nothing more in general. But, we try to optimize the
                  // case of a break and a direct: This would normally be
                  //   if (break?) { break; } ..
                  // but if we make sure to nest the else, we can save the
                  // break,
                  //   if (!break?) { .. }
                  // This is also better because the more canonical nested
                  // form is easier to further optimize later. The
                  // downside is more nesting, which adds to size in builds with
                  // whitespace.
                  // Note that we avoid switches, as it complicates control flow
                  // and is not relevant for the common case we optimize here.
                  bool Found = false;
                  bool Abort = false;
                  for (const auto &iter : Simple->Inner->ProcessedBranchesOut) {
                    Block *Target = iter.first;
                    Branch *Details = iter.second.get();
                    if (Details->Type == Branch::Break) {
                      Found = true;
                      if (!contains(NaturalBlocks, Target))
                        Abort = true;
                    } else if (Details->Type != Branch::Direct)
                      Abort = true;
                  }
                  if (Found && !Abort) {
                    for (const auto &iter : Simple->Inner->ProcessedBranchesOut) {
                      Branch *Details = iter.second.get();
                      if (Details->Type == Branch::Break) {
                        Details->Type = Branch::Direct;
                        if (MultipleShape *Multiple =
                                dyn_cast<MultipleShape>(Details->Ancestor))
                          Multiple->Breaks--;
                      } else {
                        assert(Details->Type == Branch::Direct);
                        Details->Type = Branch::Nested;
                      }
                    }
                  }
                  Depth++; // this optimization increases depth, for us and all
                           // our next chain (i.e., until this call returns)
                }
                Next = Simple->Next;
              } else {
                // If there is no next then Natural is where we will
                // go to by doing nothing, so we can potentially optimize some
                // branches to direct.
                for (const auto &iter : Simple->Inner->ProcessedBranchesOut) {
                  Block *Target = iter.first;
                  Branch *Details = iter.second.get();
                  if (Details->Type != Branch::Direct &&
                      contains(NaturalBlocks,
                               Target)) { // note: cannot handle split blocks
                    Details->Type = Branch::Direct;
                    if (MultipleShape *Multiple =
                            dyn_cast<MultipleShape>(Details->Ancestor))
                      Multiple->Breaks--;
                  } else if (Details->Type == Branch::Break && LastLoop &&
                             LastLoop->Natural == Details->Ancestor->Natural) {
                    // it is important to simplify breaks, as simpler breaks
                    // enable other optimizations
                    Details->Labeled = false;
                    if (MultipleShape *Multiple =
                            dyn_cast<MultipleShape>(Details->Ancestor))
                      Multiple->Breaks--;
                  }
                }
              }
            }, [&](MultipleShape* Multiple)
            {
              for (const auto &iter : Multiple->InnerMap) {
                RemoveUnneededFlows(iter.second, Multiple->Next,
                                    Multiple->Breaks ? nullptr : LastLoop,
                                    Depth + 1);
              }
              Next = Multiple->Next;
            }, [&](LoopShape* Loop)
            {
              RemoveUnneededFlows(Loop->Inner, Loop->Inner, Loop, Depth + 1);
              Next = Loop->Next;
            });
      }
    }

    // After we know which loops exist, we can calculate which need to be
    // labeled
    void FindLabeledLoops(Shape *Root) {
      Shape *Next = Root;
      while (Next) {
        Root = Next;
        Next = nullptr;

        ShapeSwitch(
            Root,
            [&](SimpleShape *Simple) {
          MultipleShape *Fused = dyn_cast<MultipleShape>(Root->Next);
          // If we are fusing a Multiple with a loop into this Simple, then
          // visit it now
          if (Fused && Fused->Breaks)
            LoopStack.push(Fused);
          if (Simple->Inner->BranchVar)
            LoopStack.push(nullptr); // a switch means breaks are now useless,
                                     // push a dummy
          if (Fused) {
            if (Fused->UseSwitch)
              LoopStack.push(nullptr); // a switch means breaks are now
                                       // useless, push a dummy
            for (const auto &iter : Fused->InnerMap) {
              FindLabeledLoops(iter.second);
            }
          }
          for (const auto &iter : Simple->Inner->ProcessedBranchesOut) {
            Branch *Details = iter.second.get();
            if (Details->Type == Branch::Break ||
                Details->Type == Branch::Continue) {
              assert(!LoopStack.empty());
              if (Details->Ancestor != LoopStack.top() && Details->Labeled) {
                if (MultipleShape *Multiple =
                        dyn_cast<MultipleShape>(Details->Ancestor)) {
                  Multiple->Labeled = true;
                } else {
                  LoopShape *Loop = cast<LoopShape>(Details->Ancestor);
                  Loop->Labeled = true;
                }
              } else {
                Details->Labeled = false;
              }
            }
            if (Fused && Fused->UseSwitch)
              LoopStack.pop();
            if (Simple->Inner->BranchVar)
              LoopStack.pop();
            if (Fused && Fused->Breaks)
              LoopStack.pop();
            if (Fused)
              Next = Fused->Next;
            else
              Next = Root->Next;
          }
          }
          , [&](MultipleShape* Multiple) {
            if (Multiple->Breaks)
              LoopStack.push(Multiple);
            for (const auto &iter : Multiple->InnerMap)
              FindLabeledLoops(iter.second);
            if (Multiple->Breaks)
              LoopStack.pop();
            Next = Root->Next;
          }
          , [&](LoopShape* Loop) {
            LoopStack.push(Loop);
            FindLabeledLoops(Loop->Inner);
            LoopStack.pop();
            Next = Root->Next;
          });
      }
    }

    void Process(Shape * Root) {
      FindNaturals(Root);
      RemoveUnneededFlows(Root);
      FindLabeledLoops(Root);
    }
  };

  PostOptimizer(this).Process(Root);
}
