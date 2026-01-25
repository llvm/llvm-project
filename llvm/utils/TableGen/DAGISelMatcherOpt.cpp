//===- DAGISelMatcherOpt.cpp - Optimize a DAG Matcher ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DAG Matcher optimizer.
//
//===----------------------------------------------------------------------===//

#include "Basic/SDNodeProperties.h"
#include "Common/CodeGenDAGPatterns.h"
#include "DAGISelMatcher.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "isel-opt"

/// ContractNodes - Turn multiple matcher node patterns like 'MoveChild+Record'
/// into single compound nodes like RecordChild.
static void ContractNodes(MatcherList &ML, const CodeGenDAGPatterns &CGP) {
  auto P = ML.before_begin();
  auto I = std::next(P);

  while (I != ML.end()) {
    Matcher *N = *I;

    // If we have a scope node, walk down all of the children.
    if (auto *Scope = dyn_cast<ScopeMatcher>(N)) {
      for (unsigned i = 0, e = Scope->getNumChildren(); i != e; ++i)
        ContractNodes(Scope->getChild(i), CGP);
      return;
    }

    // If we found a movechild node with a node that comes in a 'foochild' form,
    // transform it.
    if (MoveChildMatcher *MC = dyn_cast<MoveChildMatcher>(N)) {
      Matcher *Next = *std::next(I);
      Matcher *New = nullptr;
      if (RecordMatcher *RM = dyn_cast<RecordMatcher>(Next))
        if (MC->getChildNo() < 8) // Only have RecordChild0...7
          New = new RecordChildMatcher(MC->getChildNo(), RM->getWhatFor(),
                                       RM->getResultNo());

      if (CheckTypeMatcher *CT = dyn_cast<CheckTypeMatcher>(Next))
        if (MC->getChildNo() < 8 && // Only have CheckChildType0...7
            CT->getResNo() == 0)    // CheckChildType checks res #0
          New = new CheckChildTypeMatcher(MC->getChildNo(), CT->getType());

      if (CheckSameMatcher *CS = dyn_cast<CheckSameMatcher>(Next))
        if (MC->getChildNo() < 4) // Only have CheckChildSame0...3
          New =
              new CheckChildSameMatcher(MC->getChildNo(), CS->getMatchNumber());

      if (CheckIntegerMatcher *CI = dyn_cast<CheckIntegerMatcher>(Next))
        if (MC->getChildNo() < 5) // Only have CheckChildInteger0...4
          New = new CheckChildIntegerMatcher(MC->getChildNo(), CI->getValue());

      if (auto *CCC = dyn_cast<CheckCondCodeMatcher>(Next))
        if (MC->getChildNo() == 2) // Only have CheckChild2CondCode
          New = new CheckChild2CondCodeMatcher(CCC->getCondCodeName());

      if (New) {
        // Erase the old node after the MoveChild.
        ML.erase_after(I);
        // Insert the new node before the MoveChild.
        I = ML.insert_after(P, New);
        continue;
      }
    }

    // Turn MoveParent->MoveChild into MoveSibling.
    if (isa<MoveParentMatcher>(N)) {
      auto J = std::next(I);
      if (auto *MC = dyn_cast<MoveChildMatcher>(*J)) {
        auto *MS = new MoveSiblingMatcher(MC->getChildNo());
        I = ML.insert_after(P, MS);
        // Erase the two old nodes.
        ML.erase_after(I, std::next(J));
        continue;
      }
    }

    // Uncontract MoveSibling if it will help form other child operations.
    if (auto *MS = dyn_cast<MoveSiblingMatcher>(N)) {
      auto J = std::next(I);
      if (auto *RM = dyn_cast<RecordMatcher>(*J)) {
        auto K = std::next(J);
        // Turn MoveSibling->Record->MoveParent into MoveParent->RecordChild.
        if (isa<MoveParentMatcher>(*K)) {
          if (MS->getSiblingNo() < 8) { // Only have RecordChild0...7
            auto *NewRCM = new RecordChildMatcher(
                MS->getSiblingNo(), RM->getWhatFor(), RM->getResultNo());
            I = ML.erase_after(P, K);
            ML.insert_after(I, NewRCM);
            continue;
          }
        }

        // Turn MoveSibling->Record->CheckType->MoveParent into
        // MoveParent->RecordChild->CheckChildType.
        if (auto *CT = dyn_cast<CheckTypeMatcher>(*K)) {
          auto L = std::next(K);
          if (isa<MoveParentMatcher>(*L)) {
            if (MS->getSiblingNo() < 8 && // Only have CheckChildType0...7
                CT->getResNo() == 0) {    // CheckChildType checks res #0
              auto *NewRCM = new RecordChildMatcher(
                  MS->getSiblingNo(), RM->getWhatFor(), RM->getResultNo());
              auto *NewCCT =
                  new CheckChildTypeMatcher(MS->getSiblingNo(), CT->getType());
              I = ML.erase_after(P, L);
              ML.insert_after(I, {NewRCM, NewCCT});
              continue;
            }
          }
        }
      }

      // Turn MoveSibling->CheckType->MoveParent into
      // MoveParent->CheckChildType.
      if (auto *CT = dyn_cast<CheckTypeMatcher>(*J)) {
        auto K = std::next(J);
        if (isa<MoveParentMatcher>(*K)) {
          if (MS->getSiblingNo() < 8 && // Only have CheckChildType0...7
              CT->getResNo() == 0) {    // CheckChildType checks res #0
            auto *NewCCT =
                new CheckChildTypeMatcher(MS->getSiblingNo(), CT->getType());
            I = ML.erase_after(P, K);
            ML.insert_after(I, NewCCT);
            continue;
          }
        }
      }

      // Turn MoveSibling->CheckInteger->MoveParent into
      // MoveParent->CheckChildInteger.
      if (auto *CI = dyn_cast<CheckIntegerMatcher>(*J)) {
        auto K = std::next(J);
        if (isa<MoveParentMatcher>(*K)) {
          if (MS->getSiblingNo() < 5) { // Only have CheckChildInteger0...4
            auto *NewCCI = new CheckChildIntegerMatcher(MS->getSiblingNo(),
                                                        CI->getValue());
            I = ML.erase_after(P, K);
            ML.insert_after(I, NewCCI);
            continue;
          }
        }

        // Turn MoveSibling->CheckInteger->CheckType->MoveParent into
        // MoveParent->CheckChildInteger->CheckType.
        if (auto *CT = dyn_cast<CheckTypeMatcher>(*K)) {
          auto L = std::next(K);
          if (isa<MoveParentMatcher>(*L)) {
            if (MS->getSiblingNo() < 5 && // Only have CheckChildInteger0...4
                CT->getResNo() == 0) {    // CheckChildType checks res #0
              auto *NewCCI = new CheckChildIntegerMatcher(MS->getSiblingNo(),
                                                          CI->getValue());
              auto *NewCCT =
                  new CheckChildTypeMatcher(MS->getSiblingNo(), CT->getType());
              I = ML.erase_after(P, L);
              ML.insert_after(I, {NewCCI, NewCCT});
              continue;
            }
          }
        }
      }

      // Turn MoveSibling->CheckCondCode->MoveParent into
      // MoveParent->CheckChild2CondCode.
      if (auto *CCC = dyn_cast<CheckCondCodeMatcher>(*J)) {
        auto K = std::next(J);
        if (isa<MoveParentMatcher>(*K)) {
          if (MS->getSiblingNo() == 2) { // Only have CheckChild2CondCode
            auto *NewCCCC =
                new CheckChild2CondCodeMatcher(CCC->getCondCodeName());
            I = ML.erase_after(P, K);
            ML.insert_after(I, NewCCCC);
            continue;
          }
        }
      }

      // Turn MoveSibling->CheckSame->MoveParent into
      // MoveParent->CheckChildSame.
      if (auto *CS = dyn_cast<CheckSameMatcher>(*J)) {
        auto K = std::next(J);
        if (isa<MoveParentMatcher>(*K)) {
          if (MS->getSiblingNo() < 4) { // Only have CheckChildSame0...3
            auto *NewCCS = new CheckChildSameMatcher(MS->getSiblingNo(),
                                                     CS->getMatchNumber());
            I = ML.erase_after(P, K);
            ML.insert_after(I, NewCCS);
            continue;
          }
        }

        // Turn MoveSibling->CheckSame->CheckType->MoveParent into
        // MoveParent->CheckChildSame->CheckChildType.
        if (auto *CT = dyn_cast<CheckTypeMatcher>(*K)) {
          auto L = std::next(K);
          if (isa<MoveParentMatcher>(*L)) {
            if (MS->getSiblingNo() < 4 && // Only have CheckChildSame0...3
                CT->getResNo() == 0) {    // CheckChildType checks res #0
              auto *NewCCS = new CheckChildSameMatcher(MS->getSiblingNo(),
                                                       CS->getMatchNumber());
              auto *NewCCT =
                  new CheckChildTypeMatcher(MS->getSiblingNo(), CT->getType());
              I = ML.erase_after(P, L);
              ML.insert_after(I, {NewCCS, NewCCT});
              continue;
            }
          }
        }
      }

      // Turn MoveSibling->MoveParent into MoveParent.
      if (isa<MoveParentMatcher>(*J)) {
        I = ML.erase_after(P, J);
        continue;
      }
    }

    // Zap movechild -> moveparent.
    if (isa<MoveChildMatcher>(N)) {
      auto J = std::next(I);
      if (isa<MoveParentMatcher>(*J)) {
        I = ML.erase_after(P, std::next(J));
        continue;
      }
    }

    // Turn EmitNode->CompleteMatch into MorphNodeTo if we can.
    if (EmitNodeMatcher *EN = dyn_cast<EmitNodeMatcher>(N)) {
      auto J = std::next(I);
      if (auto *CM = dyn_cast<CompleteMatchMatcher>(*J)) {
        // We can only use MorphNodeTo if the result values match up.
        unsigned RootResultFirst = EN->getFirstResultSlot();
        bool ResultsMatch = true;
        for (unsigned i = 0, e = CM->getNumResults(); i != e; ++i)
          if (CM->getResult(i) != RootResultFirst + i)
            ResultsMatch = false;

        // If the selected node defines a subset of the glue/chain results, we
        // can't use MorphNodeTo.  For example, we can't use MorphNodeTo if the
        // matched pattern has a chain but the root node doesn't.
        const PatternToMatch &Pattern = CM->getPattern();

        if (!EN->hasChain() &&
            Pattern.getSrcPattern().NodeHasProperty(SDNPHasChain, CGP))
          ResultsMatch = false;

        // If the matched node has glue and the output root doesn't, we can't
        // use MorphNodeTo.
        //
        // NOTE: Strictly speaking, we don't have to check for glue here
        // because the code in the pattern generator doesn't handle it right. We
        // do it anyway for thoroughness.
        if (!EN->hasOutGlue() &&
            Pattern.getSrcPattern().NodeHasProperty(SDNPOutGlue, CGP))
          ResultsMatch = false;

#if 0
        // If the root result node defines more results than the source root
        // node *and* has a chain or glue input, then we can't match it because
        // it would end up replacing the extra result with the chain/glue.
        if ((EN->hasGlue() || EN->hasChain()) &&
            EN->getNumNonChainGlueVTs() > ...need to get no results reliably...)
          ResultMatch = false;
#endif

        if (ResultsMatch) {
          ArrayRef<ValueTypeByHwMode> VTs = EN->getVTList();
          ArrayRef<unsigned> Operands = EN->getOperandList();
          auto *MNT = new MorphNodeToMatcher(
              EN->getInstruction(), VTs, Operands, EN->hasChain(),
              EN->hasInGlue(), EN->hasOutGlue(), EN->hasMemRefs(),
              EN->getNumFixedArityOperands(), Pattern);
          ML.erase_after(P, std::next(J));
          ML.insert_after(P, MNT);
          return;
        }
      }
    }

    // If we have a Record node followed by a CheckOpcode, invert the two nodes.
    // We prefer to do structural checks before type checks, as this opens
    // opportunities for factoring on targets like X86 where many operations are
    // valid on multiple types.
    if (isa<RecordMatcher>(N) && isa<CheckOpcodeMatcher>(*std::next(I))) {
      ML.splice_after(P, ML, I);
      // Restore I to the node after P.
      I = std::next(P);
      continue;
    }

    // Move to next node.
    P = I;
    ++I;
  }
}

/// FindNodeWithKind - Scan a series of matchers looking for a matcher with a
/// specified kind.  Return null if we didn't find one otherwise return the
/// matcher.
static std::pair<MatcherList::iterator, MatcherList::iterator>
FindNodeWithKind(MatcherList &ML, Matcher::KindTy Kind) {
  auto P = ML.before_begin();
  auto I = std::next(P);
  while (I != ML.end()) {
    if (I->getKind() == Kind)
      break;

    P = I;
    ++I;
  }

  return std::make_pair(P, I);
}

/// Return true if \p M is already the front, or if we can move \p M past
/// all of the nodes before \p M.
static bool canMoveToFront(const MatcherList &ML,
                           MatcherList::const_iterator M) {
  for (auto Other = ML.begin(); Other != ML.end(); ++Other) {
    if (M == Other)
      return true;

    // We have to be able to move this node across the Other node.
    if (!M->canMoveBeforeNode(*Other))
      return false;
  }

  llvm_unreachable("M not part of list?");
}

static void FactorNodes(MatcherList &ML);

/// Turn matches like this:
///   Scope
///     OPC_CheckType i32
///       ABC
///     OPC_CheckType i32
///       XYZ
/// into:
///   OPC_CheckType i32
///     Scope
///       ABC
///       XYZ
///
/// \p ML is a list that ends with a ScopeMatcher. \p Prev is the iterator
/// before the ScopeMatcher in \p ML.
static void FactorScope(MatcherList &ML, MatcherList::iterator Prev) {
  ScopeMatcher *Scope = cast<ScopeMatcher>(*std::next(Prev));

  SmallVectorImpl<MatcherList> &OptionsToMatch = Scope->getChildren();

  for (auto I = OptionsToMatch.begin(); I != OptionsToMatch.end();) {
    assert(!I->empty());
    // Factor the subexpression.
    FactorNodes(*I);

    // If the child is a ScopeMatcher we can just merge its contents.
    if (auto *SM = dyn_cast<ScopeMatcher>(I->front())) {
      // Move the MatcherList to a temporary so we can overwrite the location
      // in OptionsToMatch without deleting the ScopeMatcher.
      MatcherList Tmp = std::move(*I);
      SmallVectorImpl<MatcherList> &Children = SM->getChildren();
      *I++ = std::move(Children.front());
      I = OptionsToMatch.insert(I,
                                std::make_move_iterator(Children.begin() + 1),
                                std::make_move_iterator(Children.end()));
      I += Children.size() - 1;
    } else {
      ++I;
    }
  }

  // Loop over options to match, merging neighboring patterns with identical
  // starting nodes into a shared matcher.
  auto E = OptionsToMatch.end();
  for (auto I = OptionsToMatch.begin(); I != E; ++I) {
    // If there are no other matchers left, there's nothing to merge with.
    auto J = std::next(I);
    if (J == E)
      break;

    // Remember where we started. We'll use this to move non-equal elements.
    auto K = J;

    // Find the set of matchers that start with this node.
    Matcher *Optn = I->front();

    // See if the next option starts with the same matcher.  If the two
    // neighbors *do* start with the same matcher, we can factor the matcher out
    // of at least these two patterns.  See what the maximal set we can merge
    // together is.
    SmallVector<MatcherList, 8> EqualMatchers;
    EqualMatchers.push_back(std::move(*I));

    // Factor all of the known-equal matchers after this one into the same
    // group.
    while (J != E && J->front()->isEqual(Optn))
      EqualMatchers.push_back(std::move(*J++));

    // If we found a non-equal matcher, see if it is contradictory with the
    // current node.  If so, we know that the ordering relation between the
    // current sets of nodes and this node don't matter.  Look past it to see if
    // we can merge anything else into this matching group.
    while (J != E) {
      Matcher *ScanMatcher = J->front();

      // If we found an entry that matches out matcher, merge it into the set to
      // handle.
      if (Optn->isEqual(ScanMatcher)) {
        // It is equal after all, add the option to EqualMatchers.
        EqualMatchers.push_back(std::move(*J++));
        continue;
      }

      // If the option we're checking for contradicts the start of the list,
      // move it earlier in OptionsToMatch for the next iteration of the outer
      // loop. Then continue searching for equal or contradictory matchers.
      if (Optn->isContradictory(ScanMatcher)) {
        if (J != K)
          *K = std::move(*J);
        ++J;
        ++K;
        continue;
      }

      // If we're scanning for a simple node, see if it occurs later in the
      // sequence.  If so, and if we can move it up, it might be contradictory
      // or the same as what we're looking for.  If so, reorder it.
      if (Optn->isSimplePredicateOrRecordNode()) {
        auto [P, M2] = FindNodeWithKind(*J, Optn->getKind());
        if (M2 != J->end() && *M2 != ScanMatcher && canMoveToFront(*J, M2) &&
            (M2->isEqual(Optn) || M2->isContradictory(Optn))) {
          J->splice_after(J->before_begin(), *J, P);
          continue;
        }
      }

      // Otherwise, we don't know how to handle this entry, we have to bail.
      break;
    }

    if (J != E &&
        // Don't print if it's obvious nothing extract could be merged anyway.
        std::next(J) != E) {
      LLVM_DEBUG(
          errs() << "Couldn't merge this:\n"; I->print(errs(), indent(4));
          errs() << "into this:\n"; J->print(errs(), indent(4));
          std::next(J)->front()->printOne(errs());
          if (std::next(J, 2) != E) std::next(J, 2)->front()->printOne(errs());
          errs() << "\n");
    }

    // If we removed any equal matchers, we may need to slide the rest of the
    // elements down for the next iteration of the outer loop.
    if (J != K)
      E = std::move(J, E, K);

    // If we only found one option starting with this matcher, no factoring is
    // possible. Put the Matcher back in OptionsToMatch.
    if (EqualMatchers.size() == 1) {
      *I = std::move(EqualMatchers[0]);
      continue;
    }

    // Factor these checks by pulling the first node off each entry and
    // discarding it.  Take the first one off the first entry to reuse.
    MatcherList Shared;
    Shared.splice_after(Shared.before_begin(), EqualMatchers[0],
                        EqualMatchers[0].before_begin());
    Optn = EqualMatchers[0].empty() ? nullptr : EqualMatchers[0].front();

    // Remove and delete the first node from the other matchers we're factoring.
    for (unsigned i = 1, e = EqualMatchers.size(); i != e; ++i) {
      EqualMatchers[i].pop_front();
      Matcher *Tmp =
          EqualMatchers[i].empty() ? nullptr : EqualMatchers[i].front();
      assert(!Optn == !Tmp && "Expected all to be null if any are null");
    }

    if (!EqualMatchers[0].empty()) {
      Shared.insert_after(Shared.begin(),
                          new ScopeMatcher(std::move(EqualMatchers)));

      // Recursively factor the newly created node.
      FactorScope(Shared, Shared.begin());
    }

    // Put the new Matcher where we started in OptionsToMatch.
    *I = std::move(Shared);
  }

  // Trim the array to match the updated end.
  OptionsToMatch.erase(E, OptionsToMatch.end());

  // If we're down to a single pattern to match, then we don't need this scope
  // anymore.
  if (OptionsToMatch.size() == 1) {
    MatcherList Tmp = std::move(OptionsToMatch[0]);
    ML.erase_after(Prev);
    ML.splice_after(Prev, Tmp);
    return;
  }

  if (OptionsToMatch.empty()) {
    ML.erase_after(Prev);
    return;
  }

  // If our factoring failed (didn't achieve anything) see if we can simplify in
  // other ways.

  // Check to see if all of the leading entries are now opcode checks.  If so,
  // we can convert this Scope to be a OpcodeSwitch instead.
  bool AllOpcodeChecks = true, AllTypeChecks = true;
  for (MatcherList &Optn : OptionsToMatch) {
    // Check to see if this breaks a series of CheckOpcodeMatchers.
    if (AllOpcodeChecks && !isa<CheckOpcodeMatcher>(Optn.front())) {
#if 0
      if (i > 3) {
        errs() << "FAILING OPC #" << i << "\n";
        Optn->dump();
      }
#endif
      AllOpcodeChecks = false;
    }

    // Check to see if this breaks a series of CheckTypeMatcher's.
    if (AllTypeChecks) {
      auto [P, I] = FindNodeWithKind(Optn, Matcher::CheckType);
      auto *CTM =
          cast_or_null<CheckTypeMatcher>(I == Optn.end() ? nullptr : *I);
      if (!CTM || !CTM->getType().isSimple() ||
          // iPTR/cPTR checks could alias any other case without us knowing,
          // don't bother with them.
          CTM->getType().getSimple() == MVT::iPTR ||
          CTM->getType().getSimple() == MVT::cPTR ||
          // SwitchType only works for result #0.
          CTM->getResNo() != 0 ||
          // If the CheckType isn't at the start of the list, see if we can move
          // it there.
          !canMoveToFront(Optn, I)) {
#if 0
        if (i > 3 && AllTypeChecks) {
          errs() << "FAILING TYPE #" << i << "\n";
          Optn->dump(); }
#endif
        AllTypeChecks = false;
      }
    }
  }

  // If all the options are CheckOpcode's, we can form the SwitchOpcode, woot.
  if (AllOpcodeChecks) {
    StringSet<> Opcodes;
    SmallVector<std::pair<const SDNodeInfo *, MatcherList>, 8> Cases;
    for (MatcherList &Optn : OptionsToMatch) {
      CheckOpcodeMatcher *COM = cast<CheckOpcodeMatcher>(Optn.front());
      assert(Opcodes.insert(COM->getOpcode().getEnumName()).second &&
             "Duplicate opcodes not factored?");
      const SDNodeInfo &Opcode = COM->getOpcode();
      Optn.erase_after(Optn.before_begin());
      Cases.emplace_back(&Opcode, std::move(Optn));
    }

    ML.erase_after(Prev);
    ML.insert_after(Prev, new SwitchOpcodeMatcher(std::move(Cases)));
    return;
  }

  // If all the options are CheckType's, we can form the SwitchType, woot.
  if (AllTypeChecks) {
    DenseMap<unsigned, unsigned> TypeEntry;
    SmallVector<std::pair<MVT, MatcherList>, 8> Cases;
    for (MatcherList &Optn : OptionsToMatch) {
      auto [P, I] = FindNodeWithKind(Optn, Matcher::CheckType);
      assert(I != Optn.end() && isa<CheckTypeMatcher>(*I) &&
             "Unknown Matcher type");

      auto *CTM = cast<CheckTypeMatcher>(*I);
      MVT CTMTy = CTM->getType().getSimple();
      Optn.erase_after(P);

      unsigned &Entry = TypeEntry[CTMTy.SimpleTy];
      if (Entry != 0) {
        // If we have unfactored duplicate types, then we should factor them.
        MatcherList &PrevMatcher = Cases[Entry - 1].second;
        if (ScopeMatcher *SM = dyn_cast<ScopeMatcher>(PrevMatcher.front())) {
          SM->getChildren().push_back(std::move(Optn));
          continue;
        }

        SmallVector<MatcherList, 2> Entries;
        Entries.push_back(std::move(PrevMatcher));
        Entries.push_back(std::move(Optn));
        assert(Cases[Entry - 1].second.empty());
        Cases[Entry - 1].second.insert_after(
            Cases[Entry - 1].second.before_begin(),
            new ScopeMatcher(std::move(Entries)));
        continue;
      }

      Entry = Cases.size() + 1;
      Cases.emplace_back(CTMTy, std::move(Optn));
    }
    ML.erase_after(Prev);

    // Make sure we recursively factor any scopes we may have created.
    for (auto &M : Cases) {
      if (isa<ScopeMatcher>(M.second.front())) {
        FactorScope(M.second, M.second.before_begin());
        assert(!M.second.empty() && "empty matcher list");
      }
    }

    if (Cases.size() != 1) {
      ML.insert_after(Prev, new SwitchTypeMatcher(std::move(Cases)));
    } else {
      // If we factored and ended up with one case, insert a type check and
      // splice the rest.
      auto I = ML.insert_after(Prev, new CheckTypeMatcher(Cases[0].first, 0));
      ML.splice_after(I, Cases[0].second);
    }
    return;
  }
}

/// Search a ScopeMatcher to factor with FactorScope.
static void FactorNodes(MatcherList &ML) {
  auto P = ML.before_begin();
  auto I = std::next(P);

  while (I != ML.end()) {
    if (isa<ScopeMatcher>(*I))
      return FactorScope(ML, P);

    P = I;
    ++I;
  }
}

void llvm::OptimizeMatcher(MatcherList &ML, const CodeGenDAGPatterns &CGP) {
  ContractNodes(ML, CGP);
  FactorNodes(ML);
}
