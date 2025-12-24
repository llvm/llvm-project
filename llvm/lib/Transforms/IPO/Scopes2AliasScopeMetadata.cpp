//===- Scopes2AliasScopeMetadata.h - Add !alias.scope metadata. -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Scopes2AliasScopeMetadata.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ModRef.h"
#include <algorithm>
#include <iterator>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "scopes2aliasscopemetadata"

namespace {
static constexpr auto ScopeNumberDelimiter = '_';
static constexpr auto NoaliasIDDelimiter = '#';

using GenMapType = DenseMap<BasicBlock *, DenseMap<Value *, MDNode *>>;
} // namespace

static void processLoadInstruction(Instruction *LI) {
  auto *MD = LI->getMetadata(llvm::LLVMContext::MD_scope);
  std::queue<Value *> Worklist;
  std::set<Value *> Visited;

  Worklist.push(LI);
  Visited.insert(LI);

  while (!Worklist.empty()) {
    Value *Current = Worklist.front();
    Worklist.pop();

    for (User *U : Current->users()) {
      if (!Visited.insert(U).second)
        continue;

      auto *GEP = dyn_cast<GetElementPtrInst>(U);
      if (!GEP)
        continue;
      auto *PrevMD = GEP->getMetadata(llvm::LLVMContext::MD_scope);
      if (PrevMD == MD)
        continue;

      GEP->setMetadata(llvm::LLVMContext::MD_scope, MD);
      Worklist.push(GEP);
    }
  }
}

bool Scopes2AliasScopeMetadataPass::propagateAliasScopes(Function &F) {
  bool Changed = false;
  LLVMContext &Ctx = F.getContext();
  MDBuilder MDB(Ctx);
  std::string DomainName = (Twine(F.getName()) + "Domain").str();
  MDNode *NewDomain = MDB.createAnonymousAliasScopeDomain(DomainName);
  for (auto &I : instructions(F)) {
    auto *MD = I.getMetadata(llvm::LLVMContext::MD_scope);
    if (!MD)
      continue;

    // FIXME: Allocas require new kind of metadata to prevent interference
    if (dyn_cast<AllocaInst>(&I))
      continue;
    for (auto *U : I.users()) {
      if (dyn_cast<StoreInst>(U) || dyn_cast<LoadInst>(U)) {
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getOperand(1) != &I)
            continue;
        auto *UserInst = dyn_cast<Instruction>(U);
        const auto *MDStr = dyn_cast<MDString>(MD->getOperand(0).get());
        assert(MDStr && "Scope metadata is not a string");
        auto *MDSet = cast<MDNode>(MD->getOperand(1).get());
        auto *IntMD =
            mdconst::extract_or_null<ConstantInt>(MDSet->operands().back());
        assert(IntMD &&
               "Metadata format is invalid: constant integer expected");
        auto IntMDVal = IntMD->getZExtValue();
        std::string ScopeName = Twine(MDStr->getString()).str();

        if (IntMDVal) {
          ScopeName.append((Twine(NoaliasIDDelimiter) + Twine(IntMDVal)).str());
          NoAliasInstrs.emplace_back(UserInst, ScopeName);
        }

        Changed |= true;
        auto It = ScopeMap.find(ScopeName);
        if (It != ScopeMap.end()) {
          UserInst->setMetadata(LLVMContext::MD_alias_scope, It->second);
          continue;
        }
        MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, ScopeName);
        auto *ScopeMDNode = MDNode::get(Ctx, NewScope);
        UserInst->setMetadata(LLVMContext::MD_alias_scope, ScopeMDNode);
        ScopeMap.emplace(ScopeName, ScopeMDNode);
      }
    }
  }

  return Changed;
}

static std::string getPureScopeNumber(StringRef MetadataStr) {
  assert(MetadataStr.contains(ScopeNumberDelimiter) &&
         "Metadata format is invalid");

  auto ScopeNumberPos = MetadataStr.rfind(ScopeNumberDelimiter) + 1;
  auto NoaliasIDPos = MetadataStr.rfind(NoaliasIDDelimiter);

  auto Count = (NoaliasIDPos == StringRef::npos)
                   ? (MetadataStr.size() - ScopeNumberPos)
                   : (NoaliasIDPos - ScopeNumberPos);
  return MetadataStr.substr(ScopeNumberPos, Count).str();
}

static bool areScopesNotAlias(APInt FirstScopeVal, APInt SecondScopeVal) {
  auto FirstMSB = FirstScopeVal.getBitWidth();
  auto SecondMSB = SecondScopeVal.getBitWidth();

  if (FirstMSB > SecondMSB)
    return false;

  if (FirstMSB == SecondMSB) {
    if (FirstScopeVal == SecondScopeVal)
      return true;
    return false;
  }

  FirstScopeVal = FirstScopeVal.zext(SecondMSB);

  if (FirstScopeVal == SecondScopeVal)
    return true;

  auto ShiftedSecond = (SecondScopeVal.lshr(SecondMSB - FirstMSB - 1));
  auto ShiftedFirst = FirstScopeVal << 1;

  auto Check = ShiftedSecond ^ ShiftedFirst;
  return Check.isZero();
}

void Scopes2AliasScopeMetadataPass::setNoAliasMetadata() const {
  for (auto &&[I, FirstScopeName] : NoAliasInstrs) {
    for (auto &&[SecondScopeName, MDNode] : ScopeMap) {
      if (FirstScopeName == SecondScopeName)
        continue;

      auto FirstScopeNumber = getPureScopeNumber(FirstScopeName);
      auto SecondScopeNumber = getPureScopeNumber(SecondScopeName);

      APInt NoAlaisScopeVal =
          APInt(FirstScopeNumber.length(), FirstScopeNumber, /*Radix*/ 2);
      APInt SecondScopeVal =
          APInt(SecondScopeNumber.length(), SecondScopeNumber, /*Radix*/ 2);
      if (areScopesNotAlias(std::move(NoAlaisScopeVal),
                            std::move(SecondScopeVal)))
        I->setMetadata(LLVMContext::MD_noalias, MDNode);
    }
  }
}

static void collectScopedAllocas(SmallSet<AllocaInst *, 8> &Vars,
                                 BasicBlock &EntryBB) {
  auto Allocas = llvm::transform(
      make_filter_range(EntryBB,
                        [](auto &&I) {
                          auto *AI = dyn_cast<AllocaInst>(&I);
                          return AI &&
                                 AI->getMetadata(llvm::LLVMContext::MD_scope);
                        }),
      std::inserter(Vars, Vars.begin()),
      [](auto &&I) { return cast<AllocaInst>(&I); });

  llvm::copy(Vars, Allocas);
}

static void initGenMap(GenMapType &GenMap,
                       const SmallSet<AllocaInst *, 8> &Vars,
                       BasicBlock &EntryBB) {
  for (auto *AI : Vars) {
    assert((AI->getMetadata(llvm::LLVMContext::MD_scope)) &&
           "Instruction must have MD_scope metadata");

    GenMap[&EntryBB][AI] = AI->getMetadata(llvm::LLVMContext::MD_scope);
  }
}

static void collectAllocaSts(SmallVector<StoreInst *, 8> &AllocaSts,
                             BasicBlock &BB,
                             const SmallSet<AllocaInst *, 8> &Vars) {
  for (auto &I : BB) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;

    auto *AI = dyn_cast<AllocaInst>(SI->getPointerOperand());
    if (!AI)
      continue;

    if (!Vars.contains(AI))
      continue;

    auto *MD = AI->getMetadata(llvm::LLVMContext::MD_scope);

    auto *MDSet = cast<MDNode>(MD->getOperand(1).get());
    auto *IntMD =
        mdconst::extract_or_null<ConstantInt>(MDSet->operands().back());
    assert(IntMD && "Metadata format is invalid: constant integer expected");
    auto IntMDVal = IntMD->getZExtValue();
    if (IntMDVal)
      continue;

    AllocaSts.push_back(SI);
  }
}

static MDNode *getSrcScopeMD(Value *V, BasicBlock *BB) {
  if (auto *AI = dyn_cast<Instruction>(V))
    return AI->getMetadata(llvm::LLVMContext::MD_scope);

  return nullptr;
}

template <typename ItTy>
static bool promoteScope(ItTy Begin, ItTy End, AllocaInst *AI,
                         MDNode *MDScope) {
  assert(AI && "Non-null alloca pointer expected");
  auto LoadsToProcess =
      make_filter_range(make_range(Begin, End), [AI, MDScope](auto &&I) {
        auto *LI = dyn_cast<LoadInst>(&I);
        return LI && (LI->getPointerOperand() == AI) &&
               (LI->getMetadata(llvm::LLVMContext::MD_scope) != MDScope);
      });
  if (!range_size(LoadsToProcess))
    return false;

  for (auto &&I : LoadsToProcess) {
    auto *LI = cast<LoadInst>(&I);
    LI->setMetadata(llvm::LLVMContext::MD_scope, MDScope);
    processLoadInstruction(LI);
  }
  return true;
}

static bool promotePHIs(BasicBlock &BB, GenMapType &GenMap) {
  bool Changed = false;
  for (auto &&PHIIt : BB.phis()) {
    SmallVector<MDNode *> MOP;
    for (auto &&V : PHIIt.incoming_values()) {
      auto *I = dyn_cast<Instruction>(&*V);
      if (!I)
        break;
      if (auto *MD = I->getMetadata(llvm::LLVMContext::MD_scope))
        MOP.emplace_back(MD);
    }

    bool AllEq = ((std::adjacent_find(MOP.begin(), MOP.end(),
                                      [](const auto &Fst, const auto &Sec) {
                                        return Fst != Sec;
                                      })) == MOP.end());
    if (MOP.size() == PHIIt.getNumIncomingValues() && AllEq)
      PHIIt.setMetadata(llvm::LLVMContext::MD_scope, MOP.front());
  }

  return Changed;
}

template <typename IterTy>
static IterTy getNextStoreInstToAIOrEnd(AllocaInst *AI, BasicBlock *BB,
                                        IterTy StartIt) {
  auto *MD = AI->getMetadata(llvm::LLVMContext::MD_scope);

  auto *MDSet = cast<MDNode>(MD->getOperand(1).get());
  auto *IntMD = mdconst::extract_or_null<ConstantInt>(MDSet->operands().back());
  assert(IntMD && "Metadata format is invalid: constant integer expected");
  auto IntMDVal = IntMD->getZExtValue();
  if (IntMDVal)
    return BB->end();

  return std::find_if(StartIt, BB->end(), [AI](auto &&I) {
    auto *OtherSI = dyn_cast<StoreInst>(&I);
    return (OtherSI && OtherSI->getPointerOperand() == AI);
  });
}

static bool meetAllocasForBB(BasicBlock *BB,
                             const SmallSet<AllocaInst *, 8> &Vars,
                             GenMapType &GenMap) {
  bool Changed = false;
  if (BB->isEntryBlock())
    return Changed;

  for (auto *V : Vars) {
    SmallVector<MDNode *> MOP;
    for (auto *PredBB :
         make_filter_range(predecessors(BB), [&GenMap](auto *PredBB) {
           return GenMap.contains(PredBB);
         }))
      MOP.emplace_back(GenMap[PredBB][V]);

    bool AllEq = ((std::adjacent_find(MOP.begin(), MOP.end(),
                                      [](const auto &Fst, const auto &Sec) {
                                        return Fst != Sec;
                                      })) == MOP.end());

    // TODO: transfer function should be more intelligent
    GenMap[BB][V] = AllEq ? MOP.front() : nullptr;
  }

  Changed |= promotePHIs(*BB, GenMap);
  for (auto *AI : Vars) {
    auto &&NextSI =
        getNextStoreInstToAIOrEnd(dyn_cast<AllocaInst>(AI), BB, BB->begin());
    Changed |= promoteScope(BB->begin(), NextSI, AI, GenMap[BB][AI]);
  }
  return Changed;
}

static bool
propagateScopesThroughTheCFG(Function &F, GenMapType &GenMap,
                             const SmallSet<AllocaInst *, 8> &Vars) {
  bool Changed = false;
  bool FixedPointReached = false;

  ReversePostOrderTraversal<Function *> RPOT(&F);
  while (!FixedPointReached) {
    FixedPointReached = true;
    for (auto *BB : RPOT) {
      FixedPointReached = !meetAllocasForBB(BB, Vars, GenMap);
      SmallVector<StoreInst *, 8> AllocaSts;
      collectAllocaSts(AllocaSts, *BB, Vars);

      for (auto *SI : AllocaSts) {
        auto *AI = cast<AllocaInst>(SI->getPointerOperand());
        auto *AISrc = SI->getValueOperand();

        auto *MDScope = getSrcScopeMD(AISrc, BB);
        if (!MDScope)
          continue;
        auto *Prev = GenMap[BB][AI];
        if (Prev == MDScope)
          continue;
        GenMap[BB][AI] = MDScope;
        FixedPointReached = false;
        auto NextIt = std::next(SI->getIterator());

        auto &&NextSI = getNextStoreInstToAIOrEnd(AI, BB, NextIt);
        FixedPointReached |= !promoteScope(NextIt, NextSI, AI, MDScope);
        Changed |= FixedPointReached;
      }
    }
  }
  return Changed;
}

static bool propagateBasedOnAllocas(Function &F,
                                    SmallSet<AllocaInst *, 8> &Vars) {
  GenMapType GenMap;

  assert(!F.empty() && "Function body expected");
  auto &EntryBB = F.front();
  initGenMap(GenMap, Vars, EntryBB);
  for (auto *V : Vars)
    promoteScope(EntryBB.begin(), EntryBB.end(), V, GenMap[&EntryBB][V]);

  auto Changed = propagateScopesThroughTheCFG(F, GenMap, Vars);
  return Changed;
}

static bool isFunctionBasiclyAllowsAliasScopePropagation(const Function &F) {
  if (!F.size())
    return false;

  // Currently we disable this pass if the frontend has already annotated
  // instructions with llvm::LLVMContext::MD_alias_scope metadata. Otherwise, we
  // may violate language-specific semantics (e.g., aliasing guarantees), as
  // this pass could reorder memory accesses in ways that conflict with these
  // annotations.
  if (any_of(instructions(F), [](const auto &I) {
        return I.getMetadata(llvm::LLVMContext::MD_alias_scope);
      }))
    return false;

  return true;
}

static bool hasDangerousAllocaUsers(const AllocaInst *AI) {
  for (auto *U : AI->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getValueOperand() == AI)
        return true;
      continue;
    }

    if (auto *II = dyn_cast<IntrinsicInst>(U)) {
      if (!II->isLifetimeStartOrEnd())
        return true;
      continue;
    }

    if (!isa<LoadInst>(U))
      return true;
  }
  return false;
}

static bool
areVarsAllowAliasScopePropagation(const SmallSet<AllocaInst *, 8> &Vars) {
  if (Vars.empty())
    return false;
  bool hasScopeMetadata = false;

  for (auto *AI : Vars) {
    hasScopeMetadata |=
        (AI->getMetadata(llvm::LLVMContext::MD_scope) != nullptr);

    // Currently we conservatively drop any cases with "escaped" allocas
    if (hasDangerousAllocaUsers(AI))
      return false;
  }

  return hasScopeMetadata;
}

bool Scopes2AliasScopeMetadataPass::runOnFunction(Function &F) {
  bool Changed = false;
  if (!isFunctionBasiclyAllowsAliasScopePropagation(F))
    return Changed;

  SmallSet<AllocaInst *, 8> Vars;
  auto &EntryBB = F.front();

  // TODO: support structures GEPs
  collectScopedAllocas(Vars, EntryBB);
  if (!areVarsAllowAliasScopePropagation(Vars))
    return Changed;

  Changed |= propagateBasedOnAllocas(F, Vars);
  Changed |= propagateAliasScopes(F);
  setNoAliasMetadata();

  return Changed;
}

bool Scopes2AliasScopeMetadataPass::convertScopeInfo2AliasScopeMetadata(
    Module &M) {
  bool Changed = false;

  for (auto &F : M)
    Changed |= runOnFunction(F);

  return Changed;
}

PreservedAnalyses
Scopes2AliasScopeMetadataPass::run(Module &M, ModuleAnalysisManager &AM) {
  return convertScopeInfo2AliasScopeMetadata(M) ? PreservedAnalyses::none()
                                                : PreservedAnalyses::all();
}
