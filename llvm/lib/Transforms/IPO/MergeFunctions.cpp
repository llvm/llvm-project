//===- MergeFunctions.cpp - Merge identical functions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass looks for equivalent functions that are mergable and folds them.
//
// A Function will not be analyzed if:
// * it is overridable at runtime (except for weak linkage), or
// * it is used by anything other than the callee parameter of a call/invoke
//
// A hash is computed from the function, based on its type and number of
// basic blocks.
//
// Once all hashes are computed, we perform an expensive equality comparison
// on each function pair. This takes n^2/2 comparisons per bucket, so it's
// important that the hash function be high quality. The equality comparison
// iterates through each instruction in each basic block.
//
// When a match is found, the functions are folded. We can only fold two
// functions when we know that the definition of one of them is not
// overridable.
// * fold a function marked internal by replacing all of its users.
// * fold extern or weak functions by replacing them with a global alias
//
//===----------------------------------------------------------------------===//
//
// Future work:
//
// * fold vector<T*>::push_back and vector<S*>::push_back.
//
// These two functions have different types, but in a way that doesn't matter
// to us. As long as we never see an S or T itself, using S* and S** is the
// same as using a T* and T**.
//
// * virtual functions.
//
// Many functions have their address taken by the virtual function table for
// the object they belong to. However, as long as it's only used for a lookup
// and call, this is irrelevant, and we'd like to fold such implementations.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mergefunc"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <vector>
using namespace llvm;

STATISTIC(NumFunctionsMerged, "Number of functions merged");
STATISTIC(NumMergeFails, "Number of identical function pairings not merged");

namespace {
  struct VISIBILITY_HIDDEN MergeFunctions : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    MergeFunctions() : ModulePass((intptr_t)&ID) {}

    bool runOnModule(Module &M);
  };
}

char MergeFunctions::ID = 0;
static RegisterPass<MergeFunctions>
X("mergefunc", "Merge Functions");

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

static unsigned long hash(const Function *F) {
  return F->size() ^ reinterpret_cast<unsigned long>(F->getType());
  //return F->size() ^ F->arg_size() ^ F->getReturnType();
}

static bool compare(const Value *V, const Value *U) {
  assert(!isa<BasicBlock>(V) && !isa<BasicBlock>(U) &&
         "Must not compare basic blocks.");

  assert(V->getType() == U->getType() &&
        "Two of the same operation have operands of different type.");

  // TODO: If the constant is an expression of F, we should accept that it's
  // equal to the same expression in terms of G.
  if (isa<Constant>(V))
    return V == U;

  // The caller has ensured that ValueMap[V] != U. Since Arguments are
  // pre-loaded into the ValueMap, and Instructions are added as we go, we know
  // that this can only be a mis-match.
  if (isa<Instruction>(V) || isa<Argument>(V))
    return false;

  if (isa<InlineAsm>(V) && isa<InlineAsm>(U)) {
    const InlineAsm *IAF = cast<InlineAsm>(V);
    const InlineAsm *IAG = cast<InlineAsm>(U);
    return IAF->getAsmString() == IAG->getAsmString() &&
           IAF->getConstraintString() == IAG->getConstraintString();
  }

  return false;
}

static bool equals(const BasicBlock *BB1, const BasicBlock *BB2,
                   DenseMap<const Value *, const Value *> &ValueMap,
                   DenseMap<const Value *, const Value *> &SpeculationMap) {
  // Specutively add it anyways. If it's false, we'll notice a difference later, and
  // this won't matter.
  ValueMap[BB1] = BB2;

  BasicBlock::const_iterator FI = BB1->begin(), FE = BB1->end();
  BasicBlock::const_iterator GI = BB2->begin(), GE = BB2->end();

  do {
    if (!FI->isSameOperationAs(const_cast<Instruction *>(&*GI)))
      return false;

    if (FI->getNumOperands() != GI->getNumOperands())
      return false;

    if (ValueMap[FI] == GI) {
      ++FI, ++GI;
      continue;
    }

    if (ValueMap[FI] != NULL)
      return false;

    for (unsigned i = 0, e = FI->getNumOperands(); i != e; ++i) {
      Value *OpF = FI->getOperand(i);
      Value *OpG = GI->getOperand(i);

      if (ValueMap[OpF] == OpG)
        continue;

      if (ValueMap[OpF] != NULL)
        return false;

      assert(OpF->getType() == OpG->getType() &&
             "Two of the same operation has operands of different type.");

      if (OpF->getValueID() != OpG->getValueID())
        return false;

      if (isa<PHINode>(FI)) {
        if (SpeculationMap[OpF] == NULL)
          SpeculationMap[OpF] = OpG;
        else if (SpeculationMap[OpF] != OpG)
          return false;
        continue;
      } else if (isa<BasicBlock>(OpF)) {
        assert(isa<TerminatorInst>(FI) &&
               "BasicBlock referenced by non-Terminator non-PHI");
        // This call changes the ValueMap, hence we can't use
        // Value *& = ValueMap[...]
        if (!equals(cast<BasicBlock>(OpF), cast<BasicBlock>(OpG), ValueMap,
                    SpeculationMap))
          return false;
      } else {
        if (!compare(OpF, OpG))
          return false;
      }

      ValueMap[OpF] = OpG;
    }

    ValueMap[FI] = GI;
    ++FI, ++GI;
  } while (FI != FE && GI != GE);

  return FI == FE && GI == GE;
}

static bool equals(const Function *F, const Function *G) {
  // We need to recheck everything, but check the things that weren't included
  // in the hash first.

  if (F->getAttributes() != G->getAttributes())
    return false;

  if (F->hasGC() != G->hasGC())
    return false;

  if (F->hasGC() && F->getGC() != G->getGC())
    return false;

  if (F->hasSection() != G->hasSection())
    return false;

  if (F->hasSection() && F->getSection() != G->getSection())
    return false;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (F->getCallingConv() != G->getCallingConv())
    return false;

  // TODO: We want to permit cases where two functions take T* and S* but
  // only load or store them into T** and S**.
  if (F->getType() != G->getType())
    return false;

  DenseMap<const Value *, const Value *> ValueMap;
  DenseMap<const Value *, const Value *> SpeculationMap;
  ValueMap[F] = G;

  assert(F->arg_size() == G->arg_size() &&
         "Identical functions have a different number of args.");

  for (Function::const_arg_iterator fi = F->arg_begin(), gi = G->arg_begin(),
         fe = F->arg_end(); fi != fe; ++fi, ++gi)
    ValueMap[fi] = gi;

  if (!equals(&F->getEntryBlock(), &G->getEntryBlock(), ValueMap,
              SpeculationMap))
    return false;

  for (DenseMap<const Value *, const Value *>::iterator
         I = SpeculationMap.begin(), E = SpeculationMap.end(); I != E; ++I) {
    if (ValueMap[I->first] != I->second)
      return false;
  }

  return true;
}

static bool fold(std::vector<Function *> &FnVec, unsigned i, unsigned j) {
  if (FnVec[i]->mayBeOverridden() && !FnVec[j]->mayBeOverridden())
    std::swap(FnVec[i], FnVec[j]);

  Function *F = FnVec[i];
  Function *G = FnVec[j];

  if (!F->mayBeOverridden()) {
    if (G->hasLocalLinkage()) {
      F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
      G->replaceAllUsesWith(F);
      G->eraseFromParent();
      ++NumFunctionsMerged;
      return true;
    }

    if (G->hasExternalLinkage() || G->hasWeakLinkage()) {
      GlobalAlias *GA = new GlobalAlias(G->getType(), G->getLinkage(), "",
                                        F, G->getParent());
      F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
      GA->takeName(G);
      GA->setVisibility(G->getVisibility());
      G->replaceAllUsesWith(GA);
      G->eraseFromParent();
      ++NumFunctionsMerged;
      return true;
    }
  }

  if (F->hasWeakLinkage() && G->hasWeakLinkage()) {
    GlobalAlias *GA_F = new GlobalAlias(F->getType(), F->getLinkage(), "",
                                        0, F->getParent());
    GA_F->takeName(F);
    GA_F->setVisibility(F->getVisibility());
    F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
    F->replaceAllUsesWith(GA_F);
    F->setName("folded." + GA_F->getName());
    F->setLinkage(GlobalValue::ExternalLinkage);
    GA_F->setAliasee(F);

    GlobalAlias *GA_G = new GlobalAlias(G->getType(), G->getLinkage(), "",
                                        F, G->getParent());
    GA_G->takeName(G);
    GA_G->setVisibility(G->getVisibility());
    G->replaceAllUsesWith(GA_G);
    G->eraseFromParent();

    ++NumFunctionsMerged;
    return true;
  }

  DOUT << "Failed on " << F->getName() << " and " << G->getName() << "\n";

  ++NumMergeFails;
  return false;
}

static bool hasAddressTaken(User *U) {
  for (User::use_iterator I = U->use_begin(), E = U->use_end(); I != E; ++I) {
    User *Use = *I;

    // 'call (bitcast @F to ...)' happens a lot.
    while (isa<ConstantExpr>(Use) && Use->hasOneUse()) {
      Use = *Use->use_begin();
    }

    if (isa<ConstantExpr>(Use)) {
      if (hasAddressTaken(Use))
        return true;
    }

    if (!isa<CallInst>(Use) && !isa<InvokeInst>(Use))
      return true;

    // Make sure we aren't passing U as a parameter to call instead of the
    // callee.
    if (CallSite(cast<Instruction>(Use)).hasArgument(U))
      return true;
  }

  return false;
}

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;

  std::map<unsigned long, std::vector<Function *> > FnMap;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration() || F->isIntrinsic())
      continue;

    if (!F->hasLocalLinkage() && !F->hasExternalLinkage() &&
        !F->hasWeakLinkage())
      continue;

    if (hasAddressTaken(F))
      continue;

    FnMap[hash(F)].push_back(F);
  }

  // TODO: instead of running in a loop, we could also fold functions in callgraph
  // order. Constructing the CFG probably isn't cheaper than just running in a loop.

  bool LocalChanged;
  do {
    LocalChanged = false;
    for (std::map<unsigned long, std::vector<Function *> >::iterator
         I = FnMap.begin(), E = FnMap.end(); I != E; ++I) {
      DOUT << "size: " << FnMap.size() << "\n";
      std::vector<Function *> &FnVec = I->second;
      DOUT << "hash (" << I->first << "): " << FnVec.size() << "\n";

      for (int i = 0, e = FnVec.size(); i != e; ++i) {
        for (int j = i + 1; j != e; ++j) {
          bool isEqual = equals(FnVec[i], FnVec[j]);

          DOUT << "  " << FnVec[i]->getName()
               << (isEqual ? " == " : " != ")
               << FnVec[j]->getName() << "\n";

          if (isEqual) {
            if (fold(FnVec, i, j)) {
              LocalChanged = true;
              FnVec.erase(FnVec.begin() + j);
              --j, --e;
            }
          }
        }
      }

    }
    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}
