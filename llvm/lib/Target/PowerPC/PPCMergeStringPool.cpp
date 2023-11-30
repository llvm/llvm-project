//===-- PPCMergeStringPool.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation tries to merge the strings in the module into one pool
// of strings. The idea is to reduce the number of TOC entries in the module so
// that instead of having one TOC entry for each string there is only one global
// TOC entry and all of the strings are referenced off of that one entry plus
// an offset.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "ppc-merge-strings"

STATISTIC(NumPooledStrings, "Number of Strings Pooled");

using namespace llvm;

static cl::opt<unsigned>
    MaxStringsPooled("ppc-max-strings-pooled", cl::Hidden, cl::init(-1),
                     cl::desc("Maximum Number of Strings to Pool."));

static cl::opt<unsigned>
    MinStringsBeforePool("ppc-min-strings-before-pool", cl::Hidden, cl::init(2),
                         cl::desc("Minimum number of string candidates before "
				  "pooling is considered."));

namespace {
struct {
  bool operator()(const GlobalVariable *LHS, const GlobalVariable *RHS) const {
    // First priority is alignment.
    // If elements are sorted in terms of alignment then there won't be an
    // issue with incorrect alignment that would require padding.
    Align LHSAlign = LHS->getAlign().valueOrOne();
    Align RHSAlign = RHS->getAlign().valueOrOne();
    if (LHSAlign > RHSAlign)
      return true;
    else if (LHSAlign < RHSAlign)
      return false;

    // Next priority is the number of uses.
    // Smaller offsets are easier to materialize because materializing a large
    // offset may require more than one instruction. (ie addis, addi).
    if (LHS->getNumUses() > RHS->getNumUses())
      return true;
    else if (LHS->getNumUses() < RHS->getNumUses())
      return false;

    const Constant *ConstLHS = LHS->getInitializer();
    const ConstantDataSequential *ConstDataLHS =
        dyn_cast<ConstantDataSequential>(ConstLHS);
    unsigned LHSSize =
        ConstDataLHS->getNumElements() * ConstDataLHS->getElementByteSize();
    const Constant *ConstRHS = RHS->getInitializer();
    const ConstantDataSequential *ConstDataRHS =
        dyn_cast<ConstantDataSequential>(ConstRHS);
    unsigned RHSSize =
        ConstDataRHS->getNumElements() * ConstDataRHS->getElementByteSize();

    // Finally smaller constants should go first. This is, again, trying to
    // minimize the offsets into the final struct.
    return LHSSize < RHSSize;
  }
} CompareConstants;

class PPCMergeStringPool : public ModulePass {
public:
  static char ID;
  PPCMergeStringPool() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return mergeModuleStringPool(M); }

  StringRef getPassName() const override { return "PPC Merge String Pool"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();
  }

private:
  // Globals in a Module are already unique so a set is not required and a
  // vector will do.
  std::vector<GlobalVariable *> MergeableStrings;
  Align MaxAlignment;
  Type *PooledStructType;
  LLVMContext *Context;
  void collectCandidateConstants(Module &M);
  bool mergeModuleStringPool(Module &M);
  void replaceUsesWithGEP(GlobalVariable *GlobalToReplace, GlobalVariable *GPool,
                          unsigned ElementIndex);
};


// In order for a constant to be pooled we need to be able to replace all of
// the uses for that constant. This function checks all of the uses to make
// sure that they can be replaced.
static bool hasReplaceableUsers(GlobalVariable &GV) {
  for (User *CurrentUser : GV.users()) {
    // Instruction users are always valid.
    if (isa<Instruction>(CurrentUser))
      continue;

    // We cannot replace GlobalValue users because they are not just nodes
    // in IR. To replace a user like this we would need to create a new
    // GlobalValue with the replacement and then try to delete the original
    // GlobalValue. Deleting the original would only happen if it has no other
    // uses.
    if (isa<GlobalValue>(CurrentUser))
      return false;

    // We only support Instruction and Constant users.
    if (!isa<Constant>(CurrentUser))
      return false;
  }

  return true;
}

// Run through all of the constants in the module and determine if they are
// valid candidates to be merged into the string pool. Valid candidates will
// be added to MergeableStrings.
void PPCMergeStringPool::collectCandidateConstants(Module &M) {
  SmallVector<GlobalValue *, 4> UsedV;
  collectUsedGlobalVariables(M, UsedV, /*CompilerUsed=*/false);
  SmallVector<GlobalValue *, 4> UsedVCompiler;
  collectUsedGlobalVariables(M, UsedVCompiler, /*CompilerUsed=*/true);
  // Combine all of the Global Variables marked as used into a SmallPtrSet for
  // faster lookup inside the loop.
  SmallPtrSet<GlobalValue *, 8> AllUsedGlobals;
  AllUsedGlobals.insert(UsedV.begin(), UsedV.end());
  AllUsedGlobals.insert(UsedVCompiler.begin(), UsedVCompiler.end());

  for (GlobalVariable &Global : M.globals()) {
    LLVM_DEBUG(dbgs() << "Looking at global:");
    LLVM_DEBUG(Global.dump());
    LLVM_DEBUG(dbgs() << "isConstant() " << Global.isConstant() << "\n");
    LLVM_DEBUG(dbgs() << "hasInitializer() " << Global.hasInitializer()
                      << "\n");

    // We can only pool constants.
    if (!Global.isConstant() || !Global.hasInitializer())
      continue;

    // If a global constant has a section we do not try to pool it because
    // there is no guarantee that other constants will also be in the same
    // section. Trying to pool constants from different sections (or no
    // section) means that the pool has to be in multiple sections at the same
    // time.
    if (Global.hasSection())
      continue;

    // Do not pool constants with metadata because we should not add metadata
    // to the pool when that metadata refers to a single constant in the pool.
    if (Global.hasMetadata())
      continue;

    ConstantDataSequential *ConstData =
        dyn_cast<ConstantDataSequential>(Global.getInitializer());

    // If the constant is undef then ConstData will be null.
    if (!ConstData)
      continue;

    // Do not pool globals that are part of llvm.used or llvm.compiler.end.
    if (AllUsedGlobals.contains(&Global))
      continue;

    if (!hasReplaceableUsers(Global))
      continue;

    Align AlignOfGlobal = Global.getAlign().valueOrOne();

    // TODO: At this point do not allow over-aligned types. Adding a type
    //       with larger alignment may lose the larger alignment once it is
    //       added to the struct.
    //       Fix this in a future patch.
    if (AlignOfGlobal.value() > ConstData->getElementByteSize())
      continue;

    // Make sure that the global is only visible inside the compilation unit.
    if (Global.getLinkage() != GlobalValue::PrivateLinkage &&
        Global.getLinkage() != GlobalValue::InternalLinkage)
      continue;

    LLVM_DEBUG(dbgs() << "Constant data of Global: ");
    LLVM_DEBUG(ConstData->dump());
    LLVM_DEBUG(dbgs() << "\n\n");

    MergeableStrings.push_back(&Global);
    if (MaxAlignment < AlignOfGlobal)
      MaxAlignment = AlignOfGlobal;

    // If we have already reached the maximum number of pooled strings then
    // there is no point in looking for more.
    if (MergeableStrings.size() >= MaxStringsPooled)
      break;
  }
}

bool PPCMergeStringPool::mergeModuleStringPool(Module &M) {

  LLVM_DEBUG(dbgs() << "Merging string pool for module: " << M.getName()
                    << "\n");
  LLVM_DEBUG(dbgs() << "Number of globals is: " << M.global_size() << "\n");

  collectCandidateConstants(M);

  // If we have too few constants in the module that are merge candidates we
  // will skip doing the merging.
  if (MergeableStrings.size() < MinStringsBeforePool)
    return false;

  // Sort the global constants to make access more efficient.
  std::sort(MergeableStrings.begin(), MergeableStrings.end(), CompareConstants);

  SmallVector<Constant *> ConstantsInStruct;
  for (GlobalVariable *GV : MergeableStrings)
    ConstantsInStruct.push_back(GV->getInitializer());

  // Use an anonymous struct to pool the strings.
  // TODO: This pass uses a single anonymous struct for all of the pooled
  // entries. This may cause a performance issue in the situation where
  // computing the offset requires two instructions (addis, addi). For the
  // future we may want to split this into multiple structs.
  Constant *ConstantPool = ConstantStruct::getAnon(ConstantsInStruct);
  PooledStructType = ConstantPool->getType();

  // The GlobalVariable constructor calls
  // MM->insertGlobalVariable(PooledGlobal).
  GlobalVariable *PooledGlobal =
      new GlobalVariable(M, PooledStructType,
                         /* isConstant */ true, GlobalValue::PrivateLinkage,
                         ConstantPool, "__ModuleStringPool");
  PooledGlobal->setAlignment(MaxAlignment);

  LLVM_DEBUG(dbgs() << "Constructing global variable for string pool: ");
  LLVM_DEBUG(PooledGlobal->dump());

  Context = &M.getContext();
  size_t ElementIndex = 0;
  for (GlobalVariable *GV : MergeableStrings) {

    LLVM_DEBUG(dbgs() << "The global:\n");
    LLVM_DEBUG(GV->dump());
    LLVM_DEBUG(dbgs() << "Has " << GV->getNumUses() << " uses.\n");

    // Access to the pooled constant strings require an offset. Add a GEP
    // before every use in order to compute this offset.
    replaceUsesWithGEP(GV, PooledGlobal, ElementIndex);

    // This GV has no more uses so we can erase it.
    if (GV->use_empty())
      GV->eraseFromParent();

    NumPooledStrings++;
    ElementIndex++;
  }
  return true;
}

static bool userHasOperand(User *TheUser, GlobalVariable *GVOperand) {
  for (Value *Op : TheUser->operands())
    if (Op == GVOperand)
      return true;
  return false;
}

// For pooled strings we need to add the offset into the pool for each string.
// This is done by adding a Get Element Pointer (GEP) before each user. This
// function adds the GEP.
void PPCMergeStringPool::replaceUsesWithGEP(GlobalVariable *GlobalToReplace,
                                            GlobalVariable *GPool,
                                            unsigned ElementIndex) {
  SmallVector<Value *, 2> Indices;
  Indices.push_back(ConstantInt::get(Type::getInt32Ty(*Context), 0));
  Indices.push_back(ConstantInt::get(Type::getInt32Ty(*Context), ElementIndex));

  // Need to save a temporary copy of each user list because we remove uses
  // as we replace them.
  SmallVector<User *> Users;
  for (User *CurrentUser : GlobalToReplace->users())
    Users.push_back(CurrentUser);

  for (User *CurrentUser : Users) {
    Instruction *UserInstruction = dyn_cast<Instruction>(CurrentUser);
    Constant *UserConstant = dyn_cast<Constant>(CurrentUser);

    // At this point we expect that the user is either an instruction or a
    // constant.
    assert((UserConstant || UserInstruction) &&
           "Expected the user to be an instruction or a constant.");

    // The user was not found so it must have been replaced earlier.
    if (!userHasOperand(CurrentUser, GlobalToReplace))
      continue;

    // We cannot replace operands in globals so we ignore those.
    if (isa<GlobalValue>(CurrentUser))
      continue;

    if (!UserInstruction) {
      // User is a constant type.
      Constant *ConstGEP = ConstantExpr::getInBoundsGetElementPtr(
          PooledStructType, GPool, Indices);
      UserConstant->handleOperandChange(GlobalToReplace, ConstGEP);
      continue;
    }

    if (PHINode *UserPHI = dyn_cast<PHINode>(UserInstruction)) {
      // GEP instructions cannot be added before PHI nodes.
      // With getInBoundsGetElementPtr we create the GEP and then replace it
      // inline into the PHI.
      Constant *ConstGEP = ConstantExpr::getInBoundsGetElementPtr(
          PooledStructType, GPool, Indices);
      UserPHI->replaceUsesOfWith(GlobalToReplace, ConstGEP);
      continue;
    }
    // The user is a valid instruction that is not a PHINode.
    GetElementPtrInst *GEPInst =
        GetElementPtrInst::Create(PooledStructType, GPool, Indices);
    GEPInst->insertBefore(UserInstruction);

    LLVM_DEBUG(dbgs() << "Inserting GEP before:\n");
    LLVM_DEBUG(UserInstruction->dump());

    LLVM_DEBUG(dbgs() << "Replacing this global:\n");
    LLVM_DEBUG(GlobalToReplace->dump());
    LLVM_DEBUG(dbgs() << "with this:\n");
    LLVM_DEBUG(GEPInst->dump());

    // After the GEP is inserted the GV can be replaced.
    CurrentUser->replaceUsesOfWith(GlobalToReplace, GEPInst);
  }
}

} // namespace

char PPCMergeStringPool::ID = 0;

INITIALIZE_PASS(PPCMergeStringPool, DEBUG_TYPE, "PPC Merge String Pool", false,
                false)

ModulePass *llvm::createPPCMergeStringPoolPass() {
  return new PPCMergeStringPool();
}
