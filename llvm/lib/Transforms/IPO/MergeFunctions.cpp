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
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <vector>
using namespace llvm;

STATISTIC(NumFunctionsMerged, "Number of functions merged");

namespace {
  struct VISIBILITY_HIDDEN MergeFunctions : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    MergeFunctions() : ModulePass(&ID) {}

    bool runOnModule(Module &M);
  };
}

char MergeFunctions::ID = 0;
static RegisterPass<MergeFunctions>
X("mergefunc", "Merge Functions");

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

// ===----------------------------------------------------------------------===
// Comparison of functions
// ===----------------------------------------------------------------------===

static unsigned long hash(const Function *F) {
  const FunctionType *FTy = F->getFunctionType();

  FoldingSetNodeID ID;
  ID.AddInteger(F->size());
  ID.AddInteger(F->getCallingConv());
  ID.AddBoolean(F->hasGC());
  ID.AddBoolean(FTy->isVarArg());
  ID.AddInteger(FTy->getReturnType()->getTypeID());
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    ID.AddInteger(FTy->getParamType(i)->getTypeID());
  return ID.ComputeHash();
}

/// IgnoreBitcasts - given a bitcast, returns the first non-bitcast found by
/// walking the chain of cast operands. Otherwise, returns the argument.
static Value* IgnoreBitcasts(Value *V) {
  while (BitCastInst *BC = dyn_cast<BitCastInst>(V))
    V = BC->getOperand(0);

  return V;
}

/// isEquivalentType - any two pointers are equivalent. Otherwise, standard
/// type equivalence rules apply.
static bool isEquivalentType(const Type *Ty1, const Type *Ty2) {
  if (Ty1 == Ty2)
    return true;
  if (Ty1->getTypeID() != Ty2->getTypeID())
    return false;

  switch(Ty1->getTypeID()) {
  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return true;

  case Type::IntegerTyID:
  case Type::OpaqueTyID:
    // Ty1 == Ty2 would have returned true earlier.
    return false;

  default:
    llvm_unreachable("Unknown type!");
    return false;

  case Type::PointerTyID: {
    const PointerType *PTy1 = cast<PointerType>(Ty1);
    const PointerType *PTy2 = cast<PointerType>(Ty2);
    return PTy1->getAddressSpace() == PTy2->getAddressSpace();
  }

  case Type::StructTyID: {
    const StructType *STy1 = cast<StructType>(Ty1);
    const StructType *STy2 = cast<StructType>(Ty2);
    if (STy1->getNumElements() != STy2->getNumElements())
      return false;

    if (STy1->isPacked() != STy2->isPacked())
      return false;

    for (unsigned i = 0, e = STy1->getNumElements(); i != e; ++i) {
      if (!isEquivalentType(STy1->getElementType(i), STy2->getElementType(i)))
        return false;
    }
    return true;
  }

  case Type::FunctionTyID: {
    const FunctionType *FTy1 = cast<FunctionType>(Ty1);
    const FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy1->getNumParams() != FTy2->getNumParams() ||
        FTy1->isVarArg() != FTy2->isVarArg())
      return false;

    if (!isEquivalentType(FTy1->getReturnType(), FTy2->getReturnType()))
      return false;

    for (unsigned i = 0, e = FTy1->getNumParams(); i != e; ++i) {
      if (!isEquivalentType(FTy1->getParamType(i), FTy2->getParamType(i)))
        return false;
    }
    return true;
  }

  case Type::ArrayTyID:
  case Type::VectorTyID: {
    const SequentialType *STy1 = cast<SequentialType>(Ty1);
    const SequentialType *STy2 = cast<SequentialType>(Ty2);
    return isEquivalentType(STy1->getElementType(), STy2->getElementType());
  }
  }
}

/// isEquivalentOperation - determine whether the two operations are the same
/// except that pointer-to-A and pointer-to-B are equivalent. This should be
/// kept in sync with Instruction::isSameOperationAs.
static bool
isEquivalentOperation(const Instruction *I1, const Instruction *I2) {
  if (I1->getOpcode() != I2->getOpcode() ||
      I1->getNumOperands() != I2->getNumOperands() ||
      !isEquivalentType(I1->getType(), I2->getType()) ||
      !I1->hasSameSubclassOptionalData(I2))
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = I1->getNumOperands(); i != e; ++i)
    if (!isEquivalentType(I1->getOperand(i)->getType(),
                          I2->getOperand(i)->getType()))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(I1))
    return LI->isVolatile() == cast<LoadInst>(I2)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I2)->getAlignment();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I1))
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I2)->getAlignment();
  if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
    return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(I1))
    return CI->isTailCall() == cast<CallInst>(I2)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I2)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<CallInst>(I2)->getAttributes().getRawPointer();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(I1))
    return CI->getCallingConv() == cast<InvokeInst>(I2)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<InvokeInst>(I2)->getAttributes().getRawPointer();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1)) {
    if (IVI->getNumIndices() != cast<InsertValueInst>(I2)->getNumIndices())
      return false;
    for (unsigned i = 0, e = IVI->getNumIndices(); i != e; ++i)
      if (IVI->idx_begin()[i] != cast<InsertValueInst>(I2)->idx_begin()[i])
        return false;
    return true;
  }
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1)) {
    if (EVI->getNumIndices() != cast<ExtractValueInst>(I2)->getNumIndices())
      return false;
    for (unsigned i = 0, e = EVI->getNumIndices(); i != e; ++i)
      if (EVI->idx_begin()[i] != cast<ExtractValueInst>(I2)->idx_begin()[i])
        return false;
    return true;
  }

  return true;
}

static bool compare(const Value *V, const Value *U) {
  assert(!isa<BasicBlock>(V) && !isa<BasicBlock>(U) &&
         "Must not compare basic blocks.");

  assert(isEquivalentType(V->getType(), U->getType()) &&
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
  // Speculatively add it anyways. If it's false, we'll notice a difference
  // later, and this won't matter.
  ValueMap[BB1] = BB2;

  BasicBlock::const_iterator FI = BB1->begin(), FE = BB1->end();
  BasicBlock::const_iterator GI = BB2->begin(), GE = BB2->end();

  do {
    if (isa<BitCastInst>(FI)) {
      ++FI;
      continue;
    }
    if (isa<BitCastInst>(GI)) {
      ++GI;
      continue;
    }

    if (!isEquivalentOperation(FI, GI))
      return false;

    if (isa<GetElementPtrInst>(FI)) {
      const GetElementPtrInst *GEPF = cast<GetElementPtrInst>(FI);
      const GetElementPtrInst *GEPG = cast<GetElementPtrInst>(GI);
      if (GEPF->hasAllZeroIndices() && GEPG->hasAllZeroIndices()) {
        // It's effectively a bitcast.
        ++FI, ++GI;
        continue;
      }

      // TODO: we only really care about the elements before the index
      if (FI->getOperand(0)->getType() != GI->getOperand(0)->getType())
        return false;
    }

    if (ValueMap[FI] == GI) {
      ++FI, ++GI;
      continue;
    }

    if (ValueMap[FI] != NULL)
      return false;

    for (unsigned i = 0, e = FI->getNumOperands(); i != e; ++i) {
      Value *OpF = IgnoreBitcasts(FI->getOperand(i));
      Value *OpG = IgnoreBitcasts(GI->getOperand(i));

      if (ValueMap[OpF] == OpG)
        continue;

      if (ValueMap[OpF] != NULL)
        return false;

      if (OpF->getValueID() != OpG->getValueID() ||
          !isEquivalentType(OpF->getType(), OpG->getType()))
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

  if (F->isVarArg() != G->isVarArg())
    return false;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (F->getCallingConv() != G->getCallingConv())
    return false;

  if (!isEquivalentType(F->getFunctionType(), G->getFunctionType()))
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

// ===----------------------------------------------------------------------===
// Folding of functions
// ===----------------------------------------------------------------------===

// Cases:
// * F is external strong, G is external strong:
//   turn G into a thunk to F    (1)
// * F is external strong, G is external weak:
//   turn G into a thunk to F    (1)
// * F is external weak, G is external weak:
//   unfoldable
// * F is external strong, G is internal:
//   address of G taken:
//     turn G into a thunk to F  (1)
//   address of G not taken:
//     make G an alias to F      (2)
// * F is internal, G is external weak
//   address of F is taken:
//     turn G into a thunk to F  (1)
//   address of F is not taken:
//     make G an alias of F      (2)
// * F is internal, G is internal:
//   address of F and G are taken:
//     turn G into a thunk to F  (1)
//   address of G is not taken:
//     make G an alias to F      (2)
//
// alias requires linkage == (external,local,weak) fallback to creating a thunk
// external means 'externally visible' linkage != (internal,private)
// internal means linkage == (internal,private)
// weak means linkage mayBeOverridable
// being external implies that the address is taken
//
// 1. turn G into a thunk to F
// 2. make G an alias to F

enum LinkageCategory {
  ExternalStrong,
  ExternalWeak,
  Internal
};

static LinkageCategory categorize(const Function *F) {
  switch (F->getLinkage()) {
  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    return Internal;

  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::ExternalWeakLinkage:
    return ExternalWeak;

  case GlobalValue::ExternalLinkage:
  case GlobalValue::AvailableExternallyLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::AppendingLinkage:
  case GlobalValue::DLLImportLinkage:
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::GhostLinkage:
  case GlobalValue::CommonLinkage:
    return ExternalStrong;
  }

  llvm_unreachable("Unknown LinkageType.");
  return ExternalWeak;
}

static void ThunkGToF(Function *F, Function *G) {
  Function *NewG = Function::Create(G->getFunctionType(), G->getLinkage(), "",
                                    G->getParent());
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "", NewG);

  std::vector<Value *> Args;
  unsigned i = 0;
  const FunctionType *FFTy = F->getFunctionType();
  for (Function::arg_iterator AI = NewG->arg_begin(), AE = NewG->arg_end();
       AI != AE; ++AI) {
    if (FFTy->getParamType(i) == AI->getType())
      Args.push_back(AI);
    else {
      Value *BCI = new BitCastInst(AI, FFTy->getParamType(i), "", BB);
      Args.push_back(BCI);
    }
    ++i;
  }

  CallInst *CI = CallInst::Create(F, Args.begin(), Args.end(), "", BB);
  CI->setTailCall();
  CI->setCallingConv(F->getCallingConv());
  if (NewG->getReturnType() == Type::getVoidTy(F->getContext())) {
    ReturnInst::Create(F->getContext(), BB);
  } else if (CI->getType() != NewG->getReturnType()) {
    Value *BCI = new BitCastInst(CI, NewG->getReturnType(), "", BB);
    ReturnInst::Create(F->getContext(), BCI, BB);
  } else {
    ReturnInst::Create(F->getContext(), CI, BB);
  }

  NewG->copyAttributesFrom(G);
  NewG->takeName(G);
  G->replaceAllUsesWith(NewG);
  G->eraseFromParent();

  // TODO: look at direct callers to G and make them all direct callers to F.
}

static void AliasGToF(Function *F, Function *G) {
  if (!G->hasExternalLinkage() && !G->hasLocalLinkage() && !G->hasWeakLinkage())
    return ThunkGToF(F, G);

  GlobalAlias *GA = new GlobalAlias(
    G->getType(), G->getLinkage(), "",
    ConstantExpr::getBitCast(F, G->getType()), G->getParent());
  F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
  GA->takeName(G);
  GA->setVisibility(G->getVisibility());
  G->replaceAllUsesWith(GA);
  G->eraseFromParent();
}

static bool fold(std::vector<Function *> &FnVec, unsigned i, unsigned j) {
  Function *F = FnVec[i];
  Function *G = FnVec[j];

  LinkageCategory catF = categorize(F);
  LinkageCategory catG = categorize(G);

  if (catF == ExternalWeak || (catF == Internal && catG == ExternalStrong)) {
    std::swap(FnVec[i], FnVec[j]);
    std::swap(F, G);
    std::swap(catF, catG);
  }

  switch (catF) {
    case ExternalStrong:
      switch (catG) {
        case ExternalStrong:
        case ExternalWeak:
          ThunkGToF(F, G);
          break;
        case Internal:
          if (G->hasAddressTaken())
            ThunkGToF(F, G);
          else
            AliasGToF(F, G);
          break;
      }
      break;

    case ExternalWeak: {
      assert(catG == ExternalWeak);

      // Make them both thunks to the same internal function.
      F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
      Function *H = Function::Create(F->getFunctionType(), F->getLinkage(), "",
                                     F->getParent());
      H->copyAttributesFrom(F);
      H->takeName(F);
      F->replaceAllUsesWith(H);

      ThunkGToF(F, G);
      ThunkGToF(F, H);

      F->setLinkage(GlobalValue::InternalLinkage);
    } break;

    case Internal:
      switch (catG) {
        case ExternalStrong:
          llvm_unreachable(0);
          // fall-through
        case ExternalWeak:
	  if (F->hasAddressTaken())
            ThunkGToF(F, G);
          else
            AliasGToF(F, G);
	  break;
        case Internal: {
          bool addrTakenF = F->hasAddressTaken();
          bool addrTakenG = G->hasAddressTaken();
          if (!addrTakenF && addrTakenG) {
            std::swap(FnVec[i], FnVec[j]);
            std::swap(F, G);
	    std::swap(addrTakenF, addrTakenG);
	  }

          if (addrTakenF && addrTakenG) {
            ThunkGToF(F, G);
          } else {
            assert(!addrTakenG);
            AliasGToF(F, G);
          }
	} break;
      }
      break;
  }

  ++NumFunctionsMerged;
  return true;
}

// ===----------------------------------------------------------------------===
// Pass definition
// ===----------------------------------------------------------------------===

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;

  std::map<unsigned long, std::vector<Function *> > FnMap;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration() || F->isIntrinsic())
      continue;

    FnMap[hash(F)].push_back(F);
  }

  // TODO: instead of running in a loop, we could also fold functions in
  // callgraph order. Constructing the CFG probably isn't cheaper than just
  // running in a loop, unless it happened to already be available.

  bool LocalChanged;
  do {
    LocalChanged = false;
    DOUT << "size: " << FnMap.size() << "\n";
    for (std::map<unsigned long, std::vector<Function *> >::iterator
         I = FnMap.begin(), E = FnMap.end(); I != E; ++I) {
      std::vector<Function *> &FnVec = I->second;
      DOUT << "hash (" << I->first << "): " << FnVec.size() << "\n";

      for (int i = 0, e = FnVec.size(); i != e; ++i) {
        for (int j = i + 1; j != e; ++j) {
          bool isEqual = equals(FnVec[i], FnVec[j]);

          DEBUG(errs() << "  " << FnVec[i]->getName()
                << (isEqual ? " == " : " != ")
                << FnVec[j]->getName() << "\n");

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
