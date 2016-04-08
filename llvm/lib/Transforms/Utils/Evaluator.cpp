//===- Evaluator.cpp - LLVM IR evaluator ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Function evaluator for LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "evaluator"

using namespace llvm;

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL);

/// Return true if the specified constant can be handled by the code generator.
/// We don't want to generate something like:
///   void *X = &X/42;
/// because the code generator doesn't have a relocation that can handle that.
///
/// This function should be called if C was not found (but just got inserted)
/// in SimpleConstants to avoid having to rescan the same constants all the
/// time.
static bool
isSimpleEnoughValueToCommitHelper(Constant *C,
                                  SmallPtrSetImpl<Constant *> &SimpleConstants,
                                  const DataLayout &DL) {
  // Simple global addresses are supported, do not allow dllimport or
  // thread-local globals.
  if (auto *GV = dyn_cast<GlobalValue>(C))
    return !GV->hasDLLImportStorageClass() && !GV->isThreadLocal();

  // Simple integer, undef, constant aggregate zero, etc are all supported.
  if (C->getNumOperands() == 0 || isa<BlockAddress>(C))
    return true;

  // Aggregate values are safe if all their elements are.
  if (isa<ConstantAggregate>(C)) {
    for (Value *Op : C->operands())
      if (!isSimpleEnoughValueToCommit(cast<Constant>(Op), SimpleConstants, DL))
        return false;
    return true;
  }

  // We don't know exactly what relocations are allowed in constant expressions,
  // so we allow &global+constantoffset, which is safe and uniformly supported
  // across targets.
  ConstantExpr *CE = cast<ConstantExpr>(C);
  switch (CE->getOpcode()) {
  case Instruction::BitCast:
    // Bitcast is fine if the casted value is fine.
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
    // int <=> ptr is fine if the int type is the same size as the
    // pointer type.
    if (DL.getTypeSizeInBits(CE->getType()) !=
        DL.getTypeSizeInBits(CE->getOperand(0)->getType()))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  // GEP is fine if it is simple + constant offset.
  case Instruction::GetElementPtr:
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
      if (!isa<ConstantInt>(CE->getOperand(i)))
        return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::Add:
    // We allow simple+cst.
    if (!isa<ConstantInt>(CE->getOperand(1)))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);
  }
  return false;
}

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL) {
  // If we already checked this constant, we win.
  if (!SimpleConstants.insert(C).second)
    return true;
  // Check the constant.
  return isSimpleEnoughValueToCommitHelper(C, SimpleConstants, DL);
}

/// Return true if this constant is simple enough for us to understand.  In
/// particular, if it is a cast to anything other than from one pointer type to
/// another pointer type, we punt.  We basically just support direct accesses to
/// globals and GEP's of globals.  This should be kept up to date with
/// CommitValueTo.
static bool isSimpleEnoughPointerToCommit(Constant *C) {
  // Conservatively, avoid aggregate types. This is because we don't
  // want to worry about them partially overlapping other stores.
  if (!cast<PointerType>(C->getType())->getElementType()->isSingleValueType())
    return false;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    // Do not allow weak/*_odr/linkonce linkage or external globals.
    return GV->hasUniqueInitializer();

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Handle a constantexpr gep.
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0)) &&
        cast<GEPOperator>(CE)->isInBounds()) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      if (!GV->hasUniqueInitializer())
        return false;

      // The first index must be zero.
      ConstantInt *CI = dyn_cast<ConstantInt>(*std::next(CE->op_begin()));
      if (!CI || !CI->isZero()) return false;

      // The remaining indices must be compile-time known integers within the
      // notional bounds of the corresponding static array types.
      if (!CE->isGEPWithNoNotionalOverIndexing())
        return false;

      return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);

    // A constantexpr bitcast from a pointer to another pointer is a no-op,
    // and we know how to evaluate it by moving the bitcast from the pointer
    // operand to the value operand.
    } else if (CE->getOpcode() == Instruction::BitCast &&
               isa<GlobalVariable>(CE->getOperand(0))) {
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      return cast<GlobalVariable>(CE->getOperand(0))->hasUniqueInitializer();
    }
  }

  return false;
}

/// Return the value that would be computed by a load from P after the stores
/// reflected by 'memory' have been performed.  If we can't decide, return null.
Constant *Evaluator::ComputeLoadResult(Constant *P) {
  // If this memory location has been recently stored, use the stored value: it
  // is the most up-to-date.
  DenseMap<Constant*, Constant*>::const_iterator I = MutatedMemory.find(P);
  if (I != MutatedMemory.end()) return I->second;

  // Access it.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(P)) {
    if (GV->hasDefinitiveInitializer())
      return GV->getInitializer();
    return nullptr;
  }

  // Handle a constantexpr getelementptr.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(P))
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0))) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      if (GV->hasDefinitiveInitializer())
        return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);
    }

  return nullptr;  // don't know how to evaluate.
}

/// Evaluate all instructions in block BB, returning true if successful, false
/// if we can't evaluate it.  NewBB returns the next BB that control flows into,
/// or null upon return.
bool Evaluator::EvaluateBlock(BasicBlock::iterator CurInst,
                              BasicBlock *&NextBB) {
  // This is the main evaluation loop.
  while (1) {
    Constant *InstResult = nullptr;

    DEBUG(dbgs() << "Evaluating Instruction: " << *CurInst << "\n");

    if (StoreInst *SI = dyn_cast<StoreInst>(CurInst)) {
      if (!SI->isSimple()) {
        DEBUG(dbgs() << "Store is not simple! Can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }
      Constant *Ptr = getVal(SI->getOperand(1));
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        DEBUG(dbgs() << "Folding constant ptr expression: " << *Ptr);
        Ptr = ConstantFoldConstantExpression(CE, DL, TLI);
        DEBUG(dbgs() << "; To: " << *Ptr << "\n");
      }
      if (!isSimpleEnoughPointerToCommit(Ptr)) {
        // If this is too complex for us to commit, reject it.
        DEBUG(dbgs() << "Pointer is too complex for us to evaluate store.");
        return false;
      }

      Constant *Val = getVal(SI->getOperand(0));

      // If this might be too difficult for the backend to handle (e.g. the addr
      // of one global variable divided by another) then we can't commit it.
      if (!isSimpleEnoughValueToCommit(Val, SimpleConstants, DL)) {
        DEBUG(dbgs() << "Store value is too complex to evaluate store. " << *Val
              << "\n");
        return false;
      }

      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          DEBUG(dbgs() << "Attempting to resolve bitcast on constant ptr.\n");
          // If we're evaluating a store through a bitcast, then we need
          // to pull the bitcast off the pointer type and push it onto the
          // stored value.
          Ptr = CE->getOperand(0);

          Type *NewTy = cast<PointerType>(Ptr->getType())->getElementType();

          // In order to push the bitcast onto the stored value, a bitcast
          // from NewTy to Val's type must be legal.  If it's not, we can try
          // introspecting NewTy to find a legal conversion.
          while (!Val->getType()->canLosslesslyBitCastTo(NewTy)) {
            // If NewTy is a struct, we can convert the pointer to the struct
            // into a pointer to its first member.
            // FIXME: This could be extended to support arrays as well.
            if (StructType *STy = dyn_cast<StructType>(NewTy)) {
              NewTy = STy->getTypeAtIndex(0U);

              IntegerType *IdxTy = IntegerType::get(NewTy->getContext(), 32);
              Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
              Constant * const IdxList[] = {IdxZero, IdxZero};

              Ptr = ConstantExpr::getGetElementPtr(nullptr, Ptr, IdxList);
              if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
                Ptr = ConstantFoldConstantExpression(CE, DL, TLI);

            // If we can't improve the situation by introspecting NewTy,
            // we have to give up.
            } else {
              DEBUG(dbgs() << "Failed to bitcast constant ptr, can not "
                    "evaluate.\n");
              return false;
            }
          }

          // If we found compatible types, go ahead and push the bitcast
          // onto the stored value.
          Val = ConstantExpr::getBitCast(Val, NewTy);

          DEBUG(dbgs() << "Evaluated bitcast: " << *Val << "\n");
        }
      }

      MutatedMemory[Ptr] = Val;
    } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(CurInst)) {
      InstResult = ConstantExpr::get(BO->getOpcode(),
                                     getVal(BO->getOperand(0)),
                                     getVal(BO->getOperand(1)));
      DEBUG(dbgs() << "Found a BinaryOperator! Simplifying: " << *InstResult
            << "\n");
    } else if (CmpInst *CI = dyn_cast<CmpInst>(CurInst)) {
      InstResult = ConstantExpr::getCompare(CI->getPredicate(),
                                            getVal(CI->getOperand(0)),
                                            getVal(CI->getOperand(1)));
      DEBUG(dbgs() << "Found a CmpInst! Simplifying: " << *InstResult
            << "\n");
    } else if (CastInst *CI = dyn_cast<CastInst>(CurInst)) {
      InstResult = ConstantExpr::getCast(CI->getOpcode(),
                                         getVal(CI->getOperand(0)),
                                         CI->getType());
      DEBUG(dbgs() << "Found a Cast! Simplifying: " << *InstResult
            << "\n");
    } else if (SelectInst *SI = dyn_cast<SelectInst>(CurInst)) {
      InstResult = ConstantExpr::getSelect(getVal(SI->getOperand(0)),
                                           getVal(SI->getOperand(1)),
                                           getVal(SI->getOperand(2)));
      DEBUG(dbgs() << "Found a Select! Simplifying: " << *InstResult
            << "\n");
    } else if (auto *EVI = dyn_cast<ExtractValueInst>(CurInst)) {
      InstResult = ConstantExpr::getExtractValue(
          getVal(EVI->getAggregateOperand()), EVI->getIndices());
      DEBUG(dbgs() << "Found an ExtractValueInst! Simplifying: " << *InstResult
                   << "\n");
    } else if (auto *IVI = dyn_cast<InsertValueInst>(CurInst)) {
      InstResult = ConstantExpr::getInsertValue(
          getVal(IVI->getAggregateOperand()),
          getVal(IVI->getInsertedValueOperand()), IVI->getIndices());
      DEBUG(dbgs() << "Found an InsertValueInst! Simplifying: " << *InstResult
                   << "\n");
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurInst)) {
      Constant *P = getVal(GEP->getOperand(0));
      SmallVector<Constant*, 8> GEPOps;
      for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end();
           i != e; ++i)
        GEPOps.push_back(getVal(*i));
      InstResult =
          ConstantExpr::getGetElementPtr(GEP->getSourceElementType(), P, GEPOps,
                                         cast<GEPOperator>(GEP)->isInBounds());
      DEBUG(dbgs() << "Found a GEP! Simplifying: " << *InstResult
            << "\n");
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurInst)) {

      if (!LI->isSimple()) {
        DEBUG(dbgs() << "Found a Load! Not a simple load, can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }

      Constant *Ptr = getVal(LI->getOperand(0));
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        Ptr = ConstantFoldConstantExpression(CE, DL, TLI);
        DEBUG(dbgs() << "Found a constant pointer expression, constant "
              "folding: " << *Ptr << "\n");
      }
      InstResult = ComputeLoadResult(Ptr);
      if (!InstResult) {
        DEBUG(dbgs() << "Failed to compute load result. Can not evaluate load."
              "\n");
        return false; // Could not evaluate load.
      }

      DEBUG(dbgs() << "Evaluated load: " << *InstResult << "\n");
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(CurInst)) {
      if (AI->isArrayAllocation()) {
        DEBUG(dbgs() << "Found an array alloca. Can not evaluate.\n");
        return false;  // Cannot handle array allocs.
      }
      Type *Ty = AI->getAllocatedType();
      AllocaTmps.push_back(
          make_unique<GlobalVariable>(Ty, false, GlobalValue::InternalLinkage,
                                      UndefValue::get(Ty), AI->getName()));
      InstResult = AllocaTmps.back().get();
      DEBUG(dbgs() << "Found an alloca. Result: " << *InstResult << "\n");
    } else if (isa<CallInst>(CurInst) || isa<InvokeInst>(CurInst)) {
      CallSite CS(&*CurInst);

      // Debug info can safely be ignored here.
      if (isa<DbgInfoIntrinsic>(CS.getInstruction())) {
        DEBUG(dbgs() << "Ignoring debug info.\n");
        ++CurInst;
        continue;
      }

      // Cannot handle inline asm.
      if (isa<InlineAsm>(CS.getCalledValue())) {
        DEBUG(dbgs() << "Found inline asm, can not evaluate.\n");
        return false;
      }

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CS.getInstruction())) {
        if (MemSetInst *MSI = dyn_cast<MemSetInst>(II)) {
          if (MSI->isVolatile()) {
            DEBUG(dbgs() << "Can not optimize a volatile memset " <<
                  "intrinsic.\n");
            return false;
          }
          Constant *Ptr = getVal(MSI->getDest());
          Constant *Val = getVal(MSI->getValue());
          Constant *DestVal = ComputeLoadResult(getVal(Ptr));
          if (Val->isNullValue() && DestVal && DestVal->isNullValue()) {
            // This memset is a no-op.
            DEBUG(dbgs() << "Ignoring no-op memset.\n");
            ++CurInst;
            continue;
          }
        }

        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
          DEBUG(dbgs() << "Ignoring lifetime intrinsic.\n");
          ++CurInst;
          continue;
        }

        if (II->getIntrinsicID() == Intrinsic::invariant_start) {
          // We don't insert an entry into Values, as it doesn't have a
          // meaningful return value.
          if (!II->use_empty()) {
            DEBUG(dbgs() << "Found unused invariant_start. Can't evaluate.\n");
            return false;
          }
          ConstantInt *Size = cast<ConstantInt>(II->getArgOperand(0));
          Value *PtrArg = getVal(II->getArgOperand(1));
          Value *Ptr = PtrArg->stripPointerCasts();
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
            Type *ElemTy = GV->getValueType();
            if (!Size->isAllOnesValue() &&
                Size->getValue().getLimitedValue() >=
                    DL.getTypeStoreSize(ElemTy)) {
              Invariants.insert(GV);
              DEBUG(dbgs() << "Found a global var that is an invariant: " << *GV
                    << "\n");
            } else {
              DEBUG(dbgs() << "Found a global var, but can not treat it as an "
                    "invariant.\n");
            }
          }
          // Continue even if we do nothing.
          ++CurInst;
          continue;
        } else if (II->getIntrinsicID() == Intrinsic::assume) {
          DEBUG(dbgs() << "Skipping assume intrinsic.\n");
          ++CurInst;
          continue;
        }

        DEBUG(dbgs() << "Unknown intrinsic. Can not evaluate.\n");
        return false;
      }

      // Resolve function pointers.
      Function *Callee = dyn_cast<Function>(getVal(CS.getCalledValue()));
      if (!Callee || Callee->isInterposable()) {
        DEBUG(dbgs() << "Can not resolve function pointer.\n");
        return false;  // Cannot resolve.
      }

      SmallVector<Constant*, 8> Formals;
      for (User::op_iterator i = CS.arg_begin(), e = CS.arg_end(); i != e; ++i)
        Formals.push_back(getVal(*i));

      if (Callee->isDeclaration()) {
        // If this is a function we can constant fold, do it.
        if (Constant *C = ConstantFoldCall(Callee, Formals, TLI)) {
          InstResult = C;
          DEBUG(dbgs() << "Constant folded function call. Result: " <<
                *InstResult << "\n");
        } else {
          DEBUG(dbgs() << "Can not constant fold function call.\n");
          return false;
        }
      } else {
        if (Callee->getFunctionType()->isVarArg()) {
          DEBUG(dbgs() << "Can not constant fold vararg function call.\n");
          return false;
        }

        Constant *RetVal = nullptr;
        // Execute the call, if successful, use the return value.
        ValueStack.emplace_back();
        if (!EvaluateFunction(Callee, RetVal, Formals)) {
          DEBUG(dbgs() << "Failed to evaluate function.\n");
          return false;
        }
        ValueStack.pop_back();
        InstResult = RetVal;

        if (InstResult) {
          DEBUG(dbgs() << "Successfully evaluated function. Result: "
                       << *InstResult << "\n\n");
        } else {
          DEBUG(dbgs() << "Successfully evaluated function. Result: 0\n\n");
        }
      }
    } else if (isa<TerminatorInst>(CurInst)) {
      DEBUG(dbgs() << "Found a terminator instruction.\n");

      if (BranchInst *BI = dyn_cast<BranchInst>(CurInst)) {
        if (BI->isUnconditional()) {
          NextBB = BI->getSuccessor(0);
        } else {
          ConstantInt *Cond =
            dyn_cast<ConstantInt>(getVal(BI->getCondition()));
          if (!Cond) return false;  // Cannot determine.

          NextBB = BI->getSuccessor(!Cond->getZExtValue());
        }
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(CurInst)) {
        ConstantInt *Val =
          dyn_cast<ConstantInt>(getVal(SI->getCondition()));
        if (!Val) return false;  // Cannot determine.
        NextBB = SI->findCaseValue(Val).getCaseSuccessor();
      } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(CurInst)) {
        Value *Val = getVal(IBI->getAddress())->stripPointerCasts();
        if (BlockAddress *BA = dyn_cast<BlockAddress>(Val))
          NextBB = BA->getBasicBlock();
        else
          return false;  // Cannot determine.
      } else if (isa<ReturnInst>(CurInst)) {
        NextBB = nullptr;
      } else {
        // invoke, unwind, resume, unreachable.
        DEBUG(dbgs() << "Can not handle terminator.");
        return false;  // Cannot handle this terminator.
      }

      // We succeeded at evaluating this block!
      DEBUG(dbgs() << "Successfully evaluated block.\n");
      return true;
    } else {
      // Did not know how to evaluate this!
      DEBUG(dbgs() << "Failed to evaluate block due to unhandled instruction."
            "\n");
      return false;
    }

    if (!CurInst->use_empty()) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(InstResult))
        InstResult = ConstantFoldConstantExpression(CE, DL, TLI);

      setVal(&*CurInst, InstResult);
    }

    // If we just processed an invoke, we finished evaluating the block.
    if (InvokeInst *II = dyn_cast<InvokeInst>(CurInst)) {
      NextBB = II->getNormalDest();
      DEBUG(dbgs() << "Found an invoke instruction. Finished Block.\n\n");
      return true;
    }

    // Advance program counter.
    ++CurInst;
  }
}

/// Evaluate a call to function F, returning true if successful, false if we
/// can't evaluate it.  ActualArgs contains the formal arguments for the
/// function.
bool Evaluator::EvaluateFunction(Function *F, Constant *&RetVal,
                                 const SmallVectorImpl<Constant*> &ActualArgs) {
  // Check to see if this function is already executing (recursion).  If so,
  // bail out.  TODO: we might want to accept limited recursion.
  if (std::find(CallStack.begin(), CallStack.end(), F) != CallStack.end())
    return false;

  CallStack.push_back(F);

  // Initialize arguments to the incoming values specified.
  unsigned ArgNo = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E;
       ++AI, ++ArgNo)
    setVal(&*AI, ActualArgs[ArgNo]);

  // ExecutedBlocks - We only handle non-looping, non-recursive code.  As such,
  // we can only evaluate any one basic block at most once.  This set keeps
  // track of what we have executed so we can detect recursive cases etc.
  SmallPtrSet<BasicBlock*, 32> ExecutedBlocks;

  // CurBB - The current basic block we're evaluating.
  BasicBlock *CurBB = &F->front();

  BasicBlock::iterator CurInst = CurBB->begin();

  while (1) {
    BasicBlock *NextBB = nullptr; // Initialized to avoid compiler warnings.
    DEBUG(dbgs() << "Trying to evaluate BB: " << *CurBB << "\n");

    if (!EvaluateBlock(CurInst, NextBB))
      return false;

    if (!NextBB) {
      // Successfully running until there's no next block means that we found
      // the return.  Fill it the return value and pop the call stack.
      ReturnInst *RI = cast<ReturnInst>(CurBB->getTerminator());
      if (RI->getNumOperands())
        RetVal = getVal(RI->getOperand(0));
      CallStack.pop_back();
      return true;
    }

    // Okay, we succeeded in evaluating this control flow.  See if we have
    // executed the new block before.  If so, we have a looping function,
    // which we cannot evaluate in reasonable time.
    if (!ExecutedBlocks.insert(NextBB).second)
      return false;  // looped!

    // Okay, we have never been in this block before.  Check to see if there
    // are any PHI nodes.  If so, evaluate them with information about where
    // we came from.
    PHINode *PN = nullptr;
    for (CurInst = NextBB->begin();
         (PN = dyn_cast<PHINode>(CurInst)); ++CurInst)
      setVal(PN, getVal(PN->getIncomingValueForBlock(CurBB)));

    // Advance to the next block.
    CurBB = NextBB;
  }
}

