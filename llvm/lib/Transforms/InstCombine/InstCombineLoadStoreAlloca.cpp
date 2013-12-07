//===- InstCombineLoadStoreAlloca.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for load, store and alloca.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

STATISTIC(NumDeadStore,    "Number of dead stores eliminated");
STATISTIC(NumGlobalCopies, "Number of allocas copied from constant global");

/// pointsToConstantGlobal - Return true if V (possibly indirectly) points to
/// some part of a constant global variable.  This intentionally only accepts
/// constant expressions because we can't rewrite arbitrary instructions.
static bool pointsToConstantGlobal(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return GV->isConstant();
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::BitCast ||
        CE->getOpcode() == Instruction::GetElementPtr)
      return pointsToConstantGlobal(CE->getOperand(0));
  return false;
}

/// isOnlyCopiedFromConstantGlobal - Recursively walk the uses of a (derived)
/// pointer to an alloca.  Ignore any reads of the pointer, return false if we
/// see any stores or other unknown uses.  If we see pointer arithmetic, keep
/// track of whether it moves the pointer (with IsOffset) but otherwise traverse
/// the uses.  If we see a memcpy/memmove that targets an unoffseted pointer to
/// the alloca, and if the source pointer is a pointer to a constant global, we
/// can optimize this.
static bool
isOnlyCopiedFromConstantGlobal(Value *V, MemTransferInst *&TheCopy,
                               SmallVectorImpl<Instruction *> &ToDelete,
                               bool IsOffset = false) {
  // We track lifetime intrinsics as we encounter them.  If we decide to go
  // ahead and replace the value with the global, this lets the caller quickly
  // eliminate the markers.

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI!=E; ++UI) {
    User *U = cast<Instruction>(*UI);

    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // Ignore non-volatile loads, they are always ok.
      if (!LI->isSimple()) return false;
      continue;
    }

    if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      // If uses of the bitcast are ok, we are ok.
      if (!isOnlyCopiedFromConstantGlobal(BCI, TheCopy, ToDelete, IsOffset))
        return false;
      continue;
    }
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      // If the GEP has all zero indices, it doesn't offset the pointer.  If it
      // doesn't, it does.
      if (!isOnlyCopiedFromConstantGlobal(
              GEP, TheCopy, ToDelete, IsOffset || !GEP->hasAllZeroIndices()))
        return false;
      continue;
    }

    if (CallSite CS = U) {
      // If this is the function being called then we treat it like a load and
      // ignore it.
      if (CS.isCallee(UI))
        continue;

      // If this is a readonly/readnone call site, then we know it is just a
      // load (but one that potentially returns the value itself), so we can
      // ignore it if we know that the value isn't captured.
      unsigned ArgNo = CS.getArgumentNo(UI);
      if (CS.onlyReadsMemory() &&
          (CS.getInstruction()->use_empty() || CS.doesNotCapture(ArgNo)))
        continue;

      // If this is being passed as a byval argument, the caller is making a
      // copy, so it is only a read of the alloca.
      if (CS.isByValArgument(ArgNo))
        continue;
    }

    // Lifetime intrinsics can be handled by the caller.
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        assert(II->use_empty() && "Lifetime markers have no result to use!");
        ToDelete.push_back(II);
        continue;
      }
    }

    // If this is isn't our memcpy/memmove, reject it as something we can't
    // handle.
    MemTransferInst *MI = dyn_cast<MemTransferInst>(U);
    if (MI == 0)
      return false;

    // If the transfer is using the alloca as a source of the transfer, then
    // ignore it since it is a load (unless the transfer is volatile).
    if (UI.getOperandNo() == 1) {
      if (MI->isVolatile()) return false;
      continue;
    }

    // If we already have seen a copy, reject the second one.
    if (TheCopy) return false;

    // If the pointer has been offset from the start of the alloca, we can't
    // safely handle this.
    if (IsOffset) return false;

    // If the memintrinsic isn't using the alloca as the dest, reject it.
    if (UI.getOperandNo() != 0) return false;

    // If the source of the memcpy/move is not a constant global, reject it.
    if (!pointsToConstantGlobal(MI->getSource()))
      return false;

    // Otherwise, the transform is safe.  Remember the copy instruction.
    TheCopy = MI;
  }
  return true;
}

/// isOnlyCopiedFromConstantGlobal - Return true if the specified alloca is only
/// modified by a copy from a constant global.  If we can prove this, we can
/// replace any uses of the alloca with uses of the global directly.
static MemTransferInst *
isOnlyCopiedFromConstantGlobal(AllocaInst *AI,
                               SmallVectorImpl<Instruction *> &ToDelete) {
  MemTransferInst *TheCopy = 0;
  if (isOnlyCopiedFromConstantGlobal(AI, TheCopy, ToDelete))
    return TheCopy;
  return 0;
}

Instruction *InstCombiner::visitAllocaInst(AllocaInst &AI) {
  // Ensure that the alloca array size argument has type intptr_t, so that
  // any casting is exposed early.
  if (TD) {
    Type *IntPtrTy = TD->getIntPtrType(AI.getType());
    if (AI.getArraySize()->getType() != IntPtrTy) {
      Value *V = Builder->CreateIntCast(AI.getArraySize(),
                                        IntPtrTy, false);
      AI.setOperand(0, V);
      return &AI;
    }
  }

  // Convert: alloca Ty, C - where C is a constant != 1 into: alloca [C x Ty], 1
  if (AI.isArrayAllocation()) {  // Check C != 1
    if (const ConstantInt *C = dyn_cast<ConstantInt>(AI.getArraySize())) {
      Type *NewTy =
        ArrayType::get(AI.getAllocatedType(), C->getZExtValue());
      AllocaInst *New = Builder->CreateAlloca(NewTy, 0, AI.getName());
      New->setAlignment(AI.getAlignment());

      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...also skip interleaved debug info
      //
      BasicBlock::iterator It = New;
      while (isa<AllocaInst>(*It) || isa<DbgInfoIntrinsic>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      Type *IdxTy = TD
                  ? TD->getIntPtrType(AI.getType())
                  : Type::getInt64Ty(AI.getContext());
      Value *NullIdx = Constant::getNullValue(IdxTy);
      Value *Idx[2] = { NullIdx, NullIdx };
      Instruction *GEP =
        GetElementPtrInst::CreateInBounds(New, Idx, New->getName() + ".sub");
      InsertNewInstBefore(GEP, *It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, GEP);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }
  }

  if (TD && AI.getAllocatedType()->isSized()) {
    // If the alignment is 0 (unspecified), assign it the preferred alignment.
    if (AI.getAlignment() == 0)
      AI.setAlignment(TD->getPrefTypeAlignment(AI.getAllocatedType()));

    // Move all alloca's of zero byte objects to the entry block and merge them
    // together.  Note that we only do this for alloca's, because malloc should
    // allocate and return a unique pointer, even for a zero byte allocation.
    if (TD->getTypeAllocSize(AI.getAllocatedType()) == 0) {
      // For a zero sized alloca there is no point in doing an array allocation.
      // This is helpful if the array size is a complicated expression not used
      // elsewhere.
      if (AI.isArrayAllocation()) {
        AI.setOperand(0, ConstantInt::get(AI.getArraySize()->getType(), 1));
        return &AI;
      }

      // Get the first instruction in the entry block.
      BasicBlock &EntryBlock = AI.getParent()->getParent()->getEntryBlock();
      Instruction *FirstInst = EntryBlock.getFirstNonPHIOrDbg();
      if (FirstInst != &AI) {
        // If the entry block doesn't start with a zero-size alloca then move
        // this one to the start of the entry block.  There is no problem with
        // dominance as the array size was forced to a constant earlier already.
        AllocaInst *EntryAI = dyn_cast<AllocaInst>(FirstInst);
        if (!EntryAI || !EntryAI->getAllocatedType()->isSized() ||
            TD->getTypeAllocSize(EntryAI->getAllocatedType()) != 0) {
          AI.moveBefore(FirstInst);
          return &AI;
        }

        // If the alignment of the entry block alloca is 0 (unspecified),
        // assign it the preferred alignment.
        if (EntryAI->getAlignment() == 0)
          EntryAI->setAlignment(
            TD->getPrefTypeAlignment(EntryAI->getAllocatedType()));
        // Replace this zero-sized alloca with the one at the start of the entry
        // block after ensuring that the address will be aligned enough for both
        // types.
        unsigned MaxAlign = std::max(EntryAI->getAlignment(),
                                     AI.getAlignment());
        EntryAI->setAlignment(MaxAlign);
        if (AI.getType() != EntryAI->getType())
          return new BitCastInst(EntryAI, AI.getType());
        return ReplaceInstUsesWith(AI, EntryAI);
      }
    }
  }

  if (AI.getAlignment()) {
    // Check to see if this allocation is only modified by a memcpy/memmove from
    // a constant global whose alignment is equal to or exceeds that of the
    // allocation.  If this is the case, we can change all users to use
    // the constant global instead.  This is commonly produced by the CFE by
    // constructs like "void foo() { int A[] = {1,2,3,4,5,6,7,8,9...}; }" if 'A'
    // is only subsequently read.
    SmallVector<Instruction *, 4> ToDelete;
    if (MemTransferInst *Copy = isOnlyCopiedFromConstantGlobal(&AI, ToDelete)) {
      unsigned SourceAlign = getOrEnforceKnownAlignment(Copy->getSource(),
                                                        AI.getAlignment(), TD);
      if (AI.getAlignment() <= SourceAlign) {
        DEBUG(dbgs() << "Found alloca equal to global: " << AI << '\n');
        DEBUG(dbgs() << "  memcpy = " << *Copy << '\n');
        for (unsigned i = 0, e = ToDelete.size(); i != e; ++i)
          EraseInstFromFunction(*ToDelete[i]);
        Constant *TheSrc = cast<Constant>(Copy->getSource());
        Constant *Cast
          = ConstantExpr::getPointerBitCastOrAddrSpaceCast(TheSrc, AI.getType());
        Instruction *NewI = ReplaceInstUsesWith(AI, Cast);
        EraseInstFromFunction(*Copy);
        ++NumGlobalCopies;
        return NewI;
      }
    }
  }

  // At last, use the generic allocation site handler to aggressively remove
  // unused allocas.
  return visitAllocSite(AI);
}


/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI,
                                        const DataLayout *TD) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  PointerType *DestTy = cast<PointerType>(CI->getType());
  Type *DestPTy = DestTy->getElementType();
  if (PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {

    // If the address spaces don't match, don't eliminate the cast.
    if (DestTy->getAddressSpace() != SrcTy->getAddressSpace())
      return 0;

    Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isIntegerTy() || DestPTy->isPointerTy() ||
         DestPTy->isVectorTy()) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            Type *IdxTy = TD
                        ? TD->getIntPtrType(SrcTy)
                        : Type::getInt64Ty(SrcTy->getContext());
            Value *Idx = Constant::getNullValue(IdxTy);
            Value *Idxs[2] = { Idx, Idx };
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if (IC.getDataLayout() &&
          (SrcPTy->isIntegerTy() || SrcPTy->isPointerTy() ||
            SrcPTy->isVectorTy()) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (SrcPTy->isPtrOrPtrVectorTy() ==
           LI.getType()->isPtrOrPtrVectorTy()) &&
          IC.getDataLayout()->getTypeSizeInBits(SrcPTy) ==
               IC.getDataLayout()->getTypeSizeInBits(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        LoadInst *NewLoad =
          IC.Builder->CreateLoad(CastOp, LI.isVolatile(), CI->getName());
        NewLoad->setAlignment(LI.getAlignment());
        NewLoad->setAtomic(LI.getOrdering(), LI.getSynchScope());
        // Now cast the result of the load.
        return new BitCastInst(NewLoad, LI.getType());
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      getOrEnforceKnownAlignment(Op, TD->getPrefTypeAlignment(LI.getType()),TD);
    unsigned LoadAlign = LI.getAlignment();
    unsigned EffectiveLoadAlign = LoadAlign != 0 ? LoadAlign :
      TD->getABITypeAlignment(LI.getType());

    if (KnownAlign > EffectiveLoadAlign)
      LI.setAlignment(KnownAlign);
    else if (LoadAlign == 0)
      LI.setAlignment(EffectiveLoadAlign);
  }

  // load (cast X) --> cast (load X) iff safe.
  if (isa<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
      return Res;

  // None of the following transforms are legal for volatile/atomic loads.
  // FIXME: Some of it is okay for atomic loads; needs refactoring.
  if (!LI.isSimple()) return 0;

  // Do really simple store-to-load forwarding and load CSE, to catch cases
  // where there are several consecutive memory accesses to the same location,
  // separated by a few arithmetic operations.
  BasicBlock::iterator BBI = &LI;
  if (Value *AvailableVal = FindAvailableLoadedValue(Op, LI.getParent(), BBI,6))
    return ReplaceInstUsesWith(LI, AvailableVal);

  // load(gep null, ...) -> unreachable
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    const Value *GEPI0 = GEPI->getOperand(0);
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(GEPI0) && GEPI->getPointerAddressSpace() == 0){
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }
  }

  // load null/undef -> unreachable
  // TODO: Consider a target hook for valid address spaces for this xform.
  if (isa<UndefValue>(Op) ||
      (isa<ConstantPointerNull>(Op) && LI.getPointerAddressSpace() == 0)) {
    // Insert a new store to null instruction before the load to indicate that
    // this code is not reachable.  We do this instead of inserting an
    // unreachable instruction directly because we cannot modify the CFG.
    new StoreInst(UndefValue::get(LI.getType()),
                  Constant::getNullValue(Op->getType()), &LI);
    return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
  }

  // Instcombine load (constantexpr_cast global) -> cast (load global)
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
    if (CE->isCast())
      if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
        return Res;

  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      unsigned Align = LI.getAlignment();
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI, Align, TD) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI, Align, TD)) {
        LoadInst *V1 = Builder->CreateLoad(SI->getOperand(1),
                                           SI->getOperand(1)->getName()+".val");
        LoadInst *V2 = Builder->CreateLoad(SI->getOperand(2),
                                           SI->getOperand(2)->getName()+".val");
        V1->setAlignment(Align);
        V2->setAlignment(Align);
        return SelectInst::Create(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }
    }
  }
  return 0;
}

/// InstCombineStoreToCast - Fold store V, (cast P) -> store (cast V), P
/// when possible.  This makes it generally easy to do alias analysis and/or
/// SROA/mem2reg of the memory object.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType());
  if (SrcTy == 0) return 0;

  Type *SrcPTy = SrcTy->getElementType();

  if (!DestPTy->isIntegerTy() && !DestPTy->isPointerTy())
    return 0;

  /// NewGEPIndices - If SrcPTy is an aggregate type, we can emit a "noop gep"
  /// to its first element.  This allows us to handle things like:
  ///   store i32 xxx, (bitcast {foo*, float}* %P to i32*)
  /// on 32-bit hosts.
  SmallVector<Value*, 4> NewGEPIndices;

  // If the source is an array, the code below will not succeed.  Check to
  // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
  // constants.
  if (SrcPTy->isArrayTy() || SrcPTy->isStructTy()) {
    // Index through pointer.
    Constant *Zero = Constant::getNullValue(Type::getInt32Ty(SI.getContext()));
    NewGEPIndices.push_back(Zero);

    while (1) {
      if (StructType *STy = dyn_cast<StructType>(SrcPTy)) {
        if (!STy->getNumElements()) /* Struct can be empty {} */
          break;
        NewGEPIndices.push_back(Zero);
        SrcPTy = STy->getElementType(0);
      } else if (ArrayType *ATy = dyn_cast<ArrayType>(SrcPTy)) {
        NewGEPIndices.push_back(Zero);
        SrcPTy = ATy->getElementType();
      } else {
        break;
      }
    }

    SrcTy = PointerType::get(SrcPTy, SrcTy->getAddressSpace());
  }

  if (!SrcPTy->isIntegerTy() && !SrcPTy->isPointerTy())
    return 0;

  // If the pointers point into different address spaces or if they point to
  // values with different sizes, we can't do the transformation.
  if (!IC.getDataLayout() ||
      SrcTy->getAddressSpace() !=
        cast<PointerType>(CI->getType())->getAddressSpace() ||
      IC.getDataLayout()->getTypeSizeInBits(SrcPTy) !=
      IC.getDataLayout()->getTypeSizeInBits(DestPTy))
    return 0;

  // Okay, we are casting from one integer or pointer type to another of
  // the same size.  Instead of casting the pointer before
  // the store, cast the value to be stored.
  Value *NewCast;
  Value *SIOp0 = SI.getOperand(0);
  Instruction::CastOps opcode = Instruction::BitCast;
  Type* CastSrcTy = SIOp0->getType();
  Type* CastDstTy = SrcPTy;
  if (CastDstTy->isPointerTy()) {
    if (CastSrcTy->isIntegerTy())
      opcode = Instruction::IntToPtr;
  } else if (CastDstTy->isIntegerTy()) {
    if (SIOp0->getType()->isPointerTy())
      opcode = Instruction::PtrToInt;
  }

  // SIOp0 is a pointer to aggregate and this is a store to the first field,
  // emit a GEP to index into its first field.
  if (!NewGEPIndices.empty())
    CastOp = IC.Builder->CreateInBoundsGEP(CastOp, NewGEPIndices);

  NewCast = IC.Builder->CreateCast(opcode, SIOp0, CastDstTy,
                                   SIOp0->getName()+".c");
  SI.setOperand(0, NewCast);
  SI.setOperand(1, CastOp);
  return &SI;
}

/// equivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool equivalentAddressValues(Value *A, Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;

  // Test if the values come form identical arithmetic instructions.
  // This uses isIdenticalToWhenDefined instead of isIdenticalTo because
  // its only used to compare two uses within the same basic block, which
  // means that they'll always either have the same value or one of them
  // will have an undefined value.
  if (isa<BinaryOperator>(A) ||
      isa<CastInst>(A) ||
      isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;

  // Otherwise they may not be equivalent.
  return false;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      getOrEnforceKnownAlignment(Ptr, TD->getPrefTypeAlignment(Val->getType()),
                                 TD);
    unsigned StoreAlign = SI.getAlignment();
    unsigned EffectiveStoreAlign = StoreAlign != 0 ? StoreAlign :
      TD->getABITypeAlignment(Val->getType());

    if (KnownAlign > EffectiveStoreAlign)
      SI.setAlignment(KnownAlign);
    else if (StoreAlign == 0)
      SI.setAlignment(EffectiveStoreAlign);
  }

  // Don't hack volatile/atomic stores.
  // FIXME: Some bits are legal for atomic stores; needs refactoring.
  if (!SI.isSimple()) return 0;

  // If the RHS is an alloca with a single use, zapify the store, making the
  // alloca dead.
  if (Ptr->hasOneUse()) {
    if (isa<AllocaInst>(Ptr))
      return EraseInstFromFunction(SI);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      if (isa<AllocaInst>(GEP->getOperand(0))) {
        if (GEP->getOperand(0)->hasOneUse())
          return EraseInstFromFunction(SI);
      }
    }
  }

  // Do really simple DSE, to catch cases where there are several consecutive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    // Don't count debug info directives, lest they affect codegen,
    // and we skip pointer-to-pointer bitcasts, which are NOPs.
    if (isa<DbgInfoIntrinsic>(BBI) ||
        (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      ScanInsts++;
      continue;
    }

    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (PrevSI->isSimple() && equivalentAddressValues(PrevSI->getOperand(1),
                                                        SI.getOperand(1))) {
        ++NumDeadStore;
        ++BBI;
        EraseInstFromFunction(*PrevSI);
        continue;
      }
      break;
    }

    // If this is a load, we have to stop.  However, if the loaded value is from
    // the pointer we're loading and is producing the pointer we're storing,
    // then *this* store is dead (X = load P; store X -> P).
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI == Val && equivalentAddressValues(LI->getOperand(0), Ptr) &&
          LI->isSimple())
        return EraseInstFromFunction(SI);

      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }

    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory() || BBI->mayReadFromMemory())
      break;
  }

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr) && SI.getPointerAddressSpace() == 0) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        Worklist.Add(U);  // Dropped a use.
    }
    return 0;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val))
    return EraseInstFromFunction(SI);

  // If the pointer destination is a cast, see if we can fold the cast into the
  // source instead.
  if (isa<CastInst>(Ptr))
    if (Instruction *Res = InstCombineStoreToCast(*this, SI))
      return Res;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->isCast())
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;


  // If this store is the last instruction in the basic block (possibly
  // excepting debug info instructions), and if the block ends with an
  // unconditional branch, try to move it to the successor block.
  BBI = &SI;
  do {
    ++BBI;
  } while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy()));
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional())
      if (SimplifyStoreAtEndOfBlock(SI))
        return 0;  // xform done!

  return 0;
}

/// SimplifyStoreAtEndOfBlock - Turn things like:
///   if () { *P = v1; } else { *P = v2 }
/// into a phi node with a store in the successor.
///
/// Simplify things like:
///   *P = v1; if () { *P = v2; }
/// into a phi node with a store in the successor.
///
bool InstCombiner::SimplifyStoreAtEndOfBlock(StoreInst &SI) {
  BasicBlock *StoreBB = SI.getParent();

  // Check to see if the successor block has exactly two incoming edges.  If
  // so, see if the other predecessor contains a store to the same location.
  // if so, insert a PHI node (if needed) and move the stores down.
  BasicBlock *DestBB = StoreBB->getTerminator()->getSuccessor(0);

  // Determine whether Dest has exactly two predecessors and, if so, compute
  // the other predecessor.
  pred_iterator PI = pred_begin(DestBB);
  BasicBlock *P = *PI;
  BasicBlock *OtherBB = 0;

  if (P != StoreBB)
    OtherBB = P;

  if (++PI == pred_end(DestBB))
    return false;

  P = *PI;
  if (P != StoreBB) {
    if (OtherBB)
      return false;
    OtherBB = P;
  }
  if (++PI != pred_end(DestBB))
    return false;

  // Bail out if all the relevant blocks aren't distinct (this can happen,
  // for example, if SI is in an infinite loop)
  if (StoreBB == DestBB || OtherBB == DestBB)
    return false;

  // Verify that the other block ends in a branch and is not otherwise empty.
  BasicBlock::iterator BBI = OtherBB->getTerminator();
  BranchInst *OtherBr = dyn_cast<BranchInst>(BBI);
  if (!OtherBr || BBI == OtherBB->begin())
    return false;

  // If the other block ends in an unconditional branch, check for the 'if then
  // else' case.  there is an instruction before the branch.
  StoreInst *OtherStore = 0;
  if (OtherBr->isUnconditional()) {
    --BBI;
    // Skip over debugging info.
    while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy())) {
      if (BBI==OtherBB->begin())
        return false;
      --BBI;
    }
    // If this isn't a store, isn't a store to the same location, or is not the
    // right kind of store, bail out.
    OtherStore = dyn_cast<StoreInst>(BBI);
    if (!OtherStore || OtherStore->getOperand(1) != SI.getOperand(1) ||
        !SI.isSameOperationAs(OtherStore))
      return false;
  } else {
    // Otherwise, the other block ended with a conditional branch. If one of the
    // destinations is StoreBB, then we have the if/then case.
    if (OtherBr->getSuccessor(0) != StoreBB &&
        OtherBr->getSuccessor(1) != StoreBB)
      return false;

    // Okay, we know that OtherBr now goes to Dest and StoreBB, so this is an
    // if/then triangle.  See if there is a store to the same ptr as SI that
    // lives in OtherBB.
    for (;; --BBI) {
      // Check to see if we find the matching store.
      if ((OtherStore = dyn_cast<StoreInst>(BBI))) {
        if (OtherStore->getOperand(1) != SI.getOperand(1) ||
            !SI.isSameOperationAs(OtherStore))
          return false;
        break;
      }
      // If we find something that may be using or overwriting the stored
      // value, or if we run out of instructions, we can't do the xform.
      if (BBI->mayReadFromMemory() || BBI->mayWriteToMemory() ||
          BBI == OtherBB->begin())
        return false;
    }

    // In order to eliminate the store in OtherBr, we have to
    // make sure nothing reads or overwrites the stored value in
    // StoreBB.
    for (BasicBlock::iterator I = StoreBB->begin(); &*I != &SI; ++I) {
      // FIXME: This should really be AA driven.
      if (I->mayReadFromMemory() || I->mayWriteToMemory())
        return false;
    }
  }

  // Insert a PHI node now if we need it.
  Value *MergedVal = OtherStore->getOperand(0);
  if (MergedVal != SI.getOperand(0)) {
    PHINode *PN = PHINode::Create(MergedVal->getType(), 2, "storemerge");
    PN->addIncoming(SI.getOperand(0), SI.getParent());
    PN->addIncoming(OtherStore->getOperand(0), OtherBB);
    MergedVal = InsertNewInstBefore(PN, DestBB->front());
  }

  // Advance to a place where it is safe to insert the new store and
  // insert it.
  BBI = DestBB->getFirstInsertionPt();
  StoreInst *NewSI = new StoreInst(MergedVal, SI.getOperand(1),
                                   SI.isVolatile(),
                                   SI.getAlignment(),
                                   SI.getOrdering(),
                                   SI.getSynchScope());
  InsertNewInstBefore(NewSI, *BBI);
  NewSI->setDebugLoc(OtherStore->getDebugLoc());

  // If the two stores had the same TBAA tag, preserve it.
  if (MDNode *TBAATag = SI.getMetadata(LLVMContext::MD_tbaa))
    if ((TBAATag = MDNode::getMostGenericTBAA(TBAATag,
                               OtherStore->getMetadata(LLVMContext::MD_tbaa))))
      NewSI->setMetadata(LLVMContext::MD_tbaa, TBAATag);


  // Nuke the old stores.
  EraseInstFromFunction(SI);
  EraseInstFromFunction(*OtherStore);
  return true;
}
