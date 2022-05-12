//===- ArgumentPromotion.cpp - Promote by-reference arguments -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass promotes "by reference" arguments to be "by value" arguments.  In
// practice, this means looking for internal functions that have pointer
// arguments.  If it can prove, through the use of alias analysis, that an
// argument is *only* loaded, then it can pass the value into the function
// instead of the address of the value.  This can cause recursive simplification
// of code and lead to the elimination of allocas (especially in C++ template
// code like the STL).
//
// This pass also handles aggregate arguments that are passed into a function,
// scalarizing them if the elements of the aggregate are only loaded.  Note that
// by default it refuses to scalarize aggregates which would require passing in
// more than three operands to the function, because passing thousands of
// operands for a large array or structure is unprofitable! This limit can be
// configured or disabled, however.
//
// Note that this transformation could also be done for arguments that are only
// stored to (returning the value instead), but does not currently.  This case
// would be best handled when and if LLVM begins supporting multiple return
// values from functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "argpromotion"

STATISTIC(NumArgumentsPromoted, "Number of pointer arguments promoted");
STATISTIC(NumByValArgsPromoted, "Number of byval arguments promoted");
STATISTIC(NumArgumentsDead, "Number of dead pointer args eliminated");

namespace {

struct ArgPart {
  Type *Ty;
  Align Alignment;
  /// A representative guaranteed-executed load instruction for use by
  /// metadata transfer.
  LoadInst *MustExecLoad;
};

using OffsetAndArgPart = std::pair<int64_t, ArgPart>;

} // end anonymous namespace

static Value *createByteGEP(IRBuilderBase &IRB, const DataLayout &DL,
                            Value *Ptr, Type *ResElemTy, int64_t Offset) {
  // For non-opaque pointers, try to create a "nice" GEP if possible, otherwise
  // fall back to an i8 GEP to a specific offset.
  unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
  APInt OrigOffset(DL.getIndexTypeSizeInBits(Ptr->getType()), Offset);
  if (!Ptr->getType()->isOpaquePointerTy()) {
    Type *OrigElemTy = Ptr->getType()->getNonOpaquePointerElementType();
    if (OrigOffset == 0 && OrigElemTy == ResElemTy)
      return Ptr;

    if (OrigElemTy->isSized()) {
      APInt TmpOffset = OrigOffset;
      Type *TmpTy = OrigElemTy;
      SmallVector<APInt> IntIndices =
          DL.getGEPIndicesForOffset(TmpTy, TmpOffset);
      if (TmpOffset == 0) {
        // Try to add trailing zero indices to reach the right type.
        while (TmpTy != ResElemTy) {
          Type *NextTy = GetElementPtrInst::getTypeAtIndex(TmpTy, (uint64_t)0);
          if (!NextTy)
            break;

          IntIndices.push_back(APInt::getZero(
              isa<StructType>(TmpTy) ? 32 : OrigOffset.getBitWidth()));
          TmpTy = NextTy;
        }

        SmallVector<Value *> Indices;
        for (const APInt &Index : IntIndices)
          Indices.push_back(IRB.getInt(Index));

        if (OrigOffset != 0 || TmpTy == ResElemTy) {
          Ptr = IRB.CreateGEP(OrigElemTy, Ptr, Indices);
          return IRB.CreateBitCast(Ptr, ResElemTy->getPointerTo(AddrSpace));
        }
      }
    }
  }

  if (OrigOffset != 0) {
    Ptr = IRB.CreateBitCast(Ptr, IRB.getInt8PtrTy(AddrSpace));
    Ptr = IRB.CreateGEP(IRB.getInt8Ty(), Ptr, IRB.getInt(OrigOffset));
  }
  return IRB.CreateBitCast(Ptr, ResElemTy->getPointerTo(AddrSpace));
}

/// DoPromotion - This method actually performs the promotion of the specified
/// arguments, and returns the new function.  At this point, we know that it's
/// safe to do so.
static Function *doPromotion(
    Function *F,
    const DenseMap<Argument *, SmallVector<OffsetAndArgPart, 4>> &ArgsToPromote,
    SmallPtrSetImpl<Argument *> &ByValArgsToTransform,
    Optional<function_ref<void(CallBase &OldCS, CallBase &NewCS)>>
        ReplaceCallSite) {
  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has modified arguments.
  FunctionType *FTy = F->getFunctionType();
  std::vector<Type *> Params;

  // Attribute - Keep track of the parameter attributes for the arguments
  // that we are *not* promoting. For the ones that we do promote, the parameter
  // attributes are lost
  SmallVector<AttributeSet, 8> ArgAttrVec;
  AttributeList PAL = F->getAttributes();

  // First, determine the new argument list
  unsigned ArgNo = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I, ++ArgNo) {
    if (ByValArgsToTransform.count(&*I)) {
      // Simple byval argument? Just add all the struct element types.
      Type *AgTy = I->getParamByValType();
      StructType *STy = cast<StructType>(AgTy);
      llvm::append_range(Params, STy->elements());
      ArgAttrVec.insert(ArgAttrVec.end(), STy->getNumElements(),
                        AttributeSet());
      ++NumByValArgsPromoted;
    } else if (!ArgsToPromote.count(&*I)) {
      // Unchanged argument
      Params.push_back(I->getType());
      ArgAttrVec.push_back(PAL.getParamAttrs(ArgNo));
    } else if (I->use_empty()) {
      // Dead argument (which are always marked as promotable)
      ++NumArgumentsDead;
    } else {
      const auto &ArgParts = ArgsToPromote.find(&*I)->second;
      for (const auto &Pair : ArgParts) {
        Params.push_back(Pair.second.Ty);
        ArgAttrVec.push_back(AttributeSet());
      }
      ++NumArgumentsPromoted;
    }
  }

  Type *RetTy = FTy->getReturnType();

  // Construct the new function type using the new arguments.
  FunctionType *NFTy = FunctionType::get(RetTy, Params, FTy->isVarArg());

  // Create the new function body and insert it into the module.
  Function *NF = Function::Create(NFTy, F->getLinkage(), F->getAddressSpace(),
                                  F->getName());
  NF->copyAttributesFrom(F);
  NF->copyMetadata(F, 0);

  // The new function will have the !dbg metadata copied from the original
  // function. The original function may not be deleted, and dbg metadata need
  // to be unique, so we need to drop it.
  F->setSubprogram(nullptr);

  LLVM_DEBUG(dbgs() << "ARG PROMOTION:  Promoting to:" << *NF << "\n"
                    << "From: " << *F);

  uint64_t LargestVectorWidth = 0;
  for (auto *I : Params)
    if (auto *VT = dyn_cast<llvm::VectorType>(I))
      LargestVectorWidth = std::max(
          LargestVectorWidth, VT->getPrimitiveSizeInBits().getKnownMinSize());

  // Recompute the parameter attributes list based on the new arguments for
  // the function.
  NF->setAttributes(AttributeList::get(F->getContext(), PAL.getFnAttrs(),
                                       PAL.getRetAttrs(), ArgAttrVec));
  AttributeFuncs::updateMinLegalVectorWidthAttr(*NF, LargestVectorWidth);
  ArgAttrVec.clear();

  F->getParent()->getFunctionList().insert(F->getIterator(), NF);
  NF->takeName(F);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in the loaded pointers.
  //
  SmallVector<Value *, 16> Args;
  const DataLayout &DL = F->getParent()->getDataLayout();
  while (!F->use_empty()) {
    CallBase &CB = cast<CallBase>(*F->user_back());
    assert(CB.getCalledFunction() == F);
    const AttributeList &CallPAL = CB.getAttributes();
    IRBuilder<NoFolder> IRB(&CB);

    // Loop over the operands, inserting GEP and loads in the caller as
    // appropriate.
    auto *AI = CB.arg_begin();
    ArgNo = 0;
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
         ++I, ++AI, ++ArgNo)
      if (!ArgsToPromote.count(&*I) && !ByValArgsToTransform.count(&*I)) {
        Args.push_back(*AI); // Unmodified argument
        ArgAttrVec.push_back(CallPAL.getParamAttrs(ArgNo));
      } else if (ByValArgsToTransform.count(&*I)) {
        // Emit a GEP and load for each element of the struct.
        Type *AgTy = I->getParamByValType();
        StructType *STy = cast<StructType>(AgTy);
        Value *Idxs[2] = {
            ConstantInt::get(Type::getInt32Ty(F->getContext()), 0), nullptr};
        const StructLayout *SL = DL.getStructLayout(STy);
        Align StructAlign = *I->getParamAlign();
        for (unsigned J = 0, Elems = STy->getNumElements(); J != Elems; ++J) {
          Idxs[1] = ConstantInt::get(Type::getInt32Ty(F->getContext()), J);
          auto *Idx =
              IRB.CreateGEP(STy, *AI, Idxs, (*AI)->getName() + "." + Twine(J));
          // TODO: Tell AA about the new values?
          Align Alignment =
              commonAlignment(StructAlign, SL->getElementOffset(J));
          Args.push_back(IRB.CreateAlignedLoad(
              STy->getElementType(J), Idx, Alignment, Idx->getName() + ".val"));
          ArgAttrVec.push_back(AttributeSet());
        }
      } else if (!I->use_empty()) {
        Value *V = *AI;
        const auto &ArgParts = ArgsToPromote.find(&*I)->second;
        for (const auto &Pair : ArgParts) {
          LoadInst *LI = IRB.CreateAlignedLoad(
              Pair.second.Ty,
              createByteGEP(IRB, DL, V, Pair.second.Ty, Pair.first),
              Pair.second.Alignment, V->getName() + ".val");
          if (Pair.second.MustExecLoad) {
            LI->setAAMetadata(Pair.second.MustExecLoad->getAAMetadata());
            LI->copyMetadata(*Pair.second.MustExecLoad,
                             {LLVMContext::MD_range, LLVMContext::MD_nonnull,
                              LLVMContext::MD_dereferenceable,
                              LLVMContext::MD_dereferenceable_or_null,
                              LLVMContext::MD_align, LLVMContext::MD_noundef});
          }
          Args.push_back(LI);
          ArgAttrVec.push_back(AttributeSet());
        }
      }

    // Push any varargs arguments on the list.
    for (; AI != CB.arg_end(); ++AI, ++ArgNo) {
      Args.push_back(*AI);
      ArgAttrVec.push_back(CallPAL.getParamAttrs(ArgNo));
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CB.getOperandBundlesAsDefs(OpBundles);

    CallBase *NewCS = nullptr;
    if (InvokeInst *II = dyn_cast<InvokeInst>(&CB)) {
      NewCS = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                                 Args, OpBundles, "", &CB);
    } else {
      auto *NewCall = CallInst::Create(NF, Args, OpBundles, "", &CB);
      NewCall->setTailCallKind(cast<CallInst>(&CB)->getTailCallKind());
      NewCS = NewCall;
    }
    NewCS->setCallingConv(CB.getCallingConv());
    NewCS->setAttributes(AttributeList::get(F->getContext(),
                                            CallPAL.getFnAttrs(),
                                            CallPAL.getRetAttrs(), ArgAttrVec));
    NewCS->copyMetadata(CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});
    Args.clear();
    ArgAttrVec.clear();

    AttributeFuncs::updateMinLegalVectorWidthAttr(*CB.getCaller(),
                                                  LargestVectorWidth);

    // Update the callgraph to know that the callsite has been transformed.
    if (ReplaceCallSite)
      (*ReplaceCallSite)(CB, *NewCS);

    if (!CB.use_empty()) {
      CB.replaceAllUsesWith(NewCS);
      NewCS->takeName(&CB);
    }

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    CB.eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transferring uses of the old arguments over to
  // the new arguments, also transferring over the names as well.
  Function::arg_iterator I2 = NF->arg_begin();
  for (Argument &Arg : F->args()) {
    if (!ArgsToPromote.count(&Arg) && !ByValArgsToTransform.count(&Arg)) {
      // If this is an unmodified argument, move the name and users over to the
      // new version.
      Arg.replaceAllUsesWith(&*I2);
      I2->takeName(&Arg);
      ++I2;
      continue;
    }

    if (ByValArgsToTransform.count(&Arg)) {
      // In the callee, we create an alloca, and store each of the new incoming
      // arguments into the alloca.
      Instruction *InsertPt = &NF->begin()->front();

      // Just add all the struct element types.
      Type *AgTy = Arg.getParamByValType();
      Align StructAlign = *Arg.getParamAlign();
      Value *TheAlloca = new AllocaInst(AgTy, DL.getAllocaAddrSpace(), nullptr,
                                        StructAlign, "", InsertPt);
      StructType *STy = cast<StructType>(AgTy);
      Value *Idxs[2] = {ConstantInt::get(Type::getInt32Ty(F->getContext()), 0),
                        nullptr};
      const StructLayout *SL = DL.getStructLayout(STy);

      for (unsigned J = 0, Elems = STy->getNumElements(); J != Elems; ++J) {
        Idxs[1] = ConstantInt::get(Type::getInt32Ty(F->getContext()), J);
        Value *Idx = GetElementPtrInst::Create(
            AgTy, TheAlloca, Idxs, TheAlloca->getName() + "." + Twine(J),
            InsertPt);
        I2->setName(Arg.getName() + "." + Twine(J));
        Align Alignment = commonAlignment(StructAlign, SL->getElementOffset(J));
        new StoreInst(&*I2++, Idx, false, Alignment, InsertPt);
      }

      // Anything that used the arg should now use the alloca.
      Arg.replaceAllUsesWith(TheAlloca);
      TheAlloca->takeName(&Arg);
      continue;
    }

    // There potentially are metadata uses for things like llvm.dbg.value.
    // Replace them with undef, after handling the other regular uses.
    auto RauwUndefMetadata = make_scope_exit(
        [&]() { Arg.replaceAllUsesWith(UndefValue::get(Arg.getType())); });

    if (Arg.use_empty())
      continue;

    SmallDenseMap<int64_t, Argument *> OffsetToArg;
    for (const auto &Pair : ArgsToPromote.find(&Arg)->second) {
      Argument &NewArg = *I2++;
      NewArg.setName(Arg.getName() + "." + Twine(Pair.first) + ".val");
      OffsetToArg.insert({Pair.first, &NewArg});
    }

    // Otherwise, if we promoted this argument, then all users are load
    // instructions (with possible casts and GEPs in between).

    SmallVector<Value *, 16> Worklist;
    SmallVector<Instruction *, 16> DeadInsts;
    append_range(Worklist, Arg.users());
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (isa<BitCastInst>(V) || isa<GetElementPtrInst>(V)) {
        DeadInsts.push_back(cast<Instruction>(V));
        append_range(Worklist, V->users());
        continue;
      }

      if (auto *LI = dyn_cast<LoadInst>(V)) {
        Value *Ptr = LI->getPointerOperand();
        APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
        Ptr =
            Ptr->stripAndAccumulateConstantOffsets(DL, Offset,
                                                   /* AllowNonInbounds */ true);
        assert(Ptr == &Arg && "Not constant offset from arg?");
        LI->replaceAllUsesWith(OffsetToArg[Offset.getSExtValue()]);
        DeadInsts.push_back(LI);
        continue;
      }

      llvm_unreachable("Unexpected user");
    }

    for (Instruction *I : DeadInsts) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  return NF;
}

/// Return true if we can prove that all callees pass in a valid pointer for the
/// specified function argument.
static bool allCallersPassValidPointerForArgument(Argument *Arg,
                                                  Align NeededAlign,
                                                  uint64_t NeededDerefBytes) {
  Function *Callee = Arg->getParent();
  const DataLayout &DL = Callee->getParent()->getDataLayout();
  APInt Bytes(64, NeededDerefBytes);

  // Check if the argument itself is marked dereferenceable and aligned.
  if (isDereferenceableAndAlignedPointer(Arg, NeededAlign, Bytes, DL))
    return true;

  // Look at all call sites of the function.  At this point we know we only have
  // direct callees.
  return all_of(Callee->users(), [&](User *U) {
    CallBase &CB = cast<CallBase>(*U);
    return isDereferenceableAndAlignedPointer(
        CB.getArgOperand(Arg->getArgNo()), NeededAlign, Bytes, DL);
  });
}

/// Determine that this argument is safe to promote, and find the argument
/// parts it can be promoted into.
static bool findArgParts(Argument *Arg, const DataLayout &DL, AAResults &AAR,
                         unsigned MaxElements, bool IsRecursive,
                         SmallVectorImpl<OffsetAndArgPart> &ArgPartsVec) {
  // Quick exit for unused arguments
  if (Arg->use_empty())
    return true;

  // We can only promote this argument if all of the uses are loads at known
  // offsets.
  //
  // Promoting the argument causes it to be loaded in the caller
  // unconditionally. This is only safe if we can prove that either the load
  // would have happened in the callee anyway (ie, there is a load in the entry
  // block) or the pointer passed in at every call site is guaranteed to be
  // valid.
  // In the former case, invalid loads can happen, but would have happened
  // anyway, in the latter case, invalid loads won't happen. This prevents us
  // from introducing an invalid load that wouldn't have happened in the
  // original code.

  SmallDenseMap<int64_t, ArgPart, 4> ArgParts;
  Align NeededAlign(1);
  uint64_t NeededDerefBytes = 0;

  // Returns None if this load is not based on the argument. Return true if
  // we can promote the load, false otherwise.
  auto HandleLoad = [&](LoadInst *LI,
                        bool GuaranteedToExecute) -> Optional<bool> {
    // Don't promote volatile or atomic loads.
    if (!LI->isSimple())
      return false;

    Value *Ptr = LI->getPointerOperand();
    APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
    Ptr = Ptr->stripAndAccumulateConstantOffsets(DL, Offset,
                                                 /* AllowNonInbounds */ true);
    if (Ptr != Arg)
      return None;

    if (Offset.getSignificantBits() >= 64)
      return false;

    Type *Ty = LI->getType();
    TypeSize Size = DL.getTypeStoreSize(Ty);
    // Don't try to promote scalable types.
    if (Size.isScalable())
      return false;

    // If this is a recursive function and one of the types is a pointer,
    // then promoting it might lead to recursive promotion.
    if (IsRecursive && Ty->isPointerTy())
      return false;

    int64_t Off = Offset.getSExtValue();
    auto Pair = ArgParts.try_emplace(
        Off, ArgPart{Ty, LI->getAlign(), GuaranteedToExecute ? LI : nullptr});
    ArgPart &Part = Pair.first->second;
    bool OffsetNotSeenBefore = Pair.second;

    // We limit promotion to only promoting up to a fixed number of elements of
    // the aggregate.
    if (MaxElements > 0 && ArgParts.size() > MaxElements) {
      LLVM_DEBUG(dbgs() << "ArgPromotion of " << *Arg << " failed: "
                        << "more than " << MaxElements << " parts\n");
      return false;
    }

    // For now, we only support loading one specific type at a given offset.
    if (Part.Ty != Ty) {
      LLVM_DEBUG(dbgs() << "ArgPromotion of " << *Arg << " failed: "
                        << "loaded via both " << *Part.Ty << " and " << *Ty
                        << " at offset " << Off << "\n");
      return false;
    }

    // If this load is not guaranteed to execute, and we haven't seen a load at
    // this offset before (or it had lower alignment), then we need to remember
    // that requirement.
    // Note that skipping loads of previously seen offsets is only correct
    // because we only allow a single type for a given offset, which also means
    // that the number of accessed bytes will be the same.
    if (!GuaranteedToExecute &&
        (OffsetNotSeenBefore || Part.Alignment < LI->getAlign())) {
      // We won't be able to prove dereferenceability for negative offsets.
      if (Off < 0)
        return false;

      // If the offset is not aligned, an aligned base pointer won't help.
      if (!isAligned(LI->getAlign(), Off))
        return false;

      NeededDerefBytes = std::max(NeededDerefBytes, Off + Size.getFixedValue());
      NeededAlign = std::max(NeededAlign, LI->getAlign());
    }

    Part.Alignment = std::max(Part.Alignment, LI->getAlign());
    return true;
  };

  // Look for loads that are guaranteed to execute on entry.
  for (Instruction &I : Arg->getParent()->getEntryBlock()) {
    if (LoadInst *LI = dyn_cast<LoadInst>(&I))
      if (Optional<bool> Res = HandleLoad(LI, /* GuaranteedToExecute */ true))
        if (!*Res)
          return false;

    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      break;
  }

  // Now look at all loads of the argument. Remember the load instructions
  // for the aliasing check below.
  SmallVector<Value *, 16> Worklist;
  SmallPtrSet<Value *, 16> Visited;
  SmallVector<LoadInst *, 16> Loads;
  auto AppendUsers = [&](Value *V) {
    for (User *U : V->users())
      if (Visited.insert(U).second)
        Worklist.push_back(U);
  };
  AppendUsers(Arg);
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (isa<BitCastInst>(V)) {
      AppendUsers(V);
      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if (!GEP->hasAllConstantIndices())
        return false;
      AppendUsers(V);
      continue;
    }

    if (auto *LI = dyn_cast<LoadInst>(V)) {
      if (!*HandleLoad(LI, /* GuaranteedToExecute */ false))
        return false;
      Loads.push_back(LI);
      continue;
    }

    // Unknown user.
    LLVM_DEBUG(dbgs() << "ArgPromotion of " << *Arg << " failed: "
                      << "unknown user " << *V << "\n");
    return false;
  }

  if (NeededDerefBytes || NeededAlign > 1) {
    // Try to prove a required deref / aligned requirement.
    if (!allCallersPassValidPointerForArgument(Arg, NeededAlign,
                                               NeededDerefBytes)) {
      LLVM_DEBUG(dbgs() << "ArgPromotion of " << *Arg << " failed: "
                        << "not dereferenceable or aligned\n");
      return false;
    }
  }

  if (ArgParts.empty())
    return true; // No users, this is a dead argument.

  // Sort parts by offset.
  append_range(ArgPartsVec, ArgParts);
  sort(ArgPartsVec,
       [](const auto &A, const auto &B) { return A.first < B.first; });

  // Make sure the parts are non-overlapping.
  // TODO: As we're doing pure load promotion here, overlap should be fine from
  // a correctness perspective. Profitability is less obvious though.
  int64_t Offset = ArgPartsVec[0].first;
  for (const auto &Pair : ArgPartsVec) {
    if (Pair.first < Offset)
      return false; // Overlap with previous part.

    Offset = Pair.first + DL.getTypeStoreSize(Pair.second.Ty);
  }

  // Okay, now we know that the argument is only used by load instructions, and
  // it is safe to unconditionally perform all of them. Use alias analysis to
  // check to see if the pointer is guaranteed to not be modified from entry of
  // the function to each of the load instructions.

  // Because there could be several/many load instructions, remember which
  // blocks we know to be transparent to the load.
  df_iterator_default_set<BasicBlock *, 16> TranspBlocks;

  for (LoadInst *Load : Loads) {
    // Check to see if the load is invalidated from the start of the block to
    // the load itself.
    BasicBlock *BB = Load->getParent();

    MemoryLocation Loc = MemoryLocation::get(Load);
    if (AAR.canInstructionRangeModRef(BB->front(), *Load, Loc, ModRefInfo::Mod))
      return false; // Pointer is invalidated!

    // Now check every path from the entry block to the load for transparency.
    // To do this, we perform a depth first search on the inverse CFG from the
    // loading block.
    for (BasicBlock *P : predecessors(BB)) {
      for (BasicBlock *TranspBB : inverse_depth_first_ext(P, TranspBlocks))
        if (AAR.canBasicBlockModify(*TranspBB, Loc))
          return false;
    }
  }

  // If the path from the entry of the function to each load is free of
  // instructions that potentially invalidate the load, we can make the
  // transformation!
  return true;
}

bool ArgumentPromotionPass::isDenselyPacked(Type *Ty, const DataLayout &DL) {
  // There is no size information, so be conservative.
  if (!Ty->isSized())
    return false;

  // If the alloc size is not equal to the storage size, then there are padding
  // bytes. For x86_fp80 on x86-64, size: 80 alloc size: 128.
  if (DL.getTypeSizeInBits(Ty) != DL.getTypeAllocSizeInBits(Ty))
    return false;

  // FIXME: This isn't the right way to check for padding in vectors with
  // non-byte-size elements.
  if (VectorType *SeqTy = dyn_cast<VectorType>(Ty))
    return isDenselyPacked(SeqTy->getElementType(), DL);

  // For array types, check for padding within members.
  if (ArrayType *SeqTy = dyn_cast<ArrayType>(Ty))
    return isDenselyPacked(SeqTy->getElementType(), DL);

  if (!isa<StructType>(Ty))
    return true;

  // Check for padding within and between elements of a struct.
  StructType *StructTy = cast<StructType>(Ty);
  const StructLayout *Layout = DL.getStructLayout(StructTy);
  uint64_t StartPos = 0;
  for (unsigned I = 0, E = StructTy->getNumElements(); I < E; ++I) {
    Type *ElTy = StructTy->getElementType(I);
    if (!isDenselyPacked(ElTy, DL))
      return false;
    if (StartPos != Layout->getElementOffsetInBits(I))
      return false;
    StartPos += DL.getTypeAllocSizeInBits(ElTy);
  }

  return true;
}

/// Checks if the padding bytes of an argument could be accessed.
static bool canPaddingBeAccessed(Argument *Arg) {
  assert(Arg->hasByValAttr());

  // Track all the pointers to the argument to make sure they are not captured.
  SmallPtrSet<Value *, 16> PtrValues;
  PtrValues.insert(Arg);

  // Track all of the stores.
  SmallVector<StoreInst *, 16> Stores;

  // Scan through the uses recursively to make sure the pointer is always used
  // sanely.
  SmallVector<Value *, 16> WorkList(Arg->users());
  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (isa<GetElementPtrInst>(V) || isa<PHINode>(V)) {
      if (PtrValues.insert(V).second)
        append_range(WorkList, V->users());
    } else if (StoreInst *Store = dyn_cast<StoreInst>(V)) {
      Stores.push_back(Store);
    } else if (!isa<LoadInst>(V)) {
      return true;
    }
  }

  // Check to make sure the pointers aren't captured
  for (StoreInst *Store : Stores)
    if (PtrValues.count(Store->getValueOperand()))
      return true;

  return false;
}

/// Check if callers and callee agree on how promoted arguments would be
/// passed.
static bool areTypesABICompatible(ArrayRef<Type *> Types, const Function &F,
                                  const TargetTransformInfo &TTI) {
  return all_of(F.uses(), [&](const Use &U) {
    CallBase *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB)
      return false;

    const Function *Caller = CB->getCaller();
    const Function *Callee = CB->getCalledFunction();
    return TTI.areTypesABICompatible(Caller, Callee, Types);
  });
}

/// PromoteArguments - This method checks the specified function to see if there
/// are any promotable arguments and if it is safe to promote the function (for
/// example, all callers are direct).  If safe to promote some arguments, it
/// calls the DoPromotion method.
static Function *
promoteArguments(Function *F, function_ref<AAResults &(Function &F)> AARGetter,
                 unsigned MaxElements,
                 Optional<function_ref<void(CallBase &OldCS, CallBase &NewCS)>>
                     ReplaceCallSite,
                 const TargetTransformInfo &TTI, bool IsRecursive) {
  // Don't perform argument promotion for naked functions; otherwise we can end
  // up removing parameters that are seemingly 'not used' as they are referred
  // to in the assembly.
  if(F->hasFnAttribute(Attribute::Naked))
    return nullptr;

  // Make sure that it is local to this module.
  if (!F->hasLocalLinkage())
    return nullptr;

  // Don't promote arguments for variadic functions. Adding, removing, or
  // changing non-pack parameters can change the classification of pack
  // parameters. Frontends encode that classification at the call site in the
  // IR, while in the callee the classification is determined dynamically based
  // on the number of registers consumed so far.
  if (F->isVarArg())
    return nullptr;

  // Don't transform functions that receive inallocas, as the transformation may
  // not be safe depending on calling convention.
  if (F->getAttributes().hasAttrSomewhere(Attribute::InAlloca))
    return nullptr;

  // First check: see if there are any pointer arguments!  If not, quick exit.
  SmallVector<Argument *, 16> PointerArgs;
  for (Argument &I : F->args())
    if (I.getType()->isPointerTy())
      PointerArgs.push_back(&I);
  if (PointerArgs.empty())
    return nullptr;

  // Second check: make sure that all callers are direct callers.  We can't
  // transform functions that have indirect callers.  Also see if the function
  // is self-recursive.
  for (Use &U : F->uses()) {
    CallBase *CB = dyn_cast<CallBase>(U.getUser());
    // Must be a direct call.
    if (CB == nullptr || !CB->isCallee(&U) ||
        CB->getFunctionType() != F->getFunctionType())
      return nullptr;

    // Can't change signature of musttail callee
    if (CB->isMustTailCall())
      return nullptr;

    if (CB->getFunction() == F)
      IsRecursive = true;
  }

  // Can't change signature of musttail caller
  // FIXME: Support promoting whole chain of musttail functions
  for (BasicBlock &BB : *F)
    if (BB.getTerminatingMustTailCall())
      return nullptr;

  const DataLayout &DL = F->getParent()->getDataLayout();

  AAResults &AAR = AARGetter(*F);

  // Check to see which arguments are promotable.  If an argument is promotable,
  // add it to ArgsToPromote.
  DenseMap<Argument *, SmallVector<OffsetAndArgPart, 4>> ArgsToPromote;
  SmallPtrSet<Argument *, 8> ByValArgsToTransform;
  for (Argument *PtrArg : PointerArgs) {
    // Replace sret attribute with noalias. This reduces register pressure by
    // avoiding a register copy.
    if (PtrArg->hasStructRetAttr()) {
      unsigned ArgNo = PtrArg->getArgNo();
      F->removeParamAttr(ArgNo, Attribute::StructRet);
      F->addParamAttr(ArgNo, Attribute::NoAlias);
      for (Use &U : F->uses()) {
        CallBase &CB = cast<CallBase>(*U.getUser());
        CB.removeParamAttr(ArgNo, Attribute::StructRet);
        CB.addParamAttr(ArgNo, Attribute::NoAlias);
      }
    }

    // If we can promote the pointer to its value.
    SmallVector<OffsetAndArgPart, 4> ArgParts;
    if (findArgParts(PtrArg, DL, AAR, MaxElements, IsRecursive, ArgParts)) {
      SmallVector<Type *, 4> Types;
      for (const auto &Pair : ArgParts)
        Types.push_back(Pair.second.Ty);

      if (areTypesABICompatible(Types, *F, TTI)) {
        ArgsToPromote.insert({PtrArg, std::move(ArgParts)});
        continue;
      }
    }

    // Otherwise, if this is a byval argument, and if the aggregate type is
    // small, just pass the elements, which is always safe, if the passed value
    // is densely packed or if we can prove the padding bytes are never
    // accessed.
    //
    // Only handle arguments with specified alignment; if it's unspecified, the
    // actual alignment of the argument is target-specific.
    Type *ByValTy = PtrArg->getParamByValType();
    bool IsSafeToPromote =
        ByValTy && PtrArg->getParamAlign() &&
        (ArgumentPromotionPass::isDenselyPacked(ByValTy, DL) ||
         !canPaddingBeAccessed(PtrArg));
    if (!IsSafeToPromote) {
      LLVM_DEBUG(dbgs() << "ArgPromotion disables passing the elements of"
                        << " the argument '" << PtrArg->getName()
                        << "' because it is not safe.\n");
      continue;
    }
    if (StructType *STy = dyn_cast<StructType>(ByValTy)) {
      if (MaxElements > 0 && STy->getNumElements() > MaxElements) {
        LLVM_DEBUG(dbgs() << "ArgPromotion disables passing the elements of"
                          << " the argument '" << PtrArg->getName()
                          << "' because it would require adding more"
                          << " than " << MaxElements
                          << " arguments to the function.\n");
        continue;
      }
      SmallVector<Type *, 4> Types;
      append_range(Types, STy->elements());

      // If all the elements are single-value types, we can promote it.
      bool AllSimple =
          all_of(Types, [](Type *Ty) { return Ty->isSingleValueType(); });

      // Safe to transform. Passing the elements as a scalar will allow sroa to
      // hack on the new alloca we introduce.
      if (AllSimple && areTypesABICompatible(Types, *F, TTI))
        ByValArgsToTransform.insert(PtrArg);
    }
  }

  // No promotable pointer arguments.
  if (ArgsToPromote.empty() && ByValArgsToTransform.empty())
    return nullptr;

  return doPromotion(F, ArgsToPromote, ByValArgsToTransform, ReplaceCallSite);
}

PreservedAnalyses ArgumentPromotionPass::run(LazyCallGraph::SCC &C,
                                             CGSCCAnalysisManager &AM,
                                             LazyCallGraph &CG,
                                             CGSCCUpdateResult &UR) {
  bool Changed = false, LocalChange;

  // Iterate until we stop promoting from this SCC.
  do {
    LocalChange = false;

    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

    bool IsRecursive = C.size() > 1;
    for (LazyCallGraph::Node &N : C) {
      Function &OldF = N.getFunction();

      // FIXME: This lambda must only be used with this function. We should
      // skip the lambda and just get the AA results directly.
      auto AARGetter = [&](Function &F) -> AAResults & {
        assert(&F == &OldF && "Called with an unexpected function!");
        return FAM.getResult<AAManager>(F);
      };

      const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(OldF);
      Function *NewF = promoteArguments(&OldF, AARGetter, MaxElements, None,
                                        TTI, IsRecursive);
      if (!NewF)
        continue;
      LocalChange = true;

      // Directly substitute the functions in the call graph. Note that this
      // requires the old function to be completely dead and completely
      // replaced by the new function. It does no call graph updates, it merely
      // swaps out the particular function mapped to a particular node in the
      // graph.
      C.getOuterRefSCC().replaceNodeFunction(N, *NewF);
      FAM.clear(OldF, OldF.getName());
      OldF.eraseFromParent();

      PreservedAnalyses FuncPA;
      FuncPA.preserveSet<CFGAnalyses>();
      for (auto *U : NewF->users()) {
        auto *UserF = cast<CallBase>(U)->getFunction();
        FAM.invalidate(*UserF, FuncPA);
      }
    }

    Changed |= LocalChange;
  } while (LocalChange);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  // We've cleared out analyses for deleted functions.
  PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
  // We've manually invalidated analyses for functions we've modified.
  PA.preserveSet<AllAnalysesOn<Function>>();
  return PA;
}

namespace {

/// ArgPromotion - The 'by reference' to 'by value' argument promotion pass.
struct ArgPromotion : public CallGraphSCCPass {
  // Pass identification, replacement for typeid
  static char ID;

  explicit ArgPromotion(unsigned MaxElements = 3)
      : CallGraphSCCPass(ID), MaxElements(MaxElements) {
    initializeArgPromotionPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getAAResultsAnalysisUsage(AU);
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool runOnSCC(CallGraphSCC &SCC) override;

private:
  using llvm::Pass::doInitialization;

  bool doInitialization(CallGraph &CG) override;

  /// The maximum number of elements to expand, or 0 for unlimited.
  unsigned MaxElements;
};

} // end anonymous namespace

char ArgPromotion::ID = 0;

INITIALIZE_PASS_BEGIN(ArgPromotion, "argpromotion",
                      "Promote 'by reference' arguments to scalars", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ArgPromotion, "argpromotion",
                    "Promote 'by reference' arguments to scalars", false, false)

Pass *llvm::createArgumentPromotionPass(unsigned MaxElements) {
  return new ArgPromotion(MaxElements);
}

bool ArgPromotion::runOnSCC(CallGraphSCC &SCC) {
  if (skipSCC(SCC))
    return false;

  // Get the callgraph information that we need to update to reflect our
  // changes.
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();

  LegacyAARGetter AARGetter(*this);

  bool Changed = false, LocalChange;

  // Iterate until we stop promoting from this SCC.
  do {
    LocalChange = false;
    // Attempt to promote arguments from all functions in this SCC.
    bool IsRecursive = SCC.size() > 1;
    for (CallGraphNode *OldNode : SCC) {
      Function *OldF = OldNode->getFunction();
      if (!OldF)
        continue;

      auto ReplaceCallSite = [&](CallBase &OldCS, CallBase &NewCS) {
        Function *Caller = OldCS.getParent()->getParent();
        CallGraphNode *NewCalleeNode =
            CG.getOrInsertFunction(NewCS.getCalledFunction());
        CallGraphNode *CallerNode = CG[Caller];
        CallerNode->replaceCallEdge(cast<CallBase>(OldCS),
                                    cast<CallBase>(NewCS), NewCalleeNode);
      };

      const TargetTransformInfo &TTI =
          getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*OldF);
      if (Function *NewF =
              promoteArguments(OldF, AARGetter, MaxElements, {ReplaceCallSite},
                               TTI, IsRecursive)) {
        LocalChange = true;

        // Update the call graph for the newly promoted function.
        CallGraphNode *NewNode = CG.getOrInsertFunction(NewF);
        NewNode->stealCalledFunctionsFrom(OldNode);
        if (OldNode->getNumReferences() == 0)
          delete CG.removeFunctionFromModule(OldNode);
        else
          OldF->setLinkage(Function::ExternalLinkage);

        // And update the SCC we're iterating as well.
        SCC.ReplaceNode(OldNode, NewNode);
      }
    }
    // Remember that we changed something.
    Changed |= LocalChange;
  } while (LocalChange);

  return Changed;
}

bool ArgPromotion::doInitialization(CallGraph &CG) {
  return CallGraphSCCPass::doInitialization(CG);
}
