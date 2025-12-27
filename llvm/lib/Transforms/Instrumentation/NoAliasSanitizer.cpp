//===- NoAliasSanitizer.cpp - NoAlias violation detector -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NoAliasSanitizer pass for detecting violations of
// the noalias attribute semantics at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/NoAliasSanitizer.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "nasan"

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumSkippedReads, "Number of skipped reads (proven safe by AA)");
STATISTIC(NumSkippedWrites, "Number of skipped writes (proven safe by AA)");
STATISTIC(NumNoAliasParams, "Number of noalias parameters found");

namespace {

class NoAliasSanitizerImpl {
public:
  NoAliasSanitizerImpl(Module &M, function_ref<AAResults &(Function &)> GetAA)
      : M(M), C(M.getContext()), DL(M.getDataLayout()), GetAA(GetAA) {
    initializeRuntimeFunctions();
  }

  bool run();

private:
  Module &M;
  LLVMContext &C;
  const DataLayout &DL;
  function_ref<AAResults &(Function &)> GetAA;

  // Runtime function declarations
  FunctionCallee CreateProvenanceFn;
  FunctionCallee DestroyProvenanceFn;
  FunctionCallee SetPointerProvenanceFn;
  FunctionCallee InheritProvenanceFn;
  FunctionCallee MergeProvenanceFn;
  FunctionCallee CheckLoadFn;
  FunctionCallee CheckStoreFn;
  FunctionCallee PropagateLoadProvenanceFn;
  FunctionCallee RecordPointerStoreFn;
  FunctionCallee HandleExceptionCleanupFn;
  FunctionCallee GetPointerProvenanceFn;

  // Per-function provenance tracking: maps pointer Value* to its provenance Value*
  DenseMap<Value*, Value*> PointerProvenance;

  void initializeRuntimeFunctions();
  bool instrumentFunction(Function &F);

  // Instruction-specific instrumentation
  void instrumentGEP(GetElementPtrInst &GEP);
  void instrumentCast(CastInst &Cast);
  void instrumentPHI(PHINode &PHI, Value *PtrArray, IRBuilder<> &EntryIRB);
  void instrumentSelect(SelectInst &Sel, Value *PtrArray, IRBuilder<> &EntryIRB);
  void instrumentLoad(LoadInst &LI);
  void instrumentStore(StoreInst &SI);
  void instrumentCall(CallBase &CB);
  void instrumentMemIntrinsic(MemIntrinsic &MI);

  // Helper methods
  void insertInheritProvenance(Value *Dst, Value *Src, Instruction &I);
  bool shouldInstrumentFunction(Function &F);
  bool shouldInstrumentAccess(Value *Ptr,
                               const SmallVectorImpl<Argument *> &NoAliasArgs,
                               AAResults &AA);
  bool isNonEscapingAlloca(Value *V);

  // Get provenance for a pointer, returns 0 constant if unknown
  Value *getProvenance(Value *Ptr, IRBuilder<> &IRB);
};

void NoAliasSanitizerImpl::initializeRuntimeFunctions() {
  Type *VoidTy = Type::getVoidTy(C);
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  PointerType *PtrTy = PointerType::getUnqual(C);

  CreateProvenanceFn = M.getOrInsertFunction(
      "__nasan_create_provenance",
      Int64Ty, PtrTy, PtrTy, PtrTy, Int32Ty);

  DestroyProvenanceFn = M.getOrInsertFunction(
      "__nasan_destroy_provenance",
      VoidTy, Int64Ty);

  SetPointerProvenanceFn = M.getOrInsertFunction(
      "__nasan_set_pointer_provenance",
      VoidTy, PtrTy, Int64Ty);

  InheritProvenanceFn = M.getOrInsertFunction(
      "__nasan_inherit_provenance",
      VoidTy, PtrTy, PtrTy);

  MergeProvenanceFn = M.getOrInsertFunction(
      "__nasan_merge_provenance",
      VoidTy, PtrTy, PtrTy, Int64Ty);

  CheckLoadFn = M.getOrInsertFunction(
      "__nasan_check_load",
      VoidTy, Int64Ty, Int64Ty, Int64Ty);  // addr, size, provenance

  CheckStoreFn = M.getOrInsertFunction(
      "__nasan_check_store",
      VoidTy, Int64Ty, Int64Ty, Int64Ty);  // addr, size, provenance

  PropagateLoadProvenanceFn = M.getOrInsertFunction(
      "__nasan_propagate_through_load",
      VoidTy, PtrTy, PtrTy);

  RecordPointerStoreFn = M.getOrInsertFunction(
      "__nasan_record_pointer_store",
      VoidTy, PtrTy, Int64Ty);  // addr, provenance_id

  HandleExceptionCleanupFn = M.getOrInsertFunction(
      "__nasan_handle_exception_cleanup",
      VoidTy, Int64Ty);

  GetPointerProvenanceFn = M.getOrInsertFunction(
      "__nasan_get_pointer_provenance",
      Int64Ty, PtrTy);
}

bool NoAliasSanitizerImpl::shouldInstrumentFunction(Function &F) {
  // Skip declarations, intrinsics, and NASan runtime functions
  if (F.isDeclaration() || F.isIntrinsic())
    return false;

  StringRef Name = F.getName();
  if (Name.starts_with("__nasan_"))
    return false;

  // Skip system library functions
  if (Name.starts_with("_Z") && Name.contains("std::"))
    return false;

  return true;
}

bool NoAliasSanitizerImpl::run() {
  bool Modified = false;

  for (Function &F : M) {
    if (shouldInstrumentFunction(F)) {
      Modified |= instrumentFunction(F);
    }
  }

  return Modified;
}

bool NoAliasSanitizerImpl::instrumentFunction(Function &F) {
  bool Modified = false;
  BasicBlock &EntryBB = F.getEntryBlock();

  // Clear per-function state
  PointerProvenance.clear();

  // Track noalias parameters and their provenance IDs
  SmallVector<Argument*, 4> NoAliasParams;
  SmallVector<Value*, 4> ProvenanceIDs;

  for (Argument &Arg : F.args()) {
    if (Arg.hasNoAliasAttr() && Arg.getType()->isPointerTy()) {
      NoAliasParams.push_back(&Arg);
    }
  }

  // If no noalias parameters, skip instrumentation
  if (NoAliasParams.empty())
    return Modified;

  // Insert all create_provenance calls at the very beginning of the function
  IRBuilder<> EntryIRB(&*EntryBB.getFirstInsertionPt());
  if (auto *FirstInst = &*EntryBB.getFirstInsertionPt()) {
    if (FirstInst->getDebugLoc())
      EntryIRB.SetCurrentDebugLocation(FirstInst->getDebugLoc());
  }

  for (Argument *Arg : NoAliasParams) {
    DISubprogram *SP = F.getSubprogram();
    Value *FuncName = EntryIRB.CreateGlobalString(F.getName());
    Value *FileName = EntryIRB.CreateGlobalString(
        SP ? SP->getFilename() : StringRef("unknown"));
    Value *LineNum = ConstantInt::get(Type::getInt32Ty(C),
        SP ? SP->getLine() : 0);

    Value *ProvID = EntryIRB.CreateCall(
        CreateProvenanceFn,
        {Arg, FuncName, FileName, LineNum});
    ProvenanceIDs.push_back(ProvID);

    // Track the provenance for this pointer at compile time
    PointerProvenance[Arg] = ProvID;

    // Also store in shadow for runtime lookups (e.g., for pointers loaded from memory)
    EntryIRB.CreateCall(SetPointerProvenanceFn, {Arg, ProvID});

    ++NumNoAliasParams;
    Modified = true;
  }

  // First pass: find max PHI/select merge size needed
  unsigned MaxMergeSize = 2;  // At least 2 for select
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *PHI = dyn_cast<PHINode>(&I)) {
        if (PHI->getType()->isPointerTy()) {
          MaxMergeSize = std::max(MaxMergeSize, PHI->getNumIncomingValues());
        }
      }
    }
  }

  // Create entry block alloca for merge operations (reused across all PHI/select)
  // Insert right after create_provenance calls
  Value *MergeArray = EntryIRB.CreateAlloca(
      PointerType::getUnqual(C),
      ConstantInt::get(Type::getInt32Ty(C), MaxMergeSize),
      "nasan.merge.array");

  // Insert destroy_provenance at all return/unwind points
  for (BasicBlock &BB : F) {
    Instruction *Term = BB.getTerminator();

    if (auto *RI = dyn_cast<ReturnInst>(Term)) {
      IRBuilder<> RetIRB(RI);
      // Destroy in reverse order (LIFO for nested scopes)
      for (auto it = ProvenanceIDs.rbegin(); it != ProvenanceIDs.rend(); ++it) {
        RetIRB.CreateCall(DestroyProvenanceFn, {*it});
      }
    }
  }

  // Handle exception cleanup (landing pads)
  for (BasicBlock &BB : F) {
    if (BB.isLandingPad()) {
      BasicBlock::iterator InsertPt = BB.getFirstNonPHIIt();
      IRBuilder<> LPIRB(&*InsertPt);
      for (auto it = ProvenanceIDs.rbegin(); it != ProvenanceIDs.rend(); ++it) {
        LPIRB.CreateCall(HandleExceptionCleanupFn, {*it});
      }
    }
  }

  // Get alias analysis results for this function
  AAResults &AA = GetAA(F);

  // Instrument all instructions in the function
  for (BasicBlock &BB : F) {
    for (Instruction &I : make_early_inc_range(BB)) {
      // Pass NoAliasParams to check methods
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        // Check if we need to instrument this load
        if (!shouldInstrumentAccess(LI->getPointerOperand(), NoAliasParams, AA)) {
          ++NumSkippedReads;
          // Still need provenance propagation for pointer loads
          if (LI->getType()->isPointerTy()) {
            IRBuilder<> PostIRB(LI->getNextNode());
            if (LI->getDebugLoc())
              PostIRB.SetCurrentDebugLocation(LI->getDebugLoc());
            PostIRB.CreateCall(PropagateLoadProvenanceFn, {LI, LI->getPointerOperand()});
          }
        } else {
          instrumentLoad(*LI);
        }
        Modified = true;
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        // Check if we need to instrument this store
        if (!shouldInstrumentAccess(SI->getPointerOperand(), NoAliasParams, AA)) {
          ++NumSkippedWrites;
          // Still need provenance tracking for pointer stores
          if (SI->getValueOperand()->getType()->isPointerTy()) {
            IRBuilder<> IRB(SI);
            if (SI->getDebugLoc())
              IRB.SetCurrentDebugLocation(SI->getDebugLoc());
            Value *ValProv = getProvenance(SI->getValueOperand(), IRB);
            IRB.CreateCall(RecordPointerStoreFn, {SI->getPointerOperand(), ValProv});
          }
        } else {
          instrumentStore(*SI);
        }
        Modified = true;
      } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
        instrumentPHI(*PHI, MergeArray, EntryIRB);
        Modified = true;
      } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
        instrumentSelect(*Sel, MergeArray, EntryIRB);
        Modified = true;
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        instrumentGEP(*GEP);
        Modified = true;
      } else if (auto *Cast = dyn_cast<CastInst>(&I)) {
        instrumentCast(*Cast);
        Modified = true;
      } else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
        instrumentMemIntrinsic(*MI);
        Modified = true;
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        instrumentCall(*CB);
        // Don't set Modified for empty call instrumentation
      }
    }
  }

  return Modified;
}

void NoAliasSanitizerImpl::instrumentGEP(GetElementPtrInst &GEP) {
  // GEP inherits provenance from base pointer
  Value *BasePtr = GEP.getPointerOperand();

  // Update compile-time provenance map if we know the base's provenance
  auto It = PointerProvenance.find(BasePtr);
  if (It != PointerProvenance.end()) {
    PointerProvenance[&GEP] = It->second;
  }

  insertInheritProvenance(&GEP, BasePtr, GEP);
}

void NoAliasSanitizerImpl::instrumentCast(CastInst &Cast) {
  if (!Cast.getType()->isPointerTy())
    return;

  if (Cast.getOperand(0)->getType()->isPointerTy()) {
    // Pointer-to-pointer cast: inherit provenance
    Value *SrcPtr = Cast.getOperand(0);

    // Update compile-time provenance map if we know the source's provenance
    auto It = PointerProvenance.find(SrcPtr);
    if (It != PointerProvenance.end()) {
      PointerProvenance[&Cast] = It->second;
    }

    insertInheritProvenance(&Cast, SrcPtr, Cast);
  } else if (isa<IntToPtrInst>(Cast)) {
    // Integer-to-pointer: lose provenance (set to 0)
    IRBuilder<> IRB(Cast.getNextNode());
    if (Cast.getDebugLoc())
      IRB.SetCurrentDebugLocation(Cast.getDebugLoc());
    IRB.CreateCall(SetPointerProvenanceFn,
                   {&Cast, ConstantInt::get(Type::getInt64Ty(C), 0)});
  }
}

void NoAliasSanitizerImpl::instrumentPHI(PHINode &PHI, Value *PtrArray,
                                          IRBuilder<> &EntryIRB) {
  if (!PHI.getType()->isPointerTy())
    return;

  // Insert after all PHI nodes in the block
  IRBuilder<> IRB(&*PHI.getParent()->getFirstNonPHIIt());
  if (PHI.getDebugLoc())
    IRB.SetCurrentDebugLocation(PHI.getDebugLoc());

  // Collect all incoming pointer values
  SmallVector<Value*, 8> IncomingPtrs;
  for (unsigned i = 0; i < PHI.getNumIncomingValues(); ++i) {
    IncomingPtrs.push_back(PHI.getIncomingValue(i));
  }

  // Store incoming pointers to the pre-allocated array
  for (unsigned i = 0; i < IncomingPtrs.size(); ++i) {
    Value *ElemPtr = IRB.CreateConstGEP1_32(
        PointerType::getUnqual(C), PtrArray, i);
    IRB.CreateStore(IncomingPtrs[i], ElemPtr);
  }

  // Call __nasan_merge_provenance
  IRB.CreateCall(MergeProvenanceFn, {
    &PHI,
    PtrArray,
    ConstantInt::get(Type::getInt64Ty(C), IncomingPtrs.size())
  });
}

void NoAliasSanitizerImpl::instrumentSelect(SelectInst &Sel, Value *PtrArray,
                                             IRBuilder<> &EntryIRB) {
  if (!Sel.getType()->isPointerTy())
    return;

  // Insert after the select
  IRBuilder<> IRB(Sel.getNextNode());
  if (Sel.getDebugLoc())
    IRB.SetCurrentDebugLocation(Sel.getDebugLoc());

  Value *TrueVal = Sel.getTrueValue();
  Value *FalseVal = Sel.getFalseValue();

  // Store to pre-allocated array
  Value *Elem0 = IRB.CreateConstGEP1_32(PointerType::getUnqual(C), PtrArray, 0);
  Value *Elem1 = IRB.CreateConstGEP1_32(PointerType::getUnqual(C), PtrArray, 1);
  IRB.CreateStore(TrueVal, Elem0);
  IRB.CreateStore(FalseVal, Elem1);

  IRB.CreateCall(MergeProvenanceFn, {
    &Sel,
    PtrArray,
    ConstantInt::get(Type::getInt64Ty(C), 2)
  });
}

void NoAliasSanitizerImpl::instrumentLoad(LoadInst &LI) {
  IRBuilder<> PreIRB(&LI);
  if (LI.getDebugLoc())
    PreIRB.SetCurrentDebugLocation(LI.getDebugLoc());

  Value *Addr = LI.getPointerOperand();
  Value *AddrInt = PreIRB.CreatePtrToInt(Addr, Type::getInt64Ty(C));
  Value *Size = ConstantInt::get(Type::getInt64Ty(C),
      DL.getTypeStoreSize(LI.getType()));

  // Get provenance from compile-time tracking
  Value *Prov = getProvenance(Addr, PreIRB);

  // Check the load access with provenance ID
  PreIRB.CreateCall(CheckLoadFn, {AddrInt, Size, Prov});
  ++NumInstrumentedReads;

  // If loading a pointer, propagate provenance from stored value
  if (LI.getType()->isPointerTy()) {
    IRBuilder<> PostIRB(LI.getNextNode());
    if (LI.getDebugLoc())
      PostIRB.SetCurrentDebugLocation(LI.getDebugLoc());
    PostIRB.CreateCall(PropagateLoadProvenanceFn, {&LI, Addr});
  }
}

void NoAliasSanitizerImpl::instrumentStore(StoreInst &SI) {
  IRBuilder<> IRB(&SI);
  if (SI.getDebugLoc())
    IRB.SetCurrentDebugLocation(SI.getDebugLoc());

  Value *Addr = SI.getPointerOperand();
  Value *Val = SI.getValueOperand();
  Value *AddrInt = IRB.CreatePtrToInt(Addr, Type::getInt64Ty(C));
  Value *Size = ConstantInt::get(Type::getInt64Ty(C),
      DL.getTypeStoreSize(Val->getType()));

  // Get provenance from compile-time tracking
  Value *Prov = getProvenance(Addr, IRB);

  // Check the store access with provenance ID
  IRB.CreateCall(CheckStoreFn, {AddrInt, Size, Prov});
  ++NumInstrumentedWrites;

  // If storing a pointer, record the provenance for later loads
  if (Val->getType()->isPointerTy()) {
    Value *ValProv = getProvenance(Val, IRB);
    IRB.CreateCall(RecordPointerStoreFn, {Addr, ValProv});
  }
}

void NoAliasSanitizerImpl::instrumentCall(CallBase &CB) {
  // Most calls are handled automatically via shadow memory
  // Just ensure we're not calling nasan functions
}

void NoAliasSanitizerImpl::instrumentMemIntrinsic(MemIntrinsic &MI) {
  IRBuilder<> IRB(&MI);
  if (MI.getDebugLoc())
    IRB.SetCurrentDebugLocation(MI.getDebugLoc());

  Value *Dest = MI.getDest();
  Value *Length = MI.getLength();

  // Extend length to i64 if needed
  if (Length->getType() != Type::getInt64Ty(C)) {
    Length = IRB.CreateZExt(Length, Type::getInt64Ty(C));
  }

  if (auto *MTI = dyn_cast<MemTransferInst>(&MI)) {
    // memcpy or memmove
    Value *Src = MTI->getSource();

    Value *DestInt = IRB.CreatePtrToInt(Dest, Type::getInt64Ty(C));
    Value *SrcInt = IRB.CreatePtrToInt(Src, Type::getInt64Ty(C));

    // Get provenance for both pointers
    Value *DestProv = getProvenance(Dest, IRB);
    Value *SrcProv = getProvenance(Src, IRB);

    IRB.CreateCall(CheckStoreFn, {DestInt, Length, DestProv});
    IRB.CreateCall(CheckLoadFn, {SrcInt, Length, SrcProv});
  } else if (isa<MemSetInst>(&MI)) {
    // memset
    Value *DestInt = IRB.CreatePtrToInt(Dest, Type::getInt64Ty(C));
    Value *DestProv = getProvenance(Dest, IRB);
    IRB.CreateCall(CheckStoreFn, {DestInt, Length, DestProv});
  }
}

void NoAliasSanitizerImpl::insertInheritProvenance(Value *Dst, Value *Src, Instruction &I) {
  // Find the next insertion point
  Instruction *InsertPt = I.getNextNode();
  if (!InsertPt)
    return;

  IRBuilder<> IRB(InsertPt);
  if (I.getDebugLoc())
    IRB.SetCurrentDebugLocation(I.getDebugLoc());
  IRB.CreateCall(InheritProvenanceFn, {Dst, Src});
}

Value *NoAliasSanitizerImpl::getProvenance(Value *Ptr, IRBuilder<> &IRB) {
  // First, try to find provenance in compile-time map by walking through
  // pointer casts and GEPs to find the base
  Value *Base = Ptr;
  while (true) {
    // Check if we have provenance for this value
    auto It = PointerProvenance.find(Base);
    if (It != PointerProvenance.end()) {
      return It->second;
    }

    // Try to walk through pointer operations
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Base)) {
      Base = GEP->getPointerOperand();
    } else if (auto *Cast = dyn_cast<CastInst>(Base)) {
      if (Cast->getOperand(0)->getType()->isPointerTy()) {
        Base = Cast->getOperand(0);
      } else {
        break;
      }
    } else {
      break;
    }
  }

  // Not found in compile-time map, fall back to runtime lookup
  return IRB.CreateCall(GetPointerProvenanceFn, {Ptr});
}

// Helper to check if a value derived from an alloca escapes
// Uses a visited set to handle cycles (e.g., from PHI nodes)
static bool isNonEscapingValueImpl(Value *V, SmallPtrSetImpl<Value*> &Visited) {
  // Avoid infinite recursion on cycles
  if (!Visited.insert(V).second)
    return true;  // Already visited, assume safe to avoid false negatives

  for (User *U : V->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      // Storing TO the value is fine, storing the value's address is not
      if (SI->getValueOperand() == V)
        return false;
    } else if (isa<CallInst>(U) || isa<InvokeInst>(U)) {
      // Passing value address to a function - it escapes
      return false;
    } else if (isa<ReturnInst>(U)) {
      // Returning value address - it escapes
      return false;
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      // Check if GEP escapes
      if (!isNonEscapingValueImpl(GEP, Visited))
        return false;
    } else if (auto *BC = dyn_cast<BitCastInst>(U)) {
      // Check if bitcast escapes
      if (!isNonEscapingValueImpl(BC, Visited))
        return false;
    } else if (auto *PHI = dyn_cast<PHINode>(U)) {
      // Check if PHI escapes (but avoid cycles)
      if (!isNonEscapingValueImpl(PHI, Visited))
        return false;
    }
    // Load/store through the value are fine
  }

  return true;
}

static bool isNonEscapingValue(Value *V) {
  SmallPtrSet<Value*, 16> Visited;
  return isNonEscapingValueImpl(V, Visited);
}

bool NoAliasSanitizerImpl::isNonEscapingAlloca(Value *V) {
  // Strip to base pointer
  Value *Base = V->stripPointerCasts();

  // Must be an alloca at the base
  auto *AI = dyn_cast<AllocaInst>(Base);
  if (!AI)
    return false;

  // Check if the alloca (and all its derived pointers) escape
  return isNonEscapingValue(AI);
}

bool NoAliasSanitizerImpl::shouldInstrumentAccess(
    Value *Ptr, const SmallVectorImpl<Argument *> &NoAliasArgs,
    AAResults &AA) {

  // Always instrument if we can't analyze the pointer
  if (!Ptr)
    return true;

  // Check if this is a non-escaping alloca
  if (isNonEscapingAlloca(Ptr)) {
    // This alloca doesn't escape, so it can't alias with parameters
    LLVM_DEBUG(dbgs() << "NASan: Skipping non-escaping alloca: " << *Ptr << "\n");
    return false;
  }

  // Use alias analysis to check if this pointer could alias with any noalias parameter
  bool CouldAlias = false;
  for (Argument *Arg : NoAliasArgs) {
    AliasResult AR = AA.alias(Ptr, Arg);
    if (AR != AliasResult::NoAlias) {
      // Could alias with this noalias parameter
      CouldAlias = true;
      break;
    }
  }

  if (!CouldAlias) {
    LLVM_DEBUG(dbgs() << "NASan: AA proves no alias for: " << *Ptr << "\n");
    return false;
  }

  // Need to instrument this access
  return true;
}

} // anonymous namespace

PreservedAnalyses NoAliasSanitizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto GetAA = [&](Function &F) -> AAResults & {
    return FAM.getResult<AAManager>(F);
  };

  NoAliasSanitizerImpl Impl(M, GetAA);

  if (Impl.run())
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}
