//===-- ShadowStackGC.cpp - GC support for uncooperative targets ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering for the llvm.gc* intrinsics for targets that do
// not natively support them (which includes the C backend). Note that the code
// generated is not quite as efficient as algorithms which generate stack maps
// to identify roots.
//
// This pass implements the code transformation described in this paper:
//   "Accurate Garbage Collection in an Uncooperative Environment"
//   Fergus Henderson, ISMM, 2002
//
// In runtime/GC/SemiSpace.cpp is a prototype runtime which is compatible with
// ShadowStackGC.
//
// In order to support this particular transformation, all stack roots are
// coallocated in the stack. This allows a fully target-independent stack map
// while introducing only minor runtime overhead.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "shadowstackgc"
#include "llvm/CodeGen/GCs.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/IRBuilder.h"

using namespace llvm;

namespace {

  class VISIBILITY_HIDDEN ShadowStackGC : public GCStrategy {
    /// RootChain - This is the global linked-list that contains the chain of GC
    /// roots.
    GlobalVariable *Head;

    /// StackEntryTy - Abstract type of a link in the shadow stack.
    ///
    const StructType *StackEntryTy;

    /// Roots - GC roots in the current function. Each is a pair of the
    /// intrinsic call and its corresponding alloca.
    std::vector<std::pair<CallInst*,AllocaInst*> > Roots;

  public:
    ShadowStackGC();

    bool initializeCustomLowering(Module &M);
    bool performCustomLowering(Function &F);

  private:
    bool IsNullValue(Value *V);
    Constant *GetFrameMap(Function &F);
    const Type* GetConcreteStackEntryType(Function &F);
    void CollectRoots(Function &F);
    static GetElementPtrInst *CreateGEP(IRBuilder<> &B, Value *BasePtr,
                                        int Idx1, const char *Name);
    static GetElementPtrInst *CreateGEP(IRBuilder<> &B, Value *BasePtr,
                                        int Idx1, int Idx2, const char *Name);
  };

}

static GCRegistry::Add<ShadowStackGC>
X("shadow-stack", "Very portable GC for uncooperative code generators");

namespace {
  /// EscapeEnumerator - This is a little algorithm to find all escape points
  /// from a function so that "finally"-style code can be inserted. In addition
  /// to finding the existing return and unwind instructions, it also (if
  /// necessary) transforms any call instructions into invokes and sends them to
  /// a landing pad.
  ///
  /// It's wrapped up in a state machine using the same transform C# uses for
  /// 'yield return' enumerators, This transform allows it to be non-allocating.
  class VISIBILITY_HIDDEN EscapeEnumerator {
    Function &F;
    const char *CleanupBBName;

    // State.
    int State;
    Function::iterator StateBB, StateE;
    IRBuilder<> Builder;

  public:
    EscapeEnumerator(Function &F, const char *N = "cleanup")
      : F(F), CleanupBBName(N), State(0) {}

    IRBuilder<> *Next() {
      switch (State) {
      default:
        return 0;

      case 0:
        StateBB = F.begin();
        StateE = F.end();
        State = 1;

      case 1:
        // Find all 'return' and 'unwind' instructions.
        while (StateBB != StateE) {
          BasicBlock *CurBB = StateBB++;

          // Branches and invokes do not escape, only unwind and return do.
          TerminatorInst *TI = CurBB->getTerminator();
          if (!isa<UnwindInst>(TI) && !isa<ReturnInst>(TI))
            continue;

          Builder.SetInsertPoint(TI->getParent(), TI);
          return &Builder;
        }

        State = 2;

        // Find all 'call' instructions.
        SmallVector<Instruction*,16> Calls;
        for (Function::iterator BB = F.begin(),
                                E = F.end(); BB != E; ++BB)
          for (BasicBlock::iterator II = BB->begin(),
                                    EE = BB->end(); II != EE; ++II)
            if (CallInst *CI = dyn_cast<CallInst>(II))
              if (!CI->getCalledFunction() ||
                  !CI->getCalledFunction()->getIntrinsicID())
                Calls.push_back(CI);

        if (Calls.empty())
          return 0;

        // Create a cleanup block.
        BasicBlock *CleanupBB = BasicBlock::Create(CleanupBBName, &F);
        UnwindInst *UI = new UnwindInst(CleanupBB);

        // Transform the 'call' instructions into 'invoke's branching to the
        // cleanup block. Go in reverse order to make prettier BB names.
        SmallVector<Value*,16> Args;
        for (unsigned I = Calls.size(); I != 0; ) {
          CallInst *CI = cast<CallInst>(Calls[--I]);

          // Split the basic block containing the function call.
          BasicBlock *CallBB = CI->getParent();
          BasicBlock *NewBB =
            CallBB->splitBasicBlock(CI, CallBB->getName() + ".cont");

          // Remove the unconditional branch inserted at the end of CallBB.
          CallBB->getInstList().pop_back();
          NewBB->getInstList().remove(CI);

          // Create a new invoke instruction.
          Args.clear();
          Args.append(CI->op_begin() + 1, CI->op_end());

          InvokeInst *II = InvokeInst::Create(CI->getOperand(0),
                                              NewBB, CleanupBB,
                                              Args.begin(), Args.end(),
                                              CI->getName(), CallBB);
          II->setCallingConv(CI->getCallingConv());
          II->setAttributes(CI->getAttributes());
          CI->replaceAllUsesWith(II);
          delete CI;
        }

        Builder.SetInsertPoint(UI->getParent(), UI);
        return &Builder;
      }
    }
  };
}

// -----------------------------------------------------------------------------

void llvm::linkShadowStackGC() { }

ShadowStackGC::ShadowStackGC() : Head(0), StackEntryTy(0) {
  InitRoots = true;
  CustomRoots = true;
}

Constant *ShadowStackGC::GetFrameMap(Function &F) {
  // doInitialization creates the abstract type of this value.

  Type *VoidPtr = PointerType::getUnqual(Type::Int8Ty);

  // Truncate the ShadowStackDescriptor if some metadata is null.
  unsigned NumMeta = 0;
  SmallVector<Constant*,16> Metadata;
  for (unsigned I = 0; I != Roots.size(); ++I) {
    Constant *C = cast<Constant>(Roots[I].first->getOperand(2));
    if (!C->isNullValue())
      NumMeta = I + 1;
    Metadata.push_back(ConstantExpr::getBitCast(C, VoidPtr));
  }

  Constant *BaseElts[] = {
    ConstantInt::get(Type::Int32Ty, Roots.size(), false),
    ConstantInt::get(Type::Int32Ty, NumMeta, false),
  };

  Constant *DescriptorElts[] = {
    ConstantStruct::get(BaseElts, 2),
    ConstantArray::get(ArrayType::get(VoidPtr, NumMeta),
                       Metadata.begin(), NumMeta)
  };

  Constant *FrameMap = ConstantStruct::get(DescriptorElts, 2);

  std::string TypeName("gc_map.");
  TypeName += utostr(NumMeta);
  F.getParent()->addTypeName(TypeName, FrameMap->getType());

  // FIXME: Is this actually dangerous as WritingAnLLVMPass.html claims? Seems
  //        that, short of multithreaded LLVM, it should be safe; all that is
  //        necessary is that a simple Module::iterator loop not be invalidated.
  //        Appending to the GlobalVariable list is safe in that sense.
  //
  //        All of the output passes emit globals last. The ExecutionEngine
  //        explicitly supports adding globals to the module after
  //        initialization.
  //
  //        Still, if it isn't deemed acceptable, then this transformation needs
  //        to be a ModulePass (which means it cannot be in the 'llc' pipeline
  //        (which uses a FunctionPassManager (which segfaults (not asserts) if
  //        provided a ModulePass))).
  Constant *GV = new GlobalVariable(FrameMap->getType(), true,
                                    GlobalVariable::InternalLinkage,
                                    FrameMap, "__gc_" + F.getName(),
                                    F.getParent());

  Constant *GEPIndices[2] = { ConstantInt::get(Type::Int32Ty, 0),
                              ConstantInt::get(Type::Int32Ty, 0) };
  return ConstantExpr::getGetElementPtr(GV, GEPIndices, 2);
}

const Type* ShadowStackGC::GetConcreteStackEntryType(Function &F) {
  // doInitialization creates the generic version of this type.
  std::vector<const Type*> EltTys;
  EltTys.push_back(StackEntryTy);
  for (size_t I = 0; I != Roots.size(); I++)
    EltTys.push_back(Roots[I].second->getAllocatedType());
  Type *Ty = StructType::get(EltTys);

  std::string TypeName("gc_stackentry.");
  TypeName += F.getName();
  F.getParent()->addTypeName(TypeName, Ty);

  return Ty;
}

/// doInitialization - If this module uses the GC intrinsics, find them now. If
/// not, exit fast.
bool ShadowStackGC::initializeCustomLowering(Module &M) {
  // struct FrameMap {
  //   int32_t NumRoots; // Number of roots in stack frame.
  //   int32_t NumMeta;  // Number of metadata descriptors. May be < NumRoots.
  //   void *Meta[];     // May be absent for roots without metadata.
  // };
  std::vector<const Type*> EltTys;
  EltTys.push_back(Type::Int32Ty); // 32 bits is ok up to a 32GB stack frame. :)
  EltTys.push_back(Type::Int32Ty); // Specifies length of variable length array.
  StructType *FrameMapTy = StructType::get(EltTys);
  M.addTypeName("gc_map", FrameMapTy);
  PointerType *FrameMapPtrTy = PointerType::getUnqual(FrameMapTy);

  // struct StackEntry {
  //   ShadowStackEntry *Next; // Caller's stack entry.
  //   FrameMap *Map;          // Pointer to constant FrameMap.
  //   void *Roots[];          // Stack roots (in-place array, so we pretend).
  // };
  OpaqueType *RecursiveTy = OpaqueType::get();

  EltTys.clear();
  EltTys.push_back(PointerType::getUnqual(RecursiveTy));
  EltTys.push_back(FrameMapPtrTy);
  PATypeHolder LinkTyH = StructType::get(EltTys);

  RecursiveTy->refineAbstractTypeTo(LinkTyH.get());
  StackEntryTy = cast<StructType>(LinkTyH.get());
  const PointerType *StackEntryPtrTy = PointerType::getUnqual(StackEntryTy);
  M.addTypeName("gc_stackentry", LinkTyH.get());  // FIXME: Is this safe from
                                                  //        a FunctionPass?

  // Get the root chain if it already exists.
  Head = M.getGlobalVariable("llvm_gc_root_chain");
  if (!Head) {
    // If the root chain does not exist, insert a new one with linkonce
    // linkage!
    Head = new GlobalVariable(StackEntryPtrTy, false,
                              GlobalValue::LinkOnceLinkage,
                              Constant::getNullValue(StackEntryPtrTy),
                              "llvm_gc_root_chain", &M);
  } else if (Head->hasExternalLinkage() && Head->isDeclaration()) {
    Head->setInitializer(Constant::getNullValue(StackEntryPtrTy));
    Head->setLinkage(GlobalValue::LinkOnceLinkage);
  }

  return true;
}

bool ShadowStackGC::IsNullValue(Value *V) {
  if (Constant *C = dyn_cast<Constant>(V))
    return C->isNullValue();
  return false;
}

void ShadowStackGC::CollectRoots(Function &F) {
  // FIXME: Account for original alignment. Could fragment the root array.
  //   Approach 1: Null initialize empty slots at runtime. Yuck.
  //   Approach 2: Emit a map of the array instead of just a count.

  assert(Roots.empty() && "Not cleaned up?");

  SmallVector<std::pair<CallInst*,AllocaInst*>,16> MetaRoots;

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;)
      if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++))
        if (Function *F = CI->getCalledFunction())
          if (F->getIntrinsicID() == Intrinsic::gcroot) {
            std::pair<CallInst*,AllocaInst*> Pair = std::make_pair(
              CI, cast<AllocaInst>(CI->getOperand(1)->stripPointerCasts()));
            if (IsNullValue(CI->getOperand(2)))
              Roots.push_back(Pair);
            else
              MetaRoots.push_back(Pair);
          }

  // Number roots with metadata (usually empty) at the beginning, so that the
  // FrameMap::Meta array can be elided.
  Roots.insert(Roots.begin(), MetaRoots.begin(), MetaRoots.end());
}

GetElementPtrInst *
ShadowStackGC::CreateGEP(IRBuilder<> &B, Value *BasePtr,
                         int Idx, int Idx2, const char *Name) {
  Value *Indices[] = { ConstantInt::get(Type::Int32Ty, 0),
                       ConstantInt::get(Type::Int32Ty, Idx),
                       ConstantInt::get(Type::Int32Ty, Idx2) };
  Value* Val = B.CreateGEP(BasePtr, Indices, Indices + 3, Name);

  assert(isa<GetElementPtrInst>(Val) && "Unexpected folded constant");

  return dyn_cast<GetElementPtrInst>(Val);
}

GetElementPtrInst *
ShadowStackGC::CreateGEP(IRBuilder<> &B, Value *BasePtr,
                         int Idx, const char *Name) {
  Value *Indices[] = { ConstantInt::get(Type::Int32Ty, 0),
                       ConstantInt::get(Type::Int32Ty, Idx) };
  Value *Val = B.CreateGEP(BasePtr, Indices, Indices + 2, Name);

  assert(isa<GetElementPtrInst>(Val) && "Unexpected folded constant");

  return dyn_cast<GetElementPtrInst>(Val);
}

/// runOnFunction - Insert code to maintain the shadow stack.
bool ShadowStackGC::performCustomLowering(Function &F) {
  // Find calls to llvm.gcroot.
  CollectRoots(F);

  // If there are no roots in this function, then there is no need to add a
  // stack map entry for it.
  if (Roots.empty())
    return false;

  // Build the constant map and figure the type of the shadow stack entry.
  Value *FrameMap = GetFrameMap(F);
  const Type *ConcreteStackEntryTy = GetConcreteStackEntryType(F);

  // Build the shadow stack entry at the very start of the function.
  BasicBlock::iterator IP = F.getEntryBlock().begin();
  IRBuilder<> AtEntry(IP->getParent(), IP);

  Instruction *StackEntry   = AtEntry.CreateAlloca(ConcreteStackEntryTy, 0,
                                                   "gc_frame");

  while (isa<AllocaInst>(IP)) ++IP;
  AtEntry.SetInsertPoint(IP->getParent(), IP);

  // Initialize the map pointer and load the current head of the shadow stack.
  Instruction *CurrentHead  = AtEntry.CreateLoad(Head, "gc_currhead");
  Instruction *EntryMapPtr  = CreateGEP(AtEntry, StackEntry,0,1,"gc_frame.map");
                              AtEntry.CreateStore(FrameMap, EntryMapPtr);

  // After all the allocas...
  for (unsigned I = 0, E = Roots.size(); I != E; ++I) {
    // For each root, find the corresponding slot in the aggregate...
    Value *SlotPtr = CreateGEP(AtEntry, StackEntry, 1 + I, "gc_root");

    // And use it in lieu of the alloca.
    AllocaInst *OriginalAlloca = Roots[I].second;
    SlotPtr->takeName(OriginalAlloca);
    OriginalAlloca->replaceAllUsesWith(SlotPtr);
  }

  // Move past the original stores inserted by GCStrategy::InitRoots. This isn't
  // really necessary (the collector would never see the intermediate state at
  // runtime), but it's nicer not to push the half-initialized entry onto the
  // shadow stack.
  while (isa<StoreInst>(IP)) ++IP;
  AtEntry.SetInsertPoint(IP->getParent(), IP);

  // Push the entry onto the shadow stack.
  Instruction *EntryNextPtr = CreateGEP(AtEntry,StackEntry,0,0,"gc_frame.next");
  Instruction *NewHeadVal   = CreateGEP(AtEntry,StackEntry, 0, "gc_newhead");
                              AtEntry.CreateStore(CurrentHead, EntryNextPtr);
                              AtEntry.CreateStore(NewHeadVal, Head);

  // For each instruction that escapes...
  EscapeEnumerator EE(F, "gc_cleanup");
  while (IRBuilder<> *AtExit = EE.Next()) {
    // Pop the entry from the shadow stack. Don't reuse CurrentHead from
    // AtEntry, since that would make the value live for the entire function.
    Instruction *EntryNextPtr2 = CreateGEP(*AtExit, StackEntry, 0, 0,
                                           "gc_frame.next");
    Value *SavedHead = AtExit->CreateLoad(EntryNextPtr2, "gc_savedhead");
                       AtExit->CreateStore(SavedHead, Head);
  }

  // Delete the original allocas (which are no longer used) and the intrinsic
  // calls (which are no longer valid). Doing this last avoids invalidating
  // iterators.
  for (unsigned I = 0, E = Roots.size(); I != E; ++I) {
    Roots[I].first->eraseFromParent();
    Roots[I].second->eraseFromParent();
  }

  Roots.clear();
  return true;
}
