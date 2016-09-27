//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass builds the coroutine frame and outlines resume and destroy parts
// of the coroutine into separate functions.
//
// We present a coroutine to an LLVM as an ordinary function with suspension
// points marked up with intrinsics. We let the optimizer party on the coroutine
// as a single function for as long as possible. Shortly before the coroutine is
// eligible to be inlined into its callers, we split up the coroutine into parts
// corresponding to an initial, resume and destroy invocations of the coroutine,
// add them to the current SCC and restart the IPO pipeline to optimize the
// coroutine subfunctions we extracted before proceeding to the caller of the
// coroutine.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "coro-split"

// Create an entry block for a resume function with a switch that will jump to
// suspend points.
static BasicBlock *createResumeEntryBlock(Function &F, coro::Shape &Shape) {
  LLVMContext &C = F.getContext();

  // resume.entry:
  //  %index.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0,
  //  i32 2
  //  % index = load i32, i32* %index.addr
  //  switch i32 %index, label %unreachable [
  //    i32 0, label %resume.0
  //    i32 1, label %resume.1
  //    ...
  //  ]

  auto *NewEntry = BasicBlock::Create(C, "resume.entry", &F);
  auto *UnreachBB = BasicBlock::Create(C, "unreachable", &F);

  IRBuilder<> Builder(NewEntry);
  auto *FramePtr = Shape.FramePtr;
  auto *FrameTy = Shape.FrameTy;
  auto *GepIndex = Builder.CreateConstInBoundsGEP2_32(
      FrameTy, FramePtr, 0, coro::Shape::IndexField, "index.addr");
  auto *Index = Builder.CreateLoad(GepIndex, "index");
  auto *Switch =
      Builder.CreateSwitch(Index, UnreachBB, Shape.CoroSuspends.size());
  Shape.ResumeSwitch = Switch;

  size_t SuspendIndex = 0;
  for (CoroSuspendInst *S : Shape.CoroSuspends) {
    ConstantInt *IndexVal = Shape.getIndex(SuspendIndex);

    // Replace CoroSave with a store to Index:
    //    %index.addr = getelementptr %f.frame... (index field number)
    //    store i32 0, i32* %index.addr1
    auto *Save = S->getCoroSave();
    Builder.SetInsertPoint(Save);
    if (S->isFinal()) {
      // Final suspend point is represented by storing zero in ResumeFnAddr.
      auto *GepIndex = Builder.CreateConstInBoundsGEP2_32(FrameTy, FramePtr, 0,
                                                          0, "ResumeFn.addr");
      auto *NullPtr = ConstantPointerNull::get(cast<PointerType>(
          cast<PointerType>(GepIndex->getType())->getElementType()));
      Builder.CreateStore(NullPtr, GepIndex);
    } else {
      auto *GepIndex = Builder.CreateConstInBoundsGEP2_32(
          FrameTy, FramePtr, 0, coro::Shape::IndexField, "index.addr");
      Builder.CreateStore(IndexVal, GepIndex);
    }
    Save->replaceAllUsesWith(ConstantTokenNone::get(C));
    Save->eraseFromParent();

    // Split block before and after coro.suspend and add a jump from an entry
    // switch:
    //
    //  whateverBB:
    //    whatever
    //    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    //    switch i8 %0, label %suspend[i8 0, label %resume
    //                                 i8 1, label %cleanup]
    // becomes:
    //
    //  whateverBB:
    //     whatever
    //     br label %resume.0.landing
    //
    //  resume.0: ; <--- jump from the switch in the resume.entry
    //     %0 = tail call i8 @llvm.coro.suspend(token none, i1 false)
    //     br label %resume.0.landing
    //
    //  resume.0.landing:
    //     %1 = phi i8[-1, %whateverBB], [%0, %resume.0]
    //     switch i8 % 1, label %suspend [i8 0, label %resume
    //                                    i8 1, label %cleanup]

    auto *SuspendBB = S->getParent();
    auto *ResumeBB =
        SuspendBB->splitBasicBlock(S, "resume." + Twine(SuspendIndex));
    auto *LandingBB = ResumeBB->splitBasicBlock(
        S->getNextNode(), ResumeBB->getName() + Twine(".landing"));
    Switch->addCase(IndexVal, ResumeBB);

    cast<BranchInst>(SuspendBB->getTerminator())->setSuccessor(0, LandingBB);
    auto *PN = PHINode::Create(Builder.getInt8Ty(), 2, "", &LandingBB->front());
    S->replaceAllUsesWith(PN);
    PN->addIncoming(Builder.getInt8(-1), SuspendBB);
    PN->addIncoming(S, ResumeBB);

    ++SuspendIndex;
  }

  Builder.SetInsertPoint(UnreachBB);
  Builder.CreateUnreachable();

  return NewEntry;
}

// In Resumers, we replace fallthrough coro.end with ret void and delete the
// rest of the block.
static void replaceFallthroughCoroEnd(IntrinsicInst *End,
                                      ValueToValueMapTy &VMap) {
  auto *NewE = cast<IntrinsicInst>(VMap[End]);
  ReturnInst::Create(NewE->getContext(), nullptr, NewE);

  // Remove the rest of the block, by splitting it into an unreachable block.
  auto *BB = NewE->getParent();
  BB->splitBasicBlock(NewE);
  BB->getTerminator()->eraseFromParent();
}

// Rewrite final suspend point handling. We do not use suspend index to
// represent the final suspend point. Instead we zero-out ResumeFnAddr in the
// coroutine frame, since it is undefined behavior to resume a coroutine
// suspended at the final suspend point. Thus, in the resume function, we can
// simply remove the last case (when coro::Shape is built, the final suspend
// point (if present) is always the last element of CoroSuspends array).
// In the destroy function, we add a code sequence to check if ResumeFnAddress
// is Null, and if so, jump to the appropriate label to handle cleanup from the
// final suspend point.
static void handleFinalSuspend(IRBuilder<> &Builder, Value *FramePtr,
                               coro::Shape &Shape, SwitchInst *Switch,
                               bool IsDestroy) {
  assert(Shape.HasFinalSuspend);
  auto FinalCase = --Switch->case_end();
  BasicBlock *ResumeBB = FinalCase.getCaseSuccessor();
  Switch->removeCase(FinalCase);
  if (IsDestroy) {
    BasicBlock *OldSwitchBB = Switch->getParent();
    auto *NewSwitchBB = OldSwitchBB->splitBasicBlock(Switch, "Switch");
    Builder.SetInsertPoint(OldSwitchBB->getTerminator());
    auto *GepIndex = Builder.CreateConstInBoundsGEP2_32(Shape.FrameTy, FramePtr,
                                                        0, 0, "ResumeFn.addr");
    auto *Load = Builder.CreateLoad(GepIndex);
    auto *NullPtr =
        ConstantPointerNull::get(cast<PointerType>(Load->getType()));
    auto *Cond = Builder.CreateICmpEQ(Load, NullPtr);
    Builder.CreateCondBr(Cond, ResumeBB, NewSwitchBB);
    OldSwitchBB->getTerminator()->eraseFromParent();
  }
}

// Create a resume clone by cloning the body of the original function, setting
// new entry block and replacing coro.suspend an appropriate value to force
// resume or cleanup pass for every suspend point.
static Function *createClone(Function &F, Twine Suffix, coro::Shape &Shape,
                             BasicBlock *ResumeEntry, int8_t FnIndex) {
  Module *M = F.getParent();
  auto *FrameTy = Shape.FrameTy;
  auto *FnPtrTy = cast<PointerType>(FrameTy->getElementType(0));
  auto *FnTy = cast<FunctionType>(FnPtrTy->getElementType());

  Function *NewF =
      Function::Create(FnTy, GlobalValue::LinkageTypes::InternalLinkage,
                       F.getName() + Suffix, M);
  NewF->addAttribute(1, Attribute::NonNull);
  NewF->addAttribute(1, Attribute::NoAlias);

  ValueToValueMapTy VMap;
  // Replace all args with undefs. The buildCoroutineFrame algorithm already
  // rewritten access to the args that occurs after suspend points with loads
  // and stores to/from the coroutine frame.
  for (Argument &A : F.getArgumentList())
    VMap[&A] = UndefValue::get(A.getType());

  SmallVector<ReturnInst *, 4> Returns;

  CloneFunctionInto(NewF, &F, VMap, /*ModuleLevelChanges=*/true, Returns);

  // If we have debug info, update it. ModuleLevelChanges = true above, does
  // the heavy lifting, we just need to repoint subprogram at the same
  // DICompileUnit as the original function F.
  if (DISubprogram *SP = F.getSubprogram())
    NewF->getSubprogram()->replaceUnit(SP->getUnit());

  // Remove old returns.
  for (ReturnInst *Return : Returns)
    changeToUnreachable(Return, /*UseLLVMTrap=*/false);

  // Remove old return attributes.
  NewF->removeAttributes(
      AttributeSet::ReturnIndex,
      AttributeSet::get(
          NewF->getContext(), AttributeSet::ReturnIndex,
          AttributeFuncs::typeIncompatible(NewF->getReturnType())));

  // Make AllocaSpillBlock the new entry block.
  auto *SwitchBB = cast<BasicBlock>(VMap[ResumeEntry]);
  auto *Entry = cast<BasicBlock>(VMap[Shape.AllocaSpillBlock]);
  Entry->moveBefore(&NewF->getEntryBlock());
  Entry->getTerminator()->eraseFromParent();
  BranchInst::Create(SwitchBB, Entry);
  Entry->setName("entry" + Suffix);

  // Clear all predecessors of the new entry block.
  auto *Switch = cast<SwitchInst>(VMap[Shape.ResumeSwitch]);
  Entry->replaceAllUsesWith(Switch->getDefaultDest());

  IRBuilder<> Builder(&NewF->getEntryBlock().front());

  // Remap frame pointer.
  Argument *NewFramePtr = &NewF->getArgumentList().front();
  Value *OldFramePtr = cast<Value>(VMap[Shape.FramePtr]);
  NewFramePtr->takeName(OldFramePtr);
  OldFramePtr->replaceAllUsesWith(NewFramePtr);

  // Remap vFrame pointer.
  auto *NewVFrame = Builder.CreateBitCast(
      NewFramePtr, Type::getInt8PtrTy(Builder.getContext()), "vFrame");
  Value *OldVFrame = cast<Value>(VMap[Shape.CoroBegin]);
  OldVFrame->replaceAllUsesWith(NewVFrame);

  // Rewrite final suspend handling as it is not done via switch (allows to
  // remove final case from the switch, since it is undefined behavior to resume
  // the coroutine suspended at the final suspend point.
  if (Shape.HasFinalSuspend) {
    auto *Switch = cast<SwitchInst>(VMap[Shape.ResumeSwitch]);
    bool IsDestroy = FnIndex != 0;
    handleFinalSuspend(Builder, NewFramePtr, Shape, Switch, IsDestroy);
  }

  // Replace coro suspend with the appropriate resume index.
  // Replacing coro.suspend with (0) will result in control flow proceeding to
  // a resume label associated with a suspend point, replacing it with (1) will
  // result in control flow proceeding to a cleanup label associated with this
  // suspend point.
  auto *NewValue = Builder.getInt8(FnIndex ? 1 : 0);
  for (CoroSuspendInst *CS : Shape.CoroSuspends) {
    auto *MappedCS = cast<CoroSuspendInst>(VMap[CS]);
    MappedCS->replaceAllUsesWith(NewValue);
    MappedCS->eraseFromParent();
  }

  // Remove coro.end intrinsics.
  replaceFallthroughCoroEnd(Shape.CoroEnds.front(), VMap);
  // FIXME: coming in upcoming patches:
  // replaceUnwindCoroEnds(Shape.CoroEnds, VMap);

  // We only store resume(0) and destroy(1) addresses in the coroutine frame.
  // The cleanup(2) clone is only used during devirtualization when coroutine is
  // eligible for heap elision and thus does not participate in indirect calls
  // and does not need its address to be stored in the coroutine frame.
  if (FnIndex < 2) {
    // Store the address of this clone in the coroutine frame.
    Builder.SetInsertPoint(Shape.FramePtr->getNextNode());
    auto *G = Builder.CreateConstInBoundsGEP2_32(Shape.FrameTy, Shape.FramePtr,
                                                 0, FnIndex, "fn.addr");
    Builder.CreateStore(NewF, G);
  }

  // Eliminate coro.free from the clones, replacing it with 'null' in cleanup,
  // to suppress deallocation code.
  coro::replaceCoroFree(cast<CoroIdInst>(VMap[Shape.CoroBegin->getId()]),
                        /*Elide=*/FnIndex == 2);

  NewF->setCallingConv(CallingConv::Fast);

  return NewF;
}

static void removeCoroEnds(coro::Shape &Shape) {
  for (CoroEndInst *CE : Shape.CoroEnds)
    CE->eraseFromParent();
}

static void replaceFrameSize(coro::Shape &Shape) {
  if (Shape.CoroSizes.empty())
    return;

  // In the same function all coro.sizes should have the same result type.
  auto *SizeIntrin = Shape.CoroSizes.back();
  Module *M = SizeIntrin->getModule();
  const DataLayout &DL = M->getDataLayout();
  auto Size = DL.getTypeAllocSize(Shape.FrameTy);
  auto *SizeConstant = ConstantInt::get(SizeIntrin->getType(), Size);

  for (CoroSizeInst *CS : Shape.CoroSizes) {
    CS->replaceAllUsesWith(SizeConstant);
    CS->eraseFromParent();
  }
}

// Create a global constant array containing pointers to functions provided and
// set Info parameter of CoroBegin to point at this constant. Example:
//
//   @f.resumers = internal constant [2 x void(%f.frame*)*]
//                    [void(%f.frame*)* @f.resume, void(%f.frame*)* @f.destroy]
//   define void @f() {
//     ...
//     call i8* @llvm.coro.begin(i8* null, i32 0, i8* null,
//                    i8* bitcast([2 x void(%f.frame*)*] * @f.resumers to i8*))
//
// Assumes that all the functions have the same signature.
static void setCoroInfo(Function &F, CoroBeginInst *CoroBegin,
                        std::initializer_list<Function *> Fns) {

  SmallVector<Constant *, 4> Args(Fns.begin(), Fns.end());
  assert(!Args.empty());
  Function *Part = *Fns.begin();
  Module *M = Part->getParent();
  auto *ArrTy = ArrayType::get(Part->getType(), Args.size());

  auto *ConstVal = ConstantArray::get(ArrTy, Args);
  auto *GV = new GlobalVariable(*M, ConstVal->getType(), /*isConstant=*/true,
                                GlobalVariable::PrivateLinkage, ConstVal,
                                F.getName() + Twine(".resumers"));

  // Update coro.begin instruction to refer to this constant.
  LLVMContext &C = F.getContext();
  auto *BC = ConstantExpr::getPointerCast(GV, Type::getInt8PtrTy(C));
  CoroBegin->getId()->setInfo(BC);
}

static void postSplitCleanup(Function &F) {
  removeUnreachableBlocks(F);
  llvm::legacy::FunctionPassManager FPM(F.getParent());

  FPM.add(createVerifierPass());
  FPM.add(createSCCPPass());
  FPM.add(createCFGSimplificationPass());
  FPM.add(createEarlyCSEPass());
  FPM.add(createCFGSimplificationPass());

  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();
}

// Coroutine has no suspend points. Remove heap allocation for the coroutine
// frame if possible.
static void handleNoSuspendCoroutine(CoroBeginInst *CoroBegin, Type *FrameTy) {
  auto *CoroId = CoroBegin->getId();
  auto *AllocInst = CoroId->getCoroAlloc();
  coro::replaceCoroFree(CoroId, /*Elide=*/AllocInst != nullptr);
  if (AllocInst) {
    IRBuilder<> Builder(AllocInst);
    // FIXME: Need to handle overaligned members.
    auto *Frame = Builder.CreateAlloca(FrameTy);
    auto *VFrame = Builder.CreateBitCast(Frame, Builder.getInt8PtrTy());
    AllocInst->replaceAllUsesWith(Builder.getFalse());
    AllocInst->eraseFromParent();
    CoroBegin->replaceAllUsesWith(VFrame);
  } else {
    CoroBegin->replaceAllUsesWith(CoroBegin->getMem());
  }
  CoroBegin->eraseFromParent();
}

// look for a very simple pattern
//    coro.save
//    no other calls
//    resume or destroy call
//    coro.suspend
//
// If there are other calls between coro.save and coro.suspend, they can
// potentially resume or destroy the coroutine, so it is unsafe to eliminate a
// suspend point.
static bool simplifySuspendPoint(CoroSuspendInst *Suspend,
                                 CoroBeginInst *CoroBegin) {
  auto *Save = Suspend->getCoroSave();
  auto *BB = Suspend->getParent();
  if (BB != Save->getParent())
    return false;

  CallSite SingleCallSite;

  // Check that we have only one CallSite.
  for (Instruction *I = Save->getNextNode(); I != Suspend;
       I = I->getNextNode()) {
    if (isa<CoroFrameInst>(I))
      continue;
    if (isa<CoroSubFnInst>(I))
      continue;
    if (CallSite CS = CallSite(I)) {
      if (SingleCallSite)
        return false;
      else
        SingleCallSite = CS;
    }
  }
  auto *CallInstr = SingleCallSite.getInstruction();
  if (!CallInstr)
    return false;

  auto *Callee = SingleCallSite.getCalledValue()->stripPointerCasts();

  // See if the callsite is for resumption or destruction of the coroutine.
  auto *SubFn = dyn_cast<CoroSubFnInst>(Callee);
  if (!SubFn)
    return false;

  // Does not refer to the current coroutine, we cannot do anything with it.
  if (SubFn->getFrame() != CoroBegin)
    return false;

  // Replace llvm.coro.suspend with the value that results in resumption over
  // the resume or cleanup path.
  Suspend->replaceAllUsesWith(SubFn->getRawIndex());
  Suspend->eraseFromParent();
  Save->eraseFromParent();

  // No longer need a call to coro.resume or coro.destroy.
  CallInstr->eraseFromParent();

  if (SubFn->user_empty())
    SubFn->eraseFromParent();

  return true;
}

// Remove suspend points that are simplified.
static void simplifySuspendPoints(coro::Shape &Shape) {
  auto &S = Shape.CoroSuspends;
  size_t I = 0, N = S.size();
  if (N == 0)
    return;
  for (;;) {
    if (simplifySuspendPoint(S[I], Shape.CoroBegin)) {
      if (--N == I)
        break;
      std::swap(S[I], S[N]);
      continue;
    }
    if (++I == N)
      break;
  }
  S.resize(N);
}

static void splitCoroutine(Function &F, CallGraph &CG, CallGraphSCC &SCC) {
  coro::Shape Shape(F);
  if (!Shape.CoroBegin)
    return;

  simplifySuspendPoints(Shape);
  buildCoroutineFrame(F, Shape);
  replaceFrameSize(Shape);

  // If there are no suspend points, no split required, just remove
  // the allocation and deallocation blocks, they are not needed.
  if (Shape.CoroSuspends.empty()) {
    handleNoSuspendCoroutine(Shape.CoroBegin, Shape.FrameTy);
    removeCoroEnds(Shape);
    postSplitCleanup(F);
    coro::updateCallGraph(F, {}, CG, SCC);
    return;
  }

  auto *ResumeEntry = createResumeEntryBlock(F, Shape);
  auto ResumeClone = createClone(F, ".resume", Shape, ResumeEntry, 0);
  auto DestroyClone = createClone(F, ".destroy", Shape, ResumeEntry, 1);
  auto CleanupClone = createClone(F, ".cleanup", Shape, ResumeEntry, 2);

  // We no longer need coro.end in F.
  removeCoroEnds(Shape);

  postSplitCleanup(F);
  postSplitCleanup(*ResumeClone);
  postSplitCleanup(*DestroyClone);
  postSplitCleanup(*CleanupClone);

  setCoroInfo(F, Shape.CoroBegin, {ResumeClone, DestroyClone, CleanupClone});
  coro::updateCallGraph(F, {ResumeClone, DestroyClone, CleanupClone}, CG, SCC);
}

// When we see the coroutine the first time, we insert an indirect call to a
// devirt trigger function and mark the coroutine that it is now ready for
// split.
static void prepareForSplit(Function &F, CallGraph &CG) {
  Module &M = *F.getParent();
#ifndef NDEBUG
  Function *DevirtFn = M.getFunction(CORO_DEVIRT_TRIGGER_FN);
  assert(DevirtFn && "coro.devirt.trigger function not found");
#endif

  F.addFnAttr(CORO_PRESPLIT_ATTR, PREPARED_FOR_SPLIT);

  // Insert an indirect call sequence that will be devirtualized by CoroElide
  // pass:
  //    %0 = call i8* @llvm.coro.subfn.addr(i8* null, i8 -1)
  //    %1 = bitcast i8* %0 to void(i8*)*
  //    call void %1(i8* null)
  coro::LowererBase Lowerer(M);
  Instruction *InsertPt = F.getEntryBlock().getTerminator();
  auto *Null = ConstantPointerNull::get(Type::getInt8PtrTy(F.getContext()));
  auto *DevirtFnAddr =
      Lowerer.makeSubFnCall(Null, CoroSubFnInst::RestartTrigger, InsertPt);
  auto *IndirectCall = CallInst::Create(DevirtFnAddr, Null, "", InsertPt);

  // Update CG graph with an indirect call we just added.
  CG[&F]->addCalledFunction(IndirectCall, CG.getCallsExternalNode());
}

// Make sure that there is a devirtualization trigger function that CoroSplit
// pass uses the force restart CGSCC pipeline. If devirt trigger function is not
// found, we will create one and add it to the current SCC.
static void createDevirtTriggerFunc(CallGraph &CG, CallGraphSCC &SCC) {
  Module &M = CG.getModule();
  if (M.getFunction(CORO_DEVIRT_TRIGGER_FN))
    return;

  LLVMContext &C = M.getContext();
  auto *FnTy = FunctionType::get(Type::getVoidTy(C), Type::getInt8PtrTy(C),
                                 /*IsVarArgs=*/false);
  Function *DevirtFn =
      Function::Create(FnTy, GlobalValue::LinkageTypes::PrivateLinkage,
                       CORO_DEVIRT_TRIGGER_FN, &M);
  DevirtFn->addFnAttr(Attribute::AlwaysInline);
  auto *Entry = BasicBlock::Create(C, "entry", DevirtFn);
  ReturnInst::Create(C, Entry);

  auto *Node = CG.getOrInsertFunction(DevirtFn);

  SmallVector<CallGraphNode *, 8> Nodes(SCC.begin(), SCC.end());
  Nodes.push_back(Node);
  SCC.initialize(Nodes);
}

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroSplit : public CallGraphSCCPass {
  static char ID; // Pass identification, replacement for typeid
  CoroSplit() : CallGraphSCCPass(ID) {}

  bool Run = false;

  // A coroutine is identified by the presence of coro.begin intrinsic, if
  // we don't have any, this pass has nothing to do.
  bool doInitialization(CallGraph &CG) override {
    Run = coro::declaresIntrinsics(CG.getModule(), {"llvm.coro.begin"});
    return CallGraphSCCPass::doInitialization(CG);
  }

  bool runOnSCC(CallGraphSCC &SCC) override {
    if (!Run)
      return false;

    // Find coroutines for processing.
    SmallVector<Function *, 4> Coroutines;
    for (CallGraphNode *CGN : SCC)
      if (auto *F = CGN->getFunction())
        if (F->hasFnAttribute(CORO_PRESPLIT_ATTR))
          Coroutines.push_back(F);

    if (Coroutines.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    createDevirtTriggerFunc(CG, SCC);

    for (Function *F : Coroutines) {
      Attribute Attr = F->getFnAttribute(CORO_PRESPLIT_ATTR);
      StringRef Value = Attr.getValueAsString();
      DEBUG(dbgs() << "CoroSplit: Processing coroutine '" << F->getName()
                   << "' state: " << Value << "\n");
      if (Value == UNPREPARED_FOR_SPLIT) {
        prepareForSplit(*F, CG);
        continue;
      }
      F->removeFnAttr(CORO_PRESPLIT_ATTR);
      splitCoroutine(*F, CG, SCC);
    }
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};
}

char CoroSplit::ID = 0;
INITIALIZE_PASS(
    CoroSplit, "coro-split",
    "Split coroutine into a set of functions driving its state machine", false,
    false)

Pass *llvm::createCoroSplitPass() { return new CoroSplit(); }
