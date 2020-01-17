//===- QthreadsABI.cpp - Lower Tapir into Qthreads runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the QthreadsABI interface, which is used to convert
// Tapir instructions to calls into the Qthreads runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/QthreadsABI.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "qthreadsabi"

static cl::opt<bool> UseCopyargs(
    "qthreads-use-fork-copyargs", cl::init(false), cl::Hidden,
    cl::desc("Use copyargs variant of fork"));

// Accessors for opaque Qthreads RTS functions
FunctionCallee QthreadsABI::get_qthread_num_workers() {
  if (QthreadNumWorkers)
    return QthreadNumWorkers;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(Type::getInt16Ty(C), {}, false);
  QthreadNumWorkers = M.getOrInsertFunction("qthread_num_workers", FTy, AL);
  return QthreadNumWorkers;
}

FunctionCallee QthreadsABI::get_qthread_fork_copyargs() {
  if (QthreadForkCopyargs)
    return QthreadForkCopyargs;

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C),
      { QthreadFTy,            // qthread_f f
        Type::getInt8PtrTy(C), // const void *arg
        DL.getIntPtrType(C),   // size_t arg_size
        Type::getInt64PtrTy(C) // aligned_t *ret
      }, false);
  
  QthreadForkCopyargs = M.getOrInsertFunction("qthread_fork_copyargs", FTy, AL);
  return QthreadForkCopyargs;
}

FunctionCallee QthreadsABI::get_qthread_initialize() {
  if (QthreadInitialize)
    return QthreadInitialize;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C), {}, false);
  
  QthreadInitialize = M.getOrInsertFunction("qthread_initialize", FTy, AL);
  return QthreadInitialize;
}

FunctionCallee QthreadsABI::get_qt_sinc_create() {
  if (QtSincCreate)
    return QtSincCreate;

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt8PtrTy(C),
      { DL.getIntPtrType(C),   // size_t size
        Type::getInt8PtrTy(C), // void *initval
        Type::getInt8PtrTy(C), // void *op
        DL.getIntPtrType(C)    // size_t expect
      },
      false);
  
  QtSincCreate = M.getOrInsertFunction("qt_sinc_create", FTy, AL);
  return QtSincCreate;
}

FunctionCallee QthreadsABI::get_qt_sinc_expect() {
  if (QtSincExpect)
    return QtSincExpect;

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(C),
      { Type::getInt8PtrTy(C), // sync_t *s
        DL.getIntPtrType(C)    // size_t incr
      },
      false);
  
  QtSincExpect = M.getOrInsertFunction("qt_sinc_expect", FTy, AL);
  return QtSincExpect;
}

FunctionCallee QthreadsABI::get_qt_sinc_submit() {
  if (QtSincSubmit)
    return QtSincSubmit;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(C),
      { Type::getInt8PtrTy(C), // sync_t *s
        Type::getInt8PtrTy(C)  // void *val
      },
      false);
  
  QtSincSubmit = M.getOrInsertFunction("qt_sinc_submit", FTy, AL);
  return QtSincSubmit;
}

FunctionCallee QthreadsABI::get_qt_sinc_wait() {
  if (QtSincWait)
    return QtSincWait;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(C),
      { Type::getInt8PtrTy(C), // sync_t *s
        Type::getInt8PtrTy(C)  // void *target
      },
      false);
  
  QtSincWait = M.getOrInsertFunction("qt_sinc_wait", FTy, AL);
  return QtSincWait;
}

FunctionCallee QthreadsABI::get_qt_sinc_destroy() {
  if (QtSincDestroy)
    return QtSincDestroy;

  LLVMContext &C = M.getContext();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(C),
      { Type::getInt8PtrTy(C), // sync_t *s
      },
      false);
  
  QtSincDestroy = M.getOrInsertFunction("qt_sinc_destroy", FTy, AL);
  return QtSincDestroy;
}

#define QTHREAD_FUNC(name) get_##name()

QthreadsABI::QthreadsABI(Module &M) : TapirTarget(M) {
  LLVMContext &C = M.getContext();
  // Initialize any types we need for lowering.
  QthreadFTy = PointerType::getUnqual(
      FunctionType::get(Type::getInt64Ty(C), { Type::getInt8PtrTy(C) }, false));
}

/// Lower a call to get the grainsize of this Tapir loop.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
Value *QthreadsABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  IRBuilder<> Builder(GrainsizeCall);

  // Get 8 * workers
  Value *Workers = Builder.CreateCall(QTHREAD_FUNC(qthread_num_workers));
  Value *WorkersX8 = Builder.CreateIntCast(
      Builder.CreateMul(Workers, ConstantInt::get(Workers->getType(), 8)),
      Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) =
  //           (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, WorkersX8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       WorkersX8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Value *Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

Value *QthreadsABI::getOrCreateSinc(Value *SyncRegion, Function *F) {
  LLVMContext &C = M.getContext();
  Value* sinc;
  if((sinc = SyncRegionToSinc[SyncRegion]))
    return sinc;
  else {
    Value* zero = ConstantInt::get(Type::getInt64Ty(C), 0);
    Value* null = Constant::getNullValue(Type::getInt8PtrTy(C));
    std::vector<Value*> createArgs = {zero, null, null, zero};
    sinc = CallInst::Create(QTHREAD_FUNC(qt_sinc_create), createArgs, "",
                            F->getEntryBlock().getTerminator());
    SyncRegionToSinc[SyncRegion] = sinc;

    // Make sure we destroy the sinc at all exit points to prevent memory leaks
    for(BasicBlock &BB : *F) {
      if(isa<ReturnInst>(BB.getTerminator())){
        CallInst::Create(QTHREAD_FUNC(qt_sinc_destroy), {sinc}, "",
                         BB.getTerminator());
      }
    }

    return sinc;
  }
}

void QthreadsABI::lowerSync(SyncInst &SI) {
  IRBuilder<> builder(&SI); 
  auto F = SI.getParent()->getParent(); 
  auto& C = M.getContext(); 
  auto null = Constant::getNullValue(Type::getInt8PtrTy(C)); 
  Value* SR = SI.getSyncRegion(); 
  auto sinc = getOrCreateSinc(SR, F); 
  std::vector<Value *> args = {sinc, null}; 
  auto sincwait = QTHREAD_FUNC(qt_sinc_wait); 
  builder.CreateCall(sincwait, args);
  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
}

void QthreadsABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
  BasicBlock *CallBlock = ReplStart->getParent();

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  // At this point, we have a call in the parent to a function containing the
  // task body.  That function takes as its argument a pointer to a structure
  // containing the inputs to the task body.  This structure is initialized in
  // the parent immediately before the call.

  // To match the Qthreads ABI, we replace the existing call with a call to
  // qthreads_fork_copyargs.
  IRBuilder<> CallerIRBuilder(ReplCall);
  Value *OutlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      Outlined, QthreadFTy);
  AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
  Type *ArgsTy = CallerArgStruct->getAllocatedType();
  Value *ArgStructPtr = CallerIRBuilder.CreateBitCast(CallerArgStruct,
                                                      Type::getInt8PtrTy(C));
  Constant *Null = Constant::getNullValue(Type::getInt64PtrTy(C));
  ConstantInt *ArgSize = ConstantInt::get(DL.getIntPtrType(C),
                                          DL.getTypeAllocSize(ArgsTy));
  CallInst *Call = CallerIRBuilder.CreateCall(
      QTHREAD_FUNC(qthread_fork_copyargs), { OutlinedFnPtr, ArgStructPtr,
                                             ArgSize, Null });
  Call->setDebugLoc(ReplCall->getDebugLoc());
  TOI.replaceReplCall(Call);
  ReplCall->eraseFromParent();

  // Add lifetime intrinsics for the argument struct.  TODO: Move this logic
  // into underlying LoweringUtils routines?
  CallerIRBuilder.SetInsertPoint(ReplStart);
  CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);
  CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
  CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgSize);

  if (TOI.ReplUnwind)
    // We assume that qthread_fork_copyargs dealt with the exception.  But
    // replacing the invocation of the helper function with the call to
    // qthread_fork_copyargs will remove the terminator from CallBlock.  Restore
    // that terminator here.
    BranchInst::Create(TOI.ReplRet, CallBlock);

  // VERIFY: If we're using fork_copyargs, we don't need a separate helper
  // function to manage the allocation of the argument structure.
}

void QthreadsABI::preProcessFunction(Function &F, TaskInfo &TI,
                                     bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  LLVMContext &C = M.getContext();
  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask())
      continue;
    DetachInst *Detach = T->getDetach();
    BasicBlock *detB = Detach->getParent();
    BasicBlock *Spawned = T->getEntry();
    Value *SR = Detach->getSyncRegion(); 
    Value *sinc = getOrCreateSinc(SR, &F);

    // Add an expect increment before spawning
    IRBuilder<> preSpawnB(detB);
    Value* one = ConstantInt::get(Type::getInt64Ty(C), 1);
    std::vector<Value*> expectArgs = {sinc, one};
    CallInst::Create(QTHREAD_FUNC(qt_sinc_expect), expectArgs, "", Detach);

    // Add a submit to end of task body
    //
    // TB: I would interpret the above comment to mean we want qt_sinc_submit()
    // before the task terminates.  But the code I see for inserting
    // qt_sinc_submit just inserts the call at the end of the entry block of the
    // task, which is not necessarily the end of the task.  I kept the code I
    // found, but I'm not sure if it is correct.
    IRBuilder<> footerB(Spawned->getTerminator());
    Value* null = Constant::getNullValue(Type::getInt8PtrTy(C));
    std::vector<Value*> submitArgs = {sinc, null};
    footerB.CreateCall(QTHREAD_FUNC(qt_sinc_submit), submitArgs);
  }
}

void QthreadsABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  CallInst::Create(QTHREAD_FUNC(qthread_initialize), "",
                   F.getEntryBlock().getFirstNonPHIOrDbg());
}

void QthreadsABI::postProcessHelper(Function &F) {}

