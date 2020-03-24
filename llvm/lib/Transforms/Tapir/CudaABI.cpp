//===- CudaABI.cpp - Lower Tapir to the Kitsune CUDA back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune CUDA ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "cudaabi"

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // currently a no-op...
}

void CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
}

void CudaABI::postProcessHelper(Function &F) {
}

void CudaABI::processOutlinedTask(Function &F) {
}

void CudaABI::processSpawner(Function &F) {
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
}

LoopOutlineProcessor *CudaABI::getLoopOutlineProcessor(
    const TapirLoopInfo *TL) const {
  return new PTXLoop(M);
}

// Static counter for assigning IDs to kernels.
unsigned PTXLoop::NextKernelID = 0;

PTXLoop::PTXLoop(Module &M)
    : LoopOutlineProcessor(M, PTXM), PTXM("ptxModule", M.getContext()) {
  // Assign an ID to this kernel.
  MyKernelID = NextKernelID++;

  // Setup an NVPTX triple.
  Triple PTXTriple(M.getTargetTriple());
  PTXTriple.setArch(Triple::nvptx64);
  PTXTriple.setOS(Triple::CUDA);

  PTXM.setTargetTriple(PTXTriple.str());

  // Find the NVPTX module pass which will create the PTX code
  std::string error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", PTXTriple, error);
  LLVM_DEBUG({
      if (!PTXTarget)
        dbgs() << "ERROR: Failed to lookup NVPTX target: " << error << "\n";
    });
  assert(PTXTarget && "Failed to find NVPTX target");

  PTXTargetMachine =
      PTXTarget->createTargetMachine(PTXTriple.getTriple(), "sm_70", "+ptx60",
                                     TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  PTXM.setDataLayout(PTXTargetMachine->createDataLayout());

  // Insert runtime-function declarations in PTX host modules.
  Type *PTXInt32Ty = Type::getInt32Ty(PTXM.getContext());
  GetThreadIdx = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                          PTXInt32Ty);
  GetBlockIdx = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ctaid.x",
                                         PTXInt32Ty);
  GetBlockDim = PTXM.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ntid.x",
                                         PTXInt32Ty);

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  KitsuneCUDAInit = M.getOrInsertFunction("__kitsune_cuda_init", VoidTy);
  KitsuneGPUInitKernel = M.getOrInsertFunction("__kitsune_gpu_init_kernel",
                                               VoidTy, Int32Ty, VoidPtrTy);
  KitsuneGPUInitField = M.getOrInsertFunction("__kitsune_gpu_init_field",
                                              VoidTy, Int32Ty, VoidPtrTy,
                                              VoidPtrTy, Int32Ty, Int64Ty,
                                              Int8Ty);
  KitsuneGPUSetRunSize = M.getOrInsertFunction("__kitsune_gpu_set_run_size",
                                               VoidTy, Int32Ty, Int64Ty,
                                               Int64Ty, Int64Ty);
  KitsuneGPURunKernel = M.getOrInsertFunction("__kitsune_gpu_run_kernel",
                                              VoidTy, Int32Ty);
  KitsuneGPUFinish = M.getOrInsertFunction("__kitsune_gpu_finish", VoidTy);
}

void PTXLoop::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the loop control inputs.

  // The first parameter defines the extent of the index space, i.e., the number
  // of threads to launch.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The third parameter defines the grainsize, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the remaining inputs
  for (Value *V : TLInputsFixed) {
    assert(!HelperArgs.count(V));
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }
}

unsigned PTXLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned PTXLoop::getLimitArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the loop limit is the first input.
  return 0;
}

void PTXLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();

  Function *Helper = Out.Outline;
  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg = cast<Instruction>(
      VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Helper->setLinkage(Function::ExternalLinkage);

  // Get the thread ID for this invocation of Helper.
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadIdx);
  Value *BlockIdx = B.CreateCall(GetBlockIdx);
  Value *BlockDim = B.CreateCall(GetBlockDim);
  Value *ThreadID = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim), "threadId"),
      PrimaryIV->getType(), false);

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter;

    // Get the grainsize value, which is either constant or the third LC arg.
    if (unsigned ConstGrainsize = TL.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*++(++OutlineArgsIter);
  }
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);

  LLVMContext &Ctx = PTXM.getContext();
  // Add the necessary NVPTX to mark the global function
  NamedMDNode *Annotations =
    PTXM.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(Helper));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
                                                     1)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  LLVM_DEBUG(dbgs() << "PTX Module: " << PTXM);

  legacy::PassManager *PassManager = new legacy::PassManager;

  PassManager->add(createVerifierPass());

  // Add in our optimization passes

  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createReassociatePass());
  PassManager->add(createGVNPass());
  PassManager->add(createCFGSimplificationPass());
  PassManager->add(createSLPVectorizerPass());
  //PassManager->add(createBreakCriticalEdgesPass());
  PassManager->add(createConstantPropagationPass());
  PassManager->add(createDeadInstEliminationPass());
  PassManager->add(createDeadStoreEliminationPass());
  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createCFGSimplificationPass());

  SmallVector<char, 65536> Buf;
  raw_svector_ostream Ostr(Buf);

  bool Fail = PTXTargetMachine->addPassesToEmitFile(
      *PassManager, Ostr, &Ostr,
      TargetMachine::CodeGenFileType::CGFT_AssemblyFile, false);
  assert(!Fail && "Failed to emit PTX");

  PassManager->run(PTXM);

  delete PassManager;

  // Create a global string to hold the PTX code
  Constant *PCS = ConstantDataArray::getString(M.getContext(),
                                               Ostr.str().str());
  PTXGlobal = new GlobalVariable(M, PCS->getType(), true,
                                 GlobalValue::PrivateLinkage, PCS,
                                 "ptx" + Twine(MyKernelID));
}

void PTXLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  Task *T = TL.getTask();
  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
  Function *Parent = ReplCall->getFunction();
  Value *RunStart = ReplCall->getArgOperand(getIVArgIndex(*Parent,
                                                          TOI.InputSet));
  Value *TripCount = ReplCall->getArgOperand(getLimitArgIndex(*Parent,
                                                              TOI.InputSet));
  IRBuilder<> B(ReplCall);

  Value *KernelID = ConstantInt::get(Int32Ty, MyKernelID);
  Value *PTXStr = B.CreateBitCast(PTXGlobal, VoidPtrTy);

  B.CreateCall(KitsuneCUDAInit, {});
  B.CreateCall(KitsuneGPUInitKernel, { KernelID, PTXStr });

  for (Value *V : TOI.InputSet) {
    Value *ElementSize = nullptr;
    Value *VPtr;
    Value *FieldName;
    Value *Size = nullptr;

    // TODO: fix
    // this is a temporary hack to get the size of the field
    // it will currently only work for a limited case

    if (BitCastInst *BC = dyn_cast<BitCastInst>(V)) {
      CallInst *CI = dyn_cast<CallInst>(BC->getOperand(0));
      assert(CI && "Unable to detect field size");

      Value *Bytes = CI->getOperand(0);
      assert(Bytes->getType()->isIntegerTy(64));

      PointerType *PT = dyn_cast<PointerType>(V->getType());
      IntegerType *IntT = dyn_cast<IntegerType>(PT->getElementType());
      assert(IntT && "Expected integer type");

      Constant *Fn = ConstantDataArray::getString(Ctx, CI->getName());
      GlobalVariable *FieldNameGlobal =
          new GlobalVariable(M, Fn->getType(), true,
                             GlobalValue::PrivateLinkage, Fn, "field.name");
      FieldName = B.CreateBitCast(FieldNameGlobal, VoidPtrTy);
      VPtr = B.CreateBitCast(V, VoidPtrTy);
      ElementSize = ConstantInt::get(Int32Ty, IntT->getBitWidth()/8);
      Size = B.CreateUDiv(Bytes, ConstantInt::get(Int64Ty,
                                                  IntT->getBitWidth()/8));
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      Constant *Fn = ConstantDataArray::getString(Ctx, AI->getName());
      GlobalVariable *FieldNameGlobal =
          new GlobalVariable(M, Fn->getType(), true,
                             GlobalValue::PrivateLinkage, Fn, "field.name");
      FieldName = B.CreateBitCast(FieldNameGlobal, VoidPtrTy);
      VPtr = B.CreateBitCast(V, VoidPtrTy);
      ArrayType *AT = dyn_cast<ArrayType>(AI->getAllocatedType());
      assert(AT && "Expected array type");
      ElementSize =
          ConstantInt::get(Int32Ty,
                           AT->getElementType()->getPrimitiveSizeInBits()/8);
      Size = ConstantInt::get(Int64Ty, AT->getNumElements());
    }

    unsigned m = 0;
    for (const User *U : V->users()) {
      if (const Instruction *I = dyn_cast<Instruction>(U)) {
        // TODO: Properly restrict this check to users within the cloned loop
        // body.  Checking the dominator tree doesn't properly check
        // exception-handling code, although it's not clear we should see such
        // code in these loops.
        if (!DT.dominates(T->getEntry(), I->getParent()))
          continue;

        if (isa<LoadInst>(U))
          m |= 1;
        else if (isa<StoreInst>(U))
          m |= 2;
      }
    }
    Value *Mode = ConstantInt::get(Int8Ty, m);
    if (ElementSize && Size)
      B.CreateCall(KitsuneGPUInitField, { KernelID, FieldName, VPtr,
                                          ElementSize, Size, Mode });
  }

  Value *RunSize = B.CreateSub(TripCount, ConstantInt::get(TripCount->getType(),
                                                           1));
  B.CreateCall(KitsuneGPUSetRunSize, { KernelID, RunSize, RunStart, RunStart });

  B.CreateCall(KitsuneGPURunKernel, { KernelID });

  B.CreateCall(KitsuneGPUFinish, {});

  ReplCall->eraseFromParent();
}
