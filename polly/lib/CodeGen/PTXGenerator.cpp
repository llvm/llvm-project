//===------ PTXGenerator.cpp -  IR helper to create loops -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create GPU parallel codes as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/PTXGenerator.h"

#ifdef GPU_CODEGEN
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"

#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace polly;

PTXGenerator::PTXGenerator(IRBuilder<> &Builder, Pass *P,
                           const std::string &Triple):
  Builder(Builder), P(P), GPUTriple(Triple), GridWidth(1), GridHeight(1),
  BlockWidth(1), BlockHeight(1), OutputBytes(0) {

  InitializeGPUDataTypes();
}

Module *PTXGenerator::getModule() {
  return Builder.GetInsertBlock()->getParent()->getParent();
}

Function *PTXGenerator::createSubfunctionDefinition(int NumArgs) {
  assert(NumArgs == 1 && "we support only one array access now.");

  Module *M = getModule();
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type*> Arguments;
  for (int i = 0; i < NumArgs; i++)
    Arguments.push_back(Builder.getInt8PtrTy());
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *FN = Function::Create(FT, Function::InternalLinkage,
      F->getName() + "_ptx_subfn", M);
  FN->setCallingConv(CallingConv::PTX_Kernel);

  // Do not run any optimization pass on the new function.
  P->getAnalysis<polly::ScopDetection>().markFunctionAsInvalid(FN);

  for (Function::arg_iterator AI = FN->arg_begin(); AI != FN->arg_end(); ++AI)
    AI->setName("ptx.Array");

  return FN;
}

void PTXGenerator::createSubfunction(SetVector<Value*> &UsedValues,
                                     SetVector<Value*> &OriginalIVS,
                                     PTXGenerator::ValueToValueMapTy &VMap,
                                     Function **SubFunction) {
  Function *FN = createSubfunctionDefinition(UsedValues.size());
  Module *M = getModule();
  LLVMContext &Context = FN->getContext();
  IntegerType *Ty = Builder.getInt64Ty();

  // Store the previous basic block.
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "ptx.setup", FN);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "ptx.exit", FN);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "ptx.loop_body", FN);

  DominatorTree &DT = P->getAnalysis<DominatorTree>();
  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(BodyBB, HeaderBB);

  Builder.SetInsertPoint(HeaderBB);

  // Insert VMap items with maps of array base address on the host to base
  // address on the device.
  Function::arg_iterator AI = FN->arg_begin();
  for (unsigned j = 0; j < UsedValues.size(); j++) {
    Value *BaseAddr = UsedValues[j];
    Type *ArrayTy = BaseAddr->getType();
    Value *Param = Builder.CreateBitCast(AI, ArrayTy);
    VMap.insert(std::make_pair<Value*, Value*>(BaseAddr, Param));
    AI++;
  }

  // FIXME: These intrinsics should be inserted on-demand. However, we insert
  // them all currently for simplicity.
  Function *GetNctaidX =
    Intrinsic::getDeclaration(M, Intrinsic::ptx_read_nctaid_x);
  Function *GetNctaidY =
    Intrinsic::getDeclaration(M, Intrinsic::ptx_read_nctaid_y);
  Function *GetCtaidX =
    Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_x);
  Function *GetCtaidY =
    Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_y);
  Function *GetNtidX = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ntid_x);
  Function *GetNtidY = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ntid_y);
  Function *GetTidX = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_x);
  Function *GetTidY = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_y);

  Value *GridWidth = Builder.CreateCall(GetNctaidX);
  GridWidth = Builder.CreateIntCast(GridWidth, Ty, false);
  Value *GridHeight = Builder.CreateCall(GetNctaidY);
  GridHeight = Builder.CreateIntCast(GridHeight, Ty, false);
  Value *BlockWidth = Builder.CreateCall(GetNtidX);
  BlockWidth = Builder.CreateIntCast(BlockWidth, Ty, false);
  Value *BlockHeight = Builder.CreateCall(GetNtidY);
  BlockHeight = Builder.CreateIntCast(BlockHeight, Ty, false);
  Value *BIDx = Builder.CreateCall(GetCtaidX);
  BIDx = Builder.CreateIntCast(BIDx, Ty, false);
  Value *BIDy = Builder.CreateCall(GetCtaidY);
  BIDy = Builder.CreateIntCast(BIDy, Ty, false);
  Value *TIDx = Builder.CreateCall(GetTidX);
  TIDx = Builder.CreateIntCast(TIDx, Ty, false);
  Value *TIDy = Builder.CreateCall(GetTidY);
  TIDy = Builder.CreateIntCast(TIDy, Ty, false);

  Builder.CreateBr(BodyBB);
  Builder.SetInsertPoint(BodyBB);

  unsigned NumDims = OriginalIVS.size();
  std::vector<Value *> Substitutions;
  Value *BlockID, *ThreadID;
  switch (NumDims) {
  case 1: {
    Value *BlockSize = Builder.CreateMul(BlockWidth, BlockHeight,
                                         "p_gpu_blocksize");
    BlockID = Builder.CreateMul(BIDy, GridWidth, "p_gpu_index_i");
    BlockID = Builder.CreateAdd(BlockID, BIDx);
    BlockID = Builder.CreateMul(BlockID, BlockSize);
    ThreadID = Builder.CreateMul(TIDy, BlockWidth, "p_gpu_index_j");
    ThreadID = Builder.CreateAdd(ThreadID, TIDx);
    ThreadID = Builder.CreateAdd(ThreadID, BlockID);
    Substitutions.push_back(ThreadID);
    break;
  }
  case 2: {
    BlockID = Builder.CreateMul(BIDy, GridWidth, "p_gpu_index_i");
    BlockID = Builder.CreateAdd(BlockID, BIDx);
    Substitutions.push_back(BlockID);
    ThreadID = Builder.CreateMul(TIDy, BlockWidth, "p_gpu_index_j");
    ThreadID = Builder.CreateAdd(ThreadID, TIDx);
    Substitutions.push_back(ThreadID);
    break;
  }
  case 3: {
    BlockID = Builder.CreateMul(BIDy, GridWidth, "p_gpu_index_i");
    BlockID = Builder.CreateAdd(BlockID, BIDx);
    Substitutions.push_back(BlockID);
    Substitutions.push_back(TIDy);
    Substitutions.push_back(TIDx);
    break;
  }
  case 4: {
    Substitutions.push_back(BIDy);
    Substitutions.push_back(BIDx);
    Substitutions.push_back(TIDy);
    Substitutions.push_back(TIDx);
    break;
  }
  default:
    assert(true &&
           "We cannot transform parallel loops whose depth is larger than 4.");
    return;
  }

  assert(OriginalIVS.size() == Substitutions.size()
         && "The size of IVS should be equal to the size of substitutions.");
  for (unsigned i = 0; i < OriginalIVS.size(); ++i) {
    VMap.insert(std::make_pair<Value*, Value*>(OriginalIVS[i],
                                               Substitutions[i]));
  }

  Builder.CreateBr(ExitBB);
  Builder.SetInsertPoint(--Builder.GetInsertPoint());
  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add the termination of the ptx-device subfunction.
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(LoopBody);
  *SubFunction = FN;
}

void PTXGenerator::startGeneration(SetVector<Value*> &UsedValues,
                                   SetVector<Value*> &OriginalIVS,
                                   ValueToValueMapTy &VMap,
                                   BasicBlock::iterator *LoopBody) {
  Function *SubFunction;
  BasicBlock::iterator PrevInsertPoint = Builder.GetInsertPoint();
  createSubfunction(UsedValues, OriginalIVS, VMap, &SubFunction);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(PrevInsertPoint);
}

IntegerType *PTXGenerator::getInt64Type() {
  return Builder.getInt64Ty();
}

PointerType *PTXGenerator::getI8PtrType() {
  return PointerType::getUnqual(Builder.getInt8Ty());
}

PointerType *PTXGenerator::getPtrI8PtrType() {
  return PointerType::getUnqual(getI8PtrType());
}

PointerType *PTXGenerator::getFloatPtrType() {
  return llvm::Type::getFloatPtrTy(getModule()->getContext());
}

PointerType *PTXGenerator::getGPUContextPtrType() {
  return PointerType::getUnqual(ContextTy);
}

PointerType *PTXGenerator::getGPUModulePtrType() {
  return PointerType::getUnqual(ModuleTy);
}

PointerType *PTXGenerator::getGPUDevicePtrType() {
  return PointerType::getUnqual(DeviceTy);
}

PointerType *PTXGenerator::getPtrGPUDevicePtrType() {
  return PointerType::getUnqual(DevDataTy);
}

PointerType *PTXGenerator::getGPUFunctionPtrType() {
  return PointerType::getUnqual(KernelTy);
}

PointerType *PTXGenerator::getGPUEventPtrType() {
  return PointerType::getUnqual(EventTy);
}

void PTXGenerator::InitializeGPUDataTypes() {
  LLVMContext &Context = getModule()->getContext();

  ContextTy = StructType::create(Context, "struct.PollyGPUContextT");
  ModuleTy = StructType::create(Context, "struct.PollyGPUModuleT");
  KernelTy = StructType::create(Context, "struct.PollyGPUFunctionT");
  DeviceTy = StructType::create(Context, "struct.PollyGPUDeviceT");
  DevDataTy = StructType::create(Context,"struct.PollyGPUDevicePtrT");
  EventTy = StructType::create(Context, "struct.PollyGPUEventT");
}

void PTXGenerator::createCallInitDevice(Value *Context, Value *Device) {
  const char *Name = "polly_initDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(PointerType::getUnqual(getGPUContextPtrType()));
    Args.push_back(PointerType::getUnqual(getGPUDevicePtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, Context, Device);
}

void PTXGenerator::createCallGetPTXModule(Value *Buffer, Value *Module) {
  const char *Name = "polly_getPTXModule";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(PointerType::getUnqual(getGPUModulePtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, Buffer, Module);
}

void PTXGenerator::createCallGetPTXKernelEntry(Value *Entry, Value *Module,
                                               Value *Kernel) {
  const char *Name = "polly_getPTXKernelEntry";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(getGPUModulePtrType());
    Args.push_back(PointerType::getUnqual(getGPUFunctionPtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, Entry, Module, Kernel);
}

void PTXGenerator::createCallAllocateMemoryForHostAndDevice(Value *HostData,
                                                            Value *DeviceData,
                                                            Value *Size) {
  const char *Name = "polly_allocateMemoryForHostAndDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getPtrI8PtrType());
    Args.push_back(PointerType::getUnqual(getPtrGPUDevicePtrType()));
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, HostData, DeviceData, Size);
}

void PTXGenerator::createCallCopyFromHostToDevice(Value *DeviceData,
                                                  Value *HostData,
                                                  Value *Size) {
  const char *Name = "polly_copyFromHostToDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getI8PtrType());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, DeviceData, HostData, Size);
}

void PTXGenerator::createCallCopyFromDeviceToHost(Value *HostData,
                                                  Value *DeviceData,
                                                  Value *Size) {
  const char *Name = "polly_copyFromDeviceToHost";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, HostData, DeviceData, Size);
}

void PTXGenerator::createCallSetKernelParameters(Value *Kernel,
                                                 Value *BlockWidth,
                                                 Value *BlockHeight,
                                                 Value *DeviceData) {
  const char *Name = "polly_setKernelParameters";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    Args.push_back(getPtrGPUDevicePtrType());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall4(F, Kernel, BlockWidth, BlockHeight, DeviceData);
}

void PTXGenerator::createCallLaunchKernel(Value *Kernel, Value *GridWidth,
                                          Value *GridHeight) {
  const char *Name = "polly_launchKernel";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, Kernel, GridWidth, GridHeight);
}

void PTXGenerator::createCallStartTimerByCudaEvent(Value *StartEvent,
                                                   Value *StopEvent) {
  const char *Name = "polly_startTimerByCudaEvent";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(PointerType::getUnqual(getGPUEventPtrType()));
    Args.push_back(PointerType::getUnqual(getGPUEventPtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, StartEvent, StopEvent);
}

void PTXGenerator::createCallStopTimerByCudaEvent(Value *StartEvent,
                                                  Value *StopEvent,
                                                  Value *Timer) {
  const char *Name = "polly_stopTimerByCudaEvent";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getGPUEventPtrType());
    Args.push_back(getGPUEventPtrType());
    Args.push_back(getFloatPtrType());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, StartEvent, StopEvent, Timer);
}

void PTXGenerator::createCallCleanupGPGPUResources(Value *HostData,
                                                   Value *DeviceData,
                                                   Value *Module,
                                                   Value *Context,
                                                   Value *Kernel) {
  const char *Name = "polly_cleanupGPGPUResources";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type*> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getGPUModulePtrType());
    Args.push_back(getGPUContextPtrType());
    Args.push_back(getGPUFunctionPtrType());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall5(F, HostData, DeviceData, Module, Context, Kernel);
}

Value *PTXGenerator::getCUDAGridWidth() {
  return ConstantInt::get(getInt64Type(), GridWidth);
}

Value *PTXGenerator::getCUDAGridHeight() {
  return ConstantInt::get(getInt64Type(), GridHeight);
}

Value *PTXGenerator::getCUDABlockWidth() {
  return ConstantInt::get(getInt64Type(), BlockWidth);
}

Value *PTXGenerator::getCUDABlockHeight() {
  return ConstantInt::get(getInt64Type(), BlockHeight);
}

Value *PTXGenerator::getOutputArraySizeInBytes() {
  return ConstantInt::get(getInt64Type(), OutputBytes);
}

Value *PTXGenerator::createPTXKernelFunction(Function *SubFunction) {
  Module *M = getModule();
  std::string LLVMKernelStr;
  raw_string_ostream NameROS(LLVMKernelStr);
  formatted_raw_ostream FOS(NameROS);
  FOS << "target triple = \"" << GPUTriple <<"\"\n";
  SubFunction->print(FOS);

  // Insert ptx intrinsics into the kernel string.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ) {
    Function *F = I++;
    // Function must be a prototype and unused.
    if (F->isDeclaration() && F->isIntrinsic()) {
      switch (F->getIntrinsicID()) {
      case Intrinsic::ptx_read_nctaid_x:
      case Intrinsic::ptx_read_nctaid_y:
      case Intrinsic::ptx_read_ctaid_x:
      case Intrinsic::ptx_read_ctaid_y:
      case Intrinsic::ptx_read_ntid_x:
      case Intrinsic::ptx_read_ntid_y:
      case Intrinsic::ptx_read_tid_x:
      case Intrinsic::ptx_read_tid_y:
        F->print(FOS);
        break;
      default:
        break;
      }
    }
  }

  Value *LLVMKernel = Builder.CreateGlobalStringPtr(LLVMKernelStr,
                                                    "llvm_kernel");
  Value *MCPU = Builder.CreateGlobalStringPtr("sm_10", "mcpu");
  Value *Features = Builder.CreateGlobalStringPtr("", "cpu_features");

  Function *GetDeviceKernel = Intrinsic::getDeclaration(M,
                                                        Intrinsic::codegen);

  return Builder.CreateCall3(GetDeviceKernel, LLVMKernel, MCPU, Features);
}

Value *PTXGenerator::getPTXKernelEntryName(Function *SubFunction) {
  StringRef Entry = SubFunction->getName();
  return Builder.CreateGlobalStringPtr(Entry, "ptx_entry");
}

void PTXGenerator::eraseUnusedFunctions(Function *SubFunction) {
  Module *M = getModule();
  SubFunction->eraseFromParent();

  if (Function *FuncPTXReadNCtaidX = M->getFunction("llvm.ptx.read.nctaid.x")) {
    FuncPTXReadNCtaidX->eraseFromParent();
  }

  if (Function *FuncPTXReadNCtaidY = M->getFunction("llvm.ptx.read.nctaid.y")) {
    FuncPTXReadNCtaidY->eraseFromParent();
  }

  if (Function *FuncPTXReadCtaidX = M->getFunction("llvm.ptx.read.ctaid.x")) {
    FuncPTXReadCtaidX->eraseFromParent();
  }

  if (Function *FuncPTXReadCtaidY = M->getFunction("llvm.ptx.read.ctaid.y")) {
    FuncPTXReadCtaidY->eraseFromParent();
  }

  if (Function *FuncPTXReadNTidX = M->getFunction("llvm.ptx.read.ntid.x")) {
    FuncPTXReadNTidX->eraseFromParent();
  }

  if (Function *FuncPTXReadNTidY = M->getFunction("llvm.ptx.read.ntid.y")) {
    FuncPTXReadNTidY->eraseFromParent();
  }

  if (Function *FuncPTXReadTidX = M->getFunction("llvm.ptx.read.tid.x")) {
    FuncPTXReadTidX->eraseFromParent();
  }

  if (Function *FuncPTXReadTidY = M->getFunction("llvm.ptx.read.tid.y")) {
    FuncPTXReadTidY->eraseFromParent();
  }
}

void PTXGenerator::finishGeneration(Function *F) {
  // Define data used by the GPURuntime library.
  AllocaInst *PtrCUContext = Builder.CreateAlloca(getGPUContextPtrType(), 0,
                                                  "phcontext");
  AllocaInst *PtrCUDevice = Builder.CreateAlloca(getGPUDevicePtrType(), 0,
                                                 "phdevice");
  AllocaInst *PtrCUModule = Builder.CreateAlloca(getGPUModulePtrType(), 0,
                                                 "phmodule");
  AllocaInst *PtrCUKernel = Builder.CreateAlloca(getGPUFunctionPtrType(), 0,
                                                 "phkernel");
  AllocaInst *PtrCUStartEvent = Builder.CreateAlloca(getGPUEventPtrType(), 0,
                                                     "pstart_timer");
  AllocaInst *PtrCUStopEvent = Builder.CreateAlloca(getGPUEventPtrType(), 0,
                                                    "pstop_timer");
  AllocaInst *PtrDevData = Builder.CreateAlloca(getPtrGPUDevicePtrType(), 0,
                                                "pdevice_data");
  AllocaInst *PtrHostData = Builder.CreateAlloca(getI8PtrType(), 0,
                                                 "phost_data");
  Type *FloatTy = llvm::Type::getFloatTy(getModule()->getContext());
  AllocaInst *PtrElapsedTimes = Builder.CreateAlloca(FloatTy, 0, "ptimer");

  // Initialize the GPU device.
  createCallInitDevice(PtrCUContext, PtrCUDevice);

  // Create the GPU kernel module and entry function.
  Value *PTXString = createPTXKernelFunction(F);
  Value *PTXEntry = getPTXKernelEntryName(F);
  createCallGetPTXModule(PTXString, PtrCUModule);
  LoadInst *CUModule = Builder.CreateLoad(PtrCUModule, "cumodule");
  createCallGetPTXKernelEntry(PTXEntry, CUModule, PtrCUKernel);

  // Allocate device memory and its corresponding host memory.
  createCallAllocateMemoryForHostAndDevice(PtrHostData, PtrDevData,
                                           getOutputArraySizeInBytes());

  // Get the pointer to the device memory and set the GPU execution parameters.
  LoadInst *DData = Builder.CreateLoad(PtrDevData, "device_data");
  LoadInst *CUKernel = Builder.CreateLoad(PtrCUKernel, "cukernel");
  createCallSetKernelParameters(CUKernel, getCUDABlockWidth(),
                                getCUDABlockHeight(), DData);

  // Create the start and end timer and record the start time.
  createCallStartTimerByCudaEvent(PtrCUStartEvent, PtrCUStopEvent);

  // Launch the GPU kernel.
  createCallLaunchKernel(CUKernel, getCUDAGridWidth(), getCUDAGridHeight());

  // Copy the results back from the GPU to the host.
  LoadInst *HData = Builder.CreateLoad(PtrHostData, "host_data");
  createCallCopyFromDeviceToHost(HData, DData, getOutputArraySizeInBytes());

  // Record the end time.
  LoadInst *CUStartEvent = Builder.CreateLoad(PtrCUStartEvent, "start_timer");
  LoadInst *CUStopEvent = Builder.CreateLoad(PtrCUStopEvent, "stop_timer");
  createCallStopTimerByCudaEvent(CUStartEvent, CUStopEvent,
                                 PtrElapsedTimes);

  // Cleanup all the resources used.
  LoadInst *CUContext = Builder.CreateLoad(PtrCUContext, "cucontext");
  createCallCleanupGPGPUResources(HData, DData, CUModule, CUContext,
                                  CUKernel);

  // Erase the ptx kernel and device subfunctions and ptx intrinsics from
  // current module.
  eraseUnusedFunctions(F);
}
#endif /* GPU_CODEGEN */
