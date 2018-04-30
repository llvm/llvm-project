//===- OpenMPABI.cpp - Interface to the OpenMP runtime system -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP ABI to converts Tapir instructions to calls
// into the OpenMP runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "ompabi"

StructType *IdentTy = nullptr;
FunctionType *Kmpc_MicroTy = nullptr;
Constant *DefaultOpenMPPSource = nullptr;
Constant *DefaultOpenMPLocation = nullptr;
PointerType *KmpRoutineEntryPtrTy = nullptr;

// instruction of its thread id.
typedef DenseMap<Function *, Value *> OpenMPThreadIDAllocaMapTy;
OpenMPThreadIDAllocaMapTy OpenMPThreadIDAllocaMap;

// Maps a funtion to the instruction where we loaded the thread id addrs
typedef DenseMap<Function *, Value *> OpenMPThreadIDLoadMapTy;
OpenMPThreadIDLoadMapTy OpenMPThreadIDLoadMap;

// Maps an extracted forked function (Using CodeExtractor) to its
// corresponding task outlined function as required by OMP runtime.
typedef DenseMap<Function *, Function *> ExtractedToOutlinedMapTy;
ExtractedToOutlinedMapTy ExtractedToOutlinedMap;

// Maps an outlined task function to its corresponding task entry function.
typedef DenseMap<Function *, Function *> OutlinedToEntryMapTy;
OutlinedToEntryMapTy OutlinedToEntryMap;

Value *emitTaskInit(Function *Caller,
                                     IRBuilder<> &CallerIRBuilder,
                                     IRBuilder<> &CallerAllocaIRBuilder,
                                     Function *ForkedFn,
                                     ArrayRef<Value *> LoadedCapturedArgs);

void emitBranch(BasicBlock *Target, IRBuilder<> &IRBuilder);

Type *createKmpTaskTTy(Module *M, PointerType *KmpRoutineEntryPtrTy);

Type *createKmpTaskTWithPrivatesTy(Type *data);
Function *emitTaskOutlinedFunction(Module *M,
                                                    Type *SharedsPtrTy,
                                                    Function *ForkedFn);

DenseMap<Argument *, Value *> startFunction(Function *Fn);

Function *emitProxyTaskFunction(
    Type *KmpTaskTWithPrivatesPtrTy, Type *SharedsPtrTy, Function *TaskFunction,
    Value *TaskPrivatesMap);

void emitTaskwaitCall(Function *Caller,
                                       IRBuilder<> &CallerIRBuilder,
                                       const DataLayout &DL);

PointerType *getIdentTyPointerTy() {
  return PointerType::getUnqual(IdentTy);
}

FunctionType *getOrCreateKmpc_MicroTy(LLVMContext &Context) {
  if (Kmpc_MicroTy == nullptr) {
    auto *Int32PtrTy = PointerType::getUnqual(Type::getInt32Ty(Context));
    Type *MicroParams[] = {Int32PtrTy, Int32PtrTy};
    Kmpc_MicroTy =
        FunctionType::get(Type::getVoidTy(Context), MicroParams, true);
  }

  return Kmpc_MicroTy;
}

PointerType *getKmpc_MicroPointerTy(LLVMContext &Context) {
  return PointerType::getUnqual(getOrCreateKmpc_MicroTy(Context));
}

Constant *createRuntimeFunction(OpenMPRuntimeFunction Function,
                                                 Module *M) {
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *Int32Ty = Type::getInt32Ty(M->getContext());
  auto *Int32PtrTy = Type::getInt32PtrTy(M->getContext());
  // TODO double check for how SizeTy get created. Eventually, it get emitted
  // as i64 on my machine.
  auto *SizeTy = Type::getInt64Ty(M->getContext());
  auto *IdentTyPtrTy = getIdentTyPointerTy();
  Constant *RTLFn = nullptr;

  switch (Function) {
  case OMPRTL__kmpc_fork_call: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty,
                          getKmpc_MicroPointerTy(M->getContext())};
    FunctionType *FnTy = FunctionType::get(VoidTy, TypeParams, true);
    RTLFn = M->getOrInsertFunction("__kmpc_fork_call", FnTy);
    break;
  }
  case OMPRTL__kmpc_for_static_init_4: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty,    Int32Ty,
                          Int32PtrTy,   Int32PtrTy, Int32PtrTy,
                          Int32PtrTy,   Int32Ty,    Int32Ty};
    FunctionType *FnTy =
      FunctionType::get(VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = M->getOrInsertFunction("__kmpc_for_static_init_4", FnTy);
    break;
  }
  case OMPRTL__kmpc_for_static_fini: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = M->getOrInsertFunction("__kmpc_for_static_fini", FnTy);
    break;
  }
  case OMPRTL__kmpc_master: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_master", FnTy);
    break;
  }
  case OMPRTL__kmpc_end_master: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_end_master", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_task_alloc: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty, Int32Ty,
                          SizeTy,       SizeTy,  KmpRoutineEntryPtrTy};
    FunctionType *FnTy =
        FunctionType::get(VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_task_alloc", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_task: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty, VoidPtrTy};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_task", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_taskwait: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_taskwait", FnTy);
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    Type *TypeParams[] = {IdentTyPtrTy};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_global_thread_num", FnTy);
    break;
  }
  case OMPRTL__kmpc_barrier: {
    // NOTE There is more elaborate logic to emitting barriers based on the
    // directive kind. This is just the simplified version currently needed.
    // Check: CGOpenMPRuntime::emitBarrierCall.
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
      FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_barrier", FnTy);
    break;
  }
  case OMPRTL__kmpc_global_num_threads: {
    Type *TypeParams[] = {IdentTyPtrTy};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_global_num_threads", FnTy);
    break;
  }
  }
  return RTLFn;
}

CallInst *emitRuntimeCall(Value *Callee, ArrayRef<Value *> Args,
                                           const Twine &Name,
                                           BasicBlock *Parent) {
  IRBuilder<> Builder(Parent);
  CallInst *call = Builder.CreateCall(Callee, Args, None, Name);
  return call;
}

CallInst *emitRuntimeCall(Value *Callee, ArrayRef<Value *> Args,
                                           const Twine &Name,
                                           IRBuilder<> &IRBuilder) {
  CallInst *call = IRBuilder.CreateCall(Callee, Args, None, Name);
  return call;
}

Value *getThreadID(Function *F, IRBuilder<> &IRBuilder) {
  Value *ThreadID = nullptr;
  auto I = OpenMPThreadIDLoadMap.find(F);
  if (I != OpenMPThreadIDLoadMap.end()) {
    ThreadID = I->second;
    assert(ThreadID != nullptr && "A null thread ID associated to F");
    return ThreadID;
  }

  auto I2 = OpenMPThreadIDAllocaMap.find(F);

  if (I2 != OpenMPThreadIDAllocaMap.end()) {
    DataLayout DL(F->getParent());
    auto Alloca = I2->second;
    auto ThreadIDAddrs = IRBuilder.CreateLoad(Alloca);
    ThreadIDAddrs->setAlignment(DL.getTypeAllocSize(ThreadIDAddrs->getType()));
    ThreadID = IRBuilder.CreateLoad(ThreadIDAddrs);
    ((LoadInst *)ThreadID)
        ->setAlignment(DL.getTypeAllocSize(ThreadID->getType()));
    auto &Elem = OpenMPThreadIDLoadMap.FindAndConstruct(F);
    Elem.second = ThreadID;
    return ThreadID;
  }

  auto GTIDFn = createRuntimeFunction(
      OpenMPRuntimeFunction::OMPRTL__kmpc_global_thread_num, F->getParent());
  ThreadID = emitRuntimeCall(GTIDFn, {DefaultOpenMPLocation}, "", IRBuilder);
  auto &Elem = OpenMPThreadIDLoadMap.FindAndConstruct(F);
  Elem.second = ThreadID;

  return ThreadID;
}

Value *getThreadID(Function *F) {
  IRBuilder <> fstart(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  return getThreadID(F, fstart);
}

/// Creates a struct that contains elements corresponding to the arguments
/// of \param F.
StructType *createSharedsTy(Function *F) {
  LLVMContext &C = F->getParent()->getContext();
  auto FnParams = F->getFunctionType()->params();

  if (FnParams.size() == 0) {
    return StructType::create(SmallVector<llvm::Type*,1>(1,Type::getInt8Ty(C)), "anon");
  }

  return StructType::create(FnParams, "anon");
}

Type *getOrCreateIdentTy(Module *M) {
  if (M->getTypeByName("ident_t") == nullptr) {
    auto *Int32Ty = Type::getInt32Ty(M->getContext());
    auto *Int8PtrTy = Type::getInt8PtrTy(M->getContext());
    IdentTy = StructType::create(ArrayRef<llvm::Type*>({Int32Ty /* reserved_1 */,
                                 Int32Ty /* flags */, Int32Ty /* reserved_2 */,
                                 Int32Ty /* reserved_3 */,
                                 Int8PtrTy /* psource */}), "ident_t");
  } else if ((IdentTy = dyn_cast<StructType>(M->getTypeByName("ident_t")))->isOpaque()) {
      auto *Int32Ty = Type::getInt32Ty(M->getContext());
      auto *Int8PtrTy = Type::getInt8PtrTy(M->getContext());
      IdentTy->setBody(ArrayRef<llvm::Type*>({Int32Ty /* reserved_1 */,
                                   Int32Ty /* flags */, Int32Ty /* reserved_2 */,
                                   Int32Ty /* reserved_3 */,
                                   Int8PtrTy /* psource */}), false);
  }
  return IdentTy;
}

PointerType *emitKmpRoutineEntryT(LLVMContext &C) {
  if (!KmpRoutineEntryPtrTy) {
    // Build typedef kmp_int32 (* kmp_routine_entry_t)(kmp_int32, void *); type.
    auto *Int32Ty = Type::getInt32Ty(C);
    std::vector<Type *> KmpRoutineEntryTyArgs = {Int32Ty, Type::getInt8PtrTy(C)};
    KmpRoutineEntryPtrTy = PointerType::getUnqual(
        FunctionType::get(Int32Ty, KmpRoutineEntryTyArgs, false));
  }
  return KmpRoutineEntryPtrTy;
}

Type *createKmpTaskTTy(LLVMContext &C) {
  auto routine = emitKmpRoutineEntryT(C);
  auto *KmpCmplrdataTy =
      StructType::create(SmallVector<llvm::Type*,1>(1,routine),"kmp_cmplrdata_t");
  auto *KmpTaskTTy = StructType::create(ArrayRef<llvm::Type*>({
      Type::getInt8PtrTy(C), routine,
      Type::getInt32Ty(C), KmpCmplrdataTy, KmpCmplrdataTy}), "kmp_task_t");

  return KmpTaskTTy;
}

Type *createKmpTaskTWithPrivatesTy(Type *data) {
  auto *KmpTaskTWithPrivatesTy =
      StructType::create(ArrayRef<llvm::Type*>({createKmpTaskTTy(data->getContext()),data}),"kmp_task_t_with_privates");
  return KmpTaskTWithPrivatesTy;
}

Value *getOrCreateDefaultLocation(Module *M) {
  if (DefaultOpenMPPSource == nullptr) {
    const std::string DefaultLocStr = ";unknown;unknown;0;0;;";
    StringRef DefaultLocStrWithNull(DefaultLocStr.c_str(),
                                    DefaultLocStr.size() + 1);
    DataLayout DL(M);
    uint64_t Alignment = DL.getTypeAllocSize(Type::getInt8Ty(M->getContext()));
    Constant *C = ConstantDataArray::getString(M->getContext(),
                                               DefaultLocStrWithNull, false);
    // NOTE Are heap allocations not recommended in general or is it OK here?
    // I couldn't find a way to statically allocate an IRBuilder for a Module!
    auto *GV =
        new GlobalVariable(*M, C->getType(), true, GlobalValue::PrivateLinkage,
                           C, ".str", nullptr, GlobalValue::NotThreadLocal);
    GV->setAlignment(Alignment);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    DefaultOpenMPPSource = cast<Constant>(GV);
    DefaultOpenMPPSource = ConstantExpr::getBitCast(
        DefaultOpenMPPSource, Type::getInt8PtrTy(M->getContext()));
  }

  if (DefaultOpenMPLocation == nullptr) {
    // Constant *C = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0,
    // true);
    auto *Int32Ty = Type::getInt32Ty(M->getContext());
    std::vector<Constant *> Members = {
        ConstantInt::get(Int32Ty, 0, true), ConstantInt::get(Int32Ty, 2, true),
        ConstantInt::get(Int32Ty, 0, true), ConstantInt::get(Int32Ty, 0, true),
        DefaultOpenMPPSource};
    Constant *C = ConstantStruct::get(IdentTy, Members);
    auto *GV =
        new GlobalVariable(*M, C->getType(), true, GlobalValue::PrivateLinkage,
                           C, "", nullptr, GlobalValue::NotThreadLocal);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GV->setAlignment(8);
    DefaultOpenMPLocation = GV;
  }

  return DefaultOpenMPLocation;
}

//##############################################################################

/// Get or create the worker count for the spawning function.
static Value *GetOrCreateWorkerCount(Function &F) {
  // TODO?: Figure out better place for these calls, but needed here due to
  // this function being called before other initialization points
  getOrCreateIdentTy(F.getParent());
  getOrCreateDefaultLocation(F.getParent());

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto NTFn = createRuntimeFunction(
      OpenMPRuntimeFunction::OMPRTL__kmpc_global_num_threads, F.getParent());
  Value *NWorkers = emitRuntimeCall(NTFn, {DefaultOpenMPLocation}, "", B);

  return NWorkers;
}

/// Lower a call to get the grainsize of this Tapir loop.
Value *llvm::OpenMPABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  Module *M = GrainsizeCall->getModule();
  IRBuilder<> Builder(GrainsizeCall);

  Value *Workers = GetOrCreateWorkerCount(*GrainsizeCall->getFunction());
  // num_threads returns 0 if not in parallel region, so need to add 1 to avoid
  // dividing by zero later in the case of fast-openmp
  // `nworkers += nworkers == 0`
  Type *Int32Ty = IntegerType::get(M->getContext(), 32);
  Value *EQCmp = Builder.CreateICmpEQ(Workers, ConstantInt::get(Int32Ty, 0));
  Value *IntEQCmp = Builder.CreateIntCast(EQCmp, Int32Ty, false);
  Value *PosWorkers = Builder.CreateAdd(Workers, IntEQCmp, "nworkers");

  Value *WorkersX8 = Builder.CreateIntCast(
      Builder.CreateMul(PosWorkers, ConstantInt::get(PosWorkers->getType(), 8)),
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
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void llvm::OpenMPABI::lowerSync(SyncInst &SI) {
  std::vector<Value *> Args = {DefaultOpenMPLocation,
                            getThreadID(SI.getParent()->getParent())};
  IRBuilder<> builder(&SI);
  emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_taskwait,
                            SI.getParent()->getParent()->getParent()),
    Args, "", builder);

  // Replace the detach with a branch to the continuation.
  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
}


typedef struct shar {
    int **pth_counter;
    int *pcounter;
    int *pj;
} *pshareds;

namespace llvm {
template<bool xcompile> class TypeBuilder<struct shar, xcompile> {
public:
  static StructType *get(LLVMContext &Context) {
    // If you cache this result, be sure to cache it separately
    // for each LLVMContext.
    return StructType::get(Context, ArrayRef<Type*>({
      TypeBuilder<int**, xcompile>::get(Context),
      TypeBuilder<int*, xcompile>::get(Context),
      TypeBuilder<int*, xcompile>::get(Context)}));
  }
  // You may find this a convenient place to put some constants
  // to help with getelementptr.  They don't have any effect on
  // the operation of TypeBuilder.
  enum Fields {
    pth_counter,
    pcounter,
    pj
  };
};
}

typedef struct task {
    pshareds shareds;
    int(* routine)(int,void*);
    int part_id;
// privates:
    unsigned long long lb; // library always uses ULONG
    unsigned long long ub;
    int st;
    int last;
    int i;
    int j;
    int th;
} *ptask, kmp_task_t;

namespace llvm {
template<bool xcompile> class TypeBuilder<struct task, xcompile> {
public:
  static StructType *get(LLVMContext &Context) {
    // If you cache this result, be sure to cache it separately
    // for each LLVMContext.
    return StructType::get(Context, ArrayRef<Type*>({
      TypeBuilder<pshareds, xcompile>::get(Context),
      TypeBuilder<int(*)(int,void*), xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context),
      TypeBuilder<unsigned long long, xcompile>::get(Context),
      TypeBuilder<unsigned long long, xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context),
      TypeBuilder<int, xcompile>::get(Context)}));
  }
  // You may find this a convenient place to put some constants
  // to help with getelementptr.  They don't have any effect on
  // the operation of TypeBuilder.
  enum Fields {
    shareds,
    routine,
    part_id,
    lb,
    ub,
    st,
    last,
    i,
    j,
    th,
  };
};
}

typedef llvm::TypeBuilder<ptask, false> ptask_builder;
typedef llvm::TypeBuilder<pshareds, false> pshareds_builder;

Function* formatFunctionToTask(Function* extracted, Instruction* CallSite) {
  // TODO: Fix this function to support call sites that are invokes instead of
  // calls.
  CallInst *cal = dyn_cast<CallInst>(CallSite);
  assert(cal && "Call instruction for task not found.");
  std::vector<Value*> LoadedCapturedArgs;
  for(auto& a:cal->arg_operands()) {
    LoadedCapturedArgs.push_back(a);
  }

  Module* M = extracted->getParent();
  LLVMContext &C = M->getContext();
  DataLayout DL(M);
  auto *SharedsTy = createSharedsTy(extracted);
  auto *SharedsPtrTy = PointerType::getUnqual(SharedsTy);
  IRBuilder<> CallerIRBuilder(cal);
  auto *SharedsTySize =
      CallerIRBuilder.getInt64(DL.getTypeAllocSize(SharedsTy));
  auto *KmpTaskTTy = createKmpTaskTTy(C);
  auto *KmpTaskTWithPrivatesTy = createKmpTaskTWithPrivatesTy(SharedsTy);//KmpTaskTTy);
  auto *KmpTaskTWithPrivatesPtrTy =
      PointerType::getUnqual(KmpTaskTWithPrivatesTy);
  auto *KmpTaskTWithPrivatesTySize =
      CallerIRBuilder.getInt64(DL.getTypeAllocSize(KmpTaskTWithPrivatesTy));

  auto *VoidTy = Type::getVoidTy(C);
  auto *Int8PtrTy = Type::getInt8PtrTy(C);
  auto *Int32Ty = Type::getInt32Ty(C);

  auto *CopyFnTy = FunctionType::get(VoidTy, {Int8PtrTy}, true);
  auto *CopyFnPtrTy = PointerType::getUnqual(CopyFnTy);

  auto *OutlinedFnTy = FunctionType::get(
      VoidTy,
      {Int32Ty, KmpTaskTWithPrivatesPtrTy},
      false);
  auto *OutlinedFn = Function::Create(
      OutlinedFnTy, GlobalValue::InternalLinkage, ".omp_outlined.", M);
  StringRef ArgNames[] = {".global_tid.", "ptask"};

  std::vector<Value*> out_args;
  for (auto &Arg : OutlinedFn->args()) {
    Arg.setName(ArgNames[out_args.size()]);
    out_args.push_back(&Arg);
  }

  OutlinedFn->setLinkage(GlobalValue::InternalLinkage);
  OutlinedFn->addFnAttr(Attribute::AlwaysInline);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::UWTable);

  auto *EntryBB = BasicBlock::Create(C, "entry", OutlinedFn, nullptr);
  IRBuilder<> IRBuilder(EntryBB);

  // Load the context struct so that we can access the task's accessed data
  auto *Context = IRBuilder.CreatePointerBitCastOrAddrSpaceCast(
    IRBuilder.CreateConstGEP2_32(cast<PointerType>(out_args[1]->getType()->getScalarType())->getElementType(), out_args[1], 0, 1), SharedsPtrTy);//.back();

  std::vector<Value *> ForkedFnArgs;


  	ValueToValueMapTy valmap;

    unsigned int argc = 0;
    for (auto& arg : extracted->args()) {
      auto *DataAddrEP = IRBuilder.CreateInBoundsGEP(
          Context, {IRBuilder.getInt32(0), IRBuilder.getInt32(argc)});
      auto *DataAddr = IRBuilder.CreateAlignedLoad(
          DataAddrEP,
          DL.getTypeAllocSize(DataAddrEP->getType()->getPointerElementType()));
      valmap.insert(std::pair<Value*,Value*>(&arg,DataAddr));
      argc++;
    }

  	SmallVector< ReturnInst *,5> retinsts;
    CloneFunctionInto(OutlinedFn, extracted, valmap, true, retinsts);
    IRBuilder.CreateBr(OutlinedFn->getBasicBlockList().getNextNode(*EntryBB));

  // We only need tied tasks for now and that's what the 1 value is for.
  auto *TaskFlags = CallerIRBuilder.getInt32(1);
  std::vector<Value *> AllocArgs = {
      DefaultOpenMPLocation,
      getThreadID(cal->getParent()->getParent()),
      TaskFlags,
      KmpTaskTWithPrivatesTySize,
      SharedsTySize,
      CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
          OutlinedFn, KmpRoutineEntryPtrTy)};
  auto *NewTask = emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_task_alloc,
                            M),
      AllocArgs, "", CallerIRBuilder);
  auto *NewTaskNewTaskTTy = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      NewTask, KmpTaskTWithPrivatesPtrTy);
  auto *AggCaptured = CallerIRBuilder.CreateInBoundsGEP(
      KmpTaskTWithPrivatesTy, NewTaskNewTaskTTy,
      {CallerIRBuilder.getInt32(0), CallerIRBuilder.getInt32(1)});

      // Store captured arguments into agg.captured
      for (unsigned i = 0; i < LoadedCapturedArgs.size(); ++i) {
        auto *AggCapturedElemPtr = CallerIRBuilder.CreateInBoundsGEP(
            SharedsTy, AggCaptured,
            {CallerIRBuilder.getInt32(0), CallerIRBuilder.getInt32(i)});
        CallerIRBuilder.CreateAlignedStore(
            LoadedCapturedArgs[i], AggCapturedElemPtr,
            DL.getTypeAllocSize(LoadedCapturedArgs[i]->getType()));
      }

  std::vector<Value *> TaskArgs =
    {DefaultOpenMPLocation, getThreadID(cal->getParent()->getParent()),
     NewTask};

  emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_task, M),
      TaskArgs, "", CallerIRBuilder);

  cal->eraseFromParent();
  extracted->eraseFromParent();
  return OutlinedFn;
}

void OpenMPABI::processOutlinedTask(Function &F) {}
void OpenMPABI::processSpawner(Function &F) {}

void OpenMPABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Function *Outline = TOI.Outline;
  Instruction *ReplCall = TOI.ReplCall;
  TOI.Outline = formatFunctionToTask(Outline, ReplCall);
}

void llvm::OpenMPABI::preProcessFunction(Function &F) {
  auto M = (Module *)F.getParent();
  getOrCreateIdentTy(M);
  getOrCreateDefaultLocation(M);
}

cl::opt<bool> fastOpenMP(
    "fast-openmp", cl::init(false), cl::Hidden,
    cl::desc("Attempt faster OpenMP implementation, "
             "assuming parallel outside set"));

void llvm::OpenMPABI::postProcessFunction(Function &F) {
  if (fastOpenMP) return;

  auto& Context = F.getContext();
  DataLayout DL(F.getParent());

  std::vector<CallInst*> tasks;
  for (BasicBlock& BB : F) {
    for(Instruction& I : BB) {
      if (CallInst* cal = dyn_cast_or_null<CallInst>(&I)){
        if (cal->getCalledFunction()->getName()=="__kmpc_omp_task_alloc") {
            tasks.push_back(cal);
        }
      }
    }
  }

  SmallVector<BasicBlock *, 32> Todo;
  SmallPtrSet<BasicBlock *, 4> Visited;
  SmallVector<BasicBlock*, 32>VisitedVec;
  for(auto a : tasks) {
    a->getParent()->splitBasicBlock(a);
    Todo.push_back(a->getParent());
  }

  while (Todo.size() > 0) {
    BasicBlock *BB = Todo.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    VisitedVec.push_back(BB);
    bool foundsync = false;
    for(Instruction& I : *BB) {
      if (CallInst* cal = dyn_cast_or_null<CallInst>(&I)){
        if (cal->getCalledFunction()->getName()=="__kmpc_omp_taskwait") {
            foundsync = true;
            BB->splitBasicBlock(++BasicBlock::iterator(cal));
            break;
        }
      }
    }

    if (!foundsync) {
      for (BasicBlock *Succ : successors(BB)) {
        Todo.push_back(Succ);
      }
    }
  }

  for(int i=1; i<VisitedVec.size(); i++) {
      for (auto P : predecessors(VisitedVec[i])) {
        if (Visited.count(P) == 0) {
          std::swap(VisitedVec[0], VisitedVec[i]);
          break;
        }
      }
  }

  CodeExtractor RegionExtractor(VisitedVec);
  Function *RegionFn = RegionExtractor.extractCodeRegion();

  std::vector<Type *> FnParams;
  std::vector<StringRef> FnArgNames;
  std::vector<AttributeSet> FnArgAttrs;

  auto *Int32PtrTy = PointerType::getUnqual(Type::getInt32Ty(Context));
  FnParams.push_back(Int32PtrTy);
  FnParams.push_back(Int32PtrTy);

  FnArgNames.push_back(".global_tid.");
  FnArgNames.push_back(".bound_tid.");

  //FnArgAttrs.push_back(AttributeSet::get(Context, 1, Attribute::NoAlias));
  //FnArgAttrs.push_back(AttributeSet::get(Context, 2, Attribute::NoAlias));

  int ArgOffset = 2;

  // For RegionFn argument add a corresponding argument to the new function.
  bool first = true;
  for (auto &Arg : RegionFn->args()) {
    if (first) {
      first = false;
      continue;
    }
    FnParams.push_back(Arg.getType());
    FnArgNames.push_back(Arg.getName());

    // Allow speculative loading from shared data.
    if (Arg.getType()->isPointerTy()) {
      AttrBuilder B;
      B.addDereferenceableAttr(
          DL.getTypeAllocSize(Arg.getType()->getPointerElementType()));
      //FnArgAttrs.push_back(AttributeSet::get(Context, ++ArgOffset, B));
    } else {
      //FnArgAttrs.push_back(AttributeSet());
      ++ArgOffset;
    }
  }

  // Create the function and set its argument properties.
  auto *VoidTy = Type::getVoidTy(Context);
    auto *OMPRegionFnTy = FunctionType::get(VoidTy, FnParams, false);
    auto Name = RegionFn->getName() + ".OMP";
    Function *OMPRegionFn = dyn_cast<Function>(
        F.getParent()->getOrInsertFunction(Name.str(), OMPRegionFnTy));

    // If this is an outermost region, skip the first 2 arguments (global_tid and
    // bound_tid) ...
    auto OMPArgIt = OMPRegionFn->arg_begin();
    Value* tmp = OMPArgIt;
    ++OMPArgIt;
    ++OMPArgIt;

    // ... then map corresponding arguments in RegionFn and OMPRegionFn

    auto *EntryBB = BasicBlock::Create(Context, "entry", OMPRegionFn, nullptr);
    IRBuilder<> IRBuilder0(EntryBB);
    tmp = IRBuilder0.CreateLoad(tmp);

    ValueToValueMapTy VMap;
    for (auto &Arg : RegionFn->args()) {
      if(tmp) {
        VMap[&Arg] = tmp;
        tmp = nullptr;
      } else {
        VMap[&Arg] = &*OMPArgIt;
        ++OMPArgIt;
      }
    }

  	SmallVector< ReturnInst *,5> retinsts;
    CloneFunctionInto(OMPRegionFn, RegionFn, VMap, false, retinsts);
    IRBuilder0.CreateBr(OMPRegionFn->getBasicBlockList().getNextNode(*EntryBB));

  auto FindCallToExtractedFn = [](Function *SpawningFn,
                                    Function *ExtractedFn) {
      // Find the call instruction to the extracted region function: RegionFn.
      CallInst *ExtractedFnCI = nullptr;
      for (auto &BB : *SpawningFn) {
        if (CallInst *CI = dyn_cast<CallInst>(BB.begin())) {
          // NOTE: I use pointer equality here, is that fine?
          if (ExtractedFn == CI->getCalledFunction()) {
            ExtractedFnCI = CI;
            break;
          }
        }
      }

      assert(ExtractedFnCI != nullptr &&
             "Couldn't find the call to the extracted region function!");

      return ExtractedFnCI;
    };
  auto *ExtractedFnCI = FindCallToExtractedFn(&F, RegionFn);

  IRBuilder<> b(ExtractedFnCI);

  auto *Int32Ty = Type::getInt32Ty(Context);
  std::vector<Value *> OMPRegionFnArgs = {
      DefaultOpenMPLocation,
      ConstantInt::getSigned(Int32Ty, RegionFn->arg_size()),
      b.CreateBitCast(OMPRegionFn, getKmpc_MicroPointerTy(Context))};

  std::vector<Value *> OMPNestedRegionFnArgs;
  auto ArgIt = ExtractedFnCI->arg_begin();
  ++ArgIt;
  // Append the rest of the region's arguments.
  while (ArgIt != ExtractedFnCI->arg_end()) {
    OMPRegionFnArgs.push_back(ArgIt->get());
    ++ArgIt;
  }

  auto ForkRTFn = createRuntimeFunction(
      OpenMPRuntimeFunction::OMPRTL__kmpc_fork_call, F.getParent());
  // Replace the old call with __kmpc_fork_call
  auto *ForkCall = emitRuntimeCall(ForkRTFn, OMPRegionFnArgs, "", b);
  ExtractedFnCI->eraseFromParent();
  RegionFn->eraseFromParent();
}

void llvm::OpenMPABI::postProcessHelper(Function &F) {}
