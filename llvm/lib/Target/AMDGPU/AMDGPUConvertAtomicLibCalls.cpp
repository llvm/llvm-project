//===- AMDGPUConvertAtomicLibCalls.cpp ------------===//
//
// Copyright(c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Convert atomic intrinsic calls to LLVM IR Instructions.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "amdloweratomic"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#include "AMDGPU.h"

using namespace llvm;

namespace {
const char MangledAtomicPrefix[] = "U7_Atomic";
};

// Check if a mangled type name is unsigned
static bool isMangledTypeUnsigned(char Mangled) {
  return Mangled == 'h'    /* uchar */
         || Mangled == 't' /* ushort */
         || Mangled == 'j' /* uint */
         || Mangled == 'm' /* ulong */;
}

// Check if a mangled function name contains unsigned atomic type
static bool containsUnsignedAtomicType(StringRef Name) {
  auto Loc = Name.find(MangledAtomicPrefix);
  if (Loc == StringRef::npos)
    return false;
  return isMangledTypeUnsigned(Name[Loc + strlen(MangledAtomicPrefix)]);
}

// Declarations for lowering OCL 1.x atomics
namespace AMDOCL1XAtomic {
llvm::cl::opt<unsigned> OCL1XAtomicOrder(
    "amd-ocl1x-atomic-order",
    llvm::cl::init(unsigned(AtomicOrdering::Monotonic)), llvm::cl::Hidden,
    llvm::cl::desc("AMD OCL 1.x atomic ordering for x86/x86-64"));

llvm::cl::opt<AMDGPUSynchronizationScope>
    OCL1XAtomicScope("amd-ocl1x-atomic-scope",
                     llvm::cl::init(AMDGPUSynchronizationScope::Agent),
                     llvm::cl::Hidden,
                     llvm::cl::desc("AMD OCL 1.x atomic scope for x86/x86-64"));

llvm::Value *LowerOCL1XAtomic(IRBuilder<> &llvmBuilder,
                             CallSite * CS);
}

// Pass for lowering OCL 2.0 atomics
namespace llvm {
class AMDGPUConvertAtomicLibCalls : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  AMDGPUConvertAtomicLibCalls() : ModulePass(ID) {
    initializeAMDGPUConvertAtomicLibCallsPass(*PassRegistry::getPassRegistry());
  }
  virtual bool runOnModule(Module &M);

private:
  Module *Mod;
  void setModule(Module *M);
  Value *lowerAtomic(StringRef Name, CallSite *CS);
  Value *lowerAtomicLoad(IRBuilder<> LlvmBuilder, CallSite *CS);
  Value *lowerAtomicStore(IRBuilder<> LlvmBuilder, StringRef Name,
                          CallSite *CS);
  Value *lowerAtomicCmpXchg(IRBuilder<> LlvmBuilder, CallSite *CS);
  Value *lowerAtomicRMW(IRBuilder<> LlvmBuilder, StringRef Name, CallSite *CS);
  Value *lowerAtomicInit(IRBuilder<> LlvmBuilder, CallSite *CS);
};
}

char AMDGPUConvertAtomicLibCalls::ID = 0;

INITIALIZE_PASS(
    AMDGPUConvertAtomicLibCalls, "amdgpu-lower-opencl-atomic-builtins",
    "Convert OpenCL atomic intrinsic calls into LLVM IR Instructions ", false,
    false);

char &llvm::AMDGPUConvertAtomicLibCallsID = AMDGPUConvertAtomicLibCalls::ID;

namespace llvm {
ModulePass *createAMDGPUConvertAtomicLibCallsPass() { return new AMDGPUConvertAtomicLibCalls(); }
}

void AMDGPUConvertAtomicLibCalls::setModule(Module *M) { Mod = M; }

static bool isOCLAtomicLoad(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      FuncName.find("atomic_load") == StringRef::npos)
    return false;
  return true;
}

static bool isOCLAtomicStore(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      FuncName.find("atomic_store") == StringRef::npos)
    return false;
  return true;
}

static bool isOCLAtomicCmpXchg(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      ((FuncName.find("atomic_compare_exchange_strong") == StringRef::npos) &&
       (FuncName.find("atomic_compare_exchange_weak") == StringRef::npos)))
    return false;
  return true;
}

static bool isOCLAtomicRMW(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      ((FuncName.find("atomic_fetch_add") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_sub") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_or") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_xor") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_and") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_min") == StringRef::npos) &&
       (FuncName.find("atomic_fetch_max") == StringRef::npos) &&
       (FuncName.find("atomic_exchange") == StringRef::npos)))
    return false;
  return true;
}

static bool isOCLAtomicTestAndSet(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      FuncName.find("atomic_flag_test_and_set") == StringRef::npos)
    return false;
  return true;
}

static bool isOCLAtomicFlagClear(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      FuncName.find("atomic_flag_clear") == StringRef::npos)
    return false;
  return true;
}

static bool isOCLAtomicInit(StringRef FuncName) {
  if (!FuncName.startswith("_Z") ||
      FuncName.find("atomic_init") == StringRef::npos)
    return false;
  return true;
}

static AtomicOrdering MemoryOrderSpir2LLVM(Value *SpirMemOrd) {
  enum memory_order {
    memory_order_relaxed = 0,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
  };
  unsigned MemOrd = dyn_cast<ConstantInt>(SpirMemOrd)->getZExtValue();
  switch (MemOrd) {
  case memory_order_relaxed:
    return AtomicOrdering::Monotonic;
  case memory_order_acquire:
    return AtomicOrdering::Acquire;
  case memory_order_release:
    return AtomicOrdering::Release;
  case memory_order_acq_rel:
    return AtomicOrdering::AcquireRelease;
  case memory_order_seq_cst:
    return AtomicOrdering::SequentiallyConsistent;
  default:
    return AtomicOrdering::NotAtomic;
  }
}

static AMDGPUSynchronizationScope
MemoryScopeOpenCL2LLVM(Value *OpenclMemScope) {
  enum memory_scope {
    memory_scope_work_item = 0,
    memory_scope_work_group,
    memory_scope_device,
    memory_scope_all_svm_devices,
    memory_scope_sub_group
  };
  unsigned MemScope = dyn_cast<ConstantInt>(OpenclMemScope)->getZExtValue();
  switch (MemScope) {
  case memory_scope_work_item:
    llvm_unreachable("memory_scope_work_item not Valid for atomic builtins");
  case memory_scope_work_group:
    return AMDGPUSynchronizationScope::WorkGroup;
  case memory_scope_device:
    return AMDGPUSynchronizationScope::Agent;
  case memory_scope_all_svm_devices:
    return AMDGPUSynchronizationScope::System;
  case memory_scope_sub_group:
    return AMDGPUSynchronizationScope::Wavefront;
  default:
    llvm_unreachable("unknown memory scope");
  }
}

static AMDGPUSynchronizationScope getDefaultMemScope(Value *Ptr) {
  unsigned AddrSpace = dyn_cast<PointerType>(Ptr->getType())->getAddressSpace();
  // for atomics on local pointers, memory scope is wg
  if (AddrSpace == 3)
    return AMDGPUSynchronizationScope::WorkGroup;
  return AMDGPUSynchronizationScope::Agent;
}

static AtomicOrdering getMemoryOrder(CallSite *CS, unsigned MemOrderPos) {
  return CS->getNumArgOperands() > MemOrderPos
    ? MemoryOrderSpir2LLVM(CS->getArgOperand(MemOrderPos))
             : AtomicOrdering::SequentiallyConsistent;
}

static AMDGPUSynchronizationScope getMemoryScope(CallSite *CS,
                                                 unsigned MemScopePos) {
  return CS->getNumArgOperands() > MemScopePos
    ? MemoryScopeOpenCL2LLVM(CS->getArgOperand(MemScopePos))
             : getDefaultMemScope(CS->getInstruction()->getOperand(0));
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomic(StringRef Name, CallSite *CS) {
  IRBuilder<> LlvmBuilder(Mod->getContext());
  LlvmBuilder.SetInsertPoint(CS->getInstruction());

  if (isOCLAtomicLoad(Name))
    return lowerAtomicLoad(LlvmBuilder, CS);
  if (isOCLAtomicStore(Name) || isOCLAtomicFlagClear(Name))
    return lowerAtomicStore(LlvmBuilder, Name, CS);
  if (isOCLAtomicCmpXchg(Name))
    return lowerAtomicCmpXchg(LlvmBuilder, CS);
  if (isOCLAtomicRMW(Name) || isOCLAtomicTestAndSet(Name))
    return lowerAtomicRMW(LlvmBuilder, Name, CS);
  if (isOCLAtomicInit(Name))
    return lowerAtomicInit(LlvmBuilder, CS);
  return AMDOCL1XAtomic::LowerOCL1XAtomic(LlvmBuilder, CS);
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicLoad(IRBuilder<> LlvmBuilder,
                                           CallSite *Inst) {
  Value *Ptr = Inst->getArgOperand(0);
  AtomicOrdering MemOrd = getMemoryOrder(Inst, 1);
  AMDGPUSynchronizationScope memScope = getMemoryScope(Inst, 2);
  Type *PtrType = Ptr->getType();
  Type *ValType = Ptr->getType()->getPointerElementType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned AddrSpace = dyn_cast<PointerType>(PtrType)->getAddressSpace();
    Type *IntType =
      IntegerType::get(Mod->getContext(), ValType->getPrimitiveSizeInBits());
    Type *IntPtrType = PointerType::get(IntType, AddrSpace);
    Ptr = LlvmBuilder.CreateCast(Instruction::BitCast, Ptr, IntPtrType);
  }
  Value *LdInst = LlvmBuilder.CreateLoad(Ptr, true);
  dyn_cast<LoadInst>(LdInst)->setOrdering(MemOrd);
  dyn_cast<LoadInst>(LdInst)->setSynchScope((SynchronizationScope)memScope);
  dyn_cast<LoadInst>(LdInst)->setAlignment(ValType->getPrimitiveSizeInBits() /
                                           8);
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    LdInst = LlvmBuilder.CreateCast(Instruction::BitCast, LdInst, ValType);
  }
  return LdInst;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicStore(IRBuilder<> LlvmBuilder,
                                            StringRef FuncName,
                                            CallSite *Inst) {
  bool isAtomicClear = FuncName.startswith("atomic_flag_clear") ? true : false;
  Value *Ptr = Inst->getArgOperand(0);
  Value *Val =
      isAtomicClear
          ? ConstantInt::get(IntegerType::get(Mod->getContext(), 32), 0)
          : Inst->getArgOperand(1);
  AtomicOrdering MemOrd =
      isAtomicClear ? getMemoryOrder(Inst, 1) : getMemoryOrder(Inst, 2);
  AMDGPUSynchronizationScope memScope =
      isAtomicClear ? getMemoryScope(Inst, 2) : getMemoryScope(Inst, 3);
  Type *ValType = Val->getType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned addrSpace =
        dyn_cast<PointerType>(Ptr->getType())->getAddressSpace();
    Type *IntType =
        IntegerType::get(Mod->getContext(), ValType->getPrimitiveSizeInBits());
    Type *IntPtrType = PointerType::get(IntType, addrSpace);
    Ptr = LlvmBuilder.CreateCast(Instruction::BitCast, Ptr, IntPtrType);
    Val = LlvmBuilder.CreateCast(Instruction::BitCast, Val, IntType);
  }
  Value *StInst = LlvmBuilder.CreateStore(Val, Ptr, true);
  dyn_cast<StoreInst>(StInst)->setOrdering(MemOrd);
  dyn_cast<StoreInst>(StInst)->setSynchScope((SynchronizationScope)memScope);
  dyn_cast<StoreInst>(StInst)->setAlignment(ValType->getPrimitiveSizeInBits() /
                                            8);
  return StInst;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicCmpXchg(IRBuilder<> llvmBuilder,
                                              CallSite *Inst) {
  LLVMContext &llvmContext = Mod->getContext();
  Value *Ptr = Inst->getArgOperand(0);
  Value *Expected = Inst->getArgOperand(1);
  Value *Desired = Inst->getArgOperand(2);
  AtomicOrdering MemOrdSuccess = getMemoryOrder(Inst, 3);
  AtomicOrdering MemOrdFailure = getMemoryOrder(Inst, 4);
  AMDGPUSynchronizationScope memScope = getMemoryScope(Inst, 5);
  Value *OrigExpected = llvmBuilder.CreateLoad(Expected, true);
  Value *Cas = llvmBuilder.CreateAtomicCmpXchg(
      Ptr, OrigExpected, Desired, MemOrdSuccess, MemOrdFailure,
      (SynchronizationScope)AMDGPUSynchronizationScope::System);
  dyn_cast<AtomicCmpXchgInst>(Cas)->setSynchScope(
      (SynchronizationScope)memScope);
  Cas = llvmBuilder.CreateExtractValue(Cas, 0);
  Type *ValType = Expected->getType();
  Value *Cmp = NULL;
  if (ValType->isFloatTy() || ValType->isDoubleTy())
    Cmp = llvmBuilder.CreateFCmp(FCmpInst::FCMP_OEQ, Cas, OrigExpected);
  else
    Cmp = llvmBuilder.CreateICmp(ICmpInst::ICMP_EQ, Cas, OrigExpected);
  Cmp = llvmBuilder.CreateCast(Instruction::BitCast, Cmp,
                               IntegerType::get(llvmContext, 1));
  Value *Select = llvmBuilder.CreateSelect(
      Cmp, ConstantInt::get(IntegerType::get(llvmContext, 1), 1),
      ConstantInt::get(IntegerType::get(llvmContext, 1), 0));
  return Select;
}

static AtomicRMWInst::BinOp atomicFetchBinOp(StringRef Name, bool IsSigned) {
  const char *FuncName = Name.data();
  if (strstr(FuncName, "atomic_fetch_add"))
    return AtomicRMWInst::Add;
  if (strstr(FuncName, "atomic_fetch_sub"))
    return AtomicRMWInst::Sub;
  if (strstr(FuncName, "atomic_fetch_and"))
    return AtomicRMWInst::And;
  if (strstr(FuncName, "atomic_fetch_or"))
    return AtomicRMWInst::Or;
  if (strstr(FuncName, "atomic_fetch_xor"))
    return AtomicRMWInst::Xor;
  if (strstr(FuncName, "atomic_fetch_max"))
    return IsSigned ? AtomicRMWInst::Max : AtomicRMWInst::UMax;
  if (strstr(FuncName, "atomic_fetch_min"))
    return IsSigned ? AtomicRMWInst::Min : AtomicRMWInst::UMin;
  if (strstr(FuncName, "atomic_exchange") ||
      strstr(FuncName, "atomic_flag_test_and_set"))
    return AtomicRMWInst::Xchg;
  assert(0 && "internal error");
  return AtomicRMWInst::BAD_BINOP;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicRMW(IRBuilder<> LlvmBuilder,
                                          StringRef FuncName, CallSite *Inst) {
  LLVMContext &llvmContext = Mod->getContext();
  bool TestAndSet =
    strstr(FuncName.data(), "atomic_flag_test_and_set") ? true : false;
  Value *Ptr = Inst->getArgOperand(0);
  Value *Val = TestAndSet
                   ? ConstantInt::get(IntegerType::get(llvmContext, 32), 1)
                   : Inst->getArgOperand(1);
  AtomicOrdering MemOrd =
    TestAndSet ? getMemoryOrder(Inst, 1) : getMemoryOrder(Inst, 2);
  AMDGPUSynchronizationScope memScope =
    TestAndSet ? getMemoryScope(Inst, 2) : getMemoryScope(Inst, 3);
  Type *PtrType = Ptr->getType();
  Type *ValType = Val->getType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned addrSpace = dyn_cast<PointerType>(PtrType)->getAddressSpace();
    Type *IntType =
        IntegerType::get(llvmContext, ValType->getPrimitiveSizeInBits());
    Type *IntPtrType = PointerType::get(IntType, addrSpace);
    Ptr = LlvmBuilder.CreateCast(Instruction::BitCast, Ptr, IntPtrType);
    Val = LlvmBuilder.CreateCast(Instruction::BitCast, Val, IntType);
  }
  AtomicRMWInst::BinOp BinOp =
    atomicFetchBinOp(FuncName, !containsUnsignedAtomicType(FuncName));
  Value *AtomicRMW = LlvmBuilder.CreateAtomicRMW(
      BinOp, Ptr, Val, MemOrd,
      (SynchronizationScope)AMDGPUSynchronizationScope::System);
  dyn_cast<AtomicRMWInst>(AtomicRMW)->setSynchScope(
      (SynchronizationScope)memScope);
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    AtomicRMW =
      LlvmBuilder.CreateCast(Instruction::BitCast, AtomicRMW, ValType);
  }
  if (TestAndSet) {
    AtomicRMW = LlvmBuilder.CreateCast(Instruction::Trunc, AtomicRMW,
                                       IntegerType::get(llvmContext, 1));
  }
  return AtomicRMW;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicInit(IRBuilder<> llvmBuilder,
                                           CallSite *Inst) {
  Value *Ptr = Inst->getArgOperand(0);
  Value *Val = Inst->getArgOperand(1);
  Value *StInst = llvmBuilder.CreateStore(Val, Ptr, true);
  return StInst;
}

const Function *getCallee(CallSite &CS) {
  const Function *Callee = CS.getCalledFunction();
  if (!Callee) { // function call via the pointer
    const Value *Val = CS.getCalledValue();
    if (const Constant *C = dyn_cast<Constant>(Val)) {
      Callee = dyn_cast_or_null<Function>(C->getOperand(0));
    }
  }
  return Callee;
}

bool AMDGPUConvertAtomicLibCalls::runOnModule(Module &M) {

  setModule(&M);
  bool Changed = false;
  for (Module::iterator MF = M.begin(), E = M.end(); MF != E; ++MF) {
    for (Function::iterator BB = MF->begin(), MFE = MF->end(); BB != MFE;
         ++BB) {
      for (BasicBlock::iterator Instr = BB->begin(), Instr_end = BB->end();
           Instr != Instr_end;) {
        CallSite CS(&*Instr);
        Instr++;
        if (!CS)
          continue;
        const Function *Callee = getCallee(CS);
        if (!Callee || !Callee->hasName())
          continue;
        Value *newAtomicInstr = lowerAtomic(Callee->getName(), &CS);
        if (newAtomicInstr) {
          CS.getInstruction()->replaceAllUsesWith(newAtomicInstr);
          CS->eraseFromParent();
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

// Functions for lowering OCL 1.x atomics
namespace AMDOCL1XAtomic {
using namespace llvm;

enum InstType { RMW, CMPXCHG, BAD };
struct Entry {
  const char *Name;
  InstType Type;
  AtomicRMWInst::BinOp Op;
  unsigned Nop;
};

static const Entry Table[] = {{"add", RMW, AtomicRMWInst::Add, 2},
                        {"sub", RMW, AtomicRMWInst::Sub, 2},
                        {"xchg", RMW, AtomicRMWInst::Xchg, 2},
                        {"inc", RMW, AtomicRMWInst::Add, 1},
                        {"dec", RMW, AtomicRMWInst::Sub, 1},
                        {"min", RMW, AtomicRMWInst::Min, 2},
                        {"max", RMW, AtomicRMWInst::Max, 2},
                        {"min", RMW, AtomicRMWInst::UMin, 2},
                        {"max", RMW, AtomicRMWInst::UMax, 2},
                        {"and", RMW, AtomicRMWInst::And, 2},
                        {"or", RMW, AtomicRMWInst::Or, 2},
                        {"xor", RMW, AtomicRMWInst::Xor, 2},
                        {"cmpxchg", CMPXCHG, AtomicRMWInst::BAD_BINOP, 3}};

bool ParseOCL1XAtomic(StringRef N, InstType &TP, AtomicRMWInst::BinOp &OP,
                      unsigned &NOP) {
  size_t Pos = N.find("atomic_"); // 32bit
  size_t Len = strlen("atomic_");
  if (Pos == StringRef::npos) {
    Pos = N.find("atom_"); // 64bit
    Len = strlen("atom_");
  }
  if (Pos != StringRef::npos) {
    StringRef Needle = N.substr(Pos + Len, N.size());
    int I;
    int N = array_lengthof(Table);
    for (I = 0; I < N; ++I) {
      if (Needle.startswith(Table[I].Name)) {
        break;
      }
    }

    if (I == N) {
      return false; // we have a user-defined function that has 'atomic_' in it's name
    } else {
      OP = Table[I].Op;
      TP = Table[I].Type;
      NOP = Table[I].Nop;
      return true;
    }
  }
  return false;
}

Value *LowerOCL1XAtomic(IRBuilder<> &Builder, CallSite * CS) {

  const Function *F = getCallee(*CS);
  if (!F || !F->hasName() || !F->getName().startswith("_Z")) {
    return NULL;
  }

  InstType Type = BAD;
  AtomicRMWInst::BinOp Op = AtomicRMWInst::BAD_BINOP;
  unsigned NumOp = 0;

  if (!ParseOCL1XAtomic(F->getName(), Type, Op, NumOp))
    return nullptr;

  assert(CS->arg_size() == NumOp && "Incorrect number of arguments");

  llvm::Value *P = CS->getArgument(0);
  llvm::Value *NI = NULL;
  AtomicOrdering Order = AtomicOrdering(OCL1XAtomicOrder.getValue());
  if (Type == RMW) {
    llvm::Value *V =
      NumOp == 2 ? CS->getArgument(1)
                   : ConstantInt::get(P->getType()->getPointerElementType(), 1);
    bool NeedCast = !V->getType()->isIntegerTy();
    if (NeedCast) {
      assert(Op == AtomicRMWInst::Xchg && "InValid atomic Instruction");
      LLVMContext &context = CS->getParent()->getContext();
      V = Builder.CreateBitCast(V, Type::getInt32Ty(context));
      P = Builder.CreateBitCast(
          P,
          Type::getInt32PtrTy(context, P->getType()->getPointerAddressSpace()));
    }
    NI = Builder.CreateAtomicRMW(
        Op, P, V, Order,
        (SynchronizationScope)AMDGPUSynchronizationScope::System);
    dyn_cast<AtomicRMWInst>(NI)->setSynchScope(
        (SynchronizationScope)OCL1XAtomicScope.getValue());
    if (NeedCast) {
      NI = Builder.CreateBitCast(NI, F->getReturnType());
    }
  } else if (Type == CMPXCHG) {
    NI = Builder.CreateAtomicCmpXchg(
      P, CS->getArgument(1), CS->getArgument(2), Order, Order,
        (SynchronizationScope)
            AMDGPUSynchronizationScope::System); // TBD Valery - what is
                                                 // FailureOrdering?
    dyn_cast<AtomicCmpXchgInst>(NI)->setSynchScope(
        (SynchronizationScope)OCL1XAtomicScope.getValue());
    NI = Builder.CreateExtractValue(NI, 0);
  } else {
    llvm_unreachable("InValid atomic builtin");
  }

  DEBUG(dbgs() << *F << " => " << *NI << '\n');
  return NI;
}
}
