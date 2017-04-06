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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Demangle/Demangle.h"
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
} // anonymous namespace

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

Value *LowerOCL1XAtomic(IRBuilder<> &llvmBuilder, const CallSite &CS,
                        const Function &Callee, StringRef DemangledName,
                        bool Unsigned);
}

// Pass for lowering OCL 2.0 atomics
namespace llvm {
class AMDGPUConvertAtomicLibCalls : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  AMDGPUConvertAtomicLibCalls() : ModulePass(ID), Context(nullptr) {
    initializeAMDGPUConvertAtomicLibCallsPass(*PassRegistry::getPassRegistry());
  }
  virtual bool runOnModule(Module &M);

private:
  LLVMContext *Context;
  Value *lowerAtomic(const CallSite &CS);
  Value *lowerAtomicLoad(IRBuilder<> &LlvmBuilder, const CallSite &CS);
  Value *lowerAtomicStore(IRBuilder<> &LlvmBuilder, StringRef Name,
                          const CallSite &CS);
  Value *lowerAtomicCmpXchg(IRBuilder<> &LlvmBuilder, const CallSite &CS,
                            bool isWeak);
  Value *lowerAtomicRMW(IRBuilder<> &LlvmBuilder, StringRef Name,
                        StringRef DemangledName, bool TestAndSet,
                        const CallSite &CS);
  Value *lowerAtomicInit(IRBuilder<> &LlvmBuilder, const CallSite &CS);
};
}

char AMDGPUConvertAtomicLibCalls::ID = 0;

INITIALIZE_PASS(
    AMDGPUConvertAtomicLibCalls, "amdgpu-lower-opencl-atomic-builtins",
    "Convert OpenCL atomic intrinsic calls into LLVM IR Instructions ", false,
    false)

char &llvm::AMDGPUConvertAtomicLibCallsID = AMDGPUConvertAtomicLibCalls::ID;

namespace llvm {
ModulePass *createAMDGPUConvertAtomicLibCallsPass() {
  return new AMDGPUConvertAtomicLibCalls();
}
}

static bool isOCLAtomicLoad(StringRef FuncName) {
  return FuncName.startswith("atomic_load");
}

static bool isOCLAtomicStore(StringRef FuncName) {
  return FuncName.startswith("atomic_store");
}

static bool isOCLAtomicCmpXchgStrong(StringRef FuncName) {
  return FuncName.startswith("atomic_compare_exchange_strong");
}

static bool isOCLAtomicCmpXchgWeak(StringRef FuncName) {
  return FuncName.startswith("atomic_compare_exchange_weak");
}

static bool isOCLAtomicRMW(StringRef FuncName) {
  return StringSwitch<bool>(FuncName)
      .StartsWith("atomic_fetch_add", true)
      .StartsWith("atomic_fetch_sub", true)
      .StartsWith("atomic_fetch_or", true)
      .StartsWith("atomic_fetch_xor", true)
      .StartsWith("atomic_fetch_and", true)
      .StartsWith("atomic_fetch_min", true)
      .StartsWith("atomic_fetch_max", true)
      .StartsWith("atomic_exchange", true)
      .Default(false);
}

static bool isOCLAtomicTestAndSet(StringRef FuncName) {
  return FuncName.startswith("atomic_flag_test_and_set");
}

static bool isOCLAtomicFlagClear(StringRef FuncName) {
  return FuncName.startswith("atomic_flag_clear");
}

static bool isOCLAtomicInit(StringRef FuncName) {
  return FuncName.startswith("atomic_init");
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

static AtomicOrdering getMemoryOrder(const CallSite &CS, unsigned MemOrderPos) {
  return CS.getNumArgOperands() > MemOrderPos
             ? MemoryOrderSpir2LLVM(CS.getArgOperand(MemOrderPos))
             : AtomicOrdering::SequentiallyConsistent;
}

static AMDGPUSynchronizationScope getMemoryScope(const CallSite &CS,
                                                 unsigned MemScopePos) {
  return CS.getNumArgOperands() > MemScopePos
             ? MemoryScopeOpenCL2LLVM(CS.getArgOperand(MemScopePos))
             : getDefaultMemScope(CS.getInstruction()->getOperand(0));
}

static const Function *getCallee(const CallSite &CS) {
  const Function *Callee = CS.getCalledFunction();
  if (!Callee) { // function call via the pointer
    const Value *Val = CS.getCalledValue();
    if (const Constant *C = dyn_cast<Constant>(Val)) {
      Callee = dyn_cast_or_null<Function>(C->getOperand(0));
    }
  }
  return Callee;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomic(const CallSite &CS) {
  const Function *Callee = getCallee(CS);
  if (!Callee || !Callee->hasName())
    return nullptr;

  StringRef Name = Callee->getName();
  IRBuilder<> LlvmBuilder(*Context);
  LlvmBuilder.SetInsertPoint(CS.getInstruction());
  std::unique_ptr<char, llvm::FreeDeleter> Buf(
    itaniumDemangle(Name.str().c_str(), nullptr, nullptr, nullptr));
  StringRef DemangledName(Buf.get());
  if (DemangledName.empty())
    return nullptr;

  if (isOCLAtomicLoad(DemangledName))
    return lowerAtomicLoad(LlvmBuilder, CS);
  if (isOCLAtomicStore(DemangledName) || isOCLAtomicFlagClear(DemangledName))
    return lowerAtomicStore(LlvmBuilder, DemangledName, CS);
  if (isOCLAtomicCmpXchgStrong(DemangledName))
    return lowerAtomicCmpXchg(LlvmBuilder, CS, false);
  if (isOCLAtomicCmpXchgWeak(DemangledName))
    return lowerAtomicCmpXchg(LlvmBuilder, CS, true);
  const bool TestAndSet = isOCLAtomicTestAndSet(DemangledName);
  if (isOCLAtomicRMW(DemangledName) || TestAndSet)
    return lowerAtomicRMW(LlvmBuilder, Name, DemangledName, TestAndSet, CS);
  if (isOCLAtomicInit(DemangledName))
    return lowerAtomicInit(LlvmBuilder, CS);
  return AMDOCL1XAtomic::LowerOCL1XAtomic(
      LlvmBuilder, CS, *Callee, DemangledName,
      isMangledTypeUnsigned(Name[Name.size() - 1]));
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicLoad(IRBuilder<> &LlvmBuilder,
                                                    const CallSite &Inst) {
  Value *Ptr = Inst.getArgOperand(0);
  AtomicOrdering MemOrd = getMemoryOrder(Inst, 1);
  AMDGPUSynchronizationScope memScope = getMemoryScope(Inst, 2);
  Type *PtrType = Ptr->getType();
  Type *ValType = Ptr->getType()->getPointerElementType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned AddrSpace = dyn_cast<PointerType>(PtrType)->getAddressSpace();
    Type *IntType =
        IntegerType::get(*Context, ValType->getPrimitiveSizeInBits());
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

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicStore(IRBuilder<> &LlvmBuilder,
                                                     StringRef FuncName,
                                                     const CallSite &Inst) {
  const bool isAtomicClear = isOCLAtomicFlagClear(FuncName);
  Value *Ptr = Inst.getArgOperand(0);
  Value *Val = isAtomicClear
                   ? ConstantInt::get(IntegerType::get(*Context, 32), 0)
                   : Inst.getArgOperand(1);
  AtomicOrdering MemOrd =
      isAtomicClear ? getMemoryOrder(Inst, 1) : getMemoryOrder(Inst, 2);
  AMDGPUSynchronizationScope memScope =
      isAtomicClear ? getMemoryScope(Inst, 2) : getMemoryScope(Inst, 3);
  Type *ValType = Val->getType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned addrSpace =
        dyn_cast<PointerType>(Ptr->getType())->getAddressSpace();
    Type *IntType =
        IntegerType::get(*Context, ValType->getPrimitiveSizeInBits());
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

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicCmpXchg(IRBuilder<> &llvmBuilder,
                                                       const CallSite &Inst,
                                                       bool isWeak) {
  Value *Ptr = Inst.getArgOperand(0);
  Value *Expected = Inst.getArgOperand(1);
  Value *Desired = Inst.getArgOperand(2);
  AtomicOrdering MemOrdSuccess = getMemoryOrder(Inst, 3);
  AtomicOrdering MemOrdFailure = getMemoryOrder(Inst, 4);
  AMDGPUSynchronizationScope memScope = getMemoryScope(Inst, 5);
  Value *OrigExpected = llvmBuilder.CreateLoad(Expected, true);
  Value *Cas = llvmBuilder.CreateAtomicCmpXchg(
      Ptr, OrigExpected, Desired, MemOrdSuccess, MemOrdFailure,
      (SynchronizationScope)AMDGPUSynchronizationScope::System);
  dyn_cast<AtomicCmpXchgInst>(Cas)->setSynchScope(
      (SynchronizationScope)memScope);
  dyn_cast<AtomicCmpXchgInst>(Cas)->setVolatile(true);
  dyn_cast<AtomicCmpXchgInst>(Cas)->setWeak(isWeak);

  Value *Cas0 = llvmBuilder.CreateExtractValue(Cas, 0);
  Value *Cas1 = llvmBuilder.CreateExtractValue(Cas, 1);
  Value *Select = llvmBuilder.CreateSelect(Cas1, OrigExpected, Cas0);
  llvmBuilder.CreateStore(Select, Expected);

  return Cas1;
}

static AtomicRMWInst::BinOp atomicFetchBinOp(StringRef Name, bool IsSigned,
                                             bool TestAndSet) {
  if (Name.startswith("atomic_fetch_add"))
    return AtomicRMWInst::Add;
  if (Name.startswith("atomic_fetch_sub"))
    return AtomicRMWInst::Sub;
  if (Name.startswith("atomic_fetch_and"))
    return AtomicRMWInst::And;
  if (Name.startswith("atomic_fetch_or"))
    return AtomicRMWInst::Or;
  if (Name.startswith("atomic_fetch_xor"))
    return AtomicRMWInst::Xor;
  if (Name.startswith("atomic_fetch_max"))
    return IsSigned ? AtomicRMWInst::Max : AtomicRMWInst::UMax;
  if (Name.startswith("atomic_fetch_min"))
    return IsSigned ? AtomicRMWInst::Min : AtomicRMWInst::UMin;
  if (Name.startswith("atomic_exchange") || TestAndSet)
    return AtomicRMWInst::Xchg;
  return AtomicRMWInst::BAD_BINOP;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicRMW(IRBuilder<> &LlvmBuilder,
                                                   StringRef FuncName,
                                                   StringRef DemangledName,
                                                   bool TestAndSet,
                                                   const CallSite &Inst) {
  Value *Ptr = Inst.getArgOperand(0);
  Value *Val = TestAndSet ? ConstantInt::get(IntegerType::get(*Context, 32), 1)
                          : Inst.getArgOperand(1);
  AtomicOrdering MemOrd =
      TestAndSet ? getMemoryOrder(Inst, 1) : getMemoryOrder(Inst, 2);
  AMDGPUSynchronizationScope memScope =
      TestAndSet ? getMemoryScope(Inst, 2) : getMemoryScope(Inst, 3);
  Type *PtrType = Ptr->getType();
  Type *ValType = Val->getType();
  if (ValType->isFloatTy() || ValType->isDoubleTy()) {
    unsigned addrSpace = dyn_cast<PointerType>(PtrType)->getAddressSpace();
    Type *IntType =
        IntegerType::get(*Context, ValType->getPrimitiveSizeInBits());
    Type *IntPtrType = PointerType::get(IntType, addrSpace);
    Ptr = LlvmBuilder.CreateCast(Instruction::BitCast, Ptr, IntPtrType);
    Val = LlvmBuilder.CreateCast(Instruction::BitCast, Val, IntType);
  }
  const AtomicRMWInst::BinOp BinOp = atomicFetchBinOp(
      DemangledName, !containsUnsignedAtomicType(FuncName), TestAndSet);
  assert(AtomicRMWInst::BAD_BINOP != BinOp);
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
                                       IntegerType::get(*Context, 1));
  }
  return AtomicRMW;
}

Value *AMDGPUConvertAtomicLibCalls::lowerAtomicInit(IRBuilder<> &llvmBuilder,
                                                    const CallSite &Inst) {
  Value *Ptr = Inst.getArgOperand(0);
  Value *Val = Inst.getArgOperand(1);
  Value *StInst = llvmBuilder.CreateStore(Val, Ptr, true);
  return StInst;
}

bool AMDGPUConvertAtomicLibCalls::runOnModule(Module &M) {

  Context = &M.getContext();
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
        Value *newAtomicInstr = lowerAtomic(CS);
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
  StringRef Name;
  InstType Type;
  AtomicRMWInst::BinOp Op;
  unsigned Nop;
};

static const Entry Table[] = {
    {"add", RMW, AtomicRMWInst::Add, 2},
    {"sub", RMW, AtomicRMWInst::Sub, 2},
    {"xchg", RMW, AtomicRMWInst::Xchg, 2},
    {"inc", RMW, AtomicRMWInst::Add, 1},
    {"dec", RMW, AtomicRMWInst::Sub, 1},
    {"min", RMW, AtomicRMWInst::Min, 2},
    {"max", RMW, AtomicRMWInst::Max, 2},
    {"and", RMW, AtomicRMWInst::And, 2},
    {"or", RMW, AtomicRMWInst::Or, 2},
    {"xor", RMW, AtomicRMWInst::Xor, 2},
    {"cmpxchg", CMPXCHG, AtomicRMWInst::BAD_BINOP, 3}};

bool ParseOCL1XAtomic(StringRef Name, InstType &TP, AtomicRMWInst::BinOp &OP,
                      unsigned &NOP, bool Unsigned) {
  const size_t Len = Name.startswith("atomic_")
                         ? strlen("atomic_")
                         : (Name.startswith("atom_") ? strlen("atom_") : 0);
  if (Len) {
    const StringRef Needle = Name.slice(Len, Name.find('('));
    size_t I;
    const size_t Num = array_lengthof(Table);
    for (I = 0; I < Num; ++I)
      if (Needle == Table[I].Name)
        break;

    if (I == Num) {
      return false; // we have a user-defined function that has 'atomic_' in
                    // it's name
    } else {
      OP = Table[I].Op;
      TP = Table[I].Type;
      NOP = Table[I].Nop;
      // Need to determine min/max or umin/umax
      if ((OP == AtomicRMWInst::Min || OP == AtomicRMWInst::Max) && Unsigned)
        OP = (OP == AtomicRMWInst::Min) ? AtomicRMWInst::UMin
                                        : AtomicRMWInst::UMax;
      return true;
    }
  }
  return false;
}

Value *LowerOCL1XAtomic(IRBuilder<> &Builder, const CallSite &CS,
                        const Function &Callee, StringRef DemangledName,
                        bool Unsigned) {

  InstType Type = BAD;
  AtomicRMWInst::BinOp Op = AtomicRMWInst::BAD_BINOP;
  unsigned NumOp = 0;

  if (!ParseOCL1XAtomic(DemangledName, Type, Op, NumOp, Unsigned))
    return nullptr;

  assert(CS.arg_size() == NumOp && "Incorrect number of arguments");

  Value *P = CS.getArgument(0);
  Value *NI = nullptr;
  AtomicOrdering Order = AtomicOrdering(OCL1XAtomicOrder.getValue());
  if (Type == RMW) {
    Value *V = NumOp == 2
                   ? CS.getArgument(1)
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
      NI = Builder.CreateBitCast(NI, Callee.getReturnType());
    }
  } else if (Type == CMPXCHG) {
    NI = Builder.CreateAtomicCmpXchg(
        P, CS.getArgument(1), CS.getArgument(2), Order, Order,
        (SynchronizationScope)
            AMDGPUSynchronizationScope::System); // TBD Valery - what is
                                                 // FailureOrdering?
    dyn_cast<AtomicCmpXchgInst>(NI)->setSynchScope(
        (SynchronizationScope)OCL1XAtomicScope.getValue());
    NI = Builder.CreateExtractValue(NI, 0);
  } else {
    llvm_unreachable("InValid atomic builtin");
  }

  DEBUG(dbgs() << Callee << " => " << *NI << '\n');
  return NI;
}
}
