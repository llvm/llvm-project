//===-- ExecutionEngine.cpp - Common Implementation shared by EEs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the common interface used by the various execution engine
// subclasses.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "llvm/ExecutionEngine/ExecutionEngine.h"

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include <cmath>
#include <cstring>
using namespace llvm;

STATISTIC(NumInitBytes, "Number of bytes of global vars initialized");
STATISTIC(NumGlobals  , "Number of global vars initialized");

ExecutionEngine *(*ExecutionEngine::JITCtor)(
  Module *M,
  std::string *ErrorStr,
  JITMemoryManager *JMM,
  bool GVsWithCode,
  TargetMachine *TM) = 0;
ExecutionEngine *(*ExecutionEngine::MCJITCtor)(
  Module *M,
  std::string *ErrorStr,
  JITMemoryManager *JMM,
  bool GVsWithCode,
  TargetMachine *TM) = 0;
ExecutionEngine *(*ExecutionEngine::InterpCtor)(Module *M,
                                                std::string *ErrorStr) = 0;

ExecutionEngine::ExecutionEngine(Module *M)
  : EEState(*this),
    LazyFunctionCreator(0),
    ExceptionTableRegister(0),
    ExceptionTableDeregister(0) {
  CompilingLazily         = false;
  GVCompilationDisabled   = false;
  SymbolSearchingDisabled = false;
  Modules.push_back(M);
  assert(M && "Module is null?");
}

ExecutionEngine::~ExecutionEngine() {
  clearAllGlobalMappings();
  for (unsigned i = 0, e = Modules.size(); i != e; ++i)
    delete Modules[i];
}

void ExecutionEngine::DeregisterAllTables() {
  if (ExceptionTableDeregister) {
    DenseMap<const Function*, void*>::iterator it = AllExceptionTables.begin();
    DenseMap<const Function*, void*>::iterator ite = AllExceptionTables.end();
    for (; it != ite; ++it)
      ExceptionTableDeregister(it->second);
    AllExceptionTables.clear();
  }
}

namespace {
/// \brief Helper class which uses a value handler to automatically deletes the
/// memory block when the GlobalVariable is destroyed.
class GVMemoryBlock : public CallbackVH {
  GVMemoryBlock(const GlobalVariable *GV)
    : CallbackVH(const_cast<GlobalVariable*>(GV)) {}

public:
  /// \brief Returns the address the GlobalVariable should be written into.  The
  /// GVMemoryBlock object prefixes that.
  static char *Create(const GlobalVariable *GV, const DataLayout& TD) {
    Type *ElTy = GV->getType()->getElementType();
    size_t GVSize = (size_t)TD.getTypeAllocSize(ElTy);
    void *RawMemory = ::operator new(
      DataLayout::RoundUpAlignment(sizeof(GVMemoryBlock),
                                   TD.getPreferredAlignment(GV))
      + GVSize);
    new(RawMemory) GVMemoryBlock(GV);
    return static_cast<char*>(RawMemory) + sizeof(GVMemoryBlock);
  }

  virtual void deleted() {
    // We allocated with operator new and with some extra memory hanging off the
    // end, so don't just delete this.  I'm not sure if this is actually
    // required.
    this->~GVMemoryBlock();
    ::operator delete(this);
  }
};
}  // anonymous namespace

char *ExecutionEngine::getMemoryForGV(const GlobalVariable *GV) {
  return GVMemoryBlock::Create(GV, *getDataLayout());
}

bool ExecutionEngine::removeModule(Module *M) {
  for(SmallVector<Module *, 1>::iterator I = Modules.begin(),
        E = Modules.end(); I != E; ++I) {
    Module *Found = *I;
    if (Found == M) {
      Modules.erase(I);
      clearGlobalMappingsFromModule(M);
      return true;
    }
  }
  return false;
}

Function *ExecutionEngine::FindFunctionNamed(const char *FnName) {
  for (unsigned i = 0, e = Modules.size(); i != e; ++i) {
    if (Function *F = Modules[i]->getFunction(FnName))
      return F;
  }
  return 0;
}


void *ExecutionEngineState::RemoveMapping(const MutexGuard &,
                                          const GlobalValue *ToUnmap) {
  GlobalAddressMapTy::iterator I = GlobalAddressMap.find(ToUnmap);
  void *OldVal;

  // FIXME: This is silly, we shouldn't end up with a mapping -> 0 in the
  // GlobalAddressMap.
  if (I == GlobalAddressMap.end())
    OldVal = 0;
  else {
    OldVal = I->second;
    GlobalAddressMap.erase(I);
  }

  GlobalAddressReverseMap.erase(OldVal);
  return OldVal;
}

void ExecutionEngine::addGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);

  DEBUG(dbgs() << "JIT: Map \'" << GV->getName()
        << "\' to [" << Addr << "]\n";);
  void *&CurVal = EEState.getGlobalAddressMap(locked)[GV];
  assert((CurVal == 0 || Addr == 0) && "GlobalMapping already established!");
  CurVal = Addr;

  // If we are using the reverse mapping, add it too.
  if (!EEState.getGlobalAddressReverseMap(locked).empty()) {
    AssertingVH<const GlobalValue> &V =
      EEState.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
}

void ExecutionEngine::clearAllGlobalMappings() {
  MutexGuard locked(lock);

  EEState.getGlobalAddressMap(locked).clear();
  EEState.getGlobalAddressReverseMap(locked).clear();
}

void ExecutionEngine::clearGlobalMappingsFromModule(Module *M) {
  MutexGuard locked(lock);

  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI)
    EEState.RemoveMapping(locked, FI);
  for (Module::global_iterator GI = M->global_begin(), GE = M->global_end();
       GI != GE; ++GI)
    EEState.RemoveMapping(locked, GI);
}

void *ExecutionEngine::updateGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);

  ExecutionEngineState::GlobalAddressMapTy &Map =
    EEState.getGlobalAddressMap(locked);

  // Deleting from the mapping?
  if (Addr == 0)
    return EEState.RemoveMapping(locked, GV);

  void *&CurVal = Map[GV];
  void *OldVal = CurVal;

  if (CurVal && !EEState.getGlobalAddressReverseMap(locked).empty())
    EEState.getGlobalAddressReverseMap(locked).erase(CurVal);
  CurVal = Addr;

  // If we are using the reverse mapping, add it too.
  if (!EEState.getGlobalAddressReverseMap(locked).empty()) {
    AssertingVH<const GlobalValue> &V =
      EEState.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
  return OldVal;
}

void *ExecutionEngine::getPointerToGlobalIfAvailable(const GlobalValue *GV) {
  MutexGuard locked(lock);

  ExecutionEngineState::GlobalAddressMapTy::iterator I =
    EEState.getGlobalAddressMap(locked).find(GV);
  return I != EEState.getGlobalAddressMap(locked).end() ? I->second : 0;
}

const GlobalValue *ExecutionEngine::getGlobalValueAtAddress(void *Addr) {
  MutexGuard locked(lock);

  // If we haven't computed the reverse mapping yet, do so first.
  if (EEState.getGlobalAddressReverseMap(locked).empty()) {
    for (ExecutionEngineState::GlobalAddressMapTy::iterator
         I = EEState.getGlobalAddressMap(locked).begin(),
         E = EEState.getGlobalAddressMap(locked).end(); I != E; ++I)
      EEState.getGlobalAddressReverseMap(locked).insert(std::make_pair(
                                                          I->second, I->first));
  }

  std::map<void *, AssertingVH<const GlobalValue> >::iterator I =
    EEState.getGlobalAddressReverseMap(locked).find(Addr);
  return I != EEState.getGlobalAddressReverseMap(locked).end() ? I->second : 0;
}

namespace {
class ArgvArray {
  char *Array;
  std::vector<char*> Values;
public:
  ArgvArray() : Array(NULL) {}
  ~ArgvArray() { clear(); }
  void clear() {
    delete[] Array;
    Array = NULL;
    for (size_t I = 0, E = Values.size(); I != E; ++I) {
      delete[] Values[I];
    }
    Values.clear();
  }
  /// Turn a vector of strings into a nice argv style array of pointers to null
  /// terminated strings.
  void *reset(LLVMContext &C, ExecutionEngine *EE,
              const std::vector<std::string> &InputArgv);
};
}  // anonymous namespace
void *ArgvArray::reset(LLVMContext &C, ExecutionEngine *EE,
                       const std::vector<std::string> &InputArgv) {
  clear();  // Free the old contents.
  unsigned PtrSize = EE->getDataLayout()->getPointerSize();
  Array = new char[(InputArgv.size()+1)*PtrSize];

  DEBUG(dbgs() << "JIT: ARGV = " << (void*)Array << "\n");
  Type *SBytePtr = Type::getInt8PtrTy(C);

  for (unsigned i = 0; i != InputArgv.size(); ++i) {
    unsigned Size = InputArgv[i].size()+1;
    char *Dest = new char[Size];
    Values.push_back(Dest);
    DEBUG(dbgs() << "JIT: ARGV[" << i << "] = " << (void*)Dest << "\n");

    std::copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
    Dest[Size-1] = 0;

    // Endian safe: Array[i] = (PointerTy)Dest;
    EE->StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Array+i*PtrSize),
                           SBytePtr);
  }

  // Null terminate it
  EE->StoreValueToMemory(PTOGV(0),
                         (GenericValue*)(Array+InputArgv.size()*PtrSize),
                         SBytePtr);
  return Array;
}

void ExecutionEngine::runStaticConstructorsDestructors(Module *module,
                                                       bool isDtors) {
  const char *Name = isDtors ? "llvm.global_dtors" : "llvm.global_ctors";
  GlobalVariable *GV = module->getNamedGlobal(Name);

  // If this global has internal linkage, or if it has a use, then it must be
  // an old-style (llvmgcc3) static ctor with __main linked in and in use.  If
  // this is the case, don't execute any of the global ctors, __main will do
  // it.
  if (!GV || GV->isDeclaration() || GV->hasLocalLinkage()) return;

  // Should be an array of '{ i32, void ()* }' structs.  The first value is
  // the init priority, which we ignore.
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (InitList == 0)
    return;
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i));
    if (CS == 0) continue;

    Constant *FP = CS->getOperand(1);
    if (FP->isNullValue())
      continue;  // Found a sentinal value, ignore.

    // Strip off constant expression casts.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
      if (CE->isCast())
        FP = CE->getOperand(0);

    // Execute the ctor/dtor function!
    if (Function *F = dyn_cast<Function>(FP))
      runFunction(F, std::vector<GenericValue>());

    // FIXME: It is marginally lame that we just do nothing here if we see an
    // entry we don't recognize. It might not be unreasonable for the verifier
    // to not even allow this and just assert here.
  }
}

void ExecutionEngine::runStaticConstructorsDestructors(bool isDtors) {
  // Execute global ctors/dtors for each module in the program.
  for (unsigned i = 0, e = Modules.size(); i != e; ++i)
    runStaticConstructorsDestructors(Modules[i], isDtors);
}

#ifndef NDEBUG
/// isTargetNullPtr - Return whether the target pointer stored at Loc is null.
static bool isTargetNullPtr(ExecutionEngine *EE, void *Loc) {
  unsigned PtrSize = EE->getDataLayout()->getPointerSize();
  for (unsigned i = 0; i < PtrSize; ++i)
    if (*(i + (uint8_t*)Loc))
      return false;
  return true;
}
#endif

int ExecutionEngine::runFunctionAsMain(Function *Fn,
                                       const std::vector<std::string> &argv,
                                       const char * const * envp) {
  std::vector<GenericValue> GVArgs;
  GenericValue GVArgc;
  GVArgc.IntVal = APInt(32, argv.size());

  // Check main() type
  unsigned NumArgs = Fn->getFunctionType()->getNumParams();
  FunctionType *FTy = Fn->getFunctionType();
  Type* PPInt8Ty = Type::getInt8PtrTy(Fn->getContext())->getPointerTo();

  // Check the argument types.
  if (NumArgs > 3)
    report_fatal_error("Invalid number of arguments of main() supplied");
  if (NumArgs >= 3 && FTy->getParamType(2) != PPInt8Ty)
    report_fatal_error("Invalid type for third argument of main() supplied");
  if (NumArgs >= 2 && FTy->getParamType(1) != PPInt8Ty)
    report_fatal_error("Invalid type for second argument of main() supplied");
  if (NumArgs >= 1 && !FTy->getParamType(0)->isIntegerTy(32))
    report_fatal_error("Invalid type for first argument of main() supplied");
  if (!FTy->getReturnType()->isIntegerTy() &&
      !FTy->getReturnType()->isVoidTy())
    report_fatal_error("Invalid return type of main() supplied");

  ArgvArray CArgv;
  ArgvArray CEnv;
  if (NumArgs) {
    GVArgs.push_back(GVArgc); // Arg #0 = argc.
    if (NumArgs > 1) {
      // Arg #1 = argv.
      GVArgs.push_back(PTOGV(CArgv.reset(Fn->getContext(), this, argv)));
      assert(!isTargetNullPtr(this, GVTOP(GVArgs[1])) &&
             "argv[0] was null after CreateArgv");
      if (NumArgs > 2) {
        std::vector<std::string> EnvVars;
        for (unsigned i = 0; envp[i]; ++i)
          EnvVars.push_back(envp[i]);
        // Arg #2 = envp.
        GVArgs.push_back(PTOGV(CEnv.reset(Fn->getContext(), this, EnvVars)));
      }
    }
  }

  return runFunction(Fn, GVArgs).IntVal.getZExtValue();
}

ExecutionEngine *ExecutionEngine::create(Module *M,
                                         bool ForceInterpreter,
                                         std::string *ErrorStr,
                                         CodeGenOpt::Level OptLevel,
                                         bool GVsWithCode) {
  EngineBuilder EB =  EngineBuilder(M)
      .setEngineKind(ForceInterpreter
                     ? EngineKind::Interpreter
                     : EngineKind::JIT)
      .setErrorStr(ErrorStr)
      .setOptLevel(OptLevel)
      .setAllocateGVsWithCode(GVsWithCode);

  return EB.create();
}

/// createJIT - This is the factory method for creating a JIT for the current
/// machine, it does not fall back to the interpreter.  This takes ownership
/// of the module.
ExecutionEngine *ExecutionEngine::createJIT(Module *M,
                                            std::string *ErrorStr,
                                            JITMemoryManager *JMM,
                                            CodeGenOpt::Level OL,
                                            bool GVsWithCode,
                                            Reloc::Model RM,
                                            CodeModel::Model CMM) {
  if (ExecutionEngine::JITCtor == 0) {
    if (ErrorStr)
      *ErrorStr = "JIT has not been linked in.";
    return 0;
  }

  // Use the defaults for extra parameters.  Users can use EngineBuilder to
  // set them.
  EngineBuilder EB(M);
  EB.setEngineKind(EngineKind::JIT);
  EB.setErrorStr(ErrorStr);
  EB.setRelocationModel(RM);
  EB.setCodeModel(CMM);
  EB.setAllocateGVsWithCode(GVsWithCode);
  EB.setOptLevel(OL);
  EB.setJITMemoryManager(JMM);

  // TODO: permit custom TargetOptions here
  TargetMachine *TM = EB.selectTarget();
  if (!TM || (ErrorStr && ErrorStr->length() > 0)) return 0;

  return ExecutionEngine::JITCtor(M, ErrorStr, JMM, GVsWithCode, TM);
}

ExecutionEngine *EngineBuilder::create(TargetMachine *TM) {
  OwningPtr<TargetMachine> TheTM(TM); // Take ownership.

  // Make sure we can resolve symbols in the program as well. The zero arg
  // to the function tells DynamicLibrary to load the program, not a library.
  if (sys::DynamicLibrary::LoadLibraryPermanently(0, ErrorStr))
    return 0;

  // If the user specified a memory manager but didn't specify which engine to
  // create, we assume they only want the JIT, and we fail if they only want
  // the interpreter.
  if (JMM) {
    if (WhichEngine & EngineKind::JIT)
      WhichEngine = EngineKind::JIT;
    else {
      if (ErrorStr)
        *ErrorStr = "Cannot create an interpreter with a memory manager.";
      return 0;
    }
  }

  // Unless the interpreter was explicitly selected or the JIT is not linked,
  // try making a JIT.
  if ((WhichEngine & EngineKind::JIT) && TheTM) {
    Triple TT(M->getTargetTriple());
    if (!TM->getTarget().hasJIT()) {
      errs() << "WARNING: This target JIT is not designed for the host"
             << " you are running.  If bad things happen, please choose"
             << " a different -march switch.\n";
    }

    if (UseMCJIT && ExecutionEngine::MCJITCtor) {
      ExecutionEngine *EE =
        ExecutionEngine::MCJITCtor(M, ErrorStr, JMM,
                                   AllocateGVsWithCode, TheTM.take());
      if (EE) return EE;
    } else if (ExecutionEngine::JITCtor) {
      ExecutionEngine *EE =
        ExecutionEngine::JITCtor(M, ErrorStr, JMM,
                                 AllocateGVsWithCode, TheTM.take());
      if (EE) return EE;
    }
  }

  // If we can't make a JIT and we didn't request one specifically, try making
  // an interpreter instead.
  if (WhichEngine & EngineKind::Interpreter) {
    if (ExecutionEngine::InterpCtor)
      return ExecutionEngine::InterpCtor(M, ErrorStr);
    if (ErrorStr)
      *ErrorStr = "Interpreter has not been linked in.";
    return 0;
  }

  if ((WhichEngine & EngineKind::JIT) && ExecutionEngine::JITCtor == 0 &&
      ExecutionEngine::MCJITCtor == 0) {
    if (ErrorStr)
      *ErrorStr = "JIT has not been linked in.";
  }

  return 0;
}

void *ExecutionEngine::getPointerToGlobal(const GlobalValue *GV) {
  if (Function *F = const_cast<Function*>(dyn_cast<Function>(GV)))
    return getPointerToFunction(F);

  MutexGuard locked(lock);
  if (void *P = EEState.getGlobalAddressMap(locked)[GV])
    return P;

  // Global variable might have been added since interpreter started.
  if (GlobalVariable *GVar =
          const_cast<GlobalVariable *>(dyn_cast<GlobalVariable>(GV)))
    EmitGlobalVariable(GVar);
  else
    llvm_unreachable("Global hasn't had an address allocated yet!");

  return EEState.getGlobalAddressMap(locked)[GV];
}

/// \brief Converts a Constant* into a GenericValue, including handling of
/// ConstantExpr values.
GenericValue ExecutionEngine::getConstantValue(const Constant *C) {
  // If its undefined, return the garbage.
  if (isa<UndefValue>(C)) {
    GenericValue Result;
    switch (C->getType()->getTypeID()) {
    case Type::IntegerTyID:
    case Type::X86_FP80TyID:
    case Type::FP128TyID:
    case Type::PPC_FP128TyID:
      // Although the value is undefined, we still have to construct an APInt
      // with the correct bit width.
      Result.IntVal = APInt(C->getType()->getPrimitiveSizeInBits(), 0);
      break;
    default:
      break;
    }
    return Result;
  }

  // Otherwise, if the value is a ConstantExpr...
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    Constant *Op0 = CE->getOperand(0);
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // Compute the index
      GenericValue Result = getConstantValue(Op0);
      SmallVector<Value*, 8> Indices(CE->op_begin()+1, CE->op_end());
      uint64_t Offset = TD->getIndexedOffset(Op0->getType(), Indices);

      char* tmp = (char*) Result.PointerVal;
      Result = PTOGV(tmp + Offset);
      return Result;
    }
    case Instruction::Trunc: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.trunc(BitWidth);
      return GV;
    }
    case Instruction::ZExt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.zext(BitWidth);
      return GV;
    }
    case Instruction::SExt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.sext(BitWidth);
      return GV;
    }
    case Instruction::FPTrunc: {
      // FIXME long double
      GenericValue GV = getConstantValue(Op0);
      GV.FloatVal = float(GV.DoubleVal);
      return GV;
    }
    case Instruction::FPExt:{
      // FIXME long double
      GenericValue GV = getConstantValue(Op0);
      GV.DoubleVal = double(GV.FloatVal);
      return GV;
    }
    case Instruction::UIToFP: {
      GenericValue GV = getConstantValue(Op0);
      if (CE->getType()->isFloatTy())
        GV.FloatVal = float(GV.IntVal.roundToDouble());
      else if (CE->getType()->isDoubleTy())
        GV.DoubleVal = GV.IntVal.roundToDouble();
      else if (CE->getType()->isX86_FP80Ty()) {
        APFloat apf = APFloat::getZero(APFloat::x87DoubleExtended);
        (void)apf.convertFromAPInt(GV.IntVal,
                                   false,
                                   APFloat::rmNearestTiesToEven);
        GV.IntVal = apf.bitcastToAPInt();
      }
      return GV;
    }
    case Instruction::SIToFP: {
      GenericValue GV = getConstantValue(Op0);
      if (CE->getType()->isFloatTy())
        GV.FloatVal = float(GV.IntVal.signedRoundToDouble());
      else if (CE->getType()->isDoubleTy())
        GV.DoubleVal = GV.IntVal.signedRoundToDouble();
      else if (CE->getType()->isX86_FP80Ty()) {
        APFloat apf = APFloat::getZero(APFloat::x87DoubleExtended);
        (void)apf.convertFromAPInt(GV.IntVal,
                                   true,
                                   APFloat::rmNearestTiesToEven);
        GV.IntVal = apf.bitcastToAPInt();
      }
      return GV;
    }
    case Instruction::FPToUI: // double->APInt conversion handles sign
    case Instruction::FPToSI: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      if (Op0->getType()->isFloatTy())
        GV.IntVal = APIntOps::RoundFloatToAPInt(GV.FloatVal, BitWidth);
      else if (Op0->getType()->isDoubleTy())
        GV.IntVal = APIntOps::RoundDoubleToAPInt(GV.DoubleVal, BitWidth);
      else if (Op0->getType()->isX86_FP80Ty()) {
        APFloat apf = APFloat(GV.IntVal);
        uint64_t v;
        bool ignored;
        (void)apf.convertToInteger(&v, BitWidth,
                                   CE->getOpcode()==Instruction::FPToSI,
                                   APFloat::rmTowardZero, &ignored);
        GV.IntVal = v; // endian?
      }
      return GV;
    }
    case Instruction::PtrToInt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t PtrWidth = TD->getTypeSizeInBits(Op0->getType());
      assert(PtrWidth <= 64 && "Bad pointer width");
      GV.IntVal = APInt(PtrWidth, uintptr_t(GV.PointerVal));
      uint32_t IntWidth = TD->getTypeSizeInBits(CE->getType());
      GV.IntVal = GV.IntVal.zextOrTrunc(IntWidth);
      return GV;
    }
    case Instruction::IntToPtr: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t PtrWidth = TD->getTypeSizeInBits(CE->getType());
      GV.IntVal = GV.IntVal.zextOrTrunc(PtrWidth);
      assert(GV.IntVal.getBitWidth() <= 64 && "Bad pointer width");
      GV.PointerVal = PointerTy(uintptr_t(GV.IntVal.getZExtValue()));
      return GV;
    }
    case Instruction::BitCast: {
      GenericValue GV = getConstantValue(Op0);
      Type* DestTy = CE->getType();
      switch (Op0->getType()->getTypeID()) {
        default: llvm_unreachable("Invalid bitcast operand");
        case Type::IntegerTyID:
          assert(DestTy->isFloatingPointTy() && "invalid bitcast");
          if (DestTy->isFloatTy())
            GV.FloatVal = GV.IntVal.bitsToFloat();
          else if (DestTy->isDoubleTy())
            GV.DoubleVal = GV.IntVal.bitsToDouble();
          break;
        case Type::FloatTyID:
          assert(DestTy->isIntegerTy(32) && "Invalid bitcast");
          GV.IntVal = APInt::floatToBits(GV.FloatVal);
          break;
        case Type::DoubleTyID:
          assert(DestTy->isIntegerTy(64) && "Invalid bitcast");
          GV.IntVal = APInt::doubleToBits(GV.DoubleVal);
          break;
        case Type::PointerTyID:
          assert(DestTy->isPointerTy() && "Invalid bitcast");
          break; // getConstantValue(Op0)  above already converted it
      }
      return GV;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      GenericValue LHS = getConstantValue(Op0);
      GenericValue RHS = getConstantValue(CE->getOperand(1));
      GenericValue GV;
      switch (CE->getOperand(0)->getType()->getTypeID()) {
      default: llvm_unreachable("Bad add type!");
      case Type::IntegerTyID:
        switch (CE->getOpcode()) {
          default: llvm_unreachable("Invalid integer opcode");
          case Instruction::Add: GV.IntVal = LHS.IntVal + RHS.IntVal; break;
          case Instruction::Sub: GV.IntVal = LHS.IntVal - RHS.IntVal; break;
          case Instruction::Mul: GV.IntVal = LHS.IntVal * RHS.IntVal; break;
          case Instruction::UDiv:GV.IntVal = LHS.IntVal.udiv(RHS.IntVal); break;
          case Instruction::SDiv:GV.IntVal = LHS.IntVal.sdiv(RHS.IntVal); break;
          case Instruction::URem:GV.IntVal = LHS.IntVal.urem(RHS.IntVal); break;
          case Instruction::SRem:GV.IntVal = LHS.IntVal.srem(RHS.IntVal); break;
          case Instruction::And: GV.IntVal = LHS.IntVal & RHS.IntVal; break;
          case Instruction::Or:  GV.IntVal = LHS.IntVal | RHS.IntVal; break;
          case Instruction::Xor: GV.IntVal = LHS.IntVal ^ RHS.IntVal; break;
        }
        break;
      case Type::FloatTyID:
        switch (CE->getOpcode()) {
          default: llvm_unreachable("Invalid float opcode");
          case Instruction::FAdd:
            GV.FloatVal = LHS.FloatVal + RHS.FloatVal; break;
          case Instruction::FSub:
            GV.FloatVal = LHS.FloatVal - RHS.FloatVal; break;
          case Instruction::FMul:
            GV.FloatVal = LHS.FloatVal * RHS.FloatVal; break;
          case Instruction::FDiv:
            GV.FloatVal = LHS.FloatVal / RHS.FloatVal; break;
          case Instruction::FRem:
            GV.FloatVal = std::fmod(LHS.FloatVal,RHS.FloatVal); break;
        }
        break;
      case Type::DoubleTyID:
        switch (CE->getOpcode()) {
          default: llvm_unreachable("Invalid double opcode");
          case Instruction::FAdd:
            GV.DoubleVal = LHS.DoubleVal + RHS.DoubleVal; break;
          case Instruction::FSub:
            GV.DoubleVal = LHS.DoubleVal - RHS.DoubleVal; break;
          case Instruction::FMul:
            GV.DoubleVal = LHS.DoubleVal * RHS.DoubleVal; break;
          case Instruction::FDiv:
            GV.DoubleVal = LHS.DoubleVal / RHS.DoubleVal; break;
          case Instruction::FRem:
            GV.DoubleVal = std::fmod(LHS.DoubleVal,RHS.DoubleVal); break;
        }
        break;
      case Type::X86_FP80TyID:
      case Type::PPC_FP128TyID:
      case Type::FP128TyID: {
        APFloat apfLHS = APFloat(LHS.IntVal);
        switch (CE->getOpcode()) {
          default: llvm_unreachable("Invalid long double opcode");
          case Instruction::FAdd:
            apfLHS.add(APFloat(RHS.IntVal), APFloat::rmNearestTiesToEven);
            GV.IntVal = apfLHS.bitcastToAPInt();
            break;
          case Instruction::FSub:
            apfLHS.subtract(APFloat(RHS.IntVal), APFloat::rmNearestTiesToEven);
            GV.IntVal = apfLHS.bitcastToAPInt();
            break;
          case Instruction::FMul:
            apfLHS.multiply(APFloat(RHS.IntVal), APFloat::rmNearestTiesToEven);
            GV.IntVal = apfLHS.bitcastToAPInt();
            break;
          case Instruction::FDiv:
            apfLHS.divide(APFloat(RHS.IntVal), APFloat::rmNearestTiesToEven);
            GV.IntVal = apfLHS.bitcastToAPInt();
            break;
          case Instruction::FRem:
            apfLHS.mod(APFloat(RHS.IntVal), APFloat::rmNearestTiesToEven);
            GV.IntVal = apfLHS.bitcastToAPInt();
            break;
          }
        }
        break;
      }
      return GV;
    }
    default:
      break;
    }

    SmallString<256> Msg;
    raw_svector_ostream OS(Msg);
    OS << "ConstantExpr not handled: " << *CE;
    report_fatal_error(OS.str());
  }

  // Otherwise, we have a simple constant.
  GenericValue Result;
  switch (C->getType()->getTypeID()) {
  case Type::FloatTyID:
    Result.FloatVal = cast<ConstantFP>(C)->getValueAPF().convertToFloat();
    break;
  case Type::DoubleTyID:
    Result.DoubleVal = cast<ConstantFP>(C)->getValueAPF().convertToDouble();
    break;
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
    Result.IntVal = cast <ConstantFP>(C)->getValueAPF().bitcastToAPInt();
    break;
  case Type::IntegerTyID:
    Result.IntVal = cast<ConstantInt>(C)->getValue();
    break;
  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(C))
      Result.PointerVal = 0;
    else if (const Function *F = dyn_cast<Function>(C))
      Result = PTOGV(getPointerToFunctionOrStub(const_cast<Function*>(F)));
    else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
      Result = PTOGV(getOrEmitGlobalVariable(const_cast<GlobalVariable*>(GV)));
    else if (const BlockAddress *BA = dyn_cast<BlockAddress>(C))
      Result = PTOGV(getPointerToBasicBlock(const_cast<BasicBlock*>(
                                                        BA->getBasicBlock())));
    else
      llvm_unreachable("Unknown constant pointer type!");
    break;
  default:
    SmallString<256> Msg;
    raw_svector_ostream OS(Msg);
    OS << "ERROR: Constant unimplemented for type: " << *C->getType();
    report_fatal_error(OS.str());
  }

  return Result;
}

/// StoreIntToMemory - Fills the StoreBytes bytes of memory starting from Dst
/// with the integer held in IntVal.
static void StoreIntToMemory(const APInt &IntVal, uint8_t *Dst,
                             unsigned StoreBytes) {
  assert((IntVal.getBitWidth()+7)/8 >= StoreBytes && "Integer too small!");
  const uint8_t *Src = (const uint8_t *)IntVal.getRawData();

  if (sys::isLittleEndianHost()) {
    // Little-endian host - the source is ordered from LSB to MSB.  Order the
    // destination from LSB to MSB: Do a straight copy.
    memcpy(Dst, Src, StoreBytes);
  } else {
    // Big-endian host - the source is an array of 64 bit words ordered from
    // LSW to MSW.  Each word is ordered from MSB to LSB.  Order the destination
    // from MSB to LSB: Reverse the word order, but not the bytes in a word.
    while (StoreBytes > sizeof(uint64_t)) {
      StoreBytes -= sizeof(uint64_t);
      // May not be aligned so use memcpy.
      memcpy(Dst + StoreBytes, Src, sizeof(uint64_t));
      Src += sizeof(uint64_t);
    }

    memcpy(Dst, Src + sizeof(uint64_t) - StoreBytes, StoreBytes);
  }
}

void ExecutionEngine::StoreValueToMemory(const GenericValue &Val,
                                         GenericValue *Ptr, Type *Ty) {
  const unsigned StoreBytes = getDataLayout()->getTypeStoreSize(Ty);

  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    StoreIntToMemory(Val.IntVal, (uint8_t*)Ptr, StoreBytes);
    break;
  case Type::FloatTyID:
    *((float*)Ptr) = Val.FloatVal;
    break;
  case Type::DoubleTyID:
    *((double*)Ptr) = Val.DoubleVal;
    break;
  case Type::X86_FP80TyID:
    memcpy(Ptr, Val.IntVal.getRawData(), 10);
    break;
  case Type::PointerTyID:
    // Ensure 64 bit target pointers are fully initialized on 32 bit hosts.
    if (StoreBytes != sizeof(PointerTy))
      memset(&(Ptr->PointerVal), 0, StoreBytes);

    *((PointerTy*)Ptr) = Val.PointerVal;
    break;
  default:
    dbgs() << "Cannot store value of type " << *Ty << "!\n";
  }

  if (sys::isLittleEndianHost() != getDataLayout()->isLittleEndian())
    // Host and target are different endian - reverse the stored bytes.
    std::reverse((uint8_t*)Ptr, StoreBytes + (uint8_t*)Ptr);
}

/// LoadIntFromMemory - Loads the integer stored in the LoadBytes bytes starting
/// from Src into IntVal, which is assumed to be wide enough and to hold zero.
static void LoadIntFromMemory(APInt &IntVal, uint8_t *Src, unsigned LoadBytes) {
  assert((IntVal.getBitWidth()+7)/8 >= LoadBytes && "Integer too small!");
  uint8_t *Dst = (uint8_t *)IntVal.getRawData();

  if (sys::isLittleEndianHost())
    // Little-endian host - the destination must be ordered from LSB to MSB.
    // The source is ordered from LSB to MSB: Do a straight copy.
    memcpy(Dst, Src, LoadBytes);
  else {
    // Big-endian - the destination is an array of 64 bit words ordered from
    // LSW to MSW.  Each word must be ordered from MSB to LSB.  The source is
    // ordered from MSB to LSB: Reverse the word order, but not the bytes in
    // a word.
    while (LoadBytes > sizeof(uint64_t)) {
      LoadBytes -= sizeof(uint64_t);
      // May not be aligned so use memcpy.
      memcpy(Dst, Src + LoadBytes, sizeof(uint64_t));
      Dst += sizeof(uint64_t);
    }

    memcpy(Dst + sizeof(uint64_t) - LoadBytes, Src, LoadBytes);
  }
}

/// FIXME: document
///
void ExecutionEngine::LoadValueFromMemory(GenericValue &Result,
                                          GenericValue *Ptr,
                                          Type *Ty) {
  const unsigned LoadBytes = getDataLayout()->getTypeStoreSize(Ty);

  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    // An APInt with all words initially zero.
    Result.IntVal = APInt(cast<IntegerType>(Ty)->getBitWidth(), 0);
    LoadIntFromMemory(Result.IntVal, (uint8_t*)Ptr, LoadBytes);
    break;
  case Type::FloatTyID:
    Result.FloatVal = *((float*)Ptr);
    break;
  case Type::DoubleTyID:
    Result.DoubleVal = *((double*)Ptr);
    break;
  case Type::PointerTyID:
    Result.PointerVal = *((PointerTy*)Ptr);
    break;
  case Type::X86_FP80TyID: {
    // This is endian dependent, but it will only work on x86 anyway.
    // FIXME: Will not trap if loading a signaling NaN.
    uint64_t y[2];
    memcpy(y, Ptr, 10);
    Result.IntVal = APInt(80, y);
    break;
  }
  default:
    SmallString<256> Msg;
    raw_svector_ostream OS(Msg);
    OS << "Cannot load value of type " << *Ty << "!";
    report_fatal_error(OS.str());
  }
}

void ExecutionEngine::InitializeMemory(const Constant *Init, void *Addr) {
  DEBUG(dbgs() << "JIT: Initializing " << Addr << " ");
  DEBUG(Init->dump());
  if (isa<UndefValue>(Init))
    return;
  
  if (const ConstantVector *CP = dyn_cast<ConstantVector>(Init)) {
    unsigned ElementSize =
      getDataLayout()->getTypeAllocSize(CP->getType()->getElementType());
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      InitializeMemory(CP->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  }
  
  if (isa<ConstantAggregateZero>(Init)) {
    memset(Addr, 0, (size_t)getDataLayout()->getTypeAllocSize(Init->getType()));
    return;
  }
  
  if (const ConstantArray *CPA = dyn_cast<ConstantArray>(Init)) {
    unsigned ElementSize =
      getDataLayout()->getTypeAllocSize(CPA->getType()->getElementType());
    for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
      InitializeMemory(CPA->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  }
  
  if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(Init)) {
    const StructLayout *SL =
      getDataLayout()->getStructLayout(cast<StructType>(CPS->getType()));
    for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
      InitializeMemory(CPS->getOperand(i), (char*)Addr+SL->getElementOffset(i));
    return;
  }

  if (const ConstantDataSequential *CDS =
               dyn_cast<ConstantDataSequential>(Init)) {
    // CDS is already laid out in host memory order.
    StringRef Data = CDS->getRawDataValues();
    memcpy(Addr, Data.data(), Data.size());
    return;
  }

  if (Init->getType()->isFirstClassType()) {
    GenericValue Val = getConstantValue(Init);
    StoreValueToMemory(Val, (GenericValue*)Addr, Init->getType());
    return;
  }

  DEBUG(dbgs() << "Bad Type: " << *Init->getType() << "\n");
  llvm_unreachable("Unknown constant type to initialize memory with!");
}

/// EmitGlobals - Emit all of the global variables to memory, storing their
/// addresses into GlobalAddress.  This must make sure to copy the contents of
/// their initializers into the memory.
void ExecutionEngine::emitGlobals() {
  // Loop over all of the global variables in the program, allocating the memory
  // to hold them.  If there is more than one module, do a prepass over globals
  // to figure out how the different modules should link together.
  std::map<std::pair<std::string, Type*>,
           const GlobalValue*> LinkedGlobalsMap;

  if (Modules.size() != 1) {
    for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
      Module &M = *Modules[m];
      for (Module::const_global_iterator I = M.global_begin(),
           E = M.global_end(); I != E; ++I) {
        const GlobalValue *GV = I;
        if (GV->hasLocalLinkage() || GV->isDeclaration() ||
            GV->hasAppendingLinkage() || !GV->hasName())
          continue;// Ignore external globals and globals with internal linkage.

        const GlobalValue *&GVEntry =
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];

        // If this is the first time we've seen this global, it is the canonical
        // version.
        if (!GVEntry) {
          GVEntry = GV;
          continue;
        }

        // If the existing global is strong, never replace it.
        if (GVEntry->hasExternalLinkage() ||
            GVEntry->hasDLLImportLinkage() ||
            GVEntry->hasDLLExportLinkage())
          continue;

        // Otherwise, we know it's linkonce/weak, replace it if this is a strong
        // symbol.  FIXME is this right for common?
        if (GV->hasExternalLinkage() || GVEntry->hasExternalWeakLinkage())
          GVEntry = GV;
      }
    }
  }

  std::vector<const GlobalValue*> NonCanonicalGlobals;
  for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
    Module &M = *Modules[m];
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      // In the multi-module case, see what this global maps to.
      if (!LinkedGlobalsMap.empty()) {
        if (const GlobalValue *GVEntry =
              LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())]) {
          // If something else is the canonical global, ignore this one.
          if (GVEntry != &*I) {
            NonCanonicalGlobals.push_back(I);
            continue;
          }
        }
      }

      if (!I->isDeclaration()) {
        addGlobalMapping(I, getMemoryForGV(I));
      } else {
        // External variable reference. Try to use the dynamic loader to
        // get a pointer to it.
        if (void *SymAddr =
            sys::DynamicLibrary::SearchForAddressOfSymbol(I->getName()))
          addGlobalMapping(I, SymAddr);
        else {
          report_fatal_error("Could not resolve external global address: "
                            +I->getName());
        }
      }
    }

    // If there are multiple modules, map the non-canonical globals to their
    // canonical location.
    if (!NonCanonicalGlobals.empty()) {
      for (unsigned i = 0, e = NonCanonicalGlobals.size(); i != e; ++i) {
        const GlobalValue *GV = NonCanonicalGlobals[i];
        const GlobalValue *CGV =
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];
        void *Ptr = getPointerToGlobalIfAvailable(CGV);
        assert(Ptr && "Canonical global wasn't codegen'd!");
        addGlobalMapping(GV, Ptr);
      }
    }

    // Now that all of the globals are set up in memory, loop through them all
    // and initialize their contents.
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->isDeclaration()) {
        if (!LinkedGlobalsMap.empty()) {
          if (const GlobalValue *GVEntry =
                LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())])
            if (GVEntry != &*I)  // Not the canonical variable.
              continue;
        }
        EmitGlobalVariable(I);
      }
    }
  }
}

// EmitGlobalVariable - This method emits the specified global variable to the
// address specified in GlobalAddresses, or allocates new memory if it's not
// already in the map.
void ExecutionEngine::EmitGlobalVariable(const GlobalVariable *GV) {
  void *GA = getPointerToGlobalIfAvailable(GV);

  if (GA == 0) {
    // If it's not already specified, allocate memory for the global.
    GA = getMemoryForGV(GV);
    addGlobalMapping(GV, GA);
  }

  // Don't initialize if it's thread local, let the client do it.
  if (!GV->isThreadLocal())
    InitializeMemory(GV->getInitializer(), GA);

  Type *ElTy = GV->getType()->getElementType();
  size_t GVSize = (size_t)getDataLayout()->getTypeAllocSize(ElTy);
  NumInitBytes += (unsigned)GVSize;
  ++NumGlobals;
}

ExecutionEngineState::ExecutionEngineState(ExecutionEngine &EE)
  : EE(EE), GlobalAddressMap(this) {
}

sys::Mutex *
ExecutionEngineState::AddressMapConfig::getMutex(ExecutionEngineState *EES) {
  return &EES->EE.lock;
}

void ExecutionEngineState::AddressMapConfig::onDelete(ExecutionEngineState *EES,
                                                      const GlobalValue *Old) {
  void *OldVal = EES->GlobalAddressMap.lookup(Old);
  EES->GlobalAddressReverseMap.erase(OldVal);
}

void ExecutionEngineState::AddressMapConfig::onRAUW(ExecutionEngineState *,
                                                    const GlobalValue *,
                                                    const GlobalValue *) {
  llvm_unreachable("The ExecutionEngine doesn't know how to handle a"
                   " RAUW on a value it has a global mapping for.");
}
