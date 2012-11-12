//===-- MCJIT.cpp - MC-based Just-in-Time Compiler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCJIT.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/DataLayout.h"

using namespace llvm;

namespace {

static struct RegisterJIT {
  RegisterJIT() { MCJIT::Register(); }
} JITRegistrator;

}

extern "C" void LLVMLinkInMCJIT() {
}

ExecutionEngine *MCJIT::createJIT(Module *M,
                                  std::string *ErrorStr,
                                  JITMemoryManager *JMM,
                                  bool GVsWithCode,
                                  TargetMachine *TM) {
  // Try to register the program as a source of symbols to resolve against.
  //
  // FIXME: Don't do this here.
  sys::DynamicLibrary::LoadLibraryPermanently(0, NULL);

  return new MCJIT(M, TM, JMM, GVsWithCode);
}

MCJIT::MCJIT(Module *m, TargetMachine *tm, RTDyldMemoryManager *MM,
             bool AllocateGVsWithCode)
  : ExecutionEngine(m), TM(tm), Ctx(0), MemMgr(MM), Dyld(MM),
    isCompiled(false), M(m)  {

  setDataLayout(TM->getDataLayout());
}

MCJIT::~MCJIT() {
  if (LoadedObject)
    NotifyFreeingObject(*LoadedObject.get());
  delete MemMgr;
  delete TM;
}

void MCJIT::emitObject(Module *m) {
  /// Currently, MCJIT only supports a single module and the module passed to
  /// this function call is expected to be the contained module.  The module
  /// is passed as a parameter here to prepare for multiple module support in
  /// the future.
  assert(M == m);

  // Get a thread lock to make sure we aren't trying to compile multiple times
  MutexGuard locked(lock);

  // FIXME: Track compilation state on a per-module basis when multiple modules
  //        are supported.
  // Re-compilation is not supported
  if (isCompiled)
    return;

  PassManager PM;

  PM.add(new DataLayout(*TM->getDataLayout()));

  // The RuntimeDyld will take ownership of this shortly
  OwningPtr<ObjectBufferStream> Buffer(new ObjectBufferStream());

  // Turn the machine code intermediate representation into bytes in memory
  // that may be executed.
  if (TM->addPassesToEmitMC(PM, Ctx, Buffer->getOStream(), false)) {
    report_fatal_error("Target does not support MC emission!");
  }

  // Initialize passes.
  PM.run(*m);
  // Flush the output buffer to get the generated code into memory
  Buffer->flush();

  // Load the object into the dynamic linker.
  // handing off ownership of the buffer
  LoadedObject.reset(Dyld.loadObject(Buffer.take()));
  if (!LoadedObject)
    report_fatal_error(Dyld.getErrorString());

  // Resolve any relocations.
  Dyld.resolveRelocations();

  // FIXME: Make this optional, maybe even move it to a JIT event listener
  LoadedObject->registerWithDebugger();

  NotifyObjectEmitted(*LoadedObject);

  // FIXME: Add support for per-module compilation state
  isCompiled = true;
}

// FIXME: Add a parameter to identify which object is being finalized when
// MCJIT supports multiple modules.
void MCJIT::finalizeObject() {
  // If the module hasn't been compiled, just do that.
  if (!isCompiled) {
    // If the call to Dyld.resolveRelocations() is removed from emitObject()
    // we'll need to do that here.
    emitObject(M);
    return;
  }

  // Resolve any relocations.
  Dyld.resolveRelocations();
}

void *MCJIT::getPointerToBasicBlock(BasicBlock *BB) {
  report_fatal_error("not yet implemented");
}

void *MCJIT::getPointerToFunction(Function *F) {
  // FIXME: This should really return a uint64_t since it's a pointer in the
  // target address space, not our local address space. That's part of the
  // ExecutionEngine interface, though. Fix that when the old JIT finally
  // dies.

  // FIXME: Add support for per-module compilation state
  if (!isCompiled)
    emitObject(M);

  if (F->isDeclaration() || F->hasAvailableExternallyLinkage()) {
    bool AbortOnFailure = !F->hasExternalWeakLinkage();
    void *Addr = getPointerToNamedFunction(F->getName(), AbortOnFailure);
    addGlobalMapping(F, Addr);
    return Addr;
  }

  // FIXME: Should the Dyld be retaining module information? Probably not.
  // FIXME: Should we be using the mangler for this? Probably.
  //
  // This is the accessor for the target address, so make sure to check the
  // load address of the symbol, not the local address.
  StringRef BaseName = F->getName();
  if (BaseName[0] == '\1')
    return (void*)Dyld.getSymbolLoadAddress(BaseName.substr(1));
  return (void*)Dyld.getSymbolLoadAddress((TM->getMCAsmInfo()->getGlobalPrefix()
                                       + BaseName).str());
}

void *MCJIT::recompileAndRelinkFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::freeMachineCodeForFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

GenericValue MCJIT::runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) {
  assert(F && "Function *F was null at entry to run()");

  void *FPtr = getPointerToFunction(F);
  assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();

  assert((FTy->getNumParams() == ArgValues.size() ||
          (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) &&
         "Wrong number of arguments passed into function!");
  assert(FTy->getNumParams() == ArgValues.size() &&
         "This doesn't support passing arguments through varargs (yet)!");

  // Handle some common cases first.  These cases correspond to common `main'
  // prototypes.
  if (RetTy->isIntegerTy(32) || RetTy->isVoidTy()) {
    switch (ArgValues.size()) {
    case 3:
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy() &&
          FTy->getParamType(2)->isPointerTy()) {
        int (*PF)(int, char **, const char **) =
          (int(*)(int, char **, const char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                 (char **)GVTOP(ArgValues[1]),
                                 (const char **)GVTOP(ArgValues[2])));
        return rv;
      }
      break;
    case 2:
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy()) {
        int (*PF)(int, char **) = (int(*)(int, char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                 (char **)GVTOP(ArgValues[1])));
        return rv;
      }
      break;
    case 1:
      if (FTy->getNumParams() == 1 &&
          FTy->getParamType(0)->isIntegerTy(32)) {
        GenericValue rv;
        int (*PF)(int) = (int(*)(int))(intptr_t)FPtr;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue()));
        return rv;
      }
      break;
    }
  }

  // Handle cases where no arguments are passed first.
  if (ArgValues.empty()) {
    GenericValue rv;
    switch (RetTy->getTypeID()) {
    default: llvm_unreachable("Unknown return type for function call!");
    case Type::IntegerTyID: {
      unsigned BitWidth = cast<IntegerType>(RetTy)->getBitWidth();
      if (BitWidth == 1)
        rv.IntVal = APInt(BitWidth, ((bool(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 8)
        rv.IntVal = APInt(BitWidth, ((char(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 16)
        rv.IntVal = APInt(BitWidth, ((short(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 32)
        rv.IntVal = APInt(BitWidth, ((int(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 64)
        rv.IntVal = APInt(BitWidth, ((int64_t(*)())(intptr_t)FPtr)());
      else
        llvm_unreachable("Integer types > 64 bits not supported");
      return rv;
    }
    case Type::VoidTyID:
      rv.IntVal = APInt(32, ((int(*)())(intptr_t)FPtr)());
      return rv;
    case Type::FloatTyID:
      rv.FloatVal = ((float(*)())(intptr_t)FPtr)();
      return rv;
    case Type::DoubleTyID:
      rv.DoubleVal = ((double(*)())(intptr_t)FPtr)();
      return rv;
    case Type::X86_FP80TyID:
    case Type::FP128TyID:
    case Type::PPC_FP128TyID:
      llvm_unreachable("long double not supported yet");
    case Type::PointerTyID:
      return PTOGV(((void*(*)())(intptr_t)FPtr)());
    }
  }

  llvm_unreachable("Full-featured argument passing not supported yet!");
}

void *MCJIT::getPointerToNamedFunction(const std::string &Name,
                                       bool AbortOnFailure) {
  // FIXME: Add support for per-module compilation state
  if (!isCompiled)
    emitObject(M);

  if (!isSymbolSearchingDisabled() && MemMgr) {
    void *ptr = MemMgr->getPointerToNamedFunction(Name, false);
    if (ptr)
      return ptr;
  }

  /// If a LazyFunctionCreator is installed, use it to get/create the function.
  if (LazyFunctionCreator)
    if (void *RP = LazyFunctionCreator(Name))
      return RP;

  if (AbortOnFailure) {
    report_fatal_error("Program used external function '"+Name+
                       "' which could not be resolved!");
  }
  return 0;
}

void MCJIT::RegisterJITEventListener(JITEventListener *L) {
  if (L == NULL)
    return;
  MutexGuard locked(lock);
  EventListeners.push_back(L);
}
void MCJIT::UnregisterJITEventListener(JITEventListener *L) {
  if (L == NULL)
    return;
  MutexGuard locked(lock);
  SmallVector<JITEventListener*, 2>::reverse_iterator I=
      std::find(EventListeners.rbegin(), EventListeners.rend(), L);
  if (I != EventListeners.rend()) {
    std::swap(*I, EventListeners.back());
    EventListeners.pop_back();
  }
}
void MCJIT::NotifyObjectEmitted(const ObjectImage& Obj) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyObjectEmitted(Obj);
  }
}
void MCJIT::NotifyFreeingObject(const ObjectImage& Obj) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyFreeingObject(Obj);
  }
}
