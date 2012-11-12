//===- MCJITTestBase.h - Common base class for MCJIT Unit tests  ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements common functionality required by the MCJIT unit tests,
// as well as logic to skip tests on unsupported architectures and operating
// systems.
//
//===----------------------------------------------------------------------===//


#ifndef MCJIT_TEST_BASE_H
#define MCJIT_TEST_BASE_H

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TypeBuilder.h"

#include "SectionMemoryManager.h"

// Used to skip tests on unsupported architectures and operating systems.
// To skip a test, add this macro at the top of a test-case in a suite that
// inherits from MCJITTestBase. See MCJITTest.cpp for examples.
#define SKIP_UNSUPPORTED_PLATFORM \
  do \
    if (!ArchSupportsMCJIT() || !OSSupportsMCJIT()) \
      return; \
  while(0);

namespace llvm {

class MCJITTestBase {
protected:

  MCJITTestBase()
    : OptLevel(CodeGenOpt::None)
    , RelocModel(Reloc::Default)
    , CodeModel(CodeModel::Default)
    , MArch("")
    , Builder(Context)
    , MM(new SectionMemoryManager)
    , HostTriple(LLVM_HOSTTRIPLE)
  {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

#ifdef LLVM_ON_WIN32
    // On Windows, generate ELF objects by specifying "-elf" in triple
    HostTriple += "-elf";
#endif // LLVM_ON_WIN32
    HostTriple = Triple::normalize(HostTriple);

    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::mips);
    SupportedArchs.push_back(Triple::x86);
    SupportedArchs.push_back(Triple::x86_64);

    // The operating systems below are known to be incompatible with MCJIT as
    // they are copied from the test/ExecutionEngine/MCJIT/lit.local.cfg and
    // should be kept in sync.
    UnsupportedOSs.push_back(Triple::Cygwin);
    UnsupportedOSs.push_back(Triple::Darwin);
  }

  /// Returns true if the host architecture is known to support MCJIT
  bool ArchSupportsMCJIT() {
    Triple Host(HostTriple);
    if (std::find(SupportedArchs.begin(), SupportedArchs.end(), Host.getArch())
        == SupportedArchs.end()) {
      return false;
    }
    return true;
  }

  /// Returns true if the host OS is known to support MCJIT
  bool OSSupportsMCJIT() {
    Triple Host(HostTriple);
    if (std::find(UnsupportedOSs.begin(), UnsupportedOSs.end(), Host.getOS())
        == UnsupportedOSs.end()) {
      return true;
    }
    return false;
  }

  Module *createEmptyModule(StringRef Name) {
    Module * M = new Module(Name, Context);
    M->setTargetTriple(Triple::normalize(HostTriple));
    return M;
  }

  template<typename FuncType>
  Function *startFunction(Module *M, StringRef Name) {
    Function *Result = Function::Create(
      TypeBuilder<FuncType, false>::get(Context),
      GlobalValue::ExternalLinkage, Name, M);

    BasicBlock *BB = BasicBlock::Create(Context, Name, Result);
    Builder.SetInsertPoint(BB);

    return Result;
  }

  void endFunctionWithRet(Function *Func, Value *RetValue) {
    Builder.CreateRet(RetValue);
  }

  // Inserts a simple function that invokes Callee and takes the same arguments:
  //    int Caller(...) { return Callee(...); }
  template<typename Signature>
  Function *insertSimpleCallFunction(Module *M, Function *Callee) {
    Function *Result = startFunction<Signature>(M, "caller");

    SmallVector<Value*, 1> CallArgs;

    Function::arg_iterator arg_iter = Result->arg_begin();
    for(;arg_iter != Result->arg_end(); ++arg_iter)
      CallArgs.push_back(arg_iter);

    Value *ReturnCode = Builder.CreateCall(Callee, CallArgs);
    Builder.CreateRet(ReturnCode);
    return Result;
  }

  // Inserts a function named 'main' that returns a uint32_t:
  //    int32_t main() { return X; }
  // where X is given by returnCode
  Function *insertMainFunction(Module *M, uint32_t returnCode) {
    Function *Result = startFunction<int32_t(void)>(M, "main");

    Value *ReturnVal = ConstantInt::get(Context, APInt(32, returnCode));
    endFunctionWithRet(Result, ReturnVal);

    return Result;
  }

  // Inserts a function
  //    int32_t add(int32_t a, int32_t b) { return a + b; }
  // in the current module and returns a pointer to it.
  Function *insertAddFunction(Module *M, StringRef Name = "add") {
    Function *Result = startFunction<int32_t(int32_t, int32_t)>(M, Name);

    Function::arg_iterator args = Result->arg_begin();
    Value *Arg1 = args;
    Value *Arg2 = ++args;
    Value *AddResult = Builder.CreateAdd(Arg1, Arg2);

    endFunctionWithRet(Result, AddResult);

    return Result;
  }

  // Inserts an declaration to a function defined elsewhere
  Function *insertExternalReferenceToFunction(Module *M, StringRef Name,
                                              FunctionType *FuncTy) {
    Function *Result = Function::Create(FuncTy,
                                        GlobalValue::ExternalLinkage,
                                        Name, M);
    return Result;
  }

  // Inserts an declaration to a function defined elsewhere
  Function *insertExternalReferenceToFunction(Module *M, Function *Func) {
    Function *Result = Function::Create(Func->getFunctionType(),
                                        GlobalValue::AvailableExternallyLinkage,
                                        Func->getName(), M);
    return Result;
  }

  // Inserts a global variable of type int32
  GlobalVariable *insertGlobalInt32(Module *M,
                                    StringRef name,
                                    int32_t InitialValue) {
    Type *GlobalTy = TypeBuilder<types::i<32>, true>::get(Context);
    Constant *IV = ConstantInt::get(Context, APInt(32, InitialValue));
    GlobalVariable *Global = new GlobalVariable(*M,
                                                GlobalTy,
                                                false,
                                                GlobalValue::ExternalLinkage,
                                                IV,
                                                name);
    return Global;
  }

  void createJIT(Module *M) {

    // Due to the EngineBuilder constructor, it is required to have a Module
    // in order to construct an ExecutionEngine (i.e. MCJIT)
    assert(M != 0 && "a non-null Module must be provided to create MCJIT");

    EngineBuilder EB(M);
    std::string Error;
    TheJIT.reset(EB.setEngineKind(EngineKind::JIT)
                 .setUseMCJIT(true) /* can this be folded into the EngineKind enum? */
                 .setJITMemoryManager(MM)
                 .setErrorStr(&Error)
                 .setOptLevel(CodeGenOpt::None)
                 .setAllocateGVsWithCode(false) /*does this do anything?*/
                 .setCodeModel(CodeModel::JITDefault)
                 .setRelocationModel(Reloc::Default)
                 .setMArch(MArch)
                 .setMCPU(sys::getHostCPUName())
                 //.setMAttrs(MAttrs)
                 .create());
    // At this point, we cannot modify the module any more.
    assert(TheJIT.get() != NULL && "error creating MCJIT with EngineBuilder");
  }

  LLVMContext Context;
  CodeGenOpt::Level OptLevel;
  Reloc::Model RelocModel;
  CodeModel::Model CodeModel;
  StringRef MArch;
  SmallVector<std::string, 1> MAttrs;
  OwningPtr<TargetMachine> TM;
  OwningPtr<ExecutionEngine> TheJIT;
  IRBuilder<> Builder;
  JITMemoryManager *MM;

  std::string HostTriple;
  SmallVector<Triple::ArchType, 4> SupportedArchs;
  SmallVector<Triple::OSType, 4> UnsupportedOSs;

  OwningPtr<Module> M;
};

} // namespace llvm

#endif // MCJIT_TEST_H
