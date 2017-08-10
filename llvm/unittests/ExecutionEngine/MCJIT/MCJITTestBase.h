//===- MCJITTestBase.h - Common base class for MCJIT Unit tests -*- C++ -*-===//
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

#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_MCJIT_MCJITTESTBASE_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_MCJIT_MCJITTESTBASE_H

#include "MCJITTestAPICommon.h"
#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

/// Helper class that can build very simple Modules
class TrivialModuleBuilder {
protected:
  LLVMContext Context;
  IRBuilder<> Builder;
  std::string BuilderTriple;

  TrivialModuleBuilder(const std::string &Triple)
    : Builder(Context), BuilderTriple(Triple) {}

  Module *createEmptyModule(StringRef Name = StringRef()) {
    Module * M = new Module(Name, Context);
    M->setTargetTriple(Triple::normalize(BuilderTriple));
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

    for (Argument &A : Result->args())
      CallArgs.push_back(&A);

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
    Value *Arg1 = &*args;
    Value *Arg2 = &*++args;
    Value *AddResult = Builder.CreateAdd(Arg1, Arg2);

    endFunctionWithRet(Result, AddResult);

    return Result;
  }

  // Inserts a declaration to a function defined elsewhere
  template <typename FuncType>
  Function *insertExternalReferenceToFunction(Module *M, StringRef Name) {
    Function *Result = Function::Create(
                         TypeBuilder<FuncType, false>::get(Context),
                         GlobalValue::ExternalLinkage, Name, M);
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
                                        GlobalValue::ExternalLinkage,
                                        Func->getName(), M);
    return Result;
  }

  // Inserts a global variable of type int32
  // FIXME: make this a template function to support any type
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

  // Inserts a function
  //   int32_t recursive_add(int32_t num) {
  //     if (num == 0) {
  //       return num;
  //     } else {
  //       int32_t recursive_param = num - 1;
  //       return num + Helper(recursive_param);
  //     }
  //   }
  // NOTE: if Helper is left as the default parameter, Helper == recursive_add.
  Function *insertAccumulateFunction(Module *M,
                                     Function *Helper = nullptr,
                                     StringRef Name = "accumulate") {
    Function *Result = startFunction<int32_t(int32_t)>(M, Name);
    if (!Helper)
      Helper = Result;

    BasicBlock *BaseCase = BasicBlock::Create(Context, "", Result);
    BasicBlock *RecursiveCase = BasicBlock::Create(Context, "", Result);

    // if (num == 0)
    Value *Param = &*Result->arg_begin();
    Value *Zero = ConstantInt::get(Context, APInt(32, 0));
    Builder.CreateCondBr(Builder.CreateICmpEQ(Param, Zero),
                         BaseCase, RecursiveCase);

    //   return num;
    Builder.SetInsertPoint(BaseCase);
    Builder.CreateRet(Param);

    //   int32_t recursive_param = num - 1;
    //   return Helper(recursive_param);
    Builder.SetInsertPoint(RecursiveCase);
    Value *One = ConstantInt::get(Context, APInt(32, 1));
    Value *RecursiveParam = Builder.CreateSub(Param, One);
    Value *RecursiveReturn = Builder.CreateCall(Helper, RecursiveParam);
    Value *Accumulator = Builder.CreateAdd(Param, RecursiveReturn);
    Builder.CreateRet(Accumulator);

    return Result;
  }

  // Populates Modules A and B:
  // Module A { Extern FB1, Function FA which calls FB1 },
  // Module B { Extern FA, Function FB1, Function FB2 which calls FA },
  void createCrossModuleRecursiveCase(std::unique_ptr<Module> &A, Function *&FA,
                                      std::unique_ptr<Module> &B,
                                      Function *&FB1, Function *&FB2) {
    // Define FB1 in B.
    B.reset(createEmptyModule("B"));
    FB1 = insertAccumulateFunction(B.get(), nullptr, "FB1");

    // Declare FB1 in A (as an external).
    A.reset(createEmptyModule("A"));
    Function *FB1Extern = insertExternalReferenceToFunction(A.get(), FB1);

    // Define FA in A (with a call to FB1).
    FA = insertAccumulateFunction(A.get(), FB1Extern, "FA");

    // Declare FA in B (as an external)
    Function *FAExtern = insertExternalReferenceToFunction(B.get(), FA);

    // Define FB2 in B (with a call to FA)
    FB2 = insertAccumulateFunction(B.get(), FAExtern, "FB2");
  }

  // Module A { Function FA },
  // Module B { Extern FA, Function FB which calls FA },
  // Module C { Extern FB, Function FC which calls FB },
  void
  createThreeModuleChainedCallsCase(std::unique_ptr<Module> &A, Function *&FA,
                                    std::unique_ptr<Module> &B, Function *&FB,
                                    std::unique_ptr<Module> &C, Function *&FC) {
    A.reset(createEmptyModule("A"));
    FA = insertAddFunction(A.get());

    B.reset(createEmptyModule("B"));
    Function *FAExtern_in_B = insertExternalReferenceToFunction(B.get(), FA);
    FB = insertSimpleCallFunction<int32_t(int32_t, int32_t)>(B.get(), FAExtern_in_B);

    C.reset(createEmptyModule("C"));
    Function *FBExtern_in_C = insertExternalReferenceToFunction(C.get(), FB);
    FC = insertSimpleCallFunction<int32_t(int32_t, int32_t)>(C.get(), FBExtern_in_C);
  }

  // Module A { Function FA },
  // Populates Modules A and B:
  // Module B { Function FB }
  void createTwoModuleCase(std::unique_ptr<Module> &A, Function *&FA,
                           std::unique_ptr<Module> &B, Function *&FB) {
    A.reset(createEmptyModule("A"));
    FA = insertAddFunction(A.get());

    B.reset(createEmptyModule("B"));
    FB = insertAddFunction(B.get());
  }

  // Module A { Function FA },
  // Module B { Extern FA, Function FB which calls FA }
  void createTwoModuleExternCase(std::unique_ptr<Module> &A, Function *&FA,
                                 std::unique_ptr<Module> &B, Function *&FB) {
    A.reset(createEmptyModule("A"));
    FA = insertAddFunction(A.get());

    B.reset(createEmptyModule("B"));
    Function *FAExtern_in_B = insertExternalReferenceToFunction(B.get(), FA);
    FB = insertSimpleCallFunction<int32_t(int32_t, int32_t)>(B.get(),
                                                             FAExtern_in_B);
  }

  // Module A { Function FA },
  // Module B { Extern FA, Function FB which calls FA },
  // Module C { Extern FB, Function FC which calls FA },
  void createThreeModuleCase(std::unique_ptr<Module> &A, Function *&FA,
                             std::unique_ptr<Module> &B, Function *&FB,
                             std::unique_ptr<Module> &C, Function *&FC) {
    A.reset(createEmptyModule("A"));
    FA = insertAddFunction(A.get());

    B.reset(createEmptyModule("B"));
    Function *FAExtern_in_B = insertExternalReferenceToFunction(B.get(), FA);
    FB = insertSimpleCallFunction<int32_t(int32_t, int32_t)>(B.get(), FAExtern_in_B);

    C.reset(createEmptyModule("C"));
    Function *FAExtern_in_C = insertExternalReferenceToFunction(C.get(), FA);
    FC = insertSimpleCallFunction<int32_t(int32_t, int32_t)>(C.get(), FAExtern_in_C);
  }
};

class MCJITTestBase : public MCJITTestAPICommon, public TrivialModuleBuilder {
protected:
  MCJITTestBase()
      : TrivialModuleBuilder(HostTriple), OptLevel(CodeGenOpt::None),
        CodeModel(CodeModel::Small), MArch(""), MM(new SectionMemoryManager) {
    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::aarch64);
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::mips);
    SupportedArchs.push_back(Triple::mipsel);
    SupportedArchs.push_back(Triple::mips64);
    SupportedArchs.push_back(Triple::mips64el);
    SupportedArchs.push_back(Triple::x86);
    SupportedArchs.push_back(Triple::x86_64);

    // Some architectures have sub-architectures in which tests will fail, like
    // ARM. These two vectors will define if they do have sub-archs (to avoid
    // extra work for those who don't), and if so, if they are listed to work
    HasSubArchs.push_back(Triple::arm);
    SupportedSubArchs.push_back("armv6");
    SupportedSubArchs.push_back("armv7");

    UnsupportedEnvironments.push_back(Triple::Cygnus);
  }

  void createJIT(std::unique_ptr<Module> M) {

    // Due to the EngineBuilder constructor, it is required to have a Module
    // in order to construct an ExecutionEngine (i.e. MCJIT)
    assert(M != 0 && "a non-null Module must be provided to create MCJIT");

    EngineBuilder EB(std::move(M));
    std::string Error;
    TheJIT.reset(EB.setEngineKind(EngineKind::JIT)
                 .setMCJITMemoryManager(std::move(MM))
                 .setErrorStr(&Error)
                 .setOptLevel(CodeGenOpt::None)
                 .setMArch(MArch)
                 .setMCPU(sys::getHostCPUName())
                 //.setMAttrs(MAttrs)
                 .create());
    // At this point, we cannot modify the module any more.
    assert(TheJIT.get() != NULL && "error creating MCJIT with EngineBuilder");
  }

  CodeGenOpt::Level OptLevel;
  CodeModel::Model CodeModel;
  StringRef MArch;
  SmallVector<std::string, 1> MAttrs;
  std::unique_ptr<ExecutionEngine> TheJIT;
  std::unique_ptr<RTDyldMemoryManager> MM;

  std::unique_ptr<Module> M;
};

} // namespace llvm

#endif // LLVM_UNITTESTS_EXECUTIONENGINE_MCJIT_MCJITTESTBASE_H
