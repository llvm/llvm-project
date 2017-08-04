//===- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include "RemoteJITUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetClient.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

class PrototypeAST;
class ExprAST;

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}

  const PrototypeAST& getProto() const;
  const std::string& getName() const;
  llvm::Function *codegen();
};

/// This will compile FnAST to IR, rename the function to add the given
/// suffix (needed to prevent a name-clash with the function's stub),
/// and then take ownership of the module that the function was compiled
/// into.
std::unique_ptr<llvm::Module>
irgenAndTakeOwnership(FunctionAST &FnAST, const std::string &Suffix);

namespace llvm {
namespace orc {

// Typedef the remote-client API.
using MyRemote = remote::OrcRemoteTargetClient<FDRPCChannel>;

class KaleidoscopeJIT {
private:
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

  using OptimizeFunction =
      std::function<std::shared_ptr<Module>(std::shared_ptr<Module>)>;

  IRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;

  JITCompileCallbackManager *CompileCallbackMgr;
  std::unique_ptr<IndirectStubsManager> IndirectStubsMgr;
  MyRemote &Remote;

public:
  using ModuleHandle = decltype(OptimizeLayer)::ModuleHandleT;

  KaleidoscopeJIT(MyRemote &Remote)
      : TM(EngineBuilder().selectTarget(Triple(Remote.getTargetTriple()), "",
                                        "", SmallVector<std::string, 0>())),
        DL(TM->createDataLayout()),
        ObjectLayer([&Remote]() {
            std::unique_ptr<MyRemote::RCMemoryManager> MemMgr;
            if (auto Err = Remote.createRemoteMemoryManager(MemMgr)) {
              logAllUnhandledErrors(std::move(Err), errs(),
                                    "Error creating remote memory manager:");
              exit(1);
            }
            return MemMgr;
          }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        OptimizeLayer(CompileLayer,
                      [this](std::shared_ptr<Module> M) {
                        return optimizeModule(std::move(M));
                      }),
        Remote(Remote) {
    auto CCMgrOrErr = Remote.enableCompileCallbacks(0);
    if (!CCMgrOrErr) {
      logAllUnhandledErrors(CCMgrOrErr.takeError(), errs(),
                            "Error enabling remote compile callbacks:");
      exit(1);
    }
    CompileCallbackMgr = &*CCMgrOrErr;
    std::unique_ptr<MyRemote::RCIndirectStubsManager> ISM;
    if (auto Err = Remote.createIndirectStubsManager(ISM)) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "Error creating indirect stubs manager:");
      exit(1);
    }
    IndirectStubsMgr = std::move(ISM);
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  TargetMachine &getTargetMachine() { return *TM; }

  ModuleHandle addModule(std::unique_ptr<Module> M) {
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    auto Resolver = createLambdaResolver(
        [&](const std::string &Name) {
          if (auto Sym = IndirectStubsMgr->findStub(Name, false))
            return Sym;
          if (auto Sym = OptimizeLayer.findSymbol(Name, false))
            return Sym;
          return JITSymbol(nullptr);
        },
        [&](const std::string &Name) {
          if (auto AddrOrErr = Remote.getSymbolAddress(Name))
            return JITSymbol(*AddrOrErr, JITSymbolFlags::Exported);
          else {
            logAllUnhandledErrors(AddrOrErr.takeError(), errs(),
                                  "Error resolving remote symbol:");
            exit(1);
          }
          return JITSymbol(nullptr);
        });

    // Add the set to the JIT with the resolver we created above and a newly
    // created SectionMemoryManager.
    return cantFail(OptimizeLayer.addModule(std::move(M),
                                            std::move(Resolver)));
  }

  Error addFunctionAST(std::unique_ptr<FunctionAST> FnAST) {
    // Create a CompileCallback - this is the re-entry point into the compiler
    // for functions that haven't been compiled yet.
    auto CCInfo = CompileCallbackMgr->getCompileCallback();

    // Create an indirect stub. This serves as the functions "canonical
    // definition" - an unchanging (constant address) entry point to the
    // function implementation.
    // Initially we point the stub's function-pointer at the compile callback
    // that we just created. In the compile action for the callback (see below)
    // we will update the stub's function pointer to point at the function
    // implementation that we just implemented.
    if (auto Err = IndirectStubsMgr->createStub(mangle(FnAST->getName()),
                                                CCInfo.getAddress(),
                                                JITSymbolFlags::Exported))
      return Err;

    // Move ownership of FnAST to a shared pointer - C++11 lambdas don't support
    // capture-by-move, which is be required for unique_ptr.
    auto SharedFnAST = std::shared_ptr<FunctionAST>(std::move(FnAST));

    // Set the action to compile our AST. This lambda will be run if/when
    // execution hits the compile callback (via the stub).
    //
    // The steps to compile are:
    // (1) IRGen the function.
    // (2) Add the IR module to the JIT to make it executable like any other
    //     module.
    // (3) Use findSymbol to get the address of the compiled function.
    // (4) Update the stub pointer to point at the implementation so that
    ///    subsequent calls go directly to it and bypass the compiler.
    // (5) Return the address of the implementation: this lambda will actually
    //     be run inside an attempted call to the function, and we need to
    //     continue on to the implementation to complete the attempted call.
    //     The JIT runtime (the resolver block) will use the return address of
    //     this function as the address to continue at once it has reset the
    //     CPU state to what it was immediately before the call.
    CCInfo.setCompileAction(
      [this, SharedFnAST]() {
        auto M = irgenAndTakeOwnership(*SharedFnAST, "$impl");
        addModule(std::move(M));
        auto Sym = findSymbol(SharedFnAST->getName() + "$impl");
        assert(Sym && "Couldn't find compiled function?");
        JITTargetAddress SymAddr = cantFail(Sym.getAddress());
        if (auto Err =
              IndirectStubsMgr->updatePointer(mangle(SharedFnAST->getName()),
                                              SymAddr)) {
          logAllUnhandledErrors(std::move(Err), errs(),
                                "Error updating function pointer: ");
          exit(1);
        }

        return SymAddr;
      });

    return Error::success();
  }

  Error executeRemoteExpr(JITTargetAddress ExprAddr) {
    return Remote.callVoidVoid(ExprAddr);
  }

  JITSymbol findSymbol(const std::string Name) {
    return OptimizeLayer.findSymbol(mangle(Name), true);
  }

  void removeModule(ModuleHandle H) {
    cantFail(OptimizeLayer.removeModule(H));
  }

private:
  std::string mangle(const std::string &Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return MangledNameStream.str();
  }

  std::shared_ptr<Module> optimizeModule(std::shared_ptr<Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    return M;
  }
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
