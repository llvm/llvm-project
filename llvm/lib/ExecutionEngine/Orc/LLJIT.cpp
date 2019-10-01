//===--------- LLJIT.cpp - An ORC-based JIT for compiling LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"

namespace {

  // A SimpleCompiler that owns its TargetMachine.
  class TMOwningSimpleCompiler : public llvm::orc::SimpleCompiler {
  public:
    TMOwningSimpleCompiler(std::unique_ptr<llvm::TargetMachine> TM)
      : llvm::orc::SimpleCompiler(*TM), TM(std::move(TM)) {}
  private:
    // FIXME: shared because std::functions (and thus
    // IRCompileLayer::CompileFunction) are not moveable.
    std::shared_ptr<llvm::TargetMachine> TM;
  };

} // end anonymous namespace

namespace llvm {
namespace orc {

Error LLJITBuilderState::prepareForConstruction() {

  if (!JTMB) {
    if (auto JTMBOrErr = JITTargetMachineBuilder::detectHost())
      JTMB = std::move(*JTMBOrErr);
    else
      return JTMBOrErr.takeError();
  }

  return Error::success();
}

LLJIT::~LLJIT() {
  if (CompileThreads)
    CompileThreads->wait();
}

Error LLJIT::defineAbsolute(StringRef Name, JITEvaluatedSymbol Sym) {
  auto InternedName = ES->intern(Name);
  SymbolMap Symbols({{InternedName, Sym}});
  return Main.define(absoluteSymbols(std::move(Symbols)));
}

Error LLJIT::addIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err = applyDataLayout(*TSM.getModule()))
    return Err;

  return CompileLayer->add(JD, std::move(TSM), ES->allocateVModule());
}

Error LLJIT::addObjectFile(JITDylib &JD, std::unique_ptr<MemoryBuffer> Obj) {
  assert(Obj && "Can not add null object");

  return ObjLinkingLayer->add(JD, std::move(Obj), ES->allocateVModule());
}

Expected<JITEvaluatedSymbol> LLJIT::lookupLinkerMangled(JITDylib &JD,
                                                        StringRef Name) {
  return ES->lookup(JITDylibSearchList({{&JD, true}}), ES->intern(Name));
}

std::unique_ptr<ObjectLayer>
LLJIT::createObjectLinkingLayer(LLJITBuilderState &S, ExecutionSession &ES) {

  // If the config state provided an ObjectLinkingLayer factory then use it.
  if (S.CreateObjectLinkingLayer)
    return S.CreateObjectLinkingLayer(ES);

  // Otherwise default to creating an RTDyldObjectLinkingLayer that constructs
  // a new SectionMemoryManager for each object.
  auto GetMemMgr = []() { return llvm::make_unique<SectionMemoryManager>(); };
  return llvm::make_unique<RTDyldObjectLinkingLayer>(ES, std::move(GetMemMgr));
}

LLJIT::LLJIT(LLJITBuilderState &S, Error &Err)
    : ES(S.ES ? std::move(S.ES) : llvm::make_unique<ExecutionSession>()),
      Main(this->ES->getMainJITDylib()), DL(""), CtorRunner(Main),
      DtorRunner(Main) {

  ErrorAsOutParameter _(&Err);

  ObjLinkingLayer = createObjectLinkingLayer(S, *ES);

  if (S.NumCompileThreads > 0) {

    // Configure multi-threaded.

    if (auto DLOrErr = S.JTMB->getDefaultDataLayoutForTarget())
      DL = std::move(*DLOrErr);
    else {
      Err = DLOrErr.takeError();
      return;
    }

    {
      auto TmpCompileLayer = llvm::make_unique<IRCompileLayer>(
          *ES, *ObjLinkingLayer, ConcurrentIRCompiler(std::move(*S.JTMB)));

      TmpCompileLayer->setCloneToNewContextOnEmit(true);
      CompileLayer = std::move(TmpCompileLayer);
    }

    CompileThreads = llvm::make_unique<ThreadPool>(S.NumCompileThreads);
    ES->setDispatchMaterialization(
        [this](JITDylib &JD, std::unique_ptr<MaterializationUnit> MU) {
          // FIXME: Switch to move capture once we have c++14.
          auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
          auto Work = [SharedMU, &JD]() { SharedMU->doMaterialize(JD); };
          CompileThreads->async(std::move(Work));
        });
  } else {

    // Configure single-threaded.

    auto TM = S.JTMB->createTargetMachine();
    if (!TM) {
      Err = TM.takeError();
      return;
    }

    DL = (*TM)->createDataLayout();

    CompileLayer = llvm::make_unique<IRCompileLayer>(
        *ES, *ObjLinkingLayer, TMOwningSimpleCompiler(std::move(*TM)));
  }
}

std::string LLJIT::mangle(StringRef UnmangledName) {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, UnmangledName, DL);
  }
  return MangledName;
}

Error LLJIT::applyDataLayout(Module &M) {
  if (M.getDataLayout().isDefault())
    M.setDataLayout(DL);

  if (M.getDataLayout() != DL)
    return make_error<StringError>(
        "Added modules have incompatible data layouts",
        inconvertibleErrorCode());

  return Error::success();
}

void LLJIT::recordCtorDtors(Module &M) {
  CtorRunner.add(getConstructors(M));
  DtorRunner.add(getDestructors(M));
}

Error LLLazyJITBuilderState::prepareForConstruction() {
  if (auto Err = LLJITBuilderState::prepareForConstruction())
    return Err;
  TT = JTMB->getTargetTriple();
  return Error::success();
}

Error LLLazyJIT::addLazyIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err = applyDataLayout(*TSM.getModule()))
    return Err;

  recordCtorDtors(*TSM.getModule());

  return CODLayer->add(JD, std::move(TSM), ES->allocateVModule());
}

LLLazyJIT::LLLazyJIT(LLLazyJITBuilderState &S, Error &Err) : LLJIT(S, Err) {

  // If LLJIT construction failed then bail out.
  if (Err)
    return;

  ErrorAsOutParameter _(&Err);

  /// Take/Create the lazy-compile callthrough manager.
  if (S.LCTMgr)
    LCTMgr = std::move(S.LCTMgr);
  else {
    if (auto LCTMgrOrErr = createLocalLazyCallThroughManager(
            S.TT, *ES, S.LazyCompileFailureAddr))
      LCTMgr = std::move(*LCTMgrOrErr);
    else {
      Err = LCTMgrOrErr.takeError();
      return;
    }
  }

  // Take/Create the indirect stubs manager builder.
  auto ISMBuilder = std::move(S.ISMBuilder);

  // If none was provided, try to build one.
  if (!ISMBuilder)
    ISMBuilder = createLocalIndirectStubsManagerBuilder(S.TT);

  // No luck. Bail out.
  if (!ISMBuilder) {
    Err = make_error<StringError>("Could not construct "
                                  "IndirectStubsManagerBuilder for target " +
                                      S.TT.str(),
                                  inconvertibleErrorCode());
    return;
  }

  // Create the transform layer.
  TransformLayer = llvm::make_unique<IRTransformLayer>(*ES, *CompileLayer);

  // Create the COD layer.
  CODLayer = llvm::make_unique<CompileOnDemandLayer>(
      *ES, *TransformLayer, *LCTMgr, std::move(ISMBuilder));

  if (S.NumCompileThreads > 0)
    CODLayer->setCloneToNewContextOnEmit(true);
}

} // End namespace orc.
} // End namespace llvm.
