//===--------- LLJIT.cpp - An ORC-based JIT for compiling LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"

namespace llvm {
namespace orc {

Error LLJITBuilderState::prepareForConstruction() {

  if (!JTMB) {
    if (auto JTMBOrErr = JITTargetMachineBuilder::detectHost())
      JTMB = std::move(*JTMBOrErr);
    else
      return JTMBOrErr.takeError();
  }

  // If the client didn't configure any linker options then auto-configure the
  // JIT linker.
  if (!CreateObjectLinkingLayer && JTMB->getCodeModel() == None &&
      JTMB->getRelocationModel() == None) {

    auto &TT = JTMB->getTargetTriple();
    if (TT.isOSBinFormatMachO() &&
        (TT.getArch() == Triple::aarch64 || TT.getArch() == Triple::x86_64)) {

      JTMB->setRelocationModel(Reloc::PIC_);
      JTMB->setCodeModel(CodeModel::Small);
      CreateObjectLinkingLayer =
          [](ExecutionSession &ES,
             const Triple &) -> std::unique_ptr<ObjectLayer> {
        return std::make_unique<ObjectLinkingLayer>(
            ES, std::make_unique<jitlink::InProcessMemoryManager>());
      };
    }
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

  if (auto Err =
          TSM.withModuleDo([&](Module &M) { return applyDataLayout(M); }))
    return Err;

  return CompileLayer->add(JD, std::move(TSM), ES->allocateVModule());
}

Error LLJIT::addObjectFile(JITDylib &JD, std::unique_ptr<MemoryBuffer> Obj) {
  assert(Obj && "Can not add null object");

  return ObjTransformLayer.add(JD, std::move(Obj), ES->allocateVModule());
}

Expected<JITEvaluatedSymbol> LLJIT::lookupLinkerMangled(JITDylib &JD,
                                                        StringRef Name) {
  return ES->lookup(
      makeJITDylibSearchOrder(&JD, JITDylibLookupFlags::MatchAllSymbols),
      ES->intern(Name));
}

std::unique_ptr<ObjectLayer>
LLJIT::createObjectLinkingLayer(LLJITBuilderState &S, ExecutionSession &ES) {

  // If the config state provided an ObjectLinkingLayer factory then use it.
  if (S.CreateObjectLinkingLayer)
    return S.CreateObjectLinkingLayer(ES, S.JTMB->getTargetTriple());

  // Otherwise default to creating an RTDyldObjectLinkingLayer that constructs
  // a new SectionMemoryManager for each object.
  auto GetMemMgr = []() { return std::make_unique<SectionMemoryManager>(); };
  auto ObjLinkingLayer =
      std::make_unique<RTDyldObjectLinkingLayer>(ES, std::move(GetMemMgr));

  if (S.JTMB->getTargetTriple().isOSBinFormatCOFF()) {
    ObjLinkingLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    ObjLinkingLayer->setAutoClaimResponsibilityForObjectSymbols(true);
  }

  // FIXME: Explicit conversion to std::unique_ptr<ObjectLayer> added to silence
  //        errors from some GCC / libstdc++ bots. Remove this conversion (i.e.
  //        just return ObjLinkingLayer) once those bots are upgraded.
  return std::unique_ptr<ObjectLayer>(std::move(ObjLinkingLayer));
}

Expected<std::unique_ptr<IRCompileLayer::IRCompiler>>
LLJIT::createCompileFunction(LLJITBuilderState &S,
                             JITTargetMachineBuilder JTMB) {

  /// If there is a custom compile function creator set then use it.
  if (S.CreateCompileFunction)
    return S.CreateCompileFunction(std::move(JTMB));

  // Otherwise default to creating a SimpleCompiler, or ConcurrentIRCompiler,
  // depending on the number of threads requested.
  if (S.NumCompileThreads > 0)
    return std::make_unique<ConcurrentIRCompiler>(std::move(JTMB));

  auto TM = JTMB.createTargetMachine();
  if (!TM)
    return TM.takeError();

  return std::make_unique<TMOwningSimpleCompiler>(std::move(*TM));
}

LLJIT::LLJIT(LLJITBuilderState &S, Error &Err)
    : ES(S.ES ? std::move(S.ES) : std::make_unique<ExecutionSession>()),
      Main(this->ES->createJITDylib("<main>")), DL(""),
      ObjLinkingLayer(createObjectLinkingLayer(S, *ES)),
      ObjTransformLayer(*this->ES, *ObjLinkingLayer), CtorRunner(Main),
      DtorRunner(Main) {

  ErrorAsOutParameter _(&Err);

  if (auto DLOrErr = S.JTMB->getDefaultDataLayoutForTarget())
    DL = std::move(*DLOrErr);
  else {
    Err = DLOrErr.takeError();
    return;
  }

  {
    auto CompileFunction = createCompileFunction(S, std::move(*S.JTMB));
    if (!CompileFunction) {
      Err = CompileFunction.takeError();
      return;
    }
    CompileLayer = std::make_unique<IRCompileLayer>(
        *ES, ObjTransformLayer, std::move(*CompileFunction));
  }

  if (S.NumCompileThreads > 0) {
    CompileLayer->setCloneToNewContextOnEmit(true);
    CompileThreads = std::make_unique<ThreadPool>(S.NumCompileThreads);
    ES->setDispatchMaterialization(
        [this](JITDylib &JD, std::unique_ptr<MaterializationUnit> MU) {
          // FIXME: Switch to move capture once we have c++14.
          auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
          auto Work = [SharedMU, &JD]() { SharedMU->doMaterialize(JD); };
          CompileThreads->async(std::move(Work));
        });
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

  if (auto Err = TSM.withModuleDo([&](Module &M) -> Error {
        if (auto Err = applyDataLayout(M))
          return Err;

        recordCtorDtors(M);
        return Error::success();
      }))
    return Err;

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
  TransformLayer = std::make_unique<IRTransformLayer>(*ES, *CompileLayer);

  // Create the COD layer.
  CODLayer = std::make_unique<CompileOnDemandLayer>(
      *ES, *TransformLayer, *LCTMgr, std::move(ISMBuilder));

  if (S.NumCompileThreads > 0)
    CODLayer->setCloneToNewContextOnEmit(true);
}

} // End namespace orc.
} // End namespace llvm.
