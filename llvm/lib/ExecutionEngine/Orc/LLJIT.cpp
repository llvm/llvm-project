//===--------- LLJIT.cpp - An ORC-based JIT for compiling LLVM IR ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
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
    // IRCompileLayer2::CompileFunction) are not moveable.
    std::shared_ptr<llvm::TargetMachine> TM;
  };

} // end anonymous namespace

namespace llvm {
namespace orc {

LLJIT::~LLJIT() {
  if (CompileThreads)
    CompileThreads->wait();
}

Expected<std::unique_ptr<LLJIT>>
LLJIT::Create(JITTargetMachineBuilder JTMB, DataLayout DL,
              unsigned NumCompileThreads) {

  if (NumCompileThreads == 0) {
    // If NumCompileThreads == 0 then create a single-threaded LLJIT instance.
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();
    return std::unique_ptr<LLJIT>(new LLJIT(llvm::make_unique<ExecutionSession>(),
                                            std::move(*TM), std::move(DL)));
  }

  return std::unique_ptr<LLJIT>(new LLJIT(llvm::make_unique<ExecutionSession>(),
                                          std::move(JTMB), std::move(DL),
                                          NumCompileThreads));
}

Error LLJIT::defineAbsolute(StringRef Name, JITEvaluatedSymbol Sym) {
  auto InternedName = ES->getSymbolStringPool().intern(Name);
  SymbolMap Symbols({{InternedName, Sym}});
  return Main.define(absoluteSymbols(std::move(Symbols)));
}

Error LLJIT::addIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err = applyDataLayout(*TSM.getModule()))
    return Err;

  auto K = ES->allocateVModule();
  return CompileLayer.add(JD, K, std::move(TSM));
}

Error LLJIT::addObjectFile(JITDylib &JD, std::unique_ptr<MemoryBuffer> Obj) {
  assert(Obj && "Can not add null object");

  auto K = ES->allocateVModule();
  return ObjLinkingLayer.add(JD, K, std::move(Obj));
}

Expected<JITEvaluatedSymbol> LLJIT::lookupLinkerMangled(JITDylib &JD,
                                                        StringRef Name) {
  return llvm::orc::lookup({&JD}, ES->getSymbolStringPool().intern(Name));
}

LLJIT::LLJIT(std::unique_ptr<ExecutionSession> ES,
             std::unique_ptr<TargetMachine> TM, DataLayout DL)
    : ES(std::move(ES)), Main(this->ES->createJITDylib("main")),
      DL(std::move(DL)),
      ObjLinkingLayer(*this->ES,
                      [this](VModuleKey K) { return getMemoryManager(K); }),
      CompileLayer(*this->ES, ObjLinkingLayer, TMOwningSimpleCompiler(std::move(TM))),
      CtorRunner(Main), DtorRunner(Main) {}

LLJIT::LLJIT(std::unique_ptr<ExecutionSession> ES,
             JITTargetMachineBuilder JTMB, DataLayout DL,
             unsigned NumCompileThreads)
    : ES(std::move(ES)), Main(this->ES->createJITDylib("main")),
      DL(std::move(DL)),
      ObjLinkingLayer(*this->ES,
                      [this](VModuleKey K) { return getMemoryManager(K); }),
      CompileLayer(*this->ES, ObjLinkingLayer, MultiThreadedSimpleCompiler(std::move(JTMB))),
      CtorRunner(Main), DtorRunner(Main) {
  assert(NumCompileThreads != 0 &&
         "Multithreaded LLJIT instance can not be created with 0 threads");

  CompileThreads = llvm::make_unique<ThreadPool>(NumCompileThreads);
  this->ES->setDispatchMaterialization([this](JITDylib &JD, std::unique_ptr<MaterializationUnit> MU) {
      // FIXME: Switch to move capture once we have c++14.
      auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
      auto Work = [SharedMU, &JD]() {
        SharedMU->doMaterialize(JD);
      };
      CompileThreads->async(std::move(Work));
    });
}

std::unique_ptr<RuntimeDyld::MemoryManager>
LLJIT::getMemoryManager(VModuleKey K) {
  return llvm::make_unique<SectionMemoryManager>();
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

Expected<std::unique_ptr<LLLazyJIT>>
  LLLazyJIT::Create(JITTargetMachineBuilder JTMB, DataLayout DL,
                    unsigned NumCompileThreads) {
  auto ES = llvm::make_unique<ExecutionSession>();

  const Triple &TT = JTMB.getTargetTriple();

  auto CCMgr = createLocalCompileCallbackManager(TT, *ES, 0);
  if (!CCMgr)
    return CCMgr.takeError();

  auto ISMBuilder = createLocalIndirectStubsManagerBuilder(TT);
  if (!ISMBuilder)
    return make_error<StringError>(
        std::string("No indirect stubs manager builder for ") + TT.str(),
        inconvertibleErrorCode());

  if (NumCompileThreads == 0) {
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();
    return std::unique_ptr<LLLazyJIT>(
        new LLLazyJIT(std::move(ES), std::move(*TM), std::move(DL),
                      std::move(*CCMgr), std::move(ISMBuilder)));
  }

  return std::unique_ptr<LLLazyJIT>(new LLLazyJIT(
      std::move(ES), std::move(JTMB), std::move(DL), NumCompileThreads,
      std::move(*CCMgr), std::move(ISMBuilder)));
}

Error LLLazyJIT::addLazyIRModule(JITDylib &JD, ThreadSafeModule TSM) {
  assert(TSM && "Can not add null module");

  if (auto Err = applyDataLayout(*TSM.getModule()))
    return Err;

  makeAllSymbolsExternallyAccessible(*TSM.getModule());

  recordCtorDtors(*TSM.getModule());

  auto K = ES->allocateVModule();
  return CODLayer.add(JD, K, std::move(TSM));
}

LLLazyJIT::LLLazyJIT(
    std::unique_ptr<ExecutionSession> ES, std::unique_ptr<TargetMachine> TM,
    DataLayout DL, std::unique_ptr<JITCompileCallbackManager> CCMgr,
    std::function<std::unique_ptr<IndirectStubsManager>()> ISMBuilder)
    : LLJIT(std::move(ES), std::move(TM), std::move(DL)),
      CCMgr(std::move(CCMgr)), TransformLayer(*this->ES, CompileLayer),
      CODLayer(*this->ES, TransformLayer, *this->CCMgr, std::move(ISMBuilder)) {
}

LLLazyJIT::LLLazyJIT(
    std::unique_ptr<ExecutionSession> ES, JITTargetMachineBuilder JTMB,
    DataLayout DL, unsigned NumCompileThreads, std::unique_ptr<JITCompileCallbackManager> CCMgr,
    std::function<std::unique_ptr<IndirectStubsManager>()> ISMBuilder)
    : LLJIT(std::move(ES), std::move(JTMB), std::move(DL), NumCompileThreads),
      CCMgr(std::move(CCMgr)), TransformLayer(*this->ES, CompileLayer),
      CODLayer(*this->ES, TransformLayer, *this->CCMgr, std::move(ISMBuilder)) {
}

} // End namespace orc.
} // End namespace llvm.
