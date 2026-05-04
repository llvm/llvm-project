//===-- EJitOrcEngine.cpp - OrcJIT Engine Wrapper -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ejit;

struct EJitOrcEngine::Impl {
  std::unique_ptr<orc::LLJIT> J;
  PeriodArrayRegistry *periodReg = nullptr;
  EJitRuntimeState *runtimeState = nullptr;
  const SpecializationContext *activeCtx = nullptr;
  std::unique_ptr<LLVMContext> ctx;
};

EJitOrcEngine::EJitOrcEngine() : P(std::make_unique<Impl>()) {}
EJitOrcEngine::~EJitOrcEngine() = default;

Expected<std::unique_ptr<EJitOrcEngine>>
EJitOrcEngine::Create(const Config &config,
                      PeriodArrayRegistry &periodReg,
                      EJitRuntimeState &runtimeState) {
  auto engine = std::unique_ptr<EJitOrcEngine>(new EJitOrcEngine());
  engine->P->periodReg = &periodReg;
  engine->P->runtimeState = &runtimeState;

  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    return JTMB.takeError();

  orc::LLJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(*JTMB);
  Builder.setNumCompileThreads(0);

  auto J = Builder.create();
  if (!J)
    return J.takeError();

  engine->P->J = std::move(*J);
  engine->P->ctx = std::make_unique<LLVMContext>();

  return engine;
}

Error EJitOrcEngine::loadBitcodeModule(StringRef bitcodeData,
                                       const std::string &funcName) {
  auto Buf = MemoryBuffer::getMemBuffer(bitcodeData, funcName + ".bc");
  return P->J->addIRModule(
      orc::ThreadSafeModule(std::make_unique<Module>("ejit_" + funcName, *P->ctx),
                            std::make_unique<LLVMContext>()));
}

Expected<void *> EJitOrcEngine::lookup(const std::string &name) {
  auto sym = P->J->lookup(name);
  if (!sym)
    return sym.takeError();
  return reinterpret_cast<void *>(sym->getValue());
}

void EJitOrcEngine::setActiveContext(const SpecializationContext *ctx) {
  P->activeCtx = ctx;
}

const SpecializationContext *EJitOrcEngine::getActiveContext() const {
  return P->activeCtx;
}

EJitJITLinkMemoryManager *EJitOrcEngine::getMemoryManager() const {
  return nullptr; // Using default LLJIT memory manager
}
