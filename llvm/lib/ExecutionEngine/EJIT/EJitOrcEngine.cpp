//===-- EJitOrcEngine.cpp - OrcJIT Engine Wrapper -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ejit;

struct EJitOrcEngine::Impl {
  std::unique_ptr<orc::LLJIT> J;
  PeriodArrayRegistry *periodReg = nullptr;
  EJitRuntimeState *runtimeState = nullptr;
  const SpecializationContext *activeCtx = nullptr;
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

  // Set up IR transform layer: runs the specialization pipeline during
  // JIT compilation (parameter substitution → InstCombine → Inline →
  // StructFieldPass → standard optimization).
  engine->P->J->getIRTransformLayer().setTransform(
      [engine = engine.get(), &periodReg](
          orc::ThreadSafeModule TSM,
          const orc::MaterializationResponsibility &R)
          -> Expected<orc::ThreadSafeModule> {
        TSM.withModuleDo([engine, &periodReg](Module &M) {
          const SpecializationContext *ctx = engine->P->activeCtx;
          if (!ctx)
            return;

          EJitOptimizer opt(periodReg);

          // 1. Parameter substitution: replace ejit_period_arr_ind args
          opt.preReplacePeriodIndices(M, *ctx);

          // 2. InstCombine: fold constant chains from substituted params
          opt.runInstCombine(M);

          // 3. Inline: expand callees so StructFieldPass can trace GEP chains
          opt.runInline(M);

          // 4. EJitStructFieldPass: replace ejit_may_const loads with
          //    runtime constants (one per function)
          for (Function &F : M.functions()) {
            if (!F.isDeclaration()) {
              EJitStructFieldPass structField(periodReg);
              FunctionAnalysisManager FAM;
              // Register analysis passes needed by the pass manager
              PassBuilder PB;
              PB.registerFunctionAnalyses(FAM);
              structField.run(F, FAM);
            }
          }

          // 5. Run the standard optimization pipeline at the configured level
          opt.runOptimizationPipeline(M, ctx->optLevel);
        });
        return std::move(TSM);
      });

  return engine;
}

Error EJitOrcEngine::loadBitcodeModule(StringRef bitcodeData,
                                       const std::string &funcName) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(bitcodeData, funcName + ".bc");
  auto ModuleOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!ModuleOrErr)
    return ModuleOrErr.takeError();
  return P->J->addIRModule(
      orc::ThreadSafeModule(std::move(*ModuleOrErr), std::move(Ctx)));
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
