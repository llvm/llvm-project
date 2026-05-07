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
#include "llvm/TargetParser/Triple.h"
#include <map>

using namespace llvm;
using namespace llvm::ejit;

struct EJitOrcEngine::Impl {
  std::unique_ptr<orc::LLJIT> J;
  PeriodArrayRegistry *periodReg = nullptr;
  EJitRuntimeState *runtimeState = nullptr;
  const SpecializationContext *activeCtx = nullptr;
  /// Per-function resource trackers so re-specializing the same function
  /// removes the old module (avoiding symbol conflicts), while different
  /// functions can coexist without invalidating each other's code.
  std::map<std::string, orc::ResourceTrackerSP> funcTrackers;
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

  // Use compile-time target triple when set (e.g. for ARM embedded),
  // otherwise detect the host architecture.
#ifdef EJIT_DEFAULT_TRIPLE
  auto JTMB = orc::JITTargetMachineBuilder(Triple(EJIT_DEFAULT_TRIPLE));
#else
  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    return JTMB.takeError();
#endif

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

          // 4. First EJitStructFieldPass: replace ejit_may_const loads
          //    before the optimization pipeline so SCCP/ADCE can propagate
          //    the resulting constants.
          for (Function &F : M.functions()) {
            if (!F.isDeclaration()) {
              EJitStructFieldPass structField(periodReg);
              FunctionAnalysisManager FAM;
              PassBuilder PB;
              PB.registerFunctionAnalyses(FAM);
              structField.run(F, FAM);
            }
          }

          // 5. Run the standard optimization pipeline at the configured level.
          //    L3 LoopFullUnroll may expose new constant-index GEP chains.
          opt.runOptimizationPipeline(M, ctx->optLevel);

          // 6. Second EJitStructFieldPass + InstCombine: catch loads that
          //    became constant-indexed after loop unrolling (L3) or inlining.
          for (Function &F : M.functions()) {
            if (!F.isDeclaration()) {
              EJitStructFieldPass structField(periodReg);
              FunctionAnalysisManager FAM;
              PassBuilder PB;
              PB.registerFunctionAnalyses(FAM);
              structField.run(F, FAM);
            }
          }
          opt.runInstCombine(M);
        });
        return std::move(TSM);
      });

  return engine;
}

Error EJitOrcEngine::loadBitcodeModule(StringRef bitcodeData,
                                       const std::string &moduleId,
                                       const std::string &origFnName) {
  // Remove the previous specialization for the same original function
  // so symbols don't conflict. Different functions (different origFnName)
  // have independent trackers and can coexist.
  auto it = P->funcTrackers.find(origFnName);
  if (it != P->funcTrackers.end()) {
    if (auto Err = it->second->remove())
      return Err;
    P->funcTrackers.erase(it);
  }

  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(bitcodeData, moduleId + ".bc");
  auto ModuleOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!ModuleOrErr)
    return ModuleOrErr.takeError();

  auto RT = P->J->getMainJITDylib().createResourceTracker();
  if (auto Err = P->J->addIRModule(RT,
      orc::ThreadSafeModule(std::move(*ModuleOrErr), std::move(Ctx))))
    return Err;

  P->funcTrackers[origFnName] = std::move(RT);
  return Error::success();
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
