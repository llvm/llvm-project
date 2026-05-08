//===-- EJitOrcEngine.cpp - OrcJIT Engine Wrapper -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
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
  /// Per-specialization JITDylib pointers so each specialization is
  /// independently compiled and symbols from different specializations
  /// never conflict.
  std::map<std::string, orc::JITDylib *> specDylibs;
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
                                       const std::string &cacheKey,
                                       const std::string &origFnName) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(bitcodeData, cacheKey + ".bc");
  auto ModuleOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!ModuleOrErr)
    return ModuleOrErr.takeError();

  // Define global variable addresses from the PeriodArrayRegistry so that
  // external global references in the bitcode resolve to AOT process memory.
  {
    auto &JD = P->J->getMainJITDylib();
    for (GlobalVariable &GV : (*ModuleOrErr)->globals()) {
      if (!GV.isDeclaration() || GV.getName().empty())
        continue;
      void *addr = nullptr;
      // Try period array lookup by var name
      if (const auto *info = P->periodReg->getArrayInfo(GV.getName().str()))
        addr = info->baseAddr;
      else
        addr = P->periodReg->getStaticVarAddr(GV.getName().str());
      if (!addr)
        continue;
      orc::SymbolMap symMap;
      symMap[P->J->mangleAndIntern(GV.getName())] =
          orc::ExecutorSymbolDef(orc::ExecutorAddr::fromPtr(addr),
                                 JITSymbolFlags::Exported);
      (void)JD.define(orc::absoluteSymbols(std::move(symMap)));
    }
  }

  // Each specialization gets its own JITDylib so that symbols from
  // different specializations (same TU bitcode loaded multiple times)
  // never conflict. Helper functions and globals coexist independently.
  auto JDOrErr = P->J->createJITDylib("spec_" + cacheKey);
  if (!JDOrErr)
    return JDOrErr.takeError();

  if (auto Err = P->J->addIRModule(*JDOrErr,
      orc::ThreadSafeModule(std::move(*ModuleOrErr), std::move(Ctx))))
    return Err;

  P->specDylibs[cacheKey] = &*JDOrErr;
  return Error::success();
}

Expected<void *> EJitOrcEngine::lookup(const std::string &cacheKey,
                                       const std::string &name) {
  auto it = P->specDylibs.find(cacheKey);
  if (it == P->specDylibs.end())
    return make_error<StringError>(
        "No specialization JITDylib found for: " + cacheKey,
        inconvertibleErrorCode());

  auto addr = P->J->lookup(*it->second, name);
  if (!addr)
    return addr.takeError();
  return reinterpret_cast<void *>(addr->getValue());
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
