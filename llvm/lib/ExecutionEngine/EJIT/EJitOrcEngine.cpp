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
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"
#include <map>

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-orc-engine"

struct EJitOrcEngine::Impl {
  std::unique_ptr<orc::LLJIT> J;
  PeriodArrayRegistry *periodReg = nullptr;
  EJitRuntimeState *runtimeState = nullptr;
  const SpecializationContext *activeCtx = nullptr;
  /// Per-specialization JITDylib pointers so each specialization is
  /// independently compiled and symbols from different specializations
  /// never conflict.
  std::map<uint32_t, orc::JITDylib *> specDylibs;
  /// User-registered symbols (functions + globals) for bare-metal.
  /// Populated via ejit_register_symbol() / addUserSymbol().
  std::map<std::string, void *> userSymbols;
  /// If non-empty, dump JIT-optimized IR to this directory.
  std::string dumpJITDir;
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
  engine->P->dumpJITDir = config.dumpJITDir;

  // Use compile-time target triple when set (e.g. for ARM embedded),
  // otherwise detect the host architecture.
#ifdef EJIT_DEFAULT_TRIPLE
  auto JTMB = orc::JITTargetMachineBuilder(Triple(EJIT_DEFAULT_TRIPLE));
#else
  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    return JTMB.takeError();

  // Use Large code model so JITLink can generate 64-bit absolute relocations.
  // With the default Small model, Delta32 fixups fail when JIT code (mmap'd
  // at a random high address) references host globals (in the data segment
  // at a distant address), because the offset exceeds ±2 GB.
  JTMB->setCodeModel(CodeModel::Large);
#endif

  orc::LLJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(*JTMB);
  Builder.setNumCompileThreads(0);

  auto J = Builder.create();
  if (!J)
    return J.takeError();

  engine->P->J = std::move(*J);

  // Register all known global variable addresses from the PeriodArrayRegistry
  // so that external global references in any loaded bitcode module resolve
  // to the AOT process's memory. Deduplicate: the constructor may run twice
  // (PASS1 + PASS2 both add to global_ctors), causing duplicate entries.
  {
    auto &JD = engine->P->J->getMainJITDylib();
    orc::SymbolMap symMap;
    for (auto &kv : periodReg.getStaticVars())
      symMap[engine->P->J->mangleAndIntern(kv.varName)] =
          orc::ExecutorSymbolDef(orc::ExecutorAddr::fromPtr(kv.varAddr),
                                 JITSymbolFlags::Exported);
    if (!symMap.empty()) {
      if (auto Err = JD.define(orc::absoluteSymbols(std::move(symMap))))
        LLVM_DEBUG(dbgs() << "ejit-orc-engine: define static vars: "
                          << toString(std::move(Err)) << "\n");
    }
  }

  // Set up IR transform layer: runs the specialization pipeline during
  // JIT compilation (parameter substitution → InstCombine → Inline →
  // StructFieldPass → standard optimization).
  engine->P->J->getIRTransformLayer().setTransform(
      [engine = engine.get(), &periodReg](
          orc::ThreadSafeModule TSM,
          const orc::MaterializationResponsibility &R)
          -> Expected<orc::ThreadSafeModule> {
        TSM.withModuleDo([engine, &periodReg](Module &M) {
          LLVM_DEBUG(dbgs() << "ejit-orc-engine: JIT transform on "
                            << M.getName() << "\n");
          const SpecializationContext *ctx = engine->P->activeCtx;
          if (!ctx)
            return;

          EJitOptimizer opt(periodReg);

          // 1. Parameter substitution: replace ejit_period_arr_ind args
          opt.preReplacePeriodIndices(M, *ctx);

          // 2. InstCombine: fold constant chains from substituted params
          opt.runInstCombine(M);

          // Dump pre-optimization IR if configured.
          if (!engine->P->dumpJITDir.empty() && ctx) {
            std::string path = engine->P->dumpJITDir + "/" +
                               ctx->fnName + "_" +
                               std::to_string(ctx->cacheKey) + "_pre.ll";
            std::error_code EC;
            llvm::raw_fd_ostream OS(path, EC);
            if (!EC)
              M.print(OS, nullptr);
          }

          // 3. First EJitStructFieldPass: replace ejit_may_const loads
          //    before the optimization pipeline so SCCP/ADCE can propagate
          //    the resulting constants.
          {
            EJitStructFieldPass structField(periodReg);
            for (Function &F : M.functions())
              if (!F.isDeclaration())
                structField.run(F, opt.getFAM());
          }

          // 4. Run the standard optimization pipeline at the configured level.
          opt.runOptimizationPipeline(M, ctx->optLevel);

          // 5. Second EJitStructFieldPass + InstCombine: catch loads exposed
          //    after loop unrolling (L3) or inlining.
          {
            EJitStructFieldPass structField(periodReg);
            for (Function &F : M.functions())
              if (!F.isDeclaration())
                structField.run(F, opt.getFAM());
          }
          opt.runInstCombine(M);

          // Dump post-optimization IR if configured.
          if (!engine->P->dumpJITDir.empty() && ctx) {
            std::string path = engine->P->dumpJITDir + "/" +
                               ctx->fnName + "_" +
                               std::to_string(ctx->cacheKey) + "_opt.ll";
            std::error_code EC;
            llvm::raw_fd_ostream OS(path, EC);
            if (!EC)
              M.print(OS, nullptr);
          }
        });
        return std::move(TSM);
      });

  return engine;
}

Error EJitOrcEngine::loadBitcodeModule(StringRef bitcodeData,
                                       uint32_t cacheKey,
                                       const std::string &origFnName) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(
      bitcodeData, ("spec_" + std::to_string(cacheKey) + ".bc"));
  auto ModuleOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!ModuleOrErr)
    return ModuleOrErr.takeError();

  // Collect global variable addresses from the registry for symbols
  // that appear as external declarations in the bitcode module.
  orc::SymbolMap globalSymbols;
  for (GlobalVariable &GV : (*ModuleOrErr)->globals()) {
    if (!GV.isDeclaration() || GV.getName().empty())
      continue;
    void *addr = nullptr;
    if (const auto *info = P->periodReg->getArrayInfo(GV.getName().str()))
      addr = info->baseAddr;
    else
      addr = P->periodReg->getStaticVarAddr(GV.getName().str());
    if (!addr)
      continue;
    globalSymbols[P->J->mangleAndIntern(GV.getName())] =
        orc::ExecutorSymbolDef(orc::ExecutorAddr::fromPtr(addr),
                               JITSymbolFlags::Exported);
  }

  // Each specialization gets its own JITDylib so that symbols from
  // different specializations (same TU bitcode loaded multiple times)
  // never conflict. Remove any stale JD from a previous compilation
  // of the same cacheKey (e.g., after ejit_clear_cache).
  auto it = P->specDylibs.find(cacheKey);
  if (it != P->specDylibs.end()) {
    if (auto Err = P->J->getExecutionSession().removeJITDylib(*it->second))
      LLVM_DEBUG(dbgs() << "ejit-orc-engine: remove stale JD: "
                        << toString(std::move(Err)) << "\n");
    P->specDylibs.erase(it);
  }

  auto JDOrErr = P->J->createJITDylib("spec_" + std::to_string(cacheKey));
  if (!JDOrErr)
    return JDOrErr.takeError();

  // Resolve undefined function symbols from user-registered table.
  // Required for bare-metal where dynamic lookup (dlsym) is unavailable.
  for (Function &F : (*ModuleOrErr)->functions()) {
    if (!F.isDeclaration() || F.getName().empty())
      continue;
    std::string name = F.getName().str();
    if (globalSymbols.count(P->J->mangleAndIntern(name)))
      continue;
    auto it = P->userSymbols.find(name);
    if (it == P->userSymbols.end())
      continue;
    globalSymbols[P->J->mangleAndIntern(name)] =
        orc::ExecutorSymbolDef(orc::ExecutorAddr::fromPtr(it->second),
                               JITSymbolFlags::Exported);
  }
  for (GlobalVariable &GV : (*ModuleOrErr)->globals()) {
    if (!GV.isDeclaration() || GV.getName().empty())
      continue;
    std::string name = GV.getName().str();
    if (globalSymbols.count(P->J->mangleAndIntern(name)))
      continue;
    auto it = P->userSymbols.find(name);
    if (it == P->userSymbols.end())
      continue;
    globalSymbols[P->J->mangleAndIntern(name)] =
        orc::ExecutorSymbolDef(orc::ExecutorAddr::fromPtr(it->second),
                               JITSymbolFlags::Exported);
  }

  // Define all collected symbols in the spec JITDylib before loading the
  // IR module so the JIT linker can resolve external references.
  if (!globalSymbols.empty())
    if (auto Err = JDOrErr->define(
            orc::absoluteSymbols(std::move(globalSymbols))))
      LLVM_DEBUG(dbgs() << "ejit-orc-engine: define globals: "
                        << toString(std::move(Err)) << "\n");

  if (auto Err = P->J->addIRModule(*JDOrErr,
      orc::ThreadSafeModule(std::move(*ModuleOrErr), std::move(Ctx))))
    return Err;

  P->specDylibs[cacheKey] = &*JDOrErr;
  return Error::success();
}

Expected<void *> EJitOrcEngine::lookup(uint32_t cacheKey,
                                       const std::string &name) {
  auto it = P->specDylibs.find(cacheKey);
  if (it == P->specDylibs.end())
    return make_error<StringError>(
        "No specialization JITDylib found for: " + std::to_string(cacheKey),
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

void EJitOrcEngine::addUserSymbol(const std::string &name, void *addr) {
  P->userSymbols[name] = addr;
}
// DEBUG
#include <cstdio>
