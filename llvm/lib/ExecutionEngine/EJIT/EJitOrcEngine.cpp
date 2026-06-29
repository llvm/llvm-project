//===-- EJitOrcEngine.cpp - OrcJIT Engine Wrapper -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
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

#ifdef EJIT_SRE_CODE_POOL
#include "llvm/ExecutionEngine/EJIT/EJitCodePoolMemoryManager.h"
#include "llvm/ExecutionEngine/EJIT/EJitSrePlatform.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#endif

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-orc-engine"

struct EJitOrcEngine::Impl {
#ifdef EJIT_SRE_CODE_POOL
  /// Dedicated 2MiB code pools backing all JIT machine code. Declared before
  /// J so it outlives the LLJIT (and the memory manager the object linking
  /// layer owns, which references it).
  std::unique_ptr<EJitCodePoolManager> codePool;
#endif
  std::unique_ptr<orc::LLJIT> J;
  PeriodArrayRegistry *periodReg = nullptr;
  EJitRuntimeState *runtimeState = nullptr;
  const SpecializationContext *activeCtx = nullptr;
  /// Per-specialization JITDylib pointers so each specialization is
  /// independently compiled and symbols from different specializations
  /// never conflict.
  std::map<uint64_t, orc::JITDylib *> specDylibs;
  /// User-registered symbols (functions + globals) for bare-metal.
  /// Populated via ejit_register_symbol() / addUserSymbol().
  std::map<std::string, void *> userSymbols;
  /// If non-empty, dump JIT-optimized IR to this directory.
  std::string dumpJITDir;
  /// Persistent optimizer — analysis managers are registered once and reused.
  std::unique_ptr<EJitOptimizer> optimizer;
};

EJitOrcEngine::EJitOrcEngine() : P(std::make_unique<Impl>()) {}
EJitOrcEngine::~EJitOrcEngine() = default;

Expected<std::unique_ptr<EJitOrcEngine>>
EJitOrcEngine::Create(const Config &config,
                      PeriodArrayRegistry &periodReg,
                      EJitRuntimeState &runtimeState) {
  EJIT_DIAG("create: opt=%d dump=%s",
            static_cast<int>(config.optLevel),
            config.dumpJITDir.empty() ? "(off)" : config.dumpJITDir.c_str());
  auto engine = std::unique_ptr<EJitOrcEngine>(new EJitOrcEngine());
  engine->P->periodReg = &periodReg;
  engine->P->runtimeState = &runtimeState;
  engine->P->dumpJITDir = config.dumpJITDir;

  // Bare-metal / cross-compiled: use compile-time target triple.
  // Native host: auto-detect via detectHost().
#if defined(EJIT_DEFAULT_TRIPLE) || defined(EJIT_FREESTANDING)
  #ifdef EJIT_DEFAULT_TRIPLE
    Expected<orc::JITTargetMachineBuilder> JTMBOrErr(
        orc::JITTargetMachineBuilder(Triple(EJIT_DEFAULT_TRIPLE)));
  #else
    #error EJIT_FREESTANDING requires EJIT_DEFAULT_TRIPLE to be set
  #endif
#else
  auto JTMBOrErr = orc::JITTargetMachineBuilder::detectHost();
#endif
  if (!JTMBOrErr) {
    EJIT_DIAG("create FAIL: target machine builder error");
    return JTMBOrErr.takeError();
  }

  // Use Large code model so JITLink can generate 64-bit absolute relocations.
  JTMBOrErr->setCodeModel(CodeModel::Large);

  orc::LLJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(*JTMBOrErr);
  Builder.setNumCompileThreads(0);
// Bare-metal: skip host process symbol search (avoids dlopen/dlsym),
// and skip ORC runtime injection / EH frames / atexit / global ctors.
#ifdef EJIT_FREESTANDING
  Builder.setLinkProcessSymbolsByDefault(false);
  Builder.setPlatformSetUp(orc::setUpInactivePlatform);
#endif

#ifdef EJIT_SRE_CODE_POOL
  // Route JIT machine-code memory through EmbeddedJIT's own 2MiB pools instead
  // of the default JITLink mmap/mprotect path. The pool manager is owned by the
  // engine (so it outlives the LLJIT); the object linking layer owns a memory
  // manager that references it. Pages are kept RW here and sealed to RX later,
  // at lookup time, by the pool manager's enable_ex sealing.
  engine->P->codePool = makeSreCodePoolManager();
  {
    EJitCodePoolManager *Pool = engine->P->codePool.get();
    Builder.setObjectLinkingLayerCreator(
        [Pool](orc::ExecutionSession &ES)
            -> Expected<std::unique_ptr<orc::ObjectLayer>> {
          // Page size only affects per-segment layout padding; we never apply
          // per-segment protections (sealing is done per 2MiB pool), so a
          // conservative 4KiB is sufficient and portable.
          constexpr size_t JitPageSize = 4096;
          return std::make_unique<orc::ObjectLinkingLayer>(
              ES, std::make_unique<EJitCodePoolMemoryManager>(*Pool,
                                                              JitPageSize));
        });
  }
#endif

  auto J = Builder.create();
  if (!J) {
    EJIT_DIAG("create FAIL: LLJIT builder error");
    return J.takeError();
  }

  engine->P->J = std::move(*J);

  // Create persistent optimizer — analysis managers are registered once here
  // and reused across compilations (cleared between runs).
  engine->P->optimizer = std::make_unique<EJitOptimizer>(periodReg);

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
      size_t n = symMap.size();
      if (auto Err = JD.define(orc::absoluteSymbols(std::move(symMap))))
        EJIT_DIAG("create: define %zu static var(s) FAILED: %s", n,
                  toString(std::move(Err)).c_str());
    }
  }
  EJIT_DIAG("create: static vars registered=%zu",
            periodReg.getStaticVars().size());

  // Set up IR transform layer: runs the specialization pipeline during
  // JIT compilation (parameter substitution → InstCombine → StructFieldPass
  // → core optimization pipeline).
  engine->P->J->getIRTransformLayer().setTransform(
      [engine = engine.get()](
          orc::ThreadSafeModule TSM,
          const orc::MaterializationResponsibility &R)
          -> Expected<orc::ThreadSafeModule> {
        TSM.withModuleDo([engine](Module &M) {
          LLVM_DEBUG(dbgs() << "ejit-orc-engine: JIT transform on "
                            << M.getName() << "\n");
          const SpecializationContext *ctx = engine->P->activeCtx;
          if (!ctx)
            return;

          // Clear stale analysis results from previous compilations
          // (each compilation uses a fresh Module with new IR unit pointers).
          engine->P->optimizer->clearAnalyses();

          // Dump pre-optimization IR (before the JIT pipeline runs).
          if (!engine->P->dumpJITDir.empty()) {
            std::string prePath = engine->P->dumpJITDir + "/" +
                                  ctx->fnName + "_" +
                                  std::to_string(ctx->cacheKey) + "_pre.ll";
            std::error_code EC;
            llvm::raw_fd_ostream preOS(prePath, EC);
            if (!EC)
              M.print(preOS, nullptr);
          }

          engine->P->optimizer->runPipeline(M, *ctx);

          // Dump post-optimization IR.
          if (!engine->P->dumpJITDir.empty()) {
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

  EJIT_DIAG("create OK: LLJIT ready");
  return engine;
}

Error EJitOrcEngine::loadBitcodeModule(StringRef bitcodeData,
                                       uint64_t cacheKey,
                                       const std::string &origFnName) {
  EJIT_DIAG("loadBitcode key=0x%016lx func=%s size=%zu", cacheKey,
            origFnName.c_str(), bitcodeData.size());
  auto Ctx = std::make_unique<LLVMContext>();
  auto Buf = MemoryBuffer::getMemBuffer(
      bitcodeData, ("spec_" + std::to_string(cacheKey) + ".bc"));
  auto ModuleOrErr = parseBitcodeFile(Buf->getMemBufferRef(), *Ctx);
  if (!ModuleOrErr) {
    EJIT_DIAG("loadBitcode FAIL key=0x%016lx: parse bitcode error", cacheKey);
    return ModuleOrErr.takeError();
  }

  Triple TT((*ModuleOrErr)->getTargetTriple());
  if (TT.isAArch64() && TT.isOSBinFormatELF()) {
    // These declarations resolve to process addresses, not co-located JIT
    // storage. Clearing dso_local forces AArch64 PIC codegen to use GOT/PLT
    // style indirection instead of near-page ADRP relocations.
    for (Function &F : (*ModuleOrErr)->functions()) {
      if (F.isDeclaration() && !F.isIntrinsic())
        F.setDSOLocal(false);
    }
    for (GlobalVariable &GV : (*ModuleOrErr)->globals()) {
      if (GV.isDeclaration())
        GV.setDSOLocal(false);
    }
  }

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
      EJIT_DIAG("loadBitcode key=0x%016lx: remove stale JD FAILED: %s",
                cacheKey, toString(std::move(Err)).c_str());
    P->specDylibs.erase(it);
  }

  auto JDOrErr = P->J->createJITDylib("spec_" + std::to_string(cacheKey));
  if (!JDOrErr) {
    EJIT_DIAG("loadBitcode FAIL key=0x%016lx: create JITDylib error", cacheKey);
    return JDOrErr.takeError();
  }

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
  if (!globalSymbols.empty()) {
    size_t nGlobals = globalSymbols.size();
    if (auto Err = JDOrErr->define(
            orc::absoluteSymbols(std::move(globalSymbols))))
      EJIT_DIAG("loadBitcode key=0x%016lx: define %zu global(s) FAILED: %s",
                cacheKey, nGlobals, toString(std::move(Err)).c_str());
  }

  if (auto Err = P->J->addIRModule(*JDOrErr,
      orc::ThreadSafeModule(std::move(*ModuleOrErr), std::move(Ctx)))) {
    EJIT_DIAG("loadBitcode FAIL key=0x%016lx: add IR module error", cacheKey);
    return Err;
  }

  P->specDylibs[cacheKey] = &*JDOrErr;
  EJIT_DIAG("loadBitcode OK key=0x%016lx func=%s", cacheKey, origFnName.c_str());
  return Error::success();
}

Expected<void *> EJitOrcEngine::lookup(uint64_t cacheKey,
                                       const std::string &name) {
  auto it = P->specDylibs.find(cacheKey);
  if (it == P->specDylibs.end()) {
    EJIT_DIAG("lookup FAIL key=0x%016lx name=%s: no spec JITDylib", cacheKey,
              name.c_str());
    return make_error<StringError>(
        "No specialization JITDylib found for: " + std::to_string(cacheKey),
        inconvertibleErrorCode());
  }

  auto addr = P->J->lookup(*it->second, name);
  if (!addr) {
    EJIT_DIAG("lookup FAIL key=0x%016lx name=%s: symbol not found", cacheKey,
              name.c_str());
    return addr.takeError();
  }
  void *Ptr = reinterpret_cast<void *>(addr->getValue());

#ifdef EJIT_SRE_CODE_POOL
  // Legacy whole-pool seal: flip the 2MiB pool that contains the resolved
  // function to RX before it is handed back. This is the only point a JIT pool
  // transitions RW->RX in whole-pool mode. Idempotent: a pool already sealed
  // (e.g. on allocation rollover) is not re-flipped, so repeated lookups of the
  // same function do not re-invoke enable_ex. Only pool-backed code is sealed;
  // an address resolved outside the pools (e.g. a process/absolute symbol) is
  // left untouched. If sealing fails we must not return a callable pointer.
  //
  // In 4K page-seal mode the seal already happened per-page at finalize (in the
  // code-pool memory manager), so nothing is done here.
  if (P->codePool && !P->codePool->usesPageSeal() &&
      P->codePool->contains(Ptr)) {
    if (auto Err = P->codePool->sealPoolContaining(Ptr)) {
      EJIT_DIAG("lookup FAIL key=0x%016lx ptr=%p: seal pool error", cacheKey,
                Ptr);
      return std::move(Err);
    }
  }
#endif

  EJIT_DIAG("lookup OK key=0x%016lx name=%s ptr=%p", cacheKey, name.c_str(),
            Ptr);
  return Ptr;
}

void EJitOrcEngine::setActiveContext(const SpecializationContext *ctx) {
  P->activeCtx = ctx;
}

const SpecializationContext *EJitOrcEngine::getActiveContext() const {
  return P->activeCtx;
}


void EJitOrcEngine::addUserSymbol(const std::string &name, void *addr) {
  P->userSymbols[name] = addr;
}

#ifdef EJIT_SRE_CODE_POOL
EJitCodePoolManager::Stats EJitOrcEngine::getCodePoolStats() const {
  if (P->codePool)
    return P->codePool->getStats();
  return EJitCodePoolManager::Stats{};
}
#endif
