//===-- EJitCompileDriver.cpp - Compilation Scheduler ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#ifndef EJIT_FREESTANDING
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#endif
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#ifndef EJIT_FREESTANDING
#include <chrono>
#endif

using namespace llvm;
using namespace llvm::ejit;

#ifdef EJIT_SRE_TASKPOOL
namespace {
/// Adapter so the taskpool can call back into the driver's cold compile path
/// through a plain function pointer (never std::function). The produced JIT
/// pointer still comes from the OrcJIT engine (SRE code pool when enabled).
bool taskpoolCompileThunk(void *ctx, const EJitCompileRequest &req,
                          void **outFn) {
  auto *drv = static_cast<EJitCompileDriver *>(ctx);
  void *fn = drv->compileNow(req.cacheKey);
  *outFn = fn;
  return fn != nullptr;
}
} // namespace
#endif

EJitCompileDriver::EJitCompileDriver(const Config &config,
                                     EJitCache &cache,
                                     EJitRuntimeState &runtimeState,
                                     EJitModuleLoader &loader,
                                     EJitLogger *logger)
    : config_(config), cache_(cache),
  runtimeState_(runtimeState), loader_(loader)
#ifndef EJIT_FREESTANDING
  , logger_(logger)
#endif
{
#ifdef EJIT_SRE_TASKPOOL
  // Build the unified scheduler and point its compile callback at this driver.
  taskPool_ = std::make_unique<EJitTaskPool>();
  taskPool_->setCompiler(&taskpoolCompileThunk, this);
  taskPool_->switchController().setMode(
      config_.compileMode == CompileMode::Async ? EJitCompileMode::Async
                                                : EJitCompileMode::Sync);
#endif
}

EJitCompileDriver::~EJitCompileDriver() = default;

void EJitCompileDriver::setSyncEngine(std::unique_ptr<EJitOrcEngine> engine) {
  syncEngine_ = std::move(engine);
}

void EJitCompileDriver::registerSymbol(const std::string &name, void *addr) {
  if (syncEngine_)
    syncEngine_->addUserSymbol(name, addr);
}

void *EJitCompileDriver::getOrCompile(uint64_t cacheKey) {
#ifdef EJIT_SRE_TASKPOOL
  // When the taskpool is built, all compile_or_get traffic flows through the
  // unified scheduler (cache + dedup + sync/async). It owns its own fixed
  // cache, so the LRU EJitCache hot path below is bypassed in this build.
  if (taskPool_) {
    uint32_t funcIndex = static_cast<uint32_t>(cacheKey >> 32);
    EJitTaskPool::CompileOrGetResult r =
        taskPool_->compileOrGet(funcIndex, cacheKey, /*fallback=*/nullptr);
    return r.fnPtr;
  }
#endif

  // ── Hot path: single hash find ───────────────────────────────────────────
  if (void *cached = cache_.getOrNull(cacheKey)) {
    EJIT_DIAG("cache HIT key=0x%016lx", cacheKey);
    return cached;
  }

  return compileCold(cacheKey, /*storeLru=*/true);
}

void *EJitCompileDriver::compileCold(uint64_t cacheKey, bool storeLru) {
  // ── Cold path: decode cacheKey, verify, compile ────────────────────────
  uint32_t funcIdx = static_cast<uint32_t>(cacheKey >> 32);
  uint8_t dims[4] = {
    static_cast<uint8_t>(cacheKey & 0xFF),
    static_cast<uint8_t>((cacheKey >> 8) & 0xFF),
    static_cast<uint8_t>((cacheKey >> 16) & 0xFF),
    static_cast<uint8_t>((cacheKey >> 24) & 0xFF),
  };

  // Resolve funcName from loader
  const std::string &funcName = loader_.getFuncNameByFuncIdx(funcIdx);
  if (funcName.empty()) {
    EJIT_DIAG("cache MISS key=0x%016lx funcIdx=%u: unknown funcIdx", cacheKey, funcIdx);
    return nullptr;
  }

  EJIT_DIAG("cache MISS key=0x%016lx func=%s dims=[%u,%u,%u,%u]",
           cacheKey, funcName.c_str(), dims[0], dims[1], dims[2], dims[3]);

  // Get bitcode
  auto bitcodeOrErr = loader_.getBitcodeByFuncIdx(funcIdx);
  if (!bitcodeOrErr) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: bitcode not found", cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_BITCODE_NOT_FOUND,
                   "No bitcode for function", funcName,
                   std::to_string(cacheKey));
#endif
    return nullptr;
  }
  StringRef bitcode = *bitcodeOrErr;

  // Resolve period names from cached metadata (parsed once per funcIdx).
  const auto &meta = loader_.getOrCacheFuncMeta(funcIdx);
  const auto &periodNames = meta.periodNames;
  unsigned dimCount = meta.dimCount;

  // Verify time-window state for each dimension.
  for (unsigned i = 0; i < dimCount; ++i) {
    if (!runtimeState_.isActive(periodNames[i], dims[i])) {
      EJIT_DIAG("compile SKIP key=0x%016lx func=%s: period %s[%u] not active",
               cacheKey, funcName.c_str(), periodNames[i].c_str(), dims[i]);
#ifndef EJIT_FREESTANDING
      if (logger_)
        logger_->log(EJIT_ERR_NOT_ACTIVE,
                     "Time window not active for " + periodNames[i],
                     funcName, std::to_string(cacheKey));
#endif
      return nullptr;
    }
  }

  // Build specialization context
  SpecializationContext ctx;
  ctx.fnName = funcName;
  ctx.cacheKey = cacheKey;
  ctx.optLevel = config_.optLevel;
  for (unsigned i = 0; i < dimCount; ++i)
    ctx.dimensions.push_back({periodNames[i], dims[i]});

  if (!syncEngine_) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: no sync engine", cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_NOT_ACTIVE,
                   "Sync engine not initialized", funcName,
                   std::to_string(cacheKey));
#endif
    return nullptr;
  }

  syncEngine_->setActiveContext(&ctx);

  if (auto Err = syncEngine_->loadBitcodeModule(bitcode, cacheKey, funcName)) {
    syncEngine_->setActiveContext(nullptr);
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: load bitcode module failed", cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_COMPILE_FAILED,
                   "Failed to load bitcode module", funcName,
                   std::to_string(cacheKey));
#else
    consumeError(std::move(Err));
#endif
    return nullptr;
  }

  auto addrOrErr = syncEngine_->lookup(cacheKey, funcName);
  syncEngine_->setActiveContext(nullptr);

  if (!addrOrErr) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: lookup after compile failed", cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_COMPILE_FAILED,
                   "Failed to look up compiled function", funcName,
                   std::to_string(cacheKey));
#else
    consumeError(addrOrErr.takeError());
#endif
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  // storeLru is false on the taskpool path: the taskpool publishes to its own
  // fixed cache and the LRU EJitCache is bypassed in that configuration.
  if (storeLru) {
    SmallVector<std::string, 4> periodDeps;
    for (unsigned i = 0; i < dimCount; ++i)
      periodDeps.push_back(periodNames[i] + "=" + std::to_string(dims[i]));

    cache_.put(cacheKey, funcPtr, bitcode.size(), periodDeps);
  }

  EJIT_DIAG("compile OK key=0x%016lx func=%s → pfn=%p", cacheKey, funcName.c_str(), funcPtr);
  return funcPtr;
}
