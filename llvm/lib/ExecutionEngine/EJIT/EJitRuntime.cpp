//===-- EJitRuntime.cpp - EmbeddedJIT C Runtime API -----------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#ifdef EJIT_SRE_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#endif

using namespace llvm;
using namespace llvm::ejit;

static EJit *gEJIT = nullptr;

static void parseConfig(const ejit_config_t *src, Config &dst) {
  if (!src)
    return;
  dst.compileMode = (src->compileMode == EJIT_COMPILE_ASYNC)
                        ? CompileMode::Async
                        : CompileMode::Sync;
  dst.optLevel = static_cast<OptimizationLevel>(src->optLevel);
  if (src->maxCodeMemory > 0)
    dst.maxCodeMemory = src->maxCodeMemory;
  if (src->maxDataMemory > 0)
    dst.maxDataMemory = src->maxDataMemory;
  if (src->maxCacheEntries > 0)
    dst.maxCacheEntries = src->maxCacheEntries;
  if (src->maxCacheSize > 0)
    dst.maxCacheSize = src->maxCacheSize;
  dst.enableLogger = src->enableLogger;
  dst.forceStaticRegistry = src->forceStaticRegistry;
  if (src->dumpJITDir && src->dumpJITDir[0])
    dst.dumpJITDir = src->dumpJITDir;
#ifdef EJIT_FREESTANDING
  dst.enableLogger = false;
#endif
}

extern "C" {

ejit_status_t ejit_init(const ejit_config_t *config) {
  if (gEJIT) {
    EJIT_DIAG("already initialized, returning OK");
    return EJIT_OK;
  }

  Config cfg;
  parseConfig(config, cfg);

  gEJIT = new (std::nothrow) EJit(cfg);
  if (!gEJIT) {
    EJIT_DIAG("failed: out of memory");
    return EJIT_ERR_MEMORY;
  }

  EJIT_DIAG("initialized: mode=%d opt=%d cache=%zu entries=%u",
            (int)cfg.compileMode, (int)cfg.optLevel, cfg.maxCacheSize,
            (unsigned)cfg.maxCacheEntries);
  return EJIT_OK;
}

void ejit_shutdown(void) {
  EJIT_DIAG("shutting down");
  delete gEJIT;
  gEJIT = nullptr;
  EJIT_DIAG("shutdown complete");
}

void ejit_register_symbol(const char *name, void *addr) {
  if (gEJIT) {
    gEJIT->registerSymbol(name, addr);
  } else {
    // Constructor-phase call (before ejit_init): stage for later consumption.
    EJitRegistrationStore::instance().registerSymbol(name, addr);
  }
}

void ejit_register_bitcode(const char *funcName,
                           const uint8_t *bitcodeData, uint64_t bitcodeSize) {
  if (gEJIT) {
    gEJIT->registerBitcode(funcName, bitcodeData,
                           static_cast<size_t>(bitcodeSize));
  } else {
    EJitRegistrationStore::instance().registerBitcode(
        funcName, bitcodeData, static_cast<size_t>(bitcodeSize));
  }
}

void ejit_register_period_array(const char *periodName,
                                const char *varName,
                                void *baseAddr, uint64_t arraySize) {
  if (gEJIT) {
    gEJIT->registerPeriodArray(periodName, varName, baseAddr, arraySize);
  } else {
    EJitRegistrationStore::instance().registerPeriodArray(
        periodName, varName, baseAddr, arraySize);
  }
}

void ejit_register_static_var(const char *varName, void *varAddr) {
  if (gEJIT) {
    gEJIT->registerStaticVar(varName, varAddr);
  } else {
    EJitRegistrationStore::instance().registerStaticVar(varName, varAddr);
  }
}

ejit_status_t ejit_activate(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT) {
    EJIT_DIAG("activate(%s,%u) failed: not initialized", periodName, cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  EJIT_DIAG("activate(%s,%u)", periodName, cellIdx);
  gEJIT->activate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT) {
    EJIT_DIAG("deactivate(%s,%u) failed: not initialized", periodName, cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  EJIT_DIAG("deactivate(%s,%u)", periodName, cellIdx);
  gEJIT->deactivate(periodName, cellIdx);
  gEJIT->invalidateByPeriod(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_array(const char *periodName, void *arrayPtr,
                                   uint8_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  // Validate that arrayPtr is a registered period array.
  auto *info = gEJIT->getRegistry().getArrayByBaseAddr(arrayPtr);
  if (!info)
    return EJIT_ERR_INVALID_PARAM;
  if (info->periodName != periodName)
    return EJIT_ERR_INVALID_PARAM;
  // Array-level activation: only this specific array's instance.
  gEJIT->activateArray(arrayPtr, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_array(const char *periodName, void *arrayPtr,
                                     uint8_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  // Validate that arrayPtr is a registered period array.
  auto *info = gEJIT->getRegistry().getArrayByBaseAddr(arrayPtr);
  if (!info)
    return EJIT_ERR_INVALID_PARAM;
  if (info->periodName != periodName)
    return EJIT_ERR_INVALID_PARAM;
  // Array-level deactivation: only this specific array's instance.
  gEJIT->deactivateArray(arrayPtr, cellIdx);
  gEJIT->invalidateByPeriod(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_all(const char *periodName) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  gEJIT->activateAll(periodName);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_all(const char *periodName) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  gEJIT->deactivateAll(periodName);
  gEJIT->invalidateAllByPeriod(periodName);
  return EJIT_OK;
}

bool ejit_is_active(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT)
    return false;
  return gEJIT->isActive(periodName, cellIdx);
}

void *ejit_compile_or_get(uint64_t cacheKey, void **out_pfn) {
  if (!gEJIT) {
    EJIT_DIAG("compile_or_get(key=0x%016lx) failed: not initialized", cacheKey);
    return nullptr;
  }

  void *result = gEJIT->getOrCompile(cacheKey);
  if (out_pfn)
    *out_pfn = result;

  EJIT_DIAG("compile_or_get(key=0x%016lx) → %s", cacheKey,
           result ? "JIT" : "NULL");
  return result;
}

void ejit_clear_cache(void) {
  EJIT_DIAG("clear_cache");
  if (gEJIT)
    gEJIT->clearCache();
}

void ejit_invalidate(const char *periodName, uint8_t cellIdx) {
  EJIT_DIAG("invalidate(%s,%u)", periodName, cellIdx);
  if (gEJIT)
    gEJIT->invalidateByPeriod(periodName, cellIdx);
}

ejit_status_t ejit_get_stats(ejit_stats_t *stats) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  if (!stats)
    return EJIT_ERR_INVALID_PARAM;

  auto s = gEJIT->getStats();
  stats->entryCount = s.entryCount;
  stats->totalCodeSize = s.totalCodeSize;
  stats->maxSize = s.maxSize;
  stats->hits = s.hits;
  stats->misses = s.misses;
  stats->evictions = s.evictions;
  return EJIT_OK;
}

const ejit_error_t *ejit_get_last_error(void) {
  if (!gEJIT)
    return nullptr;
  static ejit_error_t err;
  const EJitError *last = gEJIT->getLastError();
  if (!last)
    return nullptr;
  err.code = last->code;
  snprintf(err.message, sizeof(err.message), "%s", last->message.c_str());
  snprintf(err.funcName, sizeof(err.funcName), "%s", last->funcName.c_str());
  return &err;
}

void ejit_set_compile_mode(ejit_compile_mode_t mode) {
  if (gEJIT)
    gEJIT->setCompileMode(mode == EJIT_COMPILE_ASYNC ? CompileMode::Async
                                                      : CompileMode::Sync);
}

ejit_compile_mode_t ejit_get_compile_mode(void) {
  if (!gEJIT)
    return EJIT_COMPILE_SYNC;
  return gEJIT->getCompileMode() == CompileMode::Async ? EJIT_COMPILE_ASYNC
                                                        : EJIT_COMPILE_SYNC;
}

#ifdef EJIT_SRE_TASKPOOL
//===-- SRE taskpool black-box API ----------------------------------------===//

static ejit_status_t taskpoolStatus(EJitCompileOrGetStatus s) {
  switch (s) {
  case EJitCompileOrGetStatus::CacheHit:
  case EJitCompileOrGetStatus::SyncCompiled:
    return EJIT_OK;
  case EJitCompileOrGetStatus::EnqueuedPending:
  case EJitCompileOrGetStatus::AlreadyPending:
    return EJIT_PENDING;
  case EJitCompileOrGetStatus::QueueFullFallback:
    return EJIT_ERR_QUEUE_FULL;
  case EJitCompileOrGetStatus::DedupFullFallback:
    return EJIT_ERR_DEDUP_FULL;
  case EJitCompileOrGetStatus::CacheFullFallback:
    return EJIT_ERR_CACHE_FULL;
  case EJitCompileOrGetStatus::DisabledFallback:
    return EJIT_ERR_DISABLED;
  case EJitCompileOrGetStatus::CompileFailed:
  default:
    return EJIT_ERR_COMPILE_FAILED;
  }
}

ejit_status_t ejit_taskpool_sync_compile(uint32_t funcIndex, uint64_t cacheKey,
                                         void **outFn) {
  if (outFn)
    *outFn = nullptr;
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return EJIT_ERR_NOT_ACTIVE;

  EJitCompileRequest req;
  req.funcIndex = funcIndex;
  req.version = tp->switchController().getVersion();
  req.cacheKey = cacheKey;
  req.fallbackPtr = 0;
  req.userData = 0;
  EJitTaskPool::CompileOrGetResult r = tp->syncCompile(req);
  if (outFn)
    *outFn = r.fnPtr;
  EJIT_DIAG("taskpool_sync_compile(func=%u key=0x%016lx) status=%u", funcIndex,
            cacheKey, (unsigned)r.status);
  return taskpoolStatus(r.status);
}

ejit_status_t ejit_taskpool_free_code(uint32_t funcIndex, uint64_t cacheKey) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return EJIT_ERR_NOT_ACTIVE;
  tp->freeCode(funcIndex, cacheKey);
  return EJIT_OK;
}

unsigned ejit_taskpool_poll_one(void) {
  if (!gEJIT)
    return 0;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return 0;
  return tp->pollOne() ? 1u : 0u;
}

unsigned ejit_taskpool_poll_budget(unsigned maxItems) {
  if (!gEJIT)
    return 0;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return 0;
  return tp->pollBudget(maxItems);
}

unsigned ejit_taskpool_worker_step(void) {
  return ejit_taskpool_poll_one();
}

unsigned ejit_taskpool_pending_count(void) {
  if (!gEJIT)
    return 0;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return 0;
  return tp->pendingCount();
}

ejit_status_t ejit_taskpool_get_stats(ejit_taskpool_stats_t *out) {
  if (!out)
    return EJIT_ERR_INVALID_PARAM;
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp)
    return EJIT_ERR_NOT_ACTIVE;

  EJitTaskPoolStatsSnapshot s;
  tp->getStats(s);
  out->cacheHits = s.cacheHits;
  out->syncCompiles = s.syncCompiles;
  out->asyncCompiles = s.asyncCompiles;
  out->asyncEnqueues = s.asyncEnqueues;
  out->alreadyPending = s.alreadyPending;
  out->queueFull = s.queueFull;
  out->dedupFull = s.dedupFull;
  out->compileFailed = s.compileFailed;
  out->publishFailed = s.publishFailed;
  out->freeCodeCalls = s.freeCodeCalls;
  out->readyEntries = s.readyEntries;
  out->pendingEntries = s.pendingEntries;
  out->queueApproxSize = s.queueApproxSize;
  out->reserved = 0;
  return EJIT_OK;
}
#endif // EJIT_SRE_TASKPOOL

} // extern "C"
