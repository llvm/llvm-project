//===-- EJitRuntime.cpp - EmbeddedJIT C Runtime API -----------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitFuncRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitLifecycleRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#ifdef EJIT_SRE_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#endif
#ifdef EJIT_SRE_SHARED_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitSharedTaskPool.h"
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

  // Registration failures during construction (funcIndex/lifecycle capacity
  // exhausted, a malformed or conflicting bitcode payload, or a null fixup
  // pointer) must fail init rather than expose a half-registered taskpool.
  if (gEJIT->initFailed()) {
    const EJitError &e = gEJIT->initError();
    EJIT_DIAG("init failed: code=%d %s (%s)", e.code, e.message.c_str(),
              e.funcName.c_str());
    ejit_status_t st = (e.code != 0) ? static_cast<ejit_status_t>(e.code)
                                     : EJIT_ERR_INVALID_PARAM;
    delete gEJIT;
    gEJIT = nullptr;
    return st;
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
  EJIT_DIAG("register_symbol name=%s addr=%p", name ? name : "<null>", addr);
  if (gEJIT) {
    gEJIT->registerSymbol(name, addr);
  } else {
    // Constructor-phase call (before ejit_init): stage for later consumption.
    EJitRegistrationStore::instance().registerSymbol(name, addr);
  }
}

void ejit_register_bitcode(const char *funcName, const uint8_t *bitcodeData,
                           uint64_t bitcodeSize) {
  EJIT_DIAG("register_bitcode name=%s size=%llu",
            funcName ? funcName : "<null>",
            static_cast<unsigned long long>(bitcodeSize));
  if (gEJIT) {
    // Post-init runtime registration: the void ABI cannot return a status, so a
    // rejection (null/zero payload, funcIndex capacity, conflicting payload) is
    // recorded in the registration-error sink for observability.
    if (!gEJIT->registerBitcode(funcName, bitcodeData,
                                static_cast<size_t>(bitcodeSize))) {
      EJitRegistrationStore::instance().recordError(
          EJIT_ERR_INVALID_PARAM, "runtime bitcode registration rejected",
          funcName ? funcName : "");
      EJIT_DIAG("register_bitcode FAIL name=%s: runtime registration rejected",
                funcName ? funcName : "<null>");
    }
  } else {
    EJitRegistrationStore::instance().registerBitcode(
        funcName, bitcodeData, static_cast<size_t>(bitcodeSize));
  }
}

void ejit_register_period_array(const char *periodName, const char *varName,
                                void *baseAddr, uint64_t arraySize) {
  EJIT_DIAG("register_period_array period=%s var=%s size=%llu",
            periodName ? periodName : "<null>", varName ? varName : "<null>",
            static_cast<unsigned long long>(arraySize));
  if (gEJIT) {
    // Post-init: rejected once registration is frozen (taskpool); the void ABI
    // records the failure for observability and mutates nothing.
    if (!gEJIT->registerPeriodArray(periodName, varName, baseAddr, arraySize)) {
      EJitRegistrationStore::instance().recordError(
          EJIT_ERR_INVALID_PARAM, "runtime period-array registration rejected",
          periodName ? periodName : "");
      EJIT_DIAG("register_period_array FAIL period=%s: rejected",
                periodName ? periodName : "<null>");
    }
  } else {
    EJitRegistrationStore::instance().registerPeriodArray(periodName, varName,
                                                          baseAddr, arraySize);
  }
}

void ejit_register_static_var(const char *varName, void *varAddr) {
  EJIT_DIAG("register_static_var var=%s addr=%p",
            varName ? varName : "<null>", varAddr);
  if (gEJIT) {
    if (!gEJIT->registerStaticVar(varName, varAddr)) {
      EJitRegistrationStore::instance().recordError(
          EJIT_ERR_INVALID_PARAM, "runtime static-var registration rejected",
          varName ? varName : "");
      EJIT_DIAG("register_static_var FAIL var=%s: rejected",
                varName ? varName : "<null>");
    }
  } else {
    EJitRegistrationStore::instance().registerStaticVar(varName, varAddr);
  }
}

void ejit_register_lifecycle(const char *lifecycleName, uint32_t *slotOut) {
  // Self-contained: the dimType-slot assignment lives in a process-global
  // registry that exists independently of the EJit instance, so this works
  // whether called from a global constructor (before ejit_init) or the static
  // registry-table walk. Idempotent — the same name always yields the same
  // slot. A capacity failure (the 9th distinct lifecycle) leaves *slotOut
  // invalid AND is recorded so ejit_init fails instead of silently continuing.
  if (!lifecycleName || !slotOut) {
    EJIT_DIAG("register_lifecycle reject: name=%p slotOut=%p",
              (const void *)lifecycleName, (void *)slotOut);
    return;
  }
  EJIT_DIAG("register_lifecycle name=%s", lifecycleName);
#ifdef EJIT_SRE_TASKPOOL
  // Once a taskpool init has frozen registration, the worker reads the registry
  // lock-free: refuse to mutate it (leave *slotOut and the registry unchanged).
  if (gEJIT && gEJIT->registrationFrozen()) {
    EJitRegistrationStore::instance().recordError(
        EJIT_ERR_INVALID_PARAM, "lifecycle registration after init is frozen",
        lifecycleName);
    EJIT_DIAG("register_lifecycle reject name=%s: registration frozen",
              lifecycleName);
    return;
  }
#endif
  uint32_t slot =
      EJitLifecycleRegistry::instance().resolveAssign(lifecycleName);
  *slotOut = slot;
  if (slot == kEJitInvalidDimType) {
    EJitRegistrationStore::instance().recordError(
        EJIT_ERR_CACHE_FULL, "lifecycle (dimType) capacity exhausted",
        lifecycleName);
    EJIT_DIAG("register_lifecycle FAIL name=%s: dimType capacity exhausted",
              lifecycleName);
  } else {
    EJIT_DIAG("register_lifecycle OK name=%s slot=%u", lifecycleName, slot);
  }
}

void ejit_register_funcindex(const char *funcName, uint32_t *slotOut) {
  // Self-contained dense-funcIndex assignment, mirroring ejit_register_
  // lifecycle. Idempotent by name. Capacity exhaustion leaves *slotOut invalid
  // (the wrapper then falls back without entering the taskpool) AND is recorded
  // so ejit_init fails rather than building a half-registered taskpool.
  if (!funcName || !slotOut) {
    EJIT_DIAG("register_funcindex reject: name=%p slotOut=%p",
              (const void *)funcName, (void *)slotOut);
    return;
  }
  EJIT_DIAG("register_funcindex name=%s", funcName);
#ifdef EJIT_SRE_TASKPOOL
  if (gEJIT && gEJIT->registrationFrozen()) {
    EJitRegistrationStore::instance().recordError(
        EJIT_ERR_INVALID_PARAM, "funcIndex registration after init is frozen",
        funcName);
    EJIT_DIAG("register_funcindex reject name=%s: registration frozen",
              funcName);
    return;
  }
#endif
  uint32_t idx = EJitFuncRegistry::instance().resolveAssign(funcName);
  *slotOut = idx;
  if (idx == kEJitInvalidFuncIndex) {
    EJitRegistrationStore::instance().recordError(
        EJIT_ERR_CACHE_FULL, "funcIndex capacity exhausted for function",
        funcName);
    EJIT_DIAG("register_funcindex FAIL name=%s: funcIndex capacity exhausted",
              funcName);
  } else {
    EJIT_DIAG("register_funcindex OK name=%s idx=%u", funcName, idx);
  }
}

ejit_status_t ejit_activate(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT) {
    EJIT_DIAG("activate(%s,%u) failed: not initialized", periodName, cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  EJIT_DIAG("activate(%s,%u)", periodName, cellIdx);
  // In a taskpool build this also syncs the SwitchController and returns false
  // for an unknown lifecycle (no state changed). In the legacy build it always
  // succeeds.
  if (!gEJIT->activate(periodName, cellIdx))
    return EJIT_ERR_INVALID_PARAM;
  return EJIT_OK;
}

ejit_status_t ejit_deactivate(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT) {
    EJIT_DIAG("deactivate(%s,%u) failed: not initialized", periodName, cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  EJIT_DIAG("deactivate(%s,%u)", periodName, cellIdx);
  if (!gEJIT->deactivate(periodName, cellIdx))
    return EJIT_ERR_INVALID_PARAM; // unknown lifecycle: nothing changed.
  gEJIT->invalidateByPeriod(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_array(const char *periodName, void *arrayPtr,
                                  uint8_t cellIdx) {
  EJIT_DIAG("activate_array(%s,%u) ptr=%p", periodName, cellIdx, arrayPtr);
  if (!gEJIT) {
    EJIT_DIAG("activate_array(%s,%u) failed: not initialized", periodName,
              cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  // Validation is unified in EJit::activateArray (registered array + matching
  // period + registered lifecycle in a taskpool build) so it stays consistent
  // with the SwitchController sync. No duplicate pre-checks here.
  if (!gEJIT->activateArray(periodName, arrayPtr, cellIdx))
    return EJIT_ERR_INVALID_PARAM;
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_array(const char *periodName, void *arrayPtr,
                                    uint8_t cellIdx) {
  EJIT_DIAG("deactivate_array(%s,%u) ptr=%p", periodName, cellIdx, arrayPtr);
  if (!gEJIT) {
    EJIT_DIAG("deactivate_array(%s,%u) failed: not initialized", periodName,
              cellIdx);
    return EJIT_ERR_NOT_ACTIVE;
  }
  if (!gEJIT->deactivateArray(periodName, arrayPtr, cellIdx))
    return EJIT_ERR_INVALID_PARAM;
  gEJIT->invalidateByPeriod(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_all(const char *periodName) {
  EJIT_DIAG("activate_all(%s)", periodName);
  if (!gEJIT) {
    EJIT_DIAG("activate_all(%s) failed: not initialized", periodName);
    return EJIT_ERR_NOT_ACTIVE;
  }
  if (!gEJIT->activateAll(periodName))
    return EJIT_ERR_INVALID_PARAM;
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_all(const char *periodName) {
  EJIT_DIAG("deactivate_all(%s)", periodName);
  if (!gEJIT) {
    EJIT_DIAG("deactivate_all(%s) failed: not initialized", periodName);
    return EJIT_ERR_NOT_ACTIVE;
  }
  if (!gEJIT->deactivateAll(periodName))
    return EJIT_ERR_INVALID_PARAM;
  gEJIT->invalidateAllByPeriod(periodName);
  return EJIT_OK;
}

bool ejit_is_active(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT) {
    EJIT_DIAG("is_active(%s,%u) failed: not initialized", periodName, cellIdx);
    return false;
  }
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
  if (!gEJIT) {
    EJIT_DIAG("get_stats failed: not initialized");
    return EJIT_ERR_NOT_ACTIVE;
  }
  if (!stats) {
    EJIT_DIAG("get_stats failed: null stats pointer");
    return EJIT_ERR_INVALID_PARAM;
  }

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
  if (!gEJIT) {
    EJIT_DIAG("get_last_error: not initialized");
    return nullptr;
  }
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
  EJIT_DIAG("set_compile_mode mode=%u", static_cast<unsigned>(mode));
  if (gEJIT)
    (void)gEJIT->setCompileMode(mode == EJIT_COMPILE_ASYNC ? CompileMode::Async
                                                           : CompileMode::Sync);
  else
    EJIT_DIAG("set_compile_mode reject: not initialized");
}

ejit_compile_mode_t ejit_get_compile_mode(void) {
  if (!gEJIT) {
    EJIT_DIAG("get_compile_mode: not initialized (default sync)");
    return EJIT_COMPILE_SYNC;
  }
  return gEJIT->getCompileMode() == CompileMode::Async ? EJIT_COMPILE_ASYNC
                                                       : EJIT_COMPILE_SYNC;
}

#ifdef EJIT_SRE_TASKPOOL
//===-- SRE taskpool black-box API ----------------------------------------===//

static ejit_status_t taskpoolStatus(EJitCompileOrGetStatus s) {
  switch (s) {
  case EJitCompileOrGetStatus::CacheHit:
    return EJIT_OK;
  case EJitCompileOrGetStatus::EnqueuedPending:
  case EJitCompileOrGetStatus::AlreadyPending:
    return EJIT_PENDING;
  case EJitCompileOrGetStatus::QueueFullFallback:
    return EJIT_ERR_QUEUE_FULL;
  case EJitCompileOrGetStatus::OffMode:
    return EJIT_ERR_DISABLED;
  case EJitCompileOrGetStatus::InstanceDisabled:
    return EJIT_ERR_INSTANCE_DISABLED;
  case EJitCompileOrGetStatus::InvalidParam:
    return EJIT_ERR_INVALID_PARAM;
  case EJitCompileOrGetStatus::CompileFailed:
  default:
    return EJIT_ERR_COMPILE_FAILED;
  }
}

namespace {
// Resolve the taskpool the C ABI drives: the cross-core SHARED pool in a shared
// build, otherwise the per-instance pool. Both expose compileOrGet /
// releaseRead / pendingCount / pollOne / pollBudget with matching shapes, so
// the common call sites use `auto *tp = activeTaskPool();`. (Stats and the
// switch-controller toggle differ in shape and branch explicitly.)
#ifdef EJIT_SRE_SHARED_TASKPOOL
inline EJitSharedTaskPool *activeTaskPool() {
  return gEJIT ? gEJIT->sharedTaskPool() : nullptr;
}
#else
inline EJitTaskPool *activeTaskPool() {
  return gEJIT ? gEJIT->taskPool() : nullptr;
}
#endif
} // namespace

ejit_status_t ejit_taskpool_compile_or_get(uint32_t funcIndex,
                                           const ejit_dim_pair_t *dims,
                                           uint32_t numDims, void **outFn,
                                           uint32_t *outBucket) {
  if (outFn)
    *outFn = nullptr;
  if (outBucket)
    *outBucket = 0;
  EJIT_DIAG("taskpool_compile_or_get func=%u dims=%u", funcIndex, numDims);
  if (!gEJIT) {
    EJIT_DIAG("taskpool_compile_or_get reject func=%u: not initialized",
              funcIndex);
    return EJIT_ERR_NOT_ACTIVE;
  }
  auto *tp = activeTaskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_compile_or_get reject func=%u: no taskpool", funcIndex);
    return EJIT_ERR_NOT_ACTIVE;
  }

  EJitDimPair localDims[4];
  if (numDims > 4) {
    EJIT_DIAG("taskpool_compile_or_get reject func=%u: numDims=%u > 4",
              funcIndex, numDims);
    return EJIT_ERR_INVALID_PARAM;
  }
  if (numDims > 0 && !dims) {
    EJIT_DIAG("taskpool_compile_or_get reject func=%u: dims=null numDims=%u",
              funcIndex, numDims);
    return EJIT_ERR_INVALID_PARAM;
  }
  for (uint32_t i = 0; i < numDims; ++i) {
    // dimType is an explicit lifecycle index in [0, MAX_DIM_TYPES); instanceId
    // is in [0, MAX_INSTANCES). Both are range-checked (spec §5.1).
    if (dims[i].dimType >= EJitSwitchController::MAX_DIM_TYPES) {
      EJIT_DIAG("taskpool_compile_or_get reject func=%u: dim[%u] dimType=%u OOR",
                funcIndex, i, dims[i].dimType);
      return EJIT_ERR_INVALID_PARAM;
    }
    if (dims[i].instanceId >= EJitSwitchController::MAX_INSTANCES) {
      EJIT_DIAG("taskpool_compile_or_get reject func=%u: dim[%u] instanceId=%u OOR",
                funcIndex, i, dims[i].instanceId);
      return EJIT_ERR_INVALID_PARAM;
    }
    localDims[i].dimType = dims[i].dimType;
    localDims[i].instanceId = dims[i].instanceId;
  }

  auto r = tp->compileOrGet(funcIndex, numDims ? localDims : nullptr, numDims,
                            /*fallback=*/nullptr);
  if (outFn)
    *outFn = r.fnPtr;
  if (outBucket)
    *outBucket = r.bucketIndex;
  EJIT_DIAG("taskpool_compile_or_get func=%u status=%u fn=%p", funcIndex,
            static_cast<unsigned>(r.status), r.fnPtr);
  return taskpoolStatus(r.status);
}

void ejit_taskpool_set_instance_enabled(uint32_t dimType, uint32_t instanceId,
                                        uint32_t enabled) {
  EJIT_DIAG("taskpool_set_instance_enabled dim=%u inst=%u enabled=%u", dimType,
            instanceId, enabled);
  if (!gEJIT) {
    EJIT_DIAG("taskpool_set_instance_enabled reject: not initialized");
    return;
  }
#ifdef EJIT_SRE_SHARED_TASKPOOL
  EJitSharedTaskPool *sp = gEJIT->sharedTaskPool();
  if (!sp) {
    EJIT_DIAG("taskpool_set_instance_enabled reject: no shared taskpool");
    return;
  }
  sp->setInstanceEnabled(dimType, instanceId, enabled != 0);
#else
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_set_instance_enabled reject: no taskpool");
    return;
  }
  tp->switchController().setEnabled(dimType, instanceId, enabled != 0);
#endif
}

void ejit_taskpool_release_read(uint32_t bucketIndex) {
  if (!gEJIT) {
    EJIT_DIAG("taskpool_release_read bucket=%u reject: not initialized",
              bucketIndex);
    return;
  }
  auto *tp = activeTaskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_release_read bucket=%u reject: no taskpool", bucketIndex);
    return;
  }
  tp->releaseRead(bucketIndex);
}

#ifdef EJIT_SRE_TASKPOOL_TESTING
unsigned ejit_taskpool_poll_one(void) {
  if (!gEJIT) {
    EJIT_DIAG("taskpool_poll_one reject: not initialized");
    return 0;
  }
  auto *tp = activeTaskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_poll_one reject: no taskpool");
    return 0;
  }
  return tp->pollOne() ? 1u : 0u;
}

unsigned ejit_taskpool_poll_budget(unsigned maxItems) {
  if (!gEJIT) {
    EJIT_DIAG("taskpool_poll_budget max=%u reject: not initialized", maxItems);
    return 0;
  }
  auto *tp = activeTaskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_poll_budget max=%u reject: no taskpool", maxItems);
    return 0;
  }
  return tp->pollBudget(maxItems);
}
#endif

unsigned ejit_taskpool_pending_count(void) {
  if (!gEJIT) {
    EJIT_DIAG("taskpool_pending_count reject: not initialized");
    return 0;
  }
  auto *tp = activeTaskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_pending_count reject: no taskpool");
    return 0;
  }
  return tp->pendingCount();
}

ejit_status_t ejit_taskpool_get_stats(ejit_taskpool_stats_t *out) {
  if (!out) {
    EJIT_DIAG("taskpool_get_stats failed: null out pointer");
    return EJIT_ERR_INVALID_PARAM;
  }
  if (!gEJIT) {
    EJIT_DIAG("taskpool_get_stats failed: not initialized");
    return EJIT_ERR_NOT_ACTIVE;
  }
#ifdef EJIT_SRE_SHARED_TASKPOOL
  EJitSharedTaskPool *sp = gEJIT->sharedTaskPool();
  if (!sp) {
    EJIT_DIAG("taskpool_get_stats failed: no shared taskpool");
    return EJIT_ERR_NOT_ACTIVE;
  }
  EJitSharedDiagnostics d;
  sp->getDiagnostics(d);
  out->cacheHits = d.cacheHits;
  out->asyncCompiles = d.asyncCompiles;
  out->asyncEnqueues = d.asyncEnqueues;
  out->alreadyPending = d.alreadyPending;
  out->queueFull = d.queueFull;
  out->compileFailed = d.compileFailed;
  out->publishFailed = d.publishFailed;
  out->instanceDisabled = d.instanceDisabled;
  out->readyEntries = d.cacheReadyCount;
  out->pendingEntries = d.pendingCount;
  out->queueApproxSize = d.queueDepth;
  out->reserved = 0;
  return EJIT_OK;
#else
  EJitTaskPool *tp = gEJIT->taskPool();
  if (!tp) {
    EJIT_DIAG("taskpool_get_stats failed: no taskpool");
    return EJIT_ERR_NOT_ACTIVE;
  }

  EJitTaskPoolStatsSnapshot s;
  tp->getStats(s);
  out->cacheHits = s.cacheHits;
  out->asyncCompiles = s.asyncCompiles;
  out->asyncEnqueues = s.asyncEnqueues;
  out->alreadyPending = s.alreadyPending;
  out->queueFull = s.queueFull;
  out->compileFailed = s.compileFailed;
  out->publishFailed = s.publishFailed;
  out->instanceDisabled = s.instanceDisabled;
  out->readyEntries = s.readyEntries;
  out->pendingEntries = s.pendingEntries;
  out->queueApproxSize = s.queueApproxSize;
  out->reserved = 0;
  return EJIT_OK;
#endif
}
#endif // EJIT_SRE_TASKPOOL

} // extern "C"
