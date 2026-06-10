//===-- EJitRuntime.cpp - EmbeddedJIT C Runtime API -----------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"

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
  if (gEJIT)
    return EJIT_OK;

  Config cfg;
  parseConfig(config, cfg);

  gEJIT = new (std::nothrow) EJit(cfg);
  if (!gEJIT)
    return EJIT_ERR_MEMORY;

  return EJIT_OK;
}

void ejit_shutdown(void) {
  delete gEJIT;
  gEJIT = nullptr;
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
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
  gEJIT->activate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate(const char *periodName, uint8_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERR_NOT_ACTIVE;
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
  gEJIT->activate(periodName, cellIdx);
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
  gEJIT->deactivate(periodName, cellIdx);
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
  if (!gEJIT)
    return nullptr;

  void *result = gEJIT->getOrCompile(cacheKey);
  if (out_pfn)
    *out_pfn = result;
  return result;
}

void ejit_clear_cache(void) {
  if (gEJIT)
    gEJIT->clearCache();
}

void ejit_invalidate(const char *periodName, uint8_t cellIdx) {
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
  err.code = static_cast<int>(last->code);
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

} // extern "C"
