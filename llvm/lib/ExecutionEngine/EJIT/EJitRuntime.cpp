//===-- EJitRuntime.cpp - EmbeddedJIT C Runtime API -----------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"

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
}

extern "C" {

ejit_status_t ejit_init(const ejit_config_t *config) {
  if (gEJIT)
    return EJIT_OK; // Already initialized

  Config cfg;
  parseConfig(config, cfg);

  gEJIT = new EJit(cfg);
  if (!gEJIT)
    return EJIT_ERROR_OOM;

  return EJIT_OK;
}

void ejit_shutdown(void) {
  delete gEJIT;
  gEJIT = nullptr;
}

ejit_status_t ejit_activate(const char *periodName, uint32_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->activate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate(const char *periodName, uint32_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->deactivate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_array(const char *periodName, void *arrayPtr,
                                   uint32_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->activate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_array(const char *periodName, void *arrayPtr,
                                     uint32_t cellIdx) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->deactivate(periodName, cellIdx);
  return EJIT_OK;
}

ejit_status_t ejit_activate_all(const char *periodName) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->activateAll(periodName);
  return EJIT_OK;
}

ejit_status_t ejit_deactivate_all(const char *periodName) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  gEJIT->deactivateAll(periodName);
  return EJIT_OK;
}

bool ejit_is_active(const char *periodName, uint32_t cellIdx) {
  if (!gEJIT)
    return false;
  return gEJIT->isActive(periodName, cellIdx);
}

void *ejit_compile_or_get(const char *funcName,
                           const ejit_dim_t *dims, uint32_t count,
                           void **out_pfn) {
  if (!gEJIT)
    return nullptr;

  SmallVector<std::pair<std::string, unsigned>, 4> cppDims;
  for (uint32_t i = 0; i < count && i < 4; ++i)
    cppDims.push_back({dims[i].periodName, dims[i].index});

  void *result = gEJIT->getOrCompile(funcName, cppDims.data(), count);
  if (out_pfn)
    *out_pfn = result;
  return result;
}

void ejit_clear_cache(void) {
  if (gEJIT)
    gEJIT->clearCache();
}

void ejit_invalidate(const char *periodName, uint32_t cellIdx) {
  if (gEJIT)
    gEJIT->invalidateByPeriod(periodName, cellIdx);
}

ejit_status_t ejit_get_stats(ejit_stats_t *stats) {
  if (!gEJIT)
    return EJIT_ERROR_NOT_INITIALIZED;
  if (!stats)
    return EJIT_ERROR_INVALID_ARG;

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
  // Return a static buffer for C API
  static ejit_error_t err;
  const EJitError *last = gEJIT->getLastError();
  if (!last)
    return nullptr;
  err.code = static_cast<int>(last->code);
  err.message = last->message.c_str();
  err.funcName = last->funcName.c_str();
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
