//===-- EJitCompileDriver.cpp - Compilation Scheduler ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#ifndef EJIT_FREESTANDING
#include <chrono>
#endif

using namespace llvm;
using namespace llvm::ejit;

EJitCompileDriver::EJitCompileDriver(const Config &config,
                                     EJitCache &cache,
                                     PeriodArrayRegistry &periodReg,
                                     EJitRuntimeState &runtimeState,
                                     EJitModuleLoader &loader,
                                     EJitLogger *logger)
    : config_(config), cache_(cache), periodReg_(periodReg),
      runtimeState_(runtimeState), loader_(loader), logger_(logger) {}

EJitCompileDriver::~EJitCompileDriver() = default;

void EJitCompileDriver::setSyncEngine(std::unique_ptr<EJitOrcEngine> engine) {
  syncEngine_ = std::move(engine);
}

void EJitCompileDriver::registerSymbol(const std::string &name, void *addr) {
  if (syncEngine_)
    syncEngine_->addUserSymbol(name, addr);
}

void *EJitCompileDriver::getOrCompile(
    const std::string &funcName,
    const std::pair<std::string, uint8_t> *dims,
    unsigned count) {

  // Build cache key: uint32_t = funcIdx(16b) | dim[0..3](4x4b)
  uint16_t funcIdx = loader_.getFuncIndex(funcName);
  uint32_t cacheKey = EJitCache::buildCacheKey(funcIdx, dims, count);

  // Check cache
  if (void *cached = cache_.getOrNull(cacheKey))
    return cached;

  // Verify time-window state
  for (unsigned i = 0; i < count; ++i) {
    if (!runtimeState_.isActive(dims[i].first, dims[i].second)) {
      if (logger_)
        logger_->log(ErrorCode::TimeWindowNotActive,
                     "Time window not active for " + dims[i].first,
                     funcName, std::to_string(cacheKey));
      return nullptr;
    }
  }

  // Get bitcode
  auto bitcodeOrErr = loader_.getBitcode(funcName);
  if (!bitcodeOrErr) {
    if (logger_)
      logger_->log(ErrorCode::BitcodeNotFound,
                   "No bitcode for function", funcName, std::to_string(cacheKey));
    return nullptr;
  }

  std::string bitcode = bitcodeOrErr->str();

  // Build specialization context
  SpecializationContext ctx;
  ctx.fnName = funcName;
  ctx.cacheKey = cacheKey;
  ctx.optLevel = config_.optLevel;
  for (unsigned i = 0; i < count; ++i)
    ctx.dimensions.push_back({dims[i].first, dims[i].second});

  // Sync compile
  if (!syncEngine_) {
    if (logger_)
      logger_->log(ErrorCode::NotActive,
                   "Sync engine not initialized", funcName,
                   std::to_string(cacheKey));
    return nullptr;
  }

#ifndef EJIT_FREESTANDING
  auto start = std::chrono::steady_clock::now();
#endif

  syncEngine_->setActiveContext(&ctx);

  // Load module with cacheKey as module ID and original funcName for
  // symbol renaming (each specialization gets a unique symbol).
  if (auto Err = syncEngine_->loadBitcodeModule(bitcode, cacheKey, funcName)) {
    syncEngine_->setActiveContext(nullptr);
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to load bitcode module", funcName, std::to_string(cacheKey));
    return nullptr;
  }

  auto addrOrErr = syncEngine_->lookup(cacheKey, funcName);
  syncEngine_->setActiveContext(nullptr);

  if (!addrOrErr) {
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to look up compiled function", funcName, std::to_string(cacheKey));
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  // Cache the result.
  // NOTE: codeSize is the bitcode size, not the compiled machine code size.
  // Getting the actual machine code size from LLJIT/JITLink requires
  // instrumenting the memory manager. For now, bitcode size serves as an
  // approximation for cache eviction decisions.
  std::set<std::string> periodDeps;
  for (unsigned i = 0; i < count; ++i)
    periodDeps.insert(dims[i].first + "=" + std::to_string(dims[i].second));

  cache_.put(cacheKey, funcPtr, bitcode.size(), periodDeps);

  return funcPtr;
}
