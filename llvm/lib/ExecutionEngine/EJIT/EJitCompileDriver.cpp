//===-- EJitCompileDriver.cpp - Compilation Scheduler ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include <chrono>

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

void *EJitCompileDriver::getOrCompile(
    const std::string &funcName,
    const std::pair<std::string, uint8_t> *dims,
    unsigned count) {

  // Build cache key
  std::string cacheKey = EJitCache::buildCacheKey(funcName, dims, count);

  // Check cache
  if (void *cached = cache_.getOrNull(cacheKey))
    return cached;

  // Verify time-window state
  for (unsigned i = 0; i < count; ++i) {
    if (!runtimeState_.isActive(dims[i].first, dims[i].second)) {
      if (logger_)
        logger_->log(ErrorCode::TimeWindowNotActive,
                     "Time window not active for " + dims[i].first,
                     funcName, cacheKey);
      return nullptr;
    }
  }

  // Get bitcode
  auto bitcodeOrErr = loader_.getBitcode(funcName);
  if (!bitcodeOrErr) {
    if (logger_)
      logger_->log(ErrorCode::BitcodeNotFound,
                   "No bitcode for function", funcName, cacheKey);
    return nullptr;
  }

  std::string bitcode = bitcodeOrErr->str();

  // Build specialization context
  SpecializationContext ctx;
  ctx.fnName = funcName;
  ctx.optLevel = config_.optLevel;
  for (unsigned i = 0; i < count; ++i)
    ctx.dimensions.push_back({dims[i].first, dims[i].second});

  // Sync compile
  if (!syncEngine_) {
    if (logger_)
      logger_->log(ErrorCode::NotActive,
                   "Sync engine not initialized", funcName, cacheKey);
    return nullptr;
  }

  auto start = std::chrono::steady_clock::now();

  syncEngine_->setActiveContext(&ctx);

  if (auto Err = syncEngine_->loadBitcodeModule(bitcode, funcName)) {
    syncEngine_->setActiveContext(nullptr);
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to load bitcode module", funcName, cacheKey);
    return nullptr;
  }

  auto addrOrErr = syncEngine_->lookup(funcName);
  syncEngine_->setActiveContext(nullptr);

  if (!addrOrErr) {
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to look up compiled function", funcName, cacheKey);
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  auto end = std::chrono::steady_clock::now();
  size_t compileTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             end - start).count();

  // Cache the result
  std::set<std::string> periodDeps;
  for (unsigned i = 0; i < count; ++i)
    periodDeps.insert(dims[i].first + "=" + std::to_string(dims[i].second));

  cache_.put(cacheKey, funcPtr, 0, periodDeps);

  return funcPtr;
}
