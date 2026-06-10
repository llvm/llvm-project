//===-- EJitCompileDriver.cpp - Compilation Scheduler ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#ifndef EJIT_FREESTANDING
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#endif
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
  runtimeState_(runtimeState), loader_(loader)
#ifndef EJIT_FREESTANDING
  , logger_(logger)
#endif
{}

EJitCompileDriver::~EJitCompileDriver() = default;

void EJitCompileDriver::setSyncEngine(std::unique_ptr<EJitOrcEngine> engine) {
  syncEngine_ = std::move(engine);
}

void EJitCompileDriver::registerSymbol(const std::string &name, void *addr) {
  if (syncEngine_)
    syncEngine_->addUserSymbol(name, addr);
}

// Common compile path (cache miss). Both string-based and funcIdx-based
// overloads delegate here after computing the cacheKey.
static void *compileMiss(EJitCompileDriver &D,
                         const std::string &funcName,
                         uint64_t cacheKey,
                         const std::pair<std::string, uint8_t> *dims,
                         unsigned count) {
  EJitCache &cache_ = D.getCache();
  EJitRuntimeState &runtimeState_ = D.getRuntimeState();
  EJitModuleLoader &loader_ = D.getLoader();

  // Verify time-window state
  for (unsigned i = 0; i < count; ++i) {
    if (!runtimeState_.isActive(dims[i].first, dims[i].second)) {
#ifndef EJIT_FREESTANDING
      if (D.getLogger())
        D.getLogger()->log(ErrorCode::TimeWindowNotActive,
                     "Time window not active for " + dims[i].first,
                     funcName, std::to_string(cacheKey));
#endif
      return nullptr;
    }
  }

  // Get bitcode via funcIdx (O(1)) when possible, fall back to string lookup
  uint32_t funcIdx = loader_.getFuncIndex(funcName);
  auto bitcodeOrErr = loader_.getBitcodeByFuncIdx(funcIdx);
  if (!bitcodeOrErr)
    bitcodeOrErr = loader_.getBitcode(funcName);
  if (!bitcodeOrErr) {
#ifndef EJIT_FREESTANDING
    if (D.getLogger())
      D.getLogger()->log(ErrorCode::BitcodeNotFound,
                   "No bitcode for function", funcName, std::to_string(cacheKey));
#endif
    return nullptr;
  }

  StringRef bitcode = *bitcodeOrErr;

  // Build specialization context
  SpecializationContext ctx;
  ctx.fnName = funcName;
  ctx.cacheKey = cacheKey;
  ctx.optLevel = D.getConfig().optLevel;
  for (unsigned i = 0; i < count; ++i)
    ctx.dimensions.push_back({dims[i].first, dims[i].second});

  EJitOrcEngine *syncEngine_ = D.getSyncEngine();
  if (!syncEngine_) {
#ifndef EJIT_FREESTANDING
    if (D.getLogger())
      D.getLogger()->log(ErrorCode::NotActive,
                   "Sync engine not initialized", funcName,
                   std::to_string(cacheKey));
#endif
    return nullptr;
  }

#ifndef EJIT_FREESTANDING
  auto start = std::chrono::steady_clock::now();
#endif

  syncEngine_->setActiveContext(&ctx);

  if (auto Err = syncEngine_->loadBitcodeModule(bitcode, cacheKey, funcName)) {
    syncEngine_->setActiveContext(nullptr);
#ifndef EJIT_FREESTANDING
    if (D.getLogger())
      D.getLogger()->log(ErrorCode::CompilationFailed,
                   "Failed to load bitcode module", funcName, std::to_string(cacheKey));
#else
    consumeError(std::move(Err));
#endif
    return nullptr;
  }

  auto addrOrErr = syncEngine_->lookup(cacheKey, funcName);
  syncEngine_->setActiveContext(nullptr);

  if (!addrOrErr) {
#ifndef EJIT_FREESTANDING
    if (D.getLogger())
      D.getLogger()->log(ErrorCode::CompilationFailed,
                   "Failed to look up compiled function", funcName, std::to_string(cacheKey));
#else
    consumeError(addrOrErr.takeError());
#endif
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  SmallVector<std::string, 4> periodDeps;
  for (unsigned i = 0; i < count; ++i)
    periodDeps.push_back(dims[i].first + "=" + std::to_string(dims[i].second));

  cache_.put(cacheKey, funcPtr, bitcode.size(), periodDeps);

  return funcPtr;
}

void *EJitCompileDriver::getOrCompile(
    const std::string &funcName,
    const std::pair<std::string, uint8_t> *dims,
    unsigned count) {

  uint32_t funcIdx = loader_.getFuncIndex(funcName);
  uint64_t cacheKey = EJitCache::buildCacheKey(funcIdx, dims, count);

  if (void *cached = cache_.getOrNull(cacheKey))
    return cached;

  return compileMiss(*this, funcName, cacheKey, dims, count);
}

void *EJitCompileDriver::getOrCompile(
    uint32_t funcIdx,
    const std::pair<std::string, uint8_t> *dims,
    unsigned count) {

  uint64_t cacheKey = EJitCache::buildCacheKey(funcIdx, dims, count);

  // Cache hit: zero string operations in this path
  if (void *cached = cache_.getOrNull(cacheKey))
    return cached;

  // Cache miss: resolve funcName from the module loader
  const std::string &funcName = loader_.getFuncNameByFuncIdx(funcIdx);
  if (funcName.empty())
    return nullptr;

  return compileMiss(*this, funcName, cacheKey, dims, count);
}
