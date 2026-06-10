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
                                     EJitRuntimeState &runtimeState,
                                     EJitModuleLoader &loader,
                                     EJitLogger *logger)
    : config_(config), cache_(cache),
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

void *EJitCompileDriver::getOrCompile(uint64_t cacheKey) {

  // ── Hot path: single hash find ──────────────────────────────────────────
  if (void *cached = cache_.getOrNull(cacheKey))
    return cached;

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
  if (funcName.empty())
    return nullptr;

  // Get bitcode
  auto bitcodeOrErr = loader_.getBitcodeByFuncIdx(funcIdx);
  if (!bitcodeOrErr) {
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(ErrorCode::BitcodeNotFound,
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
#ifndef EJIT_FREESTANDING
      if (logger_)
        logger_->log(ErrorCode::TimeWindowNotActive,
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
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(ErrorCode::NotActive,
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
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
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
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to look up compiled function", funcName,
                   std::to_string(cacheKey));
#else
    consumeError(addrOrErr.takeError());
#endif
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  SmallVector<std::string, 4> periodDeps;
  for (unsigned i = 0; i < dimCount; ++i)
    periodDeps.push_back(periodNames[i] + "=" + std::to_string(dims[i]));

  cache_.put(cacheKey, funcPtr, bitcode.size(), periodDeps);

  return funcPtr;
}
