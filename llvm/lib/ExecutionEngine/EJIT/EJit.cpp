//===-- EJit.cpp - EmbeddedJIT Main C++ API -------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace llvm::ejit;

EJit::EJit(const Config &config) : config_(config) {
  // Create all runtime components
  runtimeState_ = std::make_unique<EJitRuntimeState>();
  moduleLoader_ = std::make_unique<EJitModuleLoader>();
  cache_ = std::make_unique<EJitCache>(config.maxCacheEntries,
                                       config.maxCacheSize,
                                       config.maxSingleFuncSize);

  if (config.enableLogger)
    logger_ = std::make_unique<EJitLogger>();

  compileDriver_ = std::make_unique<EJitCompileDriver>(
      config, *cache_, runtimeState_->getRegistry(),
      *runtimeState_, *moduleLoader_, logger_.get());

  // Consume registration data from the staging store
  StoredData data = EJitRegistrationStore::instance().consume();

  // Populate bitcode tracker
  for (auto &be : data.bitcodes)
    moduleLoader_->registerBitcode(be.funcName, be.data, be.size);

  // Populate period registry
  PeriodArrayRegistry &reg = runtimeState_->getRegistry();
  for (auto &pa : data.periodArrays)
    reg.registerArray(pa.periodName, pa.varName, pa.baseAddr, pa.arraySize);

  for (auto &sv : data.staticVars)
    reg.registerStaticVar(sv.varName, sv.varAddr);

  // Create sync JIT engine (target must be initialized first).
  // Use InitializeAll* instead of InitializeNative* so that cross-compiled
  // builds (e.g. AArch64 target built on x86 host) also work correctly.
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  auto engine = EJitOrcEngine::Create(config, reg, *runtimeState_);
  if (engine) {
    // Forward auto-registered user symbols to the engine.
    for (auto &sym : data.userSymbols)
      (*engine)->addUserSymbol(sym.name, sym.addr);
    compileDriver_->setSyncEngine(std::move(*engine));
  } else {
    std::string errStr;
    llvm::handleAllErrors(engine.takeError(),
                          [&](const llvm::ErrorInfoBase &E) { errStr = E.message(); });
    if (logger_)
      logger_->log(ErrorCode::CompilationFailed,
                   "Failed to create OrcJIT engine: " + errStr, "", "");
  }
}

EJit::~EJit() {
  // Destroy in reverse order (compile driver holds references to other components)
  compileDriver_.reset();
  cache_.reset();
  moduleLoader_.reset();
  logger_.reset();
  runtimeState_.reset();
}

void EJit::activate(const std::string &periodName, uint8_t cellIdx) {
  runtimeState_->activate(periodName, cellIdx);
}

void EJit::deactivate(const std::string &periodName, uint8_t cellIdx) {
  runtimeState_->deactivate(periodName, cellIdx);
}

void EJit::activateAll(const std::string &periodName) {
  runtimeState_->activateAll(periodName);
}

void EJit::deactivateAll(const std::string &periodName) {
  runtimeState_->deactivateAll(periodName);
}

bool EJit::isActive(const std::string &periodName, uint8_t cellIdx) const {
  return runtimeState_->isActive(periodName, cellIdx);
}

void *EJit::getOrCompile(const std::string &funcName,
                         const std::pair<std::string, uint8_t> *dims,
                         unsigned count) {
  return compileDriver_->getOrCompile(funcName, dims, count);
}

void EJit::clearCache() {
  cache_->clear();
}

void EJit::invalidateByPeriod(const std::string &periodName,
                              uint8_t cellIdx) {
  cache_->invalidateByPeriod(periodName, cellIdx);
}

void EJit::invalidateAllByPeriod(const std::string &periodName) {
  // Invalidate all known cellIdx entries for this period.
  // Iterate over registered arrays and invalidate each cell index.
  const auto *arrs = getRegistry().getArrays(periodName);
  if (!arrs)
    return;
  for (const auto &info : *arrs) {
    for (size_t i = 0; i < info.arraySize; i++)
      cache_->invalidateByPeriod(periodName, static_cast<uint8_t>(i));
  }
}

void EJit::registerSymbol(const std::string &name, void *addr) {
  if (compileDriver_)
    compileDriver_->registerSymbol(name, addr);
}

void EJit::setCompileMode(CompileMode mode) {
  config_.compileMode = mode;
}

CompileMode EJit::getCompileMode() const {
  return config_.compileMode;
}

void EJit::setOptimizationLevel(OptimizationLevel level) {
  config_.optLevel = level;
}

OptimizationLevel EJit::getOptimizationLevel() const {
  return config_.optLevel;
}

EJitCache::Stats EJit::getStats() const {
  return cache_->getStats();
}

const EJitError *EJit::getLastError() const {
  if (!logger_)
    return nullptr;
  // Copy into stable storage so the caller gets a snapshot that won't be
  // overwritten by concurrent log() calls on other threads.
  static thread_local EJitError lastErr;
  if (logger_->copyLastError(lastErr))
    return &lastErr;
  return nullptr;
}
