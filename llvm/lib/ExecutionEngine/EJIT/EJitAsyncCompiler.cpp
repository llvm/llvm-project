//===-- EJitAsyncCompiler.cpp - Asynchronous JIT Compiler -----------------===//

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitAsyncCompiler.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitSyncCompiler.h"

using namespace llvm;
using namespace llvm::ejit;

EJitAsyncCompiler::EJitAsyncCompiler(const Config &config,
                                     EJitCache &cache,
                                     EJitRuntimeState &runtimeState)
    : config_(config), cache_(cache), runtimeState_(runtimeState) {}

EJitAsyncCompiler::~EJitAsyncCompiler() {
  stop();
}

void EJitAsyncCompiler::start() {
  if (running_.exchange(true))
    return;

  syncCompiler_ = std::make_unique<EJitSyncCompiler>();

  // Create isolated engine for the worker thread
  auto engine = EJitOrcEngine::Create(
      config_, runtimeState_.getRegistry(), runtimeState_);
  if (engine)
    workerEngine_ = std::move(*engine);

  workerThread_ = std::thread(&EJitAsyncCompiler::workerLoop, this);
}

void EJitAsyncCompiler::stop() {
  if (!running_.exchange(false))
    return;

  stopping_ = true;
  queueCV_.notify_all();

  if (workerThread_.joinable())
    workerThread_.join();

  stopping_ = false;
}

void EJitAsyncCompiler::submitRequest(CompileRequest req) {
  // Dedup: skip if same key is already in flight
  {
    std::lock_guard<std::mutex> lock(inFlightMutex_);
    if (requestsInFlight_.count(req.ctx.cacheKey))
      return;
    requestsInFlight_.insert(req.ctx.cacheKey);
  }

  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    requestQueue_.push(std::move(req));
  }
  queueCV_.notify_one();
}

void EJitAsyncCompiler::workerLoop() {
  while (!stopping_) {
    CompileRequest req;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCV_.wait(lock, [this] {
        return !requestQueue_.empty() || stopping_;
      });
      if (stopping_)
        break;
      req = std::move(requestQueue_.front());
      requestQueue_.pop();
    }

    compileOne(req);

    {
      std::lock_guard<std::mutex> lock(inFlightMutex_);
      requestsInFlight_.erase(req.ctx.cacheKey);
    }
  }
}

void EJitAsyncCompiler::compileOne(const CompileRequest &req) {
  // Re-check time window state before compiling
  for (auto &dim : req.ctx.dimensions) {
    if (!runtimeState_.isActive(dim.periodName, dim.cellIdx))
      return;
  }

  if (!workerEngine_ || !syncCompiler_)
    return;

  auto result =
      syncCompiler_->compile(*workerEngine_, req.bitcodeData, req.ctx);

  if (result.funcPtr) {
    SmallVector<std::string, 4> deps;
    for (auto &dim : req.ctx.dimensions)
      deps.push_back(dim.periodName + "=" + std::to_string(dim.cellIdx));
    cache_.put(req.ctx.cacheKey, result.funcPtr, result.codeSize, deps);
  }
}

#endif // EJIT_FREESTANDING
