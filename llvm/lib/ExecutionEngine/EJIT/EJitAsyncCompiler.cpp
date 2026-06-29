//===-- EJitAsyncCompiler.cpp - Asynchronous JIT Compiler -----------------===//

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitAsyncCompiler.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
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
  EJIT_DIAG("async compiler start");

  syncCompiler_ = std::make_unique<EJitSyncCompiler>();

  // Create isolated engine for the worker thread
  auto engine = EJitOrcEngine::Create(
      config_, runtimeState_.getRegistry(), runtimeState_);
  if (engine) {
    workerEngine_ = std::move(*engine);
  } else {
    EJIT_DIAG("async compiler start FAIL: worker engine create failed");
    consumeError(engine.takeError());
  }

  workerThread_ = std::thread(&EJitAsyncCompiler::workerLoop, this);
  EJIT_DIAG("async compiler worker thread started");
}

void EJitAsyncCompiler::stop() {
  if (!running_.exchange(false))
    return;
  EJIT_DIAG("async compiler stop begin");

  stopping_ = true;
  queueCV_.notify_all();

  if (workerThread_.joinable())
    workerThread_.join();

  stopping_ = false;
  EJIT_DIAG("async compiler stop complete");
}

void EJitAsyncCompiler::submitRequest(CompileRequest req) {
  EJIT_DIAG("async submit key=0x%016lx func=%s", req.ctx.cacheKey,
            req.ctx.fnName.c_str());
  // Dedup: skip if same key is already in flight
  {
    std::lock_guard<std::mutex> lock(inFlightMutex_);
    if (requestsInFlight_.count(req.ctx.cacheKey)) {
      EJIT_DIAG("async submit key=0x%016lx: dropped, already in flight",
                req.ctx.cacheKey);
      return;
    }
    requestsInFlight_.insert(req.ctx.cacheKey);
  }

  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    requestQueue_.push(std::move(req));
  }
  queueCV_.notify_one();
}

void EJitAsyncCompiler::workerLoop() {
  EJIT_DIAG("async worker loop enter");
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
  EJIT_DIAG("async worker loop leave");
}

void EJitAsyncCompiler::compileOne(const CompileRequest &req) {
  EJIT_DIAG("async compileOne key=0x%016lx func=%s", req.ctx.cacheKey,
            req.ctx.fnName.c_str());
  // Re-check time window state before compiling
  for (auto &dim : req.ctx.dimensions) {
    if (!runtimeState_.isActive(dim.periodName, dim.cellIdx)) {
      EJIT_DIAG("async compileOne SKIP key=0x%016lx: period %s[%u] not active",
                req.ctx.cacheKey, dim.periodName.c_str(), dim.cellIdx);
      return;
    }
  }

  if (!workerEngine_ || !syncCompiler_) {
    EJIT_DIAG("async compileOne SKIP key=0x%016lx: engine=%p sync=%p",
              req.ctx.cacheKey, workerEngine_.get(), syncCompiler_.get());
    return;
  }

  auto result =
      syncCompiler_->compile(*workerEngine_, req.bitcodeData, req.ctx);

  if (result.funcPtr) {
    SmallVector<std::string, 4> deps;
    for (auto &dim : req.ctx.dimensions)
      deps.push_back(dim.periodName + "=" + std::to_string(dim.cellIdx));
    cache_.put(req.ctx.cacheKey, result.funcPtr, result.codeSize, deps);
    EJIT_DIAG("async compileOne OK key=0x%016lx pfn=%p cached", req.ctx.cacheKey,
              result.funcPtr);
  } else {
    EJIT_DIAG("async compileOne FAIL key=0x%016lx: no funcPtr", req.ctx.cacheKey);
  }
}

#endif // EJIT_FREESTANDING
