//===-- EJitCompileDriver.cpp - Compilation Scheduler ---------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCompileDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#ifndef EJIT_FREESTANDING
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#endif
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#ifdef EJIT_SRE_CODE_POOL
#include "llvm/ExecutionEngine/EJIT/EJitSrePlatform.h"
#endif
#ifdef EJIT_SRE_SHARED_TASKPOOL
#include "llvm/ExecutionEngine/EJIT/EJitFuncRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitLifecycleRegistry.h"
#endif
#ifndef EJIT_FREESTANDING
#include <chrono>
#endif

using namespace llvm;
using namespace llvm::ejit;

#ifdef EJIT_SRE_TASKPOOL
namespace {
/// Adapter so the taskpool can call back into the driver's cold compile path
/// through a plain function pointer (never std::function). The produced JIT
/// pointer still comes from the OrcJIT engine (SRE code pool when enabled).
bool taskpoolCompileThunk(void *ctx, const EJitCompileRequest &req,
                          void **outFn) {
  auto *drv = static_cast<EJitCompileDriver *>(ctx);
  void *fn = drv->compileNow(req);
  *outFn = fn;
  return fn != nullptr;
}

#ifdef EJIT_SRE_SHARED_TASKPOOL
bool sharedPrepareCodeThunk(void * /*ctx*/, const void *fnPtr) {
#ifdef EJIT_SRE_CODE_POOL
  return prepareSreCodeForCurrentCore(fnPtr);
#else
  (void)fnPtr;
  return false;
#endif
}
#endif
} // namespace
#endif

#ifdef EJIT_SRE_SHARED_TASKPOOL
namespace {
// The single process-global shared taskpool state. Placed in the cross-core
// shared section (an empty attribute on host, where one address space already
// exists). Every EJit instance's driver binds to THIS same blob and elects a
// single worker owner across cores via CAS.
EJIT_SHARED_SECTION EJitSharedTaskPoolState gEJitSharedTaskPoolState;
} // namespace
#endif

EJitCompileDriver::EJitCompileDriver(const Config &config, EJitCache &cache,
                                     EJitRuntimeState &runtimeState,
                                     EJitModuleLoader &loader,
                                     EJitLogger *logger)
    : config_(config), cache_(cache), runtimeState_(runtimeState),
      loader_(loader)
#ifndef EJIT_FREESTANDING
      ,
      logger_(logger)
#endif
{
#ifdef EJIT_SRE_TASKPOOL
  // Build the unified scheduler with the worker STOPPED. The worker must not
  // run until EJit has consumed all registration, completed the funcIndex/
  // lifecycle fixup, frozen registration, and installed the ORC engine — EJit
  // calls startTaskPoolWorker() once everything is ready (spec §3.4).
  taskPool_ = std::make_unique<EJitTaskPool>(EJIT_SRE_TASKPOOL_QUEUE_CAPACITY,
                                             /*autoStartWorker=*/false);
  taskPool_->setCompiler(&taskpoolCompileThunk, this);
  taskPool_->switchController().setMode(
      config_.compileMode == CompileMode::Async ? EJitCompileMode::Async
                                                : EJitCompileMode::Off);
#endif
#ifdef EJIT_SRE_SHARED_TASKPOOL
  // Bind the cross-core shared pool to the process-global shared state and wire
  // the owner-private hooks. Election + the single worker start happen later in
  // startSharedTaskPool() (called by EJit once registration is frozen and the
  // ORC engine is installed). Cross-core fnPtr sharing is OFF by default until
  // a platform asserts same-VA, sealed, I/D-coherent code (spec §11).
  sharedPool_.bind(&gEJitSharedTaskPoolState);
  sharedPool_.setCompiler(&taskpoolCompileThunk, this);
  sharedPool_.setWorkerHooks(&EJitCompileDriver::sharedWorkerStart,
                             &EJitCompileDriver::sharedWorkerStop, this);
  // Inject the platform yield so the worker never busy-spins while waiting for
  // Ready or on an empty queue (spec §11): a high-priority worker that spun
  // could starve the owner core trying to publish Ready / a producer enqueuing.
  sharedPool_.setWorkerIdleHook(&EJitCompileDriver::sharedWorkerIdle, this);
  sharedPool_.setMode(config_.compileMode == CompileMode::Async
                          ? EJitCompileMode::Async
                          : EJitCompileMode::Off);
  // Cross-core fnPtr sharing is gated by the build capability flag
  // EJIT_SRE_SHARED_CODE_POINTERS (default OFF -> clean fallback for non-owner
  // cores). Only the platform may assert same-VA + sealed + I/D-cache-coherent
  // code (spec §11); we never auto-detect it.
#ifdef EJIT_SRE_SHARED_CODE_POINTERS
  sharedPool_.setCodeSharingEnabled(true);
  sharedPool_.setPrepareCodeCallback(&sharedPrepareCodeThunk, this);
#else
  sharedPool_.setCodeSharingEnabled(false);
#endif
#endif
}

EJitCompileDriver::~EJitCompileDriver() {
#ifdef EJIT_SRE_SHARED_TASKPOOL
  // Stop + join the single shared worker (if this driver is the owner) BEFORE
  // owner-private ORC/driver state is destroyed — no use-after-free.
  sharedPool_.ownerShutdown();
#endif
}

#ifdef EJIT_SRE_SHARED_TASKPOOL
bool EJitCompileDriver::sharedWorkerStart(
    void *ctx, EJitSharedTaskPool::WorkerEntryFn entry, void *entryCtx,
    uint64_t *outTaskId) {
  auto *drv = static_cast<EJitCompileDriver *>(ctx);
  if (!EJitSreTask::create(drv->sharedWorkerTask_, entry, entryCtx,
                           "ejit-shared-worker"))
    return false;
  if (outTaskId)
    *outTaskId = 1; // host has no numeric task id; diagnostic only.
  return true;
}

void EJitCompileDriver::sharedWorkerStop(void *ctx) {
  EJitSreTask::destroy(
      static_cast<EJitCompileDriver *>(ctx)->sharedWorkerTask_);
}

void EJitCompileDriver::sharedWorkerIdle(void * /*ctx*/) {
  // Platform yield: SRE_TaskDelay(1) on freestanding, std::this_thread::yield()
  // on host. The shared taskpool core never names SRE_TaskDelay directly.
  EJitSreTask::yield();
}

bool EJitCompileDriver::startSharedTaskPool() {
  // Publish this core's funcIndex/dimType registration digest so a peer with a
  // divergent mapping is cleanly rejected at attach (spec §11), never silently
  // running against mismatched indices.
  sharedPool_.setRegistrationFingerprint(
      EJitFuncRegistry::instance().fingerprint() * 0x9e3779b97f4a7c15ULL ^
      EJitLifecycleRegistry::instance().fingerprint());
  switch (sharedPool_.init()) {
  case EJitSharedTaskPool::InitResult::BecameOwner:
  case EJitSharedTaskPool::InitResult::AttachedReady:
    return true;
  case EJitSharedTaskPool::InitResult::OwnerFailed:
  case EJitSharedTaskPool::InitResult::InitInProgress:
  case EJitSharedTaskPool::InitResult::AbiMismatch:
  case EJitSharedTaskPool::InitResult::FingerprintMismatch:
  case EJitSharedTaskPool::InitResult::NoState:
    return false;
  }
  return false;
}
#endif

void EJitCompileDriver::setSyncEngine(std::unique_ptr<EJitOrcEngine> engine) {
  syncEngine_ = std::move(engine);
}

void EJitCompileDriver::registerSymbol(const std::string &name, void *addr) {
  if (syncEngine_)
    syncEngine_->addUserSymbol(name, addr);
}

void *EJitCompileDriver::getOrCompile(uint64_t cacheKey) {
  // Legacy ABI (ejit_compile_or_get): this entry point has no bucket/release
  // capability, so it must NOT enter the taskpool's read-token cache — doing so
  // would hand out a JIT pointer whose read token is released before the caller
  // ever executes it (use-after-free window). The legacy ABI therefore always
  // uses the LRU EJitCache path below. Only the new AOT wrapper ABI
  // (ejit_taskpool_compile_or_get) drives the taskpool, where the caller owns
  // the bucket and calls ejit_taskpool_release_read after using fnPtr.

  // ── Hot path: single hash find ───────────────────────────────────────────
  if (void *cached = cache_.getOrNull(cacheKey)) {
    EJIT_DIAG("cache HIT key=0x%016lx", cacheKey);
    return cached;
  }

  return compileCold(cacheKey, /*storeLru=*/true);
}

void *EJitCompileDriver::compileCold(uint64_t cacheKey, bool storeLru) {
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
  if (funcName.empty()) {
    EJIT_DIAG("cache MISS key=0x%016lx funcIdx=%u: unknown funcIdx", cacheKey,
              funcIdx);
    return nullptr;
  }

  EJIT_DIAG("cache MISS key=0x%016lx func=%s dims=[%u,%u,%u,%u]", cacheKey,
            funcName.c_str(), dims[0], dims[1], dims[2], dims[3]);

  // Get bitcode
  auto bitcodeOrErr = loader_.getBitcodeByFuncIdx(funcIdx);
  if (!bitcodeOrErr) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: bitcode not found", cacheKey,
              funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_BITCODE_NOT_FOUND, "No bitcode for function",
                   funcName, std::to_string(cacheKey));
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
#ifdef EJIT_SRE_SHARED_TASKPOOL
    // Cross-core: gate on the SHARED enabled bit (the one the producer's
    // ejit_activate writes), NOT the owner-private runtimeState_. In an
    // owner!=producer split the worker compiles on the owner core, whose
    // private runtimeState_ is empty (the owner never calls activate_all), so
    // the legacy gate would reject every compile. The shared SwitchController
    // defaults to active; a deactivate flips the bit + bumps version. Race
    // protection during compilation is handled by runCompile's version
    // checkpoints (cp1/cp2), not this gate.
    uint32_t dt = meta.dimTypes[i];
    if (dt == kEJitInvalidDimType || !sharedPool_.isInstanceActive(dt, dims[i])) {
      EJIT_DIAG("compile SKIP key=0x%016lx func=%s: period %s[%u] not active",
                cacheKey, funcName.c_str(), periodNames[i].c_str(), dims[i]);
      return nullptr;
    }
#else
    if (!runtimeState_.isActive(periodNames[i], dims[i])) {
      EJIT_DIAG("compile SKIP key=0x%016lx func=%s: period %s[%u] not active",
                cacheKey, funcName.c_str(), periodNames[i].c_str(), dims[i]);
#ifndef EJIT_FREESTANDING
      if (logger_)
        logger_->log(EJIT_ERR_NOT_ACTIVE,
                     "Time window not active for " + periodNames[i], funcName,
                     std::to_string(cacheKey));
#endif
      return nullptr;
    }
#endif
  }

  // Build specialization context
  SpecializationContext ctx;
  ctx.fnName = funcName;
  ctx.cacheKey = cacheKey;
  ctx.optLevel = config_.optLevel;
  for (unsigned i = 0; i < dimCount; ++i)
    ctx.dimensions.push_back({periodNames[i], dims[i]});

  if (!syncEngine_) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: no sync engine", cacheKey,
              funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_NOT_ACTIVE, "Sync engine not initialized", funcName,
                   std::to_string(cacheKey));
#endif
    return nullptr;
  }

  syncEngine_->setActiveContext(&ctx);

  if (auto Err = syncEngine_->loadBitcodeModule(bitcode, cacheKey, funcName)) {
    syncEngine_->setActiveContext(nullptr);
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: load bitcode module failed",
              cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_COMPILE_FAILED, "Failed to load bitcode module",
                   funcName, std::to_string(cacheKey));
#else
    consumeError(std::move(Err));
#endif
    return nullptr;
  }

  auto addrOrErr = syncEngine_->lookup(cacheKey, funcName);
  syncEngine_->setActiveContext(nullptr);

  if (!addrOrErr) {
    EJIT_DIAG("compile FAIL key=0x%016lx func=%s: lookup after compile failed",
              cacheKey, funcName.c_str());
#ifndef EJIT_FREESTANDING
    if (logger_)
      logger_->log(EJIT_ERR_COMPILE_FAILED,
                   "Failed to look up compiled function", funcName,
                   std::to_string(cacheKey));
#else
    consumeError(addrOrErr.takeError());
#endif
    return nullptr;
  }

  void *funcPtr = *addrOrErr;

  // storeLru is false on the taskpool path: the taskpool publishes to its own
  // fixed cache and the LRU EJitCache is bypassed in that configuration.
  if (storeLru) {
    SmallVector<std::string, 4> periodDeps;
    for (unsigned i = 0; i < dimCount; ++i)
      periodDeps.push_back(periodNames[i] + "=" + std::to_string(dims[i]));

    cache_.put(cacheKey, funcPtr, bitcode.size(), periodDeps);
  }

  EJIT_DIAG("compile OK key=0x%016lx func=%s → pfn=%p", cacheKey,
            funcName.c_str(), funcPtr);
  return funcPtr;
}

#ifdef EJIT_SRE_TASKPOOL
void *EJitCompileDriver::compileNow(const EJitCompileRequest &req) {
  if (req.numDims > 4)
    return nullptr;

  // Validate the request: instanceIds must be encodable in the legacy 8-bit
  // cacheKey slots, and no two dims may share a dimType (a duplicated lifecycle
  // dimension).
  SmallVector<uint32_t, 4> seenDimTypes;
  for (uint32_t i = 0; i < req.numDims; ++i) {
    if (req.dims[i].instanceId > 255u)
      return nullptr;
    if (llvm::is_contained(seenDimTypes, req.dims[i].dimType))
      return nullptr;
    seenDimTypes.push_back(req.dims[i].dimType);
  }

  // meta.dimTypes[i] is the explicit dimType slot the loader read back BY NAME
  // from the process-global EJitLifecycleRegistry — the SAME slot the wrapper
  // baked into req.dims via its per-lifecycle global. No
  // recomputation/guessing.
  const auto &meta = loader_.getOrCacheFuncMeta(req.funcIndex);
  uint8_t packedDims[4] = {0, 0, 0, 0};
  for (unsigned i = 0; i < meta.dimCount && i < 4; ++i) {
    uint32_t wantedType = meta.dimTypes[i];
    if (wantedType == kEJitInvalidDimType)
      return nullptr;
    bool found = false;
    for (uint32_t j = 0; j < req.numDims; ++j) {
      if (req.dims[j].dimType == wantedType) {
        packedDims[i] = static_cast<uint8_t>(req.dims[j].instanceId);
        found = true;
        break;
      }
    }
    if (!found)
      return nullptr;
  }

  uint64_t cacheKey = (static_cast<uint64_t>(req.funcIndex) << 32) |
                      static_cast<uint64_t>(packedDims[0]) |
                      (static_cast<uint64_t>(packedDims[1]) << 8) |
                      (static_cast<uint64_t>(packedDims[2]) << 16) |
                      (static_cast<uint64_t>(packedDims[3]) << 24);
  return compileCold(cacheKey, /*storeLru=*/false);
}
#endif
