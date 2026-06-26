//===-- EJitWorker.cpp - Internal taskpool worker -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitWorker.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"

using namespace llvm;
using namespace llvm::ejit;

bool EJitWorker::start() {
  // Idempotent: a second start while already owning a task is a no-op success.
  uint32_t expected = 0;
  if (!started_.compareExchange(expected, 1)) {
    EJIT_DIAG("worker start ignored: already started");
    return true;
  }
  EJIT_DIAG("worker start name=%s", name_ ? name_ : "<null>");
  if (!EJitSreTask::create(task_, &EJitWorker::taskEntry, this, name_)) {
    started_.storeRelease(0);
    EJIT_DIAG("worker start failed name=%s", name_ ? name_ : "<null>");
    return false;
  }
  EJIT_DIAG("worker start accepted name=%s", name_ ? name_ : "<null>");
  return true;
}

void EJitWorker::stop() {
  // Idempotent: safe to call when never started or already stopped. destroy()
  // requests a soft stop and joins, so this waits for the loop to exit.
  if (!task_.handle_) {
    started_.storeRelease(0);
    EJIT_DIAG("worker stop ignored: no task");
    return;
  }
  EJIT_DIAG("worker stop begin");
  EJitSreTask::destroy(task_);
  started_.storeRelease(0);
  EJIT_DIAG("worker stop complete processed=%llu spins=%llu",
            static_cast<unsigned long long>(processed_.loadRelaxed()),
            static_cast<unsigned long long>(spins_.loadRelaxed()));
}

void EJitWorker::taskEntry(void *ctx) { static_cast<EJitWorker *>(ctx)->run(); }

void EJitWorker::run() {
  EJIT_DIAG("worker loop enter");
  running_.storeRelease(1);
  while (!task_.stopRequested()) {
    if (pool_.pollOne())
      processed_.fetchAdd(1);
    else {
      spins_.fetchAdd(1);
      // Idle hint so the single consumer does not spin the core at full speed.
      EJitSreTask::yield();
    }
  }
  running_.storeRelease(0);
  EJIT_DIAG("worker loop leave");
}
