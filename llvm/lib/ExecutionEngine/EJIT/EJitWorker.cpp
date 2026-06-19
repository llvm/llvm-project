//===-- EJitWorker.cpp - Internal taskpool worker -------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitWorker.h"
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"

using namespace llvm;
using namespace llvm::ejit;

bool EJitWorker::start() {
  // Idempotent: a second start while already owning a task is a no-op success.
  uint32_t expected = 0;
  if (!started_.compareExchange(expected, 1))
    return true;
  if (!EJitSreTask::create(task_, &EJitWorker::taskEntry, this, name_)) {
    started_.storeRelease(0);
    return false;
  }
  return true;
}

void EJitWorker::stop() {
  // Idempotent: safe to call when never started or already stopped. destroy()
  // requests a soft stop and joins, so this waits for the loop to exit.
  if (!task_.handle_) {
    started_.storeRelease(0);
    return;
  }
  EJitSreTask::destroy(task_);
  started_.storeRelease(0);
}

void EJitWorker::taskEntry(void *ctx) { static_cast<EJitWorker *>(ctx)->run(); }

void EJitWorker::run() {
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
}
