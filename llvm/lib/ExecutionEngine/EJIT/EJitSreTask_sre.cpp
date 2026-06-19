//===-- EJitSreTask_sre.cpp - SRE task implementation shim ---------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSreTask.h"

#ifdef EJIT_FREESTANDING

using namespace llvm;
using namespace llvm::ejit;

extern "C" {
void *ejit_sre_task_create(EJitSreTask::EntryFn Entry, void *Ctx,
                           const char *Name);
void ejit_sre_task_join_destroy(void *Handle);
}

bool EJitSreTask::create(EJitSreTask &out, EntryFn entry, void *ctx,
                         const char *name) {
  if (out.handle_)
    return true;
  out.stopFlag_.storeRelease(0);
  out.entry_ = entry;
  out.ctx_ = ctx;
  out.handle_ = ejit_sre_task_create(entry, ctx, name);
  return out.handle_ != nullptr;
}

void EJitSreTask::destroy(EJitSreTask &task) {
  if (!task.handle_)
    return;
  task.stopFlag_.storeRelease(1);
  ejit_sre_task_join_destroy(task.handle_);
  task.handle_ = nullptr;
  task.entry_ = nullptr;
  task.ctx_ = nullptr;
}

// Idle hint for the worker loop. Uses the AArch64 `yield` hint instruction when
// available (a CPU hint, not a function call — no platform symbol dependency),
// otherwise a compiler reordering barrier. The taskpool never *requires* an
// external yield/pause symbol.
void EJitSreTask::yield() {
#if defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}

#endif
