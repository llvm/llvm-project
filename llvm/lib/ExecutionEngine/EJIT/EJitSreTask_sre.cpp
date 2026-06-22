//===-- EJitSreTask_sre.cpp - SRE task implementation shim ---------------===//

#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreTask.h"

#ifdef EJIT_FREESTANDING

#include <cstdint>

using namespace llvm;
using namespace llvm::ejit;

using TSK_PRIOR_T = uint16_t;
using UINTARG = uint64_t;
using TSK_ARG_T = UINTARG;
using TSK_ENTRY_FUNC = void (*)(TSK_ARG_T, TSK_ARG_T, TSK_ARG_T, TSK_ARG_T);

struct TSK_INIT_PARAM_S {
  TSK_ENTRY_FUNC pfnTaskEntry;
  TSK_PRIOR_T usTaskPrio;
  uint16_t usResved;
  TSK_ARG_T auwArgs[4];
  uint32_t uwStackSize;
  const char *pcName;
  uint32_t uwResved;
  TSK_ARG_T uwPrivateData;
};

namespace {

#ifndef EJIT_SRE_TASK_PRIORITY
#define EJIT_SRE_TASK_PRIORITY 20u
#endif

// Worker task stack size: the SINGLE source of truth is the CMake option
// EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE (default 1 MiB, defined whenever
// EJIT_SRE_TASKPOOL is ON). The LLVM optimize/codegen/ORC/JITLink pipeline runs
// on THIS stack, so 64 KiB is not enough. The fallback below exists ONLY for a
// non-CMake compile; CMake builds always pass -D so it is never used there.
#ifndef EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE
#define EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE (1024u * 1024u)
#endif
static_assert(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE > 0u,
              "worker stack size must be non-zero");
static_assert(
    EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE % 16u == 0u,
    "worker stack size must be 16-byte aligned (AArch64 SP alignment)");
static_assert(
    static_cast<unsigned long long>(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE) <=
        0xFFFFFFFFull,
    "worker stack size must fit the 32-bit TSK_INIT_PARAM_S.uwStackSize");

constexpr uint32_t kSreOk = 0;
constexpr const char kDefaultTaskName[] = "ejit_worker";

void taskTrampoline(TSK_ARG_T entryValue, TSK_ARG_T ctxValue, TSK_ARG_T,
                    TSK_ARG_T) {
  auto entry = reinterpret_cast<EJitSreTask::EntryFn>(
      static_cast<uintptr_t>(entryValue));
  void *ctx = reinterpret_cast<void *>(static_cast<uintptr_t>(ctxValue));
  if (entry)
    entry(ctx);
}

void *encodeTaskPid(uint32_t taskPid) {
  // handle_ uses nullptr as "not created", so store PID + 1.
  return reinterpret_cast<void *>(static_cast<uintptr_t>(taskPid) + 1u);
}

uint32_t decodeTaskPid(const void *handle) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(handle) - 1u);
}

} // namespace

extern "C" {
uint32_t SRE_TaskCreate(uint32_t *taskPid, TSK_INIT_PARAM_S *initParam);
uint32_t SRE_TaskDelete(uint32_t taskPid);
uint32_t SRE_TaskDelay(uint32_t tick);
}

bool EJitSreTask::create(EJitSreTask &out, EntryFn entry, void *ctx,
                         const char *name) {
  if (out.handle_) {
    EJIT_DIAG("platform task create ignored: handle=%p", out.handle_);
    return true;
  }
  EJIT_DIAG("platform task create begin name=%s entry=%p ctx=%p",
            name ? name : "<null>", reinterpret_cast<void *>(entry), ctx);

  TSK_INIT_PARAM_S init{};
  init.pfnTaskEntry = &taskTrampoline;
  init.usTaskPrio = static_cast<TSK_PRIOR_T>(EJIT_SRE_TASK_PRIORITY);
  init.auwArgs[0] = static_cast<TSK_ARG_T>(reinterpret_cast<uintptr_t>(entry));
  init.auwArgs[1] = static_cast<TSK_ARG_T>(reinterpret_cast<uintptr_t>(ctx));
  init.uwStackSize = static_cast<uint32_t>(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE);
  init.pcName = name ? name : kDefaultTaskName;
  EJIT_DIAG("platform task stack size=%u bytes",
            static_cast<unsigned>(init.uwStackSize));

  out.stopFlag_.storeRelease(0);
  out.entry_ = entry;
  out.ctx_ = ctx;

  uint32_t taskPid = 0;
  uint32_t result = SRE_TaskCreate(&taskPid, &init);
  if (result != kSreOk) {
    out.entry_ = nullptr;
    out.ctx_ = nullptr;
    EJIT_DIAG("platform task create failed rc=%u", result);
    return false;
  }

  out.handle_ = encodeTaskPid(taskPid);
  EJIT_DIAG("platform task create end pid=%u handle=%p", taskPid, out.handle_);
  return true;
}

void EJitSreTask::destroy(EJitSreTask &task) {
  if (!task.handle_) {
    EJIT_DIAG("platform task destroy ignored: no handle");
    return;
  }
  uint32_t taskPid = decodeTaskPid(task.handle_);
  EJIT_DIAG("platform task destroy begin pid=%u handle=%p", taskPid,
            task.handle_);
  task.stopFlag_.storeRelease(1);
  // Contract (spec §11): SRE_TaskDelete MUST block until the worker task has
  // actually exited (a real join), so the owner can safely destroy its private
  // ORC/driver afterwards. The worker loop also exits on its own when it sees
  // the Stopping/Uninitialized state, and the shared dedup is generation-aware,
  // so even a platform whose delete is not a perfect join cannot corrupt a new
  // generation's in-flight slots — but a true join is still required to avoid a
  // worker callback touching a destroyed driver.
  uint32_t result = SRE_TaskDelete(taskPid);
  EJIT_DIAG("platform task destroy end pid=%u rc=%u", taskPid, result);
  task.handle_ = nullptr;
  task.entry_ = nullptr;
  task.ctx_ = nullptr;
}

// Let the worker sleep for one scheduler tick instead of spinning on an empty
// queue. The queue/state visibility contract remains in EJitAtomic acquire /
// release operations; TaskDelay is only a scheduling hint.
void EJitSreTask::yield() { (void)SRE_TaskDelay(1); }

#endif
