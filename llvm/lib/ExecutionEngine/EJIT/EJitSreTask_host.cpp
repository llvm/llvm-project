//===-- EJitSreTask_host.cpp - Host task implementation -------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSreTask.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"

#ifndef EJIT_FREESTANDING
#include <new>
#include <thread>

using namespace llvm;
using namespace llvm::ejit;

namespace {

struct HostTaskHandle {
  std::thread th;
  explicit HostTaskHandle(std::thread T) : th(std::move(T)) {}
};

} // namespace

bool EJitSreTask::create(EJitSreTask &out, EntryFn entry, void *ctx,
                         const char *name) {
  (void)name;
  if (out.handle_) {
    EJIT_DIAG("host task create ignored: handle=%p", out.handle_);
    return true;
  }
  EJIT_DIAG("host task create begin name=%s entry=%p ctx=%p",
            name ? name : "<null>", reinterpret_cast<void *>(entry), ctx);
  out.stopFlag_.storeRelease(0);
  out.entry_ = entry;
  out.ctx_ = ctx;
  auto *H = new (std::nothrow) HostTaskHandle(std::thread([&out]() {
    if (out.entry_)
      out.entry_(out.ctx_);
  }));
  if (!H) {
    EJIT_DIAG("host task create failed: out of memory");
    return false;
  }
  out.handle_ = H;
  EJIT_DIAG("host task create OK handle=%p", out.handle_);
  return true;
}

void EJitSreTask::destroy(EJitSreTask &task) {
  if (!task.handle_) {
    EJIT_DIAG("host task destroy ignored: no handle");
    return;
  }
  EJIT_DIAG("host task destroy begin handle=%p", task.handle_);
  task.stopFlag_.storeRelease(1);
  auto *H = static_cast<HostTaskHandle *>(task.handle_);
  if (H->th.joinable())
    H->th.join();
  delete H;
  task.handle_ = nullptr;
  task.entry_ = nullptr;
  task.ctx_ = nullptr;
  EJIT_DIAG("host task destroy complete");
}

void EJitSreTask::yield() { std::this_thread::yield(); }

#endif
