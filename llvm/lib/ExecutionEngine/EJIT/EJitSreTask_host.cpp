//===-- EJitSreTask_host.cpp - Host task implementation -------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSreTask.h"

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
  if (out.handle_)
    return true;
  out.stopFlag_.storeRelease(0);
  out.entry_ = entry;
  out.ctx_ = ctx;
  auto *H = new (std::nothrow) HostTaskHandle(std::thread([&out]() {
    if (out.entry_)
      out.entry_(out.ctx_);
  }));
  if (!H)
    return false;
  out.handle_ = H;
  return true;
}

void EJitSreTask::destroy(EJitSreTask &task) {
  if (!task.handle_)
    return;
  task.stopFlag_.storeRelease(1);
  auto *H = static_cast<HostTaskHandle *>(task.handle_);
  if (H->th.joinable())
    H->th.join();
  delete H;
  task.handle_ = nullptr;
  task.entry_ = nullptr;
  task.ctx_ = nullptr;
}

void EJitSreTask::yield() { std::this_thread::yield(); }

#endif
