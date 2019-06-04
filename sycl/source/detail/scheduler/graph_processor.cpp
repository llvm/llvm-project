//===-- graph_processor.cpp - SYCL Graph Processor --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

#include <memory>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

static Command *getCommand(const EventImplPtr &Event) {
  return (Command *)Event->getCommand();
}

std::vector<EventImplPtr>
Scheduler::GraphProcessor::getWaitList(EventImplPtr Event) {
  std::vector<EventImplPtr> Result;
  Command *Cmd = getCommand(Event);
  for (const DepDesc &Dep : Cmd->MDeps) {
    if (Dep.MDepCommand)
      Result.push_back(Dep.MDepCommand->getEvent());
  }
  return Result;
}

void Scheduler::GraphProcessor::waitForEvent(EventImplPtr Event) {
  Command *Cmd = getCommand(Event);
  assert(Cmd && "Event has no associated command?");
  Command *FailedCommand = enqueueCommand(Cmd);
  if (FailedCommand)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.");

  cl_event &CLEvent = Cmd->getEvent()->getHandleRef();
  if (CLEvent)
    CHECK_OCL_CODE(clWaitForEvents(1, &CLEvent));
}

Command *Scheduler::GraphProcessor::enqueueCommand(Command *Cmd) {
  if (!Cmd || Cmd->isEnqueued())
    return nullptr;

  for (DepDesc &Dep : Cmd->MDeps) {
    Command *FailedCommand = enqueueCommand(Dep.MDepCommand);
    if (FailedCommand)
      return FailedCommand;
  }

  cl_int Result = Cmd->enqueue();
  return CL_SUCCESS == Result ? nullptr : Cmd;
}

} // namespace detail
} // namespace sycl
} // namespace cl
