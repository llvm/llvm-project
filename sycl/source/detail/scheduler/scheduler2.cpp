//===-- scheduler.cpp - SYCL Schedule ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/sycl_mem_obj.hpp"
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/device_selector.hpp>

#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

void Scheduler::waitForRecordToFinish(GraphBuilder::MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeafs) {
    Command *FailedCommand = GraphProcessor::enqueueCommand(Cmd);
    if (FailedCommand) {
      assert(!FailedCommand && "Command failed to enqueue");
      throw runtime_error("Enqueue process failed.");
    }
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (Command *Cmd : Record->MWriteLeafs) {
    Command *FailedCommand = GraphProcessor::enqueueCommand(Cmd);
    if (FailedCommand) {
      assert(!FailedCommand && "Command failed to enqueue");
      throw runtime_error("Enqueue process failed.");
    }
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (AllocaCommand *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    Command *FailedCommand = GraphProcessor::enqueueCommand(ReleaseCmd);
    if (FailedCommand) {
      assert(!FailedCommand && "Command failed to enqueue");
      throw runtime_error("Enqueue process failed.");
    }
    GraphProcessor::waitForEvent(ReleaseCmd->getEvent());
  }
}

EventImplPtr Scheduler::addCG(std::unique_ptr<detail::CG> CommandGroup,
                              QueueImplPtr Queue) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  Command *NewCmd = nullptr;
  switch (CommandGroup->getType()) {
  case CG::UPDATE_HOST:
    NewCmd = MGraphBuilder.addCGUpdateHost(std::move(CommandGroup),
                                           DefaultHostQueue);
    break;
  default:
    NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), std::move(Queue));
  }

  // TODO: Check if lazy mode.
  Command *FailedCommand = GraphProcessor::enqueueCommand(NewCmd);
  MGraphBuilder.cleanupCommands();
  if (FailedCommand)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.");
  return NewCmd->getEvent();
}

EventImplPtr Scheduler::addCopyBack(Requirement *Req) {
  Command *NewCmd = MGraphBuilder.addCopyBack(Req);
  // Command was not creted because there were no operations with
  // buffer.
  if (!NewCmd)
    return nullptr;
  Command *FailedCommand = GraphProcessor::enqueueCommand(NewCmd);
  if (FailedCommand)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.");
  return NewCmd->getEvent();
}

Scheduler::~Scheduler() {
  // TODO: Make running wait and release on destruction configurable?
  // TODO: Process release commands only?
  //std::lock_guard<std::mutex> lock(MGraphLock);
  //for (GraphBuilder::MemObjRecord &Record : MGraphBuilder.MMemObjRecords)
    //waitForRecordToFinish(&Record);
  //MGraphBuilder.cleanupCommands([>CleanupReleaseCommands = <] true);
}

Scheduler &Scheduler::getInstance() {
  static Scheduler instance;
  return instance;
}

std::vector<EventImplPtr> Scheduler::getWaitList(EventImplPtr Event) {
  std::lock_guard<std::mutex> lock(MGraphLock);
  return GraphProcessor::getWaitList(std::move(Event));
}

void Scheduler::waitForEvent(EventImplPtr Event) {
  std::lock_guard<std::mutex> lock(MGraphLock);
  GraphProcessor::waitForEvent(std::move(Event));
}

void Scheduler::removeMemoryObject(detail::SYCLMemObjT *MemObj) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  GraphBuilder::MemObjRecord *Record = MGraphBuilder.getMemObjRecord(MemObj);
  if (!Record) {
    assert("No operations were performed on the mem object?");
    return;
  }
  waitForRecordToFinish(Record);
  MGraphBuilder.cleanupCommands(/*CleanupReleaseCommands = */ true);
  MGraphBuilder.removeRecordForMemObj(MemObj);
}

EventImplPtr Scheduler::addHostAccessor(Requirement *Req) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  EventImplPtr RetEvent;
  Command *NewCmd = MGraphBuilder.addHostAccessor(Req, RetEvent);

  if (!NewCmd)
    return nullptr;
  Command *FailedCommand = GraphProcessor::enqueueCommand(NewCmd);
  if (FailedCommand)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.");
  return RetEvent;
}

Scheduler::Scheduler() {
  sycl::device HostDevice;
  DefaultHostQueue = QueueImplPtr(
      new queue_impl(HostDevice, /*AsyncHandler=*/{}, /*PropList=*/{}));
}

} // namespace detail
} // namespace sycl
} // namespace cl
