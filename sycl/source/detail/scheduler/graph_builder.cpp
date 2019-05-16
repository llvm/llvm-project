//===-- graph_builder.cpp - SYCL Graph Builder ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/exception.hpp>

#include <memory>
#include <set>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

// The function check whether two requirements overlaps or not. This
// information can be used to prove that executing two kernels that
// work on different parts of the memory object in parallel is legal.
static bool doOverlap(const Requirement *LHS, const Requirement *RHS) {
  // TODO: Implement check for one dimensional case only. It will be
  // enough for most of the cases because 2d and 3d sub-buffers cannot
  // be mapped to OpenCL's ones.
  return true;
}

// Returns record for the memory objects passed, nullptr if doesn't exist.
Scheduler::GraphBuilder::MemObjRecord *
Scheduler::GraphBuilder::getMemObjRecord(SYCLMemObjT *MemObject) {
  const auto It = std::find_if(MMemObjRecords.begin(), MMemObjRecords.end(),
                               [MemObject](const MemObjRecord &Record) {
                                 return Record.MMemObj == MemObject;
                               });
  return (MMemObjRecords.end() != It) ? &*It : nullptr;
}

// Returns record for the memory object requirement refers to, if doesn't
// exist, creates new one add populate it with initial alloca command.
Scheduler::GraphBuilder::MemObjRecord *
Scheduler::GraphBuilder::getOrInsertMemObjRecord(const QueueImplPtr &Queue,
                                                 Requirement *Req) {
  SYCLMemObjT *MemObject = Req->MSYCLMemObj;
  Scheduler::GraphBuilder::MemObjRecord *Record = getMemObjRecord(MemObject);
  if (nullptr != Record)
    return Record;

  // Construct requirement which describes full buffer because we allocate
  // only full-sized memory objects.
  Requirement AllocaReq(/*Offset*/ {0, 0, 0}, Req->MMemoryRange,
                        Req->MMemoryRange, access::mode::discard_write,
                        MemObject, Req->MDims, Req->MElemSize);

  AllocaCommand *AllocaCmd = new AllocaCommand(Queue, std::move(AllocaReq));
  MemObjRecord NewRecord{MemObject,
                         /*WriteLeafs*/ {AllocaCmd},
                         /*ReadLeafs*/ {},
                         {AllocaCmd},
                         false};

  MMemObjRecords.push_back(std::move(NewRecord));
  return &MMemObjRecords.back();
}

// Helper function which removes all values in Cmds from Leafs
void Scheduler::GraphBuilder::UpdateLeafs(
    const std::set<Command *> &Cmds,
    Scheduler::GraphBuilder::MemObjRecord *Record, Requirement *Req) {

  const bool ReadOnlyReq = Req->MAccessMode == access::mode::read;
  if(ReadOnlyReq)
    return;

  for (const Command *Cmd : Cmds) {
    auto NewEnd =
        std::remove(Record->MReadLeafs.begin(), Record->MReadLeafs.end(), Cmd);
    Record->MReadLeafs.erase(NewEnd, Record->MReadLeafs.end());

    NewEnd = std::remove(Record->MWriteLeafs.begin(), Record->MWriteLeafs.end(),
                         Cmd);
    Record->MWriteLeafs.erase(NewEnd, Record->MWriteLeafs.end());
  }
}

void Scheduler::GraphBuilder::AddNodeToLeafs(
    Scheduler::GraphBuilder::MemObjRecord *Record, Command *Cmd,
    Requirement *Req) {
  if (Req->MAccessMode == access::mode::read)
    Record->MReadLeafs.push_back(Cmd);
  else
    Record->MWriteLeafs.push_back(Cmd);
}

MemCpyCommand *
Scheduler::GraphBuilder::insertMemCpyCmd(MemObjRecord *Record, Requirement *Req,
                                         const QueueImplPtr &Queue,
                                         bool UseExclusiveQueue) {

  Requirement FullReq(/*Offset*/ {0, 0, 0}, Req->MMemoryRange,
                      Req->MMemoryRange, access::mode::read_write,
                      Req->MSYCLMemObj, Req->MDims, Req->MElemSize);

  std::set<Command *> Deps = findDepsForReq(Record, &FullReq, Queue);
  QueueImplPtr SrcQueue = (*Deps.begin())->getQueue();
  AllocaCommand *AllocaCmdDst = findAllocaForReq(Record, &FullReq, Queue);

  if (!AllocaCmdDst) {
    std::unique_ptr<AllocaCommand> AllocaCmdUniquePtr(
        new AllocaCommand(Queue, FullReq));

    if (!AllocaCmdUniquePtr)
      throw runtime_error("Out of host memory");

    Record->MAllocaCommands.push_back(AllocaCmdUniquePtr.get());
    AllocaCmdDst = AllocaCmdUniquePtr.release();
    Deps.insert(AllocaCmdDst);
  }

  AllocaCommand *AllocaCmdSrc = findAllocaForReq(Record, Req, SrcQueue);

  MemCpyCommand *MemCpyCmd = new MemCpyCommand(
      *AllocaCmdSrc->getAllocationReq(), AllocaCmdSrc, *Req, AllocaCmdDst,
      AllocaCmdSrc->getQueue(), AllocaCmdDst->getQueue(), UseExclusiveQueue);

  for (Command *Dep : Deps) {
    MemCpyCmd->addDep(DepDesc{Dep, &MemCpyCmd->MDstReq, AllocaCmdDst});
    Dep->addUser(MemCpyCmd);
  }
  UpdateLeafs(Deps, Record, Req);
  AddNodeToLeafs(Record, MemCpyCmd, &FullReq);
  return MemCpyCmd;
}

// The function adds copy operation of the up to date'st memory to the memory
// pointed by Req.
Command *Scheduler::GraphBuilder::addCopyBack(Requirement *Req) {

  QueueImplPtr HostQueue = Scheduler::getInstance().getDefaultHostQueue();
  SYCLMemObjT *MemObj = Req->MSYCLMemObj;
  Scheduler::GraphBuilder::MemObjRecord *Record = getMemObjRecord(MemObj);

  // Do nothing if there were no or only read operations with the memory object.
  if (nullptr == Record || !Record->MMemModified)
    return nullptr;

  std::set<Command *> Deps = findDepsForReq(Record, Req, HostQueue);
  QueueImplPtr SrcQueue = (*Deps.begin())->getQueue();
  AllocaCommand *SrcAllocaCmd = findAllocaForReq(Record, Req, SrcQueue);

  std::unique_ptr<MemCpyCommandHost> MemCpyCmdUniquePtr(
      new MemCpyCommandHost(*SrcAllocaCmd->getAllocationReq(), SrcAllocaCmd,
                            Req, std::move(SrcQueue), std::move(HostQueue)));

  if (!MemCpyCmdUniquePtr)
    throw runtime_error("Out of host memory");

  MemCpyCommandHost *MemCpyCmd = MemCpyCmdUniquePtr.release();
  for (Command *Dep : Deps) {
    MemCpyCmd->addDep(DepDesc{Dep, &MemCpyCmd->MDstReq, SrcAllocaCmd});
    Dep->addUser(MemCpyCmd);
  }

  UpdateLeafs(Deps, Record, Req);
  AddNodeToLeafs(Record, MemCpyCmd, Req);
  return MemCpyCmd;
}

// The function implements SYCL host accessor logic: host accessor
// should provide access to the buffer in user space, then during
// destruction the memory should be written back(if access mode is not read
// only) to the memory object. No operations with buffer allowed during host
// accessor lifetime.
Command *Scheduler::GraphBuilder::addHostAccessor(Requirement *Req,
                                                  EventImplPtr &RetEvent) {
  QueueImplPtr HostQueue = Scheduler::getInstance().getDefaultHostQueue();
  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  markModifiedIfWrite(Record, Req);

  std::set<Command *> Deps = findDepsForReq(Record, Req, HostQueue);
  QueueImplPtr SrcQueue = (*Deps.begin())->getQueue();

  AllocaCommand *SrcAllocaCmd = findAllocaForReq(Record, Req, SrcQueue);
  Requirement *SrcReq = SrcAllocaCmd->getAllocationReq();

  if (SrcQueue->is_host()) {
    MemCpyCommand *DevToHostCmd = insertMemCpyCmd(Record, Req, HostQueue);
    DevToHostCmd->setAccessorToUpdate(Req);
    RetEvent = DevToHostCmd->getEvent();
    return DevToHostCmd;
  }

  // Prepare "user" event that will block second operation(unmap of copy) until
  // host accessor is destructed.
  ContextImplPtr SrcContext = detail::getSyclObjImpl(SrcQueue->get_context());
  Req->BlockingEvent.reset(new detail::event_impl());
  Req->BlockingEvent->setContextImpl(SrcContext);
  cl_event &CLEvent = Req->BlockingEvent->getHandleRef();
  cl_int Error = CL_SUCCESS;
  CLEvent = clCreateUserEvent(SrcContext->getHandleRef(), &Error);
  CHECK_OCL_CODE(Error);

  // In case of memory is 1 dimensional and located on OpenCL device we
  // can use map/unmap operation.
  if (!SrcQueue->is_host() && Req->MDims == 1 &&
      Req->MAccessRange == Req->MMemoryRange) {

    std::unique_ptr<MapMemObject> MapCmdUniquePtr(
        new MapMemObject(*SrcReq, SrcAllocaCmd, Req, SrcQueue));

    /*
    [SYCL] Use exclusive queues for blocked commands.

    SYCL host accessor must wait in c'tor until the memory it provides
    access to is available on the host and should write memory back on
    destruction. No operations with the memory object are allowed
    during lifetime of the host accessor.

    In order to implement host accessor logic SYCL RT enqueues two tasks:
    read from device to host/map and write from host to device/unmap.
    The read/map operation should be completed during host accessor
    construction while write/unmap should be blocked until host accessor
    is destructed.

    To achieve blocking of write/unmap operation SYCL RT blocks it on
    user event then unblock during host accessor destruction.

    For the code:
    {
      ...
      Q.submit(...); // <== 1 Kernel
      auto HostAcc = Buf.get_access<...>(); // <== Host acc creation
      Q.submit(...); // <== 2 Kernel
    } // <== Host acc desctruction

    We generate the following graph(arrows represent dependencies)

              +-------------+
              |  1 Kernel   |
              +-------------+
                    ^
                    |
              +-------------+
              |  Read/Map   |  <== This task should be completed
              +-------------+      during host acc creation
                    ^
                    |
              +-------------+
              | Write/Unmap |  <== This is blocked by user event
              +-------------+      Can be completed after host acc
                    ^              desctruction
                    |
              +-------------+
              |  2 Kernel   |
              +-------------+

    And the following content in OpenCL command queue:

        +----------------------------------------------+
    Q1: | 1 Kernel | Read/Map | Write/Unmap | 2 Kernel |
        +----------------------------------------------+
                                      ^
                                      |
       This is blocked by user event -+

    This works fine, but for example below problems can happen:

    For the code:
    {
      ...
      Q.submit(...); // <== 1 Kernel
      auto HostAcc1 = Buf1.get_access<...>(); // <== Host acc 1 creation
      auto HostAcc2 = Buf2.get_access<...>(); // <== Host acc 2 creation
      Q.submit(...); // <== 2 Kernel
    } // <== Host acc 1 and 2 desctruction

    We generate the following graph(arrows represent dependencies)

                      +-------------+
                      |  1 Kernel   |
                      +-------------+
                            ^
            +---------------+---------------+
      +-------------+                +-------------+
      |  Read/Map 1 |                |  Read/Map 2 |  <== This task should be
      +-------------+                +-------------+       completed during host
            ^                               ^              acc creation
            |                               |
      +-------------+                +-------------+
      |Write/Unmap 1|                |Write/Unmap 2|  <== This is blocked by
      +-------------+                +-------------+      user event. Can be
            ^                               ^             completed after host
            +---------------+---------------+             accdesctruction
                      +-------------+
                      |  2 Kernel   |
                      +-------------+

    And the following content in OpenCL command queue:

        +-------------------------------------------------------------------+
    Q1: | 1K | Read/Map 1 | Write/Unmap 1 | Read/Map 2 | Write/Unmap 2 | 2K |
        +-------------------------------------------------------------------+
                                      ^                       ^
                                      |                       |
       This is blocked by user event -+-----------------------+

    In the situation above there is "Write/Unmap 1" command already in
    command queue which is blocked by user event and cannot be executed
    and "Read/Map 2" command which is enqueued after "Write/Unmap 1" but
    we should wait for the completion of this command before exiting
    construction of the second host accessor.

    Such cases is not supported in some OpenCL implementations. They
    assume that the commands the are submitted before one user waits on
    eventually completes.

    This patch workarounds problem by using separate(exclusive) queues for
    such tasks while still using one(common) queue for all other tasks.

    So, for the second example SYCL RT creates 3 OpenCL queues, where
    second and third queues are used for "Write/Unmap 1" and "Write/Unmap 2"
    command respectively:

        +-----------------------------------------------+
    Q1: | 1 Kernel | Read/Map 1 | Read/Map 2 | 2 Kernel |
        +-----------------------------------------------+

        +---------------+
    Q2: | Write/Unmap 1 |<----+
        +---------------+     |
                              |-----This is blocked by user event
        +---------------+     |
    Q3: | Write/Unmap 2 |<----+
        +---------------+
    */

    std::unique_ptr<UnMapMemObject> UnMapCmdUniquePtr(new UnMapMemObject(
        *SrcReq, SrcAllocaCmd, Req, SrcQueue, /*UseExclusiveQueue*/ true));

    if (!MapCmdUniquePtr || !UnMapCmdUniquePtr)
      throw runtime_error("Out of host memory");

    MapMemObject *MapCmd = MapCmdUniquePtr.release();
    for (Command *Dep : Deps) {
      MapCmd->addDep(DepDesc{Dep, &MapCmd->MDstReq, SrcAllocaCmd});
      Dep->addUser(MapCmd);
    }

    Command *UnMapCmd = UnMapCmdUniquePtr.release();
    UnMapCmd->addDep(DepDesc{MapCmd, &MapCmd->MDstReq, SrcAllocaCmd});
    MapCmd->addUser(UnMapCmd);

    UpdateLeafs(Deps, Record, Req);
    AddNodeToLeafs(Record, UnMapCmd, Req);

    UnMapCmd->addDep(Req->BlockingEvent);

    RetEvent = MapCmd->getEvent();
    return UnMapCmd;
  }

  // In other cases insert two mem copy operations.
  MemCpyCommand *DevToHostCmd = insertMemCpyCmd(Record, Req, HostQueue);
  DevToHostCmd->setAccessorToUpdate(Req);
  Command *HostToDevCmd =
      insertMemCpyCmd(Record, Req, SrcQueue, /*UseExclusiveQueue*/ true);
  HostToDevCmd->addDep(Req->BlockingEvent);

  RetEvent = DevToHostCmd->getEvent();
  return HostToDevCmd;
}

Command *Scheduler::GraphBuilder::addCGUpdateHost(
    std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr HostQueue) {
  // Dummy implementation of update host logic, just copy memory to the host
  // device. We could avoid copying if there is no allocation of host memory.

  CGUpdateHost *UpdateHost = (CGUpdateHost *)CommandGroup.get();
  Requirement *Req = UpdateHost->getReqToUpdate();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  return insertMemCpyCmd(Record, Req, HostQueue);
}

// The functions finds dependencies for the requirement. It starts searching
// from list of "leaf" commands for the record and check if the examining
// command can be executed in parallel with new one with regard to the memory
// object. If can, then continue searching through dependencies of that
// command. There are several rules used:
//
// 1. New and examined commands only read -> can bypass
// 2. New and examined commands has non-overlapping requirements -> can bypass
// 3. New and examined commands has different contexts -> cannot bypass
std::set<Command *>
Scheduler::GraphBuilder::findDepsForReq(MemObjRecord *Record, Requirement *Req,
                                        QueueImplPtr Queue) {
  sycl::context Context = Queue->get_context();
  std::set<Command *> RetDeps;
  std::set<Command *> Visited;
  const bool ReadOnlyReq = Req->MAccessMode == access::mode::read;

  std::vector<Command *> ToAnalyze;

  if (ReadOnlyReq)
    ToAnalyze = Record->MWriteLeafs;
  else {
    ToAnalyze = Record->MReadLeafs;
    ToAnalyze.insert(ToAnalyze.end(), Record->MWriteLeafs.begin(),
                     Record->MWriteLeafs.end());
  }

  while (!ToAnalyze.empty()) {
    Command *DepCmd = ToAnalyze.back();
    ToAnalyze.pop_back();

    std::vector<Command *> NewAnalyze;

    for (const DepDesc &Dep : DepCmd->MDeps) {
      if (Dep.MReq->MSYCLMemObj != Req->MSYCLMemObj)
        continue;

      bool CanBypassDep = false;
      // If both only read
      CanBypassDep |=
          Dep.MReq->MAccessMode == access::mode::read && ReadOnlyReq;

      // If not overlap
      CanBypassDep |= !doOverlap(Dep.MReq, Req);

      // Going through copying memory between contexts is not supported.
      if (Dep.MDepCommand)
        CanBypassDep &= Context == Dep.MDepCommand->getQueue()->get_context();

      if (!CanBypassDep) {
        RetDeps.insert(DepCmd);
        // No need to analyze deps of examining command as it's dependency
        // itself.
        NewAnalyze.clear();
        break;
      }

      if (Visited.insert(Dep.MDepCommand).second)
        NewAnalyze.push_back(Dep.MDepCommand);
    }
    ToAnalyze.insert(ToAnalyze.end(), NewAnalyze.begin(), NewAnalyze.end());
  }
  return RetDeps;
}

// The function searchs for the alloca command matching context and requirement.
AllocaCommand *Scheduler::GraphBuilder::findAllocaForReq(MemObjRecord *Record,
                                                         Requirement *Req,
                                                         QueueImplPtr Queue) {
  auto IsSuitableAlloca = [&Queue](const AllocaCommand *AllocaCmd) {
    return AllocaCmd->getQueue()->get_context() == Queue->get_context();
  };
  const auto It = std::find_if(Record->MAllocaCommands.begin(),
                               Record->MAllocaCommands.end(), IsSuitableAlloca);
  return (Record->MAllocaCommands.end() != It) ? *It : nullptr;
}

// The function sets MemModified flag in record if requirement has write access.
void Scheduler::GraphBuilder::markModifiedIfWrite(
    GraphBuilder::MemObjRecord *Record, Requirement *Req) {
  switch (Req->MAccessMode) {
  case access::mode::write:
  case access::mode::read_write:
  case access::mode::discard_write:
  case access::mode::discard_read_write:
  case access::mode::atomic:
    Record->MMemModified = true;
  case access::mode::read:
    break;
  }
}

Command *
Scheduler::GraphBuilder::addCG(std::unique_ptr<detail::CG> CommandGroup,
                               QueueImplPtr Queue) {
  std::vector<Requirement *> Reqs = CommandGroup->getRequirements();
  std::unique_ptr<ExecCGCommand> NewCmd(
      new ExecCGCommand(std::move(CommandGroup), Queue));
  if (!NewCmd)
    throw runtime_error("Out of host memory");

  for (Requirement *Req : Reqs) {
    MemObjRecord *Record = getOrInsertMemObjRecord(Queue, Req);
    markModifiedIfWrite(Record, Req);
    std::set<Command *> Deps = findDepsForReq(Record, Req, Queue);

    // If contexts of dependency and new command don't match insert
    // memcpy command.
    for (const Command *Dep : Deps)
      if (Dep->getQueue()->get_context() != Queue->get_context()) {
        // Cannot directly copy memory from OpenCL device to OpenCL device -
        // create to copies device->host and host->device.
        if (!Dep->getQueue()->is_host() && !Queue->is_host())
          insertMemCpyCmd(Record, Req,
                          Scheduler::getInstance().getDefaultHostQueue());
        insertMemCpyCmd(Record, Req, Queue);
        // Need to search for dependencies again as we modified the graph.
        Deps = findDepsForReq(Record, Req, Queue);
        break;
      }
    AllocaCommand *AllocaCmd = findAllocaForReq(Record, Req, Queue);

    for (Command *Dep : Deps) {
      NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd});
    }
  }

  // Set new command as user for dependencies and update leafs.
  for (DepDesc &Dep : NewCmd->MDeps) {
    Dep.MDepCommand->addUser(NewCmd.get());
    Requirement *Req = Dep.MReq;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    UpdateLeafs({Dep.MDepCommand}, Record, Req);
    AddNodeToLeafs(Record, NewCmd.get(), Req);
  }

  return NewCmd.release();
}

void Scheduler::GraphBuilder::cleanupCommands(bool CleanupReleaseCommands) {
  // TODO: Implement.
}

void Scheduler::GraphBuilder::removeRecordForMemObj(SYCLMemObjT *MemObject) {
  const auto It = std::find_if(MMemObjRecords.begin(), MMemObjRecords.end(),
                               [MemObject](const MemObjRecord &Record) {
                                 return Record.MMemObj == MemObject;
                               });
  MMemObjRecords.erase(It);
}

} // namespace detail
} // namespace sycl
} // namespace cl
