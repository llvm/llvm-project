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
  Requirement AllocaReq(/*Offset*/ {0, 0, 0}, Req->MOrigRange, Req->MOrigRange,
                        access::mode::discard_write, MemObject, Req->MDims,
                        Req->MElemSize);

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
                                         const QueueImplPtr &Queue) {

  Requirement FullReq(/*Offset*/ {0, 0, 0}, Req->MOrigRange, Req->MOrigRange,
                      access::mode::read_write, Req->MSYCLMemObj, Req->MDims,
                      Req->MElemSize);

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
      AllocaCmdSrc->getQueue(), AllocaCmdDst->getQueue());

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
      Req->MRange == Req->MOrigRange) {

    std::unique_ptr<MapMemObject> MapCmdUniquePtr(
        new MapMemObject(*SrcReq, SrcAllocaCmd, Req, SrcQueue));
    std::unique_ptr<UnMapMemObject> UnMapCmdUniquePtr(
        new UnMapMemObject(*SrcReq, SrcAllocaCmd, Req, SrcQueue));

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
  Command *HostToDevCmd = insertMemCpyCmd(Record, Req, SrcQueue);
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
    UpdateLeafs(Deps, Record, Req);

    for (Command *Dep : Deps) {
      NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd});
      Dep->addUser(NewCmd.get());
    }
  }

  for (Requirement *Req : Reqs) {
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
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
