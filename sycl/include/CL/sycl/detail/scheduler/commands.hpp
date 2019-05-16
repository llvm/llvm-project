//==-------------- commands.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/cg.hpp>

namespace cl {
namespace sycl {
namespace detail {

class queue_impl;
class event_impl;
class context_impl;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

class Command;
class AllocaCommand;
class ReleaseCommand;

// DepDesc represents dependency between two commands
struct DepDesc {
  DepDesc(Command *DepCommand, Requirement *Req, AllocaCommand *AllocaCmd)
      : MDepCommand(DepCommand), MReq(Req), MAllocaCmd(AllocaCmd) {}

  friend bool operator<(const DepDesc &Lhs, const DepDesc &Rhs) {
    return std::tie(Lhs.MReq, Lhs.MDepCommand) <
           std::tie(Rhs.MReq, Rhs.MDepCommand);
  }

  // The actual dependency command.
  Command *MDepCommand = nullptr;
  // Requirement for the dependency.
  Requirement *MReq = nullptr;
  // Allocation command for the memory object we have requirement for.
  // Used to simplify searching for memory handle.
  AllocaCommand *MAllocaCmd = nullptr;
};

// The Command represents some action that needs to be performed on one or more
// memory objects. The command has vector of Depdesc objects that represent
// dependencies of the command. It has vector of pointer to commands that depend
// on the command. It has pointer to sycl::queue object. And has event that is
// associated with the command.
class Command {
public:
  enum CommandType {
    RUN_CG,
    COPY_MEMORY,
    ALLOCA,
    RELEASE,
    MAP_MEM_OBJ,
    UNMAP_MEM_OBJ
  };

  Command(CommandType Type, QueueImplPtr Queue, bool UseExclusiveQueue = false);

  void addDep(DepDesc NewDep) {
    if (NewDep.MDepCommand)
      MDepsEvents.push_back(NewDep.MDepCommand->getEvent());
    MDeps.push_back(NewDep);
  }

  void addDep(EventImplPtr Event) { MDepsEvents.push_back(std::move(Event)); }

  void addUser(Command *NewUser) { MUsers.push_back(NewUser); }

  // Return type of the command, e.g. Allocate, MemoryCopy.
  CommandType getType() const { return MType; }

  // The method checks if the command is enqueued, call enqueueImp if not and
  // returns CL_SUCCESS on success.
  cl_int enqueue();

  bool isFinished();

  bool isEnqueued() const { return MEnqueued; }

  std::shared_ptr<queue_impl> getQueue() const { return MQueue; }

  std::shared_ptr<event_impl> getEvent() const { return MEvent; }

protected:
  EventImplPtr MEvent;
  QueueImplPtr MQueue;
  std::vector<EventImplPtr> MDepsEvents;

  std::vector<cl_event> prepareEvents(ContextImplPtr Context);

  bool MUseExclusiveQueue = false;

  // Private interface. Derived classes should implement this method.
  virtual cl_int enqueueImp() = 0;

public:
  std::vector<DepDesc> MDeps;
  std::vector<Command *> MUsers;

private:
  CommandType MType;
  std::atomic<bool> MEnqueued;
};

// The command enqueues release instance of memory allocated on Host or
// underlying framework.
class ReleaseCommand : public Command {
public:
  ReleaseCommand(QueueImplPtr Queue, AllocaCommand *AllocaCmd)
      : Command(CommandType::RELEASE, std::move(Queue)), MAllocaCmd(AllocaCmd) {
  }
private:
  cl_int enqueueImp() override;

  AllocaCommand *MAllocaCmd = nullptr;
};

// The command enqueues allocation of instance of memory object on Host or
// underlying framework.
class AllocaCommand : public Command {
public:
  AllocaCommand(QueueImplPtr Queue, Requirement Req,
                bool InitFromUserData = true)
      : Command(CommandType::ALLOCA, Queue), MReleaseCmd(Queue, this),
        MInitFromUserData(InitFromUserData), MReq(std::move(Req)) {
    addDep(DepDesc(nullptr, &MReq, this));
  }
  ReleaseCommand *getReleaseCmd() { return &MReleaseCmd; }

  SYCLMemObjT *getSYCLMemObj() const { return MReq.MSYCLMemObj; }

  void *getMemAllocation() const { return MMemAllocation; }

  Requirement *getAllocationReq() { return &MReq; }

private:
  cl_int enqueueImp() override;

  ReleaseCommand MReleaseCmd;
  void *MMemAllocation = nullptr;
  bool MInitFromUserData = false;
  Requirement MReq;
};

class MapMemObject : public Command {
public:
  MapMemObject(Requirement SrcReq, AllocaCommand *SrcAlloca,
               Requirement *DstAcc, QueueImplPtr Queue);

  Requirement MSrcReq;
  AllocaCommand *MSrcAlloca = nullptr;
  Requirement *MDstAcc = nullptr;
  Requirement MDstReq;

private:
  cl_int enqueueImp() override;
};

class UnMapMemObject : public Command {
public:
  UnMapMemObject(Requirement SrcReq, AllocaCommand *SrcAlloca,
                 Requirement *DstAcc, QueueImplPtr Queue,
                 bool UseExclusiveQueue = false);

private:
  cl_int enqueueImp() override;

  Requirement MSrcReq;
  AllocaCommand *MSrcAlloca = nullptr;
  Requirement *MDstAcc = nullptr;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommand : public Command {
public:
  MemCpyCommand(Requirement SrcReq, AllocaCommand *SrcAlloca,
                Requirement DstReq, AllocaCommand *DstAlloca,
                QueueImplPtr SrcQueue, QueueImplPtr DstQueue,
                bool UseExclusiveQueue = false);

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommand *MSrcAlloca = nullptr;
  Requirement MDstReq;
  AllocaCommand *MDstAlloca = nullptr;
  Requirement *MAccToUpdate = nullptr;

  void setAccessorToUpdate(Requirement *AccToUpdate) {
    MAccToUpdate = AccToUpdate;
  }

private:
  cl_int enqueueImp() override;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommandHost : public Command {
public:
  MemCpyCommandHost(Requirement SrcReq, AllocaCommand *SrcAlloca,
                    Requirement *DstAcc, QueueImplPtr SrcQueue,
                    QueueImplPtr DstQueue);

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommand *MSrcAlloca = nullptr;
  Requirement MDstReq;
  Requirement *MDstAcc = nullptr;

private:
  cl_int enqueueImp() override;
};

// The command enqueues execution of kernel or explicit memory operation.
class ExecCGCommand : public Command {
public:
  ExecCGCommand(std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr Queue)
      : Command(CommandType::RUN_CG, std::move(Queue)),
        MCommandGroup(std::move(CommandGroup)) {}

private:
  // Implementation of enqueueing of ExecCGCommand.
  cl_int enqueueImp() override;

  AllocaCommand *getAllocaForReq(Requirement *Req);

  std::unique_ptr<detail::CG> MCommandGroup;
};

} // namespace detail
} // namespace sycl
} // namespace cl
