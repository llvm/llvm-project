//==-------------- scheduler.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/scheduler/commands.hpp>
#include <CL/sycl/detail/sycl_mem_obj.hpp>

#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

class queue_impl;
class event_impl;
class context_impl;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

class Scheduler {
public:
  // Registers command group, adds it to the dependency graph and returns an
  // event object that can be used for waiting later. It's called by SYCL's
  // queue.submit.
  EventImplPtr addCG(std::unique_ptr<detail::CG> CommandGroup,
                     QueueImplPtr Queue);

  EventImplPtr addCopyBack(Requirement *Req);

  // Blocking call that waits for the event passed. For the eager execution mode
  // this method invokes corresponding function of device API. In the lazy
  // execution mode the method may enqueue the command associated with the event
  // passed and its dependency before calling device API.
  void waitForEvent(EventImplPtr Event);

  // Removes buffer pointed by MemObj from the graph: ensures all commands
  // accessing the memory objects are executed and triggers deallocation of all
  // memory assigned to the memory object. It's called from the sycl::buffer and
  // sycl::image destructors.
  void removeMemoryObject(detail::SYCLMemObjT *MemObj);

  EventImplPtr addHostAccessor(Requirement *Req);

  // Returns an instance of the scheduler object.
  static Scheduler &getInstance();

  // Returns list of "immediate" dependencies for the Event given.
  std::vector<EventImplPtr> getWaitList(EventImplPtr Event);

  QueueImplPtr getDefaultHostQueue() { return DefaultHostQueue; }

private:
  Scheduler();
  ~Scheduler();

  // The graph builder provides interfaces that can change already existing
  // graph (e.g. add/remove edges/nodes).
  class GraphBuilder {
  public:
    // Registers command group, adds it to the dependency graph and returns an
    // command that represents command group execution. It's called by SYCL's
    // queue::submit.
    Command *addCG(std::unique_ptr<detail::CG> CommandGroup,
                   QueueImplPtr Queue);

    Command *addCGUpdateHost(std::unique_ptr<detail::CG> CommandGroup,
                             QueueImplPtr HostQueue);

    Command *addCopyBack(Requirement *Req);
    Command *addHostAccessor(Requirement *Req, EventImplPtr &RetEvent);

    // [Provisional] Optimizes the whole graph.
    void optimize();

    // [Provisional] Optimizes subgraph that consists of command associated with
    // Event passed and its dependencies.
    void optimize(EventImplPtr Event);

    // Removes unneeded commands from the graph.
    void cleanupCommands(bool CleanupReleaseCommands = false);

    // Reschedules command passed using Queue provided. this can lead to
    // rescheduling of all dependent commands. This can be used when user
    // provides "secondary" queue to submit method which may be used when
    // command fails to enqueue/execute in primary queue.
    void rescheduleCommand(Command *Cmd, QueueImplPtr Queue);

    // The MemObjRecord is created for each memory object used in command
    // groups. There should be only one MemObjRecord for SYCL memory object.

    struct MemObjRecord {
      // Used to distinguish one memory object from another.
      detail::SYCLMemObjT *MMemObj;

      // Contains all allocation commands for the memory object.
      std::vector<AllocaCommand *> MAllocaCommands;

      // Contains latest read only commands working with memory object.
      std::vector<Command *> MReadLeafs;

      // Contains latest write commands working with memory object.
      std::vector<Command *> MWriteLeafs;

      // The flag indicates that the content of the memory object was/will be
      // modified. Used while deciding if copy back needed.
      bool MMemModified;
    };

    MemObjRecord *getMemObjRecord(SYCLMemObjT *MemObject);
    // Returns pointer to MemObjRecord for pointer to memory object.
    // Return nullptr if there the record is not found.
    MemObjRecord *getOrInsertMemObjRecord(const QueueImplPtr &Queue,
                                          Requirement *Req);

    // Removes MemObjRecord for memory object passed.
    void removeRecordForMemObj(SYCLMemObjT *MemObject);

    // Add new command to leafs if needed.
    void AddNodeToLeafs(MemObjRecord *Record, Command *Cmd, Requirement *Req);

    // Removes commands from leafs.
    void UpdateLeafs(const std::set<Command *> &Cmds, MemObjRecord *Record,
                     Requirement *Req);

    std::vector<MemObjRecord> MMemObjRecords;

  private:
    // The method inserts memory copy operation from the context where the
    // memory current lives to the context bound to Queue.
    MemCpyCommand *insertMemCpyCmd(MemObjRecord *Record, Requirement *Req,
                                   const QueueImplPtr &Queue);

    std::set<Command *> findDepsForReq(MemObjRecord *Record, Requirement *Req,
                                       QueueImplPtr Context);

    AllocaCommand *findAllocaForReq(MemObjRecord *Record, Requirement *Req,
                                    QueueImplPtr Queue);

    void markModifiedIfWrite(GraphBuilder::MemObjRecord *Record,
                             Requirement *Req);
  };

  // The class that provides interfaces for enqueueing command and its
  // dependencies to the underlying runtime. Methods of this class must not
  // modify the graph.
  class GraphProcessor {
  public:
    // Returns a list of events that represent immediate dependencies of the
    // command associated with Event passed.
    static std::vector<EventImplPtr> getWaitList(EventImplPtr Event);

    // Wait for the command, associated with Event passed, is completed.
    static void waitForEvent(EventImplPtr Event);

    // Enqueue the command passed to the underlying device.
    // Returns pointer to command which failed to enqueue, so this command
    // with all commands that depend on it can be rescheduled.
    static Command *enqueueCommand(Command *Cmd);
  };

  void waitForRecordToFinish(GraphBuilder::MemObjRecord *Record);

  GraphBuilder MGraphBuilder;
  // Use read-write mutex in future.
  std::mutex MGraphLock;

  QueueImplPtr DefaultHostQueue;
};

} // namespace detail
} // namespace sycl
} // namespace cl
