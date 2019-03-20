//==----------- scheduler.cpp ----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/scheduler/commands.h>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/event.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/queue.hpp>

#include <cassert>
#include <fstream>
#include <set>
#include <unordered_set>
#include <vector>

namespace cl {
namespace sycl {
namespace simple_scheduler {

Scheduler::Scheduler() {
  if (std::getenv("SS_DUMP_TEXT")) {
    m_DumpOptions[DumpOptions::Text] = 1;
  }
  if (std::getenv("SS_DUMP_WHOLE_GRAPH")) {
    m_DumpOptions[DumpOptions::WholeGraph] = 1;
  }
  if (std::getenv("SS_DUMP_RUN_GRAPH")) {
    m_DumpOptions[DumpOptions::RunGraph] = 1;
  }
}

void Node::addInteropArg(shared_ptr_class<void> Ptr, size_t Size, int ArgIndex,
                         BufferReqPtr BufReq) {
  m_InteropArgs.emplace_back(Ptr, Size, ArgIndex, BufReq);
}

CommandPtr Scheduler::getCmdForEvent(EventImplPtr Event) {
  // TODO: Currently, this method searches for the command in
  // m_BuffersEvolution, which seems expensive, especially
  // taking into account that this operation may be called
  // from another loop. Need to optimize this method, for example,
  // by adding a direct link from 'event' to the 'command' it
  // is associated with.
  for (auto &BufEvolution : m_BuffersEvolution) {
    for (auto &Cmd : BufEvolution.second) {
      if (detail::getSyclObjImpl(Cmd->getEvent()) == Event) {
        return Cmd;
      }
    }
  }
  return nullptr;
}

// Waits for the event passed.
void Scheduler::waitForEvent(EventImplPtr Event) {
  auto Cmd = getCmdForEvent(Event);
  if (Cmd) {
    enqueueAndWaitForCommand(Cmd);
    return;
  }

  for (auto &Evnt : m_EventsWithoutRequirements) {
    if (Evnt == Event) {
      Evnt->waitInternal();
      return;
    }
  }
}

// Calls async handler for the given command Cmd and those other
// commands that Cmd depends on.
void Scheduler::throwForCmdRecursive(std::shared_ptr<Command> Cmd) {
  if (Cmd == nullptr) {
    return;
  }

  auto QImpl = Cmd->getQueue();
  QImpl->throw_asynchronous();

  std::vector<std::pair<std::shared_ptr<Command>, BufferReqPtr>> Deps =
      Cmd->getDependencies();
  for (auto D : Deps) {
    throwForCmdRecursive(D.first);
  }
}

// Calls async handler for the given event Event and those other
// events that Event depends on.
void Scheduler::throwForEventRecursive(EventImplPtr Event) {
  auto Cmd = getCmdForEvent(Event);
  if (Cmd) {
    throwForCmdRecursive(Cmd);
  }
}

void Scheduler::getDepEventsHelper(
    std::unordered_set<cl::sycl::event> &EventsSet,
    EventImplPtr Event) {
  auto Cmd = getCmdForEvent(Event);
  if (Cmd == nullptr) {
    return;
  }

  std::vector<std::pair<std::shared_ptr<Command>, BufferReqPtr>> Deps =
      Cmd->getDependencies();
  for (auto D : Deps) {
    auto DepEvent = D.first->getEvent();
    EventsSet.insert(DepEvent);

    auto DepEventImpl = cl::sycl::detail::getSyclObjImpl(DepEvent);
    getDepEventsHelper(EventsSet, DepEventImpl);
  }
}

vector_class<event> Scheduler::getDepEventsRecursive(EventImplPtr Event) {
  std::unordered_set<event> DepEventsSet;
  getDepEventsHelper(DepEventsSet, Event);

  vector_class<event> DepEventsVec(DepEventsSet.begin(), DepEventsSet.end());
  return DepEventsVec;
}

void Scheduler::print(std::ostream &Stream) const {
  Stream << "======================================" << std::endl;
  Stream << "Graph dump" << std::endl;
  Stream << "======================================" << std::endl;

  for (auto It : m_BuffersEvolution) {
    Stream << std::endl;
    Stream << "Evolution of buffer " << It.first->getUniqID() << std::endl;
    for (auto Elem : It.second) {
      Elem->print(Stream);
    }
  }
}

void Scheduler::printDot(std::ostream &Stream) const {
  Stream << "strict digraph {" << std::endl;
  for (auto It : m_BuffersEvolution) {
    Stream << "label=\"" << It.first << "\"" << std::endl;
    for (auto Elem : It.second) {
      Elem->printDot(Stream);
    }
  }
  Stream << "}" << std::endl;
}

void Scheduler::dumpGraphForCommand(CommandPtr Cmd) const {
  std::string FileName = "graph_run" + std::to_string(Cmd->getID()) + ".dot";
  std::fstream GraphDot(FileName, std::ios::out);
  GraphDot << "strict digraph {" << std::endl;

  printGraphForCommand(std::move(Cmd), GraphDot);

  GraphDot << "}" << std::endl;
}

// Converts the following:
//
//  =========    =========     =========
// | kernel1 |<-| kernel2 |<--| kernel3 |
// | write A |  | read A  |   | read A  |
//  =========    =========     =========
//
// to: ---------------------------
//     \/                        |
//  =========    =========     =========
// | kernel1 |<-| kernel2 |   | kernel3 |
// | write A |  | read A  |   | read A  |
//  =========    =========     =========
//
void Scheduler::parallelReadOpt() {
  for (auto BufEvolution : m_BuffersEvolution) {
    auto &Buf = BufEvolution.first;
    for (auto Node : BufEvolution.second) {
      if (!Node->isEnqueued() && Node->getType() == Command::RUN_KERNEL &&
          Node->getAccessModeForReqBuf(Buf) == cl::sycl::access::mode::read) {
        CommandPtr Dep = Node->getDepCommandForReqBuf(Buf);
        assert(nullptr != Dep);
        if (Dep->getType() == Command::RUN_KERNEL &&
            Dep->getAccessModeForReqBuf(Buf) == cl::sycl::access::mode::read) {
          Node->replaceDepCommandForReqBuf(Buf,
                                           Dep->getDepCommandForReqBuf(Buf));
        }
      }
    }
  }
}

Scheduler::~Scheduler() {
  // TODO: Find a better way to break recursive shared_ptr desctruction.
  // This is needed because in cases when there are a lot of commands and
  // they depend on each other we can run out of stack memory because
  // destruction of latest command involves destruction of all it's
  // dependencies.
  for (auto Evol : m_BuffersEvolution) {
    for (auto Cmd : Evol.second) {
      Cmd->removeAllDeps();
    }
  }
}

void Scheduler::enqueueAndWaitForCommand(CommandPtr Cmd) {
  cl::sycl::event Event = EnqueueCommand(std::move(Cmd));
  detail::getSyclObjImpl(Event)->waitInternal();
}

bool Scheduler::getDumpFlagValue(DumpOptions DumpOption) {
  return m_DumpOptions[DumpOption];
}

// Enqueues Cmd command and all its dependencies.
cl::sycl::event Scheduler::EnqueueCommand(CommandPtr Cmd) {
  if (getDumpFlagValue(DumpOptions::Text)) {
    dump();
  }
  if (getDumpFlagValue(DumpOptions::WholeGraph)) {
    dumpGraph();
  }
  if (getDumpFlagValue(DumpOptions::RunGraph)) {
    dumpGraphForCommand(Cmd);
  }

  return dispatch(std::move(Cmd));
}

cl::sycl::event Scheduler::dispatch(CommandPtr Cmd) {
  if (Cmd->isEnqueued()) {
    return Cmd->getEvent();
  }
  std::vector<cl::sycl::event> EventDeps;
  for (auto Dep : Cmd->getDependencies()) {
    EventDeps.push_back(dispatch(Dep.first));
  }
  return Cmd->enqueue(std::move(EventDeps));
}

// Recursively generates dot records for the command passed and all that the
// command depends on.
void Scheduler::printGraphForCommand(CommandPtr Cmd,
                                     std::ostream &Stream) const {
  for (const auto &Dep : Cmd->getDependencies()) {
    printGraphForCommand(Dep.first, Stream);
  }
  Cmd->printDot(Stream);
}

Scheduler &Scheduler::getInstance() {
  static Scheduler Instance;
  return Instance;
}

CommandPtr Scheduler::insertUpdateHostCmd(const BufferReqPtr &BufStor) {
  // TODO: Find a better way to say that we need copy to HOST, just nullptr?
  cl::sycl::device HostDevice;
  CommandPtr UpdateHostCmd = std::make_shared<MemMoveCommand>(
      BufStor, m_BuffersEvolution[BufStor].back()->getQueue(),
      detail::getSyclObjImpl(cl::sycl::queue(HostDevice)),
      cl::sycl::access::mode::read_write);

  // Add dependency if there was operations with the buffer already.
  UpdateHostCmd->addDep(m_BuffersEvolution[BufStor].back(), BufStor);

  m_BuffersEvolution[BufStor].push_back(UpdateHostCmd);
  return UpdateHostCmd;
}

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
