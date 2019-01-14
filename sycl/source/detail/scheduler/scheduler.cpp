//==----------- scheduler.cpp ----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/scheduler/commands.h>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/event.hpp>
#include <CL/sycl/nd_range.hpp>

#include <cassert>
#include <fstream>
#include <set>
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

void Node::addInteropArg(shared_ptr_class<void> Ptr, size_t Size,
                         int ArgIndex, BufferReqPtr BufReq) {
  m_InteropArgs.emplace_back(Ptr, Size, ArgIndex, BufReq);
}

// Waits for the event passed.
void Scheduler::waitForEvent(EventImplPtr Event) {
  for (auto &BufEvolution : m_BuffersEvolution) {
    for (auto &Cmd : BufEvolution.second) {
      if (detail::getSyclObjImpl(Cmd->getEvent()) == Event) {
        enqueueAndWaitForCommand(Cmd);
        return;
      }
    }
  }
  for (auto &Evnt : m_EventsWithoutRequirements) {
    if (Evnt == Event) {
      Evnt->waitInternal();
      return;
    }
  }
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

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
