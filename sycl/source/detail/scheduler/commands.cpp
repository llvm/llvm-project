//==----------- commands.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl/detail/scheduler/commands.h>

#include <atomic>
#include <cassert>

namespace cl {
namespace sycl {
namespace simple_scheduler {

Command::Command(CommandType Type, QueueImplPtr Queue)
    : m_Type(Type), m_Enqueued(false), m_Queue(std::move(Queue)) {
  static std::atomic<size_t> CommandGlobalID(0);
  m_ID = CommandGlobalID++;
}

void MemMoveCommand::enqueueImp(std::vector<cl::sycl::event> DepEvents,
                                EventImplPtr Event) {
  assert(nullptr != m_Buf && "Buf is nullptr");
  m_Buf->moveMemoryTo(m_Queue, std::move(DepEvents), std::move(Event));
}

void AllocaCommand::enqueueImp(std::vector<cl::sycl::event> DepEvents,
                               EventImplPtr Event) {
  assert(nullptr != m_Buf && "Buf is nullptr");
  m_Buf->allocate(m_Queue, std::move(DepEvents), std::move(Event));
}

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
