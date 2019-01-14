//==----------- scheduler.cpp ----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
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

template <typename T, int Dimensions, typename AllocatorT>
static BufferReqPtr
getReqForBuffer(const std::set<BufferReqPtr, classcomp> &BufReqs,
                const detail::buffer_impl<T, Dimensions, AllocatorT> &Buf) {
  for (const auto &Req : BufReqs) {
    if (Req->getUniqID() == &Buf) {
      return Req;
    }
  }
  return nullptr;
}

// Adds a buffer requirement for this node.
template <access::mode Mode, access::target Target, typename T, int Dimensions,
          typename AllocatorT>
void Node::addBufRequirement(
    detail::buffer_impl<T, Dimensions, AllocatorT> &Buf) {
  BufferReqPtr Req = getReqForBuffer(m_Bufs, Buf);

  // Check if there is requirement for the same buffer already.
  if (nullptr != Req) {
    Req->addAccessMode(Mode);
  } else {
    BufferReqPtr BufStor = std::make_shared<
        BufferStorage<T, Dimensions, AllocatorT, Mode, Target>>(Buf);
    m_Bufs.insert(BufStor);
  }
}

// Adds an accessor requirement for this node.
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
void Node::addAccRequirement(
    accessor<dataT, dimensions, accessMode, accessTarget, isPlaceholder> &&Acc,
    int argIndex) {
  detail::buffer_impl<dataT, dimensions> *buf =
      Acc.template accessor_base<dataT, dimensions, accessMode, accessTarget,
                                 isPlaceholder>::__impl()
          ->m_Buf;
  addBufRequirement<accessMode, accessTarget, dataT, dimensions>(*buf);
  addInteropArg(nullptr, buf->get_size(), argIndex,
                getReqForBuffer(m_Bufs, *buf));
}

// Adds a kernel to this node, maps to single task.
template <typename KernelType>
void Node::addKernel(const std::string &KernelName, const int KernelArgsNum,
                     const detail::kernel_param_desc_t *KernelArgs,
                     KernelType KernelFunc, cl_kernel ClKernel) {
  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel =
      std::make_shared<ExecuteKernelCommand<KernelType,
                                            /*Dimensions=*/1, range<1>, id<1>,
                                            /*SingleTask=*/true>>(
          KernelFunc, KernelName, KernelArgsNum, KernelArgs, range<1>(1),
          m_Queue, ClKernel);
}

// Adds kernel to this node, maps on range parallel for.
template <typename KernelType, int Dimensions, typename KernelArgType>
void Node::addKernel(const std::string &KernelName, const int KernelArgsNum,
                     const detail::kernel_param_desc_t *KernelArgs,
                     KernelType KernelFunc, range<Dimensions> NumWorkItems,
                     cl_kernel ClKernel) {
  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel =
      std::make_shared<ExecuteKernelCommand<KernelType, Dimensions,
                                            range<Dimensions>, KernelArgType>>(
          KernelFunc, KernelName, KernelArgsNum, KernelArgs, NumWorkItems,
          m_Queue, ClKernel);
}

// Adds kernel to this node, maps to range parallel for with offset.
template <typename KernelType, int Dimensions, typename KernelArgType>
void Node::addKernel(const std::string &KernelName, const int KernelArgsNum,
                     const detail::kernel_param_desc_t *KernelArgs,
                     KernelType KernelFunc, range<Dimensions> NumWorkItems,
                     id<Dimensions> WorkItemOffset, cl_kernel ClKernel) {
  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel =
      std::make_shared<ExecuteKernelCommand<KernelType, Dimensions,
                                            range<Dimensions>, KernelArgType>>(
          KernelFunc, KernelName, KernelArgsNum, KernelArgs, NumWorkItems,
          m_Queue, ClKernel, WorkItemOffset);
}
// Adds kernel to this node, maps on nd_range parallel for.
template <typename KernelType, int Dimensions>
void Node::addKernel(const std::string &KernelName, const int KernelArgsNum,
                     const detail::kernel_param_desc_t *KernelArgs,
                     KernelType KernelFunc, nd_range<Dimensions> ExecutionRange,
                     cl_kernel ClKernel) {
  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel = std::make_shared<ExecuteKernelCommand<
      KernelType, Dimensions, nd_range<Dimensions>, nd_item<Dimensions>>>(
      KernelFunc, KernelName, KernelArgsNum, KernelArgs, ExecutionRange,
      m_Queue, ClKernel);
}

// Adds explicit memory operation to this node, maps on handler fill method
template <typename T, int Dimensions, access::mode mode, access::target tgt,
          access::placeholder isPlaceholder>
void Node::addExplicitMemOp(
    accessor<T, Dimensions, mode, tgt, isPlaceholder> &Dest, T Src) {
  auto *DestBase = Dest.template accessor_base<T, Dimensions, mode, tgt,
                                               isPlaceholder>::__impl();
  assert(DestBase != nullptr &&
         "Accessor should have an initialized accessor_base");
  detail::buffer_impl<T, Dimensions> *Buf = DestBase->m_Buf;

  range<Dimensions> Range = DestBase->Range;
  id<Dimensions> Offset = DestBase->Offset;

  BufferReqPtr Req = getReqForBuffer(m_Bufs, *Buf);
  assert(Buf != nullptr && "Accessor should have an initialized buffer_impl");
  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel = std::make_shared<FillCommand<T, Dimensions>>(Req, Src, m_Queue,
                                                          Range, Offset);
}

// Adds explicit memory operation to this node, maps on handler copy method
template <typename T_src, int dim_src, access::mode mode_src,
          access::target tgt_src, typename T_dest, int dim_dest,
          access::mode mode_dest, access::target tgt_dest,
          access::placeholder isPlaceholder_src,
          access::placeholder isPlaceholder_dest>
void Node::addExplicitMemOp(
    accessor<T_src, dim_src, mode_src, tgt_src, isPlaceholder_src> Src,
    accessor<T_dest, dim_dest, mode_dest, tgt_dest, isPlaceholder_dest> Dest) {
  auto *SrcBase = Src.template accessor_base<T_src, dim_src, mode_src, tgt_src,
                                             isPlaceholder_src>::__impl();
  assert(SrcBase != nullptr &&
         "Accessor should have an initialized accessor_base");
  auto *DestBase =
      Dest.template accessor_base<T_dest, dim_dest, mode_dest, tgt_dest,
                                  isPlaceholder_dest>::__impl();
  assert(DestBase != nullptr &&
         "Accessor should have an initialized accessor_base");

  detail::buffer_impl<T_src, dim_src> *SrcBuf = SrcBase->m_Buf;
  assert(SrcBuf != nullptr &&
         "Accessor should have an initialized buffer_impl");
  detail::buffer_impl<T_dest, dim_dest> *DestBuf = DestBase->m_Buf;
  assert(DestBuf != nullptr &&
         "Accessor should have an initialized buffer_impl");

  range<dim_src> SrcRange = SrcBase->Range;
  id<dim_src> SrcOffset = SrcBase->Offset;
  id<dim_dest> DestOffset = DestBase->Offset;

  range<dim_src> BuffSrcRange = SrcBase->m_Buf->get_range();

  BufferReqPtr SrcReq = getReqForBuffer(m_Bufs, *SrcBuf);
  BufferReqPtr DestReq = getReqForBuffer(m_Bufs, *DestBuf);

  assert(!m_Kernel && "This node already contains an execution command");
  m_Kernel = std::make_shared<CopyCommand<dim_src, dim_dest>>(
      SrcReq, DestReq, m_Queue, SrcRange, SrcOffset, DestOffset, sizeof(T_src),
      SrcBase->get_count(), BuffSrcRange);
}

// Updates host data of the specified accessor
template <typename T, int Dimensions, access::mode mode, access::target tgt,
          access::placeholder isPlaceholder>
void Scheduler::updateHost(
    accessor<T, Dimensions, mode, tgt, isPlaceholder> &Acc,
    cl::sycl::event &Event) {
  auto *AccBase = Acc.template accessor_base<T, Dimensions, mode, tgt,
                                             isPlaceholder>::__impl();
  assert(AccBase != nullptr &&
         "Accessor should have an initialized accessor_base");
  detail::buffer_impl<T, Dimensions> *Buf = AccBase->m_Buf;

  updateHost<mode, tgt>(*Buf, Event);
}

template <access::mode Mode, access::target Target, typename T, int Dimensions,
          typename AllocatorT>
void Scheduler::copyBack(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf) {
  cl::sycl::event Event;
  updateHost<Mode, Target>(Buf, Event);
  detail::getSyclObjImpl(Event)->waitInternal();
}

// Updates host data of the specified buffer_impl
template <access::mode Mode, access::target Target, typename T, int Dimensions,
          typename AllocatorT>
void Scheduler::updateHost(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf,
                           cl::sycl::event &Event) {
  CommandPtr UpdateHostCmd;
  BufferReqPtr BufStor =
      std::make_shared<BufferStorage<T, Dimensions, AllocatorT, Mode, Target>>(
          Buf);

  if (0 == m_BuffersEvolution.count(BufStor)) {
    return;
  }

  // TODO: Find a better way to say that we need copy to HOST, just nullptr?
  cl::sycl::device HostDevice;
  UpdateHostCmd = std::make_shared<MemMoveCommand>(
      BufStor, m_BuffersEvolution[BufStor].back()->getQueue(),
      detail::getSyclObjImpl(cl::sycl::queue(HostDevice)),
      cl::sycl::access::mode::read_write);

  // Add dependency if there was operations with the buffer already.
  UpdateHostCmd->addDep(m_BuffersEvolution[BufStor].back(), BufStor);

  m_BuffersEvolution[BufStor].push_back(UpdateHostCmd);
  Event = EnqueueCommand(std::move(UpdateHostCmd));
}

template <typename T, int Dimensions, typename AllocatorT>
void Scheduler::removeBuffer(
    detail::buffer_impl<T, Dimensions, AllocatorT> &Buf) {
  BufferReqPtr BufStor = std::make_shared<
      BufferStorage<T, Dimensions, AllocatorT, access::mode::read_write,
                    access::target::host_buffer>>(Buf);

  if (0 == m_BuffersEvolution.count(BufStor)) {
    return;
  }

  for (auto Cmd : m_BuffersEvolution[BufStor]) {
    Cmd->removeAllDeps();
  }

  m_BuffersEvolution.erase(BufStor);
}

static bool cmdsHaveEqualCxtAndDev(const CommandPtr &LHS,
                                   const CommandPtr &RHS) {
  return LHS->getQueue()->get_device() == RHS->getQueue()->get_device() &&
         LHS->getQueue()->get_context() == LHS->getQueue()->get_context();
}

// Adds new node to graph, creating an Alloca and MemMove commands if
// needed.
inline cl::sycl::event Scheduler::addNode(Node NewNode) {
  // Process global buffers.
  CommandPtr Cmd = NewNode.getKernel();
  for (auto Buf : NewNode.getRequirements()) {
    // If it's the first command for buffer - insert alloca command.
    if (m_BuffersEvolution[Buf].empty()) {
      CommandPtr AllocaCmd =
          std::make_shared<AllocaCommand>(Buf, std::move(NewNode.getQueue()),
                                          cl::sycl::access::mode::read_write);
      m_BuffersEvolution[Buf].push_back(AllocaCmd);
    }
    // If targets of previous and new command differ - insert memmove command.
    if (!cmdsHaveEqualCxtAndDev(m_BuffersEvolution[Buf].back(), Cmd)) {
      CommandPtr MemMoveCmd = std::make_shared<MemMoveCommand>(
          Buf, std::move(m_BuffersEvolution[Buf].back()->getQueue()),
          std::move(NewNode.getQueue()), cl::sycl::access::mode::read_write);
      MemMoveCmd->addDep(m_BuffersEvolution[Buf].back(), Buf);
      m_BuffersEvolution[Buf].push_back(MemMoveCmd);
    }
    // Finally insert command to the buffer evolution vector.
    Cmd->addDep(m_BuffersEvolution[Buf].back(), Buf);
    m_BuffersEvolution[Buf].push_back(Cmd);
  }
  // Process arguments set via interoperability interface
  for (auto Arg : NewNode.getInteropArgs()) {
    Cmd->addInteropArg(Arg);
  }
  // If the kernel has no requirements, store the event
  if (NewNode.getRequirements().empty()) {
    m_EventsWithoutRequirements.push_back(
        detail::getSyclObjImpl(Cmd->getEvent()));
  }
  return EnqueueCommand(Cmd);
}
//}
} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
