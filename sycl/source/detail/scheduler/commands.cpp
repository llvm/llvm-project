//===----------- commands.cpp - SYCL commands -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/access/access.hpp"
#include <CL/cl.h>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/commands.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/sampler.hpp>

#include <vector>

namespace cl {
namespace sycl {
namespace detail {

void EventCompletionClbk(cl_event, cl_int, void *data) {
  // TODO: Handle return values. Store errors to async handler.
  clSetUserEventStatus((cl_event)data, CL_COMPLETE);
}

// Method prepares cl_event's from list sycl::event's
std::vector<cl_event> Command::prepareEvents(ContextImplPtr Context) {
  std::vector<cl_event> Result;
  std::vector<EventImplPtr> GlueEvents;
  for (EventImplPtr &Event : MDepsEvents) {
    // Async work is not supported for host device.
    if (Event->getContextImpl()->is_host()) {
      Event->waitInternal();
      continue;
    }
    // The event handle can be null in case of, for example, alloca command,
    // which is currently synchrounious, so don't generate OpenCL event.
    if (Event->getHandleRef() == nullptr) {
      continue;
    }
    ContextImplPtr EventContext = Event->getContextImpl();

    // If contexts don't match - connect them using user event
    if (EventContext != Context && !Context->is_host()) {
      cl_int Error = CL_SUCCESS;

      EventImplPtr GlueEvent(new detail::event_impl());
      GlueEvent->setContextImpl(Context);

      cl_event &GlueEventHandle = GlueEvent->getHandleRef();
      GlueEventHandle = clCreateUserEvent(Context->getHandleRef(), &Error);
      CHECK_OCL_CODE(Error);

      Error = clSetEventCallback(Event->getHandleRef(), CL_COMPLETE,
                                 EventCompletionClbk, /*data=*/GlueEventHandle);
      CHECK_OCL_CODE(Error);
      GlueEvents.push_back(std::move(GlueEvent));
      Result.push_back(GlueEventHandle);
      continue;
    }
    Result.push_back(Event->getHandleRef());
  }
  MDepsEvents.insert(MDepsEvents.end(), GlueEvents.begin(), GlueEvents.end());
  return Result;
}

Command::Command(CommandType Type, QueueImplPtr Queue, bool UseExclusiveQueue)
    : MQueue(std::move(Queue)), MUseExclusiveQueue(UseExclusiveQueue),
      MType(Type), MEnqueued(false) {
  MEvent.reset(new detail::event_impl());
  MEvent->setCommand(this);
  MEvent->setContextImpl(detail::getSyclObjImpl(MQueue->get_context()));
}

cl_int Command::enqueue() {
  bool Expected = false;
  if (MEnqueued.compare_exchange_strong(Expected, true))
    return enqueueImp();
  return CL_SUCCESS;
}

cl_int AllocaCommand::enqueueImp() {
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  cl_event &Event = MEvent->getHandleRef();
  MMemAllocation = MemoryManager::allocate(
      detail::getSyclObjImpl(MQueue->get_context()), getSYCLMemObj(),
      MInitFromUserData, std::move(RawEvents), Event);
  return CL_SUCCESS;
}

cl_int ReleaseCommand::enqueueImp() {
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  cl_event &Event = MEvent->getHandleRef();
  MemoryManager::release(detail::getSyclObjImpl(MQueue->get_context()),
                         MAllocaCmd->getSYCLMemObj(),
                         MAllocaCmd->getMemAllocation(), std::move(RawEvents),
                         Event);
  return CL_SUCCESS;
}

MapMemObject::MapMemObject(Requirement SrcReq, AllocaCommand *SrcAlloca,
                           Requirement *DstAcc, QueueImplPtr Queue)
    : Command(CommandType::MAP_MEM_OBJ, std::move(Queue)),
      MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca), MDstAcc(DstAcc),
      MDstReq(*DstAcc) {}

cl_int MapMemObject::enqueueImp() {
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));
  assert(MDstReq.MDims == 1);

  cl_event &Event = MEvent->getHandleRef();
  void *MappedPtr = MemoryManager::map(
      MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MQueue,
      MDstReq.MAccessMode, MDstReq.MDims, MDstReq.MMemoryRange,
      MDstReq.MAccessRange, MDstReq.MOffset, MDstReq.MElemSize,
      std::move(RawEvents), Event);
  MDstAcc->MData = MappedPtr;
  return CL_SUCCESS;
}

UnMapMemObject::UnMapMemObject(Requirement SrcReq, AllocaCommand *SrcAlloca,
                               Requirement *DstAcc, QueueImplPtr Queue,
                               bool UseExclusiveQueue)
    : Command(CommandType::UNMAP_MEM_OBJ, std::move(Queue), UseExclusiveQueue),
      MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca), MDstAcc(DstAcc) {}

cl_int UnMapMemObject::enqueueImp() {
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  cl_event &Event = MEvent->getHandleRef();
  MemoryManager::unmap(MSrcAlloca->getSYCLMemObj(),
                       MSrcAlloca->getMemAllocation(), MQueue, MDstAcc->MData,
                       std::move(RawEvents), MUseExclusiveQueue, Event);
  return CL_SUCCESS;
}

MemCpyCommand::MemCpyCommand(Requirement SrcReq, AllocaCommand *SrcAlloca,
                             Requirement DstReq, AllocaCommand *DstAlloca,
                             QueueImplPtr SrcQueue, QueueImplPtr DstQueue,
                             bool UseExclusiveQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue), UseExclusiveQueue),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca),
      MDstReq(std::move(DstReq)), MDstAlloca(DstAlloca) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(detail::getSyclObjImpl(MSrcQueue->get_context()));
}

cl_int MemCpyCommand::enqueueImp() {
  std::vector<cl_event> RawEvents;
  QueueImplPtr Queue = MQueue->is_host() ? MSrcQueue : MQueue;
  RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(Queue->get_context()));

  cl_event &Event = MEvent->getHandleRef();

  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write ||
      MSrcAlloca->getMemAllocation() == MDstAlloca->getMemAllocation()) {

    if (!RawEvents.empty()) {
      if (Queue->is_host()) {
        CHECK_OCL_CODE(clWaitForEvents(RawEvents.size(), &RawEvents[0]));
      } else {
        CHECK_OCL_CODE(clEnqueueMarkerWithWaitList(
            Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event));
      }
    }
  } else {
    MemoryManager::copy(
        MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MSrcQueue,
        MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
        MSrcReq.MOffset, MSrcReq.MElemSize, MDstAlloca->getMemAllocation(),
        MQueue, MDstReq.MDims, MDstReq.MMemoryRange, MDstReq.MAccessRange,
        MDstReq.MOffset, MDstReq.MElemSize, std::move(RawEvents),
        MUseExclusiveQueue, Event);
  }

  if (MAccToUpdate)
    MAccToUpdate->MData = MDstAlloca->getMemAllocation();
  return CL_SUCCESS;
}

AllocaCommand *ExecCGCommand::getAllocaForReq(Requirement *Req) {
  for (const DepDesc &Dep : MDeps) {
    if (Dep.MReq == Req)
      return Dep.MAllocaCmd;
  }
  throw runtime_error("Alloca for command not found");
}

MemCpyCommandHost::MemCpyCommandHost(Requirement SrcReq,
                                     AllocaCommand *SrcAlloca,
                                     Requirement *DstAcc, QueueImplPtr SrcQueue,
                                     QueueImplPtr DstQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue)),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca),
      MDstReq(*DstAcc), MDstAcc(DstAcc) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(detail::getSyclObjImpl(MSrcQueue->get_context()));
}

cl_int MemCpyCommandHost::enqueueImp() {
  QueueImplPtr Queue = MQueue->is_host() ? MSrcQueue : MQueue;
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(Queue->get_context()));

  cl_event &Event = MEvent->getHandleRef();
  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write) {

    if (!RawEvents.empty()) {
      if (Queue->is_host()) {
        CHECK_OCL_CODE(clWaitForEvents(RawEvents.size(), &RawEvents[0]));
      } else {
        CHECK_OCL_CODE(clEnqueueMarkerWithWaitList(
            Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event));
      }
    }
    return CL_SUCCESS;
  }

  MemoryManager::copy(
      MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MSrcQueue,
      MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, MDstAcc->MData, MQueue, MDstReq.MDims,
      MDstReq.MMemoryRange, MDstReq.MAccessRange, MDstReq.MOffset,
      MDstReq.MElemSize, std::move(RawEvents), MUseExclusiveQueue, Event);
  return CL_SUCCESS;
}

cl_int ExecCGCommand::enqueueImp() {
  std::vector<cl_event> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  cl_event &Event = MEvent->getHandleRef();

  switch (MCommandGroup->getType()) {

  case CG::CGTYPE::UPDATE_HOST: {
    assert(!"Update host should be handled by the Scheduler.");
    throw runtime_error("Update host should be handled by the Scheduler.");
  }
  case CG::CGTYPE::COPY_ACC_TO_PTR: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommand *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, Copy->getDst(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange, /*DstOffset=*/{0, 0, 0},
        Req->MElemSize, std::move(RawEvents), MUseExclusiveQueue, Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::COPY_PTR_TO_ACC: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommand *AllocaCmd = getAllocaForReq(Req);

    Scheduler::getInstance().getDefaultHostQueue();

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), Copy->getSrc(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange, /*SrcOffset*/ {0, 0, 0},
        Req->MElemSize, AllocaCmd->getMemAllocation(), MQueue, Req->MDims,
        Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
        std::move(RawEvents), MUseExclusiveQueue, Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::COPY_ACC_TO_ACC: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommand *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommand *AllocaCmdDst = getAllocaForReq(ReqDst);

    MemoryManager::copy(
        AllocaCmdSrc->getSYCLMemObj(), AllocaCmdSrc->getMemAllocation(), MQueue,
        ReqSrc->MDims, ReqSrc->MMemoryRange, ReqSrc->MAccessRange,
        ReqSrc->MOffset, ReqSrc->MElemSize, AllocaCmdDst->getMemAllocation(),
        MQueue, ReqDst->MDims, ReqDst->MMemoryRange, ReqDst->MAccessRange,
        ReqDst->MOffset, ReqDst->MElemSize, std::move(RawEvents),
        MUseExclusiveQueue, Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::FILL: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommand *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::fill(AllocaCmd->getSYCLMemObj(),
                        AllocaCmd->getMemAllocation(), MQueue,
                        Fill->MPattern.size(), Fill->MPattern.data(),
                        Req->MDims, Req->MMemoryRange, Req->MAccessRange,
                        Req->MOffset, Req->MElemSize, std::move(RawEvents),
                        Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::KERNEL: {
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    NDRDescT &NDRDesc = ExecKernel->MNDRDesc;

    if (MQueue->is_host()) {
      for (ArgDesc &Arg : ExecKernel->MArgs)
        if (kernel_param_kind_t::kind_accessor == Arg.MType) {
          Requirement *Req = (Requirement *)(Arg.MPtr);
          AllocaCommand *AllocaCmd = getAllocaForReq(Req);
          Req->MData = AllocaCmd->getMemAllocation();
        }
      if (!RawEvents.empty())
        CHECK_OCL_CODE(clWaitForEvents(RawEvents.size(), &RawEvents[0]));
      ExecKernel->MHostKernel->call(NDRDesc);
      return CL_SUCCESS;
    }

    // Run OpenCL kernel
    sycl::context Context = MQueue->get_context();
    cl_kernel Kernel = nullptr;

    if (nullptr != ExecKernel->MSyclKernel) {
      assert(ExecKernel->MSyclKernel->get_context() == Context);
      Kernel = ExecKernel->MSyclKernel->getHandleRef();
    } else
      Kernel = detail::ProgramManager::getInstance().getOrCreateKernel(
          ExecKernel->MOSModuleHandle, Context, ExecKernel->MKernelName);

    for (ArgDesc &Arg : ExecKernel->MArgs) {
      switch (Arg.MType) {
      case kernel_param_kind_t::kind_accessor: {
        Requirement *Req = (Requirement *)(Arg.MPtr);
        AllocaCommand *AllocaCmd = getAllocaForReq(Req);
        cl_mem MemArg = (cl_mem)AllocaCmd->getMemAllocation();

        CHECK_OCL_CODE(
            clSetKernelArg(Kernel, Arg.MIndex, sizeof(cl_mem), &MemArg));
        break;
      }
      case kernel_param_kind_t::kind_std_layout: {
        CHECK_OCL_CODE(clSetKernelArg(Kernel, Arg.MIndex, Arg.MSize, Arg.MPtr));
        break;
      }
      case kernel_param_kind_t::kind_sampler: {
        sampler *SamplerPtr = (sampler *)Arg.MPtr;
        cl_sampler CLSampler =
            detail::getSyclObjImpl(*SamplerPtr)->getOrCreateSampler(Context);
        CHECK_OCL_CODE(
            clSetKernelArg(Kernel, Arg.MIndex, sizeof(cl_sampler), &CLSampler));
        break;
      }
      default:
        assert(!"Unhandled");
      }
    }

    cl_int Error = CL_SUCCESS;
    Error = clEnqueueNDRangeKernel(
        MQueue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
        &NDRDesc.GlobalSize[0],
        NDRDesc.LocalSize[0] ? &NDRDesc.LocalSize[0] : nullptr,
        RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0], &Event);
    CHECK_OCL_CODE(Error);
    return CL_SUCCESS;
  }
  }

  assert(!"CG type not implemented");
  throw runtime_error("CG type not implemented.");
}

} // namespace detail
} // namespace sycl
} // namespace cl
