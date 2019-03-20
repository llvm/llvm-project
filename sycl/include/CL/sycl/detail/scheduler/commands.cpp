//==----------- commands.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/exception.hpp>

#include <cassert>

namespace csd = cl::sycl::detail;

namespace cl {
namespace sycl {
namespace simple_scheduler {

template <typename Dst, typename Src>
const Dst *getParamAddress(const Src *ptr, uint64_t Offset) {
  return reinterpret_cast<const Dst *>((const char *)ptr + Offset);
}

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask>
void ExecuteKernelCommand<
    KernelType, Dimensions, RangeType, KernelArgType,
    SingleTask>::executeKernel(std::vector<cl::sycl::event> DepEvents,
                               EventImplPtr Event) {
  if (m_Queue->is_host()) {
    detail::waitEvents(DepEvents);
    Event->setContextImpl(detail::getSyclObjImpl(m_Queue->get_context()));
    runOnHost();
    return;
  }
  context Context = m_Queue->get_context();
  if (!m_ClKernel) {
    m_ClKernel = detail::ProgramManager::getInstance().getOrCreateKernel(
        Context, m_KernelName.c_str());
  }

  if (m_KernelArgs != nullptr) {
    for (unsigned I = 0; I < m_KernelArgsNum; ++I) {
      switch (m_KernelArgs[I].kind) {
      case csd::kernel_param_kind_t::kind_std_layout: {
        const void *Ptr =
            getParamAddress<void>(&m_HostKernel, m_KernelArgs[I].offset);
        CHECK_OCL_CODE(
            clSetKernelArg(m_ClKernel, I, m_KernelArgs[I].info, Ptr));
        break;
      }
      case csd::kernel_param_kind_t::kind_accessor: {
        switch (static_cast<cl::sycl::access::target>(m_KernelArgs[I].info)) {
        case cl::sycl::access::target::global_buffer:
        case cl::sycl::access::target::constant_buffer: {
          auto *Ptr = *(getParamAddress<
                        cl::sycl::detail::buffer_impl<std::allocator<char>> *>(
              &m_HostKernel, m_KernelArgs[I].offset));
          cl_mem CLBuf = Ptr->getOpenCLMem();
          CHECK_OCL_CODE(clSetKernelArg(m_ClKernel, I, sizeof(cl_mem), &CLBuf));
          break;
        }
        case cl::sycl::access::target::local: {
          auto *Ptr =
              getParamAddress<size_t>(&m_HostKernel, m_KernelArgs[I].offset);
          CHECK_OCL_CODE(clSetKernelArg(m_ClKernel, I, *Ptr, nullptr));
          break;
        }
        // TODO handle these cases
        case cl::sycl::access::target::image:
        case cl::sycl::access::target::host_buffer:
        case cl::sycl::access::target::host_image:
        case cl::sycl::access::target::image_array:
          assert(0);
        }
        break;
      }
      // TODO implement
      case csd::kernel_param_kind_t::kind_sampler:
        assert(0);
      }
    }
  }
  for (const auto &Arg : m_InteropArgs) {
    if (Arg.m_Ptr.get() != nullptr) {
      CHECK_OCL_CODE(clSetKernelArg(m_ClKernel, Arg.m_ArgIndex, Arg.m_Size,
                                    Arg.m_Ptr.get()));
    } else {
      cl_mem CLBuf = Arg.m_BufReq->getCLMemObject();
      CHECK_OCL_CODE(
          clSetKernelArg(m_ClKernel, Arg.m_ArgIndex, sizeof(cl_mem), &CLBuf));
    }
  }

  std::vector<cl_event> CLEvents = detail::getOrWaitEvents(
      std::move(DepEvents), detail::getSyclObjImpl(Context));
  cl_event &CLEvent = Event->getHandleRef();
  CLEvent = runEnqueueNDRangeKernel(m_Queue->getHandleRef(), m_ClKernel,
                                    std::move(CLEvents));
  Event->setContextImpl(detail::getSyclObjImpl(m_Queue->get_context()));
}

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask>
template <typename R>
typename std::enable_if<std::is_same<R, range<Dimensions>>::value,
                        cl_event>::type
ExecuteKernelCommand<
    KernelType, Dimensions, RangeType, KernelArgType,
    SingleTask>::runEnqueueNDRangeKernel(cl_command_queue &EnvQueue,
                                         cl_kernel &Kernel,
                                         std::vector<cl_event> CLEvents) {
  size_t GlobalWorkSize[Dimensions];
  size_t GlobalWorkOffset[Dimensions];
  for (int I = 0; I < Dimensions; I++) {
    GlobalWorkSize[I] = m_WorkItemsRange[I];
    GlobalWorkOffset[I] = m_WorkItemsOffset[I];
  }
  cl_event CLEvent;
  cl_int error = clEnqueueNDRangeKernel(
      EnvQueue, Kernel, Dimensions, GlobalWorkOffset, GlobalWorkSize, nullptr,
      CLEvents.size(), CLEvents.data(), &CLEvent);
  CHECK_OCL_CODE(error);
  return CLEvent;
}

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask>
template <typename R>
typename std::enable_if<std::is_same<R, nd_range<Dimensions>>::value,
                        cl_event>::type
ExecuteKernelCommand<
    KernelType, Dimensions, RangeType, KernelArgType,
    SingleTask>::runEnqueueNDRangeKernel(cl_command_queue &EnvQueue,
                                         cl_kernel &Kernel,
                                         std::vector<cl_event> CLEvents) {
  size_t GlobalWorkSize[Dimensions];
  size_t LocalWorkSize[Dimensions];
  size_t GlobalWorkOffset[Dimensions];
  for (int I = 0; I < Dimensions; I++) {
    GlobalWorkSize[I] = m_WorkItemsRange.get_global_range()[I];
    LocalWorkSize[I] = m_WorkItemsRange.get_local_range()[I];
    GlobalWorkOffset[I] = m_WorkItemsRange.get_offset()[I];
  }
  cl_event CLEvent;
  cl_int Err = clEnqueueNDRangeKernel(
      EnvQueue, Kernel, Dimensions, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, CLEvents.size(), CLEvents.data(), &CLEvent);
  CHECK_OCL_CODE(Err);
  return CLEvent;
}

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
