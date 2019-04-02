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

template <int AccessDimensions, typename KernelType>
uint passGlobalAccessorAsArg(uint I, int LambdaOffset, cl_kernel ClKernel,
                             const KernelType &HostKernel) {
  using AccType = accessor<char, AccessDimensions, access::mode::read,
                           access::target::global_buffer,
                           access::placeholder::false_t>;
  const AccType *Acc = getParamAddress<AccType>(&HostKernel, LambdaOffset);
  cl_mem CLBuf = Acc->getBufImpl()->getOpenCLMem();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I, sizeof(cl_mem), &CLBuf));

  range<AccessDimensions> AccessRange = Acc->getAccessRange();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 1,
                                sizeof(range<AccessDimensions>),
                                &AccessRange));
  range<AccessDimensions> MemRange = Acc->getMemRange();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 2,
                                sizeof(range<AccessDimensions>), &MemRange));
  id<AccessDimensions> Offset = Acc->getOffset();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 3,
                                sizeof(id<AccessDimensions>), &Offset));
  return 4;
}

template <int AccessDimensions, typename KernelType>
uint passLocalAccessorAsArg(uint I, int LambdaOffset, cl_kernel ClKernel,
                            const KernelType &HostKernel) {
  using AccType = accessor<char, AccessDimensions, access::mode::read,
                           access::target::local,
                           access::placeholder::false_t>;
  const AccType *Acc = getParamAddress<AccType>(&HostKernel, LambdaOffset);
  size_t ByteSize = Acc->getByteSize();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I, ByteSize, nullptr));

  range<AccessDimensions> AccessRange = Acc->getAccessRange();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 1,
                                sizeof(range<AccessDimensions>),
                                &AccessRange));
  range<AccessDimensions> MemRange = Acc->getMemRange();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 2,
                                sizeof(range<AccessDimensions>), &MemRange));
  id<AccessDimensions> Offset = Acc->getOffset();
  CHECK_OCL_CODE(clSetKernelArg(ClKernel, I + 3,
                                sizeof(id<AccessDimensions>), &Offset));
  return 4;
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
        m_OSModule, Context, m_KernelName);
  }

  if (m_KernelArgs != nullptr) {
    unsigned ArgumentID = 0;
    for (unsigned I = 0; I < m_KernelArgsNum; ++I) {
      switch (m_KernelArgs[I].kind) {
      case csd::kernel_param_kind_t::kind_std_layout: {
        const void *Ptr =
            getParamAddress<void>(&m_HostKernel, m_KernelArgs[I].offset);
        CHECK_OCL_CODE(
            clSetKernelArg(m_ClKernel, ArgumentID, m_KernelArgs[I].info, Ptr));
        ArgumentID++;
        break;
      }
      case csd::kernel_param_kind_t::kind_accessor: {
        int AccDims = m_KernelArgs[I].info >> 11;
        int AccTarget = m_KernelArgs[I].info & 0x7ff;
        switch (static_cast<cl::sycl::access::target>(AccTarget)) {
        case access::target::global_buffer:
        case access::target::constant_buffer: {
          switch (AccDims) {
          case 1:
            ArgumentID += passGlobalAccessorAsArg<1, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 2:
            ArgumentID += passGlobalAccessorAsArg<2, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 3:
            ArgumentID += passGlobalAccessorAsArg<3, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 0:
          default:
            assert(0 && "Passing accessor with dimensions=0 is unsupported");
            break;
          }
          break;
        }
        case access::target::local: {
          switch (AccDims) {
          case 1:
            ArgumentID += passLocalAccessorAsArg<1, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 2:
            ArgumentID += passLocalAccessorAsArg<2, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 3:
            ArgumentID += passLocalAccessorAsArg<3, KernelType>(
                ArgumentID, m_KernelArgs[I].offset, m_ClKernel, m_HostKernel);
            break;
          case 0:
          default:
            assert(0 && "Passing accessor with dimensions=0 is unsupported");
            break;
          }
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
