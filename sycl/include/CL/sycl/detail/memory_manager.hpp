//==-------------- memory_manager.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/sycl_mem_obj.hpp>
#include <CL/sycl/range.hpp>

#include <memory>
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

// The class contains methods that work with memory. All operations with
// device memory should go through MemoryManager.

class MemoryManager {
public:
  // The following method releases memory allocation of memory object.
  // Depending on the context it releases memory on host or on device.
  static void release(ContextImplPtr TargetContext, SYCLMemObjT *MemObj,
                      void *MemAllocation, std::vector<cl_event> DepEvents,
                      cl_event &OutEvent);

  // The following method allocates memory allocation of memory object.
  // Depending on the context it allocates memory on host or on device.
  static void *allocate(ContextImplPtr TargetContext, SYCLMemObjT *MemObj,
                        bool InitFromUserData, std::vector<cl_event> DepEvents,
                        cl_event &OutEvent);

  // Allocates buffer in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemBuffer(ContextImplPtr TargetContext,
                                 SYCLMemObjT *MemObj, void *UserPtr,
                                 bool HostPtrReadOnly, size_t Size,
                                 const EventImplPtr &InteropEvent,
                                 const ContextImplPtr &InteropContext,
                                 cl_event &OutEventToWait);

  // Releases buffer. TargetContext should be device one(not host).
  static void releaseMemBuf(ContextImplPtr TargetContext, SYCLMemObjT *MemObj,
                            void *MemAllocation, void *UserPtr);

  // Copies memory between: host and device, host and host,
  // device and device if memory objects bound to the one context.
  static void copy(SYCLMemObjT *SYCLMemObj, void *SrcMem, QueueImplPtr SrcQueue,
                   unsigned int DimSrc, sycl::range<3> SrcSize,
                   sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
                   unsigned int SrcElemSize, void *DstMem,
                   QueueImplPtr TgtQueue, unsigned int DimDst,
                   sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
                   sycl::id<3> DstOffset, unsigned int DstElemSize,
                   std::vector<cl_event> DepEvents, bool UseExclusiveQueue,
                   cl_event &OutEvent);

  static void fill(SYCLMemObjT *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   size_t PatternSize, const char *Pattern, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<cl_event> DepEvents, cl_event &OutEvent);

  static void *map(SYCLMemObjT *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   access::mode AccessMode, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<cl_event> DepEvents, cl_event &OutEvent);

  static void unmap(SYCLMemObjT *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                    void *MappedPtr, std::vector<cl_event> DepEvents,
                    bool UseExclusiveQueue, cl_event &OutEvent);
};
} // namespace detail
} // namespace sycl
} // namespace cl
