//==-------------- memory_manager.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

static void waitForEvents(const std::vector<cl_event> &Events) {
  if (!Events.empty())
    CHECK_OCL_CODE(clWaitForEvents(Events.size(), &Events[0]));
}

void MemoryManager::release(ContextImplPtr TargetContext, SYCLMemObjT *MemObj,
                            void *MemAllocation,
                            std::vector<cl_event> DepEvents,
                            cl_event &OutEvent) {
  // There is no async API for memory releasing. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;
  MemObj->releaseMem(TargetContext, MemAllocation);
}

void MemoryManager::releaseMemBuf(ContextImplPtr TargetContext,
                                  SYCLMemObjT *MemObj, void *MemAllocation,
                                  void *UserPtr) {
  if (UserPtr == MemAllocation) {
    // Do nothing as it's user provided memory.
    return;
  }

  if (TargetContext->is_host()) {
    MemObj->releaseHostMem(MemAllocation);
    return;
  }

  CHECK_OCL_CODE(clReleaseMemObject((cl_mem)MemAllocation));
}

void *MemoryManager::allocate(ContextImplPtr TargetContext, SYCLMemObjT *MemObj,
                              bool InitFromUserData,
                              std::vector<cl_event> DepEvents,
                              cl_event &OutEvent) {
  // There is no async API for memory allocation. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  return MemObj->allocateMem(TargetContext, InitFromUserData, OutEvent);
}

void *MemoryManager::allocateMemBuffer(ContextImplPtr TargetContext,
                                       SYCLMemObjT *MemObj, void *UserPtr,
                                       bool HostPtrReadOnly, size_t Size,
                                       const EventImplPtr &InteropEvent,
                                       const ContextImplPtr &InteropContext,
                                       cl_event &OutEventToWait) {
  if (TargetContext->is_host()) {
    // Can return user pointer directly if it points to writable memory.
    if (UserPtr && HostPtrReadOnly == false)
      return UserPtr;

    void *NewMem = MemObj->allocateHostMem();

    // Need to initialize new memory if user provides pointer to read only
    // memory.
    if (UserPtr && HostPtrReadOnly == true)
      std::memcpy((char *)NewMem, (char *)UserPtr, Size);
    return NewMem;
  }

  // If memory object is created with interop c'tor.
  if (UserPtr && InteropContext) {
    // Return cl_mem as is if contexts match.
    if (TargetContext == InteropContext) {
      OutEventToWait = InteropEvent->getHandleRef();
      return UserPtr;
    }
    // Allocate new cl_mem and initialize from user provided one.
    assert(!"Not implemented");
    return nullptr;
  }

  // Create read_write mem object by default to handle arbitrary uses.
  cl_mem_flags CreationFlags = CL_MEM_READ_WRITE;

  if (UserPtr)
    CreationFlags |=
        HostPtrReadOnly ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
  cl_int Error = CL_SUCCESS;
  cl_mem NewMem = clCreateBuffer(TargetContext->getHandleRef(), CreationFlags,
                                 Size, UserPtr, &Error);
  CHECK_OCL_CODE(Error);
  return NewMem;
}

void copyH2D(SYCLMemObjT *SYCLMemObj, char *SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, cl_mem DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<cl_event> DepEvents,
             bool UseExclusiveQueue, cl_event &OutEvent) {
  // TODO: Handle images.

  // Adjust first dimension of copy range and offset as OpenCL expects size in
  // bytes.
  DstOffset[0] *= DstElemSize;
  SrcOffset[0] *= SrcElemSize;
  SrcAccessRange[0] *= SrcElemSize;
  DstAccessRange[0] *= DstElemSize;
  SrcSize[0] *= SrcElemSize;
  DstSize[0] *= DstElemSize;

  cl_command_queue CLQueue = UseExclusiveQueue
                                 ? TgtQueue->getExclusiveQueueHandleRef()
                                 : TgtQueue->getHandleRef();

  if (1 == DimDst && 1 == DimSrc) {
    CHECK_OCL_CODE(clEnqueueWriteBuffer(
        CLQueue, DstMem,
        /*blocking_write=*/CL_FALSE, DstOffset[0], DstAccessRange[0],
        SrcMem + DstOffset[0], DepEvents.size(), &DepEvents[0], &OutEvent));
  } else {
    size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
    size_t BufferSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;

    size_t HostRowPitch = (1 == DimDst) ? 0 : DstSize[0];
    size_t HostSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;
    CHECK_OCL_CODE(clEnqueueWriteBufferRect(
        CLQueue, DstMem,
        /*blocking_write=*/CL_FALSE, &DstOffset[0], &SrcOffset[0],
        &DstAccessRange[0], BufferRowPitch, BufferSlicePitch, HostRowPitch,
        HostSlicePitch, SrcMem, DepEvents.size(), &DepEvents[0], &OutEvent));
  }
}

void copyD2H(SYCLMemObjT *SYCLMemObj, cl_mem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, char *DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<cl_event> DepEvents,
             bool UseExclusiveQueue, cl_event &OutEvent) {
  // TODO: Handle images.

  // Adjust sizes of 1 dimensions as OpenCL expects size in bytes.
  DstOffset[0] *= DstElemSize;
  SrcOffset[0] *= SrcElemSize;
  SrcAccessRange[0] *= SrcElemSize;
  DstAccessRange[0] *= DstElemSize;
  SrcSize[0] *= SrcElemSize;
  DstSize[0] *= DstElemSize;

  cl_command_queue CLQueue = UseExclusiveQueue
                                 ? SrcQueue->getExclusiveQueueHandleRef()
                                 : SrcQueue->getHandleRef();

  if (1 == DimDst && 1 == DimSrc) {
    CHECK_OCL_CODE(clEnqueueReadBuffer(
        CLQueue, SrcMem,
        /*blocking_read=*/CL_FALSE, DstOffset[0], DstAccessRange[0],
        DstMem + DstOffset[0], DepEvents.size(), &DepEvents[0], &OutEvent));
  } else {
    size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
    size_t BufferSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;

    size_t HostRowPitch = (1 == DimDst) ? 0 : DstSize[0];
    size_t HostSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;
    CHECK_OCL_CODE(clEnqueueReadBufferRect(
        CLQueue, SrcMem,
        /*blocking_read=*/CL_FALSE, &SrcOffset[0], &DstOffset[0],
        &SrcAccessRange[0], BufferRowPitch, BufferSlicePitch, HostRowPitch,
        HostSlicePitch, DstMem, DepEvents.size(), &DepEvents[0], &OutEvent));
  }
}

void copyD2D(SYCLMemObjT *SYCLMemObj, cl_mem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, cl_mem DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<cl_event> DepEvents,
             bool UseExclusiveQueue, cl_event &OutEvent) {
  // TODO: Handle images.

  // Adjust sizes of 1 dimensions as OpenCL expects size in bytes.
  DstOffset[0] *= DstElemSize;
  SrcOffset[0] *= SrcElemSize;
  SrcAccessRange[0] *= SrcElemSize;
  SrcSize[0] *= SrcElemSize;
  DstSize[0] *= DstElemSize;

  cl_command_queue CLQueue = UseExclusiveQueue
                                 ? SrcQueue->getExclusiveQueueHandleRef()
                                 : SrcQueue->getHandleRef();

  if (1 == DimDst && 1 == DimSrc) {
    CHECK_OCL_CODE(clEnqueueCopyBuffer(
        CLQueue, SrcMem, DstMem, SrcOffset[0], DstOffset[0],
        SrcAccessRange[0], DepEvents.size(), &DepEvents[0], &OutEvent));
  } else {
    size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
    size_t BufferSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;

    size_t HostRowPitch = (1 == DimDst) ? 0 : DstSize[0];
    size_t HostSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;

    CHECK_OCL_CODE(clEnqueueCopyBufferRect(
        CLQueue, SrcMem, DstMem, &SrcOffset[0], &DstOffset[0],
        &SrcAccessRange[0], BufferRowPitch, BufferSlicePitch, HostRowPitch,
        HostSlicePitch, DepEvents.size(), &DepEvents[0], &OutEvent));
  }
}

static void copyH2H(SYCLMemObjT *SYCLMemObj, char *SrcMem,
                    QueueImplPtr SrcQueue, unsigned int DimSrc,
                    sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                    sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                    char *DstMem, QueueImplPtr TgtQueue, unsigned int DimDst,
                    sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
                    sycl::id<3> DstOffset, unsigned int DstElemSize,
                    std::vector<cl_event> DepEvents, bool UseExclusiveQueue,
                    cl_event &OutEvent) {
  if ((DimSrc != 1 || DimDst != 1) &&
      (SrcOffset != id<3>{0, 0, 0} || DstOffset != id<3>{0, 0, 0} ||
       SrcSize != SrcAccessRange || DstSize != DstAccessRange)) {
    assert(!"Not supported configuration of memcpy requested");
    throw runtime_error("Not supported configuration of memcpy requested");
  }

  DstOffset[0] *= DstElemSize;
  SrcOffset[0] *= SrcElemSize;

  size_t BytesToCopy =
      SrcAccessRange[0] * SrcElemSize * SrcAccessRange[1] * SrcAccessRange[2];

  std::memcpy(DstMem + DstOffset[0], SrcMem + SrcOffset[0], BytesToCopy);
}

// Copies memory between: host and device, host and host,
// device and device if memory objects bound to the one context.
void MemoryManager::copy(SYCLMemObjT *SYCLMemObj, void *SrcMem,
                         QueueImplPtr SrcQueue, unsigned int DimSrc,
                         sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                         sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                         void *DstMem, QueueImplPtr TgtQueue,
                         unsigned int DimDst, sycl::range<3> DstSize,
                         sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                         unsigned int DstElemSize,
                         std::vector<cl_event> DepEvents,
                         bool UseExclusiveQueue, cl_event &OutEvent) {

  if (SrcQueue->is_host()) {
    if (TgtQueue->is_host())
      copyH2H(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), UseExclusiveQueue, OutEvent);

    else
      copyH2D(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (cl_mem)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), UseExclusiveQueue, OutEvent);
  } else {
    if (TgtQueue->is_host())
      copyD2H(SYCLMemObj, (cl_mem)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), UseExclusiveQueue, OutEvent);
    else
      copyD2D(SYCLMemObj, (cl_mem)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (cl_mem)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), UseExclusiveQueue, OutEvent);
  }
}

void MemoryManager::fill(SYCLMemObjT *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         size_t PatternSize, const char *Pattern,
                         unsigned int Dim, sycl::range<3> Size,
                         sycl::range<3> Range, sycl::id<3> Offset,
                         unsigned int ElementSize,
                         std::vector<cl_event> DepEvents, cl_event &OutEvent) {
  // TODO: Handle images.

  if (Dim == 1) {
    CHECK_OCL_CODE(clEnqueueFillBuffer(
        Queue->getHandleRef(), (cl_mem)Mem, Pattern, PatternSize, Offset[0],
        Range[0] * ElementSize, DepEvents.size(), &DepEvents[0], &OutEvent));
    return;
  }

  assert(!"Not supported configuration of fill requested");
  throw runtime_error("Not supported configuration of fill requested");
}

void *MemoryManager::map(SYCLMemObjT *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         access::mode AccessMode, unsigned int Dim,
                         sycl::range<3> Size, sycl::range<3> AccessRange,
                         sycl::id<3> AccessOffset, unsigned int ElementSize,
                         std::vector<cl_event> DepEvents, cl_event &OutEvent) {
  if (Queue->is_host() || Dim != 1) {
    assert(!"Not supported configuration of map requested");
    throw runtime_error("Not supported configuration of map requested");
  }

  cl_map_flags Flags = 0;

  switch (AccessMode) {
  case access::mode::read:
    Flags |= CL_MAP_READ;
    break;
  case access::mode::write:
    Flags |= CL_MAP_WRITE;
    break;
  case access::mode::read_write:
  case access::mode::atomic:
    Flags = CL_MAP_WRITE | CL_MAP_READ;
    break;
  case access::mode::discard_write:
  case access::mode::discard_read_write:
    Flags |= CL_MAP_WRITE_INVALIDATE_REGION;
    break;
  }

  AccessOffset[0] *= ElementSize;
  AccessRange[0] *= ElementSize;

  cl_int Error = CL_SUCCESS;
  void *MappedPtr = clEnqueueMapBuffer(
      Queue->getHandleRef(), (cl_mem)Mem, CL_FALSE, Flags, AccessOffset[0],
      AccessRange[0], DepEvents.size(),
      DepEvents.empty() ? nullptr : &DepEvents[0], &OutEvent, &Error);
  CHECK_OCL_CODE(Error);
  return MappedPtr;
}

void MemoryManager::unmap(SYCLMemObjT *SYCLMemObj, void *Mem,
                          QueueImplPtr Queue, void *MappedPtr,
                          std::vector<cl_event> DepEvents,
                          bool UseExclusiveQueue, cl_event &OutEvent) {
  cl_int Error = CL_SUCCESS;
  Error = clEnqueueUnmapMemObject(
      UseExclusiveQueue ? Queue->getExclusiveQueueHandleRef()
                        : Queue->getHandleRef(),
      (cl_mem)Mem, MappedPtr, DepEvents.size(),
      DepEvents.empty() ? nullptr : &DepEvents[0], &OutEvent);
  CHECK_OCL_CODE(Error);
}

} // namespace detail
} // namespace sycl
} // namespace cl
