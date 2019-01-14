//==---------- buffer_impl.hpp --- SYCL buffer ----------------*- C++-*---==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>

#include <functional>
#include <memory>
#include <type_traits>

namespace cl {
namespace sycl {
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<cl::sycl::detail::event_impl>;
// Forward declarations
template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;
template <typename T, int dimensions, typename AllocatorT> class buffer;
class handler;
class queue;
template <int dimentions> class id;
template <int dimentions> class range;
template <class T> using buffer_allocator = std::allocator<T>;
namespace detail {
template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator<T>>
class buffer_impl {
public:
  buffer_impl(const range<dimensions> &bufferRange,
              const property_list &propList = {})
      : buffer_impl((T *)nullptr, bufferRange, propList) {}

  buffer_impl(T *hostData, const range<dimensions> &bufferRange,
              const property_list &propList = {})
      : Range(bufferRange), Props(propList) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      BufPtr = hostData;
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<T *>(BufData.data());
      if (hostData != nullptr) {
        set_final_data(hostData);
        std::copy(hostData, hostData + get_count(), BufPtr);
      }
    }
  }

  // TODO temporary solution for allowing initialisation with const data
  buffer_impl(const T *hostData, const range<dimensions> &bufferRange,
              const property_list &propList = {})
      : Range(bufferRange), Props(propList) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      // TODO make this buffer read only
      BufPtr = const_cast<T *>(hostData);
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<T *>(BufData.data());
      if (hostData != nullptr) {
        std::copy(hostData, hostData + get_count(), BufPtr);
      }
    }
  }

  buffer_impl(const shared_ptr_class<T> &hostData,
              const range<dimensions> &bufferRange,
              const property_list &propList = {})
      : Range(bufferRange), Props(propList) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      BufPtr = hostData.get();
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<T *>(BufData.data());
      if (hostData.get() != nullptr) {
        weak_ptr_class<T> hostDataWeak = hostData;
        set_final_data(hostDataWeak);
        std::copy(hostData.get(), hostData.get() + get_count(), BufPtr);
      }
    }
  }

  template <class InputIterator, int N = dimensions,
            typename = std::enable_if<N == 1>>
  buffer_impl(InputIterator first, InputIterator last,
              const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))), Props(propList) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      BufPtr = &*first;
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<T *>(BufData.data());
      std::copy(first, last, BufPtr);
    }
  }

  template <int N = dimensions, typename = std::enable_if<N == 1>>
  buffer_impl(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent = {})
      : OpenCLInterop(true), AvailableEvent(AvailableEvent) {
    if (SyclContext.is_host())
      throw cl::sycl::invalid_parameter_error(
          "Creation of interoperability buffer using host context is not "
          "allowed");

    CHECK_OCL_CODE(clGetMemObjectInfo(MemObject, CL_MEM_CONTEXT,
                                      sizeof(OpenCLContext), &OpenCLContext, nullptr));
    if (SyclContext.get() != OpenCLContext)
      throw cl::sycl::invalid_parameter_error(
          "Input context must be the same as the context of cl_mem");
    OCLState.Mem = MemObject;
    CHECK_OCL_CODE(clRetainMemObject(MemObject));
  }

  range<dimensions> get_range() const { return Range; }

  size_t get_count() const { return Range.size(); }

  size_t get_size() const { return get_count() * sizeof(T); }

  ~buffer_impl() {
    if (!OpenCLInterop)
      // TODO. Use node instead?
      simple_scheduler::Scheduler::getInstance()
          .copyBack<access::mode::read_write, access::target::host_buffer>(
              *this);

    if (uploadData != nullptr) {
      uploadData();
    }

    // TODO. Use node instead?
    simple_scheduler::Scheduler::getInstance().removeBuffer(*this);

    if (OpenCLInterop)
      CHECK_OCL_CODE_NO_EXC(clReleaseMemObject(OCLState.Mem));
  }

  void set_final_data(std::nullptr_t) { uploadData = nullptr; }

  void set_final_data(weak_ptr_class<T> final_data) {
    if (OpenCLInterop)
      throw cl::sycl::runtime_error(
          "set_final_data could not be used with interoperability buffer");
    uploadData = [this, final_data]() {
      if (auto finalData = final_data.lock()) {
        std::copy(BufPtr, BufPtr + get_count(), finalData.get());
      }
    };
  }

  template <typename Destination> void set_final_data(Destination final_data) {
    if (OpenCLInterop)
      throw cl::sycl::runtime_error(
          "set_final_data could not be used with interoperability buffer");
    static_assert(!std::is_const<Destination>::value,
                  "Сan not write in a constant Destination. Destination should "
                  "not be const.");
    uploadData = [this, final_data]() mutable {
      std::copy(BufPtr, BufPtr + get_count(), final_data);
    };
  }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             handler &commandGroupHandler) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t>(
        Buffer, commandGroupHandler);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t>(Buffer);
  }

public:
  void moveMemoryTo(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                    EventImplPtr Event);

  void fill(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, const void *Pattern, size_t PatternSize,
            int Dim, size_t *Offset, size_t *Range);

  void copy(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, simple_scheduler::BufferReqPtr SrcReq,
            const int DimSrc, const size_t *const SrcRange,
            const size_t *const SrcOffset, const size_t *const DestOffset,
            const size_t SizeTySrc, const size_t SizeSrc,
            const size_t *const BuffSrcRange);

  size_t convertSycl2OCLMode(cl::sycl::access::mode mode);

  bool isValidAccessToMem(cl::sycl::access::mode AccessMode);

  void allocate(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                EventImplPtr Event, cl::sycl::access::mode mode);

  cl_mem getOpenCLMem() const;

private:
  // There are internal structures in this section.
  enum DeviceMemoryState {
    DMS_NULL,     // No data were transferred between host and device.
    DMS_COPIED,   // Data were copied from host to device.
    DMS_MODIFIED, // Data in device memory were modified.
    DMS_HOST      // Use host pointer for device memory
  };
  // Contains the latest virtual state of buffer during commands enqueueing.
  // TODO: Need to find better solution, at least make state for each device.
  struct OpenCLMemState {
    QueueImplPtr Queue;
    cl_mem Mem = nullptr;
  };

private:
  // This field must be the first to guarantee that it's safe to use
  // reinterpret casting while setting kernel arguments in order to get cl_mem
  // value from the buffer regardless of its dimensionality.
  OpenCLMemState OCLState;
  bool OpenCLInterop = false;
  event AvailableEvent;
  cl_context OpenCLContext = nullptr;
  T *BufPtr = nullptr;
  vector_class<byte> BufData;
  // TODO: enable support of cl_mem objects from multiple contexts
  // TODO: at the current moment, using a buffer on multiple devices
  // or on a device and a host simultaneously is not supported (the
  // implementation is incorrect).
  range<dimensions> Range;
  property_list Props;
  std::function<void(void)> uploadData = nullptr;
  template <typename DataT, int Dimensions, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  friend class cl::sycl::accessor;
};

template <typename T, int dimensions, typename AllocatorT>
void buffer_impl<T, dimensions, AllocatorT>::fill(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event, const void *Pattern, size_t PatternSize, int Dim,
    size_t *OffsetArr, size_t *RangeArr) {

  assert(dimensions == 1 &&
         "OpenCL doesn't support multidimensional fill method.");
  assert(!Queue->is_host() && "Host case is handled in other place.");

  size_t Offset = OffsetArr[0];
  size_t Size = RangeArr[0] * PatternSize;

  cl::sycl::context Context = Queue->get_context();

  OCLState.Queue = std::move(Queue);
  Event->setIsHostEvent(false);

  cl_event &BufEvent = Event->getHandleRef();
  std::vector<cl_event> CLEvents =
      detail::getOrWaitEvents(std::move(DepEvents), Context);

  cl_command_queue CommandQueue = OCLState.Queue->get();
  cl_int Error = clEnqueueFillBuffer(
      CommandQueue, OCLState.Mem, Pattern, PatternSize, Offset, Size,
      CLEvents.size(), CLEvents.data(), &BufEvent);

  CHECK_OCL_CODE(Error);
  CHECK_OCL_CODE(clReleaseCommandQueue(CommandQueue));
}

template <typename T, int dimensions, typename AllocatorT>
void buffer_impl<T, dimensions, AllocatorT>::copy(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event, simple_scheduler::BufferReqPtr SrcReq, const int DimSrc,
    const size_t *const SrcRange, const size_t *const SrcOffset,
    const size_t *const DestOffset, const size_t SizeTySrc,
    const size_t SizeSrc, const size_t *const BuffSrcRange) {
  assert(!Queue->is_host() && "Host case is handled in other place.");

  size_t *BuffDestRange = &get_range()[0];
  size_t SizeTyDest = sizeof(T);
  const int DimDest = dimensions;

  cl::sycl::context Context = Queue->get_context();

  cl_event &BufEvent = Event->getHandleRef();
  std::vector<cl_event> CLEvents =
      detail::getOrWaitEvents(std::move(DepEvents), Context);
  cl_int Error;

  cl_command_queue CommandQueue = Queue->get();
  if (1 == DimSrc && 1 == DimDest) {
    Error = clEnqueueCopyBuffer(CommandQueue, SrcReq->getCLMemObject(),
                                OCLState.Mem, SrcOffset[0], DestOffset[0],
                                SizeSrc * SizeTySrc, CLEvents.size(),
                                CLEvents.data(), &BufEvent);
  } else {
    size_t SrcOrigin[3] = {SrcOffset[0] * SizeTySrc,
                            (1 == DimSrc) ? 0 : SrcOffset[1],
                            (3 == DimSrc) ? SrcOffset[2] : 0};
    size_t DstOrigin[3] = {DestOffset[0] * SizeTyDest,
                            (1 == DimDest) ? 0 : DestOffset[1],
                            (3 == DimDest) ? DestOffset[2] : 0};
    size_t Region[3] = {SrcRange[0] * SizeTySrc,
                        (1 == DimSrc) ? 1 : SrcRange[1],
                        (3 == DimSrc) ? SrcRange[2] : 1};
    size_t SrcRowPitch = (1 == DimSrc) ? 0 : SizeTySrc * BuffSrcRange[0];
    size_t SrcSlicePitch =
        (3 == DimSrc) ? SizeTySrc * BuffSrcRange[0] * BuffSrcRange[1] : 0;
    size_t DstRowPitch = (1 == DimSrc) ? 0 : SizeTyDest * BuffDestRange[0];
    size_t DstSlicePitch =
        (3 == DimSrc) ? SizeTyDest * BuffDestRange[0] * BuffDestRange[1] : 0;

    Error = clEnqueueCopyBufferRect(
        CommandQueue, SrcReq->getCLMemObject(), OCLState.Mem, SrcOrigin,
        DstOrigin, Region, SrcRowPitch, SrcSlicePitch, DstRowPitch,
        DstSlicePitch, CLEvents.size(), CLEvents.data(), &BufEvent);
  }
  CHECK_OCL_CODE(Error);
  CHECK_OCL_CODE(clReleaseCommandQueue(CommandQueue));
  OCLState.Queue = std::move(Queue);
  Event->setIsHostEvent(false);
}

template <typename T, int dimensions, typename AllocatorT>
void buffer_impl<T, dimensions, AllocatorT>::moveMemoryTo(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event) {

  cl::sycl::context Context = Queue->get_context();

  if (OpenCLInterop && (Context.get() != OpenCLContext))
    throw cl::sycl::runtime_error(
        "Interoperability buffer could not be used in a context other than the "
        "context associated with the OpenCL memory object.");

  // TODO: Move all implementation specific commands to separate file?
  // TODO: Make allocation in separate command?

  // Special case, move to "user host"
  // TODO: Check discuss if "user host" and "host device" are the same.
  if ((Queue->is_host()) && (OCLState.Queue->is_host())) {
    detail::waitEvents(DepEvents);
    Event->setIsHostEvent(true);
    OCLState.Queue = std::move(Queue);
    return;
  }

  assert(OCLState.Queue->get_context() != Context ||
         OCLState.Queue->get_device() != Queue->get_device() &&
             "Attempt to move to the same env");

  // Copy from OCL device to host device.
  if (!OCLState.Queue->is_host() && Queue->is_host()) {
    const size_t ByteSize = get_size();

    std::vector<cl_event> CLEvents =
        detail::getOrWaitEvents(std::move(DepEvents), Context);

    // TODO: Handle different situations with host PTR.
    // Enqueue copying from OCL buffer to host.
    cl_event &ReadBufEvent = Event->getHandleRef();
    cl_int Error = clEnqueueReadBuffer(
        OCLState.Queue->getHandleRef(), OCLState.Mem,
        /*blocking_read=*/CL_FALSE, /*offset=*/0, ByteSize, BufPtr,
        CLEvents.size(), CLEvents.data(), &ReadBufEvent);
    CHECK_OCL_CODE(Error);

    Event->setIsHostEvent(false);

    OCLState.Queue = std::move(Queue);
    OCLState.Mem = nullptr;
    return;
  }
  // Copy from host to OCL device.
  if (OCLState.Queue->is_host() && !Queue->is_host()) {
    const size_t ByteSize = get_size();
    cl_int Error;
    cl_mem Mem = clCreateBuffer(Context.get(), CL_MEM_READ_WRITE, ByteSize,
                                /*host_ptr=*/nullptr, &Error);
    CHECK_OCL_CODE(Error);

    OCLState.Queue = std::move(Queue);
    OCLState.Mem = Mem;

    // Just exit if nothing to read from host.
    if (nullptr == BufPtr) {
      return;
    }
    std::vector<cl_event> CLEvents =
        detail::getOrWaitEvents(std::move(DepEvents), Context);
    cl_event &WriteBufEvent = Event->getHandleRef();
    // Enqueue copying from host to new OCL buffer.
    Error =
        clEnqueueWriteBuffer(OCLState.Queue->getHandleRef(), Mem,
                             /*blocking_write=*/CL_FALSE, /*offset=*/0,
                             ByteSize, BufPtr, CLEvents.size(), CLEvents.data(),
                             &WriteBufEvent); // replace &WriteBufEvent to NULL
    CHECK_OCL_CODE(Error);
    Event->setIsHostEvent(false);

    return;
  }

  assert(0 && "Not handled");
}

template <typename T, int dimensions, typename AllocatorT>
size_t buffer_impl<T, dimensions, AllocatorT>::convertSycl2OCLMode(
    cl::sycl::access::mode mode) {
  switch (mode) {
  case cl::sycl::access::mode::read:
    return CL_MEM_READ_ONLY;
  case cl::sycl::access::mode::write:
    return CL_MEM_WRITE_ONLY;
  case cl::sycl::access::mode::read_write:
  case cl::sycl::access::mode::atomic:
    return CL_MEM_READ_WRITE;
  default:
    assert(0 && "Unhandled conversion from Sycl access mode to OCL one.");
    return 0;
  }
}

template <typename T, int dimensions, typename AllocatorT>
bool buffer_impl<T, dimensions, AllocatorT>::isValidAccessToMem(
    cl::sycl::access::mode AccessMode) {
  cl_mem_flags Flags;
  assert(OCLState.Mem != nullptr &&
         "OpenCL memory associated with the buffer is null");
  CHECK_OCL_CODE(clGetMemObjectInfo(OCLState.Mem, CL_MEM_FLAGS, sizeof(Flags),
                                    &Flags, nullptr));
  if (((Flags & CL_MEM_READ_WRITE) == 0) &&
      ((convertSycl2OCLMode(AccessMode) & Flags) == 0))
    return false;
  return true;
}

template <typename T, int dimensions, typename AllocatorT>
void buffer_impl<T, dimensions, AllocatorT>::allocate(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event, cl::sycl::access::mode mode) {

  detail::waitEvents(DepEvents);

  cl::sycl::context Context = Queue->get_context();

  if (OpenCLInterop && (Context.get() != OpenCLContext))
    throw cl::sycl::runtime_error(
        "Interoperability buffer could not be used in a context other than the "
        "context associated with the OpenCL memory object.");

  if (OpenCLInterop) {
    AvailableEvent.wait();
    OCLState.Queue = std::move(Queue);
    Event->setIsHostEvent(true);
    return;
  }

  if (!Queue->is_host()) {
    size_t ByteSize = get_size();
    cl_int Error;

    cl_mem Mem = clCreateBuffer(Context.get(), convertSycl2OCLMode(mode),
                                ByteSize, nullptr, &Error);
    CHECK_OCL_CODE(Error);

    cl_event &WriteBufEvent = Event->getHandleRef();
    Error = clEnqueueWriteBuffer(Queue->getHandleRef(), Mem,
                                 /*blocking_write=*/CL_FALSE, /*offset=*/0,
                                 ByteSize, BufPtr, /*num_of_events=*/0,
                                 /*dep_list=*/nullptr, &WriteBufEvent);
    CHECK_OCL_CODE(Error);

    OCLState.Queue = std::move(Queue);
    OCLState.Mem = Mem;

    Event->setIsHostEvent(false);

    return;
  }
  if (Queue->is_host()) {
    Event->setIsHostEvent(true);
    OCLState.Queue = std::move(Queue);
    return;
  }
  assert(0 && "Unhandled Alloca");
}

template <typename T, int dimensions, typename AllocatorT>
cl_mem buffer_impl<T, dimensions, AllocatorT>::getOpenCLMem() const {
  assert(nullptr != OCLState.Mem);
  return OCLState.Mem;
}

} // namespace detail
} // namespace sycl
} // namespace cl
