//==---------- buffer_impl.hpp --- SYCL buffer ----------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
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
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;
// Forward declarations
template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;
template <typename T, int dimensions, typename AllocatorT> class buffer;
class handler;
class queue;
template <int dimentions> class id;
template <int dimentions> class range;
using buffer_allocator = std::allocator<char>;
namespace detail {
template <typename AllocatorT> class buffer_impl {
public:
  buffer_impl(const size_t sizeInBytes, const property_list &propList,
              AllocatorT allocator = AllocatorT())
      : buffer_impl((void *)nullptr, sizeInBytes, propList, allocator) {}

  buffer_impl(void *hostData, const size_t sizeInBytes,
              const property_list &propList,
              AllocatorT allocator = AllocatorT())
      : SizeInBytes(sizeInBytes), Props(propList), MAllocator(allocator) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      BufPtr = hostData;
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<void *>(BufData.data());
      if (hostData != nullptr) {
        auto HostPtr = reinterpret_cast<char *>(hostData);
        set_final_data(HostPtr);
        std::copy(HostPtr, HostPtr + SizeInBytes, BufData.data());
      }
    }
  }

  // TODO temporary solution for allowing initialisation with const data
  buffer_impl(const void *hostData, const size_t sizeInBytes,
              const property_list &propList,
              AllocatorT allocator = AllocatorT())
      : SizeInBytes(sizeInBytes), Props(propList), MAllocator(allocator) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      // TODO make this buffer read only
      BufPtr = const_cast<void *>(hostData);
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<void *>(BufData.data());
      if (hostData != nullptr) {
        std::copy((char *)hostData, (char *)hostData + SizeInBytes,
                  BufData.data());
      }
    }
  }

  template <typename T>
  buffer_impl(const shared_ptr_class<T> &hostData, const size_t sizeInBytes,
              const property_list &propList,
              AllocatorT allocator = AllocatorT())
      : SizeInBytes(sizeInBytes), Props(propList), MAllocator(allocator) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      BufPtr = hostData.get();
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<void *>(BufData.data());
      if (hostData.get() != nullptr) {
        weak_ptr_class<T> hostDataWeak = hostData;
        set_final_data(hostDataWeak);
        std::copy((char *)hostData.get(), (char *)hostData.get() + SizeInBytes,
                  BufData.data());
      }
    }
  }

  template <class InputIterator>
  buffer_impl(InputIterator first, InputIterator last, const size_t sizeInBytes,
              const property_list &propList,
              AllocatorT allocator = AllocatorT())
      : SizeInBytes(sizeInBytes), Props(propList), MAllocator(allocator) {
    if (Props.has_property<property::buffer::use_host_ptr>()) {
      // TODO next line looks unsafe
      BufPtr = &*first;
    } else {
      BufData.resize(get_size());
      BufPtr = reinterpret_cast<void *>(BufData.data());
      // We need cast BufPtr to pointer to the iteration type to get correct
      // offset in std::copy when it will increment destination pointer.
      auto *Ptr = reinterpret_cast<
          typename std::iterator_traits<InputIterator>::pointer>(BufPtr);
      std::copy(first, last, Ptr);
    }
  }

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent = {})
      : OpenCLInterop(true), AvailableEvent(AvailableEvent) {
    if (SyclContext.is_host())
      throw cl::sycl::invalid_parameter_error(
          "Creation of interoperability buffer using host context is not "
          "allowed");

    CHECK_OCL_CODE(clGetMemObjectInfo(MemObject, CL_MEM_CONTEXT,
                                      sizeof(OpenCLContext), &OpenCLContext,
                                      nullptr));
    if (detail::getSyclObjImpl(SyclContext)->getHandleRef() != OpenCLContext)
      throw cl::sycl::invalid_parameter_error(
          "Input context must be the same as the context of cl_mem");
    OCLState.Mem = MemObject;
    CHECK_OCL_CODE(clRetainMemObject(MemObject));
  }

  size_t get_size() const { return SizeInBytes; }

  ~buffer_impl() {
    if (!OpenCLInterop)
      // TODO. Use node instead?
      simple_scheduler::Scheduler::getInstance()
          .copyBack<access::mode::read_write, access::target::host_buffer>(
              *this);

    if (uploadData != nullptr && NeedWriteBack) {
      uploadData();
    }

    // TODO. Use node instead?
    simple_scheduler::Scheduler::getInstance().removeBuffer(*this);

    if (OpenCLInterop)
      CHECK_OCL_CODE_NO_EXC(clReleaseMemObject(OCLState.Mem));
  }

  void set_final_data(std::nullptr_t) { uploadData = nullptr; }

  template <typename T> void set_final_data(weak_ptr_class<T> final_data) {
    if (OpenCLInterop)
      throw cl::sycl::runtime_error(
          "set_final_data could not be used with interoperability buffer");
    uploadData = [this, final_data]() {
      if (auto finalData = final_data.lock()) {
        T *Ptr = reinterpret_cast<T *>(BufPtr);
        std::copy(Ptr, Ptr + SizeInBytes / sizeof(T), finalData.get());
      }
    };
  }

  template <typename Destination> void set_final_data(Destination final_data) {
    if (OpenCLInterop)
      throw cl::sycl::runtime_error(
          "set_final_data could not be used with interoperability buffer");
    static_assert(!std::is_const<Destination>::value,
                  "Can not write in a constant Destination. Destination should "
                  "not be const.");
    uploadData = [this, final_data]() mutable {
      auto *Ptr =
          reinterpret_cast<typename std::iterator_traits<Destination>::pointer>(
              BufPtr);
      size_t ValSize =
          sizeof(typename std::iterator_traits<Destination>::value_type);
      std::copy(Ptr, Ptr + SizeInBytes / ValSize, final_data);
    };
  }

  void set_write_back(bool flag) { NeedWriteBack = flag; }

  AllocatorT get_allocator() const { return MAllocator; }

  template <typename T, int dimensions, access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             handler &commandGroupHandler) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t>(
        Buffer, commandGroupHandler);
  }

  template <typename T, int dimensions, access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t>(Buffer);
  }

  template <typename T, int dimensions, access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t>(
        Buffer, commandGroupHandler, accessRange, accessOffset);
  }

  template <typename T, int dimensions, access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             range<dimensions> accessRange, id<dimensions> accessOffset) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t>(Buffer, accessRange,
                                                  accessOffset);
  }

  template <typename propertyT> bool has_property() const {
    return Props.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return Props.get_property<propertyT>();
  }

public:
  void moveMemoryTo(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                    EventImplPtr Event);

  void fill(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, const void *Pattern, size_t PatternSize,
            int Dim, size_t *Offset, size_t *Range);

  void copy(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, simple_scheduler::BufferReqPtr SrcReq,
            const int DimSrc, const int DimDest, const size_t *const SrcRange,
            const size_t *const SrcOffset, const size_t *const DestOffset,
            const size_t SizeTySrc, const size_t SizeTyDest,
            const size_t SizeSrc, const size_t *const BuffSrcRange,
            const size_t *const BuffDestRange);

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
  AllocatorT MAllocator;
  OpenCLMemState OCLState;
  bool OpenCLInterop = false;
  bool NeedWriteBack = true;
  event AvailableEvent;
  cl_context OpenCLContext = nullptr;
  void *BufPtr = nullptr;
  vector_class<byte> BufData;
  // TODO: enable support of cl_mem objects from multiple contexts
  // TODO: at the current moment, using a buffer on multiple devices
  // or on a device and a host simultaneously is not supported (the
  // implementation is incorrect).
  size_t SizeInBytes = 0;
  property_list Props;
  std::function<void(void)> uploadData = nullptr;
  template <typename DataT, int Dimensions, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  friend class cl::sycl::accessor;
};

template <typename AllocatorT>
void buffer_impl<AllocatorT>::fill(QueueImplPtr Queue,
                                   std::vector<cl::sycl::event> DepEvents,
                                   EventImplPtr Event, const void *Pattern,
                                   size_t PatternSize, int Dim,
                                   size_t *OffsetArr, size_t *RangeArr) {

  assert(Dim == 1 && "OpenCL doesn't support multidimensional fill method.");
  assert(!Queue->is_host() && "Host case is handled in other place.");

  size_t Offset = OffsetArr[0];
  size_t Size = RangeArr[0] * PatternSize;

  ContextImplPtr Context = detail::getSyclObjImpl(Queue->get_context());

  OCLState.Queue = std::move(Queue);

  cl_event &BufEvent = Event->getHandleRef();
  std::vector<cl_event> CLEvents =
      detail::getOrWaitEvents(std::move(DepEvents), Context);

  cl_command_queue CommandQueue = OCLState.Queue->get();
  cl_int Error = clEnqueueFillBuffer(CommandQueue, OCLState.Mem, Pattern,
                                     PatternSize, Offset, Size, CLEvents.size(),
                                     CLEvents.data(), &BufEvent);

  CHECK_OCL_CODE(Error);
  CHECK_OCL_CODE(clReleaseCommandQueue(CommandQueue));
  Event->setContextImpl(Context);
}

template <typename AllocatorT>
void buffer_impl<AllocatorT>::copy(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event, simple_scheduler::BufferReqPtr SrcReq, const int DimSrc,
    const int DimDest, const size_t *const SrcRange,
    const size_t *const SrcOffset, const size_t *const DestOffset,
    const size_t SizeTySrc, const size_t SizeTyDest, const size_t SizeSrc,
    const size_t *const BuffSrcRange, const size_t *const BuffDestRange) {
  assert(!Queue->is_host() && "Host case is handled in other place.");

  ContextImplPtr Context = detail::getSyclObjImpl(Queue->get_context());

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
  Event->setContextImpl(Context);
}

template <typename AllocatorT>
void buffer_impl<AllocatorT>::moveMemoryTo(
    QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
    EventImplPtr Event) {

  ContextImplPtr Context = detail::getSyclObjImpl(Queue->get_context());

  if (OpenCLInterop && (Context->getHandleRef() != OpenCLContext))
    throw cl::sycl::runtime_error(
        "Interoperability buffer could not be used in a context other than the "
        "context associated with the OpenCL memory object.");

  // TODO: Move all implementation specific commands to separate file?
  // TODO: Make allocation in separate command?

  // Special case, move to "user host"
  // TODO: Check discuss if "user host" and "host device" are the same.
  if ((Queue->is_host()) && (OCLState.Queue->is_host())) {
    detail::waitEvents(DepEvents);
    Event->setContextImpl(Context);
    OCLState.Queue = std::move(Queue);
    return;
  }

  assert(OCLState.Queue->get_context() != Queue->get_context() ||
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
    Event->setContextImpl(
        detail::getSyclObjImpl(OCLState.Queue->get_context()));

    OCLState.Queue = std::move(Queue);
    return;
  }
  // Copy from host to OCL device.
  if (OCLState.Queue->is_host() && !Queue->is_host()) {
    if (nullptr == BufPtr) {
      return;
    }

    cl_int Error;
    const size_t ByteSize = get_size();

    // We don't create new OpenCL buffer object to copy from OCL device to host
    // when we already have them in OCLState. But if contexts of buffer object
    // from OCLState and input Queue are not same - we should create new OpenCL
    // buffer object.
    bool NeedToCreateCLBuffer = true;

    if (OCLState.Mem != nullptr) {
      cl_context MemCtx;
      Error = clGetMemObjectInfo(OCLState.Mem, CL_MEM_CONTEXT,
                                 sizeof(cl_context), &MemCtx, nullptr);
      CHECK_OCL_CODE(Error);
      NeedToCreateCLBuffer = MemCtx != Context->getHandleRef();
    }

    if (NeedToCreateCLBuffer) {
      OCLState.Mem =
          clCreateBuffer(Context->getHandleRef(), CL_MEM_READ_WRITE, ByteSize,
                         /*host_ptr=*/nullptr, &Error);
      CHECK_OCL_CODE(Error);
    }

    OCLState.Queue = std::move(Queue);

    std::vector<cl_event> CLEvents =
        detail::getOrWaitEvents(std::move(DepEvents), Context);
    cl_event &WriteBufEvent = Event->getHandleRef();
    // Enqueue copying from host to new OCL buffer.
    Error =
        clEnqueueWriteBuffer(OCLState.Queue->getHandleRef(), OCLState.Mem,
                             /*blocking_write=*/CL_FALSE, /*offset=*/0,
                             ByteSize, BufPtr, CLEvents.size(), CLEvents.data(),
                             &WriteBufEvent); // replace &WriteBufEvent to NULL
    CHECK_OCL_CODE(Error);
    Event->setContextImpl(Context);

    return;
  }

  assert(0 && "Not handled");
}

template <typename AllocatorT>
size_t
buffer_impl<AllocatorT>::convertSycl2OCLMode(cl::sycl::access::mode mode) {
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

template <typename AllocatorT>
bool buffer_impl<AllocatorT>::isValidAccessToMem(
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

template <typename AllocatorT>
void buffer_impl<AllocatorT>::allocate(QueueImplPtr Queue,
                                       std::vector<cl::sycl::event> DepEvents,
                                       EventImplPtr Event,
                                       cl::sycl::access::mode mode) {

  detail::waitEvents(DepEvents);

  ContextImplPtr Context = detail::getSyclObjImpl(Queue->get_context());

  if (OpenCLInterop && (Context->getHandleRef() != OpenCLContext))
    throw cl::sycl::runtime_error(
        "Interoperability buffer could not be used in a context other than the "
        "context associated with the OpenCL memory object.");

  if (OpenCLInterop) {
    // For interoperability instance of the SYCL buffer class being constructed
    // must wait for the SYCL event parameter, if one is provided,
    // availableEvent to signal that the cl_mem instance is ready to be used
    // Move availableEvent to SYCL scheduler ownership to handle dependencies.
    Event = detail::getSyclObjImpl(AvailableEvent);
    OCLState.Queue = std::move(Queue);
    return;
  }

  if (!Queue->is_host()) {
    size_t ByteSize = get_size();
    cl_int Error;

    cl_mem Mem =
        clCreateBuffer(Context->getHandleRef(), convertSycl2OCLMode(mode),
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
    Event->setContextImpl(Context);

    return;
  }
  if (Queue->is_host()) {
    Event->setContextImpl(Context);
    OCLState.Queue = std::move(Queue);
    return;
  }
  assert(0 && "Unhandled Alloca");
}

template <typename AllocatorT>
cl_mem buffer_impl<AllocatorT>::getOpenCLMem() const {
  assert(nullptr != OCLState.Mem);
  return OCLState.Mem;
}

} // namespace detail
} // namespace sycl
} // namespace cl
