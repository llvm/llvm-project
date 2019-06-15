//==----------------- buffer_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/sycl_mem_obj.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>

#include <functional>
#include <memory>
#include <type_traits>

namespace cl {
namespace sycl {
// Forward declarations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;
template <typename T, int Dimensions, typename AllocatorT> class buffer;
class handler;

using buffer_allocator = aligned_allocator<char, /*Alignment*/64>;

namespace detail {
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

using cl::sycl::detail::SYCLMemObjT;

using cl::sycl::detail::MemoryManager;

template <typename AllocatorT> class buffer_impl : public SYCLMemObjT {
public:
  buffer_impl(size_t SizeInBytes, const property_list &PropList,
              AllocatorT Allocator = AllocatorT())
      : buffer_impl((void *)nullptr, SizeInBytes, PropList, Allocator) {}

  buffer_impl(void *HostData, size_t SizeInBytes, const property_list &Props,
              AllocatorT Allocator = AllocatorT())
      : MSizeInBytes(SizeInBytes), MProps(Props), MAllocator(Allocator) {

    if (!HostData)
      return;

    set_final_data(reinterpret_cast<char *>(HostData));
    if (MProps.has_property<property::buffer::use_host_ptr>()) {
      MUserPtr = HostData;
      return;
    }

    // TODO: Reuse user's pointer if it has sufficient alignment.
    MShadowCopy = allocateHostMem();
    MUserPtr = MShadowCopy;
    std::memcpy(MUserPtr, HostData, SizeInBytes);
  }

  buffer_impl(const void *HostData, size_t SizeInBytes,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : buffer_impl(const_cast<void *>(HostData), SizeInBytes, Props,
                    Allocator) {
    MHostPtrReadOnly = true;
  }

  template <typename T>
  buffer_impl(const shared_ptr_class<T> &HostData, const size_t SizeInBytes,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : MSizeInBytes(SizeInBytes), MProps(Props), MAllocator(Allocator) {
    // HostData can be destructed by the user so need to make copy
    MUserPtr = MShadowCopy = allocateHostMem();

    std::copy(HostData.get(), HostData.get() + SizeInBytes / sizeof(T),
              (T *)MUserPtr);

    set_final_data(weak_ptr_class<T>(HostData));
  }

  template <typename Iterator> struct is_const_iterator {
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    static constexpr bool value =
        std::is_const<typename std::remove_pointer<pointer>::type>::value;
  };

  template <typename Iterator>
  using EnableIfConstIterator =
      typename std::enable_if<is_const_iterator<Iterator>::value,
                              Iterator>::type;

  template <typename Iterator>
  using EnableIfNotConstIterator =
      typename std::enable_if<!is_const_iterator<Iterator>::value,
                              Iterator>::type;

  template <class InputIterator>
  buffer_impl(EnableIfNotConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, const property_list &Props,
              AllocatorT Allocator = AllocatorT())
      : MSizeInBytes(SizeInBytes), MProps(Props), MAllocator(Allocator) {

    // TODO: There is contradiction is the spec. It says SYCL RT must not
    // allocate additional memory on the host if use_host_ptr prop was passed.
    // On the other hand it says that SYCL RT should allocate temporal memory in
    // this c'tor.

    if (0) {
      MUserPtr = MShadowCopy = allocateHostMem();
    } else {
      size_t AllocatorValueSize = sizeof(typename AllocatorT::value_type);
      size_t AllocationSize = get_size() / AllocatorValueSize;
      AllocationSize += (get_size() % AllocatorValueSize) ? 1 : 0;
      MUserPtr = MShadowCopy = MAllocator.allocate(AllocationSize);
    }

    // We need to cast MUserPtr to pointer to the iteration type to get correct
    // offset in std::copy when it will increment destination pointer.
    auto *Ptr =
        reinterpret_cast<typename std::iterator_traits<InputIterator>::pointer>(
            MUserPtr);
    std::copy(First, Last, Ptr);

    // TODO: There is contradiction in the spec, in one place it says
    // the data is not copied back at all if the buffer is construted
    // using this c'tor, another section says that the data will be
    // copied back if iterators passed are not const.
    set_final_data(First);
  }

  template <class InputIterator>
  buffer_impl(EnableIfConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, const property_list &Props,
              AllocatorT Allocator = AllocatorT())
      : MSizeInBytes(SizeInBytes), MProps(Props), MAllocator(Allocator) {

    // TODO: There is contradiction is the spec. It says SYCL RT must not
    // allocate addtional memory on the host if use_host_ptr prop was passed. On
    // the other hand it says that SYCL RT should allocate temporal memory in
    // this c'tor.
    //

    if (0) {
      MUserPtr = MShadowCopy = allocateHostMem();
    } else {
      size_t AllocatorValueSize = sizeof(typename AllocatorT::value_type);
      size_t AllocationSize = get_size() / AllocatorValueSize;
      AllocationSize += (get_size() % AllocatorValueSize) ? 1 : 0;
      MUserPtr = MShadowCopy = MAllocator.allocate(AllocationSize);
    }

    // We need to cast MUserPtr to pointer to the iteration type to get correct
    // offset in std::copy when it will increment destination pointer.
    using value = typename std::iterator_traits<InputIterator>::value_type;
    auto *Ptr = reinterpret_cast<typename std::add_pointer<
        typename std::remove_const<value>::type>::type>(MUserPtr);
    std::copy(First, Last, Ptr);
  }

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent = {})
      : MInteropMemObject(MemObject), MOpenCLInterop(true),
        MSizeInBytes(SizeInBytes),
        MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
        MInteropContext(detail::getSyclObjImpl(SyclContext)) {

    if (MInteropContext->is_host())
      throw cl::sycl::invalid_parameter_error(
          "Creation of interoperability buffer using host context is not "
          "allowed");

    cl_context Context = nullptr;
    CHECK_OCL_CODE(clGetMemObjectInfo(MInteropMemObject, CL_MEM_CONTEXT,
                                      sizeof(Context), &Context, nullptr));
    if (MInteropContext->getHandleRef() != Context)
      throw cl::sycl::invalid_parameter_error(
          "Input context must be the same as the context of cl_mem");
    CHECK_OCL_CODE(clRetainMemObject(MInteropMemObject));
  }

  size_t get_size() const { return MSizeInBytes; }

  void set_write_back(bool flag) { MNeedWriteBack = flag; }

  AllocatorT get_allocator() const { return MAllocator; }

  ~buffer_impl() {
    if (MUploadDataFn != nullptr && MNeedWriteBack) {
      MUploadDataFn();
    }

    Scheduler::getInstance().removeMemoryObject(this);
    releaseHostMem(MShadowCopy);

    if (MOpenCLInterop)
      CHECK_OCL_CODE_NO_EXC(clReleaseMemObject(MInteropMemObject));
  }

  void set_final_data(std::nullptr_t) { MUploadDataFn = nullptr; }

  template <typename T> void set_final_data(weak_ptr_class<T> FinalData) {
    MUploadDataFn = [this, FinalData]() {
      if (auto finalData = FinalData.lock()) {
        void *TempPtr = finalData.get();
        detail::Requirement AccImpl({0, 0, 0}, {MSizeInBytes, 1, 1},
                                    {MSizeInBytes, 1, 1}, access::mode::read,
                                    this, 1, sizeof(char));
        AccImpl.MData = TempPtr;

        detail::EventImplPtr Event =
            Scheduler::getInstance().addCopyBack(&AccImpl);
        if (Event)
          Event->wait(Event);
      }
    };
  }

  template <template <typename T> class C, typename T>
  void set_final_data(
      C<T> FinalData,
      typename std::enable_if<
          std::is_convertible<C<T>, weak_ptr_class<T>>::value>::type * = 0) {
    weak_ptr_class<T> WeakFinalData(FinalData);
    set_final_data(WeakFinalData);
  }

  template <typename Destination>
  void set_final_data(
      Destination FinalData,
      typename std::enable_if<std::is_pointer<Destination>::value>::type * =
          0) {
    static_assert(!std::is_const<Destination>::value,
                  "Сan not write in a constant Destination. Destination should "
                  "not be const.");
    MUploadDataFn = [this, FinalData]() mutable {

      detail::Requirement AccImpl({0, 0, 0}, {MSizeInBytes, 1, 1},
                                  {MSizeInBytes, 1, 1}, access::mode::read,
                                  this, 1, sizeof(char));
      AccImpl.MData = FinalData;

      detail::EventImplPtr Event =
          Scheduler::getInstance().addCopyBack(&AccImpl);
      if (Event)
        Event->wait(Event);
    };
  }

  template <typename Destination>
  void set_final_data(
      Destination FinalData,
      typename std::enable_if<!std::is_pointer<Destination>::value>::type * =
          0) {
    static_assert(!std::is_const<Destination>::value,
                  "Сan not write in a constant Destination. Destination should "
                  "not be const.");
    MUploadDataFn = [this, FinalData]() mutable {
      using FinalDataType =
          typename std::iterator_traits<Destination>::value_type;

      // addCopyBack method expects consecutive memory while iterator
      // passed can point to non consecutive one.
      // Can be optmized if iterator papssed is consecutive one.
      std::vector<FinalDataType> TempBuffer(MSizeInBytes /
                                            sizeof(FinalDataType));
      void *TempPtr = TempBuffer.data();

      detail::Requirement AccImpl({0, 0, 0}, {MSizeInBytes, 1, 1},
                                  {MSizeInBytes, 1, 1}, access::mode::read,
                                  this, 1, sizeof(char));
      AccImpl.MData = TempPtr;

      detail::EventImplPtr Event =
          Scheduler::getInstance().addCopyBack(&AccImpl);
      if (Event) {
        Event->wait(Event);
        std::copy(TempBuffer.begin(), TempBuffer.end(), FinalData);
      }
    };
  }

  template <typename T, int Dimensions, access::mode Mode,
            access::target Target = access::target::global_buffer>
  accessor<T, Dimensions, Mode, Target, access::placeholder::false_t>
  get_access(buffer<T, Dimensions, AllocatorT> &Buffer,
             handler &CommandGroupHandler) {
    return accessor<T, Dimensions, Mode, Target, access::placeholder::false_t>(
        Buffer, CommandGroupHandler);
  }

  template <typename T, int Dimensions, access::mode Mode>
  accessor<T, Dimensions, Mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, Dimensions, AllocatorT> &Buffer) {
    return accessor<T, Dimensions, Mode, access::target::host_buffer,
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

  void *allocateHostMem() override {
    assert(
        !MProps.has_property<property::buffer::use_host_ptr>() &&
        "Cannot allocate additional memory if use_host_ptr property is set.");
    size_t AllocatorValueSize = sizeof(typename AllocatorT::value_type);
    size_t AllocationSize = get_size() / AllocatorValueSize;
    AllocationSize += (get_size() % AllocatorValueSize) ? 1 : 0;
    return MAllocator.allocate(AllocationSize);
  }

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    cl_event &OutEventToWait) override {

    void *UserPtr = InitFromUserData ? getUserPtr() : nullptr;

    return MemoryManager::allocateMemBuffer(
        std::move(Context), this, UserPtr, MHostPtrReadOnly, get_size(),
        MInteropEvent, MInteropContext, OutEventToWait);
  }

  MemObjType getType() const override { return MemObjType::BUFFER; }

  void releaseHostMem(void *Ptr) override {
    MAllocator.deallocate((typename AllocatorT::pointer)Ptr, get_size());
  }

  void releaseMem(ContextImplPtr Context, void *MemAllocation) override {
    return MemoryManager::releaseMemBuf(Context, this, MemAllocation,
                                        getUserPtr());
  }

  void *getUserPtr() const {
    return MOpenCLInterop ? (void *)MInteropMemObject : MUserPtr;
  }

  template <typename propertyT> bool has_property() const {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

private:
  bool MOpenCLInterop = false;
  bool MHostPtrReadOnly = false;

  bool MNeedWriteBack = true;

  EventImplPtr MInteropEvent;
  ContextImplPtr MInteropContext;
  cl_mem MInteropMemObject = nullptr;

  void *MUserPtr = nullptr;
  void *MShadowCopy = nullptr;
  size_t MSizeInBytes = 0;

  property_list MProps;
  std::function<void(void)> MUploadDataFn = nullptr;
  AllocatorT MAllocator;
};

} // namespace detail
} // namespace sycl
} // namespace cl
