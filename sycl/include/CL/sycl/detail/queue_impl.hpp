//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>

namespace cl {
namespace sycl {
namespace detail {

class queue_impl {
public:
  queue_impl(const device &SyclDevice, async_handler AsyncHandler,
             const property_list &PropList)
      : m_Device(SyclDevice), m_Context(m_Device), m_AsyncHandler(AsyncHandler),
        m_PropList(PropList), m_HostQueue(m_Device.is_host()) {
    m_OpenCLInterop = !m_HostQueue;
    if (!m_HostQueue) {
      cl_command_queue_properties CreationFlags =
          CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

      if (m_PropList.has_property<property::queue::enable_profiling>()) {
        CreationFlags |= CL_QUEUE_PROFILING_ENABLE;
      }

      cl_int Error = CL_SUCCESS;
#ifdef CL_VERSION_2_0
      vector_class<cl_queue_properties> CreationFlagProperties = {
          CL_QUEUE_PROPERTIES, CreationFlags, 0};
      m_CommandQueue = clCreateCommandQueueWithProperties(
          m_Context.get(), m_Device.get(), CreationFlagProperties.data(),
          &Error);
#else
      m_CommandQueue = clCreateCommandQueue(m_Context.get(), m_Device.get(),
                                            CreationFlags, &Error);
#endif
      CHECK_OCL_CODE(Error);
      // TODO catch an exception and put it to list of asynchronous exceptions
    }
  }

  queue_impl(cl_command_queue CLQueue, const context &SyclContext,
             const async_handler &AsyncHandler)
      : m_Context(SyclContext), m_AsyncHandler(AsyncHandler),
        m_CommandQueue(CLQueue), m_OpenCLInterop(true), m_HostQueue(false) {

    cl_device_id CLDevice = nullptr;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetCommandQueueInfo(m_CommandQueue, CL_QUEUE_DEVICE,
                                         sizeof(CLDevice), &CLDevice, nullptr));
    m_Device = device(CLDevice);
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clRetainCommandQueue(m_CommandQueue));
  }

  ~queue_impl() {
    if (m_OpenCLInterop) {
      CHECK_OCL_CODE_NO_EXC(clReleaseCommandQueue(m_CommandQueue));
    }
  }

  cl_command_queue get() {
    if (m_OpenCLInterop) {
      CHECK_OCL_CODE(clRetainCommandQueue(m_CommandQueue));
      return m_CommandQueue;
    }
    throw invalid_object_error(
        "This instance of queue doesn't support OpenCL interoperability");
  }

  context get_context() const { return m_Context; }

  device get_device() const { return m_Device; }

  bool is_host() const { return m_HostQueue; }

  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

  template <typename T> event submit(T cgf, std::shared_ptr<queue_impl> self,
                                     std::shared_ptr<queue_impl> second_queue) {
    event Event;
    try {
      Event = submit_impl(cgf, self);
    } catch (...) {
      m_Exceptions.push_back(std::current_exception());
      Event = second_queue->submit(cgf, second_queue);
    }
    return Event;
  }

  template <typename T> event submit(T cgf, std::shared_ptr<queue_impl> self) {
    event Event;
    try {
      Event = submit_impl(cgf, self);
    } catch(...) {
      m_Exceptions.push_back(std::current_exception());
    }
    return Event;
  }

  void wait() {
    // TODO: Make thread safe.
    for (auto &evnt : m_Events)
      evnt.wait();
    m_Events.clear();
  }

  exception_list getExceptionList() const { return m_Exceptions; }

  void wait_and_throw() {
    wait();
    throw_asynchronous();
  }

  void throw_asynchronous() {
    if (m_AsyncHandler && m_Exceptions.size()) {
      m_AsyncHandler(m_Exceptions);
    }
    m_Exceptions.clear();
  }

  cl_command_queue &getHandleRef() { return m_CommandQueue; }

  template <typename propertyT> bool has_property() const {
    return m_PropList.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return m_PropList.get_property<propertyT>();
  }

private:
  template <typename T>
  event submit_impl(T cgf, std::shared_ptr<queue_impl> self) {
    handler Handler(std::move(self), m_HostQueue);
    cgf(Handler);
    event Event = Handler.finalize();
    // TODO: Make thread safe.
    m_Events.push_back(Event);
    return Event;
  }

  device m_Device;
  context m_Context;
  vector_class<event> m_Events;
  exception_list m_Exceptions;
  async_handler m_AsyncHandler;
  property_list m_PropList;

  cl_command_queue m_CommandQueue = nullptr;
  bool m_OpenCLInterop = false;
  bool m_HostQueue = false;
};

} // namespace detail
} // namespace sycl
} // namespace cl
