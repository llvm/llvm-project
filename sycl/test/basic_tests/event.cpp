// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
//==--------------- event.cpp - SYCL event test ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

int main() {
  try {
    std::cout << "Create default event" << std::endl;
    cl::sycl::event e;
  } catch (cl::sycl::device_error e) {
    std::cout << "Failed to create device for event" << std::endl;
  }
  try {
    std::cout << "Try create OpenCL event" << std::endl;
    cl::sycl::context c;
    if (!c.is_host()) {
      ::cl_int error;
      cl_event u_e = clCreateUserEvent(c.get(), &error);
      cl::sycl::event cl_e(u_e, c);
      std::cout << "OpenCL event: " << std::hex << cl_e.get()
                << ((cl_e.get() == u_e) ? " matches " : " does not match ")
                << u_e << std::endl;

    } else {
      std::cout << "Failed to create OpenCL context" << std::endl;
    }
  } catch (cl::sycl::device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  {
    std::cout << "move constructor" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event MovedEvent(std::move(Event));
    assert(hash == std::hash<cl::sycl::event>()(MovedEvent));
  }

  {
    std::cout << "move assignment operator" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event WillMovedEvent;
    WillMovedEvent = std::move(Event);
    assert(hash == std::hash<cl::sycl::event>()(WillMovedEvent));
  }

  {
    std::cout << "copy constructor" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event EventCopy(Event);
    assert(hash == std::hash<cl::sycl::event>()(Event));
    assert(hash == std::hash<cl::sycl::event>()(EventCopy));
    assert(Event == EventCopy);
  }

  {
    std::cout << "copy assignment operator" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event WillEventCopy;
    WillEventCopy = Event;
    assert(hash == std::hash<cl::sycl::event>()(Event));
    assert(hash == std::hash<cl::sycl::event>()(WillEventCopy));
    assert(Event == WillEventCopy);
  }
}
