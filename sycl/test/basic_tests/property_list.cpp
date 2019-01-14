// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
//
// CHECK: PASSED
//==--------------- property_list.cpp - SYCL property list test ------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

namespace sycl_property = cl::sycl::property;

int main() {
  bool Failed = false;

  {
    cl::sycl::property_list Empty{};
    if (Empty.has_property<sycl_property::buffer::use_host_ptr>()) {
      std::cerr << "Error: empty property list has property." << std::endl;
      Failed = true;
    }
  }

  {
    cl::sycl::context SYCLContext;
    sycl_property::buffer::context_bound ContextBound(SYCLContext);

    cl::sycl::property_list SeveralProps{sycl_property::image::use_host_ptr(),
                                         sycl_property::buffer::use_host_ptr(),
                                         sycl_property::image::use_host_ptr(),
                                         ContextBound};

    if (!SeveralProps.has_property<sycl_property::buffer::use_host_ptr>()) {
      std::cerr << "Error: property list has no property while should have."
                << std::endl;
      Failed = true;
    }

    if (!SeveralProps.has_property<sycl_property::image::use_host_ptr>()) {
      std::cerr << "Error: property list has no property while should have."
                << std::endl;
      Failed = true;
    }

    try {
      sycl_property::buffer::context_bound ContextBoundRet =
          SeveralProps.get_property<sycl_property::buffer::context_bound>();
      if (SYCLContext != ContextBoundRet.get_context()) {
        std::cerr << "Error: returned SYCL context is not the same that was "
                     "passed to c'tor earlier."
                  << std::endl;
        Failed = true;
      }

    } catch (cl::sycl::invalid_object_error &Error) {
      Error.what();
      std::cerr << "Error: exception was thrown in get_property method."
                << std::endl;
      Failed = true;
    }
  }

  std::cerr << "Test status : " << (Failed ? "FAILED" : "PASSED") << std::endl;

  return Failed;
}
