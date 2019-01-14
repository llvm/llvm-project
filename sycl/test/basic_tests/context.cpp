// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out

//==--------------- context.cpp - SYCL context test ------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  try {
    context c;
  } catch (device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  auto devices = device::get_devices();
  device &deviceA = devices[0];
  device &deviceB = (devices.size() > 1 ? devices[1] : devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    context Context(deviceA);
    size_t hash = hash_class<context>()(Context);
    context MovedContext(std::move(Context));
    assert(hash == hash_class<context>()(MovedContext));
    assert(deviceA.is_host() == MovedContext.is_host());
    if (!deviceA.is_host()) {
      assert(MovedContext.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    context Context(deviceA);
    size_t hash = hash_class<context>()(Context);
    context WillMovedContext(deviceB);
    WillMovedContext = std::move(Context);
    assert(hash == hash_class<context>()(WillMovedContext));
    assert(deviceA.is_host() == WillMovedContext.is_host());
    if (!deviceA.is_host()) {
      assert(WillMovedContext.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    context Context(deviceA);
    size_t hash = hash_class<context>()(Context);
    context ContextCopy(Context);
    assert(hash == hash_class<context>()(Context));
    assert(hash == hash_class<context>()(ContextCopy));
    assert(Context == ContextCopy);
    assert(Context.is_host() == ContextCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    context Context(deviceA);
    size_t hash = hash_class<context>()(Context);
    context WillContextCopy(deviceB);
    WillContextCopy = Context;
    assert(hash == hash_class<context>()(Context));
    assert(hash == hash_class<context>()(WillContextCopy));
    assert(Context == WillContextCopy);
    assert(Context.is_host() == WillContextCopy.is_host());
  }
}
