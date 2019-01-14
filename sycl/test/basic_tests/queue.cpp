// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
//==--------------- queue.cpp - SYCL queue test ----------------------------==//
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

string_class get_type(const device &dev) {
  return ((dev.is_host()) ? "host"
                          : (dev.is_gpu() ? "OpenCL.GPU" : "OpenCL.CPU"));
}

void print_queue_info(const queue &q) {
  std::cout << "ID=" << std::hex
            << ((q.get_device().is_host()) ? nullptr : q.get()) << std::endl;
  std::cout << "queue wraps " << get_type(q.get_device()) << " device"
            << std::endl;
}
int main() {
  try {
    std::cout << "Create default queue." << std::endl;
    queue q;
    print_queue_info(q);

  } catch (device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  auto devices = device::get_devices();
  device &deviceA = devices[0];
  device &deviceB = (devices.size() > 1 ? devices[1] : devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    queue Queue(deviceA);
    size_t hash = hash_class<queue>()(Queue);
    queue MovedQueue(std::move(Queue));
    assert(hash == hash_class<queue>()(MovedQueue));
    assert(deviceA.is_host() == MovedQueue.is_host());
    if (!deviceA.is_host()) {
      assert(MovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    queue Queue(deviceA);
    size_t hash = hash_class<queue>()(Queue);
    queue WillMovedQueue(deviceB);
    WillMovedQueue = std::move(Queue);
    assert(hash == hash_class<queue>()(WillMovedQueue));
    assert(deviceA.is_host() == WillMovedQueue.is_host());
    if (!deviceA.is_host()) {
      assert(WillMovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    queue Queue(deviceA);
    size_t hash = hash_class<queue>()(Queue);
    queue QueueCopy(Queue);
    assert(hash == hash_class<queue>()(Queue));
    assert(hash == hash_class<queue>()(QueueCopy));
    assert(Queue == QueueCopy);
    assert(Queue.is_host() == QueueCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    queue Queue(deviceA);
    size_t hash = hash_class<queue>()(Queue);
    queue WillQueueCopy(deviceB);
    WillQueueCopy = Queue;
    assert(hash == hash_class<queue>()(Queue));
    assert(hash == hash_class<queue>()(WillQueueCopy));
    assert(Queue == WillQueueCopy);
    assert(Queue.is_host() == WillQueueCopy.is_host());
  }

  {
    property_list pl = {};
    queue Queue(pl);
    try {
      Queue.throw_asynchronous();
    }
    catch (const std::bad_function_call& e) {
      std::cout << "Default asynchronous handler call failed: " << e.what() << std::endl;
      throw;
    }
  }
}
