// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out
//==--------------- platform.cpp - SYCL platform test ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <typeinfo>

using namespace cl::sycl;

int main() {
  int i = 1;
  vector_class<platform> openclPlatforms;
  for (const auto &plt : platform::get_platforms()) {
    std::cout << "Platform " << i++
              << " is available: " << ((plt.is_host()) ? "host: " : "OpenCL: ")
              << std::hex << ((plt.is_host()) ? nullptr : plt.get())
              << std::endl;
  }

  auto platforms = platform::get_platforms();
  platform &platformA = platforms[0];
  platform &platformB = (platforms.size() > 1 ? platforms[1] : platforms[0]);
  {
    std::cout << "move constructor" << std::endl;
    platform Platform(platformA);
    size_t hash = hash_class<platform>()(Platform);
    platform MovedPlatform(std::move(Platform));
    assert(hash == hash_class<platform>()(MovedPlatform));
    assert(platformA.is_host() == MovedPlatform.is_host());
    if (!platformA.is_host()) {
      assert(MovedPlatform.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    platform Platform(platformA);
    size_t hash = hash_class<platform>()(Platform);
    platform WillMovedPlatform(platformB);
    WillMovedPlatform = std::move(Platform);
    assert(hash == hash_class<platform>()(WillMovedPlatform));
    assert(platformA.is_host() == WillMovedPlatform.is_host());
    if (!platformA.is_host()) {
      assert(WillMovedPlatform.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    platform Platform(platformA);
    size_t hash = hash_class<platform>()(Platform);
    platform PlatformCopy(Platform);
    assert(hash == hash_class<platform>()(Platform));
    assert(hash == hash_class<platform>()(PlatformCopy));
    assert(Platform == PlatformCopy);
    assert(Platform.is_host() == PlatformCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    platform Platform(platformA);
    size_t hash = hash_class<platform>()(Platform);
    platform WillPlatformCopy(platformB);
    WillPlatformCopy = Platform;
    assert(hash == hash_class<platform>()(Platform));
    assert(hash == hash_class<platform>()(WillPlatformCopy));
    assert(Platform == WillPlatformCopy);
    assert(Platform.is_host() == WillPlatformCopy.is_host());
  }
}
