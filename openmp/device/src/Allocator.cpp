//===------ State.cpp - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Shared/Environment.h"

#include "Allocator.h"
#include "Configuration.h"
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Mapping.h"
#include "Synchronization.h"
#include "Platform.h"

using namespace ompx;
using namespace allocator;

// Provide a default implementation of malloc / free for AMDGPU platforms built
// without 'libc' support.
extern "C" {

[[gnu::noinline]] uint64_t __asan_malloc_impl(uint64_t bufsz, uint64_t pc);
[[gnu::noinline]] void __asan_free_impl(uint64_t ptr, uint64_t pc);
[[gnu::noinline]] uint64_t __ockl_dm_alloc(uint64_t bufsz);
[[gnu::noinline]] void __ockl_dm_dealloc(uint64_t ptr);

#ifdef __AMDGPU__
[[gnu::noinline]] void *__alt_libc_malloc(size_t sz);
[[gnu::noinline]] void __alt_libc_free(void *ptr);

[[gnu::noinline]] uint64_t __ockl_devmem_request(uint64_t addr, uint64_t size) {
  if (size) { // allocation request
    [[clang::noinline]] return (uint64_t)__alt_libc_malloc((size_t)size);
  } else { // free request
    [[clang::noinline]] __alt_libc_free((void *)addr);
    return 0;
  }
}
#endif

#if defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
[[gnu::weak]] void *malloc(size_t Size) { return allocator::alloc(Size); }
[[gnu::weak]] void free(void *Ptr) { allocator::free(Ptr); }
#else
[[gnu::leaf]] void *malloc(size_t Size);
[[gnu::leaf]] void free(void *Ptr);
#endif
}

static constexpr uint64_t MEMORY_SIZE = /* 1 MiB */ 1024 * 1024;
alignas(ALIGNMENT) static uint8_t Memory[MEMORY_SIZE] = {0};

// Fallback bump pointer interface for platforms without a functioning
// allocator.
struct BumpAllocatorTy final {
  uint64_t Offset = 0;

  void *alloc(uint64_t Size) {
    Size = utils::roundUp(Size, uint64_t(allocator::ALIGNMENT));

    uint64_t OldData = atomic::add(&Offset, Size, atomic::seq_cst);
    if (OldData + Size >= MEMORY_SIZE)
      __builtin_trap();

    return &Memory[OldData];
  }

  void free(void *) {}
};

BumpAllocatorTy BumpAllocator;

/// allocator namespace implementation
///
///{

void *allocator::alloc(uint64_t Size) {
#if defined(__AMDGPU__) && defined(SANITIZER_AMDGPU)
  return reinterpret_cast<void *>(
      __asan_malloc_impl(Size, uint64_t(__builtin_return_address(0))));
#elif defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
  return reinterpret_cast<void *>(__ockl_dm_alloc(Size));
#else
  return ::malloc(Size);
#endif
}

void allocator::free(void *Ptr) {
#if defined(__AMDGPU__) && defined(SANITIZER_AMDGPU)
  __asan_free_impl(reinterpret_cast<uint64_t>(Ptr),
                   uint64_t(__builtin_return_address(0)));
#elif defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
  __ockl_dm_dealloc(reinterpret_cast<uint64_t>(Ptr));
#else
  ::free(Ptr);
#endif
}

///}
