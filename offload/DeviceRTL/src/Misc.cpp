//===--------- Misc.cpp - OpenMP device misc interfaces ----------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Allocator.h"
#include "Configuration.h"
#include "DeviceTypes.h"
#include "Shared/RPCOpcodes.h"
#include "shared/rpc.h"

#include "Debug.h"

namespace ompx {
namespace impl {

/// AMDGCN Implementation
///
///{
#ifdef __AMDGPU__

double getWTick() {
  // The number of ticks per second for the AMDGPU clock varies by card and can
  // only be retrieved by querying the driver. We rely on the device environment
  // to inform us what the proper frequency is.
  return 1.0 / config::getClockFrequency();
}

double getWTime() {
  return static_cast<double>(__builtin_readsteadycounter()) * getWTick();
}

#endif

/// NVPTX Implementation
///
///{
#ifdef __NVPTX__

double getWTick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

double getWTime() {
  uint64_t nsecs = __nvvm_read_ptx_sreg_globaltimer();
  return static_cast<double>(nsecs) * getWTick();
}

#endif

/// Lookup a device-side function using a host pointer /p HstPtr using the table
/// provided by the device plugin. The table is an ordered pair of host and
/// device pointers sorted on the value of the host pointer.
void *indirectCallLookup(void *HstPtr) {
  if (!HstPtr)
    return nullptr;

  struct IndirectCallTable {
    void *HstPtr;
    void *DevPtr;
  };
  IndirectCallTable *Table =
      reinterpret_cast<IndirectCallTable *>(config::getIndirectCallTablePtr());
  uint64_t TableSize = config::getIndirectCallTableSize();

  // If the table is empty we assume this is device pointer.
  if (!Table || !TableSize)
    return HstPtr;

  uint32_t Left = 0;
  uint32_t Right = TableSize;

  // If the pointer is definitely not contained in the table we exit early.
  if (HstPtr < Table[Left].HstPtr || HstPtr > Table[Right - 1].HstPtr)
    return HstPtr;

  while (Left != Right) {
    uint32_t Current = Left + (Right - Left) / 2;
    if (Table[Current].HstPtr == HstPtr)
      return Table[Current].DevPtr;

    if (HstPtr < Table[Current].HstPtr)
      Right = Current;
    else
      Left = Current;
  }

  // If we searched the whole table and found nothing this is a device pointer.
  return HstPtr;
}

/// The openmp client instance used to communicate with the server.
[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client Client asm("__llvm_rpc_client");

} // namespace impl
} // namespace ompx

/// Interfaces
///
///{

extern "C" {
int32_t __kmpc_cancellationpoint(IdentTy *, int32_t, int32_t) { return 0; }

int32_t __kmpc_cancel(IdentTy *, int32_t, int32_t) { return 0; }

double omp_get_wtick(void) { return ompx::impl::getWTick(); }

double omp_get_wtime(void) { return ompx::impl::getWTime(); }

void *__llvm_omp_indirect_call_lookup(void *HstPtr) {
  return ompx::impl::indirectCallLookup(HstPtr);
}

void *omp_alloc(size_t size, omp_allocator_handle_t allocator) {
  switch (allocator) {
  case omp_default_mem_alloc:
  case omp_large_cap_mem_alloc:
  case omp_const_mem_alloc:
  case omp_high_bw_mem_alloc:
  case omp_low_lat_mem_alloc:
    return malloc(size);
  default:
    return nullptr;
  }
}

void omp_free(void *ptr, omp_allocator_handle_t allocator) {
  switch (allocator) {
  case omp_default_mem_alloc:
  case omp_large_cap_mem_alloc:
  case omp_const_mem_alloc:
  case omp_high_bw_mem_alloc:
  case omp_low_lat_mem_alloc:
    free(ptr);
  case omp_null_allocator:
  default:
    return;
  }
}

unsigned long long __llvm_omp_host_call(void *fn, void *data, size_t size) {
  rpc::Client::Port Port = ompx::impl::Client.open<OFFLOAD_HOST_CALL>();
  Port.send_n(data, size);
  Port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(fn);
  });
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
  Port.close();
  return Ret;
}

// Calls to __alt_libc_malloc and __alt_libc_free are
// made by _ockl_devmem_request
__attribute__((noinline)) void *__alt_libc_malloc(size_t sz) {
  void *ptr = nullptr;
  rpc::Client::Port Port = ompx::impl::Client.open<ALT_LIBC_MALLOC>();
  Port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = (uint64_t)sz; },
      [&](rpc::Buffer *buffer, uint32_t) {
        ptr = reinterpret_cast<void *>(buffer->data[0]);
      });
  Port.close();
  return ptr;
}
__attribute__((noinline)) void __alt_libc_free(void *ptr) {
  unsigned long long Ret;
  rpc::Client::Port Port = ompx::impl::Client.open<ALT_LIBC_FREE>();
  Port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = (uint64_t)ptr;
  });
  Port.close();
  return;
}
// Calls to __llvm_omp_emissary_rpc and __llvm_omp_emissary_premalloc are
// generated by device codegen when calls to the vargs function _emissary_exec
// ae encountered. See clang/lib/CodeGen/CGEmitEmissaryExec.cpp
__attribute__((noinline)) void *__llvm_omp_emissary_premalloc64(size_t sz) {
  void *ptr = nullptr;
  rpc::Client::Port Port = ompx::impl::Client.open<EMISSARY_PREMALLOC>();
  Port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = (uint64_t)sz; },
      [&](rpc::Buffer *buffer, uint32_t) {
        ptr = reinterpret_cast<void *>(buffer->data[0]);
      });
  Port.close();
  return ptr;
}
void *__llvm_omp_emissary_premalloc(uint32_t sz32) {
  return __llvm_omp_emissary_premalloc64((size_t)sz32);
}
__attribute__((noinline)) void __llvm_omp_emissary_free(void *ptr) {
  unsigned long long Ret;
  rpc::Client::Port Port = ompx::impl::Client.open<EMISSARY_FREE>();
  Port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = (uint64_t)ptr;
  });
  Port.close();
  return;
}
__attribute__((noinline)) unsigned long long
__llvm_omp_emissary_rpc(void* fn, void *data) {
  rpc::Client::Port Port = ompx::impl::Client.open<OFFLOAD_EMISSARY>();
  Port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(data);
  });
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
  Port.close();
  return Ret;
}
}

///}
