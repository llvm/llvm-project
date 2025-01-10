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

#pragma omp begin declare target device_type(nohost)

namespace ompx {
namespace impl {

OMP_ATTRS double getWTick();

OMP_ATTRS double getWTime();

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

OMP_ATTRS double getWTick() {
  // The number of ticks per second for the AMDGPU clock varies by card and can
  // only be retrived by querying the driver. We rely on the device environment
  // to inform us what the proper frequency is.
  return 1.0 / config::getClockFrequency();
}

OMP_ATTRS double getWTime() {
  return static_cast<double>(__builtin_readsteadycounter()) * getWTick();
}

#pragma omp end declare variant

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

OMP_ATTRS double getWTick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

OMP_ATTRS double getWTime() {
  uint64_t nsecs = __nvvm_read_ptx_sreg_globaltimer();
  return static_cast<double>(nsecs) * getWTick();
}

#pragma omp end declare variant

/// Lookup a device-side function using a host pointer /p HstPtr using the table
/// provided by the device plugin. The table is an ordered pair of host and
/// device pointers sorted on the value of the host pointer.
OMP_ATTRS void *indirectCallLookup(void *HstPtr) {
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
/// FIXME: This is marked as 'retain' so that it is not removed via
/// `-mlink-builtin-bitcode`
#ifdef __NVPTX__
[[gnu::visibility("protected"), gnu::weak,
  gnu::retain]] rpc::Client Client asm("__llvm_rpc_client");
#else
[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client Client asm("__llvm_rpc_client");
#endif

} // namespace impl
} // namespace ompx

/// Interfaces
///
///{

extern "C" {
OMP_ATTRS int32_t __kmpc_cancellationpoint(IdentTy *, int32_t, int32_t) {
  return 0;
}

OMP_ATTRS int32_t __kmpc_cancel(IdentTy *, int32_t, int32_t) { return 0; }

OMP_ATTRS double omp_get_wtick(void) { return ompx::impl::getWTick(); }

OMP_ATTRS double omp_get_wtime(void) { return ompx::impl::getWTime(); }

OMP_ATTRS void *__llvm_omp_indirect_call_lookup(void *HstPtr) {
  return ompx::impl::indirectCallLookup(HstPtr);
}

OMP_ATTRS void *omp_alloc(size_t size, omp_allocator_handle_t allocator) {
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

OMP_ATTRS void omp_free(void *ptr, omp_allocator_handle_t allocator) {
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

OMP_ATTRS unsigned long long __llvm_omp_host_call(void *fn, void *data,
                                                  size_t size) {
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
}

///}
#pragma omp end declare target
