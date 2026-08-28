//===-- LanguageRuntime.h - Kernel language runtime API declarations ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_RUNTIME_H
#define LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_RUNTIME_H

#include "LanguageLaunch.h"
#include "Types.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "LanguageErrors.h"

struct DeviceProp_t {
  char name[256];
  size_t totalGlobalMem;
  int warpSize;
  int multiProcessorCount;
  int major;
  int minor;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int memoryBusWidth;
};

enum MemcpyKind {
  MemcpyHostToHost = 0,
  MemcpyHostToDevice = 1,
  MemcpyDeviceToHost = 2,
  MemcpyDeviceToDevice = 3,
  MemcpyDefault = 4
};

/// Flags passed to HostAlloc
enum : unsigned int {
  HostAllocDefault = 0x00,
  HostAllocPortable = 0x01,
  HostAllocMapped = 0x02,
  HostAllocWriteCombined = 0x04,
};

/// Flags passed to StreamCreateWithFlags
enum : unsigned int {
  StreamDefault = 0x00,
  StreamNonBlocking = 0x01,
};

typedef struct Stream_st *Stream_t;

/// Malloc, with type template overlay.
///{
Error_t Malloc(void **Dev_Ptr, size_t Size);

template <class T> static inline Error_t Malloc(T **dev_Ptr, size_t Size) {
  return ::Malloc((void **)dev_Ptr, Size);
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags);

template <class T>
static inline Error_t HostAlloc(T **Ptr, size_t Size, unsigned int Flags) {
  return ::HostAlloc((void **)Ptr, Size, Flags);
}

Error_t MallocHost(void **Ptr, size_t Size);

template <class T> static inline Error_t MallocHost(T **Ptr, size_t Size) {
  return ::MallocHost((void **)Ptr, Size);
}
///}

/// Free, no type template necessary.
Error_t Free(void *Dev_Ptr);

/// Memcpy, with type template overlay.
///{
Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind);

template <class T>
static inline Error_t Memcpy(T *Dst, const T *Src, size_t Size,
                             MemcpyKind Kind) {
  return ::Memcpy((void *)Dst, (const void *)Src, Size, Kind);
}
///}

Error_t DeviceSynchronize();

Error_t GetDevice(int *DeviceNo);

Error_t GetDeviceCount(int *Count);

Error_t SetDevice(int DeviceNo);

Error_t FreeHost(void *Ptr);

Error_t GetDeviceProperties(DeviceProp_t *DeviceProp, int DeviceNo);

Error_t StreamCreate(Stream_t *stream);

Error_t StreamCreateWithFlags(Stream_t *stream, unsigned int flags);

Error_t StreamDestroy(Stream_t stream);

Error_t StreamSynchronize(Stream_t stream);

#if defined(__AMDGPU__) || defined(__NVPTX__)
#include <gpuintrin.h>

/// Define \p FIELD as a property backed by component \p OFFSET of Vec.
#define __LLVM_OFFLOAD_DEVICE_BUILTIN(FIELD, OFFSET)                           \
  __declspec(property(get = __get_##FIELD,                                     \
                      put = __put_##FIELD)) unsigned int FIELD;                \
  __device__ inline __attribute__((always_inline)) T __get_##FIELD(void)       \
      const {                                                                  \
    return Vec[OFFSET];                                                        \
  }                                                                            \
  __device__ inline __attribute__((always_inline)) T __put_##FIELD(T V) {      \
    return Vec[OFFSET] = V;                                                    \
  }

/// Common storage for CUDA/HIP vector aliases such as int4 and float3.
///
/// Provides array-style indexing and x/y/z/w component properties over Clang
/// ext_vector_type storage.
template <class T, int Size> struct BaseVector {
  using VT = float __attribute__((ext_vector_type(Size)));
  VT Vec;

  __device__ __host__ BaseVector() = default;

  /// Construct a vector from component values.
  template <typename... Args>
  __device__ __host__ BaseVector(Args... args) : BaseVector({args...}) {}

  /// Return component \p Idx.
  __device__ __host__ T &operator[](int Idx) { return Vec[Idx]; }
  __device__ __host__ const T &operator[](int Idx) const { return Vec[Idx]; }

  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, 0);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, 1);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, 2);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(w, 3);
};

/// Define the vector alias TY##SIZE and its make_TY##SIZE constructor helper.
#define __VECTOR_DEF_IMPL(TY, SIZE)                                            \
  using TY##SIZE = BaseVector<TY, SIZE>;                                       \
                                                                               \
  template <typename... Args>                                                  \
  __device__ __host__ TY##SIZE make_##TY##SIZE(Args... args) {                 \
    return TY##SIZE(args...);                                                  \
  }

/// Define the standard 1/2/3/4/8/16 element vector aliases for \p TY.
#define __VECTOR_DEF(TY)                                                       \
  __VECTOR_DEF_IMPL(TY, 1)                                                     \
  __VECTOR_DEF_IMPL(TY, 2)                                                     \
  __VECTOR_DEF_IMPL(TY, 3)                                                     \
  __VECTOR_DEF_IMPL(TY, 4)                                                     \
  __VECTOR_DEF_IMPL(TY, 8)                                                     \
  __VECTOR_DEF_IMPL(TY, 16)

/// Instantiate CUDA/HIP-style vector types and make_* helpers.
__VECTOR_DEF(float)
__VECTOR_DEF(double)
__VECTOR_DEF(int8_t)
__VECTOR_DEF(int16_t)
__VECTOR_DEF(int32_t)
__VECTOR_DEF(int64_t)
__VECTOR_DEF(uint8_t)
__VECTOR_DEF(uint16_t)
__VECTOR_DEF(uint32_t)
__VECTOR_DEF(uint64_t)
__VECTOR_DEF(char)
__VECTOR_DEF(short)
__VECTOR_DEF(int)
__VECTOR_DEF(unsigned)
__VECTOR_DEF(long)

#undef __VECTOR_DEF_IMPL
#undef __VECTOR_DEF
#undef __LLVM_OFFLOAD_DEVICE_BUILTIN

#endif

#endif // LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_RUNTIME_H
