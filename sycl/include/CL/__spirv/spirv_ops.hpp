//==---------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_types.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cl {
namespace __spirv {

#ifdef __SYCL_DEVICE_ONLY__

template <typename dataT>
extern OpTypeEvent *
OpGroupAsyncCopy(int32_t Scope, __local dataT *Dest, __global dataT *Src,
                 size_t NumElements, size_t Stride, OpTypeEvent *E) noexcept;

template <typename dataT>
extern OpTypeEvent *
OpGroupAsyncCopy(int32_t Scope, __global dataT *Dest, __local dataT *Src,
                 size_t NumElements, size_t Stride, OpTypeEvent *E) noexcept;

#define OpGroupAsyncCopyGlobalToLocal OpGroupAsyncCopy
#define OpGroupAsyncCopyLocalToGlobal OpGroupAsyncCopy

// Atomic SPIR-V builtins
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern Type OpAtomicLoad(AS const Type *P, Scope S, MemorySemantics O);
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern void OpAtomicStore(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern Type OpAtomicExchange(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern Type OpAtomicCompareExchange(AS Type *P, Scope S, MemorySemantics E,  \
                                      MemorySemantics U, Type V, Type C);
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern Type OpAtomicIAdd(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern Type OpAtomicISub(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern Type OpAtomicSMin(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern Type OpAtomicUMin(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern Type OpAtomicSMax(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern Type OpAtomicUMax(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern Type OpAtomicAnd(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern Type OpAtomicOr(AS Type *P, Scope S, MemorySemantics O, Type V);
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern Type OpAtomicXor(AS Type *P, Scope S, MemorySemantics O, Type V);

#define __SPIRV_ATOMIC_FLOAT(AS, Type)                                         \
  __SPIRV_ATOMIC_LOAD(AS, Type)                                                \
  __SPIRV_ATOMIC_STORE(AS, Type)                                               \
  __SPIRV_ATOMIC_EXCHANGE(AS, Type)

#define __SPIRV_ATOMIC_BASE(AS, Type)                                          \
  __SPIRV_ATOMIC_FLOAT(AS, Type)                                               \
  __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                        \
  __SPIRV_ATOMIC_IADD(AS, Type)                                                \
  __SPIRV_ATOMIC_ISUB(AS, Type)                                                \
  __SPIRV_ATOMIC_AND(AS, Type)                                                 \
  __SPIRV_ATOMIC_OR(AS, Type)                                                  \
  __SPIRV_ATOMIC_XOR(AS, Type)

#define __SPIRV_ATOMIC_SIGNED(AS, Type)                                        \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_SMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_SMAX(AS, Type)

#define __SPIRV_ATOMIC_UNSIGNED(AS, Type)                                      \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_UMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_UMAX(AS, Type)

// Helper atomic operations which select correct signed/unsigned version
// of atomic min/max based on the signed-ness of the type
#define __SPIRV_ATOMIC_MINMAX(AS, Op)                                          \
  template <typename T>                                                        \
  typename std::enable_if<std::is_signed<T>::value, T>::type OpAtomic##Op(     \
      AS T *Ptr, Scope Scope, MemorySemantics Semantics, T Value) {            \
    return OpAtomicS##Op(Ptr, Scope, Semantics, Value);                        \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<!std::is_signed<T>::value, T>::type OpAtomic##Op(    \
      AS T *Ptr, Scope Scope, MemorySemantics Semantics, T Value) {            \
    return OpAtomicU##Op(Ptr, Scope, Semantics, Value);                        \
  }

#define __SPIRV_ATOMICS(macro, Arg) macro(__global, Arg) macro(__local, Arg)

__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, float)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Min)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Max)

extern bool OpGroupAll(int32_t Scope, bool Predicate) noexcept;

extern bool OpGroupAny(int32_t Scope, bool Predicate) noexcept;

template <typename dataT>
extern dataT OpGroupBroadcast(int32_t Scope, dataT Value,
                              uint32_t LocalId) noexcept;

template <typename dataT>
extern dataT OpGroupIAdd(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupFAdd(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupUMin(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupSMin(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupFMin(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupUMax(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupSMax(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpGroupFMax(int32_t Scope, int32_t Op, dataT Value) noexcept;
template <typename dataT>
extern dataT OpSubgroupShuffleINTEL(dataT Data, uint32_t InvocationId) noexcept;
template <typename dataT>
extern dataT OpSubgroupShuffleDownINTEL(dataT Current, dataT Next,
                                        uint32_t Delta) noexcept;
template <typename dataT>
extern dataT OpSubgroupShuffleUpINTEL(dataT Previous, dataT Current,
                                      uint32_t Delta) noexcept;
template <typename dataT>
extern dataT OpSubgroupShuffleXorINTEL(dataT Data, uint32_t Value) noexcept;

template <typename dataT>
extern dataT OpSubgroupBlockReadINTEL(const __global uint16_t *Ptr) noexcept;

template <typename dataT>
extern void OpSubgroupBlockWriteINTEL(__global uint16_t *Ptr,
                                      dataT Data) noexcept;

template <typename dataT>
extern dataT OpSubgroupBlockReadINTEL(const __global uint32_t *Ptr) noexcept;

template <typename dataT>
extern void OpSubgroupBlockWriteINTEL(__global uint32_t *Ptr,
                                      dataT Data) noexcept;

extern void prefetch(const __global char *Ptr, size_t NumBytes) noexcept;

#else // if !__SYCL_DEVICE_ONLY__

template <typename dataT>
extern OpTypeEvent *
OpGroupAsyncCopyGlobalToLocal(int32_t Scope, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              OpTypeEvent *E) noexcept {
  for (int i = 0; i < NumElements; i++) {
    Dest[i] = Src[i * Stride];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

template <typename dataT>
extern OpTypeEvent *
OpGroupAsyncCopyLocalToGlobal(int32_t Scope, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              OpTypeEvent *E) noexcept {
  for (int i = 0; i < NumElements; i++) {
    Dest[i * Stride] = Src[i];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

extern void prefetch(const char *Ptr, size_t NumBytes) noexcept;

#endif // !__SYCL_DEVICE_ONLY__

extern void OpControlBarrier(Scope Execution, Scope Memory,
                             uint32_t Semantics) noexcept;

extern void OpMemoryBarrier(Scope Memory, uint32_t Semantics) noexcept;

extern void OpGroupWaitEvents(int32_t Scope, uint32_t NumEvents,
                              OpTypeEvent ** WaitEvents) noexcept;

} // namespace __spirv
} // namespace cl
