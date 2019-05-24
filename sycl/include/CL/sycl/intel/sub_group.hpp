//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>
#include <type_traits>
#ifdef __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;
namespace intel {
template <typename>

struct is_vec : std::false_type {};
template <typename T, std::size_t N>
struct is_vec<cl::sycl::vec<T, N>> : std::true_type {};

struct minimum {
  template <typename T, GroupOperation O>
  static typename std::enable_if<
      !std::is_floating_point<T>::value && std::is_signed<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupSMin(::Scope::Subgroup, O, x);
  }

  template <typename T, ::GroupOperation O>
  static typename std::enable_if<
      !std::is_floating_point<T>::value && std::is_unsigned<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupUMin(::Scope::Subgroup, O, x);
  }

  template <typename T, ::GroupOperation O>
  static typename std::enable_if<std::is_floating_point<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupFMin(::Scope::Subgroup, O, x);
  }
};

struct maximum {
  template <typename T, ::GroupOperation O>
  static typename std::enable_if<
      !std::is_floating_point<T>::value && std::is_signed<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupSMax(::Scope::Subgroup, O, x);
  }

  template <typename T, ::GroupOperation O>
  static typename std::enable_if<
      !std::is_floating_point<T>::value && std::is_unsigned<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupUMax(::Scope::Subgroup, O, x);
  }

  template <typename T, ::GroupOperation O>
  static typename std::enable_if<std::is_floating_point<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupFMax(::Scope::Subgroup, O, x);
  }
};

struct plus {
  template <typename T, ::GroupOperation O>
  static typename std::enable_if<
      !std::is_floating_point<T>::value && std::is_integral<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupIAdd<T>(::Scope::Subgroup, O, x);
  }
  template <typename T, ::GroupOperation O>
  static typename std::enable_if<std::is_floating_point<T>::value, T>::type
  calc(T x) {
    return __spirv_GroupFAdd<T>(::Scope::Subgroup, O, x);
  }
};
struct sub_group {
  /* --- common interface members --- */

  id<1> get_local_id() const {
    return __spirv_BuiltInSubgroupLocalInvocationId;
  }
  range<1> get_local_range() const { return __spirv_BuiltInSubgroupSize; }

  range<1> get_max_local_range() const {
    return __spirv_BuiltInSubgroupMaxSize;
  }

  id<1> get_group_id() const { return __spirv_BuiltInSubgroupId; }

  unsigned int get_group_range() const {
    return __spirv_BuiltInNumSubgroups;
  }

  unsigned int get_uniform_group_range() const {
    return __spirv_BuiltInNumEnqueuedSubgroups;
  }

  /* --- vote / ballot functions --- */

  bool any(bool predicate) {
    return __spirv_GroupAny(::Scope::Subgroup, predicate);
  }

  bool all(bool predicate) {
    return __spirv_GroupAll(::Scope::Subgroup, predicate);
  }

  /* --- collectives --- */

  template <typename T>
  T broadcast(typename std::enable_if<std::is_arithmetic<T>::value, T>::type x,
              id<1> local_id) {
    return __spirv_GroupBroadcast<T>(::Scope::Subgroup, x,
                                            local_id.get(0));
  }

  template <typename T, class BinaryOperation>
  T reduce(typename std::enable_if<std::is_arithmetic<T>::value, T>::type x) {
    return BinaryOperation::template calc<T, ::Reduce>(x);
  }

  template <typename T, class BinaryOperation>
  T exclusive_scan(
      typename std::enable_if<std::is_arithmetic<T>::value, T>::type x) {
    return BinaryOperation::template calc<T, ::ExclusiveScan>(x);
  }

  template <typename T, class BinaryOperation>
  T inclusive_scan(
      typename std::enable_if<std::is_arithmetic<T>::value, T>::type x) {
    return BinaryOperation::template calc<T, ::InclusiveScan>(x);
  }

  template <typename T>
  using EnableIfIsArithmeticOrHalf = typename std::enable_if<
      (std::is_arithmetic<T>::value ||
       std::is_same<typename std::remove_const<T>::type, half>::value),
      T>::type;


  /* --- one - input shuffles --- */
  /* indices in [0 , sub - group size ) */

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle(T x, id<1> local_id) {
    return __spirv_SubgroupShuffleINTEL(x, local_id.get(0));
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type shuffle(T x,
                                                             id<1> local_id) {
    return __spirv_SubgroupShuffleINTEL((typename T::vector_t)x,
                                               local_id.get(0));
  }

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle_down(T x, uint32_t delta) {
    return shuffle_down(x, x, delta);
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type
  shuffle_down(T x, uint32_t delta) {
    return shuffle_down(x, x, delta);
  }

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle_up(T x, uint32_t delta) {
    return shuffle_up(x, x, delta);
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type
  shuffle_up(T x, uint32_t delta) {
    return shuffle_up(x, x, delta);
  }

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle_xor(T x, id<1> value) {
    return __spirv_SubgroupShuffleXorINTEL(x, (uint32_t)value.get(0));
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type shuffle_xor(T x,
                                                                 id<1> value) {
    return __spirv_SubgroupShuffleXorINTEL((typename T::vector_t)x,
                                                  (uint32_t)value.get(0));
  }

  /* --- two - input shuffles --- */
  /* indices in [0 , 2* sub - group size ) */
  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle(T x, T y, id<1> local_id) {
    return __spirv_SubgroupShuffleDownINTEL(
        x, y, local_id.get(0) - get_local_id().get(0));
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type shuffle(T x, T y,
                                                             id<1> local_id) {
    return __spirv_SubgroupShuffleDownINTEL(
        (typename T::vector_t)x, (typename T::vector_t)y,
        local_id.get(0) - get_local_id().get(0));
  }

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle_down(T current, T next, uint32_t delta) {
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type
  shuffle_down(T current, T next, uint32_t delta) {
    return __spirv_SubgroupShuffleDownINTEL(
        (typename T::vector_t)current, (typename T::vector_t)next, delta);
  }

  template <typename T>
  EnableIfIsArithmeticOrHalf<T>
  shuffle_up(T previous, T current, uint32_t delta) {
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);
  }

  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type
  shuffle_up(T previous, T current, uint32_t delta) {
    return __spirv_SubgroupShuffleUpINTEL(
        (typename T::vector_t)previous, (typename T::vector_t)current, delta);
  }

  /* --- sub - group load / stores --- */
  /* these can map to SIMD or block read / write hardware where available */

  template <typename T, access::address_space Space>
  typename std::enable_if<(sizeof(T) == sizeof(uint32_t) ||
                           sizeof(T) == sizeof(uint16_t)) &&
                              Space == access::address_space::global_space,
                          T>::type
  load(const multi_ptr<T, Space> src) {
    if (sizeof(T) == sizeof(uint32_t)) {
      uint32_t t = __spirv_SubgroupBlockReadINTEL<uint32_t>(
          (const __global uint32_t *)src.get());
      return *((T *)(&t));
    }
    uint16_t t = __spirv_SubgroupBlockReadINTEL<uint16_t>(
        (const __global uint16_t *)src.get());
    return *((T *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<(sizeof(T) == sizeof(uint32_t) ||
                               sizeof(T) == sizeof(uint16_t)) &&
                                  Space == access::address_space::global_space,
                              T>::type,
      N>
  load(const multi_ptr<T, Space> src) {
    if (N == 1) {
      return load<T, Space>(src);
    }
    if (sizeof(T) == sizeof(uint32_t)) {
      typedef uint32_t ocl_t __attribute__((ext_vector_type(N)));

      ocl_t t = __spirv_SubgroupBlockReadINTEL<ocl_t>(
          (const __global uint32_t *)src.get());
      return *((typename vec<T, N>::vector_t *)(&t));
    }
    typedef uint16_t ocl_t __attribute__((ext_vector_type(N)));

    ocl_t t = __spirv_SubgroupBlockReadINTEL<ocl_t>(
        (const __global uint16_t *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const typename std::enable_if<
            (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint16_t)) &&
                Space == access::address_space::global_space,
            T>::type &x) {
    if (sizeof(T) == sizeof(uint32_t)) {
      __spirv_SubgroupBlockWriteINTEL<uint32_t>(
          (__global uint32_t *)dst.get(), *((uint32_t *)&x));
    } else {
      __spirv_SubgroupBlockWriteINTEL<uint16_t>(
          (__global uint16_t *)dst.get(), *((uint16_t *)&x));
    }
  }

  template <int N, typename T, access::address_space Space>
  void store(multi_ptr<T, Space> dst,
             const vec<typename std::enable_if<N == 1, T>::type, N> &x) {
    store<T, Space>(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const vec<typename std::enable_if<
                    (sizeof(T) == sizeof(uint32_t) ||
                     sizeof(T) == sizeof(uint16_t)) &&
                        N != 1 && Space == access::address_space::global_space,
                    T>::type,
                N> &x) {
    if (sizeof(T) == sizeof(uint32_t)) {
      typedef uint32_t ocl_t __attribute__((ext_vector_type(N)));
      __spirv_SubgroupBlockWriteINTEL((__global uint32_t *)dst.get(),
                                             *((ocl_t *)&x));
    } else {
      typedef uint16_t ocl_t __attribute__((ext_vector_type(N)));
      __spirv_SubgroupBlockWriteINTEL((__global uint16_t *)dst.get(),
                                             *((ocl_t *)&x));
    }
  }

  /* --- synchronization functions --- */
  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    uint32_t flags = ::MemorySemantics::SequentiallyConsistent;
    switch (accessSpace) {
    case access::fence_space::global_space:
      flags |= ::MemorySemantics::CrossWorkgroupMemory;
      break;
    case access::fence_space::local_space:
      flags |= ::MemorySemantics::SubgroupMemory;
      break;
    case access::fence_space::global_and_local:
    default:
      flags |= ::MemorySemantics::CrossWorkgroupMemory |
               ::MemorySemantics::SubgroupMemory;
      break;
    }
    __spirv_ControlBarrier(::Scope::Subgroup,
                                  ::Scope::Workgroup, flags);
  }

protected:
  template <int dimensions> friend struct cl::sycl::nd_item;
  sub_group() = default;
};
} // namespace intel
} // namespace sycl
} // namespace cl
#else
#include <CL/sycl/intel/sub_group_host.hpp>
#endif
