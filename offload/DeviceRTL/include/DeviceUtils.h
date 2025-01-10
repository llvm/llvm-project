//===--- DeviceUtils.h - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_DEVICE_UTILS_H
#define OMPTARGET_DEVICERTL_DEVICE_UTILS_H

#include "DeviceTypes.h"
#include "Shared/Utils.h"

#pragma omp begin declare target device_type(nohost)

namespace utils {

template <typename T> struct type_identity {
  using type = T;
};

template <typename T, T v> struct integral_constant {
  inline static constexpr T value = v;
};

/// Freestanding SFINAE helpers.
template <class T> struct remove_cv : type_identity<T> {};
template <class T> struct remove_cv<const T> : type_identity<T> {};
template <class T> struct remove_cv<volatile T> : type_identity<T> {};
template <class T> struct remove_cv<const volatile T> : type_identity<T> {};
template <class T> using remove_cv_t = typename remove_cv<T>::type;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename T, typename U> struct is_same : false_type {};
template <typename T> struct is_same<T, T> : true_type {};
template <typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

template <typename T> struct is_floating_point {
  inline static constexpr bool value =
      is_same_v<remove_cv<T>, float> || is_same_v<remove_cv<T>, double>;
};
template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <bool B, typename T = void> struct enable_if;
template <typename T> struct enable_if<true, T> : type_identity<T> {};
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <class T> struct remove_addrspace : type_identity<T> {};
template <class T, int N>
struct remove_addrspace<T [[clang::address_space(N)]]> : type_identity<T> {};
template <class T>
using remove_addrspace_t = typename remove_addrspace<T>::type;

template <typename To, typename From> inline To bitCast(From V) {
  static_assert(sizeof(To) == sizeof(From), "Bad conversion");
  return __builtin_bit_cast(To, V);
}

/// Return the value \p Var from thread Id \p SrcLane in the warp if the thread
/// is identified by \p Mask.
int32_t shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane, int32_t Width);

int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta, int32_t Width);

int64_t shuffleDown(uint64_t Mask, int64_t Var, uint32_t Delta, int32_t Width);

uint64_t ballotSync(uint64_t Mask, int32_t Pred);

/// Return \p LowBits and \p HighBits packed into a single 64 bit value.
uint64_t pack(uint32_t LowBits, uint32_t HighBits);

/// Unpack \p Val into \p LowBits and \p HighBits.
void unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits);

/// Return true iff \p Ptr is pointing into shared (local) memory (AS(3)).
bool isSharedMemPtr(void *Ptr);

/// Return true iff \p Ptr is pointing into (thread) local memory (AS(5)).
bool isThreadLocalMemPtr(void *Ptr);

/// A  pointer variable that has by design an `undef` value. Use with care.
[[clang::loader_uninitialized]] static void *const UndefPtr;

#define OMP_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define OMP_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)

} // namespace utils

#pragma omp end declare target

#endif
