//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace cl {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class fence_space {
  local_space,
  global_space,
  global_and_local
};

enum class placeholder { false_t, true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space
};

}  // namespace access

namespace detail {

constexpr bool isTargetHostAccess(access::target T) {
  return T == access::target::host_buffer || T == access::target::host_image;
}

constexpr bool modeNeedsOldData(access::mode m) {
  return m == access::mode::read || m == access::mode::write ||
         m == access::mode::read_write || m == access::mode::atomic;
}

constexpr bool modeWritesNewData(access::mode m) {
  return m != access::mode::read;
}

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_GLOBAL_AS __global
#define SYCL_LOCAL_AS __local
#define SYCL_CONSTANT_AS __constant
#define SYCL_PRIVATE_AS __private
#else
#define SYCL_GLOBAL_AS
#define SYCL_LOCAL_AS
#define SYCL_CONSTANT_AS
#define SYCL_PRIVATE_AS
#endif

template <typename dataT, access::target accessTarget>
struct DeviceValueType;

template <typename dataT>
struct DeviceValueType<dataT, access::target::global_buffer> {
  using type = SYCL_GLOBAL_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::constant_buffer> {
  using type = SYCL_CONSTANT_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::local> {
  using type = SYCL_LOCAL_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::host_buffer> {
  using type = dataT;
};

template <typename ElementType, access::address_space addressSpace>
struct PtrValueType;

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::private_space> {
  using type = SYCL_PRIVATE_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::global_space> {
  using type = SYCL_GLOBAL_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::constant_space> {
  using type = SYCL_CONSTANT_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::local_space> {
  using type = SYCL_LOCAL_AS ElementType;
};

template <class T>
struct remove_AS {
  typedef T type;
};

#ifdef __SYCL_DEVICE_ONLY__
template <class T>
struct remove_AS<SYCL_GLOBAL_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_PRIVATE_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_LOCAL_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_CONSTANT_AS T> {
  typedef T type;
};
#endif

#undef SYCL_GLOBAL_AS
#undef SYCL_LOCAL_AS
#undef SYCL_CONSTANT_AS
#undef SYCL_PRIVATE_AS

} // namespace detail

}  // namespace sycl
}  // namespace cl
