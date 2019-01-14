//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define __NOEXCEPT noexcept
namespace cl {
namespace __spirv {
extern size_t BuiltInSubgroupLocalInvocationId() __NOEXCEPT;
extern size_t BuiltInSubgroupSize() __NOEXCEPT;
extern size_t BuiltInSubgroupMaxSize() __NOEXCEPT;
extern size_t BuiltInSubgroupId() __NOEXCEPT;
extern size_t BuiltInNumSubgroups() __NOEXCEPT;
extern size_t BuiltInNumEnqueuedSubgroups() __NOEXCEPT;
} // namespace __spirv
} // namespace cl

// TODO: rework to use SPIRV
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint3 __attribute__((ext_vector_type(3)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));
typedef ushort ushort2 __attribute__((ext_vector_type(2)));
typedef ushort ushort3 __attribute__((ext_vector_type(3)));
typedef ushort ushort4 __attribute__((ext_vector_type(4)));
typedef ushort ushort8 __attribute__((ext_vector_type(8)));
size_t get_sub_group_local_id();      // BuiltInSubgroupLocalInvocationId
size_t get_sub_group_size();          // BuiltInSubgroupSize
size_t get_max_sub_group_size();      // BuiltInSubgroupMaxSize
size_t get_sub_group_id();            // BuiltInSubgroupId
size_t get_num_sub_groups();          // BuiltInNumSubgroups
size_t get_enqueued_num_sub_groups(); // BuiltInNumEnqueuedSubgroups
int sub_group_any(int);
int sub_group_all(int);
int sub_group_broadcast(int x, uint sub_grou_local_id);
int sub_group_reduce_min(int x);
int sub_group_reduce_max(int x);
int sub_group_reduce_add(int x);
int sub_group_scan_exclusive_add(int x);
int sub_group_scan_exclusive_max(int x);
int sub_group_scan_exclusive_min(int x);
int sub_group_scan_inclusive_add(int x);
int sub_group_scan_inclusive_max(int x);
int sub_group_scan_inclusive_min(int x);
int intel_sub_group_shuffle(int data, uint c);
int intel_sub_group_shuffle_up(int prev, int cur, uint c);
int intel_sub_group_shuffle_down(int cur, int next, uint c);
int intel_sub_group_shuffle_xor(int data, uint c);
uint intel_sub_group_block_read(const __global uint *p);
uint2 intel_sub_group_block_read2(const __global uint *p);
uint4 intel_sub_group_block_read4(const __global uint *p);
uint8 intel_sub_group_block_read8(const __global uint *p);
void intel_sub_group_block_write(__global uint *p, uint data);
void intel_sub_group_block_write2(__global uint *p, uint2 data);
void intel_sub_group_block_write4(__global uint *p, uint4 data);
void intel_sub_group_block_write8(__global uint *p, uint8 data);

ushort intel_sub_group_block_read_us(const __global ushort *p);
ushort2 intel_sub_group_block_read_us2(const __global ushort *p);
ushort4 intel_sub_group_block_read_us4(const __global ushort *p);
ushort8 intel_sub_group_block_read_us8(const __global ushort *p);
void intel_sub_group_block_write_us(__global ushort *p, ushort data);
void intel_sub_group_block_write_us2(__global ushort *p, ushort2 data);
void intel_sub_group_block_write_us4(__global ushort *p, ushort4 data);
void intel_sub_group_block_write_us8(__global ushort *p, ushort8 data);
void sub_group_barrier(cl::sycl::detail::cl_mem_fence_flags flags);

namespace cl {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;
namespace intel {

enum class Operation { exclusive_scan, inclusive_scan, reduce };

struct minimum {
  Operation o;
  minimum(Operation op) : o(op) {}
  template <typename T> T operator()(T x) {
    switch (o) {
    case Operation::exclusive_scan: {
      return sub_group_scan_exclusive_min(x);
    }
    case Operation::inclusive_scan: {
      return sub_group_scan_inclusive_min(x);
    }
    case Operation::reduce: {
      return sub_group_reduce_min(x);
    }
    }
  }
};

struct maximum {
  Operation o;
  maximum(Operation op) : o(op) {}
  template <typename T> T operator()(T x) {
    switch (o) {
    case Operation::exclusive_scan: {
      return sub_group_scan_exclusive_max(x);
    }
    case Operation::inclusive_scan: {
      return sub_group_scan_inclusive_max(x);
    }
    case Operation::reduce: {
      return sub_group_reduce_max(x);
    }
    }
  }
};

struct plus {
  Operation o;
  plus(Operation op) : o(op) {}
  template <typename T> T operator()(T x) {
    switch (o) {
    case Operation::exclusive_scan: {
      return sub_group_scan_exclusive_add(x);
    }
    case Operation::inclusive_scan: {
      return sub_group_scan_inclusive_add(x);
    }
    case Operation::reduce: {
      return sub_group_reduce_add(x);
    }
    }
  }
};
struct sub_group {
  /* --- common interface members --- */

  id<1> get_local_id() const {
    return get_sub_group_local_id(); //*cl::__spirv::BuiltInSubgroupLocalInvocationId();
  }
  range<1> get_local_range() const {
    return get_sub_group_size(); // cl::__spirv::BuiltInSubgroupSize();
  }

  range<1> get_max_local_range() const {
    return get_max_sub_group_size(); // cl::__spirv::BuiltInSubgroupMaxSize();
  }

  id<1> get_group_id() const {
    return get_sub_group_id(); // cl::__spirv::BuiltInSubgroupId();
  }

  size_t get_group_range() const {
    return get_num_sub_groups(); // cl::__spirv::BuiltInNumSubgroups();
  }

  size_t get_uniform_group_range() const {
    return get_enqueued_num_sub_groups(); // cl::__spirv::BuiltInNumEnqueuedSubgroups();
  }

  /* --- vote / ballot functions --- */

  bool any(bool predicate) { return sub_group_any(predicate); }

  bool all(bool predicate) { return sub_group_all(predicate); }

  /* --- collectives --- */

  template <typename T> T broadcast(T x, id<1> local_id) {
    return sub_group_broadcast(x, local_id.get(0));
  }

  template <typename T, class BinaryOperation> T reduce(T x) {
    BinaryOperation o(Operation::reduce);
    return o(x);
  }

  template <typename T, class BinaryOperation> T exclusive_scan(T x) {
    BinaryOperation o(Operation::exclusive_scan);
    return o(x);
  }

  template <typename T, class BinaryOperation> T inclusive_scan(T x) {
    BinaryOperation o(Operation::inclusive_scan);
    return o(x);
  }

  /* --- one - input shuffles --- */
  /* indices in [0 , sub - group size ) */

  template <typename T> T shuffle(T x, id<1> local_id) {
    return intel_sub_group_shuffle(x, local_id.get(0));
  }

  template <typename T> T shuffle_down(T x, uint32_t delta) {
    return intel_sub_group_shuffle_down(x, x, delta);
  }

  template <typename T> T shuffle_up(T x, uint32_t delta) {
    return intel_sub_group_shuffle_up(x, x, delta);
  }

  template <typename T> T shuffle_xor(T x, id<1> value) {
    return intel_sub_group_shuffle_xor(x, value.get(0));
  }

  /* --- two - input shuffles --- */
  /* indices in [0 , 2* sub - group size ) */
  template <typename T> T shuffle(T x, T y, id<1> local_id) {
    return intel_sub_group_shuffle_down(
        x, y, local_id.get(0) - get_local_id().get(0));
  }

  template <typename T> T shuffle_down(T current, T next, uint32_t delta) {
    return intel_sub_group_shuffle_down(current, next, delta);
  }
  template <typename T> T shuffle_up(T previous, T current, uint32_t delta) {
    return intel_sub_group_shuffle_up(previous, current, delta);
  }

  /* --- sub - group load / stores --- */
  /* these can map to SIMD or block read / write hardware where available */

  template <typename T, access::address_space Space>
  typename std::enable_if<sizeof(T) == sizeof(uint), T>::type
  load(const multi_ptr<T, Space> src) {
    uint t = intel_sub_group_block_read((const __global uint *)src.get());
    return *((T *)&t);
  }

  template <typename T, access::address_space Space>
  typename std::enable_if<sizeof(T) == sizeof(ushort), T>::type
  load(const multi_ptr<T, Space> src) {
    ushort t =
        intel_sub_group_block_read_us((const __global ushort *)src.get());
    return *((T *)&t);
  }

  template <int N, typename T, access::address_space Space>
  typename std::enable_if<sizeof(T) == sizeof(uint) && N == 1, T>::type
  load(const multi_ptr<T, Space> src) {
    uint t = intel_sub_group_block_read((const __global uint *)src.get());
    return *((T *)&t);
  }

  template <int N, typename T, access::address_space Space>
  typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 1, T>::type
  load(const multi_ptr<T, Space> src) {
    uint t = intel_sub_group_block_read_us((const __global ushort *)src.get());
    return *((T *)&t);
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(uint) && N == 2, T>::type, N>
  load(const multi_ptr<T, Space> src) {
    uint2 t = intel_sub_group_block_read2((const __global uint *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 2, T>::type,
      N>
  load(const multi_ptr<T, Space> src) {
    ushort2 t =
        intel_sub_group_block_read_us2((const __global ushort *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(uint) && N == 4, T>::type, N>
  load(const multi_ptr<T, Space> src) {
    uint4 t = intel_sub_group_block_read4((const __global uint *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 4, T>::type,
      N>
  load(const multi_ptr<T, Space> src) {
    ushort4 t =
        intel_sub_group_block_read_us4((const __global ushort *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(uint) && N == 8, T>::type, N>
  load(const multi_ptr<T, Space> src) {
    uint8 t = intel_sub_group_block_read8((const __global uint *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 8, T>::type,
      N>
  load(const multi_ptr<T, Space> src) {
    ushort8 t =
        intel_sub_group_block_read_us8((const __global ushort *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const typename std::enable_if<sizeof(T) == sizeof(uint), T>::type &x) {
    intel_sub_group_block_write((__global uint *)dst.get(), *((uint *)&x));
  }

  template <typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const typename std::enable_if<sizeof(T) == sizeof(ushort), T>::type &x) {
    intel_sub_group_block_write_us((__global ushort *)dst.get(),
                                   *((ushort *)&x));
  }

  template <int N, typename T, access::address_space Space>
  void store(multi_ptr<T, Space> dst,
             const typename std::enable_if<sizeof(T) == sizeof(uint) && N == 1,
                                           T>::type &x) {
    intel_sub_group_block_write((__global uint *)dst.get(), *((uint *)&x));
  }

  template <int N, typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 1,
                                      T>::type &x) {
    intel_sub_group_block_write_us((__global ushort *)dst.get(),
                                   *((ushort *)&x));
  }

  template <int N, typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const vec<
          typename std::enable_if<sizeof(T) == sizeof(uint) && N == 2, T>::type,
          N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write2((__global uint *)dst.get(), *((uint2 *)&t));
  }
  template <int N, typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 2,
                                          T>::type,
                  N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write_us2((__global ushort *)dst.get(),
                                    *((ushort2 *)&t));
  }

  template <int N, typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const vec<
          typename std::enable_if<sizeof(T) == sizeof(uint) && N == 4, T>::type,
          N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write4((__global uint *)dst.get(), *((uint4 *)&t));
  }

  template <int N, typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 4,
                                          T>::type,
                  N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write_us4((__global ushort *)dst.get(),
                                    *((ushort4 *)&t));
  }

  template <int N, typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const vec<
          typename std::enable_if<sizeof(T) == sizeof(uint) && N == 8, T>::type,
          N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write8((__global uint *)dst.get(), *((uint8 *)&t));
  }

  template <int N, typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const vec<typename std::enable_if<sizeof(T) == sizeof(ushort) && N == 8,
                                          T>::type,
                  N> &x) {
    typename vec<T, N>::vector_t t = x;
    intel_sub_group_block_write_us8((__global ushort *)dst.get(),
                                    *((ushort8 *)&t));
  }

  /* --- synchronization functions --- */
  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    cl::sycl::detail::cl_mem_fence_flags flags;
    switch (accessSpace) {
    case access::fence_space::local_space:
      flags = cl::sycl::detail::CLK_LOCAL_MEM_FENCE;
      break;
    case access::fence_space::global_space:
      flags = cl::sycl::detail::CLK_GLOBAL_MEM_FENCE;
      break;
    case access::fence_space::global_and_local:
    default:
      flags = cl::sycl::detail::CLK_LOCAL_MEM_FENCE |
              cl::sycl::detail::CLK_GLOBAL_MEM_FENCE;
      break;
    }
    ::sub_group_barrier(flags);
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
