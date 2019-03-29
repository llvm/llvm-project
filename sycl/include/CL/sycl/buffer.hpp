//==----------- buffer.hpp --- SYCL buffer ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

// TODO: 4.3.4 Properties

namespace cl {
namespace sycl {
class handler;
class queue;
template <int dimentions> class range;

template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator>
class buffer {
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        get_count() * sizeof(T), propList);
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        get_count() * sizeof(T), propList, allocator);
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList);
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList, allocator);
  }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList);
  }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList, allocator);
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList, allocator);
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), propList);
  }

  template <class InputIterator>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        first, last, get_count() * sizeof(T), propList, allocator);
  }

  template <class InputIterator, int N = dimensions,
            typename = std::enable_if<N == 1>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        first, last, get_count() * sizeof(T), propList);
  }

  // buffer(buffer<T, dimensions, AllocatorT> b, const id<dimensions>
  // &baseIndex, const range<dimensions> &subRange) {
  //     impl = std::make_shared<detail::buffer_impl>(b, baseIndex, subRange);
  // }

  template <int N = dimensions, typename = std::enable_if<N == 1>>
  buffer(cl_mem MemObject, const context &SyclContext,
         event AvailableEvent = {}) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        MemObject, SyclContext, AvailableEvent);
  }

  buffer(const buffer &rhs) = default;

  buffer(buffer &&rhs) = default;

  buffer &operator=(const buffer &rhs) = default;

  buffer &operator=(buffer &&rhs) = default;

  ~buffer() = default;

  bool operator==(const buffer &rhs) const { return impl == rhs.impl; }

  bool operator!=(const buffer &rhs) const { return !(*this == rhs); }

  /* -- common interface members -- */

  /* -- property interface members -- */

  range<dimensions> get_range() const { return Range; }

  size_t get_count() const { return Range.size(); }

  size_t get_size() const { return impl->get_size(); }

  AllocatorT get_allocator() const { return impl->get_allocator(); }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler) {
    return impl->template get_access<T, dimensions, mode, target>(
        *this, commandGroupHandler);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access() {
    return impl->template get_access<T, dimensions, mode>(*this);
  }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset = {}) {
    return impl->template get_access<T, dimensions, mode, target>(
        *this, commandGroupHandler, accessRange, accessOffset);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(range<dimensions> accessRange, id<dimensions> accessOffset = {}) {
    return impl->template get_access<T, dimensions, mode>(*this, accessRange,
                                                          accessOffset);
  }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    impl->set_final_data(finalData);
  }

  // void set_write_back(bool flag = true) { return impl->set_write_back(flag);
  // }

  // bool is_sub_buffer() const { return impl->is_sub_buffer(); }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if (sizeof(ReinterpretT) * reinterpretRange.size() != get_size())
      throw cl::sycl::invalid_object_error(
          "Total size in bytes represented by the type and range of the "
          "reinterpreted SYCL buffer does not equal the total size in bytes "
          "represented by the type and range of this SYCL buffer");
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(impl,
                                                            reinterpretRange);
  }

  template <typename propertyT> bool has_property() const {
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
  }

private:
  shared_ptr_class<detail::buffer_impl<AllocatorT>> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <typename A, int dims, typename C> friend class buffer;
  range<dimensions> Range;

  // Reinterpret contructor
  buffer(shared_ptr_class<detail::buffer_impl<AllocatorT>> Impl,
         range<dimensions> reinterpretRange)
      : impl(Impl), Range(reinterpretRange){};
};
} // namespace sycl
} // namespace cl

namespace std {
template <typename T, int dimensions, typename AllocatorT>
struct hash<cl::sycl::buffer<T, dimensions, AllocatorT>> {
  size_t
  operator()(const cl::sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return hash<std::shared_ptr<cl::sycl::detail::buffer_impl<AllocatorT>>>()(
        cl::sycl::detail::getSyclObjImpl(b));
  }
};
} // namespace std
