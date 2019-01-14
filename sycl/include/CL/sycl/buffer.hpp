//==----------- buffer.hpp --- SYCL buffer ---------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
          typename AllocatorT = cl::sycl::buffer_allocator<T>>
class buffer {
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
        bufferRange, propList);
  }

  // buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
  // const property_list &propList = {}) {
  //     impl = std::make_shared<detail::buffer_impl>(bufferRange, allocator,
  //     propList);
  // }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
        hostData, bufferRange, propList);
  }

  // buffer(T *hostData, const range<dimensions> &bufferRange,
  // AllocatorT allocator, const property_list &propList = {}) {
  //     impl = std::make_shared<detail::buffer_impl>(hostData, bufferRange,
  //     allocator, propList);
  // }

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
        hostData, bufferRange, propList);
  }

  // buffer(const T *hostData, const range<dimensions> &bufferRange,
  // AllocatorT allocator, const property_list &propList = {}) {
  //     impl = std::make_shared<detail::buffer_impl>(hostData, bufferRange,
  //     allocator, propList);
  // }

  // buffer(const shared_ptr_class<T> &hostData,
  // const range<dimensions> &bufferRange, AllocatorT allocator,
  // const property_list &propList = {}) {
  //     impl = std::make_shared<detail::buffer_impl>(hostData, bufferRange,
  //     allocator, propList);
  // }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
        hostData, bufferRange, propList);
  }

  // template <class InputIterator>
  // buffer<T, 1>(InputIterator first, InputIterator last, AllocatorT allocator,
  // const property_list &propList = {}) {
  //     impl = std::make_shared<detail::buffer_impl>(first, last, allocator,
  //     propList);
  // }

  template <class InputIterator, int N = dimensions,
            typename = std::enable_if<N == 1>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
        first, last, propList);
  }

  // buffer(buffer<T, dimensions, AllocatorT> b, const id<dimensions>
  // &baseIndex, const range<dimensions> &subRange) {
  //     impl = std::make_shared<detail::buffer_impl>(b, baseIndex, subRange);
  // }

  template <int N = dimensions, typename = std::enable_if<N == 1>>
  buffer(cl_mem MemObject, const context &SyclContext,
         event AvailableEvent = {}) {
    impl = std::make_shared<detail::buffer_impl<T, dimensions, AllocatorT>>(
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

  range<dimensions> get_range() const { return impl->get_range(); }

  size_t get_count() const { return impl->get_count(); }

  size_t get_size() const { return impl->get_size(); }

  AllocatorT get_allocator() const { return impl->get_allocator(); }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler) {
    return impl->template get_access<mode, target>(*this, commandGroupHandler);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access() {
    return impl->template get_access<mode>(*this);
  }

  // template <access::mode mode, access::target target =
  // access::target::global_buffer> accessor<T, dimensions, mode, target,
  // access::placeholder::false_t> get_access( handler &commandGroupHandler,
  // range<dimensions> accessRange, id<dimensions> accessOffset = {}) {
  //     return impl->get_access(commandGroupHandler, accessRange,
  //     accessOffset);
  // }

  // template <access::mode mode>
  // accessor<T, dimensions, mode, access::target::host_buffer,
  // access::placeholder::false_t> get_access( range<dimensions> accessRange,
  // id<dimensions> accessOffset = {}) {
  //     return impl->get_access(accessRange, accessOffset);
  // }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    impl->set_final_data(finalData);
  }

  // void set_write_back(bool flag = true) { return impl->set_write_back(flag);
  // }

  // bool is_sub_buffer() const { return impl->is_sub_buffer(); }

  // template <typename ReinterpretT, int ReinterpretDim>
  // buffer<ReinterpretT, ReinterpretDim, AllocatorT>
  // reinterpret(range<ReinterpretDim> reinterpretRange) const {
  //     return impl->reinterpret((reinterpretRange));
  // }

private:
  shared_ptr_class<detail::buffer_impl<T, dimensions, AllocatorT>> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
};
} // namespace sycl
} // namespace cl

namespace std {
template <typename T, int dimensions, typename AllocatorT>
struct hash<cl::sycl::buffer<T, dimensions, AllocatorT>> {
  size_t
  operator()(const cl::sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return hash<std::shared_ptr<
        cl::sycl::detail::buffer_impl<T, dimensions, AllocatorT>>>()(
        cl::sycl::detail::getSyclObjImpl(b));
  }
};
} // namespace std
