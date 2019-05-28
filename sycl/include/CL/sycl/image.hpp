//==------------ image.hpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/image_impl.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>
#include <cstddef>

namespace cl {
namespace sycl {

enum class image_channel_order : unsigned int {
  a,
  r,
  rx,
  rg,
  rgx,
  ra,
  rgb,
  rgbx,
  rgba,
  argb,
  bgra,
  intensity,
  luminance,
  abgr
};

enum class image_channel_type : unsigned int {
  snorm_int8,
  snorm_int16,
  unorm_int8,
  unorm_int16,
  unorm_short_565,
  unorm_short_555,
  unorm_int_101010,
  signed_int8,
  signed_int16,
  signed_int32,
  unsigned_int8,
  unsigned_int16,
  unsigned_int32,
  fp16,
  fp32
};

using byte = unsigned char;

using image_allocator = std::allocator<byte>;

template <int Dimensions = 1, typename AllocatorT = cl::sycl::image_allocator>
class image {
public:
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        Order, Type, Range, PropList);
  }

  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        Order, Type, Range, Allocator, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        Order, Type, Range, Pitch, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        Order, Type, Range, Pitch, Allocator, PropList);
  }

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, PropList);
  }

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Allocator, PropList);
  }

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, PropList);
  }

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Allocator, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Pitch, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Pitch, Allocator, PropList);
  }

  image(shared_ptr_class<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, PropList);
  }

  image(shared_ptr_class<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Allocator, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(shared_ptr_class<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Pitch, PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(shared_ptr_class<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename std::enable_if<B, range<Dimensions - 1>>::type &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions, AllocatorT>>(
        HostPointer, Order, Type, Range, Pitch, Allocator, PropList);
  }

  image(cl_mem ClMemObject, const context &SyclContext,
        event AvailableEvent = {});

  /* -- common interface members -- */

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  bool operator==(const image &rhs) const { return impl == rhs.impl; }

  bool operator!=(const image &rhs) const { return !(*this == rhs); }

  /* -- property interface members -- */
  template <typename propertyT> bool has_property() const {
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
  }

  range<Dimensions> get_range() const { return impl->get_range(); }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  typename std::enable_if<B, range<Dimensions - 1>>::type get_pitch() const {
    return impl->get_pitch();
  }

  // Returns the size of the image storage in bytes
  size_t get_size() const { return impl->get_size(); }

  // Returns the total number of elements in the image
  size_t get_count() const { return impl->get_count(); }

  // Returns the allocator provided to the image
  AllocatorT get_allocator() const { return impl->get_allocator(); }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination FinalData = nullptr) {
    if (true)
      throw cl::sycl::feature_not_supported("Feature Not Implemented");
    return;
  }

  void set_write_back(bool Flag = true) {
    if (true)
      throw cl::sycl::feature_not_supported("Feature Not Implemented");
    return;
  }

private:
  shared_ptr_class<detail::image_impl<Dimensions, AllocatorT>> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
};

} // namespace sycl
} // namespace cl

namespace std {
template <int Dimensions, typename AllocatorT>
struct hash<cl::sycl::image<Dimensions, AllocatorT>> {
  size_t operator()(const cl::sycl::image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<
        cl::sycl::detail::image_impl<Dimensions, AllocatorT>>>()(
        cl::sycl::detail::getSyclObjImpl(I));
  }
};
} // namespace std
