//==------------ image.hpp -------------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/image_impl.hpp>
#include <cstddef>

namespace cl {
namespace sycl {

using byte = unsigned char;

using image_allocator = std::allocator<byte>;

template <int dimentions> class range;

template <int dimensions = 1,
          typename AllocatorT = cl::sycl::image_allocator>
class image {
public:
  image(image_channel_order order, image_channel_type type,
        const range<dimensions> &range, const property_list &propList = {}) {
    impl = std::make_shared<detail::image_impl<dimensions, AllocatorT>>(
        order, type, range, propList);
  }

  //image(image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, AllocatorT allocator,
        //const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, const range<dimensions - 1> &pitch,
        //const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, const range<dimensions - 1> &pitch,
        //AllocatorT allocator, const property_list &propList = {});

  //image(void *hostPointer, image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, const property_list &propList = {});

  //image(void *hostPointer, image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, AllocatorT allocator,
        //const property_list &propList = {});

  //image(const void *hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //const property_list &propList = {});

  //image(const void *hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //AllocatorT allocator, const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(void *hostPointer, image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, range<dimensions - 1> &pitch,
        //const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(void *hostPointer, image_channel_order order, image_channel_type type,
        //const range<dimensions> &range, range<dimensions - 1> &pitch,
        //AllocatorT allocator, const property_list &propList = {});

  //image(shared_ptr_class<void> &hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //const property_list &propList = {});

  //image(shared_ptr_class<void> &hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //AllocatorT allocator, const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(shared_ptr_class<void> &hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //const range<dimensions - 1> &pitch, const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  //image(shared_ptr_class<void> &hostPointer, image_channel_order order,
        //image_channel_type type, const range<dimensions> &range,
        //const range<dimensions - 1> &pitch, AllocatorT allocator,
        //const property_list &propList = {});

  image(cl_mem clMemObject, const context &syclContext,
        event availableEvent = {});

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  bool operator==(const image &rhs) const { return impl == rhs.impl; }

  bool operator!=(const image &rhs) const { return !(*this == rhs); }

  /* -- common interface members -- */

  /* -- property interface members -- */

  range<dimensions> get_range() const { return impl->get_range(); }

  /* Available only when: dimensions > 1 */
  range<dimensions - 1> get_pitch() const { return impl->get_pitch(); }

  size_t get_size() const { return impl->get_size(); }

  size_t get_count() const { return impl->get_count(); }

  AllocatorT get_allocator() const { return impl->get_allocator(); }

  template <typename dataT, access::mode accessMode>
  accessor<dataT, dimensions, accessMode, access::target::image>
  get_access(handler &commandGroupHandler) {
    return impl->template get_access<dataT, accessMode>();
  }

  template <typename dataT, access::mode accessMode>
  accessor<dataT, dimensions, accessMode, access::target::host_image>
  get_access() {
    return impl->template get_access<dataT, accessMode>();
  }

  //template <typename Destination = std::nullptr_t>
  //void set_final_data(Destination finalData = std::nullptr);

  void set_write_back(bool flag = true) { impl->set_write_back(flag); }

private:
  shared_ptr_class<detail::image_impl<dimensions, AllocatorT>> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
};

} // namespace sycl
} // namespace cl

namespace std {
template <int dimensions, typename AllocatorT>
struct hash<cl::sycl::image<dimensions, AllocatorT>> {
  size_t operator()(const cl::sycl::image<dimensions, AllocatorT> &i) const {
    return hash<std::shared_ptr<
        cl::sycl::detail::image_impl<dimensions, AllocatorT>>>()(i.impl);
  }
};
} // namespace std
