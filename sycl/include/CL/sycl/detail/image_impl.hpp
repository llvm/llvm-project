//==------------ image_impl.hpp --------------------------------------------==//
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

namespace detail {

template <int dimensions, typename AllocatorT> class image_impl {
public:
  image_impl(image_channel_order order, image_channel_type type,
             const range<dimensions> &range,
             const property_list &propList) {
    assert(!"Not implemented");
  }

  //image_impl(image_channel_order order, image_channel_type type,
             //const range<dimensions> &range, AllocatorT allocator,
             //const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  // image_impl(image_channel_order order, image_channel_type type,
  // const range<dimensions> &range, const range<dimensions - 1> &pitch,
  // const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  // image_impl(image_channel_order order, image_channel_type type,
  // const range<dimensions> &range, const range<dimensions - 1> &pitch,
  // AllocatorT allocator, const property_list &propList = {});

  //image_impl(void *hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //const property_list &propList = {});

  //image_impl(void *hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //AllocatorT allocator, const property_list &propList = {});

  //image_impl(const void *hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //const property_list &propList = {});

  //image_impl(const void *hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //AllocatorT allocator, const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  // image_impl(void *hostPointer, image_channel_order order, image_channel_type
  // type,
  // const range<dimensions> &range, range<dimensions - 1> &pitch,
  // const property_list &propList = {}) {assert(!"Not implemented");}

  /* Available only when: dimensions > 1 */
  // image_impl(void *hostPointer, image_channel_order order, image_channel_type
  // type,
  // const range<dimensions> &range, range<dimensions - 1> &pitch,
  // AllocatorT allocator, const property_list &propList = {}) {assert(!"Not
  // implemented");}

  //image_impl(shared_ptr_class<void> &hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //const property_list &propList = {});

  //image_impl(shared_ptr_class<void> &hostPointer, image_channel_order order,
             //image_channel_type type, const range<dimensions> &range,
             //AllocatorT allocator, const property_list &propList = {});

  /* Available only when: dimensions > 1 */
  // image_impl(shared_ptr_class<void> &hostPointer, image_channel_order order,
  // image_channel_type type, const range<dimensions> &range,
  // const range<dimensions - 1> &pitch, const property_list &propList = {})
  // {assert(!"Not implemented");}

  /* Available only when: dimensions > 1 */
  // image_impl(shared_ptr_class<void> &hostPointer, image_channel_order order,
  // image_channel_type type, const range<dimensions> &range,
  // const range<dimensions - 1> &pitch, AllocatorT allocator,
  // const property_list &propList = {}) {assert(!"Not implemented");}

  //image_impl(cl_mem clMemObject, const context &syclContext,
             //event availableEvent = {});

  /* -- property interface members -- */

  range<dimensions> get_range() const { assert(!"Not implemented"); }

  /* Available only when: dimensions > 1 */
  range<dimensions - 1> get_pitch() const { assert(!"Not implemented"); }

  size_t get_size() const { assert(!"Not implemented"); return 0;}

  size_t get_count() const { assert(!"Not implemented"); return 0; }

  AllocatorT get_allocator() const { assert(!"Not implemented"); }

  template <typename dataT, access::mode accessMode>
  accessor<dataT, dimensions, accessMode, access::target::image>
  get_access(handler &commandGroupHandler) {
    assert(!"Not implemented");
  }

  template <typename dataT, access::mode accessMode>
  accessor<dataT, dimensions, accessMode, access::target::host_image>
  get_access() {
    assert(!"Not implemented");
  }

  // template <typename Destination = std::nullptr_t>
  // void set_final_data(Destination finalData = std::nullptr);

  void set_write_back(bool flag) { assert(!"Not implemented"); }
};

} // namespace detail

} // namespace sycl
} // namespace cl
