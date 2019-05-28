//==------------ image_impl.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/image.hpp>

namespace cl {
namespace sycl {
namespace detail {

uint8_t getImageNumberChannels(image_channel_order Order) {
  switch (Order) {
  case image_channel_order::a:
  case image_channel_order::r:
  case image_channel_order::rx:
  case image_channel_order::intensity:
  case image_channel_order::luminance:
    return 1;
  case image_channel_order::rg:
  case image_channel_order::rgx:
  case image_channel_order::ra:
    return 2;
  case image_channel_order::rgb:
  case image_channel_order::rgbx:
    return 3;
  case image_channel_order::rgba:
  case image_channel_order::argb:
  case image_channel_order::bgra:
  case image_channel_order::abgr:
    return 4;
  }
  assert(!"Unhandled image channel order");
  return 0;
}

// Returns the number of bytes per image element
uint8_t getImageElementSize(uint8_t NumChannels, image_channel_type Type) {
  size_t Retval = 0;
  switch (Type) {
  case image_channel_type::snorm_int8:
  case image_channel_type::unorm_int8:
  case image_channel_type::signed_int8:
  case image_channel_type::unsigned_int8:
    Retval = NumChannels;
    break;
  case image_channel_type::snorm_int16:
  case image_channel_type::unorm_int16:
  case image_channel_type::signed_int16:
  case image_channel_type::unsigned_int16:
  case image_channel_type::fp16:
    Retval = 2 * NumChannels;
    break;
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int32:
  case image_channel_type::fp32:
    Retval = 4 * NumChannels;
    break;
  case image_channel_type::unorm_short_565:
  case image_channel_type::unorm_short_555:
    Retval = 2;
    break;
  case image_channel_type::unorm_int_101010:
    Retval = 4;
    break;
  default:
    assert(!"Unhandled image channel type");
  }
  // OpenCL states that "The number of bits per element determined by the
  // image_channel_type and image_channel_order must be a power of two"
  assert(((Retval - 1) & Retval) == 0);
  return Retval;
}

} // namespace detail
} // namespace sycl
} // namespace cl
