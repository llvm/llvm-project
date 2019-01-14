// RUN: %clang -std=c++11 -fsyntax-only %s

// Check that the test can be compiled with device compiler as well.
// RUN: %clang --sycl -fsyntax-only %s
//==--- accessor_static_check.cpp - Static checks for SYCL accessors -------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

namespace sycl {
  using namespace cl::sycl;
}

struct SomeStructure {
  char a;
  float b;
  union {
    int x;
    double y;
  } v;
};

// Check that accessor_impl is the only data field in accessor class,
// and that the accessor is a standard-layout structure. A pointer to
// a standard-layout class may be converted (with reinterpret_cast) to
// a pointer to its first non-static data member and vice versa.
// Along the way, many specializations of accessor are instantiated.

#define CHECK_ACCESSOR_SIZEOF(DataT, Dimensions, AccessMode, AccessTarget,     \
                              IsPlaceholder)                                   \
  static_assert(                                                               \
      std::is_standard_layout<sycl::accessor<                                  \
          DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>>::value, \
      "accessor is not a standard-layout structure");                          \
  static_assert(                                                               \
      sizeof(sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,       \
                            IsPlaceholder>) ==                                 \
          sizeof(sycl::detail::accessor_impl<                                  \
                 typename sycl::detail::DeviceValueType<DataT,                 \
                                                        AccessTarget>::type,   \
                 Dimensions, AccessMode, AccessTarget, IsPlaceholder>),        \
      "accessor_impl is not the only data field in accessor class");

#define CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode, AccessTarget) \
  CHECK_ACCESSOR_SIZEOF(DataT, Dimensions, AccessMode, AccessTarget,          \
                        sycl::access::placeholder::true_t);                   \
  CHECK_ACCESSOR_SIZEOF(DataT, Dimensions, AccessMode, AccessTarget,          \
                        sycl::access::placeholder::false_t);

#define CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions, AccessMode)    \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::global_buffer);   \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::constant_buffer); \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::local);           \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::host_buffer);

#if 0
// TODO:
// The following checks should be enabled after the corresponding
// access::targets are supported by DeviceValueType metafunction.
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::image);           \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::host_image);      \
  CHECK_ACCESSOR_SIZEOF_PH(DataT, Dimensions, AccessMode,          \
                           sycl::access::target::image_array);
#endif

#define CHECK_ACCESSOR_SIZEOF_AM(DataT, Dimensions)                            \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions, sycl::access::mode::read);       \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions, sycl::access::mode::write);      \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions, sycl::access::mode::read_write); \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions,                                  \
                           sycl::access::mode::discard_write);                 \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions,                                  \
                           sycl::access::mode::discard_read_write);            \
  CHECK_ACCESSOR_SIZEOF_AT(DataT, Dimensions, sycl::access::mode::atomic);

#define CHECK_ACCESSOR_SIZEOF_DIM(DataT) \
  CHECK_ACCESSOR_SIZEOF_AM(DataT, 0); \
  CHECK_ACCESSOR_SIZEOF_AM(DataT, 1); \
  CHECK_ACCESSOR_SIZEOF_AM(DataT, 2); \
  CHECK_ACCESSOR_SIZEOF_AM(DataT, 3);

#define CHECK_ACCESSOR_SIZEOF_ALL \
  CHECK_ACCESSOR_SIZEOF_DIM(char); \
  CHECK_ACCESSOR_SIZEOF_DIM(unsigned); \
  CHECK_ACCESSOR_SIZEOF_DIM(long long); \
  CHECK_ACCESSOR_SIZEOF_DIM(double); \
  CHECK_ACCESSOR_SIZEOF_DIM(SomeStructure);

CHECK_ACCESSOR_SIZEOF_ALL
