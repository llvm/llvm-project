//==------------ pointers.hpp - SYCL pointers classes ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/access/access.hpp>


namespace cl {
namespace sycl {

template <typename ElementType, access::address_space Space> class multi_ptr;
// Template specialization aliases for different pointer address spaces

template <typename ElementType>
using global_ptr = multi_ptr<ElementType, access::address_space::global_space>;

template <typename ElementType>
using local_ptr = multi_ptr<ElementType, access::address_space::local_space>;

template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space>;

template <typename ElementType>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space>;

} // namespace sycl
} // namespace cl
