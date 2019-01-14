//==------------ pointers.hpp - SYCL pointers classes ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
