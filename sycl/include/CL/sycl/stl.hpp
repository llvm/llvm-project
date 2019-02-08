//==----------- stl.hpp - basic STL implementation -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.5 C++ Standard library classes required for the interface

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cl {
namespace sycl {

 template < class T, class Alloc = std::allocator<T> >
 using vector_class = std::vector<T, Alloc>;

 using string_class = std::string;

 template <class Sig>
 using function_class = std::function<Sig>;

 using mutex_class = std::mutex;

 template <class T, class Deleter = std::default_delete<T>>
 using unique_ptr_class = std::unique_ptr<T, Deleter>;

 template <class T>
 using shared_ptr_class = std::shared_ptr<T>;

 template <class T>
 using weak_ptr_class = std::weak_ptr<T>;

 template <class T>
 using hash_class = std::hash<T>;

 using exception_ptr_class = std::exception_ptr;

} // sycl
} // cl

