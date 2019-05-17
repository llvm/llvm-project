//==----------------- stream_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/queue.hpp>

namespace cl {
namespace sycl {

namespace detail {
class stream_impl {
public:
  using AccessorType = accessor<char, 1, cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer,
                                cl::sycl::access::placeholder::false_t>;

  using OffsetAccessorType =
      accessor<unsigned, 1, cl::sycl::access::mode::atomic,
               cl::sycl::access::target::global_buffer,
               cl::sycl::access::placeholder::false_t>;

  stream_impl(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  // Method to provide an access to the stream buffer
  AccessorType getAccessor(handler &CGH) {
    return Buf.get_access<cl::sycl::access::mode::read_write>(
        CGH, range<1>(BufferSize_), id<1>(OffsetSize));
  }

  // Method to provide an atomic access to the offset in the stream buffer
  OffsetAccessorType getOffsetAccessor(handler &CGH) {
    auto OffsetSubBuf = buffer<char, 1>(Buf, id<1>(0), range<1>(OffsetSize));
    auto ReinterpretedBuf = OffsetSubBuf.reinterpret<unsigned, 1>(range<1>(1));
    return ReinterpretedBuf.get_access<cl::sycl::access::mode::atomic>(
        CGH, range<1>(1), id<1>(0));
  }

  // Copy stream buffer to the host and print the contents
  void flush();

  size_t get_size() const;

  size_t get_max_statement_size() const;

private:
  // Size of the stream buffer
  size_t BufferSize_;

  // Maximum number of symbols which could be streamed from the beginning of a
  // statement till the semicolon
  size_t MaxStatementSize_;

  // Size of the variable which is used as an offset in the stream buffer.
  // Additinonal memory is allocated in the beginning of the stream buffer for
  // this variable.
  static const size_t OffsetSize = sizeof(unsigned);

  // Vector on the host side which is used to initialize the stream buffer
  std::vector<char> Data;

  // Stream buffer
  buffer<char, 1> Buf;
};
} // namespace detail
} // namespace sycl
} // namespace cl

