//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of wrappers that manage device resources
/// like buffers, binaries, and kernels.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_DEVICERESOURCES_HPP
#define MATHTEST_DEVICERESOURCES_HPP

#include "mathtest/OffloadForward.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace mathtest {

class DeviceContext;

namespace detail {

void freeDeviceMemory(void *Address) noexcept;
} // namespace detail

//===----------------------------------------------------------------------===//
// ManagedBuffer
//===----------------------------------------------------------------------===//

template <typename T> class [[nodiscard]] ManagedBuffer {
public:
  ~ManagedBuffer() noexcept {
    if (Address)
      detail::freeDeviceMemory(Address);
  }

  ManagedBuffer(const ManagedBuffer &) = delete;
  ManagedBuffer &operator=(const ManagedBuffer &) = delete;

  ManagedBuffer(ManagedBuffer &&Other) noexcept
      : Address(Other.Address), Size(Other.Size) {
    Other.Address = nullptr;
    Other.Size = 0;
  }

  ManagedBuffer &operator=(ManagedBuffer &&Other) noexcept {
    if (this == &Other)
      return *this;

    if (Address)
      detail::freeDeviceMemory(Address);

    Address = Other.Address;
    Size = Other.Size;

    Other.Address = nullptr;
    Other.Size = 0;

    return *this;
  }

  [[nodiscard]] T *data() noexcept { return Address; }

  [[nodiscard]] const T *data() const noexcept { return Address; }

  [[nodiscard]] std::size_t getSize() const noexcept { return Size; }

  [[nodiscard]] operator llvm::MutableArrayRef<T>() noexcept {
    return llvm::MutableArrayRef<T>(data(), getSize());
  }

  [[nodiscard]] operator llvm::ArrayRef<T>() const noexcept {
    return llvm::ArrayRef<T>(data(), getSize());
  }

private:
  friend class DeviceContext;

  explicit ManagedBuffer(T *Address, std::size_t Size) noexcept
      : Address(Address), Size(Size) {}

  T *Address = nullptr;
  std::size_t Size = 0;
};

//===----------------------------------------------------------------------===//
// DeviceImage
//===----------------------------------------------------------------------===//

class [[nodiscard]] DeviceImage {
public:
  ~DeviceImage() noexcept;
  DeviceImage &operator=(DeviceImage &&Other) noexcept;

  DeviceImage(const DeviceImage &) = delete;
  DeviceImage &operator=(const DeviceImage &) = delete;

  DeviceImage(DeviceImage &&Other) noexcept;

private:
  friend class DeviceContext;

  explicit DeviceImage(ol_device_handle_t DeviceHandle,
                       ol_program_handle_t Handle) noexcept;

  ol_device_handle_t DeviceHandle = nullptr;
  ol_program_handle_t Handle = nullptr;
};

//===----------------------------------------------------------------------===//
// DeviceKernel
//===----------------------------------------------------------------------===//

template <typename KernelSignature> class [[nodiscard]] DeviceKernel {
public:
  DeviceKernel() = delete;

  DeviceKernel(const DeviceKernel &) = default;
  DeviceKernel &operator=(const DeviceKernel &) = default;
  DeviceKernel(DeviceKernel &&) noexcept = default;
  DeviceKernel &operator=(DeviceKernel &&) noexcept = default;

private:
  friend class DeviceContext;

  explicit DeviceKernel(std::shared_ptr<DeviceImage> Image,
                        ol_symbol_handle_t Kernel)
      : Image(std::move(Image)), Handle(Kernel) {}

  std::shared_ptr<DeviceImage> Image;
  ol_symbol_handle_t Handle = nullptr;
};
} // namespace mathtest

#endif // MATHTEST_DEVICERESOURCES_HPP
