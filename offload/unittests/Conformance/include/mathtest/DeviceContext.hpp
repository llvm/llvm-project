//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the DeviceContext class, which serves
/// as the high-level interface to a particular device (GPU).
///
/// This class provides methods for allocating buffers, loading binaries, and
/// getting and launching kernels on the device.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_DEVICECONTEXT_HPP
#define MATHTEST_DEVICECONTEXT_HPP

#include "mathtest/DeviceResources.hpp"
#include "mathtest/Dim.hpp"
#include "mathtest/ErrorHandling.hpp"
#include "mathtest/Support.hpp"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mathtest {

const llvm::SetVector<llvm::StringRef> &getPlatforms();

namespace detail {

void allocManagedMemory(ol_device_handle_t DeviceHandle, std::size_t Size,
                        void **AllocationOut) noexcept;
} // namespace detail

class DeviceContext {
  // For simplicity, the current design of this class doesn't have support for
  // asynchronous operations and all types of memory allocation.
  //
  // Other use cases could benefit from operations like enqueued kernel launch
  // and enqueued memcpy, as well as device and host memory allocation.

public:
  explicit DeviceContext(std::size_t GlobalDeviceId = 0);

  explicit DeviceContext(llvm::StringRef Platform, std::size_t DeviceId = 0);

  template <typename T>
  ManagedBuffer<T> createManagedBuffer(std::size_t Size) const noexcept {
    void *UntypedAddress = nullptr;

    detail::allocManagedMemory(DeviceHandle, Size * sizeof(T), &UntypedAddress);
    T *TypedAddress = static_cast<T *>(UntypedAddress);

    return ManagedBuffer<T>(TypedAddress, Size);
  }

  [[nodiscard]] std::shared_ptr<DeviceImage>
  loadBinary(llvm::StringRef Directory, llvm::StringRef BinaryName) const;

  [[nodiscard]] std::optional<std::shared_ptr<DeviceImage>>
  tryLoadBinary(llvm::StringRef Directory, llvm::StringRef BinaryName) const;

  template <typename KernelSignature>
  DeviceKernel<KernelSignature>
  getKernel(const std::shared_ptr<DeviceImage> &Image,
            llvm::StringRef KernelName) const noexcept {
    assert(Image && "Image provided to getKernel is null");

    if (Image->DeviceHandle != DeviceHandle)
      FATAL_ERROR("Image provided to getKernel was created for a different "
                  "device");

    auto KernelHandleOrErr = getKernelImpl(Image->Handle, KernelName);

    if (auto Err = KernelHandleOrErr.takeError())
      FATAL_ERROR(llvm::toString(std::move(Err)));

    return DeviceKernel<KernelSignature>(Image, *KernelHandleOrErr);
  }

  template <typename KernelSignature>
  [[nodiscard]] std::optional<DeviceKernel<KernelSignature>>
  tryGetKernel(const std::shared_ptr<DeviceImage> &Image,
               llvm::StringRef KernelName) const noexcept {
    assert(Image && "Image provided to getKernel is null");

    if (Image->DeviceHandle != DeviceHandle)
      return std::nullopt;

    auto KernelHandleOrErr = getKernelImpl(Image->Handle, KernelName);

    if (auto Err = KernelHandleOrErr.takeError()) {
      llvm::consumeError(std::move(Err));
      return std::nullopt;
    }

    return DeviceKernel<KernelSignature>(Image, *KernelHandleOrErr);
  }

  template <typename KernelSignature, typename... ArgTypes>
  void launchKernel(DeviceKernel<KernelSignature> Kernel, Dim NumGroups,
                    Dim GroupSize, ArgTypes &&...Args) const noexcept {
    using ExpectedTypes =
        typename FunctionTypeTraits<KernelSignature>::ArgTypesTuple;
    using ProvidedTypes = std::tuple<std::decay_t<ArgTypes>...>;

    static_assert(std::is_same_v<ExpectedTypes, ProvidedTypes>,
                  "Argument types provided to launchKernel do not match the "
                  "kernel's signature");

    if (Kernel.Image->DeviceHandle != DeviceHandle)
      FATAL_ERROR("Kernel provided to launchKernel was created for a different "
                  "device");

    if constexpr (sizeof...(Args) == 0) {
      launchKernelImpl(Kernel.Handle, NumGroups, GroupSize, nullptr, 0);
    } else {
      auto KernelArgs = makeKernelArgsPack(std::forward<ArgTypes>(Args)...);

      static_assert(
          (std::is_trivially_copyable_v<std::decay_t<ArgTypes>> && ...),
          "Argument types provided to launchKernel must be trivially copyable");

      launchKernelImpl(Kernel.Handle, NumGroups, GroupSize, &KernelArgs,
                       sizeof(KernelArgs));
    }
  }

  [[nodiscard]] llvm::StringRef getName() const noexcept;

  [[nodiscard]] llvm::StringRef getPlatform() const noexcept;

private:
  [[nodiscard]] llvm::Expected<std::shared_ptr<DeviceImage>>
  loadBinaryImpl(llvm::StringRef Directory, llvm::StringRef BinaryName) const;

  [[nodiscard]] llvm::Expected<ol_symbol_handle_t>
  getKernelImpl(ol_program_handle_t ProgramHandle,
                llvm::StringRef KernelName) const noexcept;

  void launchKernelImpl(ol_symbol_handle_t KernelHandle, const Dim &NumGroups,
                        const Dim &GroupSize, const void *KernelArgs,
                        std::size_t KernelArgsSize) const noexcept;

  std::size_t GlobalDeviceId;
  ol_device_handle_t DeviceHandle;
};
} // namespace mathtest

#endif // MATHTEST_DEVICECONTEXT_HPP
