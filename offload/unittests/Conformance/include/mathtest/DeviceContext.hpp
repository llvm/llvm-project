#pragma once

#include "mathtest/DeviceResources.hpp"
#include "mathtest/Dim.hpp"
#include "mathtest/ErrorHandling.hpp"
#include "mathtest/Support.hpp"

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mathtest {

std::size_t countDevices();

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
  // TODO: Add a constructor that also takes a 'Provider'.
  explicit DeviceContext(std::size_t DeviceId = 0);

  template <typename T>
  ManagedBuffer<T> createManagedBuffer(std::size_t Size) const noexcept {
    void *UntypedAddress = nullptr;

    detail::allocManagedMemory(DeviceHandle, Size * sizeof(T), &UntypedAddress);
    T *TypedAddress = static_cast<T *>(UntypedAddress);

    return ManagedBuffer<T>(TypedAddress, Size);
  }

  [[nodiscard]] std::shared_ptr<DeviceImage>
  loadBinary(llvm::StringRef Directory, llvm::StringRef BinaryName,
             llvm::StringRef Extension) const;

  [[nodiscard]] std::shared_ptr<DeviceImage>
  loadBinary(llvm::StringRef Directory, llvm::StringRef BinaryName) const;

  template <typename KernelSignature>
  DeviceKernel<KernelSignature>
  getKernel(const std::shared_ptr<DeviceImage> &Image,
            llvm::StringRef KernelName) const noexcept {
    assert(Image && "Image provided to getKernel is null");

    if (Image->DeviceHandle != this->DeviceHandle) {
      FATAL_ERROR("Image provided to getKernel was created for a different "
                  "device");
    }

    ol_symbol_handle_t KernelHandle = nullptr;
    getKernelImpl(Image->Handle, KernelName, &KernelHandle);

    return DeviceKernel<KernelSignature>(Image, KernelHandle);
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

    if (Kernel.Image->DeviceHandle != DeviceHandle) {
      FATAL_ERROR("Kernel provided to launchKernel was created for a different "
                  "device");
    }

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

  [[nodiscard]] std::size_t getId() const noexcept { return DeviceId; }

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] std::string getPlatform() const;

private:
  void getKernelImpl(ol_program_handle_t ProgramHandle,
                     llvm::StringRef KernelName,
                     ol_symbol_handle_t *KernelHandle) const noexcept;

  void launchKernelImpl(ol_symbol_handle_t KernelHandle, const Dim &NumGroups,
                        const Dim &GroupSize, const void *KernelArgs,
                        std::size_t KernelArgsSize) const noexcept;

  std::size_t DeviceId;
  ol_device_handle_t DeviceHandle;
};
} // namespace mathtest
