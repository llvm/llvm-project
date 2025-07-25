//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of helpers and non-template member
/// functions for the DeviceContext class.
///
//===----------------------------------------------------------------------===//

#include "mathtest/DeviceContext.hpp"

#include "mathtest/ErrorHandling.hpp"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <OffloadAPI.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace mathtest;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

// The static 'Wrapper' instance ensures olInit() is called once at program
// startup and olShutDown() is called once at program termination
struct OffloadInitWrapper {
  OffloadInitWrapper() { OL_CHECK(olInit()); }
  ~OffloadInitWrapper() { OL_CHECK(olShutDown()); }
};
static OffloadInitWrapper Wrapper{};

[[nodiscard]] std::string getDeviceName(ol_device_handle_t DeviceHandle) {
  std::size_t PropSize = 0;
  OL_CHECK(olGetDeviceInfoSize(DeviceHandle, OL_DEVICE_INFO_NAME, &PropSize));

  if (PropSize == 0)
    return "";

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_NAME, PropSize,
                           PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}

[[nodiscard]] ol_platform_handle_t
getDevicePlatform(ol_device_handle_t DeviceHandle) noexcept {
  ol_platform_handle_t PlatformHandle = nullptr;
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                           sizeof(PlatformHandle), &PlatformHandle));
  return PlatformHandle;
}

[[nodiscard]] std::string getPlatformName(ol_platform_handle_t PlatformHandle) {
  std::size_t PropSize = 0;
  OL_CHECK(
      olGetPlatformInfoSize(PlatformHandle, OL_PLATFORM_INFO_NAME, &PropSize));

  if (PropSize == 0)
    return "";

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_NAME, PropSize,
                             PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return llvm::StringRef(PropValue).lower();
}

[[nodiscard]] ol_platform_backend_t
getPlatformBackend(ol_platform_handle_t PlatformHandle) noexcept {
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  OL_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_BACKEND,
                             sizeof(Backend), &Backend));
  return Backend;
}

struct Device {
  ol_device_handle_t Handle;
  std::string Name;
  std::string Platform;
  ol_platform_backend_t Backend;
};

const std::vector<Device> &getDevices() {
  // Thread-safe initialization of a static local variable
  static auto Devices = []() {
    std::vector<Device> TmpDevices;

    // Discovers all devices that are not the host
    const auto *const ResultFromIterate = olIterateDevices(
        [](ol_device_handle_t DeviceHandle, void *Data) {
          ol_platform_handle_t PlatformHandle = getDevicePlatform(DeviceHandle);
          ol_platform_backend_t Backend = getPlatformBackend(PlatformHandle);

          if (Backend != OL_PLATFORM_BACKEND_HOST) {
            auto Name = getDeviceName(DeviceHandle);
            auto Platform = getPlatformName(PlatformHandle);

            static_cast<std::vector<Device> *>(Data)->push_back(
                {DeviceHandle, Name, Platform, Backend});
          }

          return true;
        },
        &TmpDevices);

    OL_CHECK(ResultFromIterate);

    return TmpDevices;
  }();

  return Devices;
}
} // namespace

const llvm::SetVector<llvm::StringRef> &mathtest::getPlatforms() {
  // Thread-safe initialization of a static local variable
  static auto Platforms = []() {
    llvm::SetVector<llvm::StringRef> TmpPlatforms;

    for (const auto &Device : getDevices())
      TmpPlatforms.insert(Device.Platform);

    return TmpPlatforms;
  }();

  return Platforms;
}

void detail::allocManagedMemory(ol_device_handle_t DeviceHandle,
                                std::size_t Size,
                                void **AllocationOut) noexcept {
  OL_CHECK(
      olMemAlloc(DeviceHandle, OL_ALLOC_TYPE_MANAGED, Size, AllocationOut));
}

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

DeviceContext::DeviceContext(std::size_t GlobalDeviceId)
    : GlobalDeviceId(GlobalDeviceId), DeviceHandle(nullptr) {
  const auto &Devices = getDevices();

  if (GlobalDeviceId >= Devices.size())
    FATAL_ERROR("Invalid GlobalDeviceId: " + llvm::Twine(GlobalDeviceId) +
                ", but the number of available devices is " +
                llvm::Twine(Devices.size()));

  DeviceHandle = Devices[GlobalDeviceId].Handle;
}

DeviceContext::DeviceContext(llvm::StringRef Platform, std::size_t DeviceId)
    : DeviceHandle(nullptr) {
  std::string NormalizedPlatform = Platform.lower();
  const auto &Platforms = getPlatforms();

  if (!Platforms.contains(NormalizedPlatform))
    FATAL_ERROR("There is no platform that matches with '" +
                llvm::Twine(Platform) +
                "'. Available platforms are: " + llvm::join(Platforms, ", "));

  const auto &Devices = getDevices();

  std::optional<std::size_t> FoundGlobalDeviceId;
  std::size_t MatchCount = 0;

  for (std::size_t Index = 0; Index < Devices.size(); ++Index) {
    if (Devices[Index].Platform == NormalizedPlatform) {
      if (MatchCount == DeviceId) {
        FoundGlobalDeviceId = Index;
        break;
      }
      MatchCount++;
    }
  }

  if (!FoundGlobalDeviceId)
    FATAL_ERROR("Invalid DeviceId: " + llvm::Twine(DeviceId) +
                ", but the number of available devices on '" + Platform +
                "' is " + llvm::Twine(MatchCount));

  GlobalDeviceId = *FoundGlobalDeviceId;
  DeviceHandle = Devices[GlobalDeviceId].Handle;
}

[[nodiscard]] llvm::Expected<std::shared_ptr<DeviceImage>>
DeviceContext::loadBinaryImpl(llvm::StringRef Directory,
                              llvm::StringRef BinaryName) const {
  auto Backend = getDevices()[GlobalDeviceId].Backend;
  llvm::StringRef Extension;

  switch (Backend) {
  case OL_PLATFORM_BACKEND_AMDGPU:
    Extension = ".amdgpu.bin";
    break;
  case OL_PLATFORM_BACKEND_CUDA:
    Extension = ".nvptx64.bin";
    break;
  default:
    llvm_unreachable("Unsupported backend to infer binary extension");
  }

  llvm::SmallString<128> FullPath(Directory);
  llvm::sys::path::append(FullPath, llvm::Twine(BinaryName) + Extension);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(FullPath);

  if (std::error_code ErrorCode = FileOrErr.getError())
    return llvm::errorCodeToError(ErrorCode);

  std::unique_ptr<llvm::MemoryBuffer> &BinaryData = *FileOrErr;

  ol_program_handle_t ProgramHandle = nullptr;
  const ol_result_t OlResult =
      olCreateProgram(DeviceHandle, BinaryData->getBufferStart(),
                      BinaryData->getBufferSize(), &ProgramHandle);

  if (OlResult != OL_SUCCESS) {
    llvm::StringRef Details =
        OlResult->Details ? OlResult->Details : "No details provided";

    return llvm::createStringError(llvm::Twine(Details) + " (Code " +
                                   llvm::Twine(OlResult->Code) + ")");
  }

  return std::shared_ptr<DeviceImage>(
      new DeviceImage(DeviceHandle, ProgramHandle));
}

[[nodiscard]] std::shared_ptr<DeviceImage>
DeviceContext::loadBinary(llvm::StringRef Directory,
                          llvm::StringRef BinaryName) const {
  auto ImageOrErr = loadBinaryImpl(Directory, BinaryName);

  if (auto Err = ImageOrErr.takeError())
    FATAL_ERROR(llvm::toString(std::move(Err)));

  return std::move(*ImageOrErr);
}

[[nodiscard]] std::optional<std::shared_ptr<DeviceImage>>
DeviceContext::tryLoadBinary(llvm::StringRef Directory,
                             llvm::StringRef BinaryName) const {
  auto ImageOrErr = loadBinaryImpl(Directory, BinaryName);

  if (auto Err = ImageOrErr.takeError()) {
    llvm::consumeError(std::move(Err));
    return std::nullopt;
  }

  return std::move(*ImageOrErr);
}

[[nodiscard]] llvm::Expected<ol_symbol_handle_t>
DeviceContext::getKernelImpl(ol_program_handle_t ProgramHandle,
                             llvm::StringRef KernelName) const noexcept {
  ol_symbol_handle_t KernelHandle = nullptr;
  llvm::SmallString<32> KernelNameBuffer(KernelName);

  const ol_result_t OlResult =
      olGetSymbol(ProgramHandle, KernelNameBuffer.c_str(),
                  OL_SYMBOL_KIND_KERNEL, &KernelHandle);

  if (OlResult != OL_SUCCESS) {
    llvm::StringRef Details =
        OlResult->Details ? OlResult->Details : "No details provided";

    return llvm::createStringError(llvm::Twine(Details) + " (Code " +
                                   llvm::Twine(OlResult->Code) + ")");
  }

  return KernelHandle;
}

void DeviceContext::launchKernelImpl(
    ol_symbol_handle_t KernelHandle, const Dim &NumGroups, const Dim &GroupSize,
    const void *KernelArgs, std::size_t KernelArgsSize) const noexcept {
  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = 3; // It seems this field is not used anywhere.
                             // Defaulting to the safest value.
  LaunchArgs.NumGroups = {NumGroups[0], NumGroups[1], NumGroups[2]};
  LaunchArgs.GroupSize = {GroupSize[0], GroupSize[1], GroupSize[2]};
  LaunchArgs.DynSharedMemory = 0;

  OL_CHECK(olLaunchKernel(nullptr, DeviceHandle, KernelHandle, KernelArgs,
                          KernelArgsSize, &LaunchArgs, nullptr));
}

[[nodiscard]] llvm::StringRef DeviceContext::getName() const noexcept {
  return getDevices()[GlobalDeviceId].Name;
}

[[nodiscard]] llvm::StringRef DeviceContext::getPlatform() const noexcept {
  return getDevices()[GlobalDeviceId].Platform;
}
