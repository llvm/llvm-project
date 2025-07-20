#include "mathtest/DeviceContext.hpp"

#include "mathtest/ErrorHandling.hpp"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
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

  if (!FoundGlobalDeviceId.has_value())
    FATAL_ERROR("Invalid DeviceId: " + llvm::Twine(DeviceId) +
                ", but the number of available devices on '" + Platform +
                "' is " + llvm::Twine(MatchCount));

  GlobalDeviceId = FoundGlobalDeviceId.value();
  DeviceHandle = Devices[GlobalDeviceId].Handle;
}

[[nodiscard]] std::shared_ptr<DeviceImage>
DeviceContext::loadBinary(llvm::StringRef Directory, llvm::StringRef BinaryName,
                          llvm::StringRef Extension) const {
  llvm::SmallString<128> FullPath(Directory);
  llvm::sys::path::append(FullPath, llvm::Twine(BinaryName) + Extension);

  // For simplicity, this implementation intentionally reads the binary from
  // disk on every call.
  //
  // Other use cases could benefit from a global, thread-safe cache to avoid
  // redundant file I/O and GPU program creation.

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(FullPath);
  if (std::error_code ErrorCode = FileOrErr.getError())
    FATAL_ERROR(llvm::Twine("Failed to read device binary file '") + FullPath +
                "': " + ErrorCode.message());

  std::unique_ptr<llvm::MemoryBuffer> &BinaryData = *FileOrErr;

  ol_program_handle_t ProgramHandle = nullptr;
  OL_CHECK(olCreateProgram(DeviceHandle, BinaryData->getBufferStart(),
                           BinaryData->getBufferSize(), &ProgramHandle));

  return std::shared_ptr<DeviceImage>(
      new DeviceImage(DeviceHandle, ProgramHandle));
}

[[nodiscard]] std::shared_ptr<DeviceImage>
DeviceContext::loadBinary(llvm::StringRef Directory,
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

  return loadBinary(Directory, BinaryName, Extension);
}

void DeviceContext::getKernelImpl(
    ol_program_handle_t ProgramHandle, llvm::StringRef KernelName,
    ol_symbol_handle_t *KernelHandle) const noexcept {
  llvm::SmallString<32> KernelNameBuffer(KernelName);
  OL_CHECK(olGetSymbol(ProgramHandle, KernelNameBuffer.c_str(),
                       OL_SYMBOL_KIND_KERNEL, KernelHandle));
}

void DeviceContext::launchKernelImpl(
    ol_symbol_handle_t KernelHandle, const Dim &NumGroups, const Dim &GroupSize,
    const void *KernelArgs, std::size_t KernelArgsSize) const noexcept {
  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = 3; // It seems this field is not used anywhere.
                             // Defaulting to the safest value
  LaunchArgs.NumGroups = {NumGroups[0], NumGroups[1], NumGroups[2]};
  LaunchArgs.GroupSize = {GroupSize[0], GroupSize[1], GroupSize[2]};
  LaunchArgs.DynSharedMemory = 0;

  OL_CHECK(olLaunchKernel(nullptr, DeviceHandle, KernelHandle, KernelArgs,
                          KernelArgsSize, &LaunchArgs, nullptr));
}

[[nodiscard]] llvm::StringRef DeviceContext::getName() const {
  return getDevices()[GlobalDeviceId].Name;
}

[[nodiscard]] llvm::StringRef DeviceContext::getPlatform() const {
  return getDevices()[GlobalDeviceId].Platform;
}
