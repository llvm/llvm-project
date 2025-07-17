#include "mathtest/DeviceContext.hpp"

#include "mathtest/ErrorHandling.hpp"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <OffloadAPI.h>
#include <cstddef>
#include <memory>
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

[[nodiscard]] ol_platform_backend_t
getBackend(ol_device_handle_t DeviceHandle) noexcept {
  ol_platform_handle_t Platform;
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                           sizeof(Platform), &Platform));
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  OL_CHECK(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                             sizeof(Backend), &Backend));
  return Backend;
}

const std::vector<ol_device_handle_t> &getDevices() {
  // Thread-safe initialization of a static local variable
  static std::vector<ol_device_handle_t> Devices =
      []() -> std::vector<ol_device_handle_t> {
    std::vector<ol_device_handle_t> TmpDevices;

    // Discovers all devices that are not the host
    const auto *const ResultFromIterate = olIterateDevices(
        [](ol_device_handle_t DeviceHandle, void *Data) {
          if (getBackend(DeviceHandle) != OL_PLATFORM_BACKEND_HOST) {
            static_cast<std::vector<ol_device_handle_t> *>(Data)->push_back(
                DeviceHandle);
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

std::size_t mathtest::countDevices() { return getDevices().size(); }

void detail::allocManagedMemory(ol_device_handle_t DeviceHandle,
                                std::size_t Size,
                                void **AllocationOut) noexcept {
  OL_CHECK(
      olMemAlloc(DeviceHandle, OL_ALLOC_TYPE_MANAGED, Size, AllocationOut));
}

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

DeviceContext::DeviceContext(std::size_t DeviceId)
    : DeviceId(DeviceId), DeviceHandle(nullptr) {
  const auto &Devices = getDevices();

  if (DeviceId >= Devices.size()) {
    FATAL_ERROR("Invalid DeviceId: " + llvm::Twine(DeviceId) + ", but only " +
                llvm::Twine(Devices.size()) + " devices are available");
  }

  DeviceHandle = Devices[DeviceId];
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
  if (std::error_code ErrorCode = FileOrErr.getError()) {
    FATAL_ERROR(llvm::Twine("Failed to read device binary file '") + FullPath +
                "': " + ErrorCode.message());
  }
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
  llvm::StringRef Extension;

  switch (getBackend(DeviceHandle)) {
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

[[nodiscard]] std::string DeviceContext::getName() const {
  std::size_t PropSize = 0;
  OL_CHECK(olGetDeviceInfoSize(DeviceHandle, OL_DEVICE_INFO_NAME, &PropSize));

  if (PropSize == 0) {
    return "";
  }

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_NAME, PropSize,
                           PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}

[[nodiscard]] std::string DeviceContext::getPlatform() const {
  ol_platform_handle_t PlatformHandle = nullptr;
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                           sizeof(ol_platform_handle_t), &PlatformHandle));

  std::size_t PropSize = 0;
  OL_CHECK(
      olGetPlatformInfoSize(PlatformHandle, OL_PLATFORM_INFO_NAME, &PropSize));

  if (PropSize == 0) {
    return "";
  }

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_NAME, PropSize,
                             PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}
