#include "mathtest/DeviceResources.hpp"

#include "mathtest/ErrorHandling.hpp"

#include <OffloadAPI.h>

using namespace mathtest;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void detail::freeDeviceMemory(void *Address) noexcept {
  if (Address) {
    OL_CHECK(olMemFree(Address));
  }
}

//===----------------------------------------------------------------------===//
// DeviceImage
//===----------------------------------------------------------------------===//

DeviceImage::~DeviceImage() noexcept {
  if (Handle) {
    OL_CHECK(olDestroyProgram(Handle));
  }
}

DeviceImage &DeviceImage::operator=(DeviceImage &&Other) noexcept {
  if (this == &Other)
    return *this;

  if (Handle) {
    OL_CHECK(olDestroyProgram(Handle));
  }

  DeviceHandle = Other.DeviceHandle;
  Handle = Other.Handle;

  Other.DeviceHandle = nullptr;
  Other.Handle = nullptr;

  return *this;
}

DeviceImage::DeviceImage(DeviceImage &&Other) noexcept
    : DeviceHandle(Other.DeviceHandle), Handle(Other.Handle) {
  Other.DeviceHandle = nullptr;
  Other.Handle = nullptr;
}

DeviceImage::DeviceImage(ol_device_handle_t DeviceHandle,
                         ol_program_handle_t Handle) noexcept
    : DeviceHandle(DeviceHandle), Handle(Handle) {}
