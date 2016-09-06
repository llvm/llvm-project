//===-- SimpleHostPlatformDevice.h - Host device for testing ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The SimpleHostPlatformDevice class is a streamexecutor::PlatformDevice that
/// is really just the host processor and memory. It is useful for testing
/// because no extra device platform is required.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_LIB_UNITTESTS_SIMPLEHOSTPLATFORMDEVICE_H
#define STREAMEXECUTOR_LIB_UNITTESTS_SIMPLEHOSTPLATFORMDEVICE_H

#include <cstdlib>
#include <cstring>

#include "streamexecutor/PlatformInterfaces.h"

namespace streamexecutor {
namespace test {

/// A streamexecutor::PlatformDevice that simply forwards all operations to the
/// host platform.
///
/// The allocate and copy methods are simple wrappers for std::malloc and
/// std::memcpy.
class SimpleHostPlatformDevice : public streamexecutor::PlatformDevice {
public:
  std::string getName() const override { return "SimpleHostPlatformDevice"; }

  streamexecutor::Expected<const void *> createStream() override {
    return nullptr;
  }

  streamexecutor::Expected<void *>
  allocateDeviceMemory(size_t ByteCount) override {
    return std::malloc(ByteCount);
  }

  streamexecutor::Error freeDeviceMemory(const void *Handle) override {
    std::free(const_cast<void *>(Handle));
    return streamexecutor::Error::success();
  }

  streamexecutor::Expected<void *>
  allocateHostMemory(size_t ByteCount) override {
    return std::malloc(ByteCount);
  }

  streamexecutor::Error freeHostMemory(void *Memory) override {
    std::free(const_cast<void *>(Memory));
    return streamexecutor::Error::success();
  }

  streamexecutor::Error registerHostMemory(void *Memory,
                                           size_t ByteCount) override {
    return streamexecutor::Error::success();
  }

  streamexecutor::Error unregisterHostMemory(void *Memory) override {
    return streamexecutor::Error::success();
  }

  streamexecutor::Error copyD2H(const void *StreamHandle,
                                const void *DeviceHandleSrc,
                                size_t SrcByteOffset, void *HostDst,
                                size_t DstByteOffset,
                                size_t ByteCount) override {
    std::memcpy(static_cast<char *>(HostDst) + DstByteOffset,
                static_cast<const char *>(DeviceHandleSrc) + SrcByteOffset,
                ByteCount);
    return streamexecutor::Error::success();
  }

  streamexecutor::Error copyH2D(const void *StreamHandle, const void *HostSrc,
                                size_t SrcByteOffset,
                                const void *DeviceHandleDst,
                                size_t DstByteOffset,
                                size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceHandleDst)) +
                    DstByteOffset,
                static_cast<const char *>(HostSrc) + SrcByteOffset, ByteCount);
    return streamexecutor::Error::success();
  }

  streamexecutor::Error
  copyD2D(const void *StreamHandle, const void *DeviceHandleSrc,
          size_t SrcByteOffset, const void *DeviceHandleDst,
          size_t DstByteOffset, size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceHandleDst)) +
                    DstByteOffset,
                static_cast<const char *>(DeviceHandleSrc) + SrcByteOffset,
                ByteCount);
    return streamexecutor::Error::success();
  }

  streamexecutor::Error synchronousCopyD2H(const void *DeviceHandleSrc,
                                           size_t SrcByteOffset, void *HostDst,
                                           size_t DstByteOffset,
                                           size_t ByteCount) override {
    std::memcpy(static_cast<char *>(HostDst) + DstByteOffset,
                static_cast<const char *>(DeviceHandleSrc) + SrcByteOffset,
                ByteCount);
    return streamexecutor::Error::success();
  }

  streamexecutor::Error synchronousCopyH2D(const void *HostSrc,
                                           size_t SrcByteOffset,
                                           const void *DeviceHandleDst,
                                           size_t DstByteOffset,
                                           size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceHandleDst)) +
                    DstByteOffset,
                static_cast<const char *>(HostSrc) + SrcByteOffset, ByteCount);
    return streamexecutor::Error::success();
  }

  streamexecutor::Error synchronousCopyD2D(const void *DeviceHandleSrc,
                                           size_t SrcByteOffset,
                                           const void *DeviceHandleDst,
                                           size_t DstByteOffset,
                                           size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceHandleDst)) +
                    DstByteOffset,
                static_cast<const char *>(DeviceHandleSrc) + SrcByteOffset,
                ByteCount);
    return streamexecutor::Error::success();
  }

  /// Gets the value at the given index from a GlobalDeviceMemory<T> instance
  /// created by this class.
  template <typename T>
  static T getDeviceValue(const streamexecutor::GlobalDeviceMemory<T> &Memory,
                          size_t Index) {
    return static_cast<const T *>(Memory.getHandle())[Index];
  }
};

} // namespace test
} // namespace streamexecutor

#endif // STREAMEXECUTOR_LIB_UNITTESTS_SIMPLEHOSTPLATFORMDEVICE_H
