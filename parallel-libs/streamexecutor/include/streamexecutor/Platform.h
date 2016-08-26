//===-- Platform.h - The Platform class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Platform class which represents a platform such as CUDA or OpenCL.
///
/// This is an abstract base class that will be overridden by each specific
/// platform.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORM_H
#define STREAMEXECUTOR_PLATFORM_H

#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

class Device;

class Platform {
public:
  virtual ~Platform();

  /// Gets the number of devices available for this platform.
  virtual size_t getDeviceCount() const = 0;

  /// Gets a pointer to a Device with the given index for this platform.
  ///
  /// Ownership of the Device instance is NOT transferred to the caller.
  virtual Expected<Device *> getDevice(size_t DeviceIndex) = 0;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORM_H
