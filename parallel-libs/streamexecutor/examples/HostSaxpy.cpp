//===-- HostSaxpy.cpp - Example of host saxpy with StreamExecutor API -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains example code demonstrating the usage of the
/// StreamExecutor API for a host platform.
///
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include "streamexecutor/StreamExecutor.h"

void Saxpy(float A, float *X, float *Y, size_t N) {
  for (size_t I = 0; I < N; ++I)
    X[I] = A * X[I] + Y[I];
}

namespace __compilergen {
using SaxpyKernel =
    streamexecutor::Kernel<float, streamexecutor::GlobalDeviceMemory<float>,
                           streamexecutor::GlobalDeviceMemory<float>, size_t>;

// Wrapper function converts argument addresses to arguments.
void SaxpyWrapper(const void *const *ArgumentAddresses) {
  Saxpy(*static_cast<const float *>(ArgumentAddresses[0]),
        static_cast<float *>(const_cast<void *>(ArgumentAddresses[1])),
        static_cast<float *>(const_cast<void *>(ArgumentAddresses[2])),
        *static_cast<const size_t *>(ArgumentAddresses[3]));
}

// The wrapper function is what gets registered.
static streamexecutor::MultiKernelLoaderSpec SaxpyLoaderSpec = []() {
  streamexecutor::MultiKernelLoaderSpec Spec;
  Spec.addHostFunction("Saxpy", SaxpyWrapper);
  return Spec;
}();
} // namespace __compilergen

int main() {
  namespace se = ::streamexecutor;
  namespace cg = ::__compilergen;

  // Create some host data.
  float A = 42.0f;
  std::vector<float> HostX = {0, 1, 2, 3};
  std::vector<float> HostY = {4, 5, 6, 7};
  size_t ArraySize = HostX.size();

  // Get a device object.
  se::Platform *Platform =
      getOrDie(se::PlatformManager::getPlatformByName("host"));
  if (Platform->getDeviceCount() == 0) {
    return EXIT_FAILURE;
  }
  se::Device *Device = getOrDie(Platform->getDevice(0));

  // Load the kernel onto the device.
  cg::SaxpyKernel Kernel =
      getOrDie(Device->createKernel<cg::SaxpyKernel>(cg::SaxpyLoaderSpec));

  se::RegisteredHostMemory<float> RegisteredX =
      getOrDie(Device->registerHostMemory<float>(HostX));
  se::RegisteredHostMemory<float> RegisteredY =
      getOrDie(Device->registerHostMemory<float>(HostY));

  // Allocate memory on the device.
  se::GlobalDeviceMemory<float> X =
      getOrDie(Device->allocateDeviceMemory<float>(ArraySize));
  se::GlobalDeviceMemory<float> Y =
      getOrDie(Device->allocateDeviceMemory<float>(ArraySize));

  // Run operations on a stream.
  se::Stream Stream = getOrDie(Device->createStream());
  Stream.thenCopyH2D(RegisteredX, X)
      .thenCopyH2D(RegisteredY, Y)
      .thenLaunch(1, 1, Kernel, A, X, Y, ArraySize)
      .thenCopyD2H(X, RegisteredX);
  // Wait for the stream to complete.
  se::dieIfError(Stream.blockHostUntilDone());

  // Process output data in HostX.
  std::vector<float> ExpectedX = {4, 47, 90, 133};
  assert(std::equal(ExpectedX.begin(), ExpectedX.end(), HostX.begin()));
}
