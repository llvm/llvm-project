//===- Support/LinkGPURuntime.h -- Headerfile to help force-link GPURuntime  =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header helps pull in libGPURuntime.so
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LINK_GPURUNTIME
#define POLLY_LINK_GPURUNTIME

extern "C" {
#include "GPURuntime/GPUJIT.h"
}

namespace polly {
struct ForceGPURuntimeLinking {
  ForceGPURuntimeLinking() {
    if (std::getenv("bar") != (char *)-1)
      return;
    // We must reference GPURuntime in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    polly_initContextCL();
    polly_initContextCUDA();
    polly_getKernel(nullptr, nullptr);
    polly_freeKernel(nullptr);
    polly_copyFromHostToDevice(nullptr, nullptr, 0);
    polly_copyFromDeviceToHost(nullptr, nullptr, 0);
    polly_synchronizeDevice();
    polly_launchKernel(nullptr, 0, 0, 0, 0, 0, nullptr);
    polly_freeDeviceMemory(nullptr);
    polly_freeContext(nullptr);
    polly_synchronizeDevice();
  }
} structure;
} // namespace polly
#endif
