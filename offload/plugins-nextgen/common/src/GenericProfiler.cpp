//===- GenericProfiler.cpp - GenericProfiler implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "GenericProfiler.h"
#include "PluginInterface.h"

#include <cstdint>
#include <memory>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

uint64_t GenericProfilerTy::getDeviceTimeStamp(GenericDeviceTy *D) {
  if (!D)
    return 0;

  return D->getDeviceTimeStamp();
}

GenericProfilerTy &getNoOpProfiler() {
  static GenericProfilerTy NoOpProfiler;
  return NoOpProfiler;
}
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
