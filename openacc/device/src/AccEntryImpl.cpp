//===- AccEntryImpl.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openacc.h"

extern "C" {
#ifdef __CUDA_ARCH__
#pragma omp declare target
int acc_on_device(acc_device_t DevType) {
  if (DevType == acc_device_not_host || DevType == acc_device_nvidia) {
    return true;
  }
  return false;
}
int acc_on_device_(acc_device_t *DevType) { return acc_on_device(*DevType); }
#pragma omp end declare target
#endif
}
