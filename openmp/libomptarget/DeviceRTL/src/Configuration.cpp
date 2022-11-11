//===- Configuration.cpp - OpenMP device configuration interface -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data object of the constant device environment and the
// query API.
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "DeviceEnvironment.h"
#include "State.h"
#include "Types.h"

using namespace _OMP;

#pragma omp begin declare target device_type(nohost)

// defined by CGOpenMPRuntimeGPU
extern uint32_t __omp_rtl_debug_kind;
extern uint32_t __omp_rtl_assume_no_thread_state;
extern uint32_t __omp_rtl_assume_no_nested_parallelism;

// TODO: We want to change the name as soon as the old runtime is gone.
// This variable should be visibile to the plugin so we override the default
// hidden visibility.
DeviceEnvironmentTy CONSTANT(omptarget_device_environment)
    __attribute__((used, retain, weak, visibility("protected")));

uint32_t config::getDebugKind() {
  return __omp_rtl_debug_kind & omptarget_device_environment.DebugKind;
}

uint32_t config::getNumDevices() {
  return omptarget_device_environment.NumDevices;
}

uint32_t config::getDeviceNum() {
  return omptarget_device_environment.DeviceNum;
}

uint64_t config::getDynamicMemorySize() {
  return omptarget_device_environment.DynamicMemSize;
}

bool config::isDebugMode(config::DebugKind Kind) {
  return config::getDebugKind() & Kind;
}

bool config::mayUseThreadStates() { return !__omp_rtl_assume_no_thread_state; }

bool config::mayUseNestedParallelism() {
  return !__omp_rtl_assume_no_nested_parallelism;
}

#pragma omp end declare target
