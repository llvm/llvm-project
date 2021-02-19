//===--- print_tracing.h - OpenMP interface definitions -------- AMDGPU -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBOMPTARGET_PLUGINS_AMGGPU_SRC_PRINT_TRACING_H_INCLUDED
#define LIBOMPTARGET_PLUGINS_AMGGPU_SRC_PRINT_TRACING_H_INCLUDED

enum PrintTraceControlBits {
  LAUNCH = 1,          // print a message to stderr for each kernel launch
  RTL_TIMING = 2,      // Print timing info around each RTL step
  STARTUP_DETAILS = 4, // Details around loading up kernel
  RTL_TO_STDOUT = 8    // Redirect RTL tracing to stdout
};

extern int print_kernel_trace; // set by environment variable

#endif
