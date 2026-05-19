//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interop support for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0INTEROP_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0INTEROP_H

namespace llvm::omp::target::plugin::L0Interop {

/// Level Zero interop property.
struct Property {
  // Use this when command queue needs to be accessed as
  // the targetsync field in interop will be changed if preferred type is sycl.
  ze_command_queue_handle_t CommandQueue;
  ze_command_list_handle_t ImmCmdList;
};

} // namespace llvm::omp::target::plugin::L0Interop

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0INTEROP_H
