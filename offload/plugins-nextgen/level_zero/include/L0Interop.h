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
  ze_command_list_handle_t ImmCmdList;
};

} // namespace llvm::omp::target::plugin::L0Interop

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0INTEROP_H
