//===-- InputGenInterface.hpp - InputGen GPU offload ABI ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_INPUTGENINTERFACE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_INPUTGENINTERFACE_H

#include <stdint.h>

enum {
#define INPUTGEN_GPU_ABI_MODE(Name, Value) Name = Value,
#include "llvm/Frontend/Offloading/InputGenGPUABI.def"
};

#ifdef __cplusplus
namespace llvm::omp::target::plugin::inputgen {
#define INPUTGEN_GPU_ENTRY_STATE(Variable, Constant, CType, Symbol)            \
  inline constexpr char Constant[] = Symbol;
#include "llvm/Frontend/Offloading/InputGenGPUABI.def"
} // namespace llvm::omp::target::plugin::inputgen
#endif

#endif
