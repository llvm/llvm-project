//===-- InputGen GPU Entry Runtime Internals -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INPUTGEN_GPU_ENTRY_INTERNAL_H
#define INPUTGEN_GPU_ENTRY_INTERNAL_H

#include <stdint.h>

#include "InputGenInterface.hpp"
#include "inputgen_gpu_instrumentor_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INPUTGEN_GPU_ENTRY_STATE(Variable, Constant, CType, Symbol)            \
  extern CType Variable __asm__(Symbol);
#include "llvm/Frontend/Offloading/InputGenGPUABI.def"

int inputgen_entry_random(void);

#ifdef __cplusplus
}
#endif

#endif
