//===-- InputGen GPU Entry Runtime State ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "inputgen_gpu_entry_internal.h"

#define INPUTGEN_GPU_ENTRY_STATE(Variable, Constant, CType, Symbol)            \
  CType Variable __asm__(Symbol);
#include "llvm/Frontend/Offloading/InputGenGPUABI.def"

int inputgen_entry_random(void) { return 9; }
