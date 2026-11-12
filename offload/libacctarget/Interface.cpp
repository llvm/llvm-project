//===- Interface.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by compilers to interact with the
// runtime. Signatures must match
// mlir/include/mlir/Dialect/OpenACC/OpenACCRuntimeFunctions.def.
//
//===----------------------------------------------------------------------===//

#include "Interface.h"
#include "RuntimeState.h"

EXTERN void __tgt_acc_rtl_init() { initAccRuntime(); }

EXTERN void __tgt_acc_rtl_deinit() { deinitAccRuntime(); }
