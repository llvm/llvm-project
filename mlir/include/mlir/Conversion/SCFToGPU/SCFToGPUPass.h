//===- SCFToGPUPass.h - Pass converting loops to GPU kernels ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
#define MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
template <typename T>
class InterfacePass;
class Pass;

#define GEN_PASS_DECL_CONVERTAFFINEFORTOGPUPASS
#define GEN_PASS_DECL_CONVERTPARALLELLOOPTOGPUPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
