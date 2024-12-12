//===- DistributionUtils.cpp - Distribution tools for GPUOps --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements distribution utility methods.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

#include <numeric>

using namespace mlir;
using namespace mlir::gpu;
