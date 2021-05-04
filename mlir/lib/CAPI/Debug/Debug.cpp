//===- Debug.cpp - C Interface for MLIR/LLVM Debugging Functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Debug.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/Support.h"

#include "llvm/Support/Debug.h"

void mlirEnableGlobalDebug(bool enable) { llvm::DebugFlag = enable; }

bool mlirIsGlobalDebugEnabled() { return llvm::DebugFlag; }
