//===- SharedCL.h - tblgen command line arguments -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_SHARED_CL_H_
#define MLIR_TOOLS_MLIRTBLGEN_SHARED_CL_H_

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace tblgen {

extern llvm::cl::opt<bool> clUseFallbackTypeIDs;

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_SHARED_CL_H_
