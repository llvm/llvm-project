//===- Rewrite.h - C API Utils for Core MLIR classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// rewrite patterns. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_REWRITE_H
#define MLIR_CAPI_REWRITE_H

#include "mlir-c/Rewrite.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/PatternMatch.h"

DEFINE_C_API_PTR_METHODS(MlirRewriterBase, mlir::RewriterBase)
DEFINE_C_API_PTR_METHODS(MlirRewritePattern, const mlir::RewritePattern)
DEFINE_C_API_PTR_METHODS(MlirRewritePatternSet, mlir::RewritePatternSet)

#endif // MLIR_CAPIREWRITER_H
