//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_SCFBUFFERIZEPASS
#define GEN_PASS_DECL_SCFFORLOOPCANONICALIZATIONPASS
#define GEN_PASS_DECL_SCFFORLOOPPEELINGPASS
#define GEN_PASS_DECL_SCFFORLOOPSPECIALIZATIONPASS
#define GEN_PASS_DECL_SCFPARALLELLOOPFUSIONPASS
#define GEN_PASS_DECL_SCFPARALLELLOOPCOLLAPSINGPASS
#define GEN_PASS_DECL_SCFPARALLELLOOPSPECIALIZATIONPASS
#define GEN_PASS_DECL_SCFPARALLELLOOPTILINGPASS
#define GEN_PASS_DECL_SCFFORLOOPRANGEFOLDINGPASS
#define GEN_PASS_DECL_SCFFORTOWHILELOOPPASS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_
