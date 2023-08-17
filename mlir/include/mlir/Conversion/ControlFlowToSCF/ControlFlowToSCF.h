//===- ControlFlowToSCF.h - ControlFlow to SCF -------------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H
#define MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_LIFTCONTROLFLOWTOSCFPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_CONTROLFLOWTOSCF_CONTROLFLOWTOSCF_H
