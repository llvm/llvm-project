//===- TranslationRegistration.h - Register translation ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the registration function for the IRDL to C++ translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_IRDLTOCPP_IRDLTOCPP_H
#define MLIR_TARGET_IRDLTOCPP_IRDLTOCPP_H

#include "mlir/Dialect/IRDL/IR/IRDL.h"

namespace mlir {
namespace irdl {

/// Translates an IRDL dialect definition to a C++ definition that can be used
/// with MLIR.
///
/// As with TableGen declarations, the following preprocessor macros will
/// generate the corresponding code:
///
///  // This define generates code for the dialect's class declarations
///  #define GEN_DIALECT_DECL_HEADER
///
///  // This define generates code for the dialect's class definitions
///  #define GEN_DIALECT_DEF
LogicalResult translateIRDLDialectToCpp(irdl::DialectOp dialect,
                                        raw_ostream &output);

} // namespace irdl
} // namespace mlir

#endif // MLIR_TARGET_IRDLTOCPP_IRDLTOCPP_H
