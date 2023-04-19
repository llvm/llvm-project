//===- OpenACC.h - MLIR OpenACC Dialect -------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ============================================================================
//
// This file declares the OpenACC dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACC_H_
#define MLIR_DIALECT_OPENACC_OPENACC_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.h.inc"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.h.inc"

namespace mlir {
namespace acc {

/// Enumeration used to encode the execution mapping on a loop construct.
/// They refer directly to the OpenACC 3.1 standard:
/// 2.9.2. gang
/// 2.9.3. worker
/// 2.9.4. vector
///
/// Value can be combined bitwise to reflect the mapping applied to the
/// construct. e.g. `acc.loop gang vector`, the `gang` and `vector` could be
/// combined and the final mapping value would be 5 (4 | 1).
enum OpenACCExecMapping { NONE = 0, VECTOR = 1, WORKER = 2, GANG = 4 };

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACC_H_
