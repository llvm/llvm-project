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
#include "mlir/IR/SymbolTable.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.h.inc"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.h.inc"

#define ACC_DATA_ENTRY_OPS                                                     \
  mlir::acc::CopyinOp, mlir::acc::CreateOp, mlir::acc::PresentOp,              \
      mlir::acc::NoCreateOp, mlir::acc::AttachOp, mlir::acc::DevicePtrOp,      \
      mlir::acc::GetDevicePtrOp, mlir::acc::PrivateOp,                         \
      mlir::acc::FirstprivateOp, mlir::acc::UpdateDeviceOp,                    \
      mlir::acc::UseDeviceOp, mlir::acc::ReductionOp,                          \
      mlir::acc::DeclareDeviceResidentOp, mlir::acc::DeclareLinkOp
#define ACC_COMPUTE_CONSTRUCT_OPS                                              \
  mlir::acc::ParallelOp, mlir::acc::KernelsOp, mlir::acc::SerialOp
#define ACC_DATA_CONSTRUCT_OPS                                                 \
  mlir::acc::DataOp, mlir::acc::EnterDataOp, mlir::acc::ExitDataOp,            \
      mlir::acc::UpdateOp, mlir::acc::HostDataOp, mlir::acc::DeclareEnterOp,   \
      mlir::acc::DeclareExitOp, mlir::acc::DeclareOp
#define ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS                                     \
  ACC_COMPUTE_CONSTRUCT_OPS, ACC_DATA_CONSTRUCT_OPS

namespace mlir {
namespace acc {

/// Enumeration used to encode the execution mapping on a loop construct.
/// They refer directly to the OpenACC 3.3 standard:
/// 2.9.2. gang
/// 2.9.3. worker
/// 2.9.4. vector
///
/// Value can be combined bitwise to reflect the mapping applied to the
/// construct. e.g. `acc.loop gang vector`, the `gang` and `vector` could be
/// combined and the final mapping value would be 5 (4 | 1).
enum OpenACCExecMapping { NONE = 0, VECTOR = 1, WORKER = 2, GANG = 4 };

/// Used to obtain the `varPtr` from a data entry operation.
/// Returns empty value if not a data entry operation.
mlir::Value getVarPtr(mlir::Operation *accDataEntryOp);

/// Used to obtain the `dataClause` from a data entry operation.
/// Returns empty optional if not a data entry operation.
std::optional<mlir::acc::DataClause>
getDataClause(mlir::Operation *accDataEntryOp);

/// Used to obtain the attribute name for declare.
static constexpr StringLiteral getDeclareAttrName() {
  return StringLiteral("acc.declare");
}

static constexpr StringLiteral getDeclareActionAttrName() {
  return StringLiteral("acc.declare_action");
}

static constexpr StringLiteral getRoutineInfoAttrName() {
  return StringLiteral("acc.routine_info");
}

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACC_H_
