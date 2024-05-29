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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.h.inc"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.h.inc"

#include "mlir/Dialect/OpenACC/OpenACCInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.h.inc"

#define ACC_DATA_ENTRY_OPS                                                     \
  mlir::acc::CopyinOp, mlir::acc::CreateOp, mlir::acc::PresentOp,              \
      mlir::acc::NoCreateOp, mlir::acc::AttachOp, mlir::acc::DevicePtrOp,      \
      mlir::acc::GetDevicePtrOp, mlir::acc::PrivateOp,                         \
      mlir::acc::FirstprivateOp, mlir::acc::UpdateDeviceOp,                    \
      mlir::acc::UseDeviceOp, mlir::acc::ReductionOp,                          \
      mlir::acc::DeclareDeviceResidentOp, mlir::acc::DeclareLinkOp,            \
      mlir::acc::CacheOp
#define ACC_DATA_EXIT_OPS                                                      \
  mlir::acc::CopyoutOp, mlir::acc::DeleteOp, mlir::acc::DetachOp,              \
      mlir::acc::UpdateHostOp
#define ACC_DATA_CLAUSE_OPS ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS
#define ACC_COMPUTE_CONSTRUCT_OPS                                              \
  mlir::acc::ParallelOp, mlir::acc::KernelsOp, mlir::acc::SerialOp
#define ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS                                     \
  ACC_COMPUTE_CONSTRUCT_OPS, mlir::acc::LoopOp
#define OPENACC_DATA_CONSTRUCT_STRUCTURED_OPS                                  \
  mlir::acc::DataOp, mlir::acc::DeclareOp
#define ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS                                    \
  mlir::acc::EnterDataOp, mlir::acc::ExitDataOp, mlir::acc::UpdateOp,          \
      mlir::acc::HostDataOp, mlir::acc::DeclareEnterOp,                        \
      mlir::acc::DeclareExitOp
#define ACC_DATA_CONSTRUCT_OPS                                                 \
  OPENACC_DATA_CONSTRUCT_STRUCTURED_OPS, ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS
#define ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS                                     \
  ACC_COMPUTE_CONSTRUCT_OPS, ACC_DATA_CONSTRUCT_OPS
#define ACC_COMPUTE_LOOP_AND_DATA_CONSTRUCT_OPS                                \
  ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS, ACC_DATA_CONSTRUCT_OPS

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

/// Used to obtain the `varPtr` from a data clause operation.
/// Returns empty value if not a data clause operation or is a data exit
/// operation with no `varPtr`.
mlir::Value getVarPtr(mlir::Operation *accDataClauseOp);

/// Used to obtain the `accPtr` from a data clause operation.
/// When a data entry operation, it obtains its result `accPtr` value.
/// If a data exit operation, it obtains its operand `accPtr` value.
/// Returns empty value if not a data clause operation.
mlir::Value getAccPtr(mlir::Operation *accDataClauseOp);

/// Used to obtain the `varPtrPtr` from a data clause operation.
/// Returns empty value if not a data clause operation.
mlir::Value getVarPtrPtr(mlir::Operation *accDataClauseOp);

/// Used to obtain `bounds` from an acc data clause operation.
/// Returns an empty vector if there are no bounds.
mlir::SmallVector<mlir::Value> getBounds(mlir::Operation *accDataClauseOp);

/// Used to obtain the `name` from an acc operation.
std::optional<llvm::StringRef> getVarName(mlir::Operation *accOp);

/// Used to obtain the `dataClause` from a data entry operation.
/// Returns empty optional if not a data entry operation.
std::optional<mlir::acc::DataClause>
getDataClause(mlir::Operation *accDataEntryOp);

/// Used to find out whether data operation is implicit.
/// Returns false if not a data operation or if it is a data operation without
/// implicit flag.
bool getImplicitFlag(mlir::Operation *accDataEntryOp);

/// Used to get an immutable range iterating over the data operands.
mlir::ValueRange getDataOperands(mlir::Operation *accOp);

/// Used to get a mutable range iterating over the data operands.
mlir::MutableOperandRange getMutableDataOperands(mlir::Operation *accOp);

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

static constexpr StringLiteral getCombinedConstructsAttrName() {
  return CombinedConstructsTypeAttr::name;
}

struct RuntimeCounters
    : public mlir::SideEffects::Resource::Base<RuntimeCounters> {
  mlir::StringRef getName() final { return "AccRuntimeCounters"; }
};

struct ConstructResource
    : public mlir::SideEffects::Resource::Base<ConstructResource> {
  mlir::StringRef getName() final { return "AccConstructResource"; }
};

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACC_H_
