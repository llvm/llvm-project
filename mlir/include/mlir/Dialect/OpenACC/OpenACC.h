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
#include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.h.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.h.inc"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <variant>

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.h.inc"

#include "mlir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.h"

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
#define ACC_DATA_CONSTRUCT_STRUCTURED_OPS                                      \
  mlir::acc::DataOp, mlir::acc::DeclareOp, mlir::acc::HostDataOp
#define ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS                                    \
  mlir::acc::EnterDataOp, mlir::acc::ExitDataOp, mlir::acc::UpdateOp,          \
      mlir::acc::DeclareEnterOp, mlir::acc::DeclareExitOp
#define ACC_DATA_CONSTRUCT_OPS                                                 \
  ACC_DATA_CONSTRUCT_STRUCTURED_OPS, ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS
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

/// Used to obtain the `var` from a data clause operation.
/// Returns empty value if not a data clause operation or is a data exit
/// operation with no `var`.
mlir::Value getVar(mlir::Operation *accDataClauseOp);

/// Used to obtain the `var` from a data clause operation if it implements
/// `PointerLikeType`.
mlir::TypedValue<mlir::acc::PointerLikeType>
getVarPtr(mlir::Operation *accDataClauseOp);

/// Used to obtains the `varType` from a data clause operation which records
/// the type of variable. When `var` is `PointerLikeType`, this returns
/// the type of the pointer target.
mlir::Type getVarType(mlir::Operation *accDataClauseOp);

/// Used to obtain the `accVar` from a data clause operation.
/// When a data entry operation, it obtains its result `accVar` value.
/// If a data exit operation, it obtains its operand `accVar` value.
/// Returns empty value if not a data clause operation.
mlir::Value getAccVar(mlir::Operation *accDataClauseOp);

/// Used to obtain the `accVar` from a data clause operation if it implements
/// `PointerLikeType`.
mlir::TypedValue<mlir::acc::PointerLikeType>
getAccPtr(mlir::Operation *accDataClauseOp);

/// Used to obtain the `varPtrPtr` from a data clause operation.
/// Returns empty value if not a data clause operation.
mlir::Value getVarPtrPtr(mlir::Operation *accDataClauseOp);

/// Used to obtain `bounds` from an acc data clause operation.
/// Returns an empty vector if there are no bounds.
mlir::SmallVector<mlir::Value> getBounds(mlir::Operation *accDataClauseOp);

/// Used to obtain `async` operands from an acc data clause operation.
/// Returns an empty vector if there are no such operands.
mlir::SmallVector<mlir::Value>
getAsyncOperands(mlir::Operation *accDataClauseOp);

/// Returns an array of acc:DeviceTypeAttr attributes attached to
/// an acc data clause operation, that correspond to the device types
/// associated with the async clauses with an async-value.
mlir::ArrayAttr getAsyncOperandsDeviceType(mlir::Operation *accDataClauseOp);

/// Returns an array of acc:DeviceTypeAttr attributes attached to
/// an acc data clause operation, that correspond to the device types
/// associated with the async clauses without an async-value.
mlir::ArrayAttr getAsyncOnly(mlir::Operation *accDataClauseOp);

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

/// Used to obtain the enclosing compute construct operation that contains
/// the provided `region`. Returns nullptr if no compute construct operation
/// is found. The returns operation is one of types defined by
///`ACC_COMPUTE_CONSTRUCT_OPS`.
mlir::Operation *getEnclosingComputeOp(mlir::Region &region);

/// Used to check whether the provided `type` implements the `PointerLikeType`
/// interface.
inline bool isPointerLikeType(mlir::Type type) {
  return mlir::isa<mlir::acc::PointerLikeType>(type);
}

/// Used to check whether the provided `type` implements the `MappableType`
/// interface.
inline bool isMappableType(mlir::Type type) {
  return mlir::isa<mlir::acc::MappableType>(type);
}

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

static constexpr StringLiteral getVarNameAttrName() {
  return VarNameAttr::name;
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

struct CurrentDeviceIdResource
    : public mlir::SideEffects::Resource::Base<CurrentDeviceIdResource> {
  mlir::StringRef getName() final { return "AccCurrentDeviceIdResource"; }
};

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACC_H_
