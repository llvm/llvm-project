//===- OpenACCUtils.h - OpenACC Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_UTILS_OPENACCUTILS_H_
#define MLIR_DIALECT_OPENACC_UTILS_OPENACCUTILS_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace mlir {
namespace acc {
/// Used to obtain the `varPtr` from a data clause operation.
/// Returns empty value if not a data clause operation or is a data exit
/// operation with no `varPtr`.
mlir::Value getVarPtr(mlir::Operation *accDataClauseOp);

/// Used to set the `varPtr` of a data clause operation.
/// Returns true if it was set successfully and false if this is not a data
/// clause operation.
bool setVarPtr(mlir::Operation *accDataClauseOp, mlir::Value varPtr);

/// Used to obtain the `accPtr` from a data clause operation.
/// When a data entry operation, it obtains its result `accPtr` value.
/// If a data exit operation, it obtains its operand `accPtr` value.
/// Returns empty value if not a data clause operation.
mlir::Value getAccPtr(mlir::Operation *accDataClauseOp);

/// Used to set the `accPtr` for a data exit operation.
/// Returns true if it was set successfully and false if is not a data exit
/// operation (data entry operations have their result as `accPtr` which
/// cannot be changed).
bool setAccPtr(mlir::Operation *accDataClauseOp, mlir::Value accPtr);

/// Used to obtain the `varPtrPtr` from a data clause operation.
/// Returns empty value if not a data clause operation.
mlir::Value getVarPtrPtr(mlir::Operation *accDataClauseOp);

/// Used to set the `varPtrPtr` for a data clause operation.
/// Returns false if the operation does not have varPtrPtr or is not a data
/// clause op.
bool setVarPtrPtr(mlir::Operation *accDataClauseOp, mlir::Value varPtrPtr);

/// Used to obtain `bounds` from an acc data clause operation.
/// Returns an empty vector if there are no bounds.
mlir::SmallVector<mlir::Value> getBounds(mlir::Operation *accDataClauseOp);

/// Used to set `bounds` for an acc data clause operation. It completely
/// replaces all bounds operands with the new list.
/// Returns false if new bounds were not set (such as when argument is not
/// an acc data clause operation).
bool setBounds(mlir::Operation *accDataClauseOp,
               mlir::SmallVector<mlir::Value> &bounds);
bool setBounds(mlir::Operation *accDataClauseOp, mlir::Value bound);

/// Used to obtain the `dataClause` from a data clause operation.
/// Returns empty optional if not a data operation.
std::optional<mlir::acc::DataClause>
getDataClause(mlir::Operation *accDataClauseOp);

/// Used to set the `dataClause` on a data clause operation.
/// Returns true if successfully set and false otherwise.
bool setDataClause(mlir::Operation *accDataClauseOp,
                   mlir::acc::DataClause dataClause);

/// Used to find out whether this data operation uses structured runtime
/// counters. Returns false if not a data operation or if it is a data operation
/// without the structured flag set.
bool getStructuredFlag(mlir::Operation *accDataClauseOp);

/// Used to update the data clause operation whether it represents structured
/// or dynamic (value of `structured` is passed as false).
/// Returns true if successfully set and false otherwise.
bool setStructuredFlag(mlir::Operation *accDataClauseOp, bool structured);

/// Used to find out whether data operation is implicit.
/// Returns false if not a data operation or if it is a data operation without
/// implicit flag.
bool getImplicitFlag(mlir::Operation *accDataClauseOp);

/// Used to update the data clause operation whether this operation is
/// implicit or explicit (`implicit` set as false).
/// Returns true if successfully set and false otherwise.
bool setImplicitFlag(mlir::Operation *accDataClauseOp, bool implicit);

/// Used to obtain the `name` from an acc operation.
std::optional<llvm::StringRef> getVarName(mlir::Operation *accDataClauseOp);

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

/// Used to get an immutable range iterating over the data operands.
mlir::ValueRange getDataOperands(mlir::Operation *accOp);

/// Used to get a mutable range iterating over the data operands.
mlir::MutableOperandRange getMutableDataOperands(mlir::Operation *accOp);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_UTILS_OPENACCUTILS_H_
