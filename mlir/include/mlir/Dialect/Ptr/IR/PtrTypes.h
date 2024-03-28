//===- PointerTypes.h - Pointer types ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Pointer dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_PTRTYPES_H
#define MLIR_DIALECT_PTR_IR_PTRTYPES_H

#include "mlir/Dialect/Ptr/IR/MemoryModel.h"
#include "mlir/IR/AsmInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace ptr {
/// The positions of different values in the data layout entry for pointers.
enum class PtrDLEntryPos { Size = 0, Abi = 1, Preferred = 2, Index = 3 };

/// Returns the value that corresponds to named position `pos` from the
/// data layout entry `attr` assuming it's a dense integer elements attribute.
/// Returns `std::nullopt` if `pos` is not present in the entry.
/// Currently only `PtrDLEntryPos::Index` is optional, and all other positions
/// may be assumed to be present.
std::optional<uint64_t> extractPointerSpecValue(Attribute attr,
                                                PtrDLEntryPos pos);
} // namespace ptr
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.h.inc"

#endif // MLIR_DIALECT_PTR_IR_PTRTYPES_H
