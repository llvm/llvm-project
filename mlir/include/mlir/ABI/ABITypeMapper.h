//===- ABITypeMapper.h - Map MLIR types to ABI types -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ABITypeMapper, which translates mlir::Type instances into
// the llvm::abi::Type hierarchy defined in llvm/ABI/Types.h.  Dialect-specific
// types are handled via MLIR's DataLayoutTypeInterface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ABI_ABITYPEMAPPER_H
#define MLIR_ABI_ABITYPEMAPPER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
namespace abi {

/// ABITypeMapper translates mlir::Type values into the llvm::abi::Type
/// hierarchy used by the LLVM ABI Lowering Library.
///
/// Standard MLIR types (IntegerType, FloatType, IndexType, VectorType,
/// MemRefType) are mapped directly.  Dialect-specific types are mapped
/// by querying the MLIR DataLayout for size and alignment.
///
/// Callers must supply a DataLayout (typically from the enclosing module)
/// so the mapper can determine sizes and alignments.
///
/// The mapper owns a BumpPtrAllocator; all returned abi::Type pointers
/// are valid for the lifetime of the mapper.
class ABITypeMapper {
public:
  explicit ABITypeMapper(const DataLayout &dl);

  /// Map an MLIR type to its ABI type representation.  Returns nullptr
  /// if the type cannot be mapped.
  const llvm::abi::Type *map(mlir::Type type);

  /// Access the underlying TypeBuilder for advanced use.
  llvm::abi::TypeBuilder &getTypeBuilder() { return builder; }

private:
  const llvm::abi::Type *mapIntegerType(mlir::IntegerType type);
  const llvm::abi::Type *mapFloatType(mlir::FloatType type);
  const llvm::abi::Type *mapIndexType(mlir::IndexType type);
  const llvm::abi::Type *mapVectorType(mlir::VectorType type);
  const llvm::abi::Type *mapMemRefType(mlir::MemRefType type);
  const llvm::abi::Type *mapNoneType(mlir::NoneType type);

  const DataLayout &dl;
  llvm::BumpPtrAllocator allocator;
  llvm::abi::TypeBuilder builder;
};

} // namespace abi
} // namespace mlir

#endif // MLIR_ABI_ABITYPEMAPPER_H
