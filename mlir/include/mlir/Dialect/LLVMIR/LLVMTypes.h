//===- LLVMTypes.h - MLIR LLVM dialect types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the LLVM dialect in MLIR. These MLIR types
// correspond to the LLVM IR type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMTYPES_H_
#define MLIR_DIALECT_LLVMIR_LLVMTYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <optional>

namespace llvm {
class ElementCount;
class TypeSize;
} // namespace llvm

namespace mlir {

class AsmParser;
class AsmPrinter;

namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMFunctionTypeStorage;
struct LLVMPointerTypeStorage;
struct LLVMStructTypeStorage;
struct LLVMTypeAndSizeStorage;
} // namespace detail
} // namespace LLVM
} // namespace mlir

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMTypes.h.inc"

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// Trivial types.
//===----------------------------------------------------------------------===//

// Batch-define trivial types.
#define DEFINE_TRIVIAL_LLVM_TYPE(ClassName, TypeName)                          \
  class ClassName : public Type::TypeBase<ClassName, Type, TypeStorage> {      \
  public:                                                                      \
    using Base::Base;                                                          \
    static constexpr StringLiteral name = TypeName;                            \
  }

DEFINE_TRIVIAL_LLVM_TYPE(LLVMVoidType, "llvm.void");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMTokenType, "llvm.token");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMLabelType, "llvm.label");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMMetadataType, "llvm.metadata");

#undef DEFINE_TRIVIAL_LLVM_TYPE

//===----------------------------------------------------------------------===//
// Printing and parsing.
//===----------------------------------------------------------------------===//

namespace detail {
/// Parses an LLVM dialect type.
Type parseType(DialectAsmParser &parser);

/// Prints an LLVM Dialect type.
void printType(Type type, AsmPrinter &printer);
} // namespace detail

/// Parse any MLIR type or a concise syntax for LLVM types.
ParseResult parsePrettyLLVMType(AsmParser &p, Type &type);
/// Print any MLIR type or a concise syntax for LLVM types.
void printPrettyLLVMType(AsmPrinter &p, Type type);

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

/// Returns `true` if the given type is compatible with the LLVM dialect. This
/// is an alias to `LLVMDialect::isCompatibleType`.
bool isCompatibleType(Type type);

/// Returns `true` if the given outer type is compatible with the LLVM dialect
/// without checking its potential nested types such as struct elements.
bool isCompatibleOuterType(Type type);

/// Returns `true` if the given type is a floating-point type compatible with
/// the LLVM dialect.
bool isCompatibleFloatingPointType(Type type);

/// Returns `true` if the given type is a vector type compatible with the LLVM
/// dialect. Compatible types include 1D built-in vector types of built-in
/// integers and floating-point values, LLVM dialect fixed vector types of LLVM
/// dialect pointers and LLVM dialect scalable vector types.
bool isCompatibleVectorType(Type type);

/// Returns the element count of any LLVM-compatible vector type.
llvm::ElementCount getVectorNumElements(Type type);

/// Returns whether a vector type is scalable or not.
bool isScalableVectorType(Type vectorType);

/// Creates an LLVM dialect-compatible vector type with the given element type
/// and length.
Type getVectorType(Type elementType, unsigned numElements,
                   bool isScalable = false);

/// Creates an LLVM dialect-compatible vector type with the given element type
/// and length.
Type getVectorType(Type elementType, const llvm::ElementCount &numElements);

/// Creates an LLVM dialect-compatible type with the given element type and
/// length.
Type getFixedVectorType(Type elementType, unsigned numElements);

/// Creates an LLVM dialect-compatible type with the given element type and
/// length.
Type getScalableVectorType(Type elementType, unsigned numElements);

/// Returns the size of the given primitive LLVM dialect-compatible type
/// (including vectors) in bits, for example, the size of i16 is 16 and
/// the size of vector<4xi16> is 64. Returns 0 for non-primitive
/// (aggregates such as struct) or types that don't have a size (such as void).
llvm::TypeSize getPrimitiveTypeSizeInBits(Type type);

/// The positions of different values in the data layout entry for pointers.
enum class PtrDLEntryPos { Size = 0, Abi = 1, Preferred = 2, Index = 3 };

/// Returns the value that corresponds to named position `pos` from the
/// data layout entry `attr` assuming it's a dense integer elements attribute.
/// Returns `std::nullopt` if `pos` is not present in the entry.
/// Currently only `PtrDLEntryPos::Index` is optional, and all other positions
/// may be assumed to be present.
std::optional<uint64_t> extractPointerSpecValue(Attribute attr,
                                                PtrDLEntryPos pos);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMTYPES_H_
