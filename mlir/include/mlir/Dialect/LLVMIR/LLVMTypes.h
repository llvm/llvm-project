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

#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <cstddef>
#include <optional>

namespace llvm {
class ElementCount;
class TypeSize;
} // namespace llvm

namespace mlir {

class AsmParser;
class AsmPrinter;
class DataLayout;

namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMFunctionTypeStorage;
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
// LLVM pointer type
//===----------------------------------------------------------------------===//

/// LLVM pointer type, this is a thin wrapper over the ptr::PtrType with
/// LLVM::AddressSpaceAttr as the memory space.
class LLVMPointerType : public ptr::PtrType {
public:
  static constexpr StringLiteral name = "llvm.ptr";
  static constexpr StringLiteral dialectName = "llvm";
  LLVMPointerType() = default;
  LLVMPointerType(std::nullptr_t) : ptr::PtrType() {}
  LLVMPointerType(const ptr::PtrType &other) : ptr::PtrType(other) {
    assert(isaLLVMPtr(other) && "not an LLVM pointer type");
  }
  static bool classof(Type type) {
    return isaLLVMPtr(dyn_cast_or_null<ptr::PtrType>(type));
  }
  /// Checks whether the given ptr::PtrType is an LLVM pointer type. That is, it
  /// uses an LLVMAddrSpaceAttrInterface as memory space.
  static bool isaLLVMPtr(ptr::PtrType type);
  /// Creates an LLVM pointer type with the given address space.
  static LLVMPointerType get(MLIRContext *context, unsigned addressSpace = 0);
  static constexpr StringLiteral getMnemonic() { return {"ptr"}; }
  static Type parse(AsmParser &odsParser);
  void print(AsmPrinter &odsPrinter) const;
  /// Returns the address space of the pointer type.
  unsigned getAddressSpace() const;
};

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

/// Returns `true` if the given type is a loadable type compatible with the LLVM
/// dialect.
bool isLoadableType(Type type);

/// Returns true if the given type is supported by atomic operations. All
/// integer, float, and pointer types with a power-of-two bitsize and a minimal
/// size of 8 bits are supported.
bool isTypeCompatibleWithAtomicOp(Type type, const DataLayout &dataLayout);

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

/// Returns the size of the given primitive LLVM dialect-compatible type
/// (including vectors) in bits, for example, the size of i16 is 16 and
/// the size of vector<4xi16> is 64. Returns 0 for non-primitive
/// (aggregates such as struct) or types that don't have a size (such as void).
llvm::TypeSize getPrimitiveTypeSizeInBits(Type type);
} // namespace LLVM

namespace detail {
template <>
class TypeIDResolver<LLVM::LLVMPointerType> {
public:
  static TypeID resolveTypeID() { return TypeID::get<ptr::PtrType>(); }
};
} // namespace detail
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMTYPES_H_
