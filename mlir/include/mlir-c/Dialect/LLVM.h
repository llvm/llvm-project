//===-- mlir-c/Dialect/LLVM.h - C API for LLVM --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_LLVM_H
#define MLIR_C_DIALECT_LLVM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLVM, llvm);

/// Creates an llvm.ptr type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMPointerTypeGet(MlirContext ctx,
                                                   unsigned addressSpace);

/// Returns `true` if the type is an LLVM dialect pointer type.
MLIR_CAPI_EXPORTED bool mlirTypeIsALLVMPointerType(MlirType type);

/// Returns address space of llvm.ptr
MLIR_CAPI_EXPORTED unsigned
mlirLLVMPointerTypeGetAddressSpace(MlirType pointerType);

/// Creates an llmv.void type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMVoidTypeGet(MlirContext ctx);

/// Creates an llvm.array type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMArrayTypeGet(MlirType elementType,
                                                 unsigned numElements);

/// Creates an llvm.func type.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                        MlirType const *argumentTypes, bool isVarArg);

/// Returns `true` if the type is an LLVM dialect struct type.
MLIR_CAPI_EXPORTED bool mlirTypeIsALLVMStructType(MlirType type);

/// Returns `true` if the type is a literal (unnamed) LLVM struct type.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsLiteral(MlirType type);

/// Returns the number of fields in the struct. Asserts if the struct is opaque
/// or not yet initialized.
MLIR_CAPI_EXPORTED intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type);

/// Returns the `positions`-th field of the struct. Asserts if the struct is
/// opaque, not yet initialized or if the position is out of range.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeGetElementType(MlirType type,
                                                             intptr_t position);

/// Returns `true` if the struct is packed.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsPacked(MlirType type);

/// Returns the identifier of the identified struct. Asserts that the struct is
/// identified, i.e., not literal.
MLIR_CAPI_EXPORTED MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type);

/// Returns `true` is the struct is explicitly opaque (will not have a body) or
/// uninitialized (will eventually have a body).
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsOpaque(MlirType type);

/// Creates an LLVM literal (unnamed) struct type. This may assert if the fields
/// have types not compatible with the LLVM dialect. For a graceful failure, use
/// the checked version.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                             MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM literal (unnamed) struct type if possible. Emits a
/// diagnostic at the given location and returns null otherwise.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc, intptr_t nFieldTypes,
                                    MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM identified struct type with no body. If a struct type with
/// this name already exists in the context, returns that type. Use
/// mlirLLVMStructTypeIdentifiedNewGet to create a fresh struct type,
/// potentially renaming it. The body should be set separatelty by calling
/// mlirLLVMStructTypeSetBody, if it isn't set already.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx,
                                                            MlirStringRef name);

/// Creates an LLVM identified struct type with no body and a name starting with
/// the given prefix. If a struct with the exact name as the given prefix
/// already exists, appends an unspecified suffix to the name so that the name
/// is unique in context.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedNewGet(
    MlirContext ctx, MlirStringRef name, intptr_t nFieldTypes,
    MlirType const *fieldTypes, bool isPacked);

MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx,
                                                        MlirStringRef name);

/// Sets the body of the identified struct if it hasn't been set yet. Returns
/// whether the operation was successful.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirLLVMStructTypeSetBody(MlirType structType, intptr_t nFieldTypes,
                          MlirType const *fieldTypes, bool isPacked);

/// Creates a LLVM CConv attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMCConvAttrGet(MlirContext ctx,
                                                      MlirLLVMCConv cconv);

/// Creates a LLVM Comdat attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMComdatAttrGet(MlirContext ctx,
                                                       MlirLLVMComdat comdat);

/// Creates a LLVM Linkage attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMLinkageAttrGet(MlirContext ctx, MlirLLVMLinkage linkage);

/// Creates a LLVM DINullType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx);

/// Creates a LLVM DIExpressionElem attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIExpressionElemAttrGet(MlirContext ctx, unsigned int opcode,
                                intptr_t nArguments, uint64_t const *arguments);

/// Creates a LLVM DIExpression attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIExpressionAttrGet(
    MlirContext ctx, intptr_t nOperations, MlirAttribute const *operations);

enum MlirLLVMTypeEncoding {
  MlirLLVMTypeEncodingAddress = 0x1,
  MlirLLVMTypeEncodingBoolean = 0x2,
  MlirLLVMTypeEncodingComplexFloat = 0x31,
  MlirLLVMTypeEncodingFloatT = 0x4,
  MlirLLVMTypeEncodingSigned = 0x5,
  MlirLLVMTypeEncodingSignedChar = 0x6,
  MlirLLVMTypeEncodingUnsigned = 0x7,
  MlirLLVMTypeEncodingUnsignedChar = 0x08,
  MlirLLVMTypeEncodingImaginaryFloat = 0x09,
  MlirLLVMTypeEncodingPackedDecimal = 0x0a,
  MlirLLVMTypeEncodingNumericString = 0x0b,
  MlirLLVMTypeEncodingEdited = 0x0c,
  MlirLLVMTypeEncodingSignedFixed = 0x0d,
  MlirLLVMTypeEncodingUnsignedFixed = 0x0e,
  MlirLLVMTypeEncodingDecimalFloat = 0x0f,
  MlirLLVMTypeEncodingUTF = 0x10,
  MlirLLVMTypeEncodingUCS = 0x11,
  MlirLLVMTypeEncodingASCII = 0x12,
  MlirLLVMTypeEncodingLoUser = 0x80,
  MlirLLVMTypeEncodingHiUser = 0xff,
};
typedef enum MlirLLVMTypeEncoding MlirLLVMTypeEncoding;

/// Creates a LLVM DIBasicType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIBasicTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name, uint64_t sizeInBits,
    MlirLLVMTypeEncoding encoding);

/// Creates a LLVM DICompositeType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDICompositeTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute recId, MlirAttribute name,
    MlirAttribute file, uint32_t line, MlirAttribute scope,
    MlirAttribute baseType, int64_t flags, uint64_t sizeInBits,
    uint64_t alignInBits, intptr_t nElements, MlirAttribute const *elements,
    MlirAttribute dataLocation, MlirAttribute rank, MlirAttribute allocated,
    MlirAttribute associated);

/// Creates a LLVM DIDerivedType attribute.  Note that `dwarfAddressSpace` is an
/// optional field, where `MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL` indicates null
/// and non-negative values indicate a value present.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIDerivedTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name,
    MlirAttribute baseType, uint64_t sizeInBits, uint32_t alignInBits,
    uint64_t offsetInBits, int64_t dwarfAddressSpace, MlirAttribute extraData);

MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIStringTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name, uint64_t sizeInBits,
    uint32_t alignInBits, MlirAttribute stringLength,
    MlirAttribute stringLengthExp, MlirAttribute stringLocationExp,
    MlirLLVMTypeEncoding encoding);

/// Constant to represent std::nullopt for dwarfAddressSpace to omit the field.
#define MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL (-1)

/// Gets the base type from a LLVM DIDerivedType attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIDerivedTypeAttrGetBaseType(MlirAttribute diDerivedType);

/// Creates a LLVM DIFileAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx,
                                                       MlirAttribute name,
                                                       MlirAttribute directory);

/// Creates a LLVM DICompileUnit attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDICompileUnitAttrGet(
    MlirContext ctx, MlirAttribute id, unsigned int sourceLanguage,
    MlirAttribute file, MlirAttribute producer, bool isOptimized,
    MlirLLVMDIEmissionKind emissionKind, MlirLLVMDINameTableKind nameTableKind);

/// Creates a LLVM DIFlags attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx,
                                                        MlirLLVMDIFlags value);

/// Creates a LLVM DILexicalBlock attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILexicalBlockAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute file, unsigned int line,
    unsigned int column);

/// Creates a LLVM DILexicalBlockFile attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILexicalBlockFileAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute file,
    unsigned int discriminator);

/// Creates a LLVM DILocalVariableAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILocalVariableAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute name,
    MlirAttribute diFile, unsigned int line, unsigned int arg,
    unsigned int alignInBits, MlirAttribute diType, int64_t flags);

/// Creates a LLVM DINamespaceAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDINamespaceAttrGet(MlirContext ctx,
                                                            MlirAttribute name,
                                                            MlirAttribute scope,
                                                            bool exportSymbols);

/// Creates a LLVM DISubprogramAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute id, MlirAttribute compileUnit,
    MlirAttribute scope, MlirAttribute name, MlirAttribute linkageName,
    MlirAttribute file, unsigned int line, unsigned int scopeLine,
    MlirLLVMDISubprogramFlags subprogramFlags, MlirAttribute type);

/// Gets the scope from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetScope(MlirAttribute diSubprogram);

/// Gets the line from this DISubprogramAttr.
MLIR_CAPI_EXPORTED unsigned int
mlirLLVMDISubprogramAttrGetLine(MlirAttribute diSubprogram);

/// Gets the scope line from this DISubprogram.
MLIR_CAPI_EXPORTED unsigned int
mlirLLVMDISubprogramAttrGetScopeLine(MlirAttribute diSubprogram);

/// Gets the linkage name from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirIdentifier
mlirLLVMDISubprogramAttrGetLinkageName(MlirAttribute diSubprogram);

/// Gets the name from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirIdentifier
mlirLLVMDISubprogramAttrGetName(MlirAttribute diSubprogram);

/// Gets the subprogram flags from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirLLVMDISubprogramFlags
mlirLLVMDISubprogramAttrGetSubprogramFlags(MlirAttribute diSubprogram);

/// Gets the compile unit from this DISubprogram.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetCompileUnit(MlirAttribute diSubprogram);

/// Gets the file from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetFile(MlirAttribute diSubprogram);

/// Gets the type from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetType(MlirAttribute diSubprogram);

/// Creates a LLVM DISubroutineTypeAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubroutineTypeAttrGet(MlirContext ctx, unsigned int callingConvention,
                                intptr_t nTypes, MlirAttribute const *types);

/// Creates a LLVM DIModuleAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIModuleAttrGet(
    MlirContext ctx, MlirAttribute file, MlirAttribute scope,
    MlirAttribute name, MlirAttribute configMacros, MlirAttribute includePath,
    MlirAttribute apinotes, unsigned int line, bool isDecl);

/// Gets the scope of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIModuleAttrGetScope(MlirAttribute diModule);

/// Gets the api notes of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirIdentifier
mlirLLVMDIModuleAttrGetApinotes(MlirAttribute diModule);

/// Gets the config macros of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirIdentifier
mlirLLVMDIModuleAttrGetConfigMacros(MlirAttribute diModule);

/// Gets the file of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIModuleAttrGetFile(MlirAttribute diModule);

/// Gets the include path of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirIdentifier
mlirLLVMDIModuleAttrGetIncludePath(MlirAttribute diModule);

/// Gets whether this DIModuleAttr is a declaration.
MLIR_CAPI_EXPORTED bool mlirLLVMDIModuleAttrGetIsDecl(MlirAttribute diModule);

/// Creates a LLVM DISubrange attribute.
///
/// All parameters have the type IntegerAttr.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDISubrangeAttrGet(
    MlirContext ctx, MlirAttribute count, MlirAttribute lowerBound,
    MlirAttribute upperBound, MlirAttribute stride);

/// Creates a LLVM AtomicOrderingAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMAtomicOrderingAttrGet(MlirContext ctx, MlirLLVMAtomicOrdering ordering);

/// Creates a LLVM AtomicBinOpAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMAtomicBinOpAttrGet(MlirContext ctx, MlirLLVMAtomicBinOp val);

/// Creates a LLVM VisibilityAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMVisibilityAttrGet(MlirContext ctx, MlirLLVMVisibility visibility);

/// Creates a LLVM UnnamedAddrAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMUnnamedAddrAttrGet(MlirContext ctx, MlirLLVMUnnamedAddr val);

/// Creates a LLVM ICmpPredicateAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMICmpPredicateAttrGet(MlirContext ctx, MlirLLVMICmpPredicate val);

/// Creates a LLVM FCmpPredicateAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMFCmpPredicateAttrGet(MlirContext ctx, MlirLLVMFCmpPredicate val);

/// Creates a LLVM FramePointerKindAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMFramePointerKindAttrGet(MlirContext ctx, MlirLLVMFramePointerKind val);

/// Creates a LLVM FastmathFlagsAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMFastmathFlagsAttrGet(MlirContext ctx, MlirLLVMFastmathFlags val);

/// Creates a LLVM ModRefInfoAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMModRefInfoAttrGet(MlirContext ctx, MlirLLVMModRefInfo val);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_LLVM_H
