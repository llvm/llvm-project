//===- LLVM.cpp - C Interface for LLVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;
using namespace mlir::LLVM;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLVM, llvm, LLVMDialect)

MlirType mlirLLVMPointerTypeGet(MlirContext ctx, unsigned addressSpace) {
  return wrap(LLVMPointerType::get(unwrap(ctx), addressSpace));
}

MlirType mlirLLVMVoidTypeGet(MlirContext ctx) {
  return wrap(LLVMVoidType::get(unwrap(ctx)));
}

MlirType mlirLLVMArrayTypeGet(MlirType elementType, unsigned numElements) {
  return wrap(LLVMArrayType::get(unwrap(elementType), numElements));
}

MlirType mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                                 MlirType const *argumentTypes, bool isVarArg) {
  SmallVector<Type, 2> argumentStorage;
  return wrap(LLVMFunctionType::get(
      unwrap(resultType),
      unwrapList(nArgumentTypes, argumentTypes, argumentStorage), isVarArg));
}

bool mlirTypeIsALLVMStructType(MlirType type) {
  return isa<LLVM::LLVMStructType>(unwrap(type));
}

bool mlirLLVMStructTypeIsLiteral(MlirType type) {
  return !cast<LLVM::LLVMStructType>(unwrap(type)).isIdentified();
}

intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).getBody().size();
}

MlirType mlirLLVMStructTypeGetElementType(MlirType type, intptr_t position) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getBody()[position]);
}

bool mlirLLVMStructTypeIsPacked(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isPacked();
}

MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getName());
}

bool mlirLLVMStructTypeIsOpaque(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isOpaque();
}

MlirType mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                                      MlirType const *fieldTypes,
                                      bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteral(
      unwrap(ctx), unwrapList(nFieldTypes, fieldTypes, fieldStorage),
      isPacked));
}

MlirType mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc,
                                             intptr_t nFieldTypes,
                                             MlirType const *fieldTypes,
                                             bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteralChecked(
      [loc]() { return emitError(unwrap(loc)); }, unwrap(loc)->getContext(),
      unwrapList(nFieldTypes, fieldTypes, fieldStorage), isPacked));
}

MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getOpaque(unwrap(name), unwrap(ctx)));
}

MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getIdentified(unwrap(ctx), unwrap(name)));
}

MlirType mlirLLVMStructTypeIdentifiedNewGet(MlirContext ctx, MlirStringRef name,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(LLVMStructType::getNewIdentified(
      unwrap(ctx), unwrap(name), unwrapList(nFieldTypes, fieldTypes, fields),
      isPacked));
}

MlirLogicalResult mlirLLVMStructTypeSetBody(MlirType structType,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(
      cast<LLVM::LLVMStructType>(unwrap(structType))
          .setBody(unwrapList(nFieldTypes, fieldTypes, fields), isPacked));
}

MlirAttribute mlirLLVMDIExpressionElemAttrGet(MlirContext ctx,
                                              unsigned int opcode,
                                              intptr_t nArguments,
                                              uint64_t const *arguments) {
  auto list = ArrayRef<uint64_t>(arguments, nArguments);
  return wrap(DIExpressionElemAttr::get(unwrap(ctx), opcode, list));
}

MlirAttribute mlirLLVMDIExpressionAttrGet(MlirContext ctx, intptr_t nOperations,
                                          MlirAttribute const *operations) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nOperations);

  return wrap(DIExpressionAttr::get(
      unwrap(ctx),
      llvm::map_to_vector(
          unwrapList(nOperations, operations, attrStorage),
          [](Attribute a) { return a.cast<DIExpressionElemAttr>(); })));
}

MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx) {
  return wrap(DINullTypeAttr::get(unwrap(ctx)));
}

MlirAttribute mlirLLVMDIBasicTypeAttrGet(MlirContext ctx, unsigned int tag,
                                         MlirAttribute name,
                                         uint64_t sizeInBits,
                                         MlirLLVMTypeEncoding encoding) {

  return wrap(DIBasicTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, encoding));
}

MlirAttribute mlirLLVMDICompositeTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute recId, MlirAttribute name,
    MlirAttribute file, uint32_t line, MlirAttribute scope,
    MlirAttribute baseType, int64_t flags, uint64_t sizeInBits,
    uint64_t alignInBits, intptr_t nElements, MlirAttribute const *elements) {
  SmallVector<Attribute> elementsStorage;
  elementsStorage.reserve(nElements);

  return wrap(DICompositeTypeAttr::get(
      unwrap(ctx), tag, cast<DistinctAttr>(unwrap(recId)),
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(file)), line,
      cast<DIScopeAttr>(unwrap(scope)), cast<DITypeAttr>(unwrap(baseType)),
      DIFlags(flags), sizeInBits, alignInBits,
      llvm::map_to_vector(unwrapList(nElements, elements, elementsStorage),
                          [](Attribute a) { return a.cast<DINodeAttr>(); })));
}

MlirAttribute mlirLLVMDIDerivedTypeAttrGet(MlirContext ctx, unsigned int tag,
                                           MlirAttribute name,
                                           MlirAttribute baseType,
                                           uint64_t sizeInBits,
                                           uint32_t alignInBits,
                                           uint64_t offsetInBits) {
  return wrap(DIDerivedTypeAttr::get(unwrap(ctx), tag,
                                     cast<StringAttr>(unwrap(name)),
                                     cast<DITypeAttr>(unwrap(baseType)),
                                     sizeInBits, alignInBits, offsetInBits));
}

MlirAttribute
mlirLLVMDIDerivedTypeAttrGetBaseType(MlirAttribute diDerivedType) {
  return wrap(cast<DIDerivedTypeAttr>(unwrap(diDerivedType)).getBaseType());
}

MlirAttribute mlirLLVMCConvAttrGet(MlirContext ctx, MlirLLVMCConv cconv) {
  return wrap(CConvAttr::get(unwrap(ctx), CConv(cconv)));
}

MlirAttribute mlirLLVMComdatAttrGet(MlirContext ctx, MlirLLVMComdat comdat) {
  return wrap(ComdatAttr::get(unwrap(ctx), comdat::Comdat(comdat)));
}

MlirAttribute mlirLLVMLinkageAttrGet(MlirContext ctx, MlirLLVMLinkage linkage) {
  return wrap(LinkageAttr::get(unwrap(ctx), linkage::Linkage(linkage)));
}

MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx, MlirAttribute name,
                                    MlirAttribute directory) {
  return wrap(DIFileAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                              cast<StringAttr>(unwrap(directory))));
}

MlirAttribute
mlirLLVMDICompileUnitAttrGet(MlirContext ctx, MlirAttribute id,
                             unsigned int sourceLanguage, MlirAttribute file,
                             MlirAttribute producer, bool isOptimized,
                             MlirLLVMDIEmissionKind emissionKind) {
  return wrap(DICompileUnitAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)), sourceLanguage,
      cast<DIFileAttr>(unwrap(file)), cast<StringAttr>(unwrap(producer)),
      isOptimized, DIEmissionKind(emissionKind)));
}

MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx, uint64_t value) {
  return wrap(DIFlagsAttr::get(unwrap(ctx), DIFlags(value)));
}

MlirAttribute mlirLLVMDILexicalBlockAttrGet(MlirContext ctx,
                                            MlirAttribute scope,
                                            MlirAttribute file,
                                            unsigned int line,
                                            unsigned int column) {
  return wrap(
      DILexicalBlockAttr::get(unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
                              cast<DIFileAttr>(unwrap(file)), line, column));
}

MlirAttribute mlirLLVMDILexicalBlockFileAttrGet(MlirContext ctx,
                                                MlirAttribute scope,
                                                MlirAttribute file,
                                                unsigned int discriminator) {
  return wrap(DILexicalBlockFileAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<DIFileAttr>(unwrap(file)), discriminator));
}

MlirAttribute
mlirLLVMDILocalVariableAttrGet(MlirContext ctx, MlirAttribute scope,
                               MlirAttribute name, MlirAttribute diFile,
                               unsigned int line, unsigned int arg,
                               unsigned int alignInBits, MlirAttribute diType) {
  return wrap(DILocalVariableAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(diFile)), line,
      arg, alignInBits, cast<DITypeAttr>(unwrap(diType))));
}

MlirAttribute mlirLLVMDISubroutineTypeAttrGet(MlirContext ctx,
                                              unsigned int callingConvention,
                                              intptr_t nTypes,
                                              MlirAttribute const *types) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nTypes);

  return wrap(DISubroutineTypeAttr::get(
      unwrap(ctx), callingConvention,
      llvm::map_to_vector(unwrapList(nTypes, types, attrStorage),
                          [](Attribute a) { return a.cast<DITypeAttr>(); })));
}

MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute id, MlirAttribute compileUnit,
    MlirAttribute scope, MlirAttribute name, MlirAttribute linkageName,
    MlirAttribute file, unsigned int line, unsigned int scopeLine,
    uint64_t subprogramFlags, MlirAttribute type) {
  return wrap(DISubprogramAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)),
      cast<DICompileUnitAttr>(unwrap(compileUnit)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(linkageName)), cast<DIFileAttr>(unwrap(file)),
      line, scopeLine, DISubprogramFlags(subprogramFlags),
      cast<DISubroutineTypeAttr>(unwrap(type))));
}

MlirAttribute mlirLLVMDISubprogramAttrGetScope(MlirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getScope());
}

unsigned int mlirLLVMDISubprogramAttrGetLine(MlirAttribute diSubprogram) {
  return cast<DISubprogramAttr>(unwrap(diSubprogram)).getLine();
}

unsigned int mlirLLVMDISubprogramAttrGetScopeLine(MlirAttribute diSubprogram) {
  return cast<DISubprogramAttr>(unwrap(diSubprogram)).getScopeLine();
}

MlirAttribute
mlirLLVMDISubprogramAttrGetCompileUnit(MlirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getCompileUnit());
}

MlirAttribute mlirLLVMDISubprogramAttrGetFile(MlirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getFile());
}

MlirAttribute mlirLLVMDISubprogramAttrGetType(MlirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getType());
}

MlirAttribute mlirLLVMDIModuleAttrGet(MlirContext ctx, MlirAttribute file,
                                      MlirAttribute scope, MlirAttribute name,
                                      MlirAttribute configMacros,
                                      MlirAttribute includePath,
                                      MlirAttribute apinotes, unsigned int line,
                                      bool isDecl) {
  return wrap(DIModuleAttr::get(
      unwrap(ctx), cast<DIFileAttr>(unwrap(file)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(configMacros)),
      cast<StringAttr>(unwrap(includePath)), cast<StringAttr>(unwrap(apinotes)),
      line, isDecl));
}

MlirAttribute mlirLLVMDIModuleAttrGetScope(MlirAttribute diModule) {
  return wrap(cast<DIModuleAttr>(unwrap(diModule)).getScope());
}
