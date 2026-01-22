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

MlirStringRef mlirLLVMPointerTypeGetName(void) {
  return wrap(LLVM::LLVMPointerType::name);
}

MlirTypeID mlirLLVMPointerTypeGetTypeID() {
  return wrap(LLVM::LLVMPointerType::getTypeID());
}

bool mlirTypeIsALLVMPointerType(MlirType type) {
  return isa<LLVM::LLVMPointerType>(unwrap(type));
}

unsigned mlirLLVMPointerTypeGetAddressSpace(MlirType pointerType) {
  return cast<LLVM::LLVMPointerType>(unwrap(pointerType)).getAddressSpace();
}

MlirType mlirLLVMVoidTypeGet(MlirContext ctx) {
  return wrap(LLVMVoidType::get(unwrap(ctx)));
}

MlirStringRef mlirLLVMVoidTypeGetName(void) { return wrap(LLVMVoidType::name); }

MlirType mlirLLVMArrayTypeGet(MlirType elementType, unsigned numElements) {
  return wrap(LLVMArrayType::get(unwrap(elementType), numElements));
}

MlirStringRef mlirLLVMArrayTypeGetName(void) {
  return wrap(LLVMArrayType::name);
}

MlirType mlirLLVMArrayTypeGetElementType(MlirType type) {
  return wrap(cast<LLVM::LLVMArrayType>(unwrap(type)).getElementType());
}

MlirType mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                                 MlirType const *argumentTypes, bool isVarArg) {
  SmallVector<Type, 2> argumentStorage;
  return wrap(LLVMFunctionType::get(
      unwrap(resultType),
      unwrapList(nArgumentTypes, argumentTypes, argumentStorage), isVarArg));
}

MlirStringRef mlirLLVMFunctionTypeGetName(void) {
  return wrap(LLVMFunctionType::name);
}

intptr_t mlirLLVMFunctionTypeGetNumInputs(MlirType type) {
  return llvm::cast<LLVM::LLVMFunctionType>(unwrap(type)).getNumParams();
}

MlirType mlirLLVMFunctionTypeGetInput(MlirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(llvm::cast<LLVM::LLVMFunctionType>(unwrap(type))
                  .getParamType(static_cast<unsigned>(pos)));
}

MlirType mlirLLVMFunctionTypeGetReturnType(MlirType type) {
  return wrap(llvm::cast<LLVM::LLVMFunctionType>(unwrap(type)).getReturnType());
}

bool mlirTypeIsALLVMStructType(MlirType type) {
  return isa<LLVM::LLVMStructType>(unwrap(type));
}

MlirTypeID mlirLLVMStructTypeGetTypeID() {
  return wrap(LLVM::LLVMStructType::getTypeID());
}

MlirStringRef mlirLLVMStructTypeGetName(void) {
  return wrap(LLVM::LLVMStructType::name);
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

MlirStringRef mlirLLVMDIExpressionElemAttrGetName(void) {
  return wrap(DIExpressionElemAttr::name);
}

MlirAttribute mlirLLVMDIExpressionAttrGet(MlirContext ctx, intptr_t nOperations,
                                          MlirAttribute const *operations) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nOperations);

  return wrap(DIExpressionAttr::get(
      unwrap(ctx),
      llvm::map_to_vector(unwrapList(nOperations, operations, attrStorage),
                          llvm::CastTo<DIExpressionElemAttr>)));
}

MlirStringRef mlirLLVMDIExpressionAttrGetName(void) {
  return wrap(DIExpressionAttr::name);
}

MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx) {
  return wrap(DINullTypeAttr::get(unwrap(ctx)));
}

MlirStringRef mlirLLVMDINullTypeAttrGetName(void) {
  return wrap(DINullTypeAttr::name);
}

MlirAttribute mlirLLVMDIBasicTypeAttrGet(MlirContext ctx, unsigned int tag,
                                         MlirAttribute name,
                                         uint64_t sizeInBits,
                                         MlirLLVMTypeEncoding encoding) {

  return wrap(DIBasicTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, encoding));
}

MlirStringRef mlirLLVMDIBasicTypeAttrGetName(void) {
  return wrap(DIBasicTypeAttr::name);
}

MlirAttribute mlirLLVMDICompositeTypeAttrGetRecSelf(MlirAttribute recId) {
  return wrap(
      DICompositeTypeAttr::getRecSelf(cast<DistinctAttr>(unwrap(recId))));
}

MlirAttribute mlirLLVMDICompositeTypeAttrGet(
    MlirContext ctx, MlirAttribute recId, bool isRecSelf, unsigned int tag,
    MlirAttribute name, MlirAttribute file, uint32_t line, MlirAttribute scope,
    MlirAttribute baseType, int64_t flags, uint64_t sizeInBits,
    uint64_t alignInBits, intptr_t nElements, MlirAttribute const *elements,
    MlirAttribute dataLocation, MlirAttribute rank, MlirAttribute allocated,
    MlirAttribute associated) {
  SmallVector<Attribute> elementsStorage;
  elementsStorage.reserve(nElements);

  return wrap(DICompositeTypeAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(recId)), isRecSelf, tag,
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(file)), line,
      cast<DIScopeAttr>(unwrap(scope)), cast<DITypeAttr>(unwrap(baseType)),
      DIFlags(flags), sizeInBits, alignInBits,
      cast<DIExpressionAttr>(unwrap(dataLocation)),
      cast<DIExpressionAttr>(unwrap(rank)),
      cast<DIExpressionAttr>(unwrap(allocated)),
      cast<DIExpressionAttr>(unwrap(associated)),
      llvm::map_to_vector(unwrapList(nElements, elements, elementsStorage),
                          llvm::CastTo<DINodeAttr>)));
}

MlirStringRef mlirLLVMDICompositeTypeAttrGetName(void) {
  return wrap(DICompositeTypeAttr::name);
}

MlirAttribute mlirLLVMDIDerivedTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name,
    MlirAttribute baseType, uint64_t sizeInBits, uint32_t alignInBits,
    uint64_t offsetInBits, int64_t dwarfAddressSpace, MlirAttribute extraData) {
  std::optional<unsigned> addressSpace = std::nullopt;
  if (dwarfAddressSpace >= 0)
    addressSpace = (unsigned)dwarfAddressSpace;
  return wrap(DIDerivedTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)),
      cast<DITypeAttr>(unwrap(baseType)), sizeInBits, alignInBits, offsetInBits,
      addressSpace, cast<DINodeAttr>(unwrap(extraData))));
}

MlirStringRef mlirLLVMDIDerivedTypeAttrGetName(void) {
  return wrap(DIDerivedTypeAttr::name);
}

MlirAttribute mlirLLVMDIStringTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name, uint64_t sizeInBits,
    uint32_t alignInBits, MlirAttribute stringLength,
    MlirAttribute stringLengthExp, MlirAttribute stringLocationExp,
    MlirLLVMTypeEncoding encoding) {
  return wrap(DIStringTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, alignInBits,
      cast<DIVariableAttr>(unwrap(stringLength)),
      cast<DIExpressionAttr>(unwrap(stringLengthExp)),
      cast<DIExpressionAttr>(unwrap(stringLocationExp)), encoding));
}

MlirStringRef mlirLLVMDIStringTypeAttrGetName(void) {
  return wrap(DIStringTypeAttr::name);
}

MlirAttribute
mlirLLVMDIDerivedTypeAttrGetBaseType(MlirAttribute diDerivedType) {
  return wrap(cast<DIDerivedTypeAttr>(unwrap(diDerivedType)).getBaseType());
}

MlirAttribute mlirLLVMCConvAttrGet(MlirContext ctx, MlirLLVMCConv cconv) {
  return wrap(CConvAttr::get(unwrap(ctx), CConv(cconv)));
}

MlirStringRef mlirLLVMCConvAttrGetName(void) { return wrap(CConvAttr::name); }

MlirAttribute mlirLLVMComdatAttrGet(MlirContext ctx, MlirLLVMComdat comdat) {
  return wrap(ComdatAttr::get(unwrap(ctx), comdat::Comdat(comdat)));
}

MlirStringRef mlirLLVMComdatAttrGetName(void) { return wrap(ComdatAttr::name); }

MlirAttribute mlirLLVMLinkageAttrGet(MlirContext ctx, MlirLLVMLinkage linkage) {
  return wrap(LinkageAttr::get(unwrap(ctx), linkage::Linkage(linkage)));
}

MlirStringRef mlirLLVMLinkageAttrGetName(void) {
  return wrap(LinkageAttr::name);
}

MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx, MlirAttribute name,
                                    MlirAttribute directory) {
  return wrap(DIFileAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                              cast<StringAttr>(unwrap(directory))));
}

MlirStringRef mlirLLVMDIFileAttrGetName(void) { return wrap(DIFileAttr::name); }

MlirAttribute mlirLLVMDICompileUnitAttrGet(
    MlirContext ctx, MlirAttribute id, unsigned int sourceLanguage,
    MlirAttribute file, MlirAttribute producer, bool isOptimized,
    MlirLLVMDIEmissionKind emissionKind, MlirLLVMDINameTableKind nameTableKind,
    MlirAttribute splitDebugFilename) {
  return wrap(DICompileUnitAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)), sourceLanguage,
      cast<DIFileAttr>(unwrap(file)), cast<StringAttr>(unwrap(producer)),
      isOptimized, DIEmissionKind(emissionKind), DINameTableKind(nameTableKind),
      cast<StringAttr>(unwrap(splitDebugFilename))));
}

MlirStringRef mlirLLVMDICompileUnitAttrGetName(void) {
  return wrap(DICompileUnitAttr::name);
}

MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx, uint64_t value) {
  return wrap(DIFlagsAttr::get(unwrap(ctx), DIFlags(value)));
}

MlirStringRef mlirLLVMDIFlagsAttrGetName(void) {
  return wrap(DIFlagsAttr::name);
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

MlirStringRef mlirLLVMDILexicalBlockAttrGetName(void) {
  return wrap(DILexicalBlockAttr::name);
}

MlirAttribute mlirLLVMDILexicalBlockFileAttrGet(MlirContext ctx,
                                                MlirAttribute scope,
                                                MlirAttribute file,
                                                unsigned int discriminator) {
  return wrap(DILexicalBlockFileAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<DIFileAttr>(unwrap(file)), discriminator));
}

MlirStringRef mlirLLVMDILexicalBlockFileAttrGetName(void) {
  return wrap(DILexicalBlockFileAttr::name);
}

MlirAttribute mlirLLVMDILocalVariableAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute name,
    MlirAttribute diFile, unsigned int line, unsigned int arg,
    unsigned int alignInBits, MlirAttribute diType, int64_t flags) {
  return wrap(DILocalVariableAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(diFile)), line,
      arg, alignInBits, cast<DITypeAttr>(unwrap(diType)), DIFlags(flags)));
}

MlirStringRef mlirLLVMDILocalVariableAttrGetName(void) {
  return wrap(DILocalVariableAttr::name);
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
                          llvm::CastTo<DITypeAttr>)));
}

MlirStringRef mlirLLVMDISubroutineTypeAttrGetName(void) {
  return wrap(DISubroutineTypeAttr::name);
}

MlirAttribute mlirLLVMDISubprogramAttrGetRecSelf(MlirAttribute recId) {
  return wrap(DISubprogramAttr::getRecSelf(cast<DistinctAttr>(unwrap(recId))));
}

MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute recId, bool isRecSelf, MlirAttribute id,
    MlirAttribute compileUnit, MlirAttribute scope, MlirAttribute name,
    MlirAttribute linkageName, MlirAttribute file, unsigned int line,
    unsigned int scopeLine, uint64_t subprogramFlags, MlirAttribute type,
    intptr_t nRetainedNodes, MlirAttribute const *retainedNodes,
    intptr_t nAnnotations, MlirAttribute const *annotations) {
  SmallVector<Attribute> nodesStorage;
  nodesStorage.reserve(nRetainedNodes);

  SmallVector<Attribute> annotationsStorage;
  annotationsStorage.reserve(nAnnotations);

  return wrap(DISubprogramAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(recId)), isRecSelf,
      cast<DistinctAttr>(unwrap(id)),
      cast<DICompileUnitAttr>(unwrap(compileUnit)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(linkageName)), cast<DIFileAttr>(unwrap(file)),
      line, scopeLine, DISubprogramFlags(subprogramFlags),
      cast<DISubroutineTypeAttr>(unwrap(type)),
      llvm::map_to_vector(
          unwrapList(nRetainedNodes, retainedNodes, nodesStorage),
          llvm::CastTo<DINodeAttr>),
      llvm::map_to_vector(
          unwrapList(nAnnotations, annotations, annotationsStorage),
          llvm::CastTo<DINodeAttr>)));
}

MlirStringRef mlirLLVMDISubprogramAttrGetName(void) {
  return wrap(DISubprogramAttr::name);
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

MlirStringRef mlirLLVMDIModuleAttrGetName(void) {
  return wrap(DIModuleAttr::name);
}

MlirAttribute mlirLLVMDIModuleAttrGetScope(MlirAttribute diModule) {
  return wrap(cast<DIModuleAttr>(unwrap(diModule)).getScope());
}

MlirAttribute mlirLLVMDIImportedEntityAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute scope,
    MlirAttribute entity, MlirAttribute file, unsigned int line,
    MlirAttribute name, intptr_t nElements, MlirAttribute const *elements) {
  SmallVector<Attribute> elementsStorage;
  elementsStorage.reserve(nElements);
  return wrap(DIImportedEntityAttr::get(
      unwrap(ctx), tag, cast<DIScopeAttr>(unwrap(scope)),
      cast<DINodeAttr>(unwrap(entity)), cast<DIFileAttr>(unwrap(file)), line,
      cast<StringAttr>(unwrap(name)),
      llvm::map_to_vector(unwrapList(nElements, elements, elementsStorage),
                          llvm::CastTo<DINodeAttr>)));
}

MlirStringRef mlirLLVMDIImportedEntityAttrGetName(void) {
  return wrap(DIImportedEntityAttr::name);
}

MlirAttribute mlirLLVMDIAnnotationAttrGet(MlirContext ctx, MlirAttribute name,
                                          MlirAttribute value) {
  return wrap(DIAnnotationAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                                    cast<StringAttr>(unwrap(value))));
}

MlirStringRef mlirLLVMDIAnnotationAttrGetName(void) {
  return wrap(DIAnnotationAttr::name);
}
