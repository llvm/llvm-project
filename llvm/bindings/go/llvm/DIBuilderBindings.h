//===- DIBuilderBindings.h - Bindings for DIBuilder -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings for the DIBuilder class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H

#include "IRBindings.h"
#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: These bindings shouldn't be Go-specific and should eventually move to
// a (somewhat) less stable collection of C APIs for use in creating bindings of
// LLVM in other languages.

typedef struct LLVMOpaqueDIBuilder *LLVMDIBuilderRef;

LLVMDIBuilderRef LLVMNewDIBuilder(LLVMModuleRef m);

void LLVMDIBuilderDestroy(LLVMDIBuilderRef d);
void LLVMDIBuilderFinalize(LLVMDIBuilderRef d);

LLVMMetadataRef
LLVMDIBuilderCreateCompileUnit(LLVMDIBuilderRef D, unsigned Language,
                               const char *File, const char *Dir,
                               const char *Producer, int Optimized,
                               const char *Flags, unsigned RuntimeVersion);

LLVMMetadataRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef D, const char *File,
                                        const char *Dir);

LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(LLVMDIBuilderRef D,
                                                LLVMMetadataRef Scope,
                                                LLVMMetadataRef File,
                                                unsigned Line, unsigned Column);

LLVMMetadataRef LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef D,
                                                    LLVMMetadataRef Scope,
                                                    LLVMMetadataRef File,
                                                    unsigned Discriminator);

LLVMMetadataRef LLVMDIBuilderCreateFunction(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name,
    const char *LinkageName, LLVMMetadataRef File, unsigned Line,
    LLVMMetadataRef CompositeType, int IsLocalToUnit, int IsDefinition,
    unsigned ScopeLine, unsigned Flags, int IsOptimized);

LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty, int AlwaysPreserve,
    unsigned Flags, uint32_t AlignInBits);

LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name, unsigned ArgNo,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty, int AlwaysPreserve,
    unsigned Flags);

LLVMMetadataRef LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef D,
                                             const char *Name,
                                             uint64_t SizeInBits,
                                             unsigned Encoding);

LLVMMetadataRef LLVMDIBuilderCreatePointerType(LLVMDIBuilderRef D,
                                               LLVMMetadataRef PointeeType,
                                               uint64_t SizeInBits,
                                               uint32_t AlignInBits,
                                               const char *Name);

LLVMMetadataRef
LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef D, LLVMMetadataRef File,
                                  LLVMMetadataRef ParameterTypes);

LLVMMetadataRef LLVMDIBuilderCreateStructType(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned Line, uint64_t SizeInBits,
    uint32_t AlignInBits, unsigned Flags, LLVMMetadataRef DerivedFrom,
    LLVMMetadataRef ElementTypes);

LLVMMetadataRef LLVMDIBuilderCreateReplaceableCompositeType(
    LLVMDIBuilderRef D, unsigned Tag, const char *Name, LLVMMetadataRef Scope,
    LLVMMetadataRef File, unsigned Line, unsigned RuntimeLang,
    uint64_t SizeInBits, uint32_t AlignInBits, unsigned Flags);

LLVMMetadataRef
LLVMDIBuilderCreateMemberType(LLVMDIBuilderRef D, LLVMMetadataRef Scope,
                              const char *Name, LLVMMetadataRef File,
                              unsigned Line, uint64_t SizeInBits,
                              uint32_t AlignInBits, uint64_t OffsetInBits,
                              unsigned Flags, LLVMMetadataRef Ty);

LLVMMetadataRef LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef D,
                                             uint64_t SizeInBits,
                                             uint32_t AlignInBits,
                                             LLVMMetadataRef ElementType,
                                             LLVMMetadataRef Subscripts);

LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef D,
                                           LLVMMetadataRef Ty, const char *Name,
                                           LLVMMetadataRef File, unsigned Line,
                                           LLVMMetadataRef Context);

LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef D, int64_t Lo,
                                                 int64_t Count);

LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef D,
                                              LLVMMetadataRef *Data,
                                              size_t Length);

LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef D,
                                                  LLVMMetadataRef *Data,
                                                  size_t Length);

LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Dref,
                                              int64_t *Addr, size_t Length);

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(LLVMDIBuilderRef D,
                                             LLVMValueRef Storage,
                                             LLVMMetadataRef VarInfo,
                                             LLVMMetadataRef Expr,
                                             LLVMBasicBlockRef Block);

LLVMValueRef LLVMDIBuilderInsertValueAtEnd(LLVMDIBuilderRef D, LLVMValueRef Val,
                                           uint64_t Offset,
                                           LLVMMetadataRef VarInfo,
                                           LLVMMetadataRef Expr,
                                           LLVMBasicBlockRef Block);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
