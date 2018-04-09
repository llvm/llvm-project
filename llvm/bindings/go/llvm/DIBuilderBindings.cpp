//===- DIBuilderBindings.cpp - Bindings for DIBuilder ---------------------===//
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

#include "DIBuilderBindings.h"
#include "IRBindings.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

using namespace llvm;

LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(
    LLVMDIBuilderRef Dref, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty, int AlwaysPreserve,
    unsigned Flags, uint32_t AlignInBits) {
  DIBuilder *D = unwrap(Dref);
  return wrap(
      D->createAutoVariable(unwrap<DIScope>(Scope), Name, unwrap<DIFile>(File),
                            Line, unwrap<DIType>(Ty), AlwaysPreserve,
                            static_cast<DINode::DIFlags>(Flags), AlignInBits));
}

LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(
    LLVMDIBuilderRef Dref, LLVMMetadataRef Scope, const char *Name,
    unsigned ArgNo, LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty,
    int AlwaysPreserve, unsigned Flags) {
  DIBuilder *D = unwrap(Dref);
  return wrap(D->createParameterVariable(
      unwrap<DIScope>(Scope), Name, ArgNo, unwrap<DIFile>(File), Line,
      unwrap<DIType>(Ty), AlwaysPreserve, static_cast<DINode::DIFlags>(Flags)));
}

LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef Dref,
                                           LLVMMetadataRef Ty, const char *Name,
                                           LLVMMetadataRef File, unsigned Line,
                                           LLVMMetadataRef Context) {
  DIBuilder *D = unwrap(Dref);
  return wrap(D->createTypedef(unwrap<DIType>(Ty), Name,
                               File ? unwrap<DIFile>(File) : nullptr, Line,
                               Context ? unwrap<DIScope>(Context) : nullptr));
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef Dref,
                                                 int64_t Lo, int64_t Count) {
  DIBuilder *D = unwrap(Dref);
  return wrap(D->getOrCreateSubrange(Lo, Count));
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef Dref,
                                              LLVMMetadataRef *Data,
                                              size_t Length) {
  DIBuilder *D = unwrap(Dref);
  Metadata **DataValue = unwrap(Data);
  ArrayRef<Metadata *> Elements(DataValue, Length);
  DINodeArray A = D->getOrCreateArray(Elements);
  return wrap(A.get());
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef Dref,
                                                  LLVMMetadataRef *Data,
                                                  size_t Length) {
  DIBuilder *D = unwrap(Dref);
  Metadata **DataValue = unwrap(Data);
  ArrayRef<Metadata *> Elements(DataValue, Length);
  DITypeRefArray A = D->getOrCreateTypeArray(Elements);
  return wrap(A.get());
}

LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Dref,
                                              int64_t *Addr, size_t Length) {
  DIBuilder *D = unwrap(Dref);
  return wrap(D->createExpression(ArrayRef<int64_t>(Addr, Length)));
}

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(LLVMDIBuilderRef Dref,
                                             LLVMValueRef Storage,
                                             LLVMMetadataRef VarInfo,
                                             LLVMMetadataRef Expr,
                                             LLVMBasicBlockRef Block) {
  // Fail immediately here until the llgo folks update their bindings.  The
  // called function is going to assert out anyway.
  llvm_unreachable("DIBuilder API change requires a DebugLoc");

  DIBuilder *D = unwrap(Dref);
  Instruction *Instr = D->insertDeclare(
      unwrap(Storage), unwrap<DILocalVariable>(VarInfo),
      unwrap<DIExpression>(Expr), /* DebugLoc */ nullptr, unwrap(Block));
  return wrap(Instr);
}

LLVMValueRef LLVMDIBuilderInsertValueAtEnd(LLVMDIBuilderRef Dref,
                                           LLVMValueRef Val,
                                           LLVMMetadataRef VarInfo,
                                           LLVMMetadataRef Expr,
                                           LLVMBasicBlockRef Block) {
  // Fail immediately here until the llgo folks update their bindings.  The
  // called function is going to assert out anyway.
  llvm_unreachable("DIBuilder API change requires a DebugLoc");

  DIBuilder *D = unwrap(Dref);
  Instruction *Instr = D->insertDbgValueIntrinsic(
      unwrap(Val), unwrap<DILocalVariable>(VarInfo), unwrap<DIExpression>(Expr),
      /* DebugLoc */ nullptr, unwrap(Block));
  return wrap(Instr);
}
