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
