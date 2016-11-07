//===- IRBindings.cpp - Additional bindings for ir ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the ir component.
//
//===----------------------------------------------------------------------===//

#include "IRBindings.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

LLVMMetadataRef LLVMConstantAsMetadata(LLVMValueRef C) {
  return wrap(ConstantAsMetadata::get(unwrap<Constant>(C)));
}

LLVMMetadataRef LLVMMDString2(LLVMContextRef C, const char *Str, unsigned SLen) {
  return wrap(MDString::get(*unwrap(C), StringRef(Str, SLen)));
}

LLVMMetadataRef LLVMMDNode2(LLVMContextRef C, LLVMMetadataRef *MDs,
                            unsigned Count) {
  return wrap(
      MDNode::get(*unwrap(C), ArrayRef<Metadata *>(unwrap(MDs), Count)));
}

LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef C, LLVMMetadataRef *MDs,
                                    unsigned Count) {
  return wrap(MDTuple::getTemporary(*unwrap(C),
                                    ArrayRef<Metadata *>(unwrap(MDs), Count))
                  .release());
}

void LLVMAddNamedMetadataOperand2(LLVMModuleRef M, const char *name,
                                  LLVMMetadataRef Val) {
  NamedMDNode *N = unwrap(M)->getOrInsertNamedMetadata(name);
  if (!N)
    return;
  if (!Val)
    return;
  N->addOperand(unwrap<MDNode>(Val));
}

void LLVMSetMetadata2(LLVMValueRef Inst, unsigned KindID, LLVMMetadataRef MD) {
  MDNode *N = MD ? unwrap<MDNode>(MD) : nullptr;
  unwrap<Instruction>(Inst)->setMetadata(KindID, N);
}

void LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef MD, LLVMMetadataRef New) {
  auto *Node = unwrap<MDNode>(MD);
  Node->replaceAllUsesWith(unwrap<Metadata>(New));
  MDNode::deleteTemporary(Node);
}

void LLVMSetCurrentDebugLocation2(LLVMBuilderRef Bref, unsigned Line,
                                  unsigned Col, LLVMMetadataRef Scope,
                                  LLVMMetadataRef InlinedAt) {
  unwrap(Bref)->SetCurrentDebugLocation(
      DebugLoc::get(Line, Col, Scope ? unwrap<MDNode>(Scope) : nullptr,
                    InlinedAt ? unwrap<MDNode>(InlinedAt) : nullptr));
}

void LLVMSetSubprogram(LLVMValueRef Func, LLVMMetadataRef SP) {
  unwrap<Function>(Func)->setSubprogram(unwrap<DISubprogram>(SP));
}
