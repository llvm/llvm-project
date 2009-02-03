//===-- Target.cpp --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C bindings for libLLVMTarget.a, which implements
// target information.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Target.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetData.h"
#include <cstring>

using namespace llvm;

LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep) {
  return wrap(new TargetData(StringRep));
}

void LLVMAddTargetData(LLVMTargetDataRef TD, LLVMPassManagerRef PM) {
  unwrap(PM)->add(new TargetData(*unwrap(TD)));
}

char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD) {
  std::string StringRep = unwrap(TD)->getStringRepresentation();
  return strdup(StringRep.c_str());
}

LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD) {
  return unwrap(TD)->isLittleEndian();
}

unsigned LLVMPointerSize(LLVMTargetDataRef TD) {
  return unwrap(TD)->getPointerSize();
}

LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD) {
  return wrap(unwrap(TD)->getIntPtrType());
}

unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypeSizeInBits(unwrap(Ty));
}

unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypeStoreSize(unwrap(Ty));
}

unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypePaddedSize(unwrap(Ty));
}

unsigned LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getABITypeAlignment(unwrap(Ty));
}

unsigned LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getCallFrameTypeAlignment(unwrap(Ty));
}

unsigned LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getPrefTypeAlignment(unwrap(Ty));
}

unsigned LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD,
                                        LLVMValueRef GlobalVar) {
  return unwrap(TD)->getPreferredAlignment(unwrap<GlobalVariable>(GlobalVar));
}

unsigned LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy,
                             unsigned long long Offset) {
  const StructType *STy = unwrap<StructType>(StructTy);
  return unwrap(TD)->getStructLayout(STy)->getElementContainingOffset(Offset);
}

unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy,
                                       unsigned Element) {
  const StructType *STy = unwrap<StructType>(StructTy);
  return unwrap(TD)->getStructLayout(STy)->getElementOffset(Element);
}

void LLVMInvalidateStructLayout(LLVMTargetDataRef TD, LLVMTypeRef StructTy) {
  unwrap(TD)->InvalidateStructLayoutInfo(unwrap<StructType>(StructTy));
}

void LLVMDisposeTargetData(LLVMTargetDataRef TD) {
  delete unwrap(TD);
}
