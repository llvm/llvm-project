/*===-- llvm-c/Target.h - Target Lib C Iface --------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMTarget.a, which             *|
|* implements target information.                                             *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TARGET_H
#define LLVM_C_TARGET_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { LLVMBigEndian, LLVMLittleEndian };
typedef int LLVMByteOrdering;

typedef struct LLVMOpaqueTargetData *LLVMTargetDataRef;
typedef struct LLVMStructLayout *LLVMStructLayoutRef;


/*===-- Target Data -------------------------------------------------------===*/

/** Creates target data from a target layout string.
    See the constructor llvm::TargetData::TargetData. */
LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep);

/** Adds target data information to a pass manager. This does not take ownership
    of the target data.
    See the method llvm::PassManagerBase::add. */
void LLVMAddTargetData(LLVMTargetDataRef, LLVMPassManagerRef);

/** Converts target data to a target layout string. The string must be disposed
    with LLVMDisposeMessage.
    See the constructor llvm::TargetData::TargetData. */
char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef);

/** Returns the byte order of a target, either LLVMBigEndian or
    LLVMLittleEndian.
    See the method llvm::TargetData::isLittleEndian. */
LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef);

/** Returns the pointer size in bytes for a target.
    See the method llvm::TargetData::getPointerSize. */
unsigned LLVMPointerSize(LLVMTargetDataRef);

/** Returns the integer type that is the same size as a pointer on a target.
    See the method llvm::TargetData::getIntPtrType. */
LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef);

/** Computes the size of a type in bytes for a target.
    See the method llvm::TargetData::getTypeSizeInBits. */
unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the storage size of a type in bytes for a target.
    See the method llvm::TargetData::getTypeStoreSize. */
unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the ABI size of a type in bytes for a target.
    See the method llvm::TargetData::getTypePaddedSize. */
unsigned long long LLVMABISizeOfType(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the ABI alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. */
unsigned LLVMABIAlignmentOfType(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the call frame alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. */
unsigned LLVMCallFrameAlignmentOfType(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the preferred alignment of a type in bytes for a target.
    See the method llvm::TargetData::getTypeABISize. */
unsigned LLVMPreferredAlignmentOfType(LLVMTargetDataRef, LLVMTypeRef);

/** Computes the preferred alignment of a global variable in bytes for a target.
    See the method llvm::TargetData::getPreferredAlignment. */
unsigned LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef,
                                        LLVMValueRef GlobalVar);

/** Computes the structure element that contains the byte offset for a target.
    See the method llvm::StructLayout::getElementContainingOffset. */
unsigned LLVMElementAtOffset(LLVMTargetDataRef, LLVMTypeRef StructTy,
                             unsigned long long Offset);

/** Computes the byte offset of the indexed struct element for a target.
    See the method llvm::StructLayout::getElementContainingOffset. */
unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef, LLVMTypeRef StructTy,
                                       unsigned Element);

/** Struct layouts are speculatively cached. If a TargetDataRef is alive when
    types are being refined and removed, this method must be called whenever a
    struct type is removed to avoid a dangling pointer in this cache.
    See the method llvm::TargetData::InvalidateStructLayoutInfo. */
void LLVMInvalidateStructLayout(LLVMTargetDataRef, LLVMTypeRef StructTy);

/** Deallocates a TargetData.
    See the destructor llvm::TargetData::~TargetData. */
void LLVMDisposeTargetData(LLVMTargetDataRef);


#ifdef __cplusplus
}

namespace llvm {
  class TargetData;

  inline TargetData *unwrap(LLVMTargetDataRef P) {
    return reinterpret_cast<TargetData*>(P);
  }
  
  inline LLVMTargetDataRef wrap(const TargetData *P) {
    return reinterpret_cast<LLVMTargetDataRef>(const_cast<TargetData*>(P));
  }
}

#endif /* defined(__cplusplus) */

#endif
