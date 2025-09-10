//===- BuildLibCalls.h - Utility builder for libcalls -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to build some C language libcalls for
// optimization passes that need to call the various functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BUILDLIBCALLS_H
#define LLVM_TRANSFORMS_UTILS_BUILDLIBCALLS_H

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
  class Value;
  class DataLayout;
  class IRBuilderBase;

  /// Analyze the name and prototype of the given function and set any
  /// applicable attributes. Note that this merely helps optimizations on an
  /// already existing function but does not consider mandatory attributes.
  ///
  /// If the library function is unavailable, this doesn't modify it.
  ///
  /// Returns true if any attributes were set and false otherwise.
  LLVM_ABI bool inferNonMandatoryLibFuncAttrs(Module *M, StringRef Name,
                                              const TargetLibraryInfo &TLI);
  LLVM_ABI bool inferNonMandatoryLibFuncAttrs(Function &F,
                                              const TargetLibraryInfo &TLI);

  /// Calls getOrInsertFunction() and then makes sure to add mandatory
  /// argument attributes.
  LLVM_ABI FunctionCallee getOrInsertLibFunc(Module *M,
                                             const TargetLibraryInfo &TLI,
                                             LibFunc TheLibFunc,
                                             FunctionType *T,
                                             AttributeList AttributeList);
  LLVM_ABI FunctionCallee getOrInsertLibFunc(Module *M,
                                             const TargetLibraryInfo &TLI,
                                             LibFunc TheLibFunc,
                                             FunctionType *T);
  template <typename... ArgsTy>
  FunctionCallee getOrInsertLibFunc(Module *M, const TargetLibraryInfo &TLI,
                               LibFunc TheLibFunc, AttributeList AttributeList,
                               Type *RetTy, ArgsTy... Args) {
    SmallVector<Type*, sizeof...(ArgsTy)> ArgTys{Args...};
    return getOrInsertLibFunc(M, TLI, TheLibFunc,
                              FunctionType::get(RetTy, ArgTys, false),
                              AttributeList);
  }
  /// Same as above, but without the attributes.
  template <typename... ArgsTy>
  FunctionCallee getOrInsertLibFunc(Module *M, const TargetLibraryInfo &TLI,
                             LibFunc TheLibFunc, Type *RetTy, ArgsTy... Args) {
    return getOrInsertLibFunc(M, TLI, TheLibFunc, AttributeList{}, RetTy,
                              Args...);
  }
  // Avoid an incorrect ordering that'd otherwise compile incorrectly.
  template <typename... ArgsTy>
  FunctionCallee
  getOrInsertLibFunc(Module *M, const TargetLibraryInfo &TLI,
                     LibFunc TheLibFunc, AttributeList AttributeList,
                     FunctionType *Invalid, ArgsTy... Args) = delete;

  // Handle -mregparm for the given function.
  // Note that this function is a rough approximation that only works for simple
  // function signatures; it does not apply other relevant attributes for
  // function signatures, including sign/zero-extension for arguments and return
  // values.
  LLVM_ABI void markRegisterParameterAttributes(Function *F);

  /// Check whether the library function is available on target and also that
  /// it in the current Module is a Function with the right type.
  LLVM_ABI bool isLibFuncEmittable(const Module *M,
                                   const TargetLibraryInfo *TLI,
                                   LibFunc TheLibFunc);
  LLVM_ABI bool isLibFuncEmittable(const Module *M,
                                   const TargetLibraryInfo *TLI,
                                   StringRef Name);

  /// Check whether the overloaded floating point function
  /// corresponding to \a Ty is available.
  LLVM_ABI bool hasFloatFn(const Module *M, const TargetLibraryInfo *TLI,
                           Type *Ty, LibFunc DoubleFn, LibFunc FloatFn,
                           LibFunc LongDoubleFn);

  /// Get the name of the overloaded floating point function
  /// corresponding to \a Ty. Return the LibFunc in \a TheLibFunc.
  LLVM_ABI StringRef getFloatFn(const Module *M, const TargetLibraryInfo *TLI,
                                Type *Ty, LibFunc DoubleFn, LibFunc FloatFn,
                                LibFunc LongDoubleFn, LibFunc &TheLibFunc);

  /// Emit a call to the strlen function to the builder, for the specified
  /// pointer. Ptr is required to be some pointer type, and the return value has
  /// 'size_t' type.
  LLVM_ABI Value *emitStrLen(Value *Ptr, IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the wcslen function to the builder, for the specified
  /// pointer. Ptr is required to be some pointer type, and the return value has
  /// 'size_t' type.
  LLVM_ABI Value *emitWcsLen(Value *Ptr, IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the strdup function to the builder, for the specified
  /// pointer. Ptr is required to be some pointer type, and the return value has
  /// 'i8*' type.
  LLVM_ABI Value *emitStrDup(Value *Ptr, IRBuilderBase &B,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the strchr function to the builder, for the specified
  /// pointer and character. Ptr is required to be some pointer type, and the
  /// return value has 'i8*' type.
  LLVM_ABI Value *emitStrChr(Value *Ptr, char C, IRBuilderBase &B,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the strncmp function to the builder.
  LLVM_ABI Value *emitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len,
                              IRBuilderBase &B, const DataLayout &DL,
                              const TargetLibraryInfo *TLI);

  /// Emit a call to the strcpy function to the builder, for the specified
  /// pointer arguments.
  LLVM_ABI Value *emitStrCpy(Value *Dst, Value *Src, IRBuilderBase &B,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the stpcpy function to the builder, for the specified
  /// pointer arguments.
  LLVM_ABI Value *emitStpCpy(Value *Dst, Value *Src, IRBuilderBase &B,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the strncpy function to the builder, for the specified
  /// pointer arguments and length.
  LLVM_ABI Value *emitStrNCpy(Value *Dst, Value *Src, Value *Len,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the stpncpy function to the builder, for the specified
  /// pointer arguments and length.
  LLVM_ABI Value *emitStpNCpy(Value *Dst, Value *Src, Value *Len,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the __memcpy_chk function to the builder. This expects that
  /// the Len and ObjSize have type 'size_t' and Dst/Src are pointers.
  LLVM_ABI Value *emitMemCpyChk(Value *Dst, Value *Src, Value *Len,
                                Value *ObjSize, IRBuilderBase &B,
                                const DataLayout &DL,
                                const TargetLibraryInfo *TLI);

  /// Emit a call to the mempcpy function.
  LLVM_ABI Value *emitMemPCpy(Value *Dst, Value *Src, Value *Len,
                              IRBuilderBase &B, const DataLayout &DL,
                              const TargetLibraryInfo *TLI);

  /// Emit a call to the memchr function. This assumes that Ptr is a pointer,
  /// Val is an 'int' value, and Len is an 'size_t' value.
  LLVM_ABI Value *emitMemChr(Value *Ptr, Value *Val, Value *Len,
                             IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the memrchr function, analogously to emitMemChr.
  LLVM_ABI Value *emitMemRChr(Value *Ptr, Value *Val, Value *Len,
                              IRBuilderBase &B, const DataLayout &DL,
                              const TargetLibraryInfo *TLI);

  /// Emit a call to the memcmp function.
  LLVM_ABI Value *emitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len,
                             IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the bcmp function.
  LLVM_ABI Value *emitBCmp(Value *Ptr1, Value *Ptr2, Value *Len,
                           IRBuilderBase &B, const DataLayout &DL,
                           const TargetLibraryInfo *TLI);

  /// Emit a call to the memccpy function.
  LLVM_ABI Value *emitMemCCpy(Value *Ptr1, Value *Ptr2, Value *Val, Value *Len,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the snprintf function.
  LLVM_ABI Value *emitSNPrintf(Value *Dest, Value *Size, Value *Fmt,
                               ArrayRef<Value *> Args, IRBuilderBase &B,
                               const TargetLibraryInfo *TLI);

  /// Emit a call to the sprintf function.
  LLVM_ABI Value *emitSPrintf(Value *Dest, Value *Fmt,
                              ArrayRef<Value *> VariadicArgs, IRBuilderBase &B,
                              const TargetLibraryInfo *TLI);

  /// Emit a call to the strcat function.
  LLVM_ABI Value *emitStrCat(Value *Dest, Value *Src, IRBuilderBase &B,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the strlcpy function.
  LLVM_ABI Value *emitStrLCpy(Value *Dest, Value *Src, Value *Size,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the strlcat function.
  LLVM_ABI Value *emitStrLCat(Value *Dest, Value *Src, Value *Size,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the strncat function.
  LLVM_ABI Value *emitStrNCat(Value *Dest, Value *Src, Value *Size,
                              IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the vsnprintf function.
  LLVM_ABI Value *emitVSNPrintf(Value *Dest, Value *Size, Value *Fmt,
                                Value *VAList, IRBuilderBase &B,
                                const TargetLibraryInfo *TLI);

  /// Emit a call to the vsprintf function.
  LLVM_ABI Value *emitVSPrintf(Value *Dest, Value *Fmt, Value *VAList,
                               IRBuilderBase &B, const TargetLibraryInfo *TLI);

  /// Emit a call to the unary function named 'Name' (e.g.  'floor'). This
  /// function is known to take a single of type matching 'Op' and returns one
  /// value with the same type. If 'Op' is a long double, 'l' is added as the
  /// suffix of name, if 'Op' is a float, we add a 'f' suffix.
  LLVM_ABI Value *emitUnaryFloatFnCall(Value *Op, const TargetLibraryInfo *TLI,
                                       StringRef Name, IRBuilderBase &B,
                                       const AttributeList &Attrs);

  /// Emit a call to the unary function DoubleFn, FloatFn or LongDoubleFn,
  /// depending of the type of Op.
  LLVM_ABI Value *emitUnaryFloatFnCall(Value *Op, const TargetLibraryInfo *TLI,
                                       LibFunc DoubleFn, LibFunc FloatFn,
                                       LibFunc LongDoubleFn, IRBuilderBase &B,
                                       const AttributeList &Attrs);

  /// Emit a call to the binary function named 'Name' (e.g. 'fmin'). This
  /// function is known to take type matching 'Op1' and 'Op2' and return one
  /// value with the same type. If 'Op1/Op2' are long double, 'l' is added as
  /// the suffix of name, if 'Op1/Op2' are float, we add a 'f' suffix.
  LLVM_ABI Value *emitBinaryFloatFnCall(Value *Op1, Value *Op2,
                                        const TargetLibraryInfo *TLI,
                                        StringRef Name, IRBuilderBase &B,
                                        const AttributeList &Attrs);

  /// Emit a call to the binary function DoubleFn, FloatFn or LongDoubleFn,
  /// depending of the type of Op1.
  LLVM_ABI Value *emitBinaryFloatFnCall(Value *Op1, Value *Op2,
                                        const TargetLibraryInfo *TLI,
                                        LibFunc DoubleFn, LibFunc FloatFn,
                                        LibFunc LongDoubleFn, IRBuilderBase &B,
                                        const AttributeList &Attrs);

  /// Emit a call to the putchar function. This assumes that Char is an 'int'.
  LLVM_ABI Value *emitPutChar(Value *Char, IRBuilderBase &B,
                              const TargetLibraryInfo *TLI);

  /// Emit a call to the puts function. This assumes that Str is some pointer.
  LLVM_ABI Value *emitPutS(Value *Str, IRBuilderBase &B,
                           const TargetLibraryInfo *TLI);

  /// Emit a call to the fputc function. This assumes that Char is an 'int', and
  /// File is a pointer to FILE.
  LLVM_ABI Value *emitFPutC(Value *Char, Value *File, IRBuilderBase &B,
                            const TargetLibraryInfo *TLI);

  /// Emit a call to the fputs function. Str is required to be a pointer and
  /// File is a pointer to FILE.
  LLVM_ABI Value *emitFPutS(Value *Str, Value *File, IRBuilderBase &B,
                            const TargetLibraryInfo *TLI);

  /// Emit a call to the fwrite function. This assumes that Ptr is a pointer,
  /// Size is an 'size_t', and File is a pointer to FILE.
  LLVM_ABI Value *emitFWrite(Value *Ptr, Value *Size, Value *File,
                             IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the malloc function.
  LLVM_ABI Value *emitMalloc(Value *Num, IRBuilderBase &B, const DataLayout &DL,
                             const TargetLibraryInfo *TLI);

  /// Emit a call to the calloc function.
  LLVM_ABI Value *emitCalloc(Value *Num, Value *Size, IRBuilderBase &B,
                             const TargetLibraryInfo &TLI, unsigned AddrSpace);

  /// Emit a call to the hot/cold operator new function.
  LLVM_ABI Value *emitHotColdNew(Value *Num, IRBuilderBase &B,
                                 const TargetLibraryInfo *TLI, LibFunc NewFunc,
                                 uint8_t HotCold);
  LLVM_ABI Value *emitHotColdNewNoThrow(Value *Num, Value *NoThrow,
                                        IRBuilderBase &B,
                                        const TargetLibraryInfo *TLI,
                                        LibFunc NewFunc, uint8_t HotCold);
  LLVM_ABI Value *emitHotColdNewAligned(Value *Num, Value *Align,
                                        IRBuilderBase &B,
                                        const TargetLibraryInfo *TLI,
                                        LibFunc NewFunc, uint8_t HotCold);
  LLVM_ABI Value *emitHotColdNewAlignedNoThrow(Value *Num, Value *Align,
                                               Value *NoThrow, IRBuilderBase &B,
                                               const TargetLibraryInfo *TLI,
                                               LibFunc NewFunc,
                                               uint8_t HotCold);
  LLVM_ABI Value *emitHotColdSizeReturningNew(Value *Num, IRBuilderBase &B,
                                              const TargetLibraryInfo *TLI,
                                              LibFunc NewFunc, uint8_t HotCold);
  LLVM_ABI Value *
  emitHotColdSizeReturningNewAligned(Value *Num, Value *Align, IRBuilderBase &B,
                                     const TargetLibraryInfo *TLI,
                                     LibFunc NewFunc, uint8_t HotCold);
}

#endif
