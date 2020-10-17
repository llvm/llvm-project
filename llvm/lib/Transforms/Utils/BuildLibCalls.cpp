//===- BuildLibCalls.cpp - Utility builder for libcalls -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some functions that will create standard C libcalls.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Analysis/MemoryBuiltins.h"

using namespace llvm;

#define DEBUG_TYPE "build-libcalls"

//- Infer Attributes ---------------------------------------------------------//

STATISTIC(NumReadNone, "Number of functions inferred as readnone");
STATISTIC(NumReadOnly, "Number of functions inferred as readonly");
STATISTIC(NumArgMemOnly, "Number of functions inferred as argmemonly");
STATISTIC(NumNoUnwind, "Number of functions inferred as nounwind");
STATISTIC(NumNoCapture, "Number of arguments inferred as nocapture");
STATISTIC(NumReadOnlyArg, "Number of arguments inferred as readonly");
STATISTIC(NumNoAlias, "Number of function returns inferred as noalias");
STATISTIC(NumNoUndef, "Number of function returns inferred as noundef returns");
STATISTIC(NumNonNull, "Number of function returns inferred as nonnull returns");
STATISTIC(NumReturnedArg, "Number of arguments inferred as returned");

static bool setDoesNotAccessMemory(Function &F) {
  if (F.doesNotAccessMemory())
    return false;
  F.setDoesNotAccessMemory();
  ++NumReadNone;
  return true;
}

static bool setOnlyReadsMemory(Function &F) {
  if (F.onlyReadsMemory())
    return false;
  F.setOnlyReadsMemory();
  ++NumReadOnly;
  return true;
}

static bool setOnlyAccessesArgMemory(Function &F) {
  if (F.onlyAccessesArgMemory())
    return false;
  F.setOnlyAccessesArgMemory();
  ++NumArgMemOnly;
  return true;
}

static bool setDoesNotThrow(Function &F) {
  if (F.doesNotThrow())
    return false;
  F.setDoesNotThrow();
  ++NumNoUnwind;
  return true;
}

static bool setRetDoesNotAlias(Function &F) {
  if (F.hasAttribute(AttributeList::ReturnIndex, Attribute::NoAlias))
    return false;
  F.addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
  ++NumNoAlias;
  return true;
}

static bool setDoesNotCapture(Function &F, unsigned ArgNo) {
  if (F.hasParamAttribute(ArgNo, Attribute::NoCapture))
    return false;
  F.addParamAttr(ArgNo, Attribute::NoCapture);
  ++NumNoCapture;
  return true;
}

static bool setDoesNotAlias(Function &F, unsigned ArgNo) {
  if (F.hasParamAttribute(ArgNo, Attribute::NoAlias))
    return false;
  F.addParamAttr(ArgNo, Attribute::NoAlias);
  ++NumNoAlias;
  return true;
}

static bool setOnlyReadsMemory(Function &F, unsigned ArgNo) {
  if (F.hasParamAttribute(ArgNo, Attribute::ReadOnly))
    return false;
  F.addParamAttr(ArgNo, Attribute::ReadOnly);
  ++NumReadOnlyArg;
  return true;
}

static bool setRetNoUndef(Function &F) {
  if (!F.getReturnType()->isVoidTy() &&
      !F.hasAttribute(AttributeList::ReturnIndex, Attribute::NoUndef)) {
    F.addAttribute(AttributeList::ReturnIndex, Attribute::NoUndef);
    ++NumNoUndef;
    return true;
  }
  return false;
}

static bool setArgsNoUndef(Function &F) {
  bool Changed = false;
  for (unsigned ArgNo = 0; ArgNo < F.arg_size(); ++ArgNo) {
    if (!F.hasParamAttribute(ArgNo, Attribute::NoUndef)) {
      F.addParamAttr(ArgNo, Attribute::NoUndef);
      ++NumNoUndef;
      Changed = true;
    }
  }
  return Changed;
}

static bool setRetAndArgsNoUndef(Function &F) {
  return setRetNoUndef(F) | setArgsNoUndef(F);
}

static bool setRetNonNull(Function &F) {
  assert(F.getReturnType()->isPointerTy() &&
         "nonnull applies only to pointers");
  if (F.hasAttribute(AttributeList::ReturnIndex, Attribute::NonNull))
    return false;
  F.addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
  ++NumNonNull;
  return true;
}

static bool setReturnedArg(Function &F, unsigned ArgNo) {
  if (F.hasParamAttribute(ArgNo, Attribute::Returned))
    return false;
  F.addParamAttr(ArgNo, Attribute::Returned);
  ++NumReturnedArg;
  return true;
}

static bool setNonLazyBind(Function &F) {
  if (F.hasFnAttribute(Attribute::NonLazyBind))
    return false;
  F.addFnAttr(Attribute::NonLazyBind);
  return true;
}

static bool setDoesNotFreeMemory(Function &F) {
  if (F.hasFnAttribute(Attribute::NoFree))
    return false;
  F.addFnAttr(Attribute::NoFree);
  return true;
}

bool llvm::inferLibFuncAttributes(Module *M, StringRef Name,
                                  const TargetLibraryInfo &TLI) {
  Function *F = M->getFunction(Name);
  if (!F)
    return false;
  return inferLibFuncAttributes(*F, TLI);
}

bool llvm::inferLibFuncAttributes(Function &F, const TargetLibraryInfo &TLI) {
  LibFunc TheLibFunc;
  if (!(TLI.getLibFunc(F, TheLibFunc) && TLI.has(TheLibFunc)))
    return false;

  bool Changed = false;

  if(!isLibFreeFunction(&F, TheLibFunc) && !isReallocLikeFn(&F,  &TLI))
    Changed |= setDoesNotFreeMemory(F);

  if (F.getParent() != nullptr && F.getParent()->getRtLibUseGOT())
    Changed |= setNonLazyBind(F);

  switch (TheLibFunc) {
  case LibFunc_strlen:
  case LibFunc_wcslen:
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_strchr:
  case LibFunc_strrchr:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc_strtol:
  case LibFunc_strtod:
  case LibFunc_strtof:
  case LibFunc_strtoul:
  case LibFunc_strtoll:
  case LibFunc_strtold:
  case LibFunc_strtoull:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_strcpy:
  case LibFunc_strncpy:
  case LibFunc_strcat:
  case LibFunc_strncat:
    Changed |= setReturnedArg(F, 0);
    LLVM_FALLTHROUGH;
  case LibFunc_stpcpy:
  case LibFunc_stpncpy:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotAlias(F, 1);
    return Changed;
  case LibFunc_strxfrm:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_strcmp:      // 0,1
  case LibFunc_strspn:      // 0,1
  case LibFunc_strncmp:     // 0,1
  case LibFunc_strcspn:     // 0,1
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_strcoll:
  case LibFunc_strcasecmp:  // 0,1
  case LibFunc_strncasecmp: //
    // Those functions may depend on the locale, which may be accessed through
    // global memory.
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_strstr:
  case LibFunc_strpbrk:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_strtok:
  case LibFunc_strtok_r:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_scanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_setbuf:
  case LibFunc_setvbuf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_strdup:
  case LibFunc_strndup:
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_stat:
  case LibFunc_statvfs:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_sscanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_sprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_snprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc_setitimer:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_system:
    // May throw; "system" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_malloc:
    Changed |= setRetNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_memcmp:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_memchr:
  case LibFunc_memrchr:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    return Changed;
  case LibFunc_modf:
  case LibFunc_modff:
  case LibFunc_modfl:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_memcpy:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setReturnedArg(F, 0);
    Changed |= setDoesNotAlias(F, 1);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_memmove:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setReturnedArg(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_mempcpy:
  case LibFunc_memccpy:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotAlias(F, 1);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_memcpy_chk:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc_memalign:
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_mkdir:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_mktime:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_realloc:
    Changed |= setRetNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_reallocf:
    Changed |= setRetNoUndef(F);
    return Changed;
  case LibFunc_read:
    // May throw; "read" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_rewind:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_rmdir:
  case LibFunc_remove:
  case LibFunc_realpath:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_rename:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_readlink:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_write:
    // May throw; "write" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_aligned_alloc:
    Changed |= setRetNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_bcopy:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_bcmp:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_bzero:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_calloc:
    Changed |= setRetNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_chmod:
  case LibFunc_chown:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_ctermid:
  case LibFunc_clearerr:
  case LibFunc_closedir:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_atoi:
  case LibFunc_atol:
  case LibFunc_atof:
  case LibFunc_atoll:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_access:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_fopen:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_fdopen:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_feof:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_free:
    Changed |= setArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_fseek:
  case LibFunc_ftell:
  case LibFunc_fgetc:
  case LibFunc_fgetc_unlocked:
  case LibFunc_fseeko:
  case LibFunc_ftello:
  case LibFunc_fileno:
  case LibFunc_fflush:
  case LibFunc_fclose:
  case LibFunc_fsetpos:
  case LibFunc_flockfile:
  case LibFunc_funlockfile:
  case LibFunc_ftrylockfile:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_ferror:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F);
    return Changed;
  case LibFunc_fputc:
  case LibFunc_fputc_unlocked:
  case LibFunc_fstat:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_frexp:
  case LibFunc_frexpf:
  case LibFunc_frexpl:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_fstatvfs:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_fgets:
  case LibFunc_fgets_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc_fread:
  case LibFunc_fread_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 3);
    return Changed;
  case LibFunc_fwrite:
  case LibFunc_fwrite_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 3);
    // FIXME: readonly #1?
    return Changed;
  case LibFunc_fputs:
  case LibFunc_fputs_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_fscanf:
  case LibFunc_fprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_fgetpos:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_getc:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_getlogin_r:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_getc_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_getenv:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_gets:
  case LibFunc_getchar:
  case LibFunc_getchar_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc_getitimer:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_getpwnam:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_ungetc:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_uname:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_unlink:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_unsetenv:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_utime:
  case LibFunc_utimes:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_putc:
  case LibFunc_putc_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_puts:
  case LibFunc_printf:
  case LibFunc_perror:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_pread:
    // May throw; "pread" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_pwrite:
    // May throw; "pwrite" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_putchar:
  case LibFunc_putchar_unlocked:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc_popen:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_pclose:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_vscanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_vsscanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_vfscanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_valloc:
    Changed |= setRetNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_vprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_vfprintf:
  case LibFunc_vsprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_vsnprintf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc_open:
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_opendir:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_tmpfile:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_times:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_htonl:
  case LibFunc_htons:
  case LibFunc_ntohl:
  case LibFunc_ntohs:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAccessMemory(F);
    return Changed;
  case LibFunc_lstat:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_lchown:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_qsort:
    // May throw; places call through function pointer.
    // Cannot give undef pointer/size
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 3);
    return Changed;
  case LibFunc_dunder_strdup:
  case LibFunc_dunder_strndup:
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_dunder_strtok_r:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_under_IO_getc:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_under_IO_putc:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_dunder_isoc99_scanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_stat64:
  case LibFunc_lstat64:
  case LibFunc_statvfs64:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_dunder_isoc99_sscanf:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_fopen64:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 0);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc_fseeko64:
  case LibFunc_ftello64:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    return Changed;
  case LibFunc_tmpfile64:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  case LibFunc_fstat64:
  case LibFunc_fstatvfs64:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_open64:
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setOnlyReadsMemory(F, 0);
    return Changed;
  case LibFunc_gettimeofday:
    // Currently some platforms have the restrict keyword on the arguments to
    // gettimeofday. To be conservative, do not add noalias to gettimeofday's
    // arguments.
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc_Znwj: // new(unsigned int)
  case LibFunc_Znwm: // new(unsigned long)
  case LibFunc_Znaj: // new[](unsigned int)
  case LibFunc_Znam: // new[](unsigned long)
  case LibFunc_msvc_new_int: // new(unsigned int)
  case LibFunc_msvc_new_longlong: // new(unsigned long long)
  case LibFunc_msvc_new_array_int: // new[](unsigned int)
  case LibFunc_msvc_new_array_longlong: // new[](unsigned long long)
    // Operator new always returns a nonnull noalias pointer
    Changed |= setRetNoUndef(F);
    Changed |= setRetNonNull(F);
    Changed |= setRetDoesNotAlias(F);
    return Changed;
  // TODO: add LibFunc entries for:
  // case LibFunc_memset_pattern4:
  // case LibFunc_memset_pattern8:
  case LibFunc_memset_pattern16:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotCapture(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  // int __nvvm_reflect(const char *)
  case LibFunc_nvvm_reflect:
    Changed |= setRetAndArgsNoUndef(F);
    Changed |= setDoesNotAccessMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;

  default:
    // FIXME: It'd be really nice to cover all the library functions we're
    // aware of here.
    return false;
  }
}

bool llvm::hasFloatFn(const TargetLibraryInfo *TLI, Type *Ty,
                      LibFunc DoubleFn, LibFunc FloatFn, LibFunc LongDoubleFn) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    return false;
  case Type::FloatTyID:
    return TLI->has(FloatFn);
  case Type::DoubleTyID:
    return TLI->has(DoubleFn);
  default:
    return TLI->has(LongDoubleFn);
  }
}

StringRef llvm::getFloatFnName(const TargetLibraryInfo *TLI, Type *Ty,
                               LibFunc DoubleFn, LibFunc FloatFn,
                               LibFunc LongDoubleFn) {
  assert(hasFloatFn(TLI, Ty, DoubleFn, FloatFn, LongDoubleFn) &&
         "Cannot get name for unavailable function!");

  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    llvm_unreachable("No name for HalfTy!");
  case Type::FloatTyID:
    return TLI->getName(FloatFn);
  case Type::DoubleTyID:
    return TLI->getName(DoubleFn);
  default:
    return TLI->getName(LongDoubleFn);
  }
}

//- Emit LibCalls ------------------------------------------------------------//

Value *llvm::castToCStr(Value *V, IRBuilderBase &B) {
  unsigned AS = V->getType()->getPointerAddressSpace();
  return B.CreateBitCast(V, B.getInt8PtrTy(AS), "cstr");
}

static Value *emitLibCall(LibFunc TheLibFunc, Type *ReturnType,
                          ArrayRef<Type *> ParamTypes,
                          ArrayRef<Value *> Operands, IRBuilderBase &B,
                          const TargetLibraryInfo *TLI,
                          bool IsVaArgs = false) {
  if (!TLI->has(TheLibFunc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef FuncName = TLI->getName(TheLibFunc);
  FunctionType *FuncType = FunctionType::get(ReturnType, ParamTypes, IsVaArgs);
  FunctionCallee Callee = M->getOrInsertFunction(FuncName, FuncType);
  inferLibFuncAttributes(M, FuncName, *TLI);
  CallInst *CI = B.CreateCall(Callee, Operands, FuncName);
  if (const Function *F =
          dyn_cast<Function>(Callee.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitStrLen(Value *Ptr, IRBuilderBase &B, const DataLayout &DL,
                        const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(LibFunc_strlen, DL.getIntPtrType(Context),
                     B.getInt8PtrTy(), castToCStr(Ptr, B), B, TLI);
}

Value *llvm::emitStrDup(Value *Ptr, IRBuilderBase &B,
                        const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_strdup, B.getInt8PtrTy(), B.getInt8PtrTy(),
                     castToCStr(Ptr, B), B, TLI);
}

Value *llvm::emitStrChr(Value *Ptr, char C, IRBuilderBase &B,
                        const TargetLibraryInfo *TLI) {
  Type *I8Ptr = B.getInt8PtrTy();
  Type *I32Ty = B.getInt32Ty();
  return emitLibCall(LibFunc_strchr, I8Ptr, {I8Ptr, I32Ty},
                     {castToCStr(Ptr, B), ConstantInt::get(I32Ty, C)}, B, TLI);
}

Value *llvm::emitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilderBase &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(
      LibFunc_strncmp, B.getInt32Ty(),
      {B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context)},
      {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, B, TLI);
}

Value *llvm::emitStrCpy(Value *Dst, Value *Src, IRBuilderBase &B,
                        const TargetLibraryInfo *TLI) {
  Type *I8Ptr = B.getInt8PtrTy();
  return emitLibCall(LibFunc_strcpy, I8Ptr, {I8Ptr, I8Ptr},
                     {castToCStr(Dst, B), castToCStr(Src, B)}, B, TLI);
}

Value *llvm::emitStpCpy(Value *Dst, Value *Src, IRBuilderBase &B,
                        const TargetLibraryInfo *TLI) {
  Type *I8Ptr = B.getInt8PtrTy();
  return emitLibCall(LibFunc_stpcpy, I8Ptr, {I8Ptr, I8Ptr},
                     {castToCStr(Dst, B), castToCStr(Src, B)}, B, TLI);
}

Value *llvm::emitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  Type *I8Ptr = B.getInt8PtrTy();
  return emitLibCall(LibFunc_strncpy, I8Ptr, {I8Ptr, I8Ptr, Len->getType()},
                     {castToCStr(Dst, B), castToCStr(Src, B), Len}, B, TLI);
}

Value *llvm::emitStpNCpy(Value *Dst, Value *Src, Value *Len, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  Type *I8Ptr = B.getInt8PtrTy();
  return emitLibCall(LibFunc_stpncpy, I8Ptr, {I8Ptr, I8Ptr, Len->getType()},
                     {castToCStr(Dst, B), castToCStr(Src, B), Len}, B, TLI);
}

Value *llvm::emitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                           IRBuilderBase &B, const DataLayout &DL,
                           const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_memcpy_chk))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  AttributeList AS;
  AS = AttributeList::get(M->getContext(), AttributeList::FunctionIndex,
                          Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  FunctionCallee MemCpy = M->getOrInsertFunction(
      "__memcpy_chk", AttributeList::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context),
      DL.getIntPtrType(Context));
  Dst = castToCStr(Dst, B);
  Src = castToCStr(Src, B);
  CallInst *CI = B.CreateCall(MemCpy, {Dst, Src, Len, ObjSize});
  if (const Function *F =
          dyn_cast<Function>(MemCpy.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitMemPCpy(Value *Dst, Value *Src, Value *Len, IRBuilderBase &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(
      LibFunc_mempcpy, B.getInt8PtrTy(),
      {B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context)},
      {Dst, Src, Len}, B, TLI);
}

Value *llvm::emitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilderBase &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(
      LibFunc_memchr, B.getInt8PtrTy(),
      {B.getInt8PtrTy(), B.getInt32Ty(), DL.getIntPtrType(Context)},
      {castToCStr(Ptr, B), Val, Len}, B, TLI);
}

Value *llvm::emitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilderBase &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(
      LibFunc_memcmp, B.getInt32Ty(),
      {B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context)},
      {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, B, TLI);
}

Value *llvm::emitBCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilderBase &B,
                      const DataLayout &DL, const TargetLibraryInfo *TLI) {
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  return emitLibCall(
      LibFunc_bcmp, B.getInt32Ty(),
      {B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context)},
      {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, B, TLI);
}

Value *llvm::emitMemCCpy(Value *Ptr1, Value *Ptr2, Value *Val, Value *Len,
                         IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  return emitLibCall(
      LibFunc_memccpy, B.getInt8PtrTy(),
      {B.getInt8PtrTy(), B.getInt8PtrTy(), B.getInt32Ty(), Len->getType()},
      {Ptr1, Ptr2, Val, Len}, B, TLI);
}

Value *llvm::emitSNPrintf(Value *Dest, Value *Size, Value *Fmt,
                          ArrayRef<Value *> VariadicArgs, IRBuilderBase &B,
                          const TargetLibraryInfo *TLI) {
  SmallVector<Value *, 8> Args{castToCStr(Dest, B), Size, castToCStr(Fmt, B)};
  Args.insert(Args.end(), VariadicArgs.begin(), VariadicArgs.end());
  return emitLibCall(LibFunc_snprintf, B.getInt32Ty(),
                     {B.getInt8PtrTy(), Size->getType(), B.getInt8PtrTy()},
                     Args, B, TLI, /*IsVaArgs=*/true);
}

Value *llvm::emitSPrintf(Value *Dest, Value *Fmt,
                         ArrayRef<Value *> VariadicArgs, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  SmallVector<Value *, 8> Args{castToCStr(Dest, B), castToCStr(Fmt, B)};
  Args.insert(Args.end(), VariadicArgs.begin(), VariadicArgs.end());
  return emitLibCall(LibFunc_sprintf, B.getInt32Ty(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy()}, Args, B, TLI,
                     /*IsVaArgs=*/true);
}

Value *llvm::emitStrCat(Value *Dest, Value *Src, IRBuilderBase &B,
                        const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_strcat, B.getInt8PtrTy(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy()},
                     {castToCStr(Dest, B), castToCStr(Src, B)}, B, TLI);
}

Value *llvm::emitStrLCpy(Value *Dest, Value *Src, Value *Size, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_strlcpy, Size->getType(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy(), Size->getType()},
                     {castToCStr(Dest, B), castToCStr(Src, B), Size}, B, TLI);
}

Value *llvm::emitStrLCat(Value *Dest, Value *Src, Value *Size, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_strlcat, Size->getType(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy(), Size->getType()},
                     {castToCStr(Dest, B), castToCStr(Src, B), Size}, B, TLI);
}

Value *llvm::emitStrNCat(Value *Dest, Value *Src, Value *Size, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_strncat, B.getInt8PtrTy(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy(), Size->getType()},
                     {castToCStr(Dest, B), castToCStr(Src, B), Size}, B, TLI);
}

Value *llvm::emitVSNPrintf(Value *Dest, Value *Size, Value *Fmt, Value *VAList,
                           IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  return emitLibCall(
      LibFunc_vsnprintf, B.getInt32Ty(),
      {B.getInt8PtrTy(), Size->getType(), B.getInt8PtrTy(), VAList->getType()},
      {castToCStr(Dest, B), Size, castToCStr(Fmt, B), VAList}, B, TLI);
}

Value *llvm::emitVSPrintf(Value *Dest, Value *Fmt, Value *VAList,
                          IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  return emitLibCall(LibFunc_vsprintf, B.getInt32Ty(),
                     {B.getInt8PtrTy(), B.getInt8PtrTy(), VAList->getType()},
                     {castToCStr(Dest, B), castToCStr(Fmt, B), VAList}, B, TLI);
}

/// Append a suffix to the function name according to the type of 'Op'.
static void appendTypeSuffix(Value *Op, StringRef &Name,
                             SmallString<20> &NameBuffer) {
  if (!Op->getType()->isDoubleTy()) {
      NameBuffer += Name;

    if (Op->getType()->isFloatTy())
      NameBuffer += 'f';
    else
      NameBuffer += 'l';

    Name = NameBuffer;
  }
}

static Value *emitUnaryFloatFnCallHelper(Value *Op, StringRef Name,
                                         IRBuilderBase &B,
                                         const AttributeList &Attrs) {
  assert((Name != "") && "Must specify Name to emitUnaryFloatFnCall");

  Module *M = B.GetInsertBlock()->getModule();
  FunctionCallee Callee =
      M->getOrInsertFunction(Name, Op->getType(), Op->getType());
  CallInst *CI = B.CreateCall(Callee, Op, Name);

  // The incoming attribute set may have come from a speculatable intrinsic, but
  // is being replaced with a library call which is not allowed to be
  // speculatable.
  CI->setAttributes(Attrs.removeAttribute(B.getContext(),
                                          AttributeList::FunctionIndex,
                                          Attribute::Speculatable));
  if (const Function *F =
          dyn_cast<Function>(Callee.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilderBase &B,
                                  const AttributeList &Attrs) {
  SmallString<20> NameBuffer;
  appendTypeSuffix(Op, Name, NameBuffer);

  return emitUnaryFloatFnCallHelper(Op, Name, B, Attrs);
}

Value *llvm::emitUnaryFloatFnCall(Value *Op, const TargetLibraryInfo *TLI,
                                  LibFunc DoubleFn, LibFunc FloatFn,
                                  LibFunc LongDoubleFn, IRBuilderBase &B,
                                  const AttributeList &Attrs) {
  // Get the name of the function according to TLI.
  StringRef Name = getFloatFnName(TLI, Op->getType(),
                                  DoubleFn, FloatFn, LongDoubleFn);

  return emitUnaryFloatFnCallHelper(Op, Name, B, Attrs);
}

static Value *emitBinaryFloatFnCallHelper(Value *Op1, Value *Op2,
                                          StringRef Name, IRBuilderBase &B,
                                          const AttributeList &Attrs) {
  assert((Name != "") && "Must specify Name to emitBinaryFloatFnCall");

  Module *M = B.GetInsertBlock()->getModule();
  FunctionCallee Callee = M->getOrInsertFunction(Name, Op1->getType(),
                                                 Op1->getType(), Op2->getType());
  CallInst *CI = B.CreateCall(Callee, { Op1, Op2 }, Name);

  // The incoming attribute set may have come from a speculatable intrinsic, but
  // is being replaced with a library call which is not allowed to be
  // speculatable.
  CI->setAttributes(Attrs.removeAttribute(B.getContext(),
                                          AttributeList::FunctionIndex,
                                          Attribute::Speculatable));
  if (const Function *F =
          dyn_cast<Function>(Callee.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitBinaryFloatFnCall(Value *Op1, Value *Op2, StringRef Name,
                                   IRBuilderBase &B,
                                   const AttributeList &Attrs) {
  assert((Name != "") && "Must specify Name to emitBinaryFloatFnCall");

  SmallString<20> NameBuffer;
  appendTypeSuffix(Op1, Name, NameBuffer);

  return emitBinaryFloatFnCallHelper(Op1, Op2, Name, B, Attrs);
}

Value *llvm::emitBinaryFloatFnCall(Value *Op1, Value *Op2,
                                   const TargetLibraryInfo *TLI,
                                   LibFunc DoubleFn, LibFunc FloatFn,
                                   LibFunc LongDoubleFn, IRBuilderBase &B,
                                   const AttributeList &Attrs) {
  // Get the name of the function according to TLI.
  StringRef Name = getFloatFnName(TLI, Op1->getType(),
                                  DoubleFn, FloatFn, LongDoubleFn);

  return emitBinaryFloatFnCallHelper(Op1, Op2, Name, B, Attrs);
}

Value *llvm::emitPutChar(Value *Char, IRBuilderBase &B,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_putchar))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef PutCharName = TLI->getName(LibFunc_putchar);
  FunctionCallee PutChar =
      M->getOrInsertFunction(PutCharName, B.getInt32Ty(), B.getInt32Ty());
  inferLibFuncAttributes(M, PutCharName, *TLI);
  CallInst *CI = B.CreateCall(PutChar,
                              B.CreateIntCast(Char,
                              B.getInt32Ty(),
                              /*isSigned*/true,
                              "chari"),
                              PutCharName);

  if (const Function *F =
          dyn_cast<Function>(PutChar.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitPutS(Value *Str, IRBuilderBase &B,
                      const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_puts))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef PutsName = TLI->getName(LibFunc_puts);
  FunctionCallee PutS =
      M->getOrInsertFunction(PutsName, B.getInt32Ty(), B.getInt8PtrTy());
  inferLibFuncAttributes(M, PutsName, *TLI);
  CallInst *CI = B.CreateCall(PutS, castToCStr(Str, B), PutsName);
  if (const Function *F =
          dyn_cast<Function>(PutS.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitFPutC(Value *Char, Value *File, IRBuilderBase &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_fputc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef FPutcName = TLI->getName(LibFunc_fputc);
  FunctionCallee F = M->getOrInsertFunction(FPutcName, B.getInt32Ty(),
                                            B.getInt32Ty(), File->getType());
  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(M, FPutcName, *TLI);
  Char = B.CreateIntCast(Char, B.getInt32Ty(), /*isSigned*/true,
                         "chari");
  CallInst *CI = B.CreateCall(F, {Char, File}, FPutcName);

  if (const Function *Fn =
          dyn_cast<Function>(F.getCallee()->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFPutS(Value *Str, Value *File, IRBuilderBase &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_fputs))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef FPutsName = TLI->getName(LibFunc_fputs);
  FunctionCallee F = M->getOrInsertFunction(FPutsName, B.getInt32Ty(),
                                            B.getInt8PtrTy(), File->getType());
  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(M, FPutsName, *TLI);
  CallInst *CI = B.CreateCall(F, {castToCStr(Str, B), File}, FPutsName);

  if (const Function *Fn =
          dyn_cast<Function>(F.getCallee()->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilderBase &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_fwrite))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  StringRef FWriteName = TLI->getName(LibFunc_fwrite);
  FunctionCallee F = M->getOrInsertFunction(
      FWriteName, DL.getIntPtrType(Context), B.getInt8PtrTy(),
      DL.getIntPtrType(Context), DL.getIntPtrType(Context), File->getType());

  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(M, FWriteName, *TLI);
  CallInst *CI =
      B.CreateCall(F, {castToCStr(Ptr, B), Size,
                       ConstantInt::get(DL.getIntPtrType(Context), 1), File});

  if (const Function *Fn =
          dyn_cast<Function>(F.getCallee()->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitMalloc(Value *Num, IRBuilderBase &B, const DataLayout &DL,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc_malloc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef MallocName = TLI->getName(LibFunc_malloc);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  FunctionCallee Malloc = M->getOrInsertFunction(MallocName, B.getInt8PtrTy(),
                                                 DL.getIntPtrType(Context));
  inferLibFuncAttributes(M, MallocName, *TLI);
  CallInst *CI = B.CreateCall(Malloc, Num, MallocName);

  if (const Function *F =
          dyn_cast<Function>(Malloc.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitCalloc(Value *Num, Value *Size, const AttributeList &Attrs,
                        IRBuilderBase &B, const TargetLibraryInfo &TLI) {
  if (!TLI.has(LibFunc_calloc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef CallocName = TLI.getName(LibFunc_calloc);
  const DataLayout &DL = M->getDataLayout();
  IntegerType *PtrType = DL.getIntPtrType((B.GetInsertBlock()->getContext()));
  FunctionCallee Calloc = M->getOrInsertFunction(
      CallocName, Attrs, B.getInt8PtrTy(), PtrType, PtrType);
  inferLibFuncAttributes(M, CallocName, TLI);
  CallInst *CI = B.CreateCall(Calloc, {Num, Size}, CallocName);

  if (const auto *F =
          dyn_cast<Function>(Calloc.getCallee()->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}
