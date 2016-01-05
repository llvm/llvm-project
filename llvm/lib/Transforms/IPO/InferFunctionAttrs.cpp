//===- InferFunctionAttrs.cpp - Infer implicit function attributes --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "inferattrs"

STATISTIC(NumReadNone, "Number of functions inferred as readnone");
STATISTIC(NumReadOnly, "Number of functions inferred as readonly");
STATISTIC(NumNoUnwind, "Number of functions inferred as nounwind");
STATISTIC(NumNoCapture, "Number of arguments inferred as nocapture");
STATISTIC(NumReadOnlyArg, "Number of arguments inferred as readonly");
STATISTIC(NumNoAlias, "Number of function returns inferred as noalias");
STATISTIC(NumNonNull, "Number of function returns inferred as nonnull returns");

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

static bool setDoesNotThrow(Function &F) {
  if (F.doesNotThrow())
    return false;
  F.setDoesNotThrow();
  ++NumNoUnwind;
  return true;
}

static bool setDoesNotCapture(Function &F, unsigned n) {
  if (F.doesNotCapture(n))
    return false;
  F.setDoesNotCapture(n);
  ++NumNoCapture;
  return true;
}

static bool setOnlyReadsMemory(Function &F, unsigned n) {
  if (F.onlyReadsMemory(n))
    return false;
  F.setOnlyReadsMemory(n);
  ++NumReadOnlyArg;
  return true;
}

static bool setDoesNotAlias(Function &F, unsigned n) {
  if (F.doesNotAlias(n))
    return false;
  F.setDoesNotAlias(n);
  ++NumNoAlias;
  return true;
}

static bool setNonNull(Function &F, unsigned n) {
  assert((n != AttributeSet::ReturnIndex ||
          F.getReturnType()->isPointerTy()) &&
         "nonnull applies only to pointers");
  if (F.getAttributes().hasAttribute(n, Attribute::NonNull))
    return false;
  F.addAttribute(n, Attribute::NonNull);
  ++NumNonNull;
  return true;
}

/// Analyze the name and prototype of the given function and set any applicable
/// attributes.
///
/// Returns true if any attributes were set and false otherwise.
static bool inferPrototypeAttributes(Function &F,
                                     const TargetLibraryInfo &TLI) {
  if (F.hasFnAttribute(Attribute::OptimizeNone))
    return false;

  FunctionType *FTy = F.getFunctionType();
  LibFunc::Func TheLibFunc;
  if (!(TLI.getLibFunc(F.getName(), TheLibFunc) && TLI.has(TheLibFunc)))
    return false;

  bool Changed = false;
  switch (TheLibFunc) {
  case LibFunc::strlen:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::strchr:
  case LibFunc::strrchr:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isIntegerTy())
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::strtol:
  case LibFunc::strtod:
  case LibFunc::strtof:
  case LibFunc::strtoul:
  case LibFunc::strtoll:
  case LibFunc::strtold:
  case LibFunc::strtoull:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::strcpy:
  case LibFunc::stpcpy:
  case LibFunc::strcat:
  case LibFunc::strncat:
  case LibFunc::strncpy:
  case LibFunc::stpncpy:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::strxfrm:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::strcmp:      // 0,1
  case LibFunc::strspn:      // 0,1
  case LibFunc::strncmp:     // 0,1
  case LibFunc::strcspn:     // 0,1
  case LibFunc::strcoll:     // 0,1
  case LibFunc::strcasecmp:  // 0,1
  case LibFunc::strncasecmp: //
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::strstr:
  case LibFunc::strpbrk:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::strtok:
  case LibFunc::strtok_r:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::scanf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::setbuf:
  case LibFunc::setvbuf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::strdup:
  case LibFunc::strndup:
    if (FTy->getNumParams() < 1 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::stat:
  case LibFunc::statvfs:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::sscanf:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::sprintf:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::snprintf:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 3);
    return Changed;
  case LibFunc::setitimer:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::system:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "system" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::malloc:
    if (FTy->getNumParams() != 1 || !FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::memcmp:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::memchr:
  case LibFunc::memrchr:
    if (FTy->getNumParams() != 3)
      return false;
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::modf:
  case LibFunc::modff:
  case LibFunc::modfl:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::memcpy:
  case LibFunc::memccpy:
  case LibFunc::memmove:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::memalign:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::mkdir:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::mktime:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::realloc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::read:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "read" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::rewind:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::rmdir:
  case LibFunc::remove:
  case LibFunc::realpath:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::rename:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::readlink:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::write:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "write" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::bcopy:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::bcmp:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::bzero:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::calloc:
    if (FTy->getNumParams() != 2 || !FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::chmod:
  case LibFunc::chown:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::ctermid:
  case LibFunc::clearerr:
  case LibFunc::closedir:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::atoi:
  case LibFunc::atol:
  case LibFunc::atof:
  case LibFunc::atoll:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::access:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::fopen:
    if (FTy->getNumParams() != 2 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fdopen:
    if (FTy->getNumParams() != 2 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::feof:
  case LibFunc::free:
  case LibFunc::fseek:
  case LibFunc::ftell:
  case LibFunc::fgetc:
  case LibFunc::fseeko:
  case LibFunc::ftello:
  case LibFunc::fileno:
  case LibFunc::fflush:
  case LibFunc::fclose:
  case LibFunc::fsetpos:
  case LibFunc::flockfile:
  case LibFunc::funlockfile:
  case LibFunc::ftrylockfile:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::ferror:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F);
    return Changed;
  case LibFunc::fputc:
  case LibFunc::fstat:
  case LibFunc::frexp:
  case LibFunc::frexpf:
  case LibFunc::frexpl:
  case LibFunc::fstatvfs:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::fgets:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 3);
    return Changed;
  case LibFunc::fread:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(3)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 4);
    return Changed;
  case LibFunc::fwrite:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(3)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 4);
    return Changed;
  case LibFunc::fputs:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::fscanf:
  case LibFunc::fprintf:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fgetpos:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::getc:
  case LibFunc::getlogin_r:
  case LibFunc::getc_unlocked:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::getenv:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::gets:
  case LibFunc::getchar:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::getitimer:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::getpwnam:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::ungetc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::uname:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::unlink:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::unsetenv:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::utime:
  case LibFunc::utimes:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::putc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::puts:
  case LibFunc::printf:
  case LibFunc::perror:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::pread:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "pread" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::pwrite:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(1)->isPointerTy())
      return false;
    // May throw; "pwrite" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::putchar:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::popen:
    if (FTy->getNumParams() != 2 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::pclose:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::vscanf:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::vsscanf:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::vfscanf:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::valloc:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::vprintf:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::vfprintf:
  case LibFunc::vsprintf:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::vsnprintf:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(2)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 3);
    return Changed;
  case LibFunc::open:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::opendir:
    if (FTy->getNumParams() != 1 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::tmpfile:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::times:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::htonl:
  case LibFunc::htons:
  case LibFunc::ntohl:
  case LibFunc::ntohs:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAccessMemory(F);
    return Changed;
  case LibFunc::lstat:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::lchown:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::qsort:
    if (FTy->getNumParams() != 4 || !FTy->getParamType(3)->isPointerTy())
      return false;
    // May throw; places call through function pointer.
    Changed |= setDoesNotCapture(F, 4);
    return Changed;
  case LibFunc::dunder_strdup:
  case LibFunc::dunder_strndup:
    if (FTy->getNumParams() < 1 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::dunder_strtok_r:
    if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::under_IO_getc:
    if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::under_IO_putc:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::dunder_isoc99_scanf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::stat64:
  case LibFunc::lstat64:
  case LibFunc::statvfs64:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::dunder_isoc99_sscanf:
    if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fopen64:
    if (FTy->getNumParams() != 2 || !FTy->getReturnType()->isPointerTy() ||
        !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fseeko64:
  case LibFunc::ftello64:
    if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::tmpfile64:
    if (!FTy->getReturnType()->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::fstat64:
  case LibFunc::fstatvfs64:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
      return false;
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::open64:
    if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
      return false;
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::gettimeofday:
    if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy() ||
        !FTy->getParamType(1)->isPointerTy())
      return false;
    // Currently some platforms have the restrict keyword on the arguments to
    // gettimeofday. To be conservative, do not add noalias to gettimeofday's
    // arguments.
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;

  case LibFunc::Znwj: // new(unsigned int)
  case LibFunc::Znwm: // new(unsigned long)
  case LibFunc::Znaj: // new[](unsigned int)
  case LibFunc::Znam: // new[](unsigned long)
  case LibFunc::msvc_new_int: // new(unsigned int)
  case LibFunc::msvc_new_longlong: // new(unsigned long long)
  case LibFunc::msvc_new_array_int: // new[](unsigned int)
  case LibFunc::msvc_new_array_longlong: // new[](unsigned long long)
    if (FTy->getNumParams() != 1)
      return false;
    // Operator new always returns a nonnull noalias pointer
    Changed |= setNonNull(F, AttributeSet::ReturnIndex);
    Changed |= setDoesNotAlias(F, AttributeSet::ReturnIndex);
    return Changed;

  default:
    // FIXME: It'd be really nice to cover all the library functions we're
    // aware of here.
    return false;
  }
}

static bool inferAllPrototypeAttributes(Module &M,
                                        const TargetLibraryInfo &TLI) {
  bool Changed = false;

  for (Function &F : M.functions())
    // We only infer things using the prototype if the definition isn't around
    // to analyze directly.
    if (F.isDeclaration())
      Changed |= inferPrototypeAttributes(F, TLI);

  return Changed;
}

PreservedAnalyses InferFunctionAttrsPass::run(Module &M,
                                              AnalysisManager<Module> *AM) {
  auto &TLI = AM->getResult<TargetLibraryAnalysis>(M);

  if (!inferAllPrototypeAttributes(M, TLI))
    // If we didn't infer anything, preserve all analyses.
    return PreservedAnalyses::all();

  // Otherwise, we may have changed fundamental function attributes, so clear
  // out all the passes.
  return PreservedAnalyses::none();
}

namespace {
struct InferFunctionAttrsLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  InferFunctionAttrsLegacyPass() : ModulePass(ID) {
    initializeInferFunctionAttrsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    return inferAllPrototypeAttributes(M, TLI);
  }
};
}

char InferFunctionAttrsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(InferFunctionAttrsLegacyPass, "inferattrs",
                      "Infer set function attributes", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(InferFunctionAttrsLegacyPass, "inferattrs",
                    "Infer set function attributes", false, false)

Pass *llvm::createInferFunctionAttrsLegacyPass() {
  return new InferFunctionAttrsLegacyPass();
}
