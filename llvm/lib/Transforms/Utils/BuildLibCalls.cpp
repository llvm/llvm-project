//===- BuildLibCalls.cpp - Utility builder for libcalls -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static bool setOnlyAccessesArgMemory(Function &F) {
  if (F.onlyAccessesArgMemory())
    return false;
  F.setOnlyAccessesArgMemory ();
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

bool llvm::inferLibFuncAttributes(Function &F, const TargetLibraryInfo &TLI) {
  LibFunc::Func TheLibFunc;
  if (!(TLI.getLibFunc(F, TheLibFunc) && TLI.has(TheLibFunc)))
    return false;

  bool Changed = false;
  switch (TheLibFunc) {
  case LibFunc::strlen:
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::strchr:
  case LibFunc::strrchr:
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
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::strxfrm:
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
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::strstr:
  case LibFunc::strpbrk:
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::strtok:
  case LibFunc::strtok_r:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::scanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::setbuf:
  case LibFunc::setvbuf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::strdup:
  case LibFunc::strndup:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::stat:
  case LibFunc::statvfs:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::sscanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::sprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::snprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 3);
    return Changed;
  case LibFunc::setitimer:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::system:
    // May throw; "system" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::malloc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::memcmp:
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::memchr:
  case LibFunc::memrchr:
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::modf:
  case LibFunc::modff:
  case LibFunc::modfl:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::memcpy:
  case LibFunc::mempcpy:
  case LibFunc::memccpy:
  case LibFunc::memmove:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::memcpy_chk:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::memalign:
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::mkdir:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::mktime:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::realloc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::read:
    // May throw; "read" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::rewind:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::rmdir:
  case LibFunc::remove:
  case LibFunc::realpath:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::rename:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::readlink:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::write:
    // May throw; "write" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::bcopy:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::bcmp:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::bzero:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::calloc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::chmod:
  case LibFunc::chown:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::ctermid:
  case LibFunc::clearerr:
  case LibFunc::closedir:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::atoi:
  case LibFunc::atol:
  case LibFunc::atof:
  case LibFunc::atoll:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::access:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::fopen:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fdopen:
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
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::ferror:
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
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::fgets:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 3);
    return Changed;
  case LibFunc::fread:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 4);
    return Changed;
  case LibFunc::fwrite:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 4);
    // FIXME: readonly #1?
    return Changed;
  case LibFunc::fputs:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::fscanf:
  case LibFunc::fprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fgetpos:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::getc:
  case LibFunc::getlogin_r:
  case LibFunc::getc_unlocked:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::getenv:
    Changed |= setDoesNotThrow(F);
    Changed |= setOnlyReadsMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::gets:
  case LibFunc::getchar:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::getitimer:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::getpwnam:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::ungetc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::uname:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::unlink:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::unsetenv:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::utime:
  case LibFunc::utimes:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::putc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::puts:
  case LibFunc::printf:
  case LibFunc::perror:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::pread:
    // May throw; "pread" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::pwrite:
    // May throw; "pwrite" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::putchar:
    Changed |= setDoesNotThrow(F);
    return Changed;
  case LibFunc::popen:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::pclose:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::vscanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::vsscanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::vfscanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::valloc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::vprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::vfprintf:
  case LibFunc::vsprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::vsnprintf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 3);
    Changed |= setOnlyReadsMemory(F, 3);
    return Changed;
  case LibFunc::open:
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::opendir:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::tmpfile:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::times:
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
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::lchown:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::qsort:
    // May throw; places call through function pointer.
    Changed |= setDoesNotCapture(F, 4);
    return Changed;
  case LibFunc::dunder_strdup:
  case LibFunc::dunder_strndup:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::dunder_strtok_r:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::under_IO_getc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::under_IO_putc:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::dunder_isoc99_scanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::stat64:
  case LibFunc::lstat64:
  case LibFunc::statvfs64:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::dunder_isoc99_sscanf:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fopen64:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 1);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  case LibFunc::fseeko64:
  case LibFunc::ftello64:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 1);
    return Changed;
  case LibFunc::tmpfile64:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotAlias(F, 0);
    return Changed;
  case LibFunc::fstat64:
  case LibFunc::fstatvfs64:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    return Changed;
  case LibFunc::open64:
    // May throw; "open" is a valid pthread cancellation point.
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setOnlyReadsMemory(F, 1);
    return Changed;
  case LibFunc::gettimeofday:
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
    // Operator new always returns a nonnull noalias pointer
    Changed |= setNonNull(F, AttributeSet::ReturnIndex);
    Changed |= setDoesNotAlias(F, AttributeSet::ReturnIndex);
    return Changed;
  //TODO: add LibFunc entries for:
  //case LibFunc::memset_pattern4:
  //case LibFunc::memset_pattern8:
  case LibFunc::memset_pattern16:
    Changed |= setOnlyAccessesArgMemory(F);
    Changed |= setDoesNotCapture(F, 1);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
    return Changed;
  // int __nvvm_reflect(const char *)
  case LibFunc::nvvm_reflect:
    Changed |= setDoesNotAccessMemory(F);
    Changed |= setDoesNotThrow(F);
    return Changed;

  default:
    // FIXME: It'd be really nice to cover all the library functions we're
    // aware of here.
    return false;
  }
}

//- Emit LibCalls ------------------------------------------------------------//

Value *llvm::castToCStr(Value *V, IRBuilder<> &B) {
  unsigned AS = V->getType()->getPointerAddressSpace();
  return B.CreateBitCast(V, B.getInt8PtrTy(AS), "cstr");
}

Value *llvm::emitStrLen(Value *Ptr, IRBuilder<> &B, const DataLayout &DL,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strlen))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrLen = M->getOrInsertFunction("strlen", DL.getIntPtrType(Context),
                                            B.getInt8PtrTy(), nullptr);
  inferLibFuncAttributes(*M->getFunction("strlen"), *TLI);
  CallInst *CI = B.CreateCall(StrLen, castToCStr(Ptr, B), "strlen");
  if (const Function *F = dyn_cast<Function>(StrLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitStrChr(Value *Ptr, char C, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Type *I8Ptr = B.getInt8PtrTy();
  Type *I32Ty = B.getInt32Ty();
  Constant *StrChr =
      M->getOrInsertFunction("strchr", I8Ptr, I8Ptr, I32Ty, nullptr);
  inferLibFuncAttributes(*M->getFunction("strchr"), *TLI);
  CallInst *CI = B.CreateCall(
      StrChr, {castToCStr(Ptr, B), ConstantInt::get(I32Ty, C)}, "strchr");
  if (const Function *F = dyn_cast<Function>(StrChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strncmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *StrNCmp = M->getOrInsertFunction("strncmp", B.getInt32Ty(),
                                          B.getInt8PtrTy(), B.getInt8PtrTy(),
                                          DL.getIntPtrType(Context), nullptr);
  inferLibFuncAttributes(*M->getFunction("strncmp"), *TLI);
  CallInst *CI = B.CreateCall(
      StrNCmp, {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, "strncmp");

  if (const Function *F = dyn_cast<Function>(StrNCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strcpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrCpy = M->getOrInsertFunction(Name, I8Ptr, I8Ptr, I8Ptr, nullptr);
  inferLibFuncAttributes(*M->getFunction(Name), *TLI);
  CallInst *CI =
      B.CreateCall(StrCpy, {castToCStr(Dst, B), castToCStr(Src, B)}, Name);
  if (const Function *F = dyn_cast<Function>(StrCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strncpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrNCpy = M->getOrInsertFunction(Name, I8Ptr, I8Ptr, I8Ptr,
                                          Len->getType(), nullptr);
  inferLibFuncAttributes(*M->getFunction(Name), *TLI);
  CallInst *CI = B.CreateCall(
      StrNCpy, {castToCStr(Dst, B), castToCStr(Src, B), Len}, "strncpy");
  if (const Function *F = dyn_cast<Function>(StrNCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                           IRBuilder<> &B, const DataLayout &DL,
                           const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcpy_chk))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  AttributeSet AS;
  AS = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                         Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCpy = M->getOrInsertFunction(
      "__memcpy_chk", AttributeSet::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context),
      DL.getIntPtrType(Context), nullptr);
  Dst = castToCStr(Dst, B);
  Src = castToCStr(Src, B);
  CallInst *CI = B.CreateCall(MemCpy, {Dst, Src, Len, ObjSize});
  if (const Function *F = dyn_cast<Function>(MemCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemChr = M->getOrInsertFunction("memchr", B.getInt8PtrTy(),
                                         B.getInt8PtrTy(), B.getInt32Ty(),
                                         DL.getIntPtrType(Context), nullptr);
  inferLibFuncAttributes(*M->getFunction("memchr"), *TLI);
  CallInst *CI = B.CreateCall(MemChr, {castToCStr(Ptr, B), Val, Len}, "memchr");

  if (const Function *F = dyn_cast<Function>(MemChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCmp = M->getOrInsertFunction("memcmp", B.getInt32Ty(),
                                         B.getInt8PtrTy(), B.getInt8PtrTy(),
                                         DL.getIntPtrType(Context), nullptr);
  inferLibFuncAttributes(*M->getFunction("memcmp"), *TLI);
  CallInst *CI = B.CreateCall(
      MemCmp, {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, "memcmp");

  if (const Function *F = dyn_cast<Function>(MemCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
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

Value *llvm::emitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                                  const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  appendTypeSuffix(Op, Name, NameBuffer);

  Module *M = B.GetInsertBlock()->getModule();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                         Op->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, Op, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitBinaryFloatFnCall(Value *Op1, Value *Op2, StringRef Name,
                                  IRBuilder<> &B, const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  appendTypeSuffix(Op1, Name, NameBuffer);

  Module *M = B.GetInsertBlock()->getModule();
  Value *Callee = M->getOrInsertFunction(Name, Op1->getType(), Op1->getType(),
                                         Op2->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, {Op1, Op2}, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitPutChar(Value *Char, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::putchar))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Value *PutChar = M->getOrInsertFunction("putchar", B.getInt32Ty(),
                                          B.getInt32Ty(), nullptr);
  CallInst *CI = B.CreateCall(PutChar,
                              B.CreateIntCast(Char,
                              B.getInt32Ty(),
                              /*isSigned*/true,
                              "chari"),
                              "putchar");

  if (const Function *F = dyn_cast<Function>(PutChar->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitPutS(Value *Str, IRBuilder<> &B,
                      const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::puts))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Value *PutS =
      M->getOrInsertFunction("puts", B.getInt32Ty(), B.getInt8PtrTy(), nullptr);
  inferLibFuncAttributes(*M->getFunction("puts"), *TLI);
  CallInst *CI = B.CreateCall(PutS, castToCStr(Str, B), "puts");
  if (const Function *F = dyn_cast<Function>(PutS->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  Constant *F = M->getOrInsertFunction("fputc", B.getInt32Ty(), B.getInt32Ty(),
                                       File->getType(), nullptr);
  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(*M->getFunction("fputc"), *TLI);
  Char = B.CreateIntCast(Char, B.getInt32Ty(), /*isSigned*/true,
                         "chari");
  CallInst *CI = B.CreateCall(F, {Char, File}, "fputc");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFPutS(Value *Str, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputs))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  StringRef FPutsName = TLI->getName(LibFunc::fputs);
  Constant *F = M->getOrInsertFunction(
      FPutsName, B.getInt32Ty(), B.getInt8PtrTy(), File->getType(), nullptr);
  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(*M->getFunction(FPutsName), *TLI);
  CallInst *CI = B.CreateCall(F, {castToCStr(Str, B), File}, "fputs");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fwrite))
    return nullptr;

  Module *M = B.GetInsertBlock()->getModule();
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  StringRef FWriteName = TLI->getName(LibFunc::fwrite);
  Constant *F = M->getOrInsertFunction(
      FWriteName, DL.getIntPtrType(Context), B.getInt8PtrTy(),
      DL.getIntPtrType(Context), DL.getIntPtrType(Context), File->getType(),
      nullptr);
  if (File->getType()->isPointerTy())
    inferLibFuncAttributes(*M->getFunction(FWriteName), *TLI);
  CallInst *CI =
      B.CreateCall(F, {castToCStr(Ptr, B), Size,
                       ConstantInt::get(DL.getIntPtrType(Context), 1), File});

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}
