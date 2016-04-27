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

/// Analyze the name and prototype of the given function and set any applicable
/// attributes.
///
/// Returns true if any attributes were set and false otherwise.
static bool inferPrototypeAttributes(Function &F,
                                     const TargetLibraryInfo &TLI) {
  if (F.hasFnAttribute(Attribute::OptimizeNone))
    return false;

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
  case LibFunc::memccpy:
  case LibFunc::memmove:
    Changed |= setDoesNotThrow(F);
    Changed |= setDoesNotCapture(F, 2);
    Changed |= setOnlyReadsMemory(F, 2);
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
    Changed |= setOnlyReadsMemory(F, 2);
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
