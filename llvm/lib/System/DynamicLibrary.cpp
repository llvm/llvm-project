//===-- DynamicLibrary.cpp - Runtime link/load libraries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system DynamicLibrary concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/DynamicLibrary.h"
#include "llvm/Config/config.h"
#include <cstdio>
#include <cstring>
#include <map>

// Collection of symbol name/value pairs to be searched prior to any libraries.
std::map<std::string, void *> &g_symbols() {
  static std::map<std::string, void *> symbols;
  return symbols;
}

void llvm::sys::DynamicLibrary::AddSymbol(const char* symbolName,
                                          void *symbolValue) {
  g_symbols()[symbolName] = symbolValue;
}

// It is not possible to use ltdl.c on VC++ builds as the terms of its LGPL
// license and special exception would cause all of LLVM to be placed under
// the LGPL.  This is because the exception applies only when libtool is
// used, and obviously libtool is not used with Visual Studio.  An entirely
// separate implementation is provided in win32/DynamicLibrary.cpp.

#ifdef LLVM_ON_WIN32

#include "Win32/DynamicLibrary.inc"

#else

//#include "ltdl.h"
#include <dlfcn.h>
#include <cassert>
using namespace llvm;
using namespace llvm::sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

//static std::vector<lt_dlhandle> OpenedHandles;
static std::vector<void *> OpenedHandles;

DynamicLibrary::DynamicLibrary() {}

DynamicLibrary::~DynamicLibrary() {
  while(!OpenedHandles.empty()) {
    void *H = OpenedHandles.back();   OpenedHandles.pop_back(); 
    dlclose(H);
  }
}

bool DynamicLibrary::LoadLibraryPermanently(const char *Filename,
                                            std::string *ErrMsg) {
  void *H = dlopen(Filename, RTLD_LAZY|RTLD_GLOBAL);
  if (H == 0) {
    if (ErrMsg)
      *ErrMsg = dlerror();
    return true;
  }
  OpenedHandles.push_back(H);
  return false;
}

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  //  check_ltdl_initialization();

  // First check symbols added via AddSymbol().
  std::map<std::string, void *>::iterator I = g_symbols().find(symbolName);
  if (I != g_symbols().end())
    return I->second;

  // Now search the libraries.
  for (std::vector<void *>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    //lt_ptr ptr = lt_dlsym(*I, symbolName);
    void *ptr = dlsym(*I, symbolName);
    if (ptr)
      return ptr;
  }

#define EXPLICIT_SYMBOL(SYM) \
   extern void *SYM; if (!strcmp(symbolName, #SYM)) return &SYM

  // If this is darwin, it has some funky issues, try to solve them here.  Some
  // important symbols are marked 'private external' which doesn't allow
  // SearchForAddressOfSymbol to find them.  As such, we special case them here,
  // there is only a small handful of them.

#ifdef __APPLE__
  {
    EXPLICIT_SYMBOL(__ashldi3);
    EXPLICIT_SYMBOL(__ashrdi3);
    EXPLICIT_SYMBOL(__cmpdi2);
    EXPLICIT_SYMBOL(__divdi3);
    EXPLICIT_SYMBOL(__eprintf);
    EXPLICIT_SYMBOL(__fixdfdi);
    EXPLICIT_SYMBOL(__fixsfdi);
    EXPLICIT_SYMBOL(__fixunsdfdi);
    EXPLICIT_SYMBOL(__fixunssfdi);
    EXPLICIT_SYMBOL(__floatdidf);
    EXPLICIT_SYMBOL(__floatdisf);
    EXPLICIT_SYMBOL(__lshrdi3);
    EXPLICIT_SYMBOL(__moddi3);
    EXPLICIT_SYMBOL(__udivdi3);
    EXPLICIT_SYMBOL(__umoddi3);
  }
#endif

#ifdef __CYGWIN__
  {
    EXPLICIT_SYMBOL(_alloca);
    EXPLICIT_SYMBOL(__main);
  }
#endif

#undef EXPLICIT_SYMBOL

// This macro returns the address of a well-known, explicit symbol
#define EXPLICIT_SYMBOL(SYM) \
   if (!strcmp(symbolName, #SYM)) return &SYM

// On linux we have a weird situation. The stderr/out/in symbols are both
// macros and global variables because of standards requirements. So, we 
// boldly use the EXPLICIT_SYMBOL macro without checking for a #define first.
#if defined(__linux__)
  {
    EXPLICIT_SYMBOL(stderr);
    EXPLICIT_SYMBOL(stdout);
    EXPLICIT_SYMBOL(stdin);
  }
#else
  // For everything else, we want to check to make sure the symbol isn't defined
  // as a macro before using EXPLICIT_SYMBOL.
  {
#ifndef stdin
    EXPLICIT_SYMBOL(stdin);
#endif
#ifndef stdout
    EXPLICIT_SYMBOL(stdout);
#endif
#ifndef stderr
    EXPLICIT_SYMBOL(stderr);
#endif
  }
#endif
#undef EXPLICIT_SYMBOL

  return 0;
}

#endif // LLVM_ON_WIN32
