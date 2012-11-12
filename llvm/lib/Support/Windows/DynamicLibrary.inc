//===- Win32/DynamicLibrary.cpp - Win32 DL Implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of DynamicLibrary.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif

#ifdef _MSC_VER
 #include <ntverp.h>
#endif

#ifdef __MINGW32__
 #if (HAVE_LIBIMAGEHLP != 1)
  #error "libimagehlp.a should be present"
 #endif
#else
 #pragma comment(lib, "dbghelp.lib")
#endif

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code.
//===----------------------------------------------------------------------===//

static DenseSet<HMODULE> *OpenedHandles;

extern "C" {

  static BOOL CALLBACK ELM_Callback(WIN32_ELMCB_PCSTR ModuleName,
                                    ULONG_PTR ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
  {
    // Ignore VC++ runtimes prior to 7.1.  Somehow some of them get loaded
    // into the process.
    if (stricmp(ModuleName, "msvci70") != 0 &&
        stricmp(ModuleName, "msvcirt") != 0 &&
        stricmp(ModuleName, "msvcp50") != 0 &&
        stricmp(ModuleName, "msvcp60") != 0 &&
        stricmp(ModuleName, "msvcp70") != 0 &&
        stricmp(ModuleName, "msvcr70") != 0 &&
#ifndef __MINGW32__
        // Mingw32 uses msvcrt.dll by default. Don't ignore it.
        // Otherwise, user should be aware, what he's doing :)
        stricmp(ModuleName, "msvcrt") != 0 &&
#endif
        stricmp(ModuleName, "msvcrt20") != 0 &&
        stricmp(ModuleName, "msvcrt40") != 0) {
      OpenedHandles->insert((HMODULE)ModuleBase);
    }
    return TRUE;
  }
}

DynamicLibrary DynamicLibrary::getPermanentLibrary(const char *filename,
                                                   std::string *errMsg) {
  SmartScopedLock<true> lock(getMutex());

  if (!filename) {
    // When no file is specified, enumerate all DLLs and EXEs in the process.
    if (OpenedHandles == 0)
      OpenedHandles = new DenseSet<HMODULE>();

    EnumerateLoadedModules(GetCurrentProcess(), ELM_Callback, 0);
    // Dummy library that represents "search all handles".
    // This is mostly to ensure that the return value still shows up as "valid".
    return DynamicLibrary(&OpenedHandles);
  }
  
  HMODULE a_handle = LoadLibrary(filename);

  if (a_handle == 0) {
    MakeErrMsg(errMsg, std::string(filename) + ": Can't open : ");
    return DynamicLibrary();
  }

  if (OpenedHandles == 0)
    OpenedHandles = new DenseSet<HMODULE>();

  // If we've already loaded this library, FreeLibrary() the handle in order to
  // keep the internal refcount at +1.
  if (!OpenedHandles->insert(a_handle).second)
    FreeLibrary(a_handle);

  return DynamicLibrary(a_handle);
}

// Stack probing routines are in the support library (e.g. libgcc), but we don't
// have dynamic linking on windows. Provide a hook.
#define EXPLICIT_SYMBOL(SYM)                    \
  extern "C" { extern void *SYM; }
#define EXPLICIT_SYMBOL2(SYMFROM, SYMTO) EXPLICIT_SYMBOL(SYMTO)

#include "explicit_symbols.inc"

#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  SmartScopedLock<true> Lock(getMutex());

  // First check symbols added via AddSymbol().
  if (ExplicitSymbols) {
    StringMap<void *>::iterator i = ExplicitSymbols->find(symbolName);

    if (i != ExplicitSymbols->end())
      return i->second;
  }

  // Now search the libraries.
  if (OpenedHandles) {
    for (DenseSet<HMODULE>::iterator I = OpenedHandles->begin(),
         E = OpenedHandles->end(); I != E; ++I) {
      FARPROC ptr = GetProcAddress((HMODULE)*I, symbolName);
      if (ptr) {
        return (void *)(intptr_t)ptr;
      }
    }
  }

  #define EXPLICIT_SYMBOL(SYM)                    \
    if (!strcmp(symbolName, #SYM)) return (void*)&SYM;
  #define EXPLICIT_SYMBOL2(SYMFROM, SYMTO)        \
    if (!strcmp(symbolName, #SYMFROM)) return (void*)&SYMTO;

  {
    #include "explicit_symbols.inc"
  }

  #undef EXPLICIT_SYMBOL
  #undef EXPLICIT_SYMBOL2

  return 0;
}


void *DynamicLibrary::getAddressOfSymbol(const char *symbolName) {
  if (!isValid())
    return NULL;
  if (Data == &OpenedHandles)
    return SearchForAddressOfSymbol(symbolName);
  return (void *)(intptr_t)GetProcAddress((HMODULE)Data, symbolName);
}


}
