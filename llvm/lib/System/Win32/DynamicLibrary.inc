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

#include "Win32.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
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

static std::vector<HMODULE> OpenedHandles;

#ifdef _WIN64
  typedef DWORD64 ModuleBaseType;
#else
  typedef ULONG ModuleBaseType;
#endif

extern "C" {
#if !defined(_MSC_VER) || _MSC_VER < 1500
  static BOOL CALLBACK ELM_Callback(PSTR  ModuleName,
                                    ModuleBaseType ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
#else
  static BOOL CALLBACK ELM_Callback(PCSTR  ModuleName,
                                    ModuleBaseType ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
#endif
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
      OpenedHandles.push_back((HMODULE)ModuleBase);
    }
    return TRUE;
  }
}

DynamicLibrary::DynamicLibrary() : handle(0) {
  handle = GetModuleHandle(NULL);
  OpenedHandles.push_back((HMODULE)handle);
}

DynamicLibrary::~DynamicLibrary() {
  if (handle == 0)
    return;

  // GetModuleHandle() does not increment the ref count, so we must not free
  // the handle to the executable.
  if (handle != GetModuleHandle(NULL))
    FreeLibrary((HMODULE)handle);
  handle = 0;

  for (std::vector<HMODULE>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    if (*I == handle) {
      // Note: don't use the swap/pop_back trick here. Order is important.
      OpenedHandles.erase(I);
    }
  }
}
 
bool DynamicLibrary::LoadLibraryPermanently(const char *filename,
                                            std::string *ErrMsg) {
  if (filename) {
    HMODULE a_handle = LoadLibrary(filename);

    if (a_handle == 0)
      return MakeErrMsg(ErrMsg, std::string(filename) + ": Can't open : ");

    OpenedHandles.push_back(a_handle);
  } else {
    // When no file is specified, enumerate all DLLs and EXEs in the
    // process.
    EnumerateLoadedModules(GetCurrentProcess(), ELM_Callback, 0);
  }

  // Because we don't remember the handle, we will never free it; hence,
  // it is loaded permanently.
  return false;
}

// Stack probing routines are in the support library (e.g. libgcc), but we don't
// have dynamic linking on windows. Provide a hook.
#if defined(__MINGW32__) || defined (_MSC_VER)
  #define EXPLICIT_SYMBOL(SYM)                    \
    if (!strcmp(symbolName, #SYM)) return (void*)&SYM
  #define EXPLICIT_SYMBOL2(SYMFROM, SYMTO)        \
    if (!strcmp(symbolName, #SYMFROM)) return (void*)&SYMTO
  #define EXPLICIT_SYMBOL_DEF(SYM)                \
    extern "C" { extern void *SYM; }

  #if defined(__MINGW32__)
    EXPLICIT_SYMBOL_DEF(_alloca);
    EXPLICIT_SYMBOL_DEF(__main);
    EXPLICIT_SYMBOL_DEF(__ashldi3);
    EXPLICIT_SYMBOL_DEF(__ashrdi3);
    EXPLICIT_SYMBOL_DEF(__cmpdi2);
    EXPLICIT_SYMBOL_DEF(__divdi3);
    EXPLICIT_SYMBOL_DEF(__eprintf);
    EXPLICIT_SYMBOL_DEF(__fixdfdi);
    EXPLICIT_SYMBOL_DEF(__fixsfdi);
    EXPLICIT_SYMBOL_DEF(__fixunsdfdi);
    EXPLICIT_SYMBOL_DEF(__fixunssfdi);
    EXPLICIT_SYMBOL_DEF(__floatdidf);
    EXPLICIT_SYMBOL_DEF(__floatdisf);
    EXPLICIT_SYMBOL_DEF(__lshrdi3);
    EXPLICIT_SYMBOL_DEF(__moddi3);
    EXPLICIT_SYMBOL_DEF(__udivdi3);
    EXPLICIT_SYMBOL_DEF(__umoddi3);
  #elif defined(_MSC_VER)
    EXPLICIT_SYMBOL_DEF(_alloca_probe);
  #endif
#endif

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  // First check symbols added via AddSymbol().
  std::map<std::string, void *>::iterator I = g_symbols().find(symbolName);
  if (I != g_symbols().end())
    return I->second;

  // Now search the libraries.
  for (std::vector<HMODULE>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    FARPROC ptr = GetProcAddress((HMODULE)*I, symbolName);
    if (ptr)
      return (void *) ptr;
  }

#if defined(__MINGW32__)
  {
    EXPLICIT_SYMBOL(_alloca);
    EXPLICIT_SYMBOL(__main);
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

    EXPLICIT_SYMBOL2(alloca, _alloca);
#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2
#undef EXPLICIT_SYMBOL_DEF    
  }
#elif defined(_MSC_VER)
  {
    EXPLICIT_SYMBOL2(alloca, _alloca_probe);
    EXPLICIT_SYMBOL2(_alloca, _alloca_probe);
#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2
#undef EXPLICIT_SYMBOL_DEF    
  }  
#endif

  return 0;
}

}

