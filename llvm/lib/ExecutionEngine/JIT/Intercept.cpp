//===-- Intercept.cpp - System function interception routines -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// If a function call occurs to an external function, the JIT is designed to use
// the dynamic loader interface to find a function to call.  This is useful for
// calling system calls and library functions that are not available in LLVM.
// Some system calls, however, need to be handled specially.  For this reason,
// we intercept some of them here and use our own stubs to handle them.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Config/config.h"
using namespace llvm;

// AtExitHandlers - List of functions to call when the program exits,
// registered with the atexit() library function.
static std::vector<void (*)()> AtExitHandlers;

/// runAtExitHandlers - Run any functions registered by the program's
/// calls to atexit(3), which we intercept and store in
/// AtExitHandlers.
///
static void runAtExitHandlers() {
  while (!AtExitHandlers.empty()) {
    void (*Fn)() = AtExitHandlers.back();
    AtExitHandlers.pop_back();
    Fn();
  }
}

//===----------------------------------------------------------------------===//
// Function stubs that are invoked instead of certain library calls
//===----------------------------------------------------------------------===//

// Force the following functions to be linked in to anything that uses the
// JIT. This is a hack designed to work around the all-too-clever Glibc
// strategy of making these functions work differently when inlined vs. when
// not inlined, and hiding their real definitions in a separate archive file
// that the dynamic linker can't see. For more info, search for
// 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
#if defined(__linux__)
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif

/* stat functions are redirecting to __xstat with a version number.  On x86-64 
 * linking with libc_nonshared.a and -Wl,--export-dynamic doesn't make 'stat' 
 * available as an exported symbol, so we have to add it explicitly.
 */
class StatSymbols {
public:
  StatSymbols() {
    sys::DynamicLibrary::AddSymbol("stat", (void*)(intptr_t)stat);
    sys::DynamicLibrary::AddSymbol("fstat", (void*)(intptr_t)fstat);
    sys::DynamicLibrary::AddSymbol("lstat", (void*)(intptr_t)lstat);
    sys::DynamicLibrary::AddSymbol("stat64", (void*)(intptr_t)stat64);
    sys::DynamicLibrary::AddSymbol("fstat64", (void*)(intptr_t)fstat64);
    sys::DynamicLibrary::AddSymbol("lstat64", (void*)(intptr_t)lstat64);
    sys::DynamicLibrary::AddSymbol("atexit", (void*)(intptr_t)atexit);
    sys::DynamicLibrary::AddSymbol("mknod", (void*)(intptr_t)mknod);
  }
};
static StatSymbols initStatSymbols;
#endif // __linux__

// jit_exit - Used to intercept the "exit" library call.
static void jit_exit(int Status) {
  runAtExitHandlers();   // Run atexit handlers...
  exit(Status);
}

// jit_atexit - Used to intercept the "atexit" library call.
static int jit_atexit(void (*Fn)(void)) {
  AtExitHandlers.push_back(Fn);    // Take note of atexit handler...
  return 0;  // Always successful
}

//===----------------------------------------------------------------------===//
//
/// getPointerToNamedFunction - This method returns the address of the specified
/// function by using the dynamic loader interface.  As such it is only useful
/// for resolving library symbols, not code generated symbols.
///
void *JIT::getPointerToNamedFunction(const std::string &Name,
                                     bool AbortOnFailure) {
  if (!isSymbolSearchingDisabled()) {
    // Check to see if this is one of the functions we want to intercept.  Note,
    // we cast to intptr_t here to silence a -pedantic warning that complains
    // about casting a function pointer to a normal pointer.
    if (Name == "exit") return (void*)(intptr_t)&jit_exit;
    if (Name == "atexit") return (void*)(intptr_t)&jit_atexit;

    const char *NameStr = Name.c_str();
    // If this is an asm specifier, skip the sentinal.
    if (NameStr[0] == 1) ++NameStr;
    
    // If it's an external function, look it up in the process image...
    void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
    if (Ptr) return Ptr;
    
    // If it wasn't found and if it starts with an underscore ('_') character,
    // and has an asm specifier, try again without the underscore.
    if (Name[0] == 1 && NameStr[0] == '_') {
      Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
      if (Ptr) return Ptr;
    }
    
    // Darwin/PPC adds $LDBLStub suffixes to various symbols like printf.  These
    // are references to hidden visibility symbols that dlsym cannot resolve.
    // If we have one of these, strip off $LDBLStub and try again.
#if defined(__APPLE__) && defined(__ppc__)
    if (Name.size() > 9 && Name[Name.size()-9] == '$' &&
        memcmp(&Name[Name.size()-8], "LDBLStub", 8) == 0) {
      // First try turning $LDBLStub into $LDBL128. If that fails, strip it off.
      // This mirrors logic in libSystemStubs.a.
      std::string Prefix = std::string(Name.begin(), Name.end()-9);
      if (void *Ptr = getPointerToNamedFunction(Prefix+"$LDBL128", false))
        return Ptr;
      if (void *Ptr = getPointerToNamedFunction(Prefix, false))
        return Ptr;
    }
#endif
  }
  
  /// If a LazyFunctionCreator is installed, use it to get/create the function.
  if (LazyFunctionCreator)
    if (void *RP = LazyFunctionCreator(Name))
      return RP;

  if (AbortOnFailure) {
    cerr << "ERROR: Program used external function '" << Name
         << "' which could not be resolved!\n";
    abort();
  }
  return 0;
}
