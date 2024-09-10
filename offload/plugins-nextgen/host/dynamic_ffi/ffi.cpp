//===--- generic-elf-64bit/dynamic_ffi/ffi.cpp -------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of the FFI api by calling into the FFI library via dlopen
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"

#include "Shared/Debug.h"
#include <memory>

#include "DLWrap.h"
#include "ffi.h"

DLWRAP_INITIALIZE()

DLWRAP(ffi_call, 4);
DLWRAP(ffi_prep_cif, 5);

DLWRAP_FINALIZE()

ffi_type ffi_type_void;
ffi_type ffi_type_pointer;

// Name of the FFI shared library.
constexpr const char *FFI_PATH = "libffi.so";

#define DYNAMIC_FFI_SUCCESS 0
#define DYNAMIC_FFI_FAIL 1

// Initializes the dynamic FFI wrapper.
uint32_t ffi_init() {
  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(FFI_PATH, &ErrMsg));

  if (!DynlibHandle->isValid()) {
    DP("Unable to load library '%s': %s!\n", FFI_PATH, ErrMsg.c_str());
    return DYNAMIC_FFI_FAIL;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, FFI_PATH);
      return DYNAMIC_FFI_FAIL;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);

    *dlwrap::pointer(I) = P;
  }

#define DYNAMIC_INIT(SYMBOL)                                                   \
  {                                                                            \
    void *SymbolPtr = DynlibHandle->getAddressOfSymbol(#SYMBOL);               \
    if (!SymbolPtr) {                                                          \
      DP("Unable to find '%s' in '%s'!\n", #SYMBOL, FFI_PATH);                 \
      return DYNAMIC_FFI_FAIL;                                                 \
    }                                                                          \
    SYMBOL = *reinterpret_cast<decltype(SYMBOL) *>(SymbolPtr);                 \
  }
  DYNAMIC_INIT(ffi_type_void);
  DYNAMIC_INIT(ffi_type_pointer);
#undef DYNAMIC_INIT

  return DYNAMIC_FFI_SUCCESS;
}
