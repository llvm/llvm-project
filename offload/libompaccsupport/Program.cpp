//===------------ Program.cpp - liboffload program abstraction ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Program.h"
#include "OmpAccError.h"

#include <cstdint>

using namespace llvm;
using namespace llvm::omp::target;

// Temporary helper to help transition of libomptarget to liboffload: returns
// the opaque plugin kernel handle backing a kernel symbol, for use with the
// legacy plugin launch interface until kernel launch itself is migrated to
// liboffload.
extern "C" void *__ol_tgt_GetKernelFromSymbol(ol_symbol_handle_t Symbol);

Expected<ProgramTy> ProgramTy::create(ol_context_handle_t Context,
                                      ol_device_handle_t Device,
                                      __tgt_device_image *Img) {
  ol_program_handle_t Handle;
  size_t ImageSize = reinterpret_cast<uintptr_t>(Img->ImageEnd) -
                     reinterpret_cast<uintptr_t>(Img->ImageStart);
  if (auto Res =
          olCreateProgram(Context, Device, Img->ImageStart, ImageSize, &Handle))
    return createError(ErrorCode::InvalidBinary, "failed to load binary %p: %s",
                       Img, Res->Details);

  return ProgramTy(Handle);
}

Expected<void *> ProgramTy::getGlobalAddress(const char *Name,
                                             size_t *Size) const {
  ol_symbol_handle_t Symbol;
  if (auto Res =
          olGetSymbol(Handle, Name, OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Symbol))
    return createError(ErrorCode::InvalidBinary,
                       "failed to find global symbol %s: %s", Name,
                       Res->Details);

  void *Address = nullptr;
  if (auto Res = olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                                 sizeof(Address), &Address))
    return createError(ErrorCode::InvalidBinary,
                       "failed to get device address of global symbol %s: %s",
                       Name, Res->Details);

  if (Size && olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                              sizeof(*Size), Size))
    *Size = 0;

  return Address;
}

Expected<void *> ProgramTy::getKernelAddress(const char *Name) const {
  ol_symbol_handle_t Symbol;
  if (auto Res = olGetSymbol(Handle, Name, OL_SYMBOL_KIND_KERNEL, &Symbol))
    return createError(ErrorCode::InvalidBinary,
                       "failed to find kernel symbol %s: %s", Name,
                       Res->Details);

  return __ol_tgt_GetKernelFromSymbol(Symbol);
}
