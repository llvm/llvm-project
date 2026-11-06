//===------------ Program.cpp - liboffload program abstraction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Program.h"
#include "OffloadError.h"

#include "llvm/BinaryFormat/Magic.h"

#include <cstdint>
#include <cstring>

using namespace llvm;

Expected<ProgramTy> ProgramTy::create(ol_context_handle_t Context,
                                      ol_device_handle_t Device,
                                      __tgt_device_image *Img) {
  ol_program_handle_t Handle;
  size_t ImageSize = reinterpret_cast<uintptr_t>(Img->ImageEnd) -
                     reinterpret_cast<uintptr_t>(Img->ImageStart);
  if (auto Res =
          olCreateProgram(Context, Device, Img->ImageStart, ImageSize, &Handle))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to load binary %p: %s", Img,
                                     Res->Details);

  StringRef Input(reinterpret_cast<const char *>(Img->ImageStart), ImageSize);
  MemoryBufferRef IRImage;
  if (identify_magic(Input) == file_magic::bitcode)
    IRImage = MemoryBufferRef(Input, "IRImage");
  return ProgramTy(Handle, IRImage);
}

Expected<MemoryBufferRef> ProgramTy::getImage() const {
  void *Base = nullptr;
  size_t Size = 0;
  if (auto Res = olGetProgramInfo(Handle, OL_PROGRAM_INFO_IMAGE_BASE,
                                  sizeof(Base), &Base))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to get program image address: %s",
                                     Res->Details);
  if (auto Res = olGetProgramInfo(Handle, OL_PROGRAM_INFO_IMAGE_SIZE,
                                  sizeof(Size), &Size))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to get program image size: %s",
                                     Res->Details);
  if (!Base || !Size)
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "program has an empty image");
  return MemoryBufferRef(StringRef(static_cast<const char *>(Base), Size),
                         "Image");
}

Expected<void *> ProgramTy::getGlobalAddress(const char *Name,
                                             size_t *Size) const {
  ol_symbol_handle_t Symbol;
  if (auto Res =
          olGetSymbol(Handle, Name, OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Symbol))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to find global symbol %s: %s",
                                     Name, Res->Details);

  void *Address = nullptr;
  if (auto Res = olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS,
                                 sizeof(Address), &Address))
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "failed to get device address of global symbol %s: %s", Name,
        Res->Details);

  if (Size) {
    if (auto Res = olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                   sizeof(*Size), Size))
      return error::createOffloadError(
          error::ErrorCode::INVALID_BINARY,
          "failed to get size of global symbol %s: %s", Name, Res->Details);
  }

  return Address;
}

Error ProgramTy::readGlobalFromImage(const char *Name, void *Dst,
                                     size_t Size) const {
  ol_symbol_handle_t Symbol;
  if (auto Res =
          olGetSymbol(Handle, Name, OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Symbol))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to find global symbol %s: %s",
                                     Name, Res->Details);

  void *ImageAddress = nullptr;
  if (auto Res =
          olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_HOST_ADDRESS,
                          sizeof(ImageAddress), &ImageAddress))
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "failed to get image address of global symbol %s: %s", Name,
        Res->Details);

  size_t SymbolSize = 0;
  if (auto Res = olGetSymbolInfo(Symbol, OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE,
                                 sizeof(SymbolSize), &SymbolSize))
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "failed to get size of global symbol %s: %s", Name, Res->Details);
  if (SymbolSize != Size)
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "global symbol %s has %zu bytes in the image but %zu were requested",
        Name, SymbolSize, Size);

  if (!ImageAddress)
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "global symbol %s has no contents in the program image", Name);

  std::memcpy(Dst, ImageAddress, Size);
  return Error::success();
}

Expected<ol_symbol_handle_t>
ProgramTy::getKernelAddress(const char *Name) const {
  ol_symbol_handle_t Symbol;
  if (auto Res = olGetSymbol(Handle, Name, OL_SYMBOL_KIND_KERNEL, &Symbol))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to find kernel symbol %s: %s",
                                     Name, Res->Details);

  return Symbol;
}
