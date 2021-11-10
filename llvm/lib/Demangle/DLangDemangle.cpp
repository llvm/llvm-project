//===--- DLangDemangle.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a demangler for the D programming language as specified
/// in the ABI specification, available at:
/// https://dlang.org/spec/abi.html#name_mangling
///
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/Utility.h"

#include <cstring>

using namespace llvm;
using llvm::itanium_demangle::OutputBuffer;

char *llvm::dlangDemangle(const char *MangledName) {
  if (MangledName == nullptr || strncmp(MangledName, "_D", 2) != 0)
    return nullptr;

  OutputBuffer Demangled;
  if (!initializeOutputBuffer(nullptr, nullptr, Demangled, 1024))
    return nullptr;

  if (strcmp(MangledName, "_Dmain") == 0)
    Demangled << "D main";

  // OutputBuffer's internal buffer is not null terminated and therefore we need
  // to add it to comply with C null terminated strings.
  if (Demangled.getCurrentPosition() > 0) {
    Demangled << '\0';
    Demangled.setCurrentPosition(Demangled.getCurrentPosition() - 1);
    return Demangled.getBuffer();
  }

  free(Demangled.getBuffer());
  return nullptr;
}
