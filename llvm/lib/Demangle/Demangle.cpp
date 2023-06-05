//===-- Demangle.cpp - Common demangling functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains definitions of common demangling functions.
///
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include <cstdlib>
#include <cstring>

static bool isItaniumEncoding(const char *S) {
  // Itanium encoding requires 1 or 3 leading underscores, followed by 'Z'.
  return std::strncmp(S, "_Z", 2) == 0 || std::strncmp(S, "___Z", 4) == 0;
}

static bool isRustEncoding(const char *S) { return S[0] == '_' && S[1] == 'R'; }

static bool isDLangEncoding(const std::string &MangledName) {
  return MangledName.size() >= 2 && MangledName[0] == '_' &&
         MangledName[1] == 'D';
}

std::string llvm::demangle(const std::string &MangledName) {
  std::string Result;
  const char *S = MangledName.c_str();

  if (nonMicrosoftDemangle(MangledName, Result))
    return Result;

  if (S[0] == '_' && nonMicrosoftDemangle(MangledName.substr(1), Result))
    return Result;

  if (char *Demangled = microsoftDemangle(S, nullptr, nullptr)) {
    Result = Demangled;
    std::free(Demangled);
    return Result;
  }

  return MangledName;
}

bool llvm::nonMicrosoftDemangle(std::string_view MangledName,
                                std::string &Result) {
  char *Demangled = nullptr;
  if (isItaniumEncoding(MangledName.data()))
    Demangled = itaniumDemangle(MangledName);
  else if (isRustEncoding(MangledName.data()))
    Demangled = rustDemangle(MangledName);
  else if (isDLangEncoding(MangledName.data()))
    Demangled = dlangDemangle(MangledName.data());

  if (!Demangled)
    return false;

  Result = Demangled;
  std::free(Demangled);
  return true;
}
