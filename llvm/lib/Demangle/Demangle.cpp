//===-- Demangle.cpp - Common demangling functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains definitions of common demangling functions.
///
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"

static bool isItaniumEncoding(const std::string &MangledName) {
  size_t Pos = MangledName.find_first_not_of('_');
  // A valid Itanium encoding requires 1-4 leading underscores, followed by 'Z'.
  return Pos > 0 && Pos <= 4 && MangledName[Pos] == 'Z';
}

std::string llvm::demangle(const std::string &MangledName) {
  char *Demangled;
  if (isItaniumEncoding(MangledName))
    Demangled = itaniumDemangle(MangledName.c_str(), nullptr, nullptr, nullptr);
  else
    Demangled =
        microsoftDemangle(MangledName.c_str(), nullptr, nullptr, nullptr);

  if (!Demangled)
    return MangledName;

  std::string Ret = Demangled;
  free(Demangled);
  return Ret;
}
