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

std::string llvm::demangle(const std::string &MangledName) {
  char *Demangled;
  if (MangledName.compare(0, 2, "_Z") == 0)
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
