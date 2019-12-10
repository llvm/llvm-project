//===- OMPConstants.cpp - Helpers related to OpenMP code generation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace omp;

Directive llvm::omp::getOpenMPDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<Directive>(Str)
#define OMP_DIRECTIVE(Enum, Str) .Case(Str, Enum)
#include "llvm/Frontend/OpenMP/OMPKinds.def"
      .Default(OMPD_unknown);
}

StringRef llvm::omp::getOpenMPDirectiveName(Directive Kind) {
  switch (Kind) {
#define OMP_DIRECTIVE(Enum, Str)                                               \
  case Enum:                                                                   \
    return Str;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Invalid OpenMP directive kind");
}
