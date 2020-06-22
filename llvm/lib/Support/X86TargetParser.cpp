//===-- X86TargetParser - Parser for X86 features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise X86 hardware features.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/X86TargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

bool checkCPUKind(llvm::X86::CPUKind Kind, bool Only64Bit) {
  using namespace X86;
  // Perform any per-CPU checks necessary to determine if this CPU is
  // acceptable.
  switch (Kind) {
  case CK_None:
    // No processor selected!
    return false;
#define PROC(ENUM, STRING, IS64BIT)                                            \
  case CK_##ENUM:                                                              \
    return IS64BIT || !Only64Bit;
#include "llvm/Support/X86TargetParser.def"
  }
  llvm_unreachable("Unhandled CPU kind");
}

X86::CPUKind llvm::X86::parseArchX86(StringRef CPU, bool Only64Bit) {
  X86::CPUKind Kind = llvm::StringSwitch<CPUKind>(CPU)
#define PROC(ENUM, STRING, IS64BIT) .Case(STRING, CK_##ENUM)
#define PROC_ALIAS(ENUM, ALIAS) .Case(ALIAS, CK_##ENUM)
#include "llvm/Support/X86TargetParser.def"
      .Default(CK_None);

  if (!checkCPUKind(Kind, Only64Bit))
    Kind = CK_None;

  return Kind;
}

void llvm::X86::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values,
                                     bool Only64Bit) {
#define PROC(ENUM, STRING, IS64BIT)                                            \
  if (IS64BIT || !Only64Bit)                                                   \
    Values.emplace_back(STRING);
  // For aliases we need to lookup the CPUKind to get the 64-bit ness.
#define PROC_ALIAS(ENUM, ALIAS)                                                \
  if (checkCPUKind(CK_##ENUM, Only64Bit))                                      \
    Values.emplace_back(ALIAS);
#include "llvm/Support/X86TargetParser.def"
}
