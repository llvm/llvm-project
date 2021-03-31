//===--- unittests/DebugInfo/DWARF/DwarfUtils.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfUtils.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

static void initLLVMIfNeeded() {
  static bool gInitialized = false;
  if (!gInitialized) {
    gInitialized = true;
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
  }
}

Triple llvm::dwarf::utils::getNormalizedDefaultTargetTriple() {
  Triple T(Triple::normalize(sys::getDefaultTargetTriple()));

  return T;
}

Triple llvm::dwarf::utils::getDefaultTargetTripleForAddrSize(uint8_t AddrSize) {
  Triple T = getNormalizedDefaultTargetTriple();

  assert((AddrSize == 4 || AddrSize == 8) &&
         "Only 32-bit/64-bit address size variants are supported");

  // If a 32-bit/64-bit address size was specified, try to convert the triple
  // if it is for the wrong variant.
  if (AddrSize == 8 && T.isArch32Bit())
    return T.get64BitArchVariant();
  if (AddrSize == 4 && T.isArch64Bit())
    return T.get32BitArchVariant();
  return T;
}

bool llvm::dwarf::utils::isConfigurationSupported(Triple &T) {
  initLLVMIfNeeded();
  std::string Err;
  return TargetRegistry::lookupTarget(T.getTriple(), Err);
}

bool llvm::dwarf::utils::isObjectEmissionSupported(Triple &T) {
  initLLVMIfNeeded();
  std::string Err;
  const Target *TheTarget = TargetRegistry::lookupTarget(T.getTriple(), Err);
  return TheTarget && TheTarget->hasMCAsmBackend();
}
