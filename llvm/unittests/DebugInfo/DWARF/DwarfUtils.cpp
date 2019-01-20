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

Triple llvm::dwarf::utils::getHostTripleForAddrSize(uint8_t AddrSize) {
  Triple T(Triple::normalize(LLVM_HOST_TRIPLE));

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
