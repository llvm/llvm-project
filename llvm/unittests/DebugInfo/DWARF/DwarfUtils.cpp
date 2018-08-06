//===--- unittests/DebugInfo/DWARF/DwarfUtils.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
