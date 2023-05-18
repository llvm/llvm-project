//= LoongArchBaseInfo.cpp - Top level definitions for LoongArch MC -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for the LoongArch target useful for the
// compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//

#include "LoongArchBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

namespace LoongArchABI {

ABI computeTargetABI(const Triple &TT, StringRef ABIName) {
  ABI ArgProvidedABI = getTargetABI(ABIName);
  bool Is64Bit = TT.isArch64Bit();
  ABI TripleABI;

  // Figure out the ABI explicitly requested via the triple's environment type.
  switch (TT.getEnvironment()) {
  case llvm::Triple::EnvironmentType::GNUSF:
    TripleABI = Is64Bit ? LoongArchABI::ABI_LP64S : LoongArchABI::ABI_ILP32S;
    break;
  case llvm::Triple::EnvironmentType::GNUF32:
    TripleABI = Is64Bit ? LoongArchABI::ABI_LP64F : LoongArchABI::ABI_ILP32F;
    break;

  // Let the fallback case behave like {ILP32,LP64}D.
  case llvm::Triple::EnvironmentType::GNUF64:
  default:
    TripleABI = Is64Bit ? LoongArchABI::ABI_LP64D : LoongArchABI::ABI_ILP32D;
    break;
  }

  switch (ArgProvidedABI) {
  case LoongArchABI::ABI_Unknown:
    // Fallback to the triple-implied ABI if ABI name is not specified or
    // invalid.
    if (!ABIName.empty())
      errs() << "'" << ABIName
             << "' is not a recognized ABI for this target, ignoring and using "
                "triple-implied ABI\n";
    return TripleABI;

  case LoongArchABI::ABI_ILP32S:
  case LoongArchABI::ABI_ILP32F:
  case LoongArchABI::ABI_ILP32D:
    if (Is64Bit) {
      errs() << "32-bit ABIs are not supported for 64-bit targets, ignoring "
                "target-abi and using triple-implied ABI\n";
      return TripleABI;
    }
    break;

  case LoongArchABI::ABI_LP64S:
  case LoongArchABI::ABI_LP64F:
  case LoongArchABI::ABI_LP64D:
    if (!Is64Bit) {
      errs() << "64-bit ABIs are not supported for 32-bit targets, ignoring "
                "target-abi and using triple-implied ABI\n";
      return TripleABI;
    }
    break;
  }

  if (!ABIName.empty() && TT.hasEnvironment() && ArgProvidedABI != TripleABI)
    errs() << "warning: triple-implied ABI conflicts with provided target-abi '"
           << ABIName << "', using target-abi\n";

  return ArgProvidedABI;
}

ABI getTargetABI(StringRef ABIName) {
  auto TargetABI = StringSwitch<ABI>(ABIName)
                       .Case("ilp32s", ABI_ILP32S)
                       .Case("ilp32f", ABI_ILP32F)
                       .Case("ilp32d", ABI_ILP32D)
                       .Case("lp64s", ABI_LP64S)
                       .Case("lp64f", ABI_LP64F)
                       .Case("lp64d", ABI_LP64D)
                       .Default(ABI_Unknown);
  return TargetABI;
}

// To avoid the BP value clobbered by a function call, we need to choose a
// callee saved register to save the value. The `last` `S` register (s9) is
// used for FP. So we choose the previous (s8) as BP.
MCRegister getBPReg() { return LoongArch::R31; }

} // end namespace LoongArchABI

} // end namespace llvm
