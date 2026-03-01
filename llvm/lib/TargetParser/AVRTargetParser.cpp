//===-- AVRTargetParser - Parser for AVR target features ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a target parser to recognise AVR hardware features.
///
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/AVRTargetParser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Errc.h"

using namespace llvm;

Expected<std::string> AVR::getFeatureSetFromEFlag(const unsigned EFlag) {
  static const DenseMap<unsigned, StringRef> EFlagToFeatureSet = {
      {ELF::EF_AVR_ARCH_AVR1, "avr1"},
      {ELF::EF_AVR_ARCH_AVR2, "avr2"},
      {ELF::EF_AVR_ARCH_AVR25, "avr25"},
      {ELF::EF_AVR_ARCH_AVR3, "avr3"},
      {ELF::EF_AVR_ARCH_AVR31, "avr31"},
      {ELF::EF_AVR_ARCH_AVR35, "avr35"},
      {ELF::EF_AVR_ARCH_AVR4, "avr4"},
      {ELF::EF_AVR_ARCH_AVR5, "avr5"},
      {ELF::EF_AVR_ARCH_AVR51, "avr51"},
      {ELF::EF_AVR_ARCH_AVR6, "avr6"},
      {ELF::EF_AVR_ARCH_AVRTINY, "avrtiny"},
      {ELF::EF_AVR_ARCH_XMEGA1, "xmega1"},
      {ELF::EF_AVR_ARCH_XMEGA2, "xmega2"},
      {ELF::EF_AVR_ARCH_XMEGA3, "xmega3"},
      {ELF::EF_AVR_ARCH_XMEGA4, "xmega4"},
      {ELF::EF_AVR_ARCH_XMEGA5, "xmega"},
      {ELF::EF_AVR_ARCH_XMEGA6, "xmega"},
      {ELF::EF_AVR_ARCH_XMEGA7, "xmega"},
  };

  auto It = EFlagToFeatureSet.find(EFlag);
  if (It != EFlagToFeatureSet.end())
    return It->second.str();

  return createStringError(errc::invalid_argument,
                           "unrecognised AVR version, 0x" +
                               Twine::utohexstr(EFlag));
}
