//===- SystemZHLASMAsmStreamer.cpp - HLASM Assembly Text Output -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZHLASMAsmStreamer.h"

void SystemZHLASMAsmStreamer::changeSection(MCSection *Section,
                                            uint32_t Subsection) {
  MCStreamer::changeSection(Section, Subsection);
}

void SystemZHLASMAsmStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCStreamer::emitLabel(Symbol, Loc);

  Symbol->print(OS, MAI);
  // TODO: update LabelSuffix in SystemZMCAsmInfoGOFF once tests have been
  // moved to HLASM syntax.
  // OS << MAI->getLabelSuffix();
  OS << '\n';
}
