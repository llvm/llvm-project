//===- lib/MC/MCELF.h - ELF MC --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some support functions used by the ELF Streamer and
// ObjectWriter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELF_H
#define LLVM_MC_MCELF_H

#include "llvm/MC/MCExpr.h"

namespace llvm {
class MCSymbolData;

class MCELF {
 public:
  static void SetBinding(MCSymbolData &SD, unsigned Binding);
  static unsigned GetBinding(const MCSymbolData &SD);
  static void SetType(MCSymbolData &SD, unsigned Type);
  static unsigned GetType(const MCSymbolData &SD);
  static void SetVisibility(MCSymbolData &SD, unsigned Visibility);
  static unsigned GetVisibility(MCSymbolData &SD);
};

}

#endif
