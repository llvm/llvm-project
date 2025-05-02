//===- llvm/MC/MasmParser.h - MASM Parser Interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_MCMASMPARSER_H
#define LLVM_MC_MCPARSER_MCMASMPARSER_H

#include "llvm/MC/MCParser/MCAsmParser.h"

namespace llvm {

/// MASM-type assembler parser interface.
class MCMasmParser : public MCAsmParser {
public:
  virtual bool getDefaultRetIsFar() const = 0;
  virtual void setDefaultRetIsFar(bool IsFar) = 0;

  bool isParsingMasm() const override { return true; }

  static bool classof(const MCAsmParser *AP) { return AP->isParsingMasm(); }
};

} // end namespace llvm

#endif // LLVM_MC_MCPARSER_MCMASMPARSER_H
