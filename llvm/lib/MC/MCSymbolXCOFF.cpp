//===- lib/MC/MCSymbolXCOFF.cpp - XCOFF Code Symbol Representation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionXCOFF.h"

using namespace llvm;

MCSectionXCOFF *MCSymbolXCOFF::getRepresentedCsect() const {
  assert(RepresentedCsect &&
         "Trying to get csect representation of this symbol but none was set.");
  assert((!getName().equals(getUnqualifiedName()) ||
          RepresentedCsect->getCSectType() == XCOFF::XTY_ER) &&
         "Symbol does not represent a csect; MCSectionXCOFF that represents "
         "the symbol should not be (but is) set.");
  assert(getSymbolTableName().equals(RepresentedCsect->getSymbolTableName()) &&
         "SymbolTableNames need to be the same for this symbol and its csect "
         "representation.");
  return RepresentedCsect;
}

void MCSymbolXCOFF::setRepresentedCsect(MCSectionXCOFF *C) {
  assert(C && "Assigned csect should not be null.");
  assert((!RepresentedCsect || RepresentedCsect == C) &&
         "Trying to set a csect that doesn't match the one that"
         "this symbol is already mapped to.");
  assert((!getName().equals(getUnqualifiedName()) ||
          C->getCSectType() == XCOFF::XTY_ER) &&
         "Symbol does not represent a csect; can only set a MCSectionXCOFF "
         "representation for a csect.");
  assert(getSymbolTableName().equals(C->getSymbolTableName()) &&
         "SymbolTableNames need to be the same for this symbol and its csect "
         "representation.");
  RepresentedCsect = C;
}
