/* --- PEMCExpr.cpp --- */

/* ------------------------------------------
author: undefined
date: 5/22/2025
------------------------------------------ */

#include "PEMCExpr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void PEMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  switch (Kd) {
  case HI:
    OS << "hi(";
    break;
  case LO:
    OS << "lo(";
    break;
  default:
    break;
  }
  Expr->print(OS, MAI, true);
  switch (Kd) {
  case HI:
  case LO:
    OS << ")";
    break;
  default:
    break;
  }
}