//===- MCSemExpr.cpp - Semantic Level Expressions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSemExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void printReg(raw_ostream &OS, MCRegister Reg, const MCRegisterInfo *MRI) {
  if (!Reg.isValid()) {
    OS << "$noreg";
    return;
  }
  if (MRI)
    OS << '$' << MRI->getName(Reg);
  else
    OS << "$physreg" << Reg.id();
}

void printScale(raw_ostream &OS, int64_t A) { OS << A << '*'; }

/// Print offset, or nothing when B is 0.
void printOffset(raw_ostream &OS, int64_t B) {
  if (B == 0)
    return;
  if (B < 0)
    OS << " - " << -(uint64_t)(B);
  else
    OS << " + " << B;
}

} // namespace

void MCSemAddrExpr::print(raw_ostream &OS, const MCRegisterInfo *MRI) const {
  if (isConstant()) {
    OS << B;
    return;
  }
  printScale(OS, A);
  printReg(OS, Reg, MRI);
  printOffset(OS, B);
}

void MCSemLeaf::print(raw_ostream &OS, const MCRegisterInfo *MRI) const {
  if (isReg()) {
    printReg(OS, Reg, MRI);
    return;
  }
  assert(isMem() && "Invalid semantic leaf kind");
  OS << "mem[";
  Addr.print(OS, MRI);
  OS << ']';
}

void MCSemExpr::print(raw_ostream &OS, const MCRegisterInfo *MRI) const {
  if (isConstant()) {
    OS << B;
    return;
  }
  printScale(OS, A);
  Leaf.print(OS, MRI);
  printOffset(OS, B);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCSemAddrExpr::dump() const {
  print(dbgs(), nullptr);
  dbgs() << '\n';
}

LLVM_DUMP_METHOD void MCSemLeaf::dump() const {
  print(dbgs(), nullptr);
  dbgs() << '\n';
}

LLVM_DUMP_METHOD void MCSemExpr::dump() const {
  print(dbgs(), nullptr);
  dbgs() << '\n';
}
#endif
