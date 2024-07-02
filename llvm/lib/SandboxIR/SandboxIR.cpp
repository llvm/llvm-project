//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm;
using namespace sandboxir;

sandboxir::Value::Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx)
    : SubclassID(SubclassID), Val(Val), Ctx(Ctx) {
#ifndef NDEBUG
  UID = 0; // FIXME: Once SBContext is available.
#endif
}

#ifndef NDEBUG
std::string sandboxir::Value::getName() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void sandboxir::Value::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void sandboxir::Value::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";
}

void sandboxir::Value::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void sandboxir::Value::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void sandboxir::Value::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void sandboxir::User::dumpCommonHeader(raw_ostream &OS) const {
  Value::dumpCommonHeader(OS);
  // TODO: This is incomplete
}
#endif // NDEBUG
