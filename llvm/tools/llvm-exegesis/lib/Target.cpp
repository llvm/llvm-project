//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Target.h"

namespace exegesis {

ExegesisTarget::~ExegesisTarget() {}  // anchor.

static ExegesisTarget* FirstTarget = nullptr;

const ExegesisTarget* ExegesisTarget::lookup(llvm::StringRef TT) {
  const llvm::Triple::ArchType Arch = llvm::Triple(TT).getArch();
  for (const ExegesisTarget* T = FirstTarget; T != nullptr; T = T->Next) {
    if (T->matchesArch(Arch)) return T;
  }
  return nullptr;
}

void ExegesisTarget::registerTarget(ExegesisTarget *Target){
  if (FirstTarget == nullptr) {
    FirstTarget = Target;
    return;
  }
  assert(Target->Next == nullptr && "target has already been registered");
  if (Target->Next != nullptr)
    return;
  Target->Next = FirstTarget;
  FirstTarget = Target;
}
}  // namespace exegesis
