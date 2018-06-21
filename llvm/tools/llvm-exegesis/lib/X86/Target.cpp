//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "../Target.h"

#include "X86.h"

namespace exegesis {

namespace {

class ExegesisX86Target : public ExegesisTarget {
  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override {
    // Lowers FP pseudo-instructions, e.g. ABS_Fp32 -> ABS_F.
    // FIXME: Enable when the exegesis assembler no longer does
    // Properties.reset(TracksLiveness);
    // PM.add(llvm::createX86FloatingPointStackifierPass());
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::x86_64 || Arch == llvm::Triple::x86;
  }
};

} // namespace

static ExegesisTarget* getTheExegesisX86Target() {
  static ExegesisX86Target Target;
  return &Target;
}

void InitializeX86ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisX86Target());
}

}  // namespace exegesis
