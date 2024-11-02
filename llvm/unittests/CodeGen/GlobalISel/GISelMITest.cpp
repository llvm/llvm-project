//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"

namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const LLT Ty) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  Ty.print(SS);
  OS << Repr;
  return OS;
}

std::ostream &
operator<<(std::ostream &OS, const MachineFunction &MF) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  MF.print(SS);
  OS << Repr;
  return OS;
}

}

std::unique_ptr<LLVMTargetMachine>
AArch64GISelMITest::createTargetMachine() const {
  Triple TargetTriple("aarch64--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      T->createTargetMachine("AArch64", "", "", Options, std::nullopt,
                             std::nullopt, CodeGenOptLevel::Aggressive)));
}

void AArch64GISelMITest::getTargetTestModuleString(SmallString<512> &S,
                                                   StringRef MIRFunc) const {
  (Twine(R"MIR(
---
...
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: _ }
  - { id: 1, class: _ }
  - { id: 2, class: _ }
  - { id: 3, class: _ }
body: |
  bb.1:
    liveins: $x0, $x1, $x2, $x4

    %0(s64) = COPY $x0
    %1(s64) = COPY $x1
    %2(s64) = COPY $x2
)MIR") +
   Twine(MIRFunc) + Twine("...\n"))
      .toNullTerminatedStringRef(S);
}

std::unique_ptr<LLVMTargetMachine>
AMDGPUGISelMITest::createTargetMachine() const {
  Triple TargetTriple("amdgcn-amd-amdhsa");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      T->createTargetMachine("amdgcn-amd-amdhsa", "gfx900", "", Options,
                             std::nullopt, std::nullopt,
                             CodeGenOptLevel::Aggressive)));
}

void AMDGPUGISelMITest::getTargetTestModuleString(
  SmallString<512> &S, StringRef MIRFunc) const {
  (Twine(R"MIR(
---
...
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: _ }
  - { id: 1, class: _ }
  - { id: 2, class: _ }
  - { id: 3, class: _ }
body: |
  bb.1:
    liveins: $vgpr0, $vgpr1, $vgpr2

    %0(s32) = COPY $vgpr0
    %1(s32) = COPY $vgpr1
    %2(s32) = COPY $vgpr2
)MIR") + Twine(MIRFunc) + Twine("...\n"))
                            .toNullTerminatedStringRef(S);
}
