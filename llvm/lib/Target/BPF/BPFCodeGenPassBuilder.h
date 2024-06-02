//===- BPFCodeGenPassBuilder.h - Build BPF codegen pipeline -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFCODEGENPASSBUILDER_H
#define LLVM_LIB_TARGET_BPF_BPFCODEGENPASSBUILDER_H

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"

namespace llvm {

class BPFTargetMachine;

class BPFCodeGenPassBuilder
    : public CodeGenPassBuilder<BPFCodeGenPassBuilder, BPFTargetMachine> {
public:
  BPFCodeGenPassBuilder(BPFTargetMachine &TM, const CGPassBuilderOption &Opts,
                        PassInstrumentationCallbacks *PIC);

  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_BPF_BPFCODEGENPASSBUILDER_H
