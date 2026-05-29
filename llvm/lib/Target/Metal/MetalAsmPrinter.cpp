//===-- MetalAsmPrinter.cpp - Metal assembly writer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stub AsmPrinter for the Metal backend. Metal `.metallib` is just an opaque
// byte blob produced by the MetalEmbedderPass; the AsmPrinter only needs to
// emit the embedded global into its requested section so the MC
// MetalLibObjectWriter can find and copy its bytes verbatim.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/MetalTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class MetalAsmPrinter : public AsmPrinter {
public:
  explicit MetalAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "Metal Assembly Printer"; }
  void emitGlobalVariable(const GlobalVariable *GV) override;
  bool runOnMachineFunction(MachineFunction &) override { return false; }
};

} // namespace

void MetalAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  // Only the embedded .metallib blob is meaningful in this backend.
  if (!GV->hasInitializer() || GV->hasImplicitSection() || !GV->hasSection())
    return;
  if (GV->getSection() == "llvm.metadata")
    return;
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GV, TM);
  MCSection *TheSection = getObjFileLowering().SectionForGlobal(GV, GVKind, TM);
  OutStreamer->switchSection(TheSection);
  emitGlobalConstant(GV->getDataLayout(), GV->getInitializer());
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeMetalAsmPrinter() {
  RegisterAsmPrinter<MetalAsmPrinter> X(getTheMetalTarget());
}
