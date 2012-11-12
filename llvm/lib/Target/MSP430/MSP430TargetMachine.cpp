//===-- MSP430TargetMachine.cpp - Define TargetMachine for MSP430 ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the MSP430 target.
//
//===----------------------------------------------------------------------===//

#include "MSP430TargetMachine.h"
#include "MSP430.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMSP430Target() {
  // Register the target.
  RegisterTargetMachine<MSP430TargetMachine> X(TheMSP430Target);
}

MSP430TargetMachine::MSP430TargetMachine(const Target &T,
                                         StringRef TT,
                                         StringRef CPU,
                                         StringRef FS,
                                         const TargetOptions &Options,
                                         Reloc::Model RM, CodeModel::Model CM,
                                         CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    // FIXME: Check DataLayout string.
    DL("e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"),
    InstrInfo(*this), TLInfo(*this), TSInfo(*this),
    FrameLowering(Subtarget), STTI(&TLInfo), VTTI(&TLInfo) { }

namespace {
/// MSP430 Code Generator Pass Configuration Options.
class MSP430PassConfig : public TargetPassConfig {
public:
  MSP430PassConfig(MSP430TargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  MSP430TargetMachine &getMSP430TargetMachine() const {
    return getTM<MSP430TargetMachine>();
  }

  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *MSP430TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MSP430PassConfig(this, PM);
}

bool MSP430PassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createMSP430ISelDag(getMSP430TargetMachine(), getOptLevel()));
  return false;
}

bool MSP430PassConfig::addPreEmitPass() {
  // Must run branch selection immediately preceding the asm printer.
  addPass(createMSP430BranchSelectionPass());
  return false;
}
