//===-- AVRTargetMachine.cpp - Define TargetMachine for AVR ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AVR specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "AVRTargetMachine.h"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"

#include "AVRTargetObjectFile.h"
#include "AVR.h"
#include "MCTargetDesc/AVRMCTargetDesc.h"

namespace llvm {

/// Processes a CPU name.
static StringRef getCPU(StringRef CPU) {
  if (CPU.empty() || CPU == "generic") {
    return "avr2";
  }

  return CPU;
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  return RM.hasValue() ? *RM : Reloc::Static;
}

AVRTargetMachine::AVRTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Optional<Reloc::Model> RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
    : LLVMTargetMachine(
          T, "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-i64:8:8-f32:8:8-f64:8:8-n8", TT,
          getCPU(CPU), FS, Options, getEffectiveRelocModel(RM), CM, OL),
      SubTarget(TT, getCPU(CPU), FS, *this) {
  this->TLOF = make_unique<AVRTargetObjectFile>();
  initAsmInfo();
}

namespace {
/// AVR Code Generator Pass Configuration Options.
class AVRPassConfig : public TargetPassConfig {
public:
  AVRPassConfig(AVRTargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  AVRTargetMachine &getAVRTargetMachine() const {
    return getTM<AVRTargetMachine>();
  }

  bool addInstSelector() override;
  void addPreSched2() override;
  void addPreRegAlloc() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *AVRTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AVRPassConfig(this, PM);
}

extern "C" void LLVMInitializeAVRTarget() {
  // Register the target.
  RegisterTargetMachine<AVRTargetMachine> X(TheAVRTarget);
}

const AVRSubtarget *AVRTargetMachine::getSubtargetImpl() const {
  return &SubTarget;
}

const AVRSubtarget *AVRTargetMachine::getSubtargetImpl(const Function &) const {
  return &SubTarget;
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool AVRPassConfig::addInstSelector() {
  return false;
}

void AVRPassConfig::addPreRegAlloc() {
}

void AVRPassConfig::addPreSched2() { }

void AVRPassConfig::addPreEmitPass() {
}

} // end of namespace llvm
