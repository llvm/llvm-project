//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetMachine.h"
#include "Mips.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMipsTarget() {
  // Register the target.
  RegisterTargetMachine<MipsebTargetMachine> X(TheMipsTarget);
  RegisterTargetMachine<MipselTargetMachine> Y(TheMipselTarget);
  RegisterTargetMachine<Mips64ebTargetMachine> A(TheMips64Target);
  RegisterTargetMachine<Mips64elTargetMachine> B(TheMips64elTarget);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
MipsTargetMachine::
MipsTargetMachine(const Target &T, StringRef TT,
                  StringRef CPU, StringRef FS, const TargetOptions &Options,
                  Reloc::Model RM, CodeModel::Model CM,
                  CodeGenOpt::Level OL,
                  bool isLittle)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, isLittle),
    DataLayout(isLittle ?
               (Subtarget.isABI_N64() ?
                "e-p:64:64:64-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32" :
                "e-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32") :
               (Subtarget.isABI_N64() ?
                "E-p:64:64:64-i8:8:32-i16:16:32-i64:64:64-f128:128:128-n32" :
                "E-p:32:32:32-i8:8:32-i16:16:32-i64:64:64-n32")),
    InstrInfo(*this),
    FrameLowering(Subtarget),
    TLInfo(*this), TSInfo(*this), JITInfo() {
}

void MipsebTargetMachine::anchor() { }

MipsebTargetMachine::
MipsebTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void MipselTargetMachine::anchor() { }

MipselTargetMachine::
MipselTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void Mips64ebTargetMachine::anchor() { }

Mips64ebTargetMachine::
Mips64ebTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void Mips64elTargetMachine::anchor() { }

Mips64elTargetMachine::
Mips64elTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace {
/// Mips Code Generator Pass Configuration Options.
class MipsPassConfig : public TargetPassConfig {
public:
  MipsPassConfig(MipsTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  MipsTargetMachine &getMipsTargetMachine() const {
    return getTM<MipsTargetMachine>();
  }

  const MipsSubtarget &getMipsSubtarget() const {
    return *getMipsTargetMachine().getSubtargetImpl();
  }

  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *MipsTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MipsPassConfig(this, PM);
}

// Install an instruction selector pass using
// the ISelDag to gen Mips code.
bool MipsPassConfig::addInstSelector() {
  PM->add(createMipsISelDag(getMipsTargetMachine()));
  return false;
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
bool MipsPassConfig::addPreEmitPass() {
  PM->add(createMipsDelaySlotFillerPass(getMipsTargetMachine()));
  return true;
}

bool MipsPassConfig::addPreRegAlloc() {
  // Do not restore $gp if target is Mips64.
  // In N32/64, $gp is a callee-saved register.
  if (!getMipsSubtarget().hasMips64())
    PM->add(createMipsEmitGPRestorePass(getMipsTargetMachine()));
  return true;
}

bool MipsPassConfig::addPreSched2() {
  PM->add(createMipsExpandPseudoPass(getMipsTargetMachine()));
  return true;
}

bool MipsTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                       JITCodeEmitter &JCE) {
  // Machine code emitter pass for Mips.
  PM.add(createMipsJITCodeEmitterPass(*this, JCE));
  return false;
}
