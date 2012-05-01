//===-- MBlazeTargetMachine.cpp - Define TargetMachine for MBlaze ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about MBlaze target spec.
//
//===----------------------------------------------------------------------===//

#include "MBlazeTargetMachine.h"
#include "MBlaze.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

extern "C" void LLVMInitializeMBlazeTarget() {
  // Register the target.
  RegisterTargetMachine<MBlazeTargetMachine> X(TheMBlazeTarget);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
MBlazeTargetMachine::
MBlazeTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    DataLayout("E-p:32:32:32-i8:8:8-i16:16:16"),
    InstrInfo(*this),
    FrameLowering(Subtarget),
    TLInfo(*this), TSInfo(*this), ELFWriterInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
}

namespace {
/// MBlaze Code Generator Pass Configuration Options.
class MBlazePassConfig : public TargetPassConfig {
public:
  MBlazePassConfig(MBlazeTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  MBlazeTargetMachine &getMBlazeTargetMachine() const {
    return getTM<MBlazeTargetMachine>();
  }

  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *MBlazeTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MBlazePassConfig(this, PM);
}

// Install an instruction selector pass using
// the ISelDag to gen MBlaze code.
bool MBlazePassConfig::addInstSelector() {
  PM->add(createMBlazeISelDag(getMBlazeTargetMachine()));
  return false;
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
bool MBlazePassConfig::addPreEmitPass() {
  PM->add(createMBlazeDelaySlotFillerPass(getMBlazeTargetMachine()));
  return true;
}
