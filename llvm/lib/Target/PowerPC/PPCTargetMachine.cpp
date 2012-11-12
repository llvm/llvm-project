//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetMachine.h"
#include "PPC.h"
#include "llvm/PassManager.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

static cl::
opt<bool> DisableCTRLoops("disable-ppc-ctrloops", cl::Hidden,
                        cl::desc("Disable CTR loops for PPC"));

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
}

PPCTargetMachine::PPCTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, is64Bit),
    DL(Subtarget.getDataLayoutString()), InstrInfo(*this),
    FrameLowering(Subtarget), JITInfo(*this, is64Bit),
    TLInfo(*this), TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()),
    STTI(&TLInfo), VTTI(&TLInfo) {

  // The binutils for the BG/P are too old for CFI.
  if (Subtarget.isBGP())
    setMCUseCFI(false);
}

void PPC32TargetMachine::anchor() { }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {
}

void PPC64TargetMachine::anchor() { }

PPC64TargetMachine::PPC64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU,  StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

namespace {
/// PPC Code Generator Pass Configuration Options.
class PPCPassConfig : public TargetPassConfig {
public:
  PPCPassConfig(PPCTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  PPCTargetMachine &getPPCTargetMachine() const {
    return getTM<PPCTargetMachine>();
  }

  virtual bool addPreRegAlloc();
  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *PPCTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PPCPassConfig(this, PM);
}

bool PPCPassConfig::addPreRegAlloc() {
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoops());

  return false;
}

bool PPCPassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createPPCISelDag(getPPCTargetMachine()));
  return false;
}

bool PPCPassConfig::addPreEmitPass() {
  // Must run branch selection immediately preceding the asm printer.
  addPass(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      JITCodeEmitter &JCE) {
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();

  // Machine code emitter pass for PowerPC.
  PM.add(createPPCJITCodeEmitterPass(*this, JCE));

  return false;
}
