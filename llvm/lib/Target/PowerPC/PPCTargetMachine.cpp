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
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::
opt<bool> DisableCTRLoops("disable-ppc-ctrloops", cl::Hidden,
                        cl::desc("Disable CTR loops for PPC"));

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
  RegisterTargetMachine<PPC64TargetMachine> C(ThePPC64LETarget);
}

/// Return the datalayout string of a subtarget.
static std::string getDataLayoutString(const PPCSubtarget &ST) {
  const Triple &T = ST.getTargetTriple();

  // PPC is big endian
  std::string Ret = "E";

  // PPC64 has 64 bit pointers, PPC32 has 32 bit pointers.
  if (ST.isPPC64())
    Ret += "-p:64:64";
  else
    Ret += "-p:32:32";

  // Note, the alignment values for f64 and i64 on ppc64 in Darwin
  // documentation are wrong; these are correct (i.e. "what gcc does").
  if (ST.isPPC64() || ST.isSVR4ABI())
    Ret += "-f64:64:64-i64:64:64";
  else
    Ret += "-f64:32:64";

  // Set support for 128 floats depending on the ABI.
  if (!ST.isPPC64() && ST.isSVR4ABI())
    Ret += "-f128:64:128";

  // Some ABIs support 128 bit vectors.
  if (ST.isPPC64() && ST.isSVR4ABI())
    Ret += "-v128:128:128";

  // PPC64 has 32 and 64 bit register, PPC32 has only 32 bit ones.
  if (ST.isPPC64())
    Ret += "-n32:64";
  else
    Ret += "-n32";

  return Ret;
}

PPCTargetMachine::PPCTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, is64Bit),
    DL(getDataLayoutString(Subtarget)), InstrInfo(*this),
    FrameLowering(Subtarget), JITInfo(*this, is64Bit),
    TLInfo(*this), TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {

  // The binutils for the BG/P are too old for CFI.
  if (Subtarget.isBGP())
    setMCUseCFI(false);
  initAsmInfo();
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

  const PPCSubtarget &getPPCSubtarget() const {
    return *getPPCTargetMachine().getSubtargetImpl();
  }

  virtual bool addPreISel();
  virtual bool addILPOpts();
  virtual bool addInstSelector();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *PPCTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PPCPassConfig(this, PM);
}

bool PPCPassConfig::addPreISel() {
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoops(getPPCTargetMachine()));

  return false;
}

bool PPCPassConfig::addILPOpts() {
  if (getPPCSubtarget().hasISEL()) {
    addPass(&EarlyIfConverterID);
    return true;
  }

  return false;
}

bool PPCPassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createPPCISelDag(getPPCTargetMachine()));

#ifndef NDEBUG
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoopsVerify());
#endif

  return false;
}

bool PPCPassConfig::addPreSched2() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID);

  return true;
}

bool PPCPassConfig::addPreEmitPass() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createPPCEarlyReturnPass());
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

void PPCTargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  // Add first the target-independent BasicTTI pass, then our PPC pass. This
  // allows the PPC pass to delegate to the target independent layer when
  // appropriate.
  PM.add(createBasicTargetTransformInfoPass(this));
  PM.add(createPPCTargetTransformInfoPass(this));
}

