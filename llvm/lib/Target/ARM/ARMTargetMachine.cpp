//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARMTargetMachine.h"
#include "ARMFrameLowering.h"
#include "ARM.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::opt<bool>
EnableGlobalMerge("global-merge", cl::Hidden,
                  cl::desc("Enable global merge pass"),
                  cl::init(true));

extern "C" void LLVMInitializeARMTarget() {
  // Register the target.
  RegisterTargetMachine<ARMTargetMachine> X(TheARMTarget);
  RegisterTargetMachine<ThumbTargetMachine> Y(TheThumbTarget);
}


/// TargetMachine ctor - Create an ARM architecture model.
///
ARMBaseTargetMachine::ARMBaseTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    JITInfo(),
    InstrItins(Subtarget.getInstrItineraryData()) {
  // Default to soft float ABI
  if (Options.FloatABIType == FloatABI::Default)
    this->Options.FloatABIType = FloatABI::Soft;
}

void ARMTargetMachine::anchor() { }

ARMTargetMachine::ARMTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
  : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    InstrInfo(Subtarget),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "v128:32:128-v64:32:64-n32-S32") :
               Subtarget.isAAPCS_ABI() ?
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "v128:64:128-v64:64:64-n32-S64") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "v128:64:128-v64:64:64-n32-S32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget) {
  if (!Subtarget.hasARMOps())
    report_fatal_error("CPU: '" + Subtarget.getCPUString() + "' does not "
                       "support ARM mode execution!");
}

void ThumbTargetMachine::anchor() { }

ThumbTargetMachine::ThumbTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    InstrInfo(Subtarget.hasThumb2()
              ? ((ARMBaseInstrInfo*)new Thumb2InstrInfo(Subtarget))
              : ((ARMBaseInstrInfo*)new Thumb1InstrInfo(Subtarget))),
    DataLayout(Subtarget.isAPCS_ABI() ?
               std::string("e-p:32:32-f64:32:64-i64:32:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:32:128-v64:32:64-a:0:32-n32-S32") :
               Subtarget.isAAPCS_ABI() ?
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:64:128-v64:64:64-a:0:32-n32-S64") :
               std::string("e-p:32:32-f64:64:64-i64:64:64-"
                           "i16:16:32-i8:8:32-i1:8:32-"
                           "v128:64:128-v64:64:64-a:0:32-n32-S32")),
    ELFWriterInfo(*this),
    TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget.hasThumb2()
              ? new ARMFrameLowering(Subtarget)
              : (ARMFrameLowering*)new Thumb1FrameLowering(Subtarget)) {
}

namespace {
/// ARM Code Generator Pass Configuration Options.
class ARMPassConfig : public TargetPassConfig {
public:
  ARMPassConfig(ARMBaseTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  ARMBaseTargetMachine &getARMTargetMachine() const {
    return getTM<ARMBaseTargetMachine>();
  }

  const ARMSubtarget &getARMSubtarget() const {
    return *getARMTargetMachine().getSubtargetImpl();
  }

  virtual bool addPreISel();
  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *ARMBaseTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new ARMPassConfig(this, PM);
}

bool ARMPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOpt::None && EnableGlobalMerge)
    PM->add(createGlobalMergePass(TM->getTargetLowering()));

  return false;
}

bool ARMPassConfig::addInstSelector() {
  PM->add(createARMISelDag(getARMTargetMachine(), getOptLevel()));
  return false;
}

bool ARMPassConfig::addPreRegAlloc() {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (getOptLevel() != CodeGenOpt::None && !getARMSubtarget().isThumb1Only())
    PM->add(createARMLoadStoreOptimizationPass(true));
  if (getOptLevel() != CodeGenOpt::None && getARMSubtarget().isCortexA9())
    PM->add(createMLxExpansionPass());
  return true;
}

bool ARMPassConfig::addPreSched2() {
  // FIXME: temporarily disabling load / store optimization pass for Thumb1.
  if (getOptLevel() != CodeGenOpt::None) {
    if (!getARMSubtarget().isThumb1Only()) {
      PM->add(createARMLoadStoreOptimizationPass());
      printAndVerify("After ARM load / store optimizer");
    }
    if (getARMSubtarget().hasNEON())
      PM->add(createExecutionDependencyFixPass(&ARM::DPRRegClass));
  }

  // Expand some pseudo instructions into multiple instructions to allow
  // proper scheduling.
  PM->add(createARMExpandPseudoPass());

  if (getOptLevel() != CodeGenOpt::None) {
    if (!getARMSubtarget().isThumb1Only())
      addPass(IfConverterID);
  }
  if (getARMSubtarget().isThumb2())
    PM->add(createThumb2ITBlockPass());

  return true;
}

bool ARMPassConfig::addPreEmitPass() {
  if (getARMSubtarget().isThumb2()) {
    if (!getARMSubtarget().prefers32BitThumb())
      PM->add(createThumb2SizeReductionPass());

    // Constant island pass work on unbundled instructions.
    addPass(UnpackMachineBundlesID);
  }

  PM->add(createARMConstantIslandPass());

  return true;
}

bool ARMBaseTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                          JITCodeEmitter &JCE) {
  // Machine code emitter pass for ARM.
  PM.add(createARMJITCodeEmitterPass(*this, JCE));
  return false;
}
