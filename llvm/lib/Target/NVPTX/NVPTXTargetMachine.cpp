//===-- NVPTXTargetMachine.cpp - Define TargetMachine for NVPTX -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetMachine.h"
#include "MCTargetDesc/NVPTXMCAsmInfo.h"
#include "NVPTX.h"
#include "NVPTXAllocaHoisting.h"
#include "NVPTXLowerAggrCopies.h"
#include "NVPTXSplitBBatBar.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

namespace llvm {
void initializeNVVMReflectPass(PassRegistry&);
void initializeGenericToNVVMPass(PassRegistry&);
}

extern "C" void LLVMInitializeNVPTXTarget() {
  // Register the target.
  RegisterTargetMachine<NVPTXTargetMachine32> X(TheNVPTXTarget32);
  RegisterTargetMachine<NVPTXTargetMachine64> Y(TheNVPTXTarget64);

  RegisterMCAsmInfo<NVPTXMCAsmInfo> A(TheNVPTXTarget32);
  RegisterMCAsmInfo<NVPTXMCAsmInfo> B(TheNVPTXTarget64);

  // FIXME: This pass is really intended to be invoked during IR optimization,
  // but it's very NVPTX-specific.
  initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
  initializeGenericToNVVMPass(*PassRegistry::getPassRegistry());
}

NVPTXTargetMachine::NVPTXTargetMachine(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL, bool is64bit)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      Subtarget(TT, CPU, FS, is64bit), DL(Subtarget.getDataLayout()),
      InstrInfo(*this), TLInfo(*this), TSInfo(*this),
      FrameLowering(
          *this, is64bit) /*FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0)*/ {}

void NVPTXTargetMachine32::anchor() {}

NVPTXTargetMachine32::NVPTXTargetMachine32(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void NVPTXTargetMachine64::anchor() {}

NVPTXTargetMachine64::NVPTXTargetMachine64(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace llvm {
class NVPTXPassConfig : public TargetPassConfig {
public:
  NVPTXPassConfig(NVPTXTargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  NVPTXTargetMachine &getNVPTXTargetMachine() const {
    return getTM<NVPTXTargetMachine>();
  }

  virtual void addIRPasses();
  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
};
}

TargetPassConfig *NVPTXTargetMachine::createPassConfig(PassManagerBase &PM) {
  NVPTXPassConfig *PassConfig = new NVPTXPassConfig(this, PM);
  return PassConfig;
}

void NVPTXPassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();
  addPass(createGenericToNVVMPass());
}

bool NVPTXPassConfig::addInstSelector() {
  addPass(createLowerAggrCopies());
  addPass(createSplitBBatBarPass());
  addPass(createAllocaHoisting());
  addPass(createNVPTXISelDag(getNVPTXTargetMachine(), getOptLevel()));
  return false;
}

bool NVPTXPassConfig::addPreRegAlloc() { return false; }
