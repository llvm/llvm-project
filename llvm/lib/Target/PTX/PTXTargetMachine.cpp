//===-- PTXTargetMachine.cpp - Define TargetMachine for PTX ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTXTargetMachine.h"
#include "PTX.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
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
  MCStreamer *createPTXAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                   bool isVerboseAsm, bool useLoc,
                                   bool useCFI, bool useDwarfDirectory,
                                   MCInstPrinter *InstPrint,
                                   MCCodeEmitter *CE,
                                   MCAsmBackend *MAB,
                                   bool ShowInst);
}

extern "C" void LLVMInitializePTXTarget() {

  RegisterTargetMachine<PTX32TargetMachine> X(ThePTX32Target);
  RegisterTargetMachine<PTX64TargetMachine> Y(ThePTX64Target);

  TargetRegistry::RegisterAsmStreamer(ThePTX32Target, createPTXAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(ThePTX64Target, createPTXAsmStreamer);
}

namespace {
  const char* DataLayout32 =
    "e-p:32:32-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
  const char* DataLayout64 =
    "e-p:64:64-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
}

// DataLayout and FrameLowering are filled with dummy data
PTXTargetMachine::PTXTargetMachine(const Target &T,
                                   StringRef TT, StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    DataLayout(is64Bit ? DataLayout64 : DataLayout32),
    Subtarget(TT, CPU, FS, is64Bit),
    FrameLowering(Subtarget),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this) {
}

void PTX32TargetMachine::anchor() { }

PTX32TargetMachine::PTX32TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {
}

void PTX64TargetMachine::anchor() { }

PTX64TargetMachine::PTX64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {
}

namespace llvm {
/// PTX Code Generator Pass Configuration Options.
class PTXPassConfig : public TargetPassConfig {
public:
  PTXPassConfig(PTXTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  PTXTargetMachine &getPTXTargetMachine() const {
      return getTM<PTXTargetMachine>();
  }

  bool addInstSelector();
  FunctionPass *createTargetRegisterAllocator(bool);
  void addOptimizedRegAlloc(FunctionPass *RegAllocPass);
  bool addPostRegAlloc();
  void addMachineLateOptimization();
  bool addPreEmitPass();
};
} // namespace

TargetPassConfig *PTXTargetMachine::createPassConfig(PassManagerBase &PM) {
  PTXPassConfig *PassConfig = new PTXPassConfig(this, PM);
  PassConfig->disablePass(PrologEpilogCodeInserterID);
  return PassConfig;
}

bool PTXPassConfig::addInstSelector() {
  PM->add(createPTXISelDag(getPTXTargetMachine(), getOptLevel()));
  return false;
}

FunctionPass *PTXPassConfig::createTargetRegisterAllocator(bool /*Optimized*/) {
  return createPTXRegisterAllocator();
}

// Modify the optimized compilation path to bypass optimized register alloction.
void PTXPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  addFastRegAlloc(RegAllocPass);
}

bool PTXPassConfig::addPostRegAlloc() {
  // PTXMFInfoExtract must after register allocation!
  //PM->add(createPTXMFInfoExtract(getPTXTargetMachine()));
  return false;
}

/// Add passes that optimize machine instructions after register allocation.
void PTXPassConfig::addMachineLateOptimization() {
  if (addPass(BranchFolderPassID) != &NoPassID)
    printAndVerify("After BranchFolding");

  if (addPass(TailDuplicateID) != &NoPassID)
    printAndVerify("After TailDuplicate");
}

bool PTXPassConfig::addPreEmitPass() {
  PM->add(createPTXMFInfoExtract(getPTXTargetMachine(), getOptLevel()));
  PM->add(createPTXFPRoundingModePass(getPTXTargetMachine(), getOptLevel()));
  return true;
}
