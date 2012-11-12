//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeX86Target() {
  // Register the target.
  RegisterTargetMachine<X86_32TargetMachine> X(TheX86_32Target);
  RegisterTargetMachine<X86_64TargetMachine> Y(TheX86_64Target);
}

void X86_32TargetMachine::anchor() { }

X86_32TargetMachine::X86_32TargetMachine(const Target &T, StringRef TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         Reloc::Model RM, CodeModel::Model CM,
                                         CodeGenOpt::Level OL)
  : X86TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false),
    DL(getSubtargetImpl()->isTargetDarwin() ?
               "e-p:32:32-f64:32:64-i64:32:64-f80:128:128-f128:128:128-"
               "n8:16:32-S128" :
               (getSubtargetImpl()->isTargetCygMing() ||
                getSubtargetImpl()->isTargetWindows()) ?
               "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-f128:128:128-"
               "n8:16:32-S32" :
               "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-f128:128:128-"
               "n8:16:32-S128"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this),
    STTI(&TLInfo), VTTI(&TLInfo) {
}

void X86_64TargetMachine::anchor() { }

X86_64TargetMachine::X86_64TargetMachine(const Target &T, StringRef TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         Reloc::Model RM, CodeModel::Model CM,
                                         CodeGenOpt::Level OL)
  : X86TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true),
    DL("e-p:64:64-s:64-f64:64:64-i64:64:64-f80:128:128-f128:128:128-"
               "n8:16:32:64-S128"),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this),
    JITInfo(*this),
    STTI(&TLInfo), VTTI(&TLInfo){
}

/// X86TargetMachine ctor - Create an X86 target.
///
X86TargetMachine::X86TargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, Options.StackAlignmentOverride, is64Bit),
    FrameLowering(*this, Subtarget),
    InstrItins(Subtarget.getInstrItineraryData()){
  // Determine the PICStyle based on the target selected.
  if (getRelocationModel() == Reloc::Static) {
    // Unless we're in PIC or DynamicNoPIC mode, set the PIC style to None.
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.is64Bit()) {
    // PIC in 64 bit mode is always rip-rel.
    Subtarget.setPICStyle(PICStyles::RIPRel);
  } else if (Subtarget.isTargetCygMing()) {
    Subtarget.setPICStyle(PICStyles::None);
  } else if (Subtarget.isTargetDarwin()) {
    if (getRelocationModel() == Reloc::PIC_)
      Subtarget.setPICStyle(PICStyles::StubPIC);
    else {
      assert(getRelocationModel() == Reloc::DynamicNoPIC);
      Subtarget.setPICStyle(PICStyles::StubDynamicNoPIC);
    }
  } else if (Subtarget.isTargetELF()) {
    Subtarget.setPICStyle(PICStyles::GOT);
  }

  // default to hard float ABI
  if (Options.FloatABIType == FloatABI::Default)
    this->Options.FloatABIType = FloatABI::Hard;
}

//===----------------------------------------------------------------------===//
// Command line options for x86
//===----------------------------------------------------------------------===//
static cl::opt<bool>
UseVZeroUpper("x86-use-vzeroupper",
  cl::desc("Minimize AVX to SSE transition penalty"),
  cl::init(true));

// Temporary option to control early if-conversion for x86 while adding machine
// models.
static cl::opt<bool>
X86EarlyIfConv("x86-early-ifcvt",
	       cl::desc("Enable early if-conversion on X86"));

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

namespace {
/// X86 Code Generator Pass Configuration Options.
class X86PassConfig : public TargetPassConfig {
public:
  X86PassConfig(X86TargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  X86TargetMachine &getX86TargetMachine() const {
    return getTM<X86TargetMachine>();
  }

  const X86Subtarget &getX86Subtarget() const {
    return *getX86TargetMachine().getSubtargetImpl();
  }

  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *X86TargetMachine::createPassConfig(PassManagerBase &PM) {
  X86PassConfig *PC = new X86PassConfig(this, PM);

  if (X86EarlyIfConv && Subtarget.hasCMov())
    PC->enablePass(&EarlyIfConverterID);

  return PC;
}

bool X86PassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createX86ISelDag(getX86TargetMachine(), getOptLevel()));

  // For ELF, cleanup any local-dynamic TLS accesses.
  if (getX86Subtarget().isTargetELF() && getOptLevel() != CodeGenOpt::None)
    addPass(createCleanupLocalDynamicTLSPass());

  // For 32-bit, prepend instructions to set the "global base reg" for PIC.
  if (!getX86Subtarget().is64Bit())
    addPass(createGlobalBaseRegPass());

  return false;
}

bool X86PassConfig::addPreRegAlloc() {
  addPass(createX86MaxStackAlignmentHeuristicPass());
  return false;  // -print-machineinstr shouldn't print after this.
}

bool X86PassConfig::addPostRegAlloc() {
  addPass(createX86FloatingPointStackifierPass());
  return true;  // -print-machineinstr should print after this.
}

bool X86PassConfig::addPreEmitPass() {
  bool ShouldPrint = false;
  if (getOptLevel() != CodeGenOpt::None && getX86Subtarget().hasSSE2()) {
    addPass(createExecutionDependencyFixPass(&X86::VR128RegClass));
    ShouldPrint = true;
  }

  if (getX86Subtarget().hasAVX() && UseVZeroUpper) {
    addPass(createX86IssueVZeroUpperPass());
    ShouldPrint = true;
  }

  return ShouldPrint;
}

bool X86TargetMachine::addCodeEmitter(PassManagerBase &PM,
                                      JITCodeEmitter &JCE) {
  PM.add(createX86JITCodeEmitterPass(*this, JCE));

  return false;
}
