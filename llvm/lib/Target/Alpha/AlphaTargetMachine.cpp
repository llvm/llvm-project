//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
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

#include "Alpha.h"
#include "AlphaJITInfo.h"
#include "AlphaTargetAsmInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// AlphaTargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int AlphaTargetMachineModule;
int AlphaTargetMachineModule = 0;

// Register the targets
static RegisterTarget<AlphaTargetMachine> X("alpha", "Alpha [experimental]");

const TargetAsmInfo *AlphaTargetMachine::createTargetAsmInfo() const {
  return new AlphaTargetAsmInfo(*this);
}

unsigned AlphaTargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "alpha*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'a' && TT[1] == 'l' && TT[2] == 'p' &&
      TT[3] == 'h' && TT[4] == 'a')
    return 20;
  // If the target triple is something non-alpha, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

unsigned AlphaTargetMachine::getJITMatchQuality() {
#ifdef __alpha
  return 10;
#else
  return 0;
#endif
}

AlphaTargetMachine::AlphaTargetMachine(const Module &M, const std::string &FS)
  : DataLayout("e-f128:128:128"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    JITInfo(*this),
    Subtarget(M, FS),
    TLInfo(*this) {
  setRelocationModel(Reloc::PIC_);
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool AlphaTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  PM.add(createAlphaISelDag(*this));
  return false;
}
bool AlphaTargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) {
  // Must run branch selection immediately preceding the asm printer
  PM.add(createAlphaBranchSelectionPass());
  return false;
}
bool AlphaTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                            raw_ostream &Out) {
  PM.add(createAlphaLLRPPass(*this));
  PM.add(createAlphaCodePrinterPass(Out, *this));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM, bool Fast,
                                        bool DumpAsm, MachineCodeEmitter &MCE) {
  PM.add(createAlphaCodeEmitterPass(*this, MCE));
  if (DumpAsm)
    PM.add(createAlphaCodePrinterPass(errs(), *this));
  return false;
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              bool Fast, bool DumpAsm,
                                              MachineCodeEmitter &MCE) {
  return addCodeEmitter(PM, Fast, DumpAsm, MCE);
}
