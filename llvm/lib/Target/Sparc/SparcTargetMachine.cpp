//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
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

#include "SparcTargetAsmInfo.h"
#include "SparcTargetMachine.h"
#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// SparcTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int SparcTargetMachineModule;
int SparcTargetMachineModule = 0;

// Register the target.
static RegisterTarget<SparcTargetMachine> X("sparc", "SPARC");

const TargetAsmInfo *SparcTargetMachine::createTargetAsmInfo() const {
  // FIXME: Handle Solaris subtarget someday :)
  return new SparcELFTargetAsmInfo(*this);
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Module &M, const std::string &FS)
  : DataLayout("E-p:32:32-f128:128:128"),
    Subtarget(M, FS), TLInfo(*this), InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

unsigned SparcTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;
  
  // If the target triple is something non-sparc, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
#ifdef __sparc__
    return 20;   // BE/32 ==> Prefer sparc on sparc
#else
    return 5;    // BE/32 ==> Prefer ppc elsewhere
#endif
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

#if defined(__sparc__)
  return 10;
#else
  return 0;
#endif
}

bool SparcTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  PM.add(createSparcISelDag(*this));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcTargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) {
  PM.add(createSparcFPMoverPass(*this));
  PM.add(createSparcDelaySlotFillerPass(*this));
  return true;
}

bool SparcTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                            raw_ostream &Out) {
  // Output assembly language.
  PM.add(createSparcCodePrinterPass(Out, *this));
  return false;
}
