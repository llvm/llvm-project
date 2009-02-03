//===-- PIC16TargetMachine.cpp - Define TargetMachine for PIC16 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PIC16 target.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachineRegistry.h"

using namespace llvm;

/// PIC16TargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int PIC16TargetMachineModule;
int PIC16TargetMachineModule = 0;


// Register the targets
static RegisterTarget<PIC16TargetMachine> 
X("pic16", "PIC16 14-bit [experimental].");
static RegisterTarget<CooperTargetMachine> 
Y("cooper", "PIC16 Cooper [experimental].");

// PIC16TargetMachine - Traditional PIC16 Machine.
PIC16TargetMachine::PIC16TargetMachine(const Module &M, const std::string &FS,
                                       bool Cooper)
: Subtarget(M, FS, Cooper),
  DataLayout("e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"), 
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0) { }

// CooperTargetMachine - Uses the same PIC16TargetMachine, but makes IsCooper
// as true.
CooperTargetMachine::CooperTargetMachine(const Module &M, const std::string &FS)
  : PIC16TargetMachine(M, FS, true) {}


const TargetAsmInfo *PIC16TargetMachine::createTargetAsmInfo() const {
  return new PIC16TargetAsmInfo(*this);
}

bool PIC16TargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  // Install an instruction selector.
  PM.add(createPIC16ISelDag(*this));
  return false;
}

bool PIC16TargetMachine::
addAssemblyEmitter(PassManagerBase &PM, bool Fast, raw_ostream &Out) {
  // Output assembly language.
  PM.add(createPIC16CodePrinterPass(Out, *this));
  return false;
}


