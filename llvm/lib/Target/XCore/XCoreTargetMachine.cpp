//===-- XCoreTargetMachine.cpp - Define TargetMachine for XCore -----------===//
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

#include "XCoreTargetAsmInfo.h"
#include "XCoreTargetMachine.h"
#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// XCoreTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int XCoreTargetMachineModule;
int XCoreTargetMachineModule = 0;

namespace {
  // Register the target.
  RegisterTarget<XCoreTargetMachine> X("xcore", "XCore");
}

const TargetAsmInfo *XCoreTargetMachine::createTargetAsmInfo() const {
  return new XCoreTargetAsmInfo(*this);
}

/// XCoreTargetMachine ctor - Create an ILP32 architecture model
///
XCoreTargetMachine::XCoreTargetMachine(const Module &M, const std::string &FS)
  : Subtarget(*this, M, FS),
    DataLayout("e-p:32:32:32-a0:0:32-f32:32:32-f64:32:32-i1:8:32-i8:8:32-"
               "i16:16:32-i32:32:32-i64:32:32"),
    InstrInfo(),
    FrameInfo(*this),
    TLInfo(*this) {
}

unsigned XCoreTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "xcore-")
    return 20;
  
  // Otherwise we don't match.
  return 0;
}

bool XCoreTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  PM.add(createXCoreISelDag(*this));
  return false;
}

bool XCoreTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                            raw_ostream &Out) {
  // Output assembly language.
  PM.add(createXCoreCodePrinterPass(Out, *this));
  return false;
}
