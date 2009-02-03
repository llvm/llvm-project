//===-- IA64TargetMachine.cpp - Define TargetMachine for IA64 -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IA64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "IA64TargetAsmInfo.h"
#include "IA64TargetMachine.h"
#include "IA64.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// IA64TargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int IA64TargetMachineModule;
int IA64TargetMachineModule = 0;

static RegisterTarget<IA64TargetMachine> X("ia64", 
                                           "IA-64 (Itanium) [experimental]");

const TargetAsmInfo *IA64TargetMachine::createTargetAsmInfo() const {
  return new IA64TargetAsmInfo(*this);
}

unsigned IA64TargetMachine::getModuleMatchQuality(const Module &M) {
  // we match [iI][aA]*64
  bool seenIA64=false;
  std::string TT = M.getTargetTriple();

  if (TT.size() >= 4) {
    if( (TT[0]=='i' || TT[0]=='I') &&
        (TT[1]=='a' || TT[1]=='A') ) {
      for(unsigned int i=2; i<(TT.size()-1); i++)
        if(TT[i]=='6' && TT[i+1]=='4')
          seenIA64=true;
    }

    if (seenIA64)
      return 20; // strong match
  }
  // If the target triple is something non-ia64, we don't match.
  if (!TT.empty()) return 0;

#if defined(__ia64__) || defined(__IA64__)
  return 5;
#else
  return 0;
#endif
}

/// IA64TargetMachine ctor - Create an LP64 architecture model
///
IA64TargetMachine::IA64TargetMachine(const Module &M, const std::string &FS)
  : DataLayout("e-f80:128:128"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    TLInfo(*this) { // FIXME? check this stuff
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool IA64TargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  PM.add(createIA64DAGToDAGInstructionSelector(*this));
  return false;
}

bool IA64TargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) {
  // Make sure everything is bundled happily
  PM.add(createIA64BundlingPass(*this));
  return true;
}
bool IA64TargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                           raw_ostream &Out) {
  PM.add(createIA64CodePrinterPass(Out, *this));
  return false;
}

