//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// MipsTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int MipsTargetMachineModule;
int MipsTargetMachineModule = 0;

// Register the target.
static RegisterTarget<MipsTargetMachine>    X("mips", "Mips");
static RegisterTarget<MipselTargetMachine>  Y("mipsel", "Mipsel");

const TargetAsmInfo *MipsTargetMachine::
createTargetAsmInfo() const 
{
  return new MipsTargetAsmInfo(*this);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables 
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
MipsTargetMachine::
MipsTargetMachine(const Module &M, const std::string &FS, bool isLittle=false):
  Subtarget(*this, M, FS, isLittle), 
  DataLayout(isLittle ? std::string("e-p:32:32:32-i8:8:32-i16:16:32") :
                        std::string("E-p:32:32:32-i8:8:32-i16:16:32")), 
  InstrInfo(*this), 
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0),
  TLInfo(*this) 
{
  // Abicall enables PIC by default
  if (Subtarget.hasABICall())
    setRelocationModel(Reloc::PIC_);  

  // TODO: create an option to enable long calls, like -mlong-calls, 
  // that would be our CodeModel::Large. It must not work with Abicall.
  if (getCodeModel() == CodeModel::Default)
    setCodeModel(CodeModel::Small);
}

MipselTargetMachine::
MipselTargetMachine(const Module &M, const std::string &FS) :
  MipsTargetMachine(M, FS, true) {}

// return 0 and must specify -march to gen MIPS code.
unsigned MipsTargetMachine::
getModuleMatchQuality(const Module &M) 
{
  // We strongly match "mips*-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && std::string(TT.begin(), TT.begin()+5) == "mips-")
    return 20;
  
  if (TT.size() >= 13 && std::string(TT.begin(), 
      TT.begin()+13) == "mipsallegrex-")
    return 20;

  return 0;
}

// return 0 and must specify -march to gen MIPSEL code.
unsigned MipselTargetMachine::
getModuleMatchQuality(const Module &M) 
{
  // We strongly match "mips*el-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 7 && std::string(TT.begin(), TT.begin()+7) == "mipsel-")
    return 20;

  if (TT.size() >= 15 && std::string(TT.begin(), 
      TT.begin()+15) == "mipsallegrexel-")
    return 20;

  if (TT.size() == 3 && std::string(TT.begin(), TT.begin()+3) == "psp")
    return 20;
  
  return 0;
}

// Install an instruction selector pass using 
// the ISelDag to gen Mips code.
bool MipsTargetMachine::
addInstSelector(PassManagerBase &PM, bool Fast) 
{
  PM.add(createMipsISelDag(*this));
  return false;
}

// Implemented by targets that want to run passes immediately before 
// machine code is emitted. return true if -print-machineinstrs should 
// print out the code after the passes.
bool MipsTargetMachine::
addPreEmitPass(PassManagerBase &PM, bool Fast) 
{
  PM.add(createMipsDelaySlotFillerPass(*this));
  return true;
}

// Implements the AssemblyEmitter for the target. Must return
// true if AssemblyEmitter is supported
bool MipsTargetMachine::
addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                   raw_ostream &Out) 
{
  // Output assembly language.
  PM.add(createMipsCodePrinterPass(Out, *this));
  return false;
}
