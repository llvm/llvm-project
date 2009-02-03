//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPURegisterNames.h"
#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"

using namespace llvm;

/// CellSPUTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int CellSPUTargetMachineModule;
int CellSPUTargetMachineModule = 0;

namespace {
  // Register the targets
  RegisterTarget<SPUTargetMachine>
  CELLSPU("cellspu", "STI CBEA Cell SPU [experimental]");
}

const std::pair<unsigned, int> *
SPUFrameInfo::getCalleeSaveSpillSlots(unsigned &NumEntries) const {
  NumEntries = 1;
  return &LR[0];
}

const TargetAsmInfo *
SPUTargetMachine::createTargetAsmInfo() const
{
  return new SPULinuxTargetAsmInfo(*this);
}

unsigned
SPUTargetMachine::getModuleMatchQuality(const Module &M)
{
  // We strongly match "spu-*" or "cellspu-*".
  std::string TT = M.getTargetTriple();
  if ((TT.size() == 3 && std::string(TT.begin(), TT.begin()+3) == "spu")
      || (TT.size() == 7 && std::string(TT.begin(), TT.begin()+7) == "cellspu")
      || (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "spu-")
      || (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "cellspu-"))
    return 20;
  
  return 0;                     // No match at all...
}

SPUTargetMachine::SPUTargetMachine(const Module &M, const std::string &FS)
  : Subtarget(*this, M, FS),
    DataLayout(Subtarget.getTargetDataString()),
    InstrInfo(*this),
    FrameInfo(*this),
    TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData())
{
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  setRelocationModel(Reloc::Static);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool
SPUTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast)
{
  // Install an instruction selector.
  PM.add(createSPUISelDag(*this));
  return false;
}

bool SPUTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                          raw_ostream &Out) {
  PM.add(createSPUAsmPrinterPass(Out, *this));
  return false;
}
