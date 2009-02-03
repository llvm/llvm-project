//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetAsmInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// PowerPCTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int PowerPCTargetMachineModule;
int PowerPCTargetMachineModule = 0;

// Register the targets
static RegisterTarget<PPC32TargetMachine>
X("ppc32", "PowerPC 32");
static RegisterTarget<PPC64TargetMachine>
Y("ppc64", "PowerPC 64");

// No assembler printer by default
PPCTargetMachine::AsmPrinterCtorFn PPCTargetMachine::AsmPrinterCtor = 0;

const TargetAsmInfo *PPCTargetMachine::createTargetAsmInfo() const {
  if (Subtarget.isDarwin())
    return new PPCDarwinTargetAsmInfo(*this);
  else
    return new PPCLinuxTargetAsmInfo(*this);
}

unsigned PPC32TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER) || defined(__PPC__)
  if (sizeof(void*) == 4)
    return 10;
#endif
  return 0;
}
unsigned PPC64TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER) || defined(__PPC__)
  if (sizeof(void*) == 8)
    return 10;
#endif
  return 0;
}

unsigned PPC32TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "powerpc-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "powerpc-")
    return 20;
  
  // If the target triple is something non-powerpc, we don't match.
  if (!TT.empty()) return 0;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return getJITMatchQuality()/2;
}

unsigned PPC64TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "powerpc64-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 10 && std::string(TT.begin(), TT.begin()+10) == "powerpc64-")
    return 20;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return getJITMatchQuality()/2;
}


PPCTargetMachine::PPCTargetMachine(const Module &M, const std::string &FS,
                                   bool is64Bit)
  : Subtarget(*this, M, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameInfo(*this, is64Bit), JITInfo(*this, is64Bit), TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()), MachOWriterInfo(*this) {

  if (getRelocationModel() == Reloc::Default) {
    if (Subtarget.isDarwin())
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::Static);
  }
}

/// Override this for PowerPC.  Tail merging happily breaks up instruction issue
/// groups, which typically degrades performance.
bool PPCTargetMachine::getEnableTailMergeDefault() const { return false; }

PPC32TargetMachine::PPC32TargetMachine(const Module &M, const std::string &FS) 
  : PPCTargetMachine(M, FS, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Module &M, const std::string &FS)
  : PPCTargetMachine(M, FS, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PPCTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));
  return false;
}

bool PPCTargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) {
  
  // Must run branch selection immediately preceding the asm printer.
  PM.add(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                          raw_ostream &Out) {
  assert(AsmPrinterCtor && "AsmPrinter was not linked in");
  if (AsmPrinterCtor)
    PM.add(AsmPrinterCtor(Out, *this));

  return false;
}

bool PPCTargetMachine::addCodeEmitter(PassManagerBase &PM, bool Fast,
                                      bool DumpAsm, MachineCodeEmitter &MCE) {
  // The JIT should use the static relocation model in ppc32 mode, PIC in ppc64.
  // FIXME: This should be moved to TargetJITInfo!!
  if (Subtarget.isPPC64()) {
    // We use PIC codegen in ppc64 mode, because otherwise we'd have to use many
    // instructions to materialize arbitrary global variable + function +
    // constant pool addresses.
    setRelocationModel(Reloc::PIC_);
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    DisableJumpTables = true;      
  } else {
    setRelocationModel(Reloc::Static);
  }
  
  // Inform the subtarget that we are in JIT mode.  FIXME: does this break macho
  // writing?
  Subtarget.SetJITMode();
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(errs(), *this));
  }

  return false;
}

bool PPCTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM, bool Fast,
                                            bool DumpAsm, MachineCodeEmitter &MCE) {
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(errs(), *this));
  }

  return false;
}
