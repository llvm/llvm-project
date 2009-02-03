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

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

/// X86TargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int X86TargetMachineModule;
int X86TargetMachineModule = 0;

// Register the target.
static RegisterTarget<X86_32TargetMachine>
X("x86",    "32-bit X86: Pentium-Pro and above");
static RegisterTarget<X86_64TargetMachine>
Y("x86-64", "64-bit X86: EM64T and AMD64");

// No assembler printer by default
X86TargetMachine::AsmPrinterCtorFn X86TargetMachine::AsmPrinterCtor = 0;

const TargetAsmInfo *X86TargetMachine::createTargetAsmInfo() const {
  if (Subtarget.isFlavorIntel())
    return new X86WinTargetAsmInfo(*this);
  else
    switch (Subtarget.TargetType) {
     case X86Subtarget::isDarwin:
      return new X86DarwinTargetAsmInfo(*this);
     case X86Subtarget::isELF:
      return new X86ELFTargetAsmInfo(*this);
     case X86Subtarget::isMingw:
     case X86Subtarget::isCygwin:
      return new X86COFFTargetAsmInfo(*this);
     case X86Subtarget::isWindows:
      return new X86WinTargetAsmInfo(*this);
     default:
      return new X86GenericTargetAsmInfo(*this);
    }
}

unsigned X86_32TargetMachine::getJITMatchQuality() {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  return 10;
#endif
  return 0;
}

unsigned X86_64TargetMachine::getJITMatchQuality() {
#if defined(__x86_64__) || defined(_M_AMD64)
  return 10;
#endif
  return 0;
}

unsigned X86_32TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "i[3-9]86-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
      TT[4] == '-' && TT[1] - '3' < 6)
    return 20;
  // If the target triple is something non-X86, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

unsigned X86_64TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "x86_64-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 7 && TT[0] == 'x' && TT[1] == '8' && TT[2] == '6' &&
      TT[3] == '_' && TT[4] == '6' && TT[5] == '4' && TT[6] == '-')
    return 20;

  // We strongly match "amd64-*".
  if (TT.size() >= 6 && TT[0] == 'a' && TT[1] == 'm' && TT[2] == 'd' &&
      TT[3] == '6' && TT[4] == '4' && TT[5] == '-')
    return 20;
  
  // If the target triple is something non-X86-64, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

X86_32TargetMachine::X86_32TargetMachine(const Module &M, const std::string &FS) 
  : X86TargetMachine(M, FS, false) {
}


X86_64TargetMachine::X86_64TargetMachine(const Module &M, const std::string &FS)
  : X86TargetMachine(M, FS, true) {
}

/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine(const Module &M, const std::string &FS,
                                   bool is64Bit)
  : Subtarget(M, FS, is64Bit),
    DataLayout(Subtarget.getDataLayout()),
    FrameInfo(TargetFrameInfo::StackGrowsDown,
              Subtarget.getStackAlignment(), Subtarget.is64Bit() ? -8 : -4),
    InstrInfo(*this), JITInfo(*this), TLInfo(*this) {
  DefRelocModel = getRelocationModel();
  // FIXME: Correctly select PIC model for Win64 stuff
  if (getRelocationModel() == Reloc::Default) {
    if (Subtarget.isTargetDarwin() ||
        (Subtarget.isTargetCygMing() && !Subtarget.isTargetWin64()))
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::Static);
  }

  // ELF doesn't have a distinct dynamic-no-PIC model. Dynamic-no-PIC
  // is defined as a model for code which may be used in static or
  // dynamic executables but not necessarily a shared library. On ELF
  // implement this by using the Static model.
  if (Subtarget.isTargetELF() &&
      getRelocationModel() == Reloc::DynamicNoPIC)
    setRelocationModel(Reloc::Static);

  if (Subtarget.is64Bit()) {
    // No DynamicNoPIC support under X86-64.
    if (getRelocationModel() == Reloc::DynamicNoPIC)
      setRelocationModel(Reloc::PIC_);
    // Default X86-64 code model is small.
    if (getCodeModel() == CodeModel::Default)
      setCodeModel(CodeModel::Small);
  }

  if (Subtarget.isTargetCygMing())
    Subtarget.setPICStyle(PICStyles::WinPIC);
  else if (Subtarget.isTargetDarwin()) {
    if (Subtarget.is64Bit())
      Subtarget.setPICStyle(PICStyles::RIPRel);
    else
      Subtarget.setPICStyle(PICStyles::Stub);
  } else if (Subtarget.isTargetELF()) {
    if (Subtarget.is64Bit())
      Subtarget.setPICStyle(PICStyles::RIPRel);
    else
      Subtarget.setPICStyle(PICStyles::GOT);
  }
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool X86TargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  // Install an instruction selector.
  PM.add(createX86ISelDag(*this, Fast));

  // If we're using Fast-ISel, clean up the mess.
  if (EnableFastISel)
    PM.add(createDeadMachineInstructionElimPass());

  // Install a pass to insert x87 FP_REG_KILL instructions, as needed.
  PM.add(createX87FPRegKillInserterPass());

  return false;
}

bool X86TargetMachine::addPreRegAlloc(PassManagerBase &PM, bool Fast) {
  // Calculate and set max stack object alignment early, so we can decide
  // whether we will need stack realignment (and thus FP).
  PM.add(createX86MaxStackAlignmentCalculatorPass());
  return false;  // -print-machineinstr shouldn't print after this.
}

bool X86TargetMachine::addPostRegAlloc(PassManagerBase &PM, bool Fast) {
  PM.add(createX86FloatingPointStackifierPass());
  return true;  // -print-machineinstr should print after this.
}

bool X86TargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                          raw_ostream &Out) {
  assert(AsmPrinterCtor && "AsmPrinter was not linked in");
  if (AsmPrinterCtor)
    PM.add(AsmPrinterCtor(Out, *this));
  return false;
}

bool X86TargetMachine::addCodeEmitter(PassManagerBase &PM, bool Fast,
                                      bool DumpAsm, MachineCodeEmitter &MCE) {
  // FIXME: Move this to TargetJITInfo!
  // On Darwin, do not override 64-bit setting made in X86TargetMachine().
  if (DefRelocModel == Reloc::Default && 
        (!Subtarget.isTargetDarwin() || !Subtarget.is64Bit()))
    setRelocationModel(Reloc::Static);
  
  // 64-bit JIT places everything in the same buffer except external functions.
  // On Darwin, use small code model but hack the call instruction for 
  // externals.  Elsewhere, do not assume globals are in the lower 4G.
  if (Subtarget.is64Bit()) {
    if (Subtarget.isTargetDarwin())
      setCodeModel(CodeModel::Small);
    else
      setCodeModel(CodeModel::Large);
  }

  PM.add(createX86CodeEmitterPass(*this, MCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(errs(), *this));
  }

  return false;
}

bool X86TargetMachine::addSimpleCodeEmitter(PassManagerBase &PM, bool Fast,
                                        bool DumpAsm, MachineCodeEmitter &MCE) {
  PM.add(createX86CodeEmitterPass(*this, MCE));
  if (DumpAsm) {
    assert(AsmPrinterCtor && "AsmPrinter was not linked in");
    if (AsmPrinterCtor)
      PM.add(AsmPrinterCtor(errs(), *this));
  }

  return false;
}

// symbolicAddressesAreRIPRel - Return true if symbolic addresses are
// RIP-relative on this machine, taking into consideration the relocation
// model and subtarget. RIP-relative addresses cannot have a separate
// base or index register.
bool X86TargetMachine::symbolicAddressesAreRIPRel() const {
  return getRelocationModel() != Reloc::Static &&
         Subtarget.isPICStyleRIPRel();
}
