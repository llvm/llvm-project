//===-- ARMTargetMachine.h - Define TargetMachine for ARM -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef ARMTARGETMACHINE_H
#define ARMTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "ARMInstrInfo.h"
#include "ARMFrameInfo.h"
#include "ARMJITInfo.h"
#include "ARMSubtarget.h"
#include "ARMISelLowering.h"

namespace llvm {

class Module;

class ARMTargetMachine : public LLVMTargetMachine {
  ARMSubtarget      Subtarget;
  const TargetData  DataLayout;       // Calculates type size & alignment
  ARMInstrInfo      InstrInfo;
  ARMFrameInfo      FrameInfo;
  ARMJITInfo        JITInfo;
  ARMTargetLowering TLInfo;
  Reloc::Model      DefRelocModel;    // Reloc model before it's overridden.

protected:
  // To avoid having target depend on the asmprinter stuff libraries, asmprinter
  // set this functions to ctor pointer at startup time if they are linked in.
  typedef FunctionPass *(*AsmPrinterCtorFn)(raw_ostream &o,
                                            ARMTargetMachine &tm);
  static AsmPrinterCtorFn AsmPrinterCtor;

public:
  ARMTargetMachine(const Module &M, const std::string &FS, bool isThumb = false);

  virtual const ARMInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const ARMFrameInfo     *getFrameInfo() const { return &FrameInfo; }
  virtual       ARMJITInfo       *getJITInfo()         { return &JITInfo; }
  virtual const ARMRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  virtual const ARMSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual       ARMTargetLowering *getTargetLowering() const {
    return const_cast<ARMTargetLowering*>(&TLInfo);
  }

  static void registerAsmPrinter(AsmPrinterCtorFn F) {
    AsmPrinterCtor = F;
  }

  static unsigned getModuleMatchQuality(const Module &M);
  static unsigned getJITMatchQuality();

  virtual const TargetAsmInfo *createTargetAsmInfo() const;

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
  virtual bool addPreEmitPass(PassManagerBase &PM, bool Fast);
  virtual bool addAssemblyEmitter(PassManagerBase &PM, bool Fast,
                                  raw_ostream &Out);
  virtual bool addCodeEmitter(PassManagerBase &PM, bool Fast,
                              bool DumpAsm, MachineCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM, bool Fast,
                                    bool DumpAsm, MachineCodeEmitter &MCE);
};

/// ThumbTargetMachine - Thumb target machine.
///
class ThumbTargetMachine : public ARMTargetMachine {
public:
  ThumbTargetMachine(const Module &M, const std::string &FS);

  static unsigned getJITMatchQuality();
  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif
