//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameInfo.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "PPCMachOWriterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const TargetData    DataLayout;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameInfo        FrameInfo;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  InstrItineraryData  InstrItins;
  PPCMachOWriterInfo  MachOWriterInfo;

protected:
  virtual const TargetAsmInfo *createTargetAsmInfo() const;

  // To avoid having target depend on the asmprinter stuff libraries, asmprinter
  // set this functions to ctor pointer at startup time if they are linked in.
  typedef FunctionPass *(*AsmPrinterCtorFn)(raw_ostream &o,
                                            PPCTargetMachine &tm);
  static AsmPrinterCtorFn AsmPrinterCtor;

public:
  PPCTargetMachine(const Module &M, const std::string &FS, bool is64Bit);

  virtual const PPCInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const PPCFrameInfo     *getFrameInfo() const { return &FrameInfo; }
  virtual       PPCJITInfo       *getJITInfo()         { return &JITInfo; }
  virtual       PPCTargetLowering *getTargetLowering() const { 
   return const_cast<PPCTargetLowering*>(&TLInfo); 
  }
  virtual const PPCRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  
  virtual const TargetData    *getTargetData() const    { return &DataLayout; }
  virtual const PPCSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItins;
  }
  virtual const PPCMachOWriterInfo *getMachOWriterInfo() const {
    return &MachOWriterInfo;
  }

  static void registerAsmPrinter(AsmPrinterCtorFn F) {
    AsmPrinterCtor = F;
  }

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
  virtual bool addPreEmitPass(PassManagerBase &PM, bool Fast);
  virtual bool addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                  raw_ostream &Out);
  virtual bool addCodeEmitter(PassManagerBase &PM, bool Fast,
                              bool DumpAsm, MachineCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM, bool Fast,
                                    bool DumpAsm, MachineCodeEmitter &MCE);
  virtual bool getEnableTailMergeDefault() const;
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
public:
  PPC32TargetMachine(const Module &M, const std::string &FS);
  
  static unsigned getJITMatchQuality();
  static unsigned getModuleMatchQuality(const Module &M);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
public:
  PPC64TargetMachine(const Module &M, const std::string &FS);
  
  static unsigned getJITMatchQuality();
  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif
