//===-- MipsTargetMachine.h - Define TargetMachine for Mips -00--*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Mips specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETMACHINE_H
#define MIPSTARGETMACHINE_H

#include "MipsSubtarget.h"
#include "MipsInstrInfo.h"
#include "MipsISelLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class raw_ostream;
  
  class MipsTargetMachine : public LLVMTargetMachine {
    MipsSubtarget       Subtarget;
    const TargetData    DataLayout; // Calculates type size & alignment
    MipsInstrInfo       InstrInfo;
    TargetFrameInfo     FrameInfo;
    MipsTargetLowering  TLInfo;
  
  protected:
    virtual const TargetAsmInfo *createTargetAsmInfo() const;
  
  public:
    MipsTargetMachine(const Module &M, const std::string &FS, bool isLittle);

    virtual const MipsInstrInfo   *getInstrInfo()     const 
    { return &InstrInfo; }
    virtual const TargetFrameInfo *getFrameInfo()     const 
    { return &FrameInfo; }
    virtual const MipsSubtarget   *getSubtargetImpl() const 
    { return &Subtarget; }
    virtual const TargetData      *getTargetData()    const 
    { return &DataLayout;}

    virtual const MipsRegisterInfo *getRegisterInfo()  const {
      return &InstrInfo.getRegisterInfo();
    }

    virtual MipsTargetLowering   *getTargetLowering() const { 
      return const_cast<MipsTargetLowering*>(&TLInfo); 
    }

    static unsigned getModuleMatchQuality(const Module &M);

    // Pass Pipeline Configuration
    virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
    virtual bool addPreEmitPass(PassManagerBase &PM, bool Fast);
    virtual bool addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                    raw_ostream &Out);
  };

/// MipselTargetMachine - Mipsel target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
public:
  MipselTargetMachine(const Module &M, const std::string &FS);

  static unsigned getModuleMatchQuality(const Module &M);
};

} // End llvm namespace

#endif
