//===-- MipsTargetMachine.h - Define TargetMachine for Mips -----*- C++ -*-===//
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

#include "MipsFrameLowering.h"
#include "MipsInstrInfo.h"
#include "MipsISelLowering.h"
#include "MipsJITInfo.h"
#include "MipsSelectionDAGInfo.h"
#include "MipsSubtarget.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetTransformImpl.h"

namespace llvm {
class formatted_raw_ostream;
class MipsRegisterInfo;

class MipsTargetMachine : public LLVMTargetMachine {
  MipsSubtarget       Subtarget;
  const DataLayout    DL; // Calculates type size & alignment
  const MipsInstrInfo *InstrInfo;
  const MipsFrameLowering *FrameLowering;
  MipsTargetLowering  TLInfo;
  MipsSelectionDAGInfo TSInfo;
  MipsJITInfo JITInfo;
  ScalarTargetTransformImpl STTI;
  VectorTargetTransformImpl VTTI;

public:
  MipsTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL,
                    bool isLittle);

  virtual ~MipsTargetMachine() { delete InstrInfo; }

  virtual const MipsInstrInfo *getInstrInfo() const
  { return InstrInfo; }
  virtual const TargetFrameLowering *getFrameLowering() const
  { return FrameLowering; }
  virtual const MipsSubtarget *getSubtargetImpl() const
  { return &Subtarget; }
  virtual const DataLayout *getDataLayout()    const
  { return &DL;}
  virtual MipsJITInfo *getJITInfo()
  { return &JITInfo; }

  virtual const MipsRegisterInfo *getRegisterInfo()  const {
    return &InstrInfo->getRegisterInfo();
  }

  virtual const MipsTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const MipsSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
  virtual bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &JCE);
};

/// MipsebTargetMachine - Mips32/64 big endian target machine.
///
class MipsebTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipsebTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

/// MipselTargetMachine - Mips32/64 little endian target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipselTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

} // End llvm namespace

#endif
