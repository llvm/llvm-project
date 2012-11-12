//===-- MSP430TargetMachine.h - Define TargetMachine for MSP430 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MSP430 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TARGET_MSP430_TARGETMACHINE_H
#define LLVM_TARGET_MSP430_TARGETMACHINE_H

#include "MSP430InstrInfo.h"
#include "MSP430ISelLowering.h"
#include "MSP430FrameLowering.h"
#include "MSP430SelectionDAGInfo.h"
#include "MSP430RegisterInfo.h"
#include "MSP430Subtarget.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetTransformImpl.h"

namespace llvm {

/// MSP430TargetMachine
///
class MSP430TargetMachine : public LLVMTargetMachine {
  MSP430Subtarget        Subtarget;
  const DataLayout       DL;       // Calculates type size & alignment
  MSP430InstrInfo        InstrInfo;
  MSP430TargetLowering   TLInfo;
  MSP430SelectionDAGInfo TSInfo;
  MSP430FrameLowering    FrameLowering;
  ScalarTargetTransformImpl STTI;
  VectorTargetTransformImpl VTTI;

public:
  MSP430TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);

  virtual const TargetFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const MSP430InstrInfo *getInstrInfo() const  { return &InstrInfo; }
  virtual const DataLayout *getDataLayout() const     { return &DL;}
  virtual const MSP430Subtarget *getSubtargetImpl() const { return &Subtarget; }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const MSP430TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const MSP430SelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
}; // MSP430TargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_MSP430_TARGETMACHINE_H
