//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86ISelLowering.h"
#include "X86FrameLowering.h"
#include "X86JITInfo.h"
#include "X86SelectionDAGInfo.h"
#include "X86Subtarget.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetTransformImpl.h"

namespace llvm {

class StringRef;

class X86TargetMachine : public LLVMTargetMachine {
  X86Subtarget       Subtarget;
  X86FrameLowering   FrameLowering;
  InstrItineraryData InstrItins;

public:
  X86TargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL,
                   bool is64Bit);

  virtual const X86InstrInfo     *getInstrInfo() const {
    llvm_unreachable("getInstrInfo not implemented");
  }
  virtual const TargetFrameLowering  *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    llvm_unreachable("getJITInfo not implemented");
  }
  virtual const X86Subtarget     *getSubtargetImpl() const{ return &Subtarget; }
  virtual const X86TargetLowering *getTargetLowering() const {
    llvm_unreachable("getTargetLowering not implemented");
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const {
    llvm_unreachable("getSelectionDAGInfo not implemented");
  }
  virtual const X86RegisterInfo  *getRegisterInfo() const {
    return &getInstrInfo()->getRegisterInfo();
  }
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }

  // Set up the pass pipeline.
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  virtual bool addCodeEmitter(PassManagerBase &PM,
                              JITCodeEmitter &JCE);
};

/// X86_32TargetMachine - X86 32-bit target machine.
///
class X86_32TargetMachine : public X86TargetMachine {
  virtual void anchor();
  const DataLayout  DL; // Calculates type size & alignment
  X86InstrInfo      InstrInfo;
  X86SelectionDAGInfo TSInfo;
  X86TargetLowering TLInfo;
  X86JITInfo        JITInfo;
  ScalarTargetTransformImpl STTI;
  X86VectorTargetTransformInfo VTTI;
public:
  X86_32TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
  virtual const DataLayout *getDataLayout() const { return &DL; }
  virtual const X86TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const X86InstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    return &JITInfo;
  }
  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }
};

/// X86_64TargetMachine - X86 64-bit target machine.
///
class X86_64TargetMachine : public X86TargetMachine {
  virtual void anchor();
  const DataLayout  DL; // Calculates type size & alignment
  X86InstrInfo      InstrInfo;
  X86SelectionDAGInfo TSInfo;
  X86TargetLowering TLInfo;
  X86JITInfo        JITInfo;
  ScalarTargetTransformImpl STTI;
  X86VectorTargetTransformInfo VTTI;
public:
  X86_64TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
  virtual const DataLayout *getDataLayout() const { return &DL; }
  virtual const X86TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const X86InstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    return &JITInfo;
  }
  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }
};

} // End llvm namespace

#endif
