//===-- XCoreTargetMachine.h - Define TargetMachine for XCore ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the XCore specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETMACHINE_H
#define XCORETARGETMACHINE_H

#include "XCoreFrameLowering.h"
#include "XCoreSubtarget.h"
#include "XCoreInstrInfo.h"
#include "XCoreISelLowering.h"
#include "XCoreSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetTransformImpl.h"
#include "llvm/DataLayout.h"

namespace llvm {

class XCoreTargetMachine : public LLVMTargetMachine {
  XCoreSubtarget Subtarget;
  const DataLayout DL;       // Calculates type size & alignment
  XCoreInstrInfo InstrInfo;
  XCoreFrameLowering FrameLowering;
  XCoreTargetLowering TLInfo;
  XCoreSelectionDAGInfo TSInfo;
  ScalarTargetTransformImpl STTI;
  VectorTargetTransformImpl VTTI;
public:
  XCoreTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);

  virtual const XCoreInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const XCoreFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const XCoreSubtarget *getSubtargetImpl() const { return &Subtarget; }
  virtual const XCoreTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const XCoreSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }
  virtual const DataLayout       *getDataLayout() const { return &DL; }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

} // end namespace llvm

#endif
