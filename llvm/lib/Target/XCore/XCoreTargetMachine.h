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

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "XCoreFrameInfo.h"
#include "XCoreSubtarget.h"
#include "XCoreInstrInfo.h"
#include "XCoreISelLowering.h"

namespace llvm {

class Module;

class XCoreTargetMachine : public LLVMTargetMachine {
  XCoreSubtarget Subtarget;
  const TargetData DataLayout;       // Calculates type size & alignment
  XCoreInstrInfo InstrInfo;
  XCoreFrameInfo FrameInfo;
  XCoreTargetLowering TLInfo;

protected:
  virtual const TargetAsmInfo *createTargetAsmInfo() const;

public:
  XCoreTargetMachine(const Module &M, const std::string &FS);

  virtual const XCoreInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const XCoreFrameInfo *getFrameInfo() const { return &FrameInfo; }
  virtual const XCoreSubtarget *getSubtargetImpl() const { return &Subtarget; }
  virtual       XCoreTargetLowering *getTargetLowering() const {
    return const_cast<XCoreTargetLowering*>(&TLInfo);
  }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  static unsigned getModuleMatchQuality(const Module &M);

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
  virtual bool addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                  raw_ostream &Out);
};

} // end namespace llvm

#endif
