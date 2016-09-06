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

#ifndef LLVM_LIB_TARGET_XCORE_XCORETARGETMACHINE_H
#define LLVM_LIB_TARGET_XCORE_XCORETARGETMACHINE_H

#include "XCoreSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class XCoreTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  XCoreSubtarget Subtarget;
public:
  XCoreTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
  ~XCoreTargetMachine() override;

  const XCoreSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const XCoreSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetIRAnalysis getTargetIRAnalysis() override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};

} // end namespace llvm

#endif
